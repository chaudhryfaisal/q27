// q27 engine: CUDA-graph token decode. One captured graph replays per token;
// pos/token/step live on device so the graph is launch-stable, and greedy
// decode chains device-side (argmax writes d_token for the next replay).
// Usage: q27 <model.q27> --tokens "1,2,3" -n 16 [--ctx 2048] [--dump-logits f.bin]
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "blocks.cuh"
#include "cuda_common.h"
#include "device_model.h"
#include "kernels.cuh"
#include "loader.h"

using q27::DevTensor;
using q27::DType;

static constexpr int N_LAYER = 64;
static constexpr int N_EMBD = 5120;
static constexpr int N_FFN = 17408;
static constexpr int N_HEAD = 24, N_KV = 4, HEAD_DIM = 256;
static constexpr int N_ROT = 64;
static constexpr float FREQ_BASE = 1e7f;
static constexpr float EPS = 1e-6f;
static constexpr int GDN_CH = 10240, GDN_V = 6144, GDN_HEADS = 48, GDN_DIM = 128;
static constexpr int VOCAB = 248320;
static constexpr int MAX_GEN_TRACK = 65536;

struct Engine {
    q27::Model model;
    q27::DeviceModel dm;
    int max_ctx;
    bool attn_layer[N_LAYER + 1] = {false};
    cudaStream_t stm;
    cudaGraphExec_t graph_exec = nullptr;

    // activations (device)
    float *h, *x1, *y, *qg, *kbuf, *vbuf, *attnout, *scratch;
    float *qkv, *convout, *z, *alpha, *betar, *g, *beta, *o, *og;
    float *ffn_g, *ffn_u, *logits;
    // device decode state
    int *d_pos, *d_token, *d_step, *d_gen;
    unsigned long long* d_amax;
    q27k::XQuant xq;
    // layer state
    float* conv_ring[N_LAYER];
    float* S[N_LAYER];
    std::vector<float*> kcache, vcache;
    std::vector<int> attn_cache_idx;

    Engine(const std::string& path, int ctx)
        : model(q27::Model::open(path)), dm(model), max_ctx(ctx) {
        CUDA_CHECK(cudaStreamCreate(&stm));
        const std::string& mj = model.meta_json;
        size_t p = mj.find("\"attn_layers\": [");
        if (p == std::string::npos) { fprintf(stderr, "no attn_layers in meta\n"); exit(1); }
        p += strlen("\"attn_layers\": [");
        while (p < mj.size() && mj[p] != ']') {
            int v = atoi(mj.c_str() + p);
            if (v <= N_LAYER) attn_layer[v] = true;
            p = mj.find_first_of(",]", p);
            if (mj[p] == ',') p++;
        }

        auto A = [](void** pp, size_t n) { CUDA_CHECK(cudaMalloc(pp, n)); };
        A((void**)&h, N_EMBD * 4); A((void**)&x1, N_EMBD * 4); A((void**)&y, N_EMBD * 4);
        A((void**)&qg, 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf, N_KV * HEAD_DIM * 4); A((void**)&vbuf, N_KV * HEAD_DIM * 4);
        A((void**)&attnout, N_HEAD * HEAD_DIM * 4);
        A((void**)&scratch, (size_t)N_HEAD * max_ctx * 4);
        A((void**)&qkv, GDN_CH * 4); A((void**)&convout, GDN_CH * 4); A((void**)&z, GDN_V * 4);
        A((void**)&alpha, GDN_HEADS * 4); A((void**)&betar, GDN_HEADS * 4);
        A((void**)&g, GDN_HEADS * 4); A((void**)&beta, GDN_HEADS * 4);
        A((void**)&o, GDN_V * 4); A((void**)&og, GDN_V * 4);
        A((void**)&ffn_g, N_FFN * 4); A((void**)&ffn_u, N_FFN * 4);
        A((void**)&logits, VOCAB * 4);
        A((void**)&d_pos, 4); A((void**)&d_token, 4); A((void**)&d_step, 4);
        A((void**)&d_gen, MAX_GEN_TRACK * 4);
        A((void**)&d_amax, 8);
        CUDA_CHECK(cudaMemset(d_pos, 0, 4));
        CUDA_CHECK(cudaMemset(d_step, 0, 4));
        xq = q27k::xquant_alloc(N_FFN);

        int cache_slot = 0;
        for (int il = 0; il < N_LAYER; il++) {
            if (attn_layer[il]) {
                float *k, *v;
                A((void**)&k, (size_t)max_ctx * N_KV * HEAD_DIM * 4);
                A((void**)&v, (size_t)max_ctx * N_KV * HEAD_DIM * 4);
                kcache.push_back(k); vcache.push_back(v);
                attn_cache_idx.push_back(cache_slot++);
                conv_ring[il] = nullptr; S[il] = nullptr;
            } else {
                A((void**)&conv_ring[il], 3 * GDN_CH * 4);
                CUDA_CHECK(cudaMemset(conv_ring[il], 0, 3 * GDN_CH * 4));
                A((void**)&S[il], (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4);
                CUDA_CHECK(cudaMemset(S[il], 0, (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4));
                attn_cache_idx.push_back(-1);
            }
        }
        fprintf(stderr, "uploading weights...\n");
        dm.upload_all();
        fprintf(stderr, "resident: %.2f GB\n", dm.bytes_resident() / 1e9);
    }

    const DevTensor& T(int il, const char* leaf) {
        char buf[96];
        snprintf(buf, sizeof buf, "blk.%d.%s", il, leaf);
        return dm.get(buf);
    }

    void qx(const float* x, int cols) { q27k::quantize_x(x, cols, xq, stm); }

    void mm(const DevTensor& w, const float* x, float* out) {
        switch (w.dtype) {
            case DType::Q4_G64:
                q27k::gemv_q4((const uint8_t*)w.data, (const __half*)w.scales, xq, out, w.rows,
                              w.cols, stm);
                break;
            case DType::Q8_G128:
                q27k::gemv_q8((const int8_t*)w.data, (const __half*)w.scales, xq, out, w.rows,
                              w.cols, stm);
                break;
            case DType::F16:
                q27k::gemv_f16((const __half*)w.data, x, out, w.rows, w.cols, stm);
                break;
            default:
                fprintf(stderr, "mm: unsupported dtype\n");
                exit(1);
        }
    }

    void gdn_block(int il, const float* xin, float* yout) {
        qx(xin, N_EMBD);
        mm(T(il, "attn_qkv.weight"), xin, qkv);
        mm(T(il, "attn_gate.weight"), xin, z);
        mm(T(il, "ssm_alpha.weight"), xin, alpha);
        mm(T(il, "ssm_beta.weight"), xin, betar);
        q27k::gdn_gates(alpha, betar, (const float*)T(il, "ssm_a").data,
                        (const float*)T(il, "ssm_dt.bias").data, g, beta, GDN_HEADS, stm);
        q27k::conv_step(conv_ring[il], qkv, (const float*)T(il, "ssm_conv1d.weight").data,
                        convout, GDN_CH, stm);
        q27k::l2norm_heads(convout, 16, GDN_DIM, EPS, stm);
        q27k::l2norm_heads(convout + 2048, 16, GDN_DIM, EPS, stm);
        q27k::delta_step(S[il], convout, g, beta, o, stm);
        q27k::gated_norm_gdn(o, (const float*)T(il, "ssm_norm.weight").data, z, og, GDN_HEADS,
                             GDN_DIM, EPS, stm);
        qx(og, GDN_V);
        mm(T(il, "ssm_out.weight"), og, yout);
    }

    void attn_block(int il, const float* xin, float* yout) {
        int ci = attn_cache_idx[il];
        qx(xin, N_EMBD);
        mm(T(il, "attn_q.weight"), xin, qg);
        q27k::rmsnorm_heads(qg, (const float*)T(il, "attn_q_norm.weight").data, qg, N_HEAD,
                            HEAD_DIM, 2 * HEAD_DIM, EPS, stm);
        mm(T(il, "attn_k.weight"), xin, kbuf);
        q27k::rmsnorm_heads(kbuf, (const float*)T(il, "attn_k_norm.weight").data, kbuf, N_KV,
                            HEAD_DIM, HEAD_DIM, EPS, stm);
        mm(T(il, "attn_v.weight"), xin, vbuf);
        q27k::rope_neox_partial(qg, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, d_pos, FREQ_BASE, stm);
        q27k::rope_neox_partial(kbuf, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, d_pos, FREQ_BASE, stm);
        q27k::kv_store(kbuf, vbuf, kcache[ci], vcache[ci], d_pos, N_KV * HEAD_DIM, stm);
        q27k::attn_decode(qg, 2 * HEAD_DIM, kcache[ci], vcache[ci], attnout, scratch, d_pos,
                          max_ctx, N_HEAD, N_KV, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM), stm);
        q27k::sigmoid_gate_mul(attnout, qg, N_HEAD, HEAD_DIM, stm);
        qx(attnout, N_HEAD * HEAD_DIM);
        mm(T(il, "attn_output.weight"), attnout, yout);
    }

    void ffn(int il, const float* xin, float* yout) {
        qx(xin, N_EMBD);
        mm(T(il, "ffn_gate.weight"), xin, ffn_g);
        mm(T(il, "ffn_up.weight"), xin, ffn_u);
        q27k::silu_mul(ffn_g, ffn_u, ffn_g, N_FFN, stm);
        qx(ffn_g, N_FFN);
        mm(T(il, "ffn_down.weight"), ffn_g, yout);
    }

    // enqueue one full token onto stm (no syncs, no allocations: graph-safe)
    void token_launches() {
        const DevTensor& emb = dm.get("token_embd.weight");
        q27k::embed_row_q8((const int8_t*)emb.data, (const __half*)emb.scales, d_token, N_EMBD, h,
                           stm);
        for (int il = 0; il < N_LAYER; il++) {
            q27k::rmsnorm(h, (const float*)T(il, "attn_norm.weight").data, x1, N_EMBD, EPS, stm);
            if (attn_layer[il]) attn_block(il, x1, y);
            else gdn_block(il, x1, y);
            q27k::add_inplace(h, y, N_EMBD, stm);
            q27k::rmsnorm(h, (const float*)T(il, "post_attention_norm.weight").data, x1, N_EMBD,
                          EPS, stm);
            ffn(il, x1, y);
            q27k::add_inplace(h, y, N_EMBD, stm);
        }
        q27k::rmsnorm(h, (const float*)dm.get("output_norm.weight").data, x1, N_EMBD, EPS, stm);
        qx(x1, N_EMBD);
        mm(dm.get("output.weight"), x1, logits);
        q27k::argmax(logits, VOCAB, d_token, d_amax, stm); // d_token becomes NEXT token
        q27k::advance(d_pos, d_step, d_gen, d_token, stm); // record + pos++
    }

    void build_graph() {
        // warm run (outside capture) so lazy CUDA state is initialized
        int zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_token, &zero, 4, cudaMemcpyHostToDevice, stm));
        token_launches();
        CUDA_CHECK(cudaStreamSynchronize(stm));
        // reset state mutated by the warm run
        CUDA_CHECK(cudaMemset(d_pos, 0, 4));
        CUDA_CHECK(cudaMemset(d_step, 0, 4));
        for (int il = 0; il < N_LAYER; il++)
            if (!attn_layer[il]) {
                CUDA_CHECK(cudaMemset(conv_ring[il], 0, 3 * GDN_CH * 4));
                CUDA_CHECK(cudaMemset(S[il], 0, (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4));
            }

        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamBeginCapture(stm, cudaStreamCaptureModeGlobal));
        token_launches();
        CUDA_CHECK(cudaStreamEndCapture(stm, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        fprintf(stderr, "token graph captured\n");
    }

    // feed one known token (prompt phase): set d_token, replay graph
    void step_with(int token) {
        CUDA_CHECK(cudaMemcpyAsync(d_token, &token, 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stm));
    }
    // generation step: d_token already holds the model's own prediction
    void step_free() { CUDA_CHECK(cudaGraphLaunch(graph_exec, stm)); }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.q27 --tokens \"1,2,3\" [-n N] [--ctx C] [--dump-logits f]\n",
                argv[0]);
        return 1;
    }
    std::string path = argv[1], dump;
    std::vector<int> toks;
    int n_gen = 8, ctx = 2048;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--tokens") && i + 1 < argc) {
            for (const char* p = argv[++i]; *p;) {
                toks.push_back(atoi(p));
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) dump = argv[++i];
    }
    if (toks.empty()) { fprintf(stderr, "need --tokens\n"); return 1; }

    Engine e(path, ctx);
    e.build_graph();

    // prompt
    for (int t : toks) e.step_with(t);
    CUDA_CHECK(cudaStreamSynchronize(e.stm));

    if (!dump.empty()) {
        std::vector<float> lg(VOCAB);
        CUDA_CHECK(cudaMemcpy(lg.data(), e.logits, (size_t)VOCAB * 4, cudaMemcpyDeviceToHost));
        FILE* f = fopen(dump.c_str(), "wb");
        fwrite(lg.data(), 4, VOCAB, f);
        fclose(f);
        fprintf(stderr, "logits -> %s\n", dump.c_str());
    }

    // generation: device-chained, zero host round-trips
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0, e.stm));
    for (int i = 0; i < n_gen; i++) e.step_free();
    CUDA_CHECK(cudaEventRecord(t1, e.stm));
    CUDA_CHECK(cudaStreamSynchronize(e.stm));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    // d_gen[i] = predicted-next after step i; generated tokens start at prompt_len-1
    std::vector<int> gen(toks.size() + n_gen);
    CUDA_CHECK(cudaMemcpy(gen.data(), e.d_gen, gen.size() * 4, cudaMemcpyDeviceToHost));
    printf("generated:");
    for (size_t i = toks.size() - 1; i < toks.size() - 1 + n_gen; i++) printf(" %d", gen[i]);
    printf("\ndecode: %d tokens in %.1f ms = %.2f t/s\n", n_gen, ms, n_gen * 1000.0f / ms);
    return 0;
}
