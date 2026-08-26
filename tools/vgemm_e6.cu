// vgemm_e6 -- attribution harness for E6 (cp.async in k_vgemm).
//
// round_weight_cost says the shipped k_vgemm moves the round's 401-weight
// sequence at ~1119 GB/s. Against the tool's 1453 GB/s constant that is 77%;
// against this machine's MEASURED streaming-read SOL it is lower still. Before
// rewriting the memory pipeline, decompose where the missing bandwidth goes.
// Four candidate sinks, and only one of them is the memory pipeline:
//
//   1. LAUNCH GAPS      -- 401 (+ up to 401 reduce) plain launches, not a graph.
//   2. THE REDUCE       -- the deterministic grid.z epilogue: z*T*rows floats
//                          written then re-read by k_reduce_z, 12-22% of the
//                          weight bytes on the big shapes.
//   3. GRID/TAIL        -- z is an integer, so CTA counts land at 1.9-2.4 waves.
//   4. THE TILE ITSELF  -- 4-byte __ldg staging through registers.
//
// The floor kernel isolates (4): identical grid, identical bytes, identical
// occupancy class, and (MODE 0/1) identical epilogue stores, but no smem
// staging and no MMA. Build the tool with -DQ27_VGEMM_NOMMA to get a third
// point -- the real kernel's staging path with the MMA compiled out -- which
// splits (4) into "memory pipeline" and "compute phase".
//
// ANSWER, 2026-08-19: grid 5%, stores 0.09 ms (L2 absorbs them), staging
// 0.25 ms, MMA phase 0.72 ms, reduce 0.62 ms. See BUILDLOG 2026-08-19 (n).
//
// Also prints an output digest -- an order-independent fold over the raw float
// bits of every lane output after each of the 401 weights. Two builds that
// agree there produced bit-identical results over the whole sequence, which is
// the only bit-identity claim available for this kernel: solo greedy decode
// stays on the GEMV (gemm_min) and the canonical CLI md5 never reaches it.
//
// Usage: vgemm_e6 model.q27 [T]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/device_model.h"
#include "../src/kernels.cuh"
#include "../src/loader.h"
#include "../src/vgemm.cuh"

#define CK(x)                                                                  \
    do {                                                                       \
        cudaError_t e = (x);                                                   \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

using q27k::VG_KB;
using q27k::VG_KG;
using q27k::VG_KS;
using q27k::VG_MR;

static constexpr int N_LAYER = 64;

// ---------------------------------------------------------------- SOL probe
__global__ void k_bwprobe(const uint4* __restrict__ p, size_t n4, float* __restrict__ sink) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (; i < n4; i += stride) {
        const uint4 v = __ldg(p + i);
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x | acc.y | acc.z | acc.w) == 0xdeadbeefu) *sink = 1.f;
}

// ------------------------------------------------------------- launch floor
__global__ void k_null() {}

// ------------------------------------- weight-read floor at k_vgemm's grid
// Same grid(1, rows/MR, z), same 256 threads, same __launch_bounds__ class, same
// per-super-step byte footprint (W + the fp16 group scales). No smem, no MMA, no
// epilogue: this is the most bandwidth this grid shape can possibly deliver.
template <bool Q4IN, int MODE>
__global__ __launch_bounds__(256, 4) void k_floor(const uint8_t* __restrict__ W,
                                                  const __half* __restrict__ S,
                                                  float* __restrict__ sink,
                                                  float* __restrict__ ws,
                                                  float* const* __restrict__ y, int64_t rows,
                                                  int64_t cols, int T, int stages_per_z) {
    constexpr int MR = VG_MR, KS = VG_KS, KG = VG_KG, KB = KG * KS;
    constexpr int WBY = Q4IN ? (MR * KB / 2) : (MR * KB); // weight bytes per super-step
    constexpr int TPR = WBY / 16 / MR;                    // 16B threads per row
    const int64_t r0 = (int64_t)blockIdx.y * MR;
    const int n_stages = (int)(cols / KS);
    const int s_begin = blockIdx.z * stages_per_z;
    const int s_end = min(n_stages, s_begin + stages_per_z);
    if (s_begin >= s_end) return;

    const int tid = threadIdx.x;
    const int rr = tid / TPR, ch = tid % TPR;
    const int64_t rstride = Q4IN ? (cols / 2) : cols;
    const int64_t sstride = cols / (Q4IN ? 64 : 128);
    const bool ok = (r0 + rr < rows);
    uint4 acc = make_uint4(0, 0, 0, 0);
    float sacc = 0.f;

    for (int sst = s_begin; sst < s_end; sst += KG) {
        const int64_t k0 = (int64_t)sst * KS;
#pragma unroll
        for (int rep = 0; rep < (256 / TPR >= MR ? 1 : MR / (256 / TPR)); rep++) {
            const int r = rr + rep * (256 / TPR);
            if (r0 + r < rows) {
                const uint8_t* base =
                    W + (r0 + r) * rstride + (Q4IN ? k0 / 2 : k0) + (int64_t)ch * 16;
                const uint4 v = __ldg((const uint4*)base);
                acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
            }
        }
        // group scales: MR*KG*(2 or 1) halves per super-step, one per thread max
        const int nsc = MR * KG * (Q4IN ? 2 : 1);
        if (tid < nsc && ok) {
            const int srr = tid / (KG * (Q4IN ? 2 : 1)), sg = tid % (KG * (Q4IN ? 2 : 1));
            if (r0 + srr < rows)
                sacc += __half2float(__ldg(S + (r0 + srr) * sstride +
                                           k0 / (Q4IN ? 64 : 128) + sg));
        }
    }
    // MODE 2 == read-only. MODE 0/1 replay k_vgemm's epilogue byte-for-byte, so
    // the floor carries the same store traffic the real kernel does.
    if constexpr (MODE != 2) {
        float a4[4];
#pragma unroll
        for (int e = 0; e < 4; e++)
            a4[e] = (float)((acc.x >> (e * 8)) & 0xff) + sacc;
        const int warp = tid / 32, lane = tid & 31;
        const int wm = warp % (MR / 16), wn = (warp / (MR / 16)) % 2;
        const int gid = lane >> 2, tg = lane & 3;
        const int64_t row0 = r0 + wm * 16 + gid;
        const int tok0 = wn * 8 + tg * 2;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int64_t row = row0 + (e >= 2 ? 8 : 0);
            int tok = tok0 + (e & 1);
            if (row < rows && tok < T) {
                if constexpr (MODE == 1) ws[((size_t)blockIdx.z * T + tok) * rows + row] = a4[e];
                else y[tok][row] = a4[e];
            }
        }
        return;
    }
    if ((acc.x | acc.y | acc.z | acc.w) == 0xdeadbeefu && sacc == 1234.5f) *sink = 1.f;
}

// -------------------------------------------- standalone copy of the reduce
// Byte-for-byte the epilogue src/vgemm.cu launches, so timing it alone
// subtracts cleanly from the total.
__global__ void k_reduce_z_probe(const float* __restrict__ ws, float* const* __restrict__ y,
                                 int64_t rows, int T, int z) {
    int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (r >= rows || t >= T) return;
    float s = 0.f;
    for (int i = 0; i < z; i++) s += ws[((size_t)i * T + t) * rows + r];
    y[t][r] = s;
}

// ---------------------------------------------------------- output digest
// Order-independent, position-sensitive fold over the raw float BITS of every
// lane output, taken after each weight. Two builds that agree here produced
// bit-identical results over the whole 401-weight sequence -- which is the
// actual E6a claim, and one the canonical CLI md5 cannot make (solo greedy
// decode stays on the GEMV and never reaches k_vgemm at all).
__global__ void k_digest(float* const* __restrict__ y, int64_t rows, int T, int wi,
                         unsigned long long* __restrict__ acc) {
    int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (r >= rows || t >= T) return;
    unsigned int b = __float_as_uint(y[t][r]);
    unsigned long long k = (unsigned long long)(r * 1000003ull + t * 31ull + wi * 7919ull + 1ull);
    atomicAdd(acc, (unsigned long long)b * k);
}

// mirror of src/vgemm.cu launch()'s z / spz derivation
static void vg_geom(int64_t rows, int64_t cols, int& z, int& spz) {
    const int n_stages = (int)(cols / VG_KS);
    z = q27k::vgemm_z(rows, cols);
    spz = (n_stages + z - 1) / z;
    spz = (spz + VG_KG - 1) / VG_KG * VG_KG;
    z = (n_stages + spz - 1) / spz;
}

// Time a launch sequence the way the ENGINE runs it: captured once into a CUDA
// graph, then replayed. Eager launches on this sequence cost ~2 us each and 801
// of them would swamp the very gap this tool is measuring.
// capture + instantiate wall for one sequence, and its node count. The engine
// pays this on every gcache MISS round (BUILDLOG 08-18: 20-28 ms/miss, mean
// ~23), so node count is a lever in its own right, separate from kernel time.
template <class F>
static double timeit_capture(F&& f, int reps, cudaStream_t st, size_t* nodes) {
    double tot = 0;
    for (int i = 0; i < reps; i++) {
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaDeviceSynchronize());
        auto t0 = std::chrono::steady_clock::now();
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        f();
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        auto t1 = std::chrono::steady_clock::now();
        tot += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (nodes) CK(cudaGraphGetNodes(g, nullptr, nodes));
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    }
    return tot / reps;
}

template <class F>
static double timeit_graph(F&& f, int reps, cudaStream_t st) {
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    f();
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaEventRecord(a, st));
    for (int i = 0; i < reps; i++) CK(cudaGraphLaunch(ge, st));
    CK(cudaEventRecord(b, st));
    CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    return ms / reps;
}

template <class F>
static double timeit(F&& f, int reps) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    f();
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; i++) f();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms / reps;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.q27 [T]\n", argv[0]); return 1; }
    const int T = argc > 2 ? atoi(argv[2]) : 16;

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));

    // measured streaming-read SOL -- every percentage below is against THIS
    double SOL;
    {
        const size_t bytes = 2ull << 30;
        uint8_t* d; CK(cudaMalloc(&d, bytes)); CK(cudaMemset(d, 0x5a, bytes));
        float* s2; CK(cudaMalloc(&s2, 4));
        const double ms = timeit([&] {
            k_bwprobe<<<prop.multiProcessorCount * 8, 256>>>((const uint4*)d, bytes / 16, s2);
        }, 20);
        SOL = bytes / (ms * 1e6);
        printf("device: %s, %d SMs, L2 %.1f MB\n", prop.name, prop.multiProcessorCount,
               prop.l2CacheSize / 1048576.0);
        printf("measured streaming-read SOL: %.0f GB/s\n\n", SOL);
        CK(cudaFree(d)); CK(cudaFree(s2));
    }

    q27::Model m = q27::Model::open(argv[1]);
    q27::DeviceModel dm(m);
    dm.upload_all();

    std::vector<bool> is_attn(N_LAYER, false);
    for (int il = 0; il < N_LAYER; il++) {
        char b[96]; snprintf(b, sizeof b, "blk.%d.attn_q.weight", il);
        is_attn[il] = m.find(b) != nullptr;
    }

    struct Item { const q27::DevTensor* t; std::string cls; };
    std::vector<Item> seq;
    size_t wbytes = 0, sbytes = 0;
    auto add = [&](const char* fmt, int il, const char* cls) {
        char b[96]; snprintf(b, sizeof b, fmt, il);
        if (!m.find(b)) return;
        const q27::DevTensor& t = dm.get(b);
        seq.push_back({&t, cls});
        const int64_t r = (int64_t)t.rows, c = (int64_t)t.cols;
        wbytes += (t.dtype == q27::DType::Q4_G64) ? (size_t)r * c / 2 : (size_t)r * c;
        sbytes += (size_t)r * (c / (t.dtype == q27::DType::Q4_G64 ? 64 : 128)) * 2;
    };
    for (int il = 0; il < N_LAYER; il++) {
        if (is_attn[il]) {
            add("blk.%d.attn_q.weight", il, "attn_q");
            add("blk.%d.attn_k.weight", il, "attn_kv");
            add("blk.%d.attn_v.weight", il, "attn_kv");
            add("blk.%d.attn_output.weight", il, "attn_out");
        } else {
            add("blk.%d.attn_qkv.weight", il, "gdn_qkv");
            add("blk.%d.attn_gate.weight", il, "gdn_gate");
            add("blk.%d.ssm_out.weight", il, "ssm_out");
        }
        add("blk.%d.ffn_gate.weight", il, "ffn_gate");
        add("blk.%d.ffn_up.weight", il, "ffn_up");
        add("blk.%d.ffn_down.weight", il, "ffn_down");
    }
    {
        const char* vh = m.find("output_q4.weight") ? "output_q4.weight" : "output.weight";
        const q27::DevTensor& t = dm.get(vh);
        seq.push_back({&t, "vocab"});
        wbytes += (t.dtype == q27::DType::Q4_G64) ? (size_t)t.rows * t.cols / 2
                                                  : (size_t)t.rows * t.cols;
        sbytes += (size_t)t.rows * (t.cols / (t.dtype == q27::DType::Q4_G64 ? 64 : 128)) * 2;
    }
    const size_t tot = wbytes + sbytes;
    printf("round sequence: %zu weights, %.2f GB codes + %.2f GB scales = %.2f GB\n",
           seq.size(), wbytes / 1e9, sbytes / 1e9, tot / 1e9);
    printf("(round_weight_cost reports codes only; scales are %.1f%% more real traffic)\n\n",
           100.0 * sbytes / wbytes);

    // ---- lanes
    const int64_t maxcols = 17408, maxrows = 248320;
    q27k::XQuant qs[16];
    float* d_x; CK(cudaMalloc(&d_x, maxcols * 4));
    std::vector<float> hx(maxcols);
    for (int i = 0; i < maxcols; i++) hx[i] = (float)((i * 2654435761u) % 1000) / 500.f - 1.f;
    CK(cudaMemcpy(d_x, hx.data(), maxcols * 4, cudaMemcpyHostToDevice));
    q27k::XLanes X{}; q27k::YLanes Y{};
    float* d_y[16];
    for (int i = 0; i < 16; i++) {
        qs[i] = q27k::xquant_alloc(maxcols);
        q27k::quantize_x(d_x, maxcols, qs[i]);
        CK(cudaMalloc(&d_y[i], maxrows * 4));
        X.nat[i] = qs[i].nat; X.xs[i] = qs[i].scale; Y.y[i] = d_y[i];
    }
    float** d_ylist; CK(cudaMalloc(&d_ylist, 16 * sizeof(float*)));
    CK(cudaMemcpy(d_ylist, d_y, 16 * sizeof(float*), cudaMemcpyHostToDevice));

    std::vector<const q27::DevTensor*> raw;
    for (auto& it : seq) raw.push_back(it.t);
    size_t wsb = q27k::vgemm_ws_bytes(raw.data(), (int)raw.size());
    float* d_ws; CK(cudaMalloc(&d_ws, wsb));
    float* sink; CK(cudaMalloc(&sink, 4));

    // how many launches does the real sequence make, and how many carry a reduce
    int n_reduce = 0;
    for (auto& it : seq) { int z, spz; vg_geom(it.t->rows, it.t->cols, z, spz); if (z > 1) n_reduce++; }
    printf("workspace %.2f MB; %zu main launches + %d reduce launches\n\n", wsb / 1e6,
           seq.size(), n_reduce);

    // bytes the MAIN kernel stores on top of the weight stream: the grid.z
    // partials (MODE 1) or the lane outputs (MODE 0).
    size_t pbytes = 0;
    for (auto& it : seq) {
        int z, spz; vg_geom(it.t->rows, it.t->cols, z, spz);
        pbytes += (size_t)(z > 1 ? z : 1) * T * (size_t)it.t->rows * 4;
    }
    printf("main-kernel epilogue stores: %.2f GB (%.1f%% on top of the weight stream)\n\n",
           pbytes / 1e9, 100.0 * pbytes / tot);

    const int REP = 10;
    auto rate = [&](double ms) { return tot / 1e9 / (ms / 1000.0); };
    auto ratew = [&](double ms) { return (tot + pbytes) / 1e9 / (ms / 1000.0); };
    cudaStream_t st; CK(cudaStreamCreate(&st));

    // ---- 1. launch-gap floor
    const int nlaunch = (int)seq.size() + n_reduce;
    auto null_seq = [&] { for (int i = 0; i < nlaunch; i++) k_null<<<1, 32, 0, st>>>(); };

    // ---- 2. floors at k_vgemm's exact grid: read-only, and read+epilogue
    auto floor1 = [&](const q27::DevTensor& w, bool store) {
        int z, spz; vg_geom(w.rows, w.cols, z, spz);
        dim3 g(1, (unsigned)((w.rows + VG_MR - 1) / VG_MR), (unsigned)z);
        const uint8_t* wd = (const uint8_t*)w.data;
        const __half* sc = (const __half*)w.scales;
        const bool q4 = w.dtype == q27::DType::Q4_G64;
        if (!store) {
            if (q4) k_floor<true, 2><<<g, 256, 0, st>>>(wd, sc, sink, d_ws, d_ylist, w.rows, w.cols, T, spz);
            else    k_floor<false, 2><<<g, 256, 0, st>>>(wd, sc, sink, d_ws, d_ylist, w.rows, w.cols, T, spz);
        } else if (z > 1) {
            if (q4) k_floor<true, 1><<<g, 256, 0, st>>>(wd, sc, sink, d_ws, d_ylist, w.rows, w.cols, T, spz);
            else    k_floor<false, 1><<<g, 256, 0, st>>>(wd, sc, sink, d_ws, d_ylist, w.rows, w.cols, T, spz);
        } else {
            if (q4) k_floor<true, 0><<<g, 256, 0, st>>>(wd, sc, sink, d_ws, d_ylist, w.rows, w.cols, T, spz);
            else    k_floor<false, 0><<<g, 256, 0, st>>>(wd, sc, sink, d_ws, d_ylist, w.rows, w.cols, T, spz);
        }
    };
    auto floor_seq = [&] { for (auto& it : seq) floor1(*it.t, false); };
    auto floorw_seq = [&] { for (auto& it : seq) floor1(*it.t, true); };

    // ---- 3. reduce epilogue alone
    auto reduce_seq = [&] {
        for (auto& it : seq) {
            const q27::DevTensor& w = *it.t;
            int z, spz; vg_geom(w.rows, w.cols, z, spz);
            if (z <= 1) continue;
            dim3 g2((unsigned)((w.rows + 255) / 256), (unsigned)T);
            k_reduce_z_probe<<<g2, 256, 0, st>>>(d_ws, d_ylist, w.rows, T, z);
        }
    };

    // ---- 4. the shipped kernel
    auto vg_seq = [&] { for (auto& it : seq) q27k::vgemm_verify(*it.t, X, Y, d_ws, T, st); };
    // same launches minus the reduce nodes -- what a fused split-K epilogue
    // would leave in the graph. Wrong results by construction; capture only.
    auto vg_main_only = [&] {
        for (auto& it : seq) {
            const q27::DevTensor& w = *it.t;
            int z, spz; vg_geom(w.rows, w.cols, z, spz);
            floor1(w, true); // one node per weight, same grid/shape class
            (void)z; (void)spz;
        }
    };

    double ms_null = timeit_graph(null_seq, REP, st);
    double ms_floor = timeit_graph(floor_seq, REP, st);
    double ms_floorw = timeit_graph(floorw_seq, REP, st);
    double ms_red = timeit_graph(reduce_seq, REP, st);
    double ms_vg = timeit_graph(vg_seq, REP, st);
    double eager_vg = timeit(vg_seq, REP);
    CK(cudaGetLastError());

    printf("== whole-round decomposition at T=%d (%d reps, CUDA-GRAPH replayed) ==\n", T, REP);
    printf("%-34s %9s %10s %8s\n", "leg", "ms", "GB/s", "%SOL");
    printf("%-34s %9.2f %10s %8s\n", "null-launch floor (801 CTAs)", ms_null, "-", "-");
    printf("%-34s %9.2f %10.0f %7.0f%%\n", "weight-read floor (same grid)", ms_floor,
           rate(ms_floor), 100.0 * rate(ms_floor) / SOL);
    printf("%-34s %9.2f %10.0f %7.0f%%   (+%.2f GB of partials)\n",
           "  + k_vgemm's epilogue stores", ms_floorw, ratew(ms_floorw),
           100.0 * ratew(ms_floorw) / SOL, pbytes / 1e9);
    printf("%-34s %9.2f %10s %8s\n", "reduce epilogue alone", ms_red, "-", "-");
    printf("%-34s %9.2f %10.0f %7.0f%%\n", "k_vgemm (shipped, incl reduce)", ms_vg, rate(ms_vg),
           100.0 * rate(ms_vg) / SOL);
    printf("%-34s %9.2f %10.0f %7.0f%%\n", "  ... minus reduce", ms_vg - ms_red,
           rate(ms_vg - ms_red), 100.0 * ratew(ms_vg - ms_red) / SOL);
    printf("%-34s %9.2f %10.0f %7.0f%%   (+%.2f ms of launch gap)\n", "same, EAGER launches",
           eager_vg, rate(eager_vg), 100.0 * rate(eager_vg) / SOL, eager_vg - ms_vg);
    {
        size_t n_all = 0, n_main = 0;
        double cap_all = timeit_capture(vg_seq, 5, st, &n_all);
        double cap_main = timeit_capture(vg_main_only, 5, st, &n_main);
        printf("\ngraph capture+instantiate: %zu nodes -> %.2f ms | %zu nodes -> %.2f ms"
               "  (%.1f us/node)\n",
               n_all, cap_all, n_main, cap_main,
               1000.0 * (cap_all - cap_main) / (double)(n_all - n_main));
    }

    printf("\nTILE gap (main kernel vs same-grid floor WITH the same stores): %.2f ms\n",
           (ms_vg - ms_red) - ms_floorw);
    printf("DETERMINISM tax  (partial stores %.2f ms + reduce %.2f ms): %.2f ms = %.0f%% of the sweep\n",
           ms_floorw - ms_floor, ms_red, (ms_floorw - ms_floor) + ms_red,
           100.0 * ((ms_floorw - ms_floor) + ms_red) / ms_vg);

    // ---- per-shape-class breakdown
    printf("\n== per class (ms over the whole round; floor vs shipped) ==\n");
    printf("%-10s %4s %7s %7s %6s %6s %9s %9s %8s\n", "class", "n", "rows", "cols", "z", "waves",
           "floor+st", "vgemm ms", "eff");
    std::vector<std::string> classes;
    for (auto& it : seq)
        if (std::find(classes.begin(), classes.end(), it.cls) == classes.end())
            classes.push_back(it.cls);
    const int co_res = prop.multiProcessorCount * 4;
    for (const std::string& c : classes) {
        std::vector<const q27::DevTensor*> sub;
        for (auto& it : seq) if (it.cls == c) sub.push_back(it.t);
        auto fl = [&] { for (const q27::DevTensor* wp : sub) floor1(*wp, true); };
        auto vg = [&] { for (const q27::DevTensor* wp : sub) q27k::vgemm_verify(*wp, X, Y, d_ws, T, st); };
        double f = timeit_graph(fl, REP, st), v = timeit_graph(vg, REP, st);
        int z, spz; vg_geom(sub[0]->rows, sub[0]->cols, z, spz);
        const int ctas = (int)((sub[0]->rows + VG_MR - 1) / VG_MR) * z;
        printf("%-10s %4zu %7lld %7lld %6d %6.2f %9.2f %9.2f %7.0f%%\n", c.c_str(), sub.size(),
               (long long)sub[0]->rows, (long long)sub[0]->cols, z, (double)ctas / co_res, f, v,
               100.0 * f / v);
    }
    // ---- bit-identity digest over the whole sequence
    {
        unsigned long long* d_acc; CK(cudaMalloc(&d_acc, 8));
        CK(cudaMemset(d_acc, 0, 8));
        int wi = 0;
        for (auto& it : seq) {
            const q27::DevTensor& w = *it.t;
            q27k::vgemm_verify(w, X, Y, d_ws, T, st);
            dim3 g((unsigned)((w.rows + 255) / 256), (unsigned)T);
            k_digest<<<g, 256, 0, st>>>(d_ylist, w.rows, T, wi++, d_acc);
        }
        CK(cudaStreamSynchronize(st));
        unsigned long long h; CK(cudaMemcpy(&h, d_acc, 8, cudaMemcpyDeviceToHost));
        printf("\noutput digest over all %zu weights at T=%d: %016llx\n", seq.size(), T, h);
    }
    return 0;
}
