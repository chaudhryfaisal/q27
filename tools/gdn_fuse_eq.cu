// gdn_fuse_eq: bitwise gate for the 2026-08-18 GDN mix fusion.
//
// Runs the OLD decode-mix launch sequence (conv_step + gdn_conv_chunk3 +
// l2norm3 + gdn_record3 + delta_step + gdn_delta_chunk3) and the NEW one
// (conv_step + gdn_convnorm3 + gdn_delta_all) on identical random inputs and
// raw-bit-compares EVERY output the mix touches: the committed state S, the
// conv ring, all lanes' convout and o, and the record arena (qkv/conv/g/beta
// rows). The fusion's claim is bitwise identity, so the tolerance is zero.
//
// Widths: vw in {2, 3, 5, 8, 12, 16} (16 = W_PLUMB, the union ceiling).
// Exit: 0 = every width bit-identical; 1 = any byte differs.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "../src/kernels.cuh"
#include "../src/blocks.cuh"
#include "../src/spec3.cuh"

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while(0)

using q27k::CP3;
using q27k::P3;

static float* dclone(const std::vector<float>& h) {
    float* d; CK(cudaMalloc(&d, h.size()*4));
    CK(cudaMemcpy(d, h.data(), h.size()*4, cudaMemcpyHostToDevice));
    return d;
}
static int diff(const char* what, const float* a, const float* b, size_t n, int vw) {
    std::vector<float> ha(n), hb(n);
    CK(cudaMemcpy(ha.data(), a, n*4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb.data(), b, n*4, cudaMemcpyDeviceToHost));
    if (memcmp(ha.data(), hb.data(), n*4) == 0) return 0;
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < n; i++)
        if (memcmp(&ha[i], &hb[i], 4)) { if (!bad) first = i; bad++; }
    printf("  vw=%-2d %-10s DIFFERS: %zu/%zu elems (first @%zu: %.9g vs %.9g)\n",
           vw, what, bad, n, first, ha[first], hb[first]);
    return 1;
}

int main() {
    const int CH = 10240, HEADS = 48, SK = 128, NL = 16;
    const size_t SB = (size_t)HEADS * SK * SK;
    srand(20260818);
    auto rnd = [&](size_t n, float sc) {
        std::vector<float> v(n);
        for (auto& x : v) x = (rand()/(float)RAND_MAX - 0.5f) * sc;
        return v;
    };
    // shared inputs
    std::vector<float> h_ring = rnd(3*(size_t)CH, 2.f), h_S = rnd(SB, 1.f);
    std::vector<float> h_w = rnd((size_t)CH*4, 1.f);
    std::vector<float> h_qkv[NL], h_g[NL], h_b[NL];
    for (int l = 0; l < NL; l++) {
        h_qkv[l] = rnd(CH, 2.f);
        h_g[l] = rnd(HEADS, 1.f);            // decay = exp(g), keep moderate
        h_b[l] = rnd(HEADS, 1.f);
    }
    float* d_w = dclone(h_w);
    float* d_qkv[NL]; float* d_g[NL]; float* d_b[NL];
    for (int l = 0; l < NL; l++) { d_qkv[l]=dclone(h_qkv[l]); d_g[l]=dclone(h_g[l]); d_b[l]=dclone(h_b[l]); }

    int fails = 0;
    const int vws[] = {2, 3, 5, 8, 12, 16};
    for (int vw : vws) {
        const int nsp = vw - 1;
        // two independent copies of every mutable buffer
        float *ringA=dclone(h_ring), *ringB=dclone(h_ring);
        float *SA=dclone(h_S), *SB_=dclone(h_S);
        float *coA[NL], *coB[NL], *oA[NL], *oB[NL];
        for (int l = 0; l < NL; l++) {
            std::vector<float> z(CH, 0.f), zo((size_t)SK*HEADS, 0.f);
            coA[l]=dclone(z); coB[l]=dclone(z); oA[l]=dclone(zo); oB[l]=dclone(zo);
        }
        std::vector<float> zrec((size_t)nsp*CH, 0.f), zrg((size_t)nsp*HEADS, 0.f);
        float *rqA=dclone(zrec), *rcA=dclone(zrec), *rgA=dclone(zrg), *rbA=dclone(zrg);
        float *rqB=dclone(zrec), *rcB=dclone(zrec), *rgB=dclone(zrg), *rbB=dclone(zrg);

        auto lanes = [&](float* const* p) { P3 s{}; for (int l=0;l<NL;l++) s.p[l]=p[l]; return s; };
        auto clanes = [&](float* const* p) { CP3 s{}; for (int l=0;l<NL;l++) s.p[l]=p[l]; return s; };
        CP3 qkvL = clanes(d_qkv), gL = clanes(d_g), bL = clanes(d_b);

        // OLD sequence (engine order pre-fusion)
        q27k::conv_step(ringA, ringA, d_qkv[0], d_w, coA[0], CH, 0);
        q27k::gdn_conv_chunk3(ringA, qkvL, d_w, lanes(coA), CH, nsp, 0);
        q27k::l2norm3(lanes(coA), 32, SK, 1e-6f, 0, vw);
        q27k::gdn_record3(qkvL, clanes(coA), gL, bL, rqA, rcA, rgA, rbA, CH, HEADS, nsp, 0);
        q27k::delta_step(SA, SA, coA[0], d_g[0], d_b[0], oA[0], 0);
        q27k::gdn_delta_chunk3(SA, clanes(coA), gL, bL, lanes(oA), nsp, 0);
        // NEW sequence
        q27k::conv_step(ringB, ringB, d_qkv[0], d_w, coB[0], CH, 0);
        q27k::gdn_convnorm3(ringB, qkvL, d_w, lanes(coB), gL, bL, rqB, rcB, rgB, rbB, CH,
                            HEADS, 1e-6f, vw, 0);
        q27k::gdn_delta_all(SB_, SB_, clanes(coB), gL, bL, lanes(oB), nsp, 0);
        CK(cudaDeviceSynchronize());

        int f = 0;
        f += diff("S", SA, SB_, SB, vw);
        f += diff("ring", ringA, ringB, 3*(size_t)CH, vw);
        for (int l = 0; l < vw; l++) {
            f += diff("convout", coA[l], coB[l], CH, vw);
            f += diff("o", oA[l], oB[l], (size_t)SK*HEADS, vw);
        }
        f += diff("rec_qkv", rqA, rqB, (size_t)nsp*CH, vw);
        f += diff("rec_conv", rcA, rcB, (size_t)nsp*CH, vw);
        f += diff("rec_g", rgA, rgB, (size_t)nsp*HEADS, vw);
        f += diff("rec_beta", rbA, rbB, (size_t)nsp*HEADS, vw);
        printf("vw=%-2d  %s\n", vw, f ? "FAIL" : "BITWISE IDENTICAL");
        fails += f;

        CK(cudaFree(ringA)); CK(cudaFree(ringB)); CK(cudaFree(SA)); CK(cudaFree(SB_));
        for (int l = 0; l < NL; l++) { CK(cudaFree(coA[l])); CK(cudaFree(coB[l]));
                                       CK(cudaFree(oA[l])); CK(cudaFree(oB[l])); }
        CK(cudaFree(rqA)); CK(cudaFree(rcA)); CK(cudaFree(rgA)); CK(cudaFree(rbA));
        CK(cudaFree(rqB)); CK(cudaFree(rcB)); CK(cudaFree(rgB)); CK(cudaFree(rbB));
    }
    printf("%s\n", fails ? "GATE FAIL" : "GATE PASS: fused mix is bitwise identical at every width");
    return fails ? 1 : 0;
}
