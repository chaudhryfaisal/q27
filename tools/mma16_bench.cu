// mma16_bench -- tensor-core verify spike (GEMM-verify reopen condition,
// BUILDLOG 2026-07-09: "reopens ONLY via tensor-core weight path at >=70% BW
// at M=16"). Measures a verify-shaped (T=16) fused q4->s8 MMA GEMM against:
//   (a) the current engine 16-lane verify cost: gemv_q4_n nb=8 called twice
//   (b) a pure-read SOL kernel over the same weight bytes (BW ceiling)
//   (c) the prefill k_gemm_mma_T shape (NT=128) at T=16 (the known-collapsed
//       29%-BW baseline from the GEMM-verify spike)
//   (d) an NT=16 fork of the same tile machinery, K-split z=1/2/4/8 across
//       CTAs (grid.z owns a K-stage slice, fp32 atomicAdd epilogue) -- the
//       FlashRT-warpsplit remedy for CTA starvation at small M
// Shapes: ffn_gate (17408x5120, tall) and ffn_down (5120x17408, long-K), the
// two dominant/opposite decode GEMM shapes; weights rotated across 4 layers
// so L2 never holds them. Numerics: XG64=false z=1 leg diffs against the
// dp4a gemv_q4 reference (int products exact, fp order differs -> ~1e-6 rel);
// atomic legs add K-slice reorder on top.
// Usage: mma16_bench model.q27
#include <cuda_fp8.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../src/device_model.h"
#include "../src/kernels.cuh"
#include "../src/loader.h"

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t err__ = (x);                                               \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---- mma primitives (forked verbatim from src/prefill.cu) ----
static __device__ __forceinline__ void mma_s8(int& d0, int& d1, int& d2, int& d3, uint32_t a0,
                                              uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                              uint32_t b1) {
    const int z = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(z), "r"(z), "r"(z), "r"(z));
}
static __device__ __forceinline__ void mma_s8_acc(int& d0, int& d1, int& d2, int& d3, uint32_t a0,
                                                  uint32_t a1, uint32_t a2, uint32_t a3,
                                                  uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---- k_gemm_mma_T fork: NT templated, optional K-slice + atomic epilogue ----
// Structure identical to src/prefill.cu k_gemm_mma_T (Q4IN=true fixed); the
// only deltas are (1) NT as a template param, (2) the stage loop runs
// [s_begin, s_end) taken from blockIdx.z, (3) ATOMIC epilogue accumulates
// into y instead of storing. Bench-only fork -- a GO promotes a reviewed
// kernel into src/, this copy stays in tools/.
template <int NT, bool XG64, bool ATOMIC>
__global__ void k_mma_verify(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                             const int8_t* __restrict__ nat, const float* __restrict__ xs,
                             float* __restrict__ y, int64_t rows, int64_t cols, int T,
                             int stages_per_z) {
    constexpr int MR = 64, KS = 128;
    constexpr int XGS = XG64 ? 64 : 32;
    constexpr int XSC = KS / XGS;
    constexpr int TS = NT / 16;
    constexpr int LDW = KS + 16, LDX = KS + 16;
    extern __shared__ unsigned char smem_raw[];
    int8_t* s_w = (int8_t*)smem_raw;
    int8_t* s_x = (int8_t*)(s_w + MR * LDW);
    float* s_ws = (float*)(s_x + NT * LDX);
    float* s_xs = (float*)(s_ws + MR * 2);

    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int wm = warp % 4, wn = warp / 4;
    const int gid = lane >> 2, tg = lane & 3;
    const int64_t r0 = (int64_t)blockIdx.y * MR;
    const int t0 = blockIdx.x * NT;
    const int n_stages = (int)(cols / KS);
    const int s_begin = blockIdx.z * stages_per_z;
    const int s_end = min(n_stages, s_begin + stages_per_z);
    if (s_begin >= s_end) return;

    float acc[TS][4];
#pragma unroll
    for (int s = 0; s < TS; s++)
#pragma unroll
        for (int e = 0; e < 4; e++) acc[s][e] = 0.f;

    constexpr int WLD = MR * (KS / 2) / 4 / 256;
    constexpr int XLD = (NT * KS / 4 + 255) / 256;
    constexpr int XSL = (NT * XSC + 255) / 256;
    const int tid = threadIdx.x;
    const int nws = MR * 2;
    uint32_t rw[WLD], rx[XLD];
    float rws = 0.f, rxs[XSL];

    auto load_stage = [&](int st) {
        const int64_t k0 = (int64_t)st * KS;
#pragma unroll
        for (int i = 0; i < WLD; i++) {
            int idx = i * 256 + tid;
            int rr = idx / 16, pb4 = idx % 16;
            rw[i] = r0 + rr < rows
                        ? __ldg((const uint32_t*)(W + (r0 + rr) * (cols / 2) + k0 / 2) + pb4)
                        : 0x88888888u;
        }
#pragma unroll
        for (int i = 0; i < XLD; i++) {
            int idx = i * 256 + tid;
            if (idx < NT * KS / 4) {
                int tt = idx / (KS / 4), u = idx % (KS / 4);
                rx[i] = t0 + tt < T
                            ? __ldg((const uint32_t*)(nat + (size_t)(t0 + tt) * cols + k0) + u)
                            : 0u;
            }
        }
        if (tid < nws) {
            int rr = tid / 2, g = tid % 2;
            rws = r0 + rr < rows
                      ? __half2float(__ldg(S + (r0 + rr) * (cols / 64) + k0 / 64 + g))
                      : 0.f;
        }
#pragma unroll
        for (int i = 0; i < XSL; i++) {
            int idx = i * 256 + tid;
            int tt = idx / XSC, cc = idx % XSC;
            rxs[i] = (idx < NT * XSC && t0 + tt < T)
                         ? __ldg(xs + (size_t)(t0 + tt) * (cols / XGS) + k0 / XGS + cc)
                         : 0.f;
        }
    };
    auto store_stage = [&]() {
#pragma unroll
        for (int i = 0; i < WLD; i++) {
            int idx = i * 256 + tid;
            int rr = idx / 16, pb4 = idx % 16;
            int8_t* dst = s_w + rr * LDW + pb4 * 8;
            const uint32_t p = rw[i];
            const uint32_t lo = p & 0x0F0F0F0Fu, hi = (p >> 4) & 0x0F0F0F0Fu;
            *(uint32_t*)dst = __vsub4(__byte_perm(lo, hi, 0x5140), 0x08080808u);
            *(uint32_t*)(dst + 4) = __vsub4(__byte_perm(lo, hi, 0x7362), 0x08080808u);
        }
#pragma unroll
        for (int i = 0; i < XLD; i++) {
            int idx = i * 256 + tid;
            if (idx < NT * KS / 4) {
                int tt = idx / (KS / 4), u = idx % (KS / 4);
                *(uint32_t*)(s_x + tt * LDX + u * 4) = rx[i];
            }
        }
        if (tid < nws) s_ws[tid] = rws;
#pragma unroll
        for (int i = 0; i < XSL; i++) {
            int idx = i * 256 + tid;
            if (idx < NT * XSC) s_xs[idx] = rxs[i];
        }
    };

    load_stage(s_begin);
    for (int st = s_begin; st < s_end; st++) {
        __syncthreads();
        store_stage();
        if (st + 1 < s_end) load_stage(st + 1);
        __syncthreads();
        if constexpr (!XG64) {
#pragma unroll
            for (int cc = 0; cc < 4; cc++) {
                const int kb = cc * 32;
                const int8_t* wrow0 = s_w + (wm * 16 + gid) * LDW + kb;
                uint32_t a0 = *(const uint32_t*)(wrow0 + tg * 4);
                uint32_t a1 = *(const uint32_t*)(wrow0 + 8 * LDW + tg * 4);
                uint32_t a2 = *(const uint32_t*)(wrow0 + tg * 4 + 16);
                uint32_t a3 = *(const uint32_t*)(wrow0 + 8 * LDW + tg * 4 + 16);
                const float wsc0 = s_ws[(wm * 16 + gid) * 2 + kb / 64];
                const float wsc1 = s_ws[(wm * 16 + gid + 8) * 2 + kb / 64];
#pragma unroll
                for (int s = 0; s < TS; s++) {
                    const int tb = wn * (NT / 2) + s * 8;
                    const int8_t* xcol = s_x + (tb + gid) * LDX + kb;
                    uint32_t b0 = *(const uint32_t*)(xcol + tg * 4);
                    uint32_t b1 = *(const uint32_t*)(xcol + tg * 4 + 16);
                    int d0, d1, d2, d3;
                    mma_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                    const float xs0 = s_xs[(tb + tg * 2) * 4 + cc];
                    const float xs1 = s_xs[(tb + tg * 2 + 1) * 4 + cc];
                    acc[s][0] += wsc0 * xs0 * (float)d0;
                    acc[s][1] += wsc0 * xs1 * (float)d1;
                    acc[s][2] += wsc1 * xs0 * (float)d2;
                    acc[s][3] += wsc1 * xs1 * (float)d3;
                }
            }
        } else {
#pragma unroll
            for (int gg = 0; gg < 2; gg++) {
                const int kb = gg * 64;
                const int8_t* wrow0 = s_w + (wm * 16 + gid) * LDW + kb;
                uint32_t a0 = *(const uint32_t*)(wrow0 + tg * 4);
                uint32_t a1 = *(const uint32_t*)(wrow0 + 8 * LDW + tg * 4);
                uint32_t a2 = *(const uint32_t*)(wrow0 + tg * 4 + 16);
                uint32_t a3 = *(const uint32_t*)(wrow0 + 8 * LDW + tg * 4 + 16);
                uint32_t a4 = *(const uint32_t*)(wrow0 + tg * 4 + 32);
                uint32_t a5 = *(const uint32_t*)(wrow0 + 8 * LDW + tg * 4 + 32);
                uint32_t a6 = *(const uint32_t*)(wrow0 + tg * 4 + 48);
                uint32_t a7 = *(const uint32_t*)(wrow0 + 8 * LDW + tg * 4 + 48);
                const float wsc0 = s_ws[(wm * 16 + gid) * 2 + gg];
                const float wsc1 = s_ws[(wm * 16 + gid + 8) * 2 + gg];
#pragma unroll
                for (int s = 0; s < TS; s++) {
                    const int tb = wn * (NT / 2) + s * 8;
                    const int8_t* xcol = s_x + (tb + gid) * LDX + kb;
                    uint32_t b0 = *(const uint32_t*)(xcol + tg * 4);
                    uint32_t b1 = *(const uint32_t*)(xcol + tg * 4 + 16);
                    uint32_t b2 = *(const uint32_t*)(xcol + tg * 4 + 32);
                    uint32_t b3 = *(const uint32_t*)(xcol + tg * 4 + 48);
                    int d0, d1, d2, d3;
                    mma_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                    mma_s8_acc(d0, d1, d2, d3, a4, a5, a6, a7, b2, b3);
                    const float xs0 = s_xs[(tb + tg * 2) * 2 + gg];
                    const float xs1 = s_xs[(tb + tg * 2 + 1) * 2 + gg];
                    acc[s][0] += wsc0 * xs0 * (float)d0;
                    acc[s][1] += wsc0 * xs1 * (float)d1;
                    acc[s][2] += wsc1 * xs0 * (float)d2;
                    acc[s][3] += wsc1 * xs1 * (float)d3;
                }
            }
        }
    }

    const int64_t row0 = r0 + wm * 16 + gid;
#pragma unroll
    for (int s = 0; s < TS; s++) {
        const int tok0 = t0 + wn * (NT / 2) + s * 8 + tg * 2;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int64_t row = row0 + (e >= 2 ? 8 : 0);
            int tok = tok0 + (e & 1);
            if (row < rows && tok < T) {
                if constexpr (ATOMIC)
                    atomicAdd(&y[(size_t)tok * rows + row], acc[s][e]);
                else
                    y[(size_t)tok * rows + row] = acc[s][e];
            }
        }
    }
}

// pure-read SOL: stream the weight bytes as uint4, fold to defeat DCE
__global__ void k_sol_read(const uint4* __restrict__ p, size_t n, unsigned* sink) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = 0;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xDEADBEEFu) *sink = acc;
}

template <typename F> static double timeit(F&& fn, int reps) {
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0));
    for (int r = 0; r < reps; r++) fn();
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return (double)ms / reps;
}

static std::vector<float> rand_vec(size_t n, unsigned seed) {
    std::vector<float> v(n);
    unsigned s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        v[i] = ((s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
    }
    return v;
}

template <int NT, bool XG64, bool ATOMIC>
static void launch_verify(const q27::DevTensor& w, const int8_t* nat, const float* xs, float* y,
                          int T, int z) {
    constexpr int MR = 64, KS = 128, LDW = KS + 16, LDX = KS + 16;
    constexpr int XSC = XG64 ? 2 : 4;
    const size_t SM = (size_t)MR * LDW + (size_t)NT * LDX + (MR * 2 + NT * XSC) * 4;
    static bool attr[2] = {false, false};
    int ai = XG64 ? 1 : 0;
    if (!attr[ai]) {
        CUDA_CHECK(cudaFuncSetAttribute(k_mma_verify<NT, XG64, ATOMIC>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
        attr[ai] = true;
    }
    int n_stages = (int)(w.cols / KS);
    int spz = (n_stages + z - 1) / z;
    dim3 grid((unsigned)((T + NT - 1) / NT), (unsigned)((w.rows + MR - 1) / MR), (unsigned)z);
    k_mma_verify<NT, XG64, ATOMIC><<<grid, 256, SM>>>((const uint8_t*)w.data,
                                                      (const __half*)w.scales, nat, xs, y,
                                                      w.rows, w.cols, T, spz);
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s model.q27\n", argv[0]); return 1; }
    q27::Model m = q27::Model::open(argv[1]);
    q27::DeviceModel dm(m);
    const int T = 16;
    const int64_t MAXC = 17408;

    // 16 activation lanes: contiguous [16][*] buffers + per-lane XQuant views
    int8_t *d_nat, *d_nat64;
    uint2* d_eo;
    float *d_scale, *d_s64;
    int* d_isum;
    CUDA_CHECK(cudaMalloc(&d_nat, (size_t)T * MAXC));
    CUDA_CHECK(cudaMalloc(&d_nat64, (size_t)T * MAXC));
    CUDA_CHECK(cudaMalloc(&d_eo, (size_t)T * (MAXC / 8) * sizeof(uint2)));
    CUDA_CHECK(cudaMalloc(&d_scale, (size_t)T * (MAXC / 32) * 4));
    CUDA_CHECK(cudaMalloc(&d_s64, (size_t)T * (MAXC / 64) * 4));
    CUDA_CHECK(cudaMalloc(&d_isum, (size_t)T * (MAXC / 32) * 4));
    float* d_x;
    CUDA_CHECK(cudaMalloc(&d_x, MAXC * 4));

    struct Shape {
        const char* tag;
        const char* names[4];
    };
    Shape shapes[2] = {
        {"ffn_gate 17408x5120",
         {"blk.0.ffn_gate.weight", "blk.1.ffn_gate.weight", "blk.2.ffn_gate.weight",
          "blk.4.ffn_gate.weight"}},
        {"ffn_down 5120x17408",
         {"blk.0.ffn_down.weight", "blk.1.ffn_down.weight", "blk.2.ffn_down.weight",
          "blk.4.ffn_down.weight"}},
    };

    unsigned* d_sink;
    CUDA_CHECK(cudaMalloc(&d_sink, 4));

    for (auto& sh : shapes) {
        const q27::DevTensor* w[4];
        for (int i = 0; i < 4; i++) w[i] = &dm.upload(sh.names[i]);
        const int64_t rows = w[0]->rows, cols = w[0]->cols;
        printf("== %s (rows=%ld cols=%ld, q4 bytes/inst=%.1fMB)\n", sh.tag, (long)rows,
               (long)cols, (rows * (cols / 2.0) + rows * (cols / 64.0) * 2) / 1e6);

        // quantize 16 distinct activation vectors at this shape's cols stride
        q27k::XQuant qv[16];
        for (int i = 0; i < T; i++) {
            std::vector<float> x = rand_vec(cols, 1234 + i * 77);
            CUDA_CHECK(cudaMemcpy(d_x, x.data(), cols * 4, cudaMemcpyHostToDevice));
            qv[i].nat = d_nat + (size_t)i * cols;
            qv[i].eo = d_eo + (size_t)i * (cols / 8);
            qv[i].scale = d_scale + (size_t)i * (cols / 32);
            qv[i].isum = d_isum + (size_t)i * (cols / 32);
            qv[i].nat64 = d_nat64 + (size_t)i * cols;
            qv[i].s64 = d_s64 + (size_t)i * (cols / 64);
            q27k::quantize_x(d_x, cols, qv[i]);
            q27k::quantize_x_g64(d_x, cols, qv[i]);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        float* ys[16];
        for (int i = 0; i < T; i++)
            CUDA_CHECK(cudaMalloc(&ys[i], rows * 4));
        float* d_y;
        CUDA_CHECK(cudaMalloc(&d_y, (size_t)T * rows * 4));

        const double wbytes = rows * (cols / 2.0) + rows * (cols / 64.0) * 2;
        int rot = 0;

        // (a) current engine, single pass: gemv_q4_n nb=8 (8 lanes, W=8 verify)
        double ms_g8 = timeit(
            [&] {
                q27k::gemv_q4_n((const uint8_t*)w[rot & 3]->data,
                                (const __half*)w[rot & 3]->scales, qv, 8, ys, rows, cols);
                rot++;
            },
            40);
        printf("  gemv_q4_n nb=8    : %8.3f ms  %7.1f GB/s\n", ms_g8, wbytes / ms_g8 / 1e6);

        // (a') 16 lanes = two passes on DISTINCT instances (L2-cold both; the
        // first bench cut called both passes on one instance -- pass 2 ran
        // from L2 at 47MB < 96MB, flattering the baseline)
        double ms_gemv = timeit(
            [&] {
                q27k::gemv_q4_n((const uint8_t*)w[rot & 3]->data,
                                (const __half*)w[rot & 3]->scales, qv, 8, ys, rows, cols);
                q27k::gemv_q4_n((const uint8_t*)w[(rot + 1) & 3]->data,
                                (const __half*)w[(rot + 1) & 3]->scales, qv + 8, 8, ys + 8,
                                rows, cols);
                rot += 2;
            },
            40);
        printf("  gemv_q4_n nb=8 x2 : %8.3f ms  (16 lanes, 2 cold passes)\n", ms_gemv);

        // (b) SOL pure read
        double ms_sol = timeit(
            [&] {
                k_sol_read<<<2048, 256>>>((const uint4*)w[rot & 3]->data,
                                          rows * (cols / 2) / 16, d_sink);
                rot++;
            },
            40);
        printf("  SOL uint4 read    : %8.3f ms  %7.1f GB/s\n", ms_sol,
               rows * (cols / 2.0) / ms_sol / 1e6);

        // (c) prefill-shaped NT=128 at T=16 (collapsed baseline)
        double ms_c = timeit(
            [&] {
                launch_verify<128, true, false>(*w[rot & 3], d_nat64, d_s64, d_y, T, 1);
                rot++;
            },
            40);
        printf("  mma NT=128 z=1    : %8.3f ms  %7.1f GB/s\n", ms_c, wbytes / ms_c / 1e6);

        // (d) NT=16 fork, K-split sweep
        for (int z : {1, 2, 4, 8, 16}) {
            double ms_d;
            if (z == 1)
                ms_d = timeit(
                    [&] {
                        launch_verify<16, true, false>(*w[rot & 3], d_nat64, d_s64, d_y, T, 1);
                        rot++;
                    },
                    40);
            else
                ms_d = timeit(
                    [&] {
                        // atomic accumulate; skip the zeroing memset in the perf
                        // loop (values unused; add order irrelevant to timing)
                        launch_verify<16, true, true>(*w[rot & 3], d_nat64, d_s64, d_y, T, z);
                        rot++;
                    },
                    40);
            printf("  mma NT=16  z=%d    : %8.3f ms  %7.1f GB/s\n", z, ms_d, wbytes / ms_d / 1e6);
        }

        // numerics: dp4a gemv reference (lane 0) vs NT=16 XG64=false z=1, and
        // vs z=4 atomic (adds K-slice reorder)
        q27k::gemv_q4((const uint8_t*)w[0]->data, (const __half*)w[0]->scales, qv[0], ys[0],
                      rows, cols);
        launch_verify<16, false, false>(*w[0], d_nat, d_scale, d_y, T, 1);
        std::vector<float> ref(rows), got(rows);
        CUDA_CHECK(cudaMemcpy(ref.data(), ys[0], rows * 4, cudaMemcpyDeviceToHost));
        double rms = 0;
        for (int64_t r = 0; r < rows; r++) rms += (double)ref[r] * ref[r];
        rms = sqrt(rms / rows) + 1e-12;
        // error normalized by rms(ref), not per-element |ref| (near-zero
        // outputs of random dots inflate per-element rel err meaninglessly)
        auto maxerr = [&](const char* tag) {
            CUDA_CHECK(cudaMemcpy(got.data(), d_y, rows * 4, cudaMemcpyDeviceToHost));
            double me = 0;
            for (int64_t r = 0; r < rows; r++) {
                double d = fabs((double)got[r] - ref[r]);
                if (d > me) me = d;
            }
            printf("  numerics vs dp4a gemv (xg32) %s: maxabs %.2e (rms(ref) %.2e, rel %.2e)\n",
                   tag, me, rms, me / rms);
        };
        maxerr("z=1");
        CUDA_CHECK(cudaMemset(d_y, 0, (size_t)T * rows * 4));
        launch_verify<16, false, true>(*w[0], d_nat, d_scale, d_y, T, 4);
        maxerr("z=4 atomic");

        for (int i = 0; i < T; i++) CUDA_CHECK(cudaFree(ys[i]));
        CUDA_CHECK(cudaFree(d_y));
    }
    printf("GO bar: NT=16 leg >= 70%% of SOL GB/s on both shapes (reopen condition).\n");
    return 0;
}
