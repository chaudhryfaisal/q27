// mma16.cuh -- the NT=16 K-split q4 MMA GEMM extracted VERBATIM from
// tools/mma16_bench.cu so more than one bench can drive it. No edits to the
// kernel: this is the exact code that measured 1075-1101 GB/s (73-76% of SOL)
// on ffn_gate/ffn_down. Extracted 2026-07-13 for the round-shape-mix probe
// (the GEMM-verify pivot's P0: does it hold on the OTHER 399 weights?).
#pragma once
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include "../src/device_model.h"
#include "../src/kernels.cuh"
#include "../src/loader.h"

#ifndef CUDA_CHECK
#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__);\
            exit(1);                                                           \
        }                                                                      \
    } while (0)
#endif

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

