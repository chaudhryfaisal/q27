// fp4 W4A4 prefill GEMM (Q27_PREFILL=fp4) -- ninfer-steals phase 2 stage B.
//
// The kernel is the microbench_mxf4 GEMM (phase 0, ratio table in the
// BUILDLOG 2026-08-15) with two deltas: fp32 output in the mmT y layout, and
// M-tail handling by padding T up to the 128-row tile (pad rows carry ZERO
// ue4m3 scales, so their contributions are exactly 0 and their stores are
// guarded; T as small as 2 reaches mmT under the server's
// Q27_PF_BATCH_MIN=2). fp4 handles EVERY T when enabled -- a T-dependent
// int/fp4 switch would make prefill numerics depend on chunk alignment, and
// at tiny T both paths are weight-bandwidth-bound at the same ~4.5 bpw.
//
// nvfp4 semantics (shared with tools/microbench_mxf4.cu and the repack
// --pf4 leg): e2m1 codes 2/byte even=low, ue4m3 scale per 16 elems along K,
// activation scale = rne(absmax/6) with the reciprocal taken from the
// ROUNDED scale. mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X
// .m16n8k64, sm_120a ONLY -- this TU compiles with MXF4FLAGS, nothing here
// may be referenced from device code in other TUs.

#include "pf4.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace q27k {
namespace {

#define PF4_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "pf4 CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
  exit(1); } } while (0)

constexpr int BM = 128, BN = 128, BK = 256;
constexpr int WARPS_M = 4, WARPS_N = 2, NWARP = WARPS_M * WARPS_N, NTHREAD = NWARP * 32;
constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
constexpr int MMA_M = WM / 16, MMA_N = WN / 8;
constexpr int K64_PER_TILE = BK / 64;
constexpr int ROW_BYTES = BK / 2;
constexpr int SEGS = ROW_BYTES / 16;
constexpr int STAGES = 2;

struct Smem {
    uint8_t  a_codes[STAGES][BM * ROW_BYTES];
    uint8_t  b_codes[STAGES][BN * ROW_BYTES];
    uint32_t a_scale[STAGES][BM * K64_PER_TILE];
    uint32_t b_scale[STAGES][BN * K64_PER_TILE];
};

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                 "r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::
                 "r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
__device__ __forceinline__ void ldmx4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                      unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldmx2(unsigned& r0, unsigned& r1, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void mma_nvfp4(float& c0, float& c1, float& c2, float& c3,
                                          unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                          unsigned b0, unsigned b1, unsigned sfa, unsigned sfb) {
    constexpr unsigned short kBid = 0, kTid = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(sfa), "h"(kBid), "h"(kTid), "r"(sfb), "h"(kBid), "h"(kTid));
}
static_assert((SEGS & (SEGS - 1)) == 0, "swz masks with SEGS-1");
__device__ __forceinline__ int swz(int row, int byte) {
    const int seg = byte >> 4;
    return ((seg ^ (row & (SEGS - 1))) << 4) + (byte & 15);
}

// A = activations (Mp x K, Mp = T padded to 128; pad-row SCALES are zero so
// pad contributions vanish), B = fp4 sidecar weight (N x K). y fp32
// [t*N + n], stores guarded by the REAL T.
__global__ __launch_bounds__(NTHREAD, 1)
void k_pf4_gemm(const uint8_t* __restrict__ a_codes, const uint8_t* __restrict__ a_scales,
                const uint8_t* __restrict__ b_codes, const uint8_t* __restrict__ b_scales,
                float* __restrict__ y, int M, int N, int K) {
    extern __shared__ Smem smem[];
    Smem& sm = smem[0];

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int k_tiles = K / BK;
    const int k64_row = K / 64;
    const int row_bytes_g = K / 2;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WARPS_N, wn = warp - wm * WARPS_N;

    const int a_mat = lane >> 3;
    const int a_row = (lane & 7) + ((a_mat & 1) << 3);
    const int a_byte = (a_mat >> 1) * 16;
    const int b_row = lane & 7;
    const int b_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row = lane >> 2;

    auto stage_tile = [&](int stage, int kt) {
        for (int t = tid; t < BM * SEGS; t += NTHREAD) {
            const int r = t / SEGS, seg = t - r * SEGS;
            cp16(sm.a_codes[stage] + r * ROW_BYTES + swz(r, seg * 16),
                 a_codes + (int64_t)(m0 + r) * row_bytes_g + kt * ROW_BYTES + seg * 16);
        }
        for (int t = tid; t < BN * SEGS; t += NTHREAD) {
            const int r = t / SEGS, seg = t - r * SEGS;
            cp16(sm.b_codes[stage] + r * ROW_BYTES + swz(r, seg * 16),
                 b_codes + (int64_t)(n0 + r) * row_bytes_g + kt * ROW_BYTES + seg * 16);
        }
        for (int t = tid; t < BM * K64_PER_TILE; t += NTHREAD) {
            const int r = t / K64_PER_TILE, k64 = t - r * K64_PER_TILE;
            cp4(&sm.a_scale[stage][r * K64_PER_TILE + k64],
                a_scales + ((int64_t)(m0 + r) * k64_row + kt * K64_PER_TILE + k64) * 4);
        }
        for (int t = tid; t < BN * K64_PER_TILE; t += NTHREAD) {
            const int r = t / K64_PER_TILE, k64 = t - r * K64_PER_TILE;
            cp4(&sm.b_scale[stage][r * K64_PER_TILE + k64],
                b_scales + ((int64_t)(n0 + r) * k64_row + kt * K64_PER_TILE + k64) * 4);
        }
        cp_commit();
    };

    stage_tile(0, 0);
    // The prologue must always commit STAGES groups (empty when there is no
    // second tile) -- cp_wait<STAGES-1> below completes the OLDEST group only
    // when STAGES are in flight. Single-tile K computes on unstaged smem
    // otherwise (caught by microbench_mxf4's exhaustive gate on 08-15).
    if (k_tiles > 1) stage_tile(1, 1);
    else cp_commit();

    float acc[MMA_M][MMA_N][4] = {};

    for (int kt = 0; kt < k_tiles; ++kt) {
        const int stage = kt & (STAGES - 1);
        cp_wait<STAGES - 1>();
        __syncthreads();

#pragma unroll
        for (int k64 = 0; k64 < K64_PER_TILE; ++k64) {
            unsigned af[MMA_M][4], bf[MMA_N][2], sa[MMA_M], sb[MMA_N];
#pragma unroll
            for (int i = 0; i < MMA_M; ++i) {
                const int r = wm * WM + i * 16 + a_row;
                ldmx4(af[i][0], af[i][1], af[i][2], af[i][3],
                      smem_u32(sm.a_codes[stage] + r * ROW_BYTES + swz(r, k64 * 32 + a_byte)));
                sa[i] = sm.a_scale[stage][(wm * WM + i * 16 + sfa_row) * K64_PER_TILE + k64];
            }
#pragma unroll
            for (int j = 0; j < MMA_N; ++j) {
                const int r = wn * WN + j * 8 + b_row;
                ldmx2(bf[j][0], bf[j][1],
                      smem_u32(sm.b_codes[stage] + r * ROW_BYTES + swz(r, k64 * 32 + b_byte)));
                sb[j] = sm.b_scale[stage][(wn * WN + j * 8 + sfb_row) * K64_PER_TILE + k64];
            }
#pragma unroll
            for (int i = 0; i < MMA_M; ++i)
#pragma unroll
                for (int j = 0; j < MMA_N; ++j)
                    mma_nvfp4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
                              af[i][0], af[i][1], af[i][2], af[i][3],
                              bf[j][0], bf[j][1], sa[i], sb[j]);
        }

        __syncthreads();
        const int nxt = kt + STAGES;
        if (nxt < k_tiles) stage_tile(stage, nxt);
        else cp_commit(); // keep the commit-group count per iteration
    }

    // fp32 stores in mmT's y[t*N + n] layout; float2 is 8B-aligned because N
    // and the fragment column are both even.
    const int cr = lane >> 2, cc = 2 * (lane & 3);
#pragma unroll
    for (int i = 0; i < MMA_M; ++i) {
        const int gm0 = m0 + wm * WM + i * 16 + cr;
#pragma unroll
        for (int j = 0; j < MMA_N; ++j) {
            const int gn = n0 + wn * WN + j * 8 + cc;
            if (gm0 < M)
                *reinterpret_cast<float2*>(y + (int64_t)gm0 * N + gn) =
                    make_float2(acc[i][j][0], acc[i][j][1]);
            if (gm0 + 8 < M)
                *reinterpret_cast<float2*>(y + (int64_t)(gm0 + 8) * N + gn) =
                    make_float2(acc[i][j][2], acc[i][j][3]);
        }
    }
}

// fp32 -> nvfp4, one 16-elem group per thread (the microbench quantizer).
__global__ void k_pf4_quant(const float* __restrict__ x, uint8_t* __restrict__ codes,
                            uint8_t* __restrict__ scales, int64_t groups, int K) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= groups) return;
    const int groups_row = K / 16;
    const int64_t m = t / groups_row;
    const int g = (int)(t - m * groups_row);
    const float* p = x + m * K + g * 16;

    float v[16], amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) { v[i] = p[i]; amax = fmaxf(amax, fabsf(v[i])); }
    // The constructor converts SATFINITE: an overflowing amax/6 clamps to the
    // max finite e4m3 (0x7e = 448), never the NaN byte -- no poison scale.
    __nv_fp8_e4m3 s8 = __nv_fp8_e4m3(amax / 6.0f);
    const uint8_t sbyte = s8.__x & 0x7f; // ue4m3: sign bit always 0 here
    __nv_fp8_e4m3 sd; sd.__x = sbyte;
    const float sdec = float(sd);
    const float inv = sdec > 0.f ? 1.0f / sdec : 0.f;

    float2 f2[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) f2[i] = make_float2(v[2 * i] * inv, v[2 * i + 1] * inv);
    uint32_t lo, hi;
    asm volatile("{\n.reg .b8 b0; .reg .b8 b1; .reg .b8 b2; .reg .b8 b3;\n"
                 ".reg .b8 b4; .reg .b8 b5; .reg .b8 b6; .reg .b8 b7;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
                 "mov.b32 %0, {b0,b1,b2,b3};\nmov.b32 %1, {b4,b5,b6,b7};\n}\n"
                 : "=r"(lo), "=r"(hi)
                 : "f"(f2[0].x), "f"(f2[0].y), "f"(f2[1].x), "f"(f2[1].y),
                   "f"(f2[2].x), "f"(f2[2].y), "f"(f2[3].x), "f"(f2[3].y),
                   "f"(f2[4].x), "f"(f2[4].y), "f"(f2[5].x), "f"(f2[5].y),
                   "f"(f2[6].x), "f"(f2[6].y), "f"(f2[7].x), "f"(f2[7].y));
    *reinterpret_cast<uint2*>(codes + m * (K / 2) + g * 8) = make_uint2(lo, hi);
    scales[m * groups_row + g] = sbyte;
}

} // namespace

bool pf4_on() {
    // Probe the CURRENT device (magic-static: thread-safe init). q27 is a
    // one-GPU-per-process engine, so this resolves once; keying on the
    // current device rather than 0 keeps it honest if that ever changes.
    static const int arch_ok = [] {
        int dev = 0;
        cudaDeviceProp p{};
        if (cudaGetDevice(&dev) != cudaSuccess) return 0;
        if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return 0;
        return p.major == 12 ? 1 : 0;
    }();
    // Re-read per call (tests flip paths via setenv; a getenv is noise next
    // to a kernel launch -- the prefill.cu convention).
    const char* e = getenv("Q27_PREFILL");
    return arch_ok == 1 && e && !strcmp(e, "fp4");
}

bool pf4_instrument() {
    const char* e = getenv("Q27_PF4_INSTRUMENT");
    return pf4_on() && e && !strcmp(e, "1");
}

void pf4_gemm_T(const void* wc, const void* ws, const float* xT, void* ac, void* as,
                float* y, int64_t rows, int64_t cols, int T, cudaStream_t st) {
    if (rows % BN != 0 || cols % BK != 0) {
        fprintf(stderr, "pf4_gemm_T: (%ld x %ld) not tileable\n", (long)rows, (long)cols);
        exit(1);
    }
    static const bool smem_set = [] { // magic-static: thread-safe one-shot
        PF4_CK(cudaFuncSetAttribute(k_pf4_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)sizeof(Smem)));
        return true;
    }();
    (void)smem_set;
    const int Tp = (T + BM - 1) & ~(BM - 1);
    // pad rows: zero SCALES only -- zero ue4m3 scale makes every pad
    // contribution exactly 0 whatever the code bytes hold.
    if (Tp > T)
        PF4_CK(cudaMemsetAsync((uint8_t*)as + (size_t)T * (cols / 16), 0,
                               (size_t)(Tp - T) * (cols / 16), st));
    const int64_t groups = (int64_t)T * cols / 16;
    k_pf4_quant<<<(unsigned)((groups + 255) / 256), 256, 0, st>>>(
        xT, (uint8_t*)ac, (uint8_t*)as, groups, (int)cols);
    k_pf4_gemm<<<dim3((unsigned)(rows / BN), (unsigned)(Tp / BM)), NTHREAD, sizeof(Smem), st>>>(
        (const uint8_t*)ac, (const uint8_t*)as, (const uint8_t*)wc, (const uint8_t*)ws, y,
        T, (int)rows, (int)cols);
    PF4_CK(cudaGetLastError());
}

} // namespace q27k
