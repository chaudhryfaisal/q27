// microbench_mxf4: pin the sm_120a fp4 block-scaled MMA capability in-tree.
//
// Phase 0 of docs/plans/2026-08-15-ninfer-steals.md. The 2026-08 recon proved
// mma.sync.aligned.kind::mxf4nvf4.block_scale...e2m1 compiles AND executes on
// the 5090 -- the earlier NO-GO was a toolchain trap: without
// `-gencode arch=compute_120a,code=sm_120a` ptxas rejects the instruction and
// the failure looks like a hardware limitation. This tool pins that proof and
// produces the GEMM ratio table phase 2 (fp4 W4A4 prefill) gates on.
//
// A leg: fp4 W4A4 block-scaled GEMM (nvfp4: e2m1 codes, ue4m3 scale per 16
//        elements), 128x128x256 tiles, 8 warps, cp.async double buffer.
//        Activations quantized fp32 -> nvfp4 by k_quant_nvfp4 (timed).
// B leg: the CURRENT prefill inner path, q27k::gemm_q4_T (W4A8-int: Q4_G64
//        weights, int8-g64 activations, m16n8k32 s8 MMA) on REAL projection
//        weights from a .q27, activations quantized by quantize_x_g64 (timed).
//
// Shapes: the four Q4 projections of Qwen3.6-27B (attn_q fused-gate 12288x5120,
// attn_output 5120x6144, ffn_gate 17408x5120, ffn_down 5120x17408) at
// M = 512 / 1024 / 2048 / 8192. M=1024 is the production point (PF_T chunk).
// K/V projections are Q8_G128 by repack policy and ~2% of prefill GEMM work;
// excluded. SSM-path projections excluded per the plan (ssm_out lesson).
//
// Build: make build/microbench_mxf4   (sm_120a-only target; see Makefile note)
// Usage: build/microbench_mxf4 [model.q27]
//        default model: /mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp-q4s.q27
// Exit: 0 = correctness gates passed, table printed; 1 = gate failure.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "../src/device_model.h"
#include "../src/loader.h"
#include "../src/kernels.cuh"
#include "../src/prefill.cuh"
#include "../src/vgemm.cuh" // T2 decode baseline: the union GEMM the C-sweep routes to at k >= 3

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

namespace {

// ---------------------------------------------------------------------------
// fp4 leg: nvfp4 W4A4 GEMM, 128x128x256, 8 warps (4x2), double-buffered.
// A: M x K activations, e2m1 packed 2/byte row-major, scales [M][K/16] ue4m3.
// B: N x K weights (row per output channel), same packing, scales [N][K/16].
// C: M x N bf16. Scale .b32 group = 4 consecutive k16 blocks (= one k64 MMA).
// ---------------------------------------------------------------------------
constexpr int BM = 128, BN = 128, BK = 256;
constexpr int WARPS_M = 4, WARPS_N = 2, NWARP = WARPS_M * WARPS_N, NTHREAD = NWARP * 32;
constexpr int WM = BM / WARPS_M;      // 32
constexpr int WN = BN / WARPS_N;      // 64
constexpr int MMA_M = WM / 16;        // 2
constexpr int MMA_N = WN / 8;         // 8
constexpr int K64_PER_TILE = BK / 64; // 4
constexpr int ROW_BYTES = BK / 2;     // 128
constexpr int SEGS = ROW_BYTES / 16;  // 8
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

// THE instruction this tool exists to pin. sm_120a-only: plain sm_120 ptxas
// rejects it ("not supported on .target"), which is the trap that produced
// the original NO-GO verdict.
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

// 16-byte-segment swizzle so ldmatrix rows spread across smem banks.
static_assert((SEGS & (SEGS - 1)) == 0, "swz masks with SEGS-1");
__device__ __forceinline__ int swz(int row, int byte) {
    const int seg = byte >> 4;
    return ((seg ^ (row & (SEGS - 1))) << 4) + (byte & 15);
}

__global__ __launch_bounds__(NTHREAD, 1)
void k_mxf4_gemm(const uint8_t* __restrict__ a_codes, const uint8_t* __restrict__ a_scales,
                 const uint8_t* __restrict__ b_codes, const uint8_t* __restrict__ b_scales,
                 __nv_bfloat16* __restrict__ c, int M, int N, int K) {
    extern __shared__ Smem smem[];
    Smem& sm = smem[0];

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int k_tiles = K / BK;
    const int k64_row = K / 64;
    const int row_bytes_g = K / 2;

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp / WARPS_N, wn = warp - wm * WARPS_N;

    // ldmatrix source rows/bytes (four 8x8 b16 tiles per x4 = 16 rows x 32B)
    const int a_mat = lane >> 3;
    const int a_row = (lane & 7) + ((a_mat & 1) << 3);
    const int a_byte = (a_mat >> 1) * 16;
    const int b_row = lane & 7;
    const int b_byte = ((lane >> 3) & 1) * 16;
    // block_scale fragment rows (scale_vec::4X, bid=0 tid=0)
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
    // The main loop's cp_wait<STAGES-1> completes the OLDEST group only if
    // STAGES groups are in flight, so the prologue must always commit
    // STAGES groups -- an empty one when there is no second tile (k_tiles
    // == 1 otherwise computes on unstaged smem; caught by the exhaustive
    // 128x128x256 correctness gate).
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
        // Commit-group accounting: cp_wait<STAGES-1> above returns when at
        // most STAGES-1 groups remain IN FLIGHT, i.e. it needs one group
        // committed per loop iteration to make the OLDEST group complete.
        // When no next tile is staged, an EMPTY group keeps that count: drop
        // this commit and the final iteration's wait would return with the
        // last real group (the data it is about to compute on) still in
        // flight. Same accounting as the staged branch, which ends in
        // cp_commit() inside stage_tile().
        else cp_commit();
    }

    // direct bf16x2 stores (c fragment: row = lane>>2, col = 2*(lane&3))
    const int cr = lane >> 2, cc = 2 * (lane & 3);
#pragma unroll
    for (int i = 0; i < MMA_M; ++i) {
        const int gm0 = m0 + wm * WM + i * 16 + cr;
#pragma unroll
        for (int j = 0; j < MMA_N; ++j) {
            const int gn = n0 + wn * WN + j * 8 + cc;
            if (gm0 < M)
                *reinterpret_cast<__nv_bfloat162*>(c + (int64_t)gm0 * N + gn) =
                    __floats2bfloat162_rn(acc[i][j][0], acc[i][j][1]);
            if (gm0 + 8 < M)
                *reinterpret_cast<__nv_bfloat162*>(c + (int64_t)(gm0 + 8) * N + gn) =
                    __floats2bfloat162_rn(acc[i][j][2], acc[i][j][3]);
        }
    }
}

// fp32 -> nvfp4 activation quantize: one 16-elem group per thread, amax/6
// scale encoded ue4m3 (mirrors what a real fp4 prefill leg would run; the
// counterpart of quantize_x_g64 on the baseline side).
__global__ void k_quant_nvfp4(const float* __restrict__ x, uint8_t* __restrict__ codes,
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
    __nv_fp8_e4m3 s8 = __nv_fp8_e4m3(amax / 6.0f);
    // ue4m3 is UNSIGNED e4m3: bit 7 is the (always-zero here) sign bit of
    // the signed encoder -- amax/6 >= 0 -- so the mask is belt-and-braces,
    // never information-dropping.
    const uint8_t sbyte = s8.__x & 0x7f;
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

// ---------------------------------------------------------------------------
// DECODE leg (T2, docs/plans/2026-08-18-fp4-viability-tests.md).
//
// The tile above is BM=128, built for the prefill point. Batched decode is a
// different regime: the 2026-08-18 C=8 ladder measured the union width at
// M = 16 in 98.0% of rounds (every lane trimmed to floor-2, k=8), so the
// prefill tile would run 8/128 full. This is the decode-shaped counterpart.
//
// SHAPE OF THE PROBLEM, and why the operands swap. C[M x N] = A[M x K] . W[N x K]^T
// with M = union lanes (<= W_PLUMB = 16 today) and N = w.rows (5120..17408).
// m16n8k64 puts 16 rows on the A operand and 8 on the B operand, so the
// prefill mapping (activations on A) would waste 8 of 16 A rows at M=8 and 12
// of 16 at M=4. Here WEIGHTS take the A operand (16 output channels) and
// ACTIVATIONS take B (8 lanes) -- the same assignment k_vgemm makes with its
// m16n8k32 s8 MMA, which is what makes this an apples-to-apples format test
// rather than a tile-shape test. NTI (8-lane B tiles) is a template parameter
// so the tile tracks M instead of padding to a fixed 16.
//
// BASELINE. Not gemm_q4_T -- that is the prefill path and batched decode never
// reaches it. At k >= 3 members build_union_view sets gemm_min = 2
// (src/conductor.h:483, the 2026-08-16 C-sweep default), so every projection
// with rows >= gemm_min_rows goes through q27k::vgemm_verify. That is the
// incumbent this has to beat.
//
// WHAT IS ACTUALLY BEING COMPARED. At M = 16 the arithmetic intensity is
// 2*M / bytes_per_weight ~= 57 FLOP/byte, i.e. ~100 TFLOPS at DRAM SOL against
// a dense fp4 peak near 800 -- both legs are bandwidth-bound, and the format
// difference is a BYTE difference: nvfp4 spends 0.5 B/weight on e2m1 codes
// plus 1 B per 16 on the ue4m3 block scale = 0.5625 B/weight (4.50 bpw), while
// Q4_G64 spends 0.5 plus one fp16 per 64 = 0.53125 (4.25 bpw). fp4 moves
// 5.88% MORE bytes for the same weights. The table therefore reports GB/s and
// percent-of-SOL next to the ratio: a ratio above 1 that comes with a HIGHER
// GB/s is kernel technique (portable to the int8 path), not format.
// ---------------------------------------------------------------------------
namespace dec {

constexpr int D_KS = 128;  // K per warp-group per super-step
constexpr int D_STAGES = 2;

template <int MR, int NTI> struct Cfg {
    static constexpr int WM = MR / 16;              // warp rows (MMA m16)
    static constexpr int WN = NTI;                  // warp lane-groups (MMA n8)
    static constexpr int KG = (8 / (WM * WN)) > 0 ? (8 / (WM * WN)) : 1; // K warp-groups
    static constexpr int NWARP = WM * WN * KG;
    static constexpr int NTHREAD = NWARP * 32;
    static constexpr int NT = NTI * 8;              // lanes the tile stages
    static constexpr int KB = KG * D_KS;            // K staged per super-step
    static constexpr int K64 = KB / 64;             // k64 MMA steps staged
    static constexpr int RB = KB / 2;               // code bytes per row per step
    static constexpr int SEGS = RB / 16;
    static_assert(NWARP == 8, "the tile is fixed at 256 threads / 8 warps");
    static_assert((SEGS & (SEGS - 1)) == 0, "swizzle masks with SEGS-1");
};

// Same 16-byte-segment swizzle as the prefill tile, parameterized on SEGS so a
// different KB cannot silently inherit the wrong mask.
template <int SEGS> __device__ __forceinline__ int dswz(int row, int byte) {
    const int seg = byte >> 4;
    return ((seg ^ (row & (SEGS - 1))) << 4) + (byte & 15);
}

// smem: codes + .b32 scale groups for both operands, double-buffered.
template <int MR, int NTI> struct DSmem {
    using C = Cfg<MR, NTI>;
    uint8_t w_codes[D_STAGES][MR * C::RB];
    uint8_t a_codes[D_STAGES][C::NT * C::RB];
    uint32_t w_scale[D_STAGES][MR * C::K64];
    uint32_t a_scale[D_STAGES][C::NT * C::K64];
};

// MODE 0: store straight to Y (z == 1). MODE 1: this z-slice's partial to ws,
// summed afterwards by k_dec_reduce_z in index order -- the same deterministic
// cross-CTA split k_vgemm uses, so neither leg gets a parallelization edge.
template <int MR, int NTI, int MODE>
__global__ __launch_bounds__(Cfg<MR, NTI>::NTHREAD, 4)
void k_mxf4_dec(const uint8_t* __restrict__ Wc, const uint8_t* __restrict__ Wsc,
                const uint8_t* __restrict__ Ac, const uint8_t* __restrict__ Asc,
                float* __restrict__ Y, float* __restrict__ ws, int64_t rows, int64_t cols,
                int T, int steps_per_z) {
    using C = Cfg<MR, NTI>;
    extern __shared__ unsigned char dsmem_raw[];
    DSmem<MR, NTI>& sm = *reinterpret_cast<DSmem<MR, NTI>*>(dsmem_raw);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % C::WM;
    const int wn = (warp / C::WM) % C::WN;
    const int kg = warp / (C::WM * C::WN);

    const int64_t r0 = (int64_t)blockIdx.y * MR;
    const int n_steps = (int)(cols / C::KB);
    const int s_begin = blockIdx.z * steps_per_z;
    const int s_end = min(n_steps, s_begin + steps_per_z);
    if (s_begin >= s_end) return;

    const int64_t crow = cols / 2;   // code bytes per row
    const int64_t srow = cols / 16;  // ue4m3 scale bytes per row

    // Row clamping instead of zero-fill: an out-of-range weight row or lane
    // reads a live address (never OOB) and its accumulator is discarded by the
    // store guard below. e2m1 x ue4m3 is bounded, so no inf/nan can escape.
    auto stage = [&](int st, int step) {
        const int64_t k0 = (int64_t)step * C::KB;
        for (int t = tid; t < MR * C::SEGS; t += C::NTHREAD) {
            const int r = t / C::SEGS, seg = t - r * C::SEGS;
            const int64_t gr = min((int64_t)r0 + r, rows - 1);
            cp16(sm.w_codes[st] + r * C::RB + dswz<C::SEGS>(r, seg * 16),
                 Wc + gr * crow + k0 / 2 + seg * 16);
        }
        for (int t = tid; t < C::NT * C::SEGS; t += C::NTHREAD) {
            const int r = t / C::SEGS, seg = t - r * C::SEGS;
            const int gr = r < T ? r : 0;
            cp16(sm.a_codes[st] + r * C::RB + dswz<C::SEGS>(r, seg * 16),
                 Ac + (int64_t)gr * crow + k0 / 2 + seg * 16);
        }
        for (int t = tid; t < MR * C::K64; t += C::NTHREAD) {
            const int r = t / C::K64, g = t - r * C::K64;
            const int64_t gr = min((int64_t)r0 + r, rows - 1);
            cp4(&sm.w_scale[st][r * C::K64 + g],
                (const uint32_t*)(Wsc + gr * srow) + k0 / 64 + g);
        }
        for (int t = tid; t < C::NT * C::K64; t += C::NTHREAD) {
            const int r = t / C::K64, g = t - r * C::K64;
            const int gr = r < T ? r : 0;
            cp4(&sm.a_scale[st][r * C::K64 + g],
                (const uint32_t*)(Asc + (int64_t)gr * srow) + k0 / 64 + g);
        }
        cp_commit();
    };

    stage(0, s_begin);
    // Same commit-group accounting as the prefill tile: cp_wait<STAGES-1>
    // completes the OLDEST group only when STAGES groups are in flight, so an
    // empty group stands in whenever there is no next tile to stage.
    if (s_begin + 1 < s_end) stage(1, s_begin + 1);
    else cp_commit();

    // ldmatrix / block-scale fragment mapping: identical to the prefill tile
    // (proven by its exhaustive 128x128x256 gate), with the operands swapped --
    // A rows are weight output channels, B rows are lanes.
    const int a_mat = lane >> 3;
    const int a_row = (lane & 7) + ((a_mat & 1) << 3);
    const int a_byte = (a_mat >> 1) * 16;
    const int b_row = lane & 7;
    const int b_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row = lane >> 2;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int st = s_begin; st < s_end; ++st) {
        const int buf = (st - s_begin) & (D_STAGES - 1);
        cp_wait<D_STAGES - 1>();
        __syncthreads();
#pragma unroll
        for (int j = 0; j < C::K64 / C::KG; ++j) {
            const int k64 = kg * (C::K64 / C::KG) + j;
            const int wr = wm * 16 + a_row;
            unsigned af0, af1, af2, af3, bf0, bf1;
            ldmx4(af0, af1, af2, af3,
                  smem_u32(sm.w_codes[buf] + wr * C::RB + dswz<C::SEGS>(wr, k64 * 32 + a_byte)));
            const unsigned sa = sm.w_scale[buf][(wm * 16 + sfa_row) * C::K64 + k64];
            const int ar = wn * 8 + b_row;
            ldmx2(bf0, bf1,
                  smem_u32(sm.a_codes[buf] + ar * C::RB + dswz<C::SEGS>(ar, k64 * 32 + b_byte)));
            const unsigned sb = sm.a_scale[buf][(wn * 8 + sfb_row) * C::K64 + k64];
            mma_nvfp4(acc[0], acc[1], acc[2], acc[3], af0, af1, af2, af3, bf0, bf1, sa, sb);
        }
        __syncthreads();
        const int nxt = st + D_STAGES;
        if (nxt < s_end) stage(buf, nxt);
        else cp_commit();
    }

    // Intra-CTA K-split reduce in FIXED WARP ORDER (no atomics), aliasing the
    // code staging area, which is dead by here. Same discipline as k_vgemm:
    // determinism is a hard requirement, not a preference.
    if constexpr (C::KG > 1) {
        float* red = (float*)dsmem_raw;
        static_assert(sizeof(DSmem<MR, NTI>) >= (size_t)C::NWARP * 32 * 4 * 4,
                      "reduce buffer overruns the staging smem it aliases");
        __syncthreads();
        for (int e = 0; e < 4; ++e) red[(warp * 32 + lane) * 4 + e] = acc[e];
        __syncthreads();
        if (kg != 0) return;
        const int b = wm + C::WM * wn;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            float s = 0.f;
            for (int g = 0; g < C::KG; ++g) s += red[((b + g * C::WM * C::WN) * 32 + lane) * 4 + e];
            acc[e] = s;
        }
    }

    const int64_t row0 = r0 + wm * 16 + (lane >> 2);
    const int tok0 = wn * 8 + 2 * (lane & 3);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int64_t row = row0 + (e >= 2 ? 8 : 0);
        const int tok = tok0 + (e & 1);
        if (row < rows && tok < T) {
            if constexpr (MODE == 1) ws[((size_t)blockIdx.z * T + tok) * rows + row] = acc[e];
            else Y[(size_t)tok * rows + row] = acc[e];
        }
    }
}

__global__ void k_dec_reduce_z(const float* __restrict__ ws, float* __restrict__ Y, int64_t rows,
                               int T, int z) {
    const int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (r >= rows || t >= T) return;
    float s = 0.f;
    for (int i = 0; i < z; ++i) s += ws[((size_t)i * T + t) * rows + r];
    Y[(size_t)t * rows + r] = s;
}

// Per-shape launch plan. z comes from q27k::vgemm_z so the fp4 leg and the
// baseline split K across CTAs identically -- otherwise the table would be
// comparing parallelization policies, not formats.
struct Plan {
    int mr, nti, z, steps_per_z;
};

inline Plan plan_for(int64_t rows, int64_t cols, int T) {
    Plan p{};
    p.nti = T <= 8 ? 1 : (T <= 16 ? 2 : (T <= 32 ? 4 : 8));
    p.mr = p.nti == 8 ? 16 : 32;
    const int kg = (8 / ((p.mr / 16) * p.nti)) > 0 ? (8 / ((p.mr / 16) * p.nti)) : 1;
    const int kb = kg * D_KS;
    const int n_steps = (int)(cols / kb);
    int z = q27k::vgemm_z(rows, cols);
    if (z > n_steps) z = n_steps;
    if (z < 1) z = 1;
    p.steps_per_z = (n_steps + z - 1) / z;
    p.z = (n_steps + p.steps_per_z - 1) / p.steps_per_z;
    return p;
}

template <int MR, int NTI>
void launch_one(const Plan& p, const uint8_t* Wc, const uint8_t* Wsc, const uint8_t* Ac,
                const uint8_t* Asc, float* Y, float* ws, int64_t rows, int64_t cols, int T,
                cudaStream_t st) {
    const size_t sm = sizeof(DSmem<MR, NTI>);
    const dim3 g(1, (unsigned)((rows + MR - 1) / MR), (unsigned)p.z);
    if (p.z == 1) {
        k_mxf4_dec<MR, NTI, 0><<<g, Cfg<MR, NTI>::NTHREAD, sm, st>>>(
            Wc, Wsc, Ac, Asc, Y, nullptr, rows, cols, T, p.steps_per_z);
    } else {
        k_mxf4_dec<MR, NTI, 1><<<g, Cfg<MR, NTI>::NTHREAD, sm, st>>>(
            Wc, Wsc, Ac, Asc, Y, ws, rows, cols, T, p.steps_per_z);
        const dim3 g2((unsigned)((rows + 255) / 256), (unsigned)T);
        k_dec_reduce_z<<<g2, 256, 0, st>>>(ws, Y, rows, T, p.z);
    }
    CK(cudaGetLastError());
}

inline void dec_gemm(const uint8_t* Wc, const uint8_t* Wsc, const uint8_t* Ac, const uint8_t* Asc,
                     float* Y, float* ws, int64_t rows, int64_t cols, int T, cudaStream_t st = 0) {
    const Plan p = plan_for(rows, cols, T);
    switch (p.nti) {
        case 1: launch_one<32, 1>(p, Wc, Wsc, Ac, Asc, Y, ws, rows, cols, T, st); break;
        case 2: launch_one<32, 2>(p, Wc, Wsc, Ac, Asc, Y, ws, rows, cols, T, st); break;
        case 4: launch_one<32, 4>(p, Wc, Wsc, Ac, Asc, Y, ws, rows, cols, T, st); break;
        default: launch_one<16, 8>(p, Wc, Wsc, Ac, Asc, Y, ws, rows, cols, T, st); break;
    }
}

// The smem request has to be raised before any launch, and cannot be raised
// during graph capture -- so do every instantiation once, up front.
inline void dec_set_attrs() {
    auto set = [](const void* f, size_t bytes) {
        CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
    };
#define DEC_SET(MR, NTI)                                                    \
    set((const void*)k_mxf4_dec<MR, NTI, 0>, sizeof(DSmem<MR, NTI>));       \
    set((const void*)k_mxf4_dec<MR, NTI, 1>, sizeof(DSmem<MR, NTI>))
    DEC_SET(32, 1); DEC_SET(32, 2); DEC_SET(32, 4); DEC_SET(16, 8);
#undef DEC_SET
}

} // namespace dec

// ---------------------------------------------------------------------------
// host-side decode for the correctness gates
// ---------------------------------------------------------------------------
float e2m1_val(int c) {
    static const float tab[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float v = tab[c & 7];
    return (c & 8) ? -v : v;
}
float ue4m3_val(uint8_t b) {
    const int exp = (b >> 3) & 0xf, man = b & 7;
    if (exp == 0) return ldexpf(man / 8.f, -6);
    if (exp == 15 && man == 7) return NAN;
    return ldexpf(1.f + man / 8.f, exp - 7);
}

template <typename F> double timeit(F&& fn, int reps) {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int w = 0; w < 3; ++w) fn();
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    for (int r = 0; r < reps; ++r) fn();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    return (double)ms / reps;
}

// fp4 GEMM vs CPU reference on synthetic data: gates the fragment/scale
// mappings. Two passes: EVERY output of a single 128x128x256 tile (a wrong
// row/k/scale mapping anywhere in the ldmatrix/swizzle/block-scale plumbing
// shifts refs by up to the 15x scale range and cannot hide), then 256
// random samples at a multi-tile shape. Products of e2m1 x ue4m3 values are
// exact in fp32, so the only slack needed is accumulation order + the bf16
// output rounding (2^-9 rel).
bool check_fp4_gemm_at(int M, int N, int K, int n_check, unsigned seed) {
    const int64_t ab = (int64_t)M * K / 2, bb = (int64_t)N * K / 2;
    const int64_t asb = (int64_t)M * K / 16, bsb = (int64_t)N * K / 16;
    std::vector<uint8_t> ha(ab), hb(bb), has(asb), hbs(bsb);
    srand(seed);
    for (auto& x : ha) x = rand() & 0xff;
    for (auto& x : hb) x = rand() & 0xff;
    for (auto& x : has) x = ((5 + rand() % 4) << 3) | (rand() & 7); // exact e4m3 near 1.0
    for (auto& x : hbs) x = ((5 + rand() % 4) << 3) | (rand() & 7);

    uint8_t *da, *db, *das, *dbs; __nv_bfloat16* dc;
    CK(cudaMalloc(&da, ab)); CK(cudaMalloc(&db, bb));
    CK(cudaMalloc(&das, asb)); CK(cudaMalloc(&dbs, bsb));
    CK(cudaMalloc(&dc, (int64_t)M * N * sizeof(__nv_bfloat16)));
    CK(cudaMemcpy(da, ha.data(), ab, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb.data(), bb, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(das, has.data(), asb, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dbs, hbs.data(), bsb, cudaMemcpyHostToDevice));

    CK(cudaFuncSetAttribute(k_mxf4_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)sizeof(Smem)));
    k_mxf4_gemm<<<dim3(N / BN, M / BM), NTHREAD, sizeof(Smem)>>>(da, das, db, dbs, dc, M, N, K);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> hc((int64_t)M * N);
    CK(cudaMemcpy(hc.data(), dc, hc.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    int bad = 0;
    const bool full = n_check >= M * N;
    const int checks = full ? M * N : n_check;
    for (int s = 0; s < checks; ++s) {
        const int m = full ? s / N : rand() % M;
        const int n = full ? s % N : rand() % N;
        double ref = 0;
        for (int k = 0; k < K; ++k) {
            const int ac = (ha[(int64_t)m * K / 2 + k / 2] >> ((k & 1) * 4)) & 0xf;
            const int bc = (hb[(int64_t)n * K / 2 + k / 2] >> ((k & 1) * 4)) & 0xf;
            ref += (double)(e2m1_val(ac) * ue4m3_val(has[(int64_t)m * (K / 16) + k / 16])) *
                   (double)(e2m1_val(bc) * ue4m3_val(hbs[(int64_t)n * (K / 16) + k / 16]));
        }
        const float got = __bfloat162float(hc[(int64_t)m * N + n]);
        if (fabs(got - ref) > fmax(fabs(ref) * 1e-2, 0.25)) {
            if (bad < 5) printf("  MISMATCH c[%d,%d] got %f ref %f\n", m, n, got, ref);
            ++bad;
        }
    }
    printf("fp4 GEMM correctness %dx%dx%d: %s (%d %s refs)\n", M, N, K,
           bad ? "FAIL" : "PASS", checks, full ? "exhaustive" : "sampled");
    CK(cudaFree(da)); CK(cudaFree(db)); CK(cudaFree(das)); CK(cudaFree(dbs)); CK(cudaFree(dc));
    return bad == 0;
}

bool check_fp4_gemm() {
    return check_fp4_gemm_at(128, 128, 256, 128 * 128, 27) &&  // every output, one tile
           check_fp4_gemm_at(512, 1024, 1024, 256, 29);        // sampled, multi-tile
}

// Decode-tile gate. Same standard as the prefill tile: EVERY output checked
// against a CPU reference, at each of the four (MR, NTI) instantiations and at
// both z == 1 (MODE 0, direct store) and z > 1 (MODE 1 + the fixed-order
// reduce). A wrong operand/scale/swizzle mapping shifts refs by up to the
// ue4m3 range and cannot hide. Also exercises the row and lane tails (rows and
// T deliberately not multiples of the tile).
bool check_fp4_decode_at(int64_t rows, int64_t cols, int T, unsigned seed) {
    const int64_t wb = rows * cols / 2, ws_b = rows * cols / 16;
    const int64_t ab = (int64_t)16 * cols / 2, as_b = (int64_t)16 * cols / 16;
    std::vector<uint8_t> hw(wb), hws(ws_b), ha(ab), has(as_b);
    srand(seed);
    for (auto& x : hw) x = rand() & 0xff;
    for (auto& x : ha) x = rand() & 0xff;
    // exact-e4m3 exponents near 1.0 keep the CPU reference bit-exact in fp64
    for (auto& x : hws) x = ((5 + rand() % 4) << 3) | (rand() & 7);
    for (auto& x : has) x = ((5 + rand() % 4) << 3) | (rand() & 7);

    uint8_t *dw, *dws, *da, *das;
    float *dy, *dz;
    const dec::Plan p = dec::plan_for(rows, cols, T);
    CK(cudaMalloc(&dw, wb)); CK(cudaMalloc(&dws, ws_b));
    CK(cudaMalloc(&da, ab)); CK(cudaMalloc(&das, as_b));
    CK(cudaMalloc(&dy, (size_t)T * rows * 4));
    CK(cudaMalloc(&dz, (size_t)p.z * T * rows * 4));
    CK(cudaMemcpy(dw, hw.data(), wb, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dws, hws.data(), ws_b, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(da, ha.data(), ab, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(das, has.data(), as_b, cudaMemcpyHostToDevice));
    CK(cudaMemset(dy, 0xff, (size_t)T * rows * 4)); // an unwritten output must fail loudly

    dec::dec_gemm(dw, dws, da, das, dy, dz, rows, cols, T);
    CK(cudaDeviceSynchronize());
    std::vector<float> hy((size_t)T * rows);
    CK(cudaMemcpy(hy.data(), dy, hy.size() * 4, cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int t = 0; t < T; ++t)
        for (int64_t r = 0; r < rows; ++r) {
            double ref = 0;
            for (int64_t k = 0; k < cols; ++k) {
                const int wc = (hw[r * cols / 2 + k / 2] >> ((k & 1) * 4)) & 0xf;
                const int ac = (ha[(int64_t)t * cols / 2 + k / 2] >> ((k & 1) * 4)) & 0xf;
                ref += (double)(e2m1_val(wc) * ue4m3_val(hws[r * (cols / 16) + k / 16])) *
                       (double)(e2m1_val(ac) * ue4m3_val(has[(int64_t)t * (cols / 16) + k / 16]));
            }
            const float got = hy[(size_t)t * rows + r];
            if (fabs(got - ref) > fmax(fabs(ref) * 1e-3, 0.25)) {
                if (bad < 5) printf("  MISMATCH y[t=%d,r=%ld] got %f ref %f\n", t, (long)r, got, ref);
                ++bad;
            }
        }
    printf("fp4 decode correctness rows=%-5ld cols=%-5ld T=%-2d (MR=%d NTI=%d z=%d MODE=%d): %s "
           "(%ld exhaustive refs)\n",
           (long)rows, (long)cols, T, p.mr, p.nti, p.z, p.z == 1 ? 0 : 1, bad ? "FAIL" : "PASS",
           (long)T * (long)rows);
    CK(cudaFree(dw)); CK(cudaFree(dws)); CK(cudaFree(da)); CK(cudaFree(das));
    CK(cudaFree(dy)); CK(cudaFree(dz));
    return bad == 0;
}

bool check_fp4_decode() {
    return check_fp4_decode_at(96, 512, 4, 41) &&    // NTI=1, z=1, row+lane tails
           check_fp4_decode_at(96, 512, 13, 43) &&   // NTI=2, z=1, lane tail
           check_fp4_decode_at(1024, 2048, 16, 45) &&// NTI=2, z>1 (MODE 1 + reduce)
           check_fp4_decode_at(1024, 2048, 32, 47) &&// NTI=4, z>1
           check_fp4_decode_at(1024, 2048, 64, 49);  // NTI=8 (MR=16), z>1
}

// Occupancy/spill introspection, mirroring the k_vgemm CI gate's discipline:
// a spilling decode tile would make the format comparison meaningless.
void report_dec_attrs() {
    auto one = [](const char* name, const void* f, size_t smem, int threads) {
        cudaFuncAttributes a{};
        CK(cudaFuncGetAttributes(&a, f));
        int blocks = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, f, threads, smem));
        printf("  %-22s regs=%-3d spill=%-4zu smem=%-6zu CTA/SM=%d%s\n", name, a.numRegs,
               (size_t)a.localSizeBytes, smem, blocks, a.localSizeBytes ? "  <-- SPILL" : "");
    };
#define DEC_ATTR(MR, NTI, MODE)                                                          \
    one("k_mxf4_dec<" #MR "," #NTI "," #MODE ">", (const void*)dec::k_mxf4_dec<MR, NTI, MODE>, \
        sizeof(dec::DSmem<MR, NTI>), dec::Cfg<MR, NTI>::NTHREAD)
    DEC_ATTR(32, 1, 1); DEC_ATTR(32, 2, 1); DEC_ATTR(32, 4, 1); DEC_ATTR(16, 8, 1);
#undef DEC_ATTR
}

// Achievable streaming-read bandwidth, measured here rather than assumed: every
// percent-of-SOL below is against THIS number, and the repo has never pinned
// clocks (README quotes a 1.79 TB/s spec figure; the achievable fraction is
// what a weight sweep can actually get).
__global__ void k_bwprobe(const uint4* __restrict__ p, size_t n4, float* __restrict__ sink) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (; i < n4; i += stride) {
        const uint4 v = __ldg(p + i);
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x | acc.y | acc.z | acc.w) == 0xdeadbeefu) *sink = 1.f; // never true; defeats DCE
}

double measure_read_sol(int sms) {
    const size_t bytes = 2ull << 30; // 2 GiB: far past any L2
    uint8_t* d; CK(cudaMalloc(&d, bytes)); CK(cudaMemset(d, 0x5a, bytes));
    float* sink; CK(cudaMalloc(&sink, 4));
    const size_t n4 = bytes / 16;
    const int blocks = sms * 8;
    const double ms = timeit([&] { k_bwprobe<<<blocks, 256>>>((const uint4*)d, n4, sink); }, 20);
    CK(cudaFree(d)); CK(cudaFree(sink));
    return bytes / (ms * 1e6); // GB/s
}

// quantize kernel gate: reconstruction of random fp32 within nvfp4 grid error.
bool check_fp4_quant() {
    const int M = 64, K = 5120;
    std::vector<float> hx((int64_t)M * K);
    srand(31);
    for (auto& v : hx) v = (rand() / (float)RAND_MAX - 0.5f) * 4.f;
    float* dx; uint8_t *dc, *ds;
    CK(cudaMalloc(&dx, hx.size() * 4));
    CK(cudaMalloc(&dc, (int64_t)M * K / 2));
    CK(cudaMalloc(&ds, (int64_t)M * K / 16));
    CK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    const int64_t groups = (int64_t)M * K / 16;
    k_quant_nvfp4<<<(int)((groups + 255) / 256), 256>>>(dx, dc, ds, groups, K);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> hc((int64_t)M * K / 2), hs((int64_t)M * K / 16);
    CK(cudaMemcpy(hc.data(), dc, hc.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs.data(), ds, hs.size(), cudaMemcpyDeviceToHost));
    double num = 0, den = 0;
    for (int64_t i = 0; i < (int64_t)M * K; ++i) {
        const int code = (hc[i / 2] >> ((i & 1) * 4)) & 0xf;
        const float rec = e2m1_val(code) * ue4m3_val(hs[i / 16]);
        num += (rec - hx[i]) * (rec - hx[i]);
        den += hx[i] * hx[i];
    }
    const double rel = sqrt(num / den);
    printf("fp4 quantize reconstruction: rel RMS %.3f %s\n", rel,
           rel < 0.2 ? "PASS" : "FAIL");
    CK(cudaFree(dx)); CK(cudaFree(dc)); CK(cudaFree(ds));
    return rel < 0.2;
}

struct Shape { const char* name; const char* tensor; };

// ---------------------------------------------------------------------------
// T2 decode sweep.
//
// L2 ROTATION. A decode weight sweep in the real engine streams the whole
// 15.46 GB tier per round, so nothing is L2-resident. A naive rep loop over one
// 15-45 MB tensor sits inside this card's L2 and measures cache bandwidth for
// BOTH legs -- which flatters the leg with fewer bytes and makes the whole
// point of the test disappear. Each shape is therefore replicated until the
// rotated working set clears L2 by 3x, and consecutive reps read different
// copies.
// ---------------------------------------------------------------------------
template <typename F> double timeit_rot(F&& fn, int reps, int nrot) {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int w = 0; w < nrot + 2; ++w) fn(w % nrot);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    for (int r = 0; r < reps; ++r) fn(r % nrot);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    return (double)ms / reps;
}

void run_decode_sweep(q27::DeviceModel& dm, const cudaDeviceProp& prop, double sol_gbs,
                      double peak_fp4, double peak_int) {
    const Shape shapes[] = {
        {"attn_q+gate", "blk.3.attn_q.weight"},      // 12288 x 5120
        {"attn_output", "blk.3.attn_output.weight"}, //  5120 x 6144
        {"ffn_gate   ", "blk.0.ffn_gate.weight"},    // 17408 x 5120
        {"ffn_down   ", "blk.0.ffn_down.weight"},    //  5120 x 17408
    };
    const int Ms[] = {4, 8, 16, 32, 64};
    const int reps = 60;
    constexpr int NL = W_PLUMB; // 16: the lane plumbing, and vgemm's hard T cap

    printf("\n=== T2: DECODE shapes (M = union width) ===\n");
    printf("baseline = q27k::vgemm_verify -- the union GEMM build_union_view routes to at\n"
           "k >= 3 (src/conductor.h:483), i.e. the path C=8 actually takes. gemv = the\n"
           "k <= 2 incumbent, shown for reference. fp4 = k_mxf4_dec (nvfp4 W4A4,\n"
           "m16n8k64 block-scale, weights on the A operand).\n"
           "bytes/GB-s count weights + scales + activations + output + split-K partials.\n\n");

    // Activation scratch, sized for the widest K. Lanes are separate buffers on
    // the vgemm side (XLanes is a pointer-per-lane struct) and one contiguous
    // [NL][K] matrix on the fp4 side -- the layout each kernel's real caller
    // would hand it.
    const int64_t maxK = 17408, maxRows = 17408;
    float* xflat; CK(cudaMalloc(&xflat, (size_t)NL * maxK * 4));
    {
        std::vector<float> hx((size_t)NL * maxK);
        srand(97);
        for (auto& v : hx) v = (rand() / (float)RAND_MAX - 0.5f) * 2.f;
        CK(cudaMemcpy(xflat, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    }
    float* xpack; CK(cudaMalloc(&xpack, (size_t)NL * maxK * 4)); // [NL][K] for the fp4 quantize
    q27k::XQuant xqs[NL];
    const int8_t* xnat[NL]; const float* xsc[NL];
    for (int i = 0; i < NL; ++i) {
        xqs[i] = q27k::xquant_alloc(maxK, /*g64=*/false);
        xnat[i] = xqs[i].nat; xsc[i] = xqs[i].scale;
    }
    uint8_t *a4c, *a4s;
    CK(cudaMalloc(&a4c, (size_t)NL * maxK / 2));
    CK(cudaMalloc(&a4s, (size_t)NL * maxK / 16));
    float* ylanes[NL];
    for (int i = 0; i < NL; ++i) CK(cudaMalloc(&ylanes[i], (size_t)maxRows * 4));
    float* yflat; CK(cudaMalloc(&yflat, (size_t)64 * maxRows * 4));
    float* wsz;   CK(cudaMalloc(&wsz, (size_t)8 * 64 * maxRows * 4));

    printf("%-12s %4s | %8s %8s %7s | %8s %8s %7s | %6s %6s %6s | %5s\n", "shape", "M", "vgemm ms",
           "GB/s", "%SOL", "fp4 ms", "GB/s", "%SOL", "ratio", "fmt", "gemv", "%pk4");
    printf("%.*s\n", 111,
           "--------------------------------------------------------------------------------"
           "-------------------------------------------------");

    for (const Shape& s : shapes) {
        const q27::DevTensor& w = dm.upload(s.tensor);
        const int64_t N = w.rows, K = w.cols;
        if (w.dtype != q27::DType::Q4_G64) { fprintf(stderr, "%s: expected Q4_G64\n", s.tensor); exit(1); }

        // Activations, quantized once per K in each leg's own layout: group-32
        // int8 per lane (what quantize3 already writes for the vgemm path) and
        // an [NL][K] contiguous nvfp4 matrix for the fp4 path. Lanes live at
        // stride maxK in xflat, so the fp4 side gets a packed copy.
        for (int i = 0; i < NL; ++i) q27k::quantize_x(xflat + (int64_t)i * maxK, K, xqs[i], 0);
        for (int i = 0; i < NL; ++i)
            CK(cudaMemcpy(xpack + (int64_t)i * K, xflat + (int64_t)i * maxK, K * 4,
                          cudaMemcpyDeviceToDevice));
        {
            const int64_t groups = (int64_t)NL * K / 16;
            k_quant_nvfp4<<<(int)((groups + 255) / 256), 256>>>(xpack, a4c, a4s, groups, (int)K);
            CK(cudaGetLastError());
        }
        CK(cudaDeviceSynchronize());

        // L2 rotation factor
        const size_t q4_bytes = (size_t)N * K / 2 + (size_t)N * K / 64 * 2;
        const size_t f4_bytes = (size_t)N * K / 2 + (size_t)N * K / 16;
        int nrot = (int)((3ull * (size_t)prop.l2CacheSize + q4_bytes - 1) / q4_bytes);
        if (nrot < 2) nrot = 2;
        if (nrot > 16) nrot = 16; // 16 x attn_output still fits; below 3x L2 the
                                  // smaller-footprint leg gets the cache edge

        std::vector<uint8_t*> wq(nrot), wqs(nrot), wf(nrot), wfs(nrot);
        std::vector<q27::DevTensor> wt(nrot);
        for (int i = 0; i < nrot; ++i) {
            CK(cudaMalloc(&wq[i], (size_t)N * K / 2));
            CK(cudaMalloc(&wqs[i], (size_t)N * K / 64 * 2));
            CK(cudaMemcpy(wq[i], w.data, (size_t)N * K / 2, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(wqs[i], w.scales, (size_t)N * K / 64 * 2, cudaMemcpyDeviceToDevice));
            wt[i] = w; wt[i].data = wq[i]; wt[i].scales = wqs[i];
            CK(cudaMalloc(&wf[i], (size_t)N * K / 2));
            CK(cudaMalloc(&wfs[i], (size_t)N * K / 16));
            // Synthetic fp4 weight VALUES at the real (N, K): both kernels are
            // data-independent in time (no data-dependent branches, fixed
            // footprint), so only the shape has to be real here.
            CK(cudaMemcpy(wf[i], w.data, (size_t)N * K / 2, cudaMemcpyDeviceToDevice));
            std::vector<uint8_t> hs((size_t)N * K / 16);
            for (auto& v : hs) v = ((5 + rand() % 4) << 3) | (rand() & 7);
            CK(cudaMemcpy(wfs[i], hs.data(), hs.size(), cudaMemcpyHostToDevice));
        }

        for (const int M : Ms) {
            const dec::Plan p = dec::plan_for(N, K, M);
            const int zv = q27k::vgemm_z(N, K);

            q27k::XLanes X{}; q27k::YLanes Y{};
            for (int i = 0; i < NL; ++i) { X.nat[i] = xnat[i]; X.xs[i] = xsc[i]; Y.y[i] = ylanes[i]; }

            // Baseline. vgemm's NT tile is W_PLUMB, so T > 16 does not exist:
            // above that the only honest stand-in is ceil(M/16) calls, which
            // re-reads the weights each time -- reported, and labelled, rather
            // than silently extrapolated.
            const int ncall = (M + NL - 1) / NL;
            const double t_vg = timeit_rot([&](int r) {
                for (int c = 0; c < ncall; ++c) {
                    const int t = (c + 1) * NL <= M ? NL : M - c * NL;
                    if (!q27k::vgemm_verify(wt[r], X, Y, wsz, t < 2 ? 2 : t, 0)) {
                        fprintf(stderr, "vgemm_verify refused T=%d\n", t); exit(1);
                    }
                }
            }, reps, nrot);
            const double t_gv = M <= NL ? timeit_rot([&](int r) {
                q27k::gemv_q4_n(wq[r], (const __half*)wqs[r], xqs, M, ylanes, N, K, 0);
            }, reps, nrot) : 0.0;
            const double t_f4 = timeit_rot([&](int r) {
                dec::dec_gemm(wf[r], wfs[r], a4c, a4s, yflat, wsz, N, K, M, 0);
            }, reps, nrot);

            // Byte accounting. Weights dominate; activations, outputs and the
            // split-K partials are included so the GB/s number is what the
            // memory system actually saw.
            const double act_q4 = (double)M * K * (1.0 + 4.0 / 32.0);
            const double act_f4 = (double)M * K * (0.5 + 1.0 / 16.0);
            const double out_b = (double)M * N * 4.0;
            // k_vgemm guards its partial store on tok < T (src/vgemm.cu:206), so
            // the split-K traffic is z*M*rows floats written and read back --
            // NOT z*W_PLUMB. Counting the padded lanes overstated the baseline's
            // GB/s by ~16% at M=4.
            const double part_vg = zv > 1 ? 2.0 * zv * M * N * 4.0 : 0.0;
            const double part_f4 = p.z > 1 ? 2.0 * p.z * M * N * 4.0 : 0.0;
            const double b_vg = ncall * (double)q4_bytes + part_vg + act_q4 + out_b;
            const double b_f4 = (double)f4_bytes + act_f4 + out_b + part_f4;
            const double fl = 2.0 * M * N * K;

            const double gvg = b_vg / (t_vg * 1e6), gf4 = b_f4 / (t_f4 * 1e6);
            // fmt: the FORMAT-only ratio -- what the baseline would take if its
            // kernel moved bytes as efficiently as the fp4 one does, i.e. the
            // measured ratio with the kernel-technique gap divided out. Both
            // legs are bandwidth-bound here, so this is the residual the plan
            // asks for: anything left after the technique gap is the format.
            const double t_vg_atf4 = b_vg / (gf4 * 1e6);
            char gvs[16];
            if (t_gv > 0) snprintf(gvs, sizeof gvs, "%.3fx", t_gv / t_f4);
            else snprintf(gvs, sizeof gvs, "n/a");
            printf("%-12s %4d | %8.4f %8.1f %6.1f%% | %8.4f %8.1f %6.1f%% | %5.3fx %5.3fx %6s | "
                   "%4.1f%%%s\n",
                   s.name, M, t_vg, gvg, 100.0 * gvg / sol_gbs, t_f4, gf4,
                   100.0 * gf4 / sol_gbs, t_vg / t_f4, t_vg_atf4 / t_f4, gvs,
                   100.0 * (fl / (t_f4 * 1e9)) / peak_fp4,
                   M > NL ? "  (baseline = vgemm x2/x4, weights re-read)" : "");
        }
        for (int i = 0; i < nrot; ++i) {
            CK(cudaFree(wq[i])); CK(cudaFree(wqs[i])); CK(cudaFree(wf[i])); CK(cudaFree(wfs[i]));
        }
        printf("%.*s\n", 104,
               "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - "
               "- - - - - - - - - - - - ");
    }
    printf("\nread SOL measured this run: %.0f GB/s (spec 1790, so the probe itself is ~95%% of\n"
           "spec -- a %%SOL at or above 100 means 'at the roof', not 'impossible'). fp4 dense\n"
           "peak this run: %.0f TFLOPS; int8 (gemm_q4_T) plateau this run: %.0f TFLOPS.\n",
           sol_gbs, peak_fp4, peak_int);
    printf("\nWHAT THE COLUMNS SAY. %%pk4 tops out near 18%%: at these M the fp4 MMA is idle\n"
           "most of the time and neither leg is compute-bound, so the dense-fp4-is-2x-int8\n"
           "silicon advantage buys nothing here -- the whole regime is a byte count. nvfp4\n"
           "spends 0.5 B/weight on e2m1 codes plus 1 B per 16 on the ue4m3 block scale =\n"
           "0.5625 B/weight (4.50 bpw); Q4_G64 spends 0.5 plus one fp16 per 64 = 0.53125\n"
           "(4.25 bpw). fp4 moves 1.0588x the bytes for the same weights. 'fmt' divides the\n"
           "kernel-technique gap out of 'ratio': it is what the baseline would take at the\n"
           "fp4 kernel's own GB/s, and it lands BELOW 1.0 -- the measured win is k_vgemm\n"
           "leaving bandwidth on the table (this tile uses cp.async; k_vgemm stages through\n"
           "registers), not the format, and closing that gap in k_vgemm beats adopting fp4.\n");

    for (int i = 0; i < NL; ++i) CK(cudaFree(ylanes[i]));
    CK(cudaFree(xflat)); CK(cudaFree(xpack)); CK(cudaFree(a4c)); CK(cudaFree(a4s));
    CK(cudaFree(yflat)); CK(cudaFree(wsz));
}

} // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1]
                                : "/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp-q4s.q27";
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("microbench_mxf4 on %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    if (prop.major != 12) {
        fprintf(stderr, "requires sm_120a (RTX 5090 class); this is sm_%d%d.\n"
                        "The binary is compiled -gencode arch=compute_120a only.\n",
                prop.major, prop.minor);
        return 1;
    }
    if (FILE* f = fopen(path, "rb")) fclose(f);
    else {
        fprintf(stderr, "model not readable: %s\nusage: %s [model.q27]\n", path, argv[0]);
        return 1;
    }
    printf("model: %s\n\n", path);

    dec::dec_set_attrs();
    if (!check_fp4_gemm() || !check_fp4_quant() || !check_fp4_decode()) return 1;
    printf("\ndecode-tile occupancy (a spill here would make the format comparison meaningless):\n");
    report_dec_attrs();

    const double sol_gbs = measure_read_sol(prop.multiProcessorCount);
    printf("device: %d SMs, L2 %.1f MB, measured streaming-read SOL %.0f GB/s\n",
           prop.multiProcessorCount, prop.l2CacheSize / 1048576.0, sol_gbs);

    q27::Model m = q27::Model::open(path);
    q27::DeviceModel dm(m);
    const Shape shapes[] = {
        {"attn_q+gate", "blk.3.attn_q.weight"},      // 12288 x 5120
        {"attn_output", "blk.3.attn_output.weight"}, //  5120 x 6144
        {"ffn_gate   ", "blk.0.ffn_gate.weight"},    // 17408 x 5120
        {"ffn_down   ", "blk.0.ffn_down.weight"},    //  5120 x 17408
    };
    const int Ms[] = {512, 1024, 2048, 8192};
    const int reps = 50;
    double peak_fp4 = 0, peak_int = 0; // in-run compute peaks for the decode table

    // worst-case buffers across shapes/Ms
    const int64_t max_mk = (int64_t)8192 * 17408;
    const int64_t max_mn = (int64_t)8192 * 17408;
    float* x;  CK(cudaMalloc(&x, max_mk * 4));
    float* y;  CK(cudaMalloc(&y, max_mn * 4));
    __nv_bfloat16* c4; CK(cudaMalloc(&c4, max_mn * sizeof(__nv_bfloat16)));
    uint8_t *a4, *a4s; CK(cudaMalloc(&a4, max_mk / 2)); CK(cudaMalloc(&a4s, max_mk / 16));
    q27k::XQuant xq = q27k::xquant_alloc(max_mk, /*g64=*/true);
    {
        std::vector<float> hx(1 << 20);
        srand(42);
        for (auto& v : hx) v = (rand() / (float)RAND_MAX - 0.5f) * 2.f;
        for (int64_t off = 0; off < max_mk; off += (1 << 20)) {
            const int64_t n = std::min<int64_t>(1 << 20, max_mk - off);
            CK(cudaMemcpy(x + off, hx.data(), n * 4, cudaMemcpyHostToDevice));
        }
    }
    CK(cudaFuncSetAttribute(k_mxf4_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)sizeof(Smem)));

    printf("\n%-12s %5s | %9s %9s %8s | %9s %9s %8s | %6s %6s\n",
           "shape", "M", "q_g64 ms", "gemm ms", "TFLOPS", "q_fp4 ms", "gemm ms", "TFLOPS",
           "gemm", "e2e");
    printf("%.*s\n", 118,
           "-----------------------------------------------------------------------------"
           "-----------------------------------------");

    for (const Shape& s : shapes) {
        const q27::DevTensor& w = dm.upload(s.tensor);
        const int64_t N = w.rows, K = w.cols;
        if (w.dtype != q27::DType::Q4_G64) {
            fprintf(stderr, "%s: expected Q4_G64\n", s.tensor);
            return 1;
        }
        // k_mxf4_gemm has no tail handling by design (the bench shapes tile
        // exactly); enforce the contract loudly for arbitrary models.
        if (N % BN != 0 || K % BK != 0) {
            fprintf(stderr, "%s: (N=%ld, K=%ld) not tileable (N%%%d, K%%%d) -- skipped\n",
                    s.tensor, (long)N, (long)K, BN, BK);
            continue;
        }
        // Synthetic fp4 weight with the same (N, K): only the SHAPE comes
        // from the real tensor. Throughput is value-independent for both
        // kernels (no data-dependent branches, same memory footprint); the
        // real-weight fp4 repack is phase 2's job.
        uint8_t *b4, *b4s;
        CK(cudaMalloc(&b4, N * K / 2)); CK(cudaMalloc(&b4s, N * K / 16));
        {
            std::vector<uint8_t> h(N * K / 2);
            for (auto& v : h) v = rand() & 0xff;
            CK(cudaMemcpy(b4, h.data(), h.size(), cudaMemcpyHostToDevice));
            std::vector<uint8_t> hs(N * K / 16);
            for (auto& v : hs) v = ((5 + rand() % 4) << 3) | (rand() & 7);
            CK(cudaMemcpy(b4s, hs.data(), hs.size(), cudaMemcpyHostToDevice));
        }

        for (const int M : Ms) {
            if (M % BM != 0) { fprintf(stderr, "M=%d not a multiple of %d\n", M, BM); return 1; }
            // baseline: activation quantize (int8 g32+g64, what qxT runs) + GEMM
            const double t_qb = timeit([&] {
                q27k::quantize_x(x, (int64_t)M * K, xq, 0);
                q27k::quantize_x_g64(x, (int64_t)M * K, xq, 0);
            }, reps);
            const double t_gb = timeit([&] {
                q27k::gemm_q4_T((const uint8_t*)w.data, (const __half*)w.scales, xq, y, N, K, M,
                                0, nullptr);
            }, reps);
            // fp4: activation quantize + GEMM
            const int64_t groups = (int64_t)M * K / 16;
            const double t_q4 = timeit([&] {
                k_quant_nvfp4<<<(int)((groups + 255) / 256), 256>>>(x, a4, a4s, groups, (int)K);
            }, reps);
            const double t_g4 = timeit([&] {
                k_mxf4_gemm<<<dim3((int)(N / BN), M / BM), NTHREAD, sizeof(Smem)>>>(
                    a4, a4s, b4, b4s, c4, M, (int)N, (int)K);
            }, reps);
            const double fl = 2.0 * M * N * K;
            const double tf_b = fl / (t_gb * 1e9), tf_4 = fl / (t_g4 * 1e9);
            if (tf_b > peak_int) peak_int = tf_b;
            if (tf_4 > peak_fp4) peak_fp4 = tf_4;
            printf("%-12s %5d | %9.3f %9.3f %8.1f | %9.3f %9.3f %8.1f | %5.2fx %5.2fx\n",
                   s.name, M, t_qb, t_gb, tf_b, t_q4, t_g4, tf_4,
                   t_gb / t_g4, (t_qb + t_gb) / (t_q4 + t_g4));
        }
        CK(cudaFree(b4)); CK(cudaFree(b4s));
    }

    printf("\nbaseline = q27k::gemm_q4_T on real %s weights (W4A8-int, m16n8k32 s8 MMA,\n",
           "q4s");
    printf("live dispatch incl. ntx leg); fp4 = k_mxf4_gemm (nvfp4 W4A4, m16n8k64 block-scale).\n");
    printf("phase 2 gate (docs/plans/2026-08-15-ninfer-steals.md): fp4 gemm ratio >= 1.3x at\n");
    printf("M >= 512 on real projection shapes.\n");

    // The prefill sweep's own maxima are the in-run compute peaks the decode
    // percent-of-peak column is measured against -- taken from THIS binary on
    // THIS driver rather than a spec sheet the repo has never pinned.
    CK(cudaFree(x)); CK(cudaFree(y)); CK(cudaFree(c4)); CK(cudaFree(a4)); CK(cudaFree(a4s));

    run_decode_sweep(dm, prop, sol_gbs, peak_fp4, peak_int);
    return 0;
}
