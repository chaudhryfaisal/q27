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

    if (!check_fp4_gemm() || !check_fp4_quant()) return 1;

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
            printf("%-12s %5d | %9.3f %9.3f %8.1f | %9.3f %9.3f %8.1f | %5.2fx %5.2fx\n",
                   s.name, M, t_qb, t_gb, fl / (t_gb * 1e9), t_q4, t_g4, fl / (t_g4 * 1e9),
                   t_gb / t_g4, (t_qb + t_gb) / (t_q4 + t_g4));
        }
        CK(cudaFree(b4)); CK(cudaFree(b4s));
    }

    printf("\nbaseline = q27k::gemm_q4_T on real %s weights (W4A8-int, m16n8k32 s8 MMA,\n",
           "q4s");
    printf("live dispatch incl. ntx leg); fp4 = k_mxf4_gemm (nvfp4 W4A4, m16n8k64 block-scale).\n");
    printf("phase 2 gate (docs/plans/2026-08-15-ninfer-steals.md): fp4 gemm ratio >= 1.3x at\n");
    printf("M >= 512 on real projection shapes.\n");
    return 0;
}
