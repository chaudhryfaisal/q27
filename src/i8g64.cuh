// int8-g64 KV quantization -- ninfer's KV profile, K-side study arm.
//
// WHY THIS EXISTS. Phase 1 of docs/plans/2026-08-15-ninfer-steals.md: ninfer
// serves the same checkpoints with INT8 group-64 per-token-absmax KV and the
// 2026-08-01 tail study proved PPL is blind on the KV axis, so the format
// earns a row in the catastrophic-position table before any serving port.
// This arm quantizes K only; V stays turbo3 like every turbo kind (the
// turbo3v/turbo5k denominator), which makes the row directly comparable on
// the K ladder: fp16 K (turbo3v, 52 catastrophic) vs 5-bit K (turbo5k, 65)
// vs this, 8-bit groupwise K, at 1056 B/token/layer K + 400 V = 25.6 KB/tok.
//
// THE CODEC IS BIT-FAITHFUL TO NINFER (recon 2026-08-15, their
// include/ninfer/ops/gqa_attention.h normative contract): 64 contiguous dims
// per group along head_dim, fp32 absmax, scale = fp16_rne(absmax/127.0f),
// and -- the load-bearing subtlety -- the quantization reciprocal is taken
// from the fp16-ROUNDED scale, not from the raw absmax/127 (getting that
// wrong shifts codes by +-1 near boundaries). Round-to-nearest-even, clamp
// to [-127,127] (-128 never emitted), all-zero group stores scale bits 0 and
// codes 0. Dequant is a bare code * scale, no zero point, no compensation.
// NO WHT rotation on K (kv_k_rotated stays false): ninfer's format has none,
// and the arm exists to measure exactly that error profile.
//
// Unlike turbo3/turbo5's norm-corrected blocks, absmax is order-independent,
// so the CPU oracle in tools/i8g64_test.cu demands EXACT code/scale equality,
// not tolerance-class agreement.
//
// q27 mapping (head_dim=256 = TWO 128-blocks, no padding): one block covers
// 128 dims = two 64-groups. 132 bytes, 2-byte aligned fields only.
#pragma once
#include <cuda_fp16.h>

namespace q27turbo {

static constexpr int QK_I8G64 = 128; // dims per block (two 64-groups)

struct block_i8g64 {
    int8_t qs[QK_I8G64];  // 128 B codes, [-127,127]
    __half s[2];          // fp16 scale per 64-group (absmax/127, rne)
};
static_assert(sizeof(block_i8g64) == QK_I8G64 + 2 * sizeof(__half),
              "i8g64 block must be 132 bytes, no padding");

// Store: 128 threads, one block; j = dim within the block. red[] is the
// caller's shared scratch (>= 128 floats). Each 64-half reduces its absmax
// independently ((j & 63) keeps the halves separate), then every thread
// derives its own group's scale from red[(j>>6)<<6] -- deterministic, no
// cross-group traffic.
__device__ __forceinline__ void i8g64_quant_group(float x, block_i8g64* dst, int j,
                                                  float* red) {
    red[j] = fabsf(x);
    __syncthreads();
#pragma unroll
    for (int s = 32; s > 0; s >>= 1) {
        if ((j & 63) < s) red[j] = fmaxf(red[j], red[j + s]);
        __syncthreads();
    }
    const float amax = red[(j >> 6) << 6];
    const __half sh = __float2half_rn(amax / 127.0f);
    const float sw = __half2float(sh);          // reciprocal of the ROUNDED scale
    const float inv = sw > 0.f ? 1.0f / sw : 0.f;
    const int q = min(127, max(-127, __float2int_rn(x * inv)));
    dst->qs[j] = (int8_t)(sw > 0.f ? q : 0);
    if ((j & 63) == 0) dst->s[j >> 6] = sh;
}

// Element d of a 256-dim head row stored as a 2-block pair (fd2/prefill
// addressing: b2 = base + (p * n_kv_heads + kvh) * 2).
__device__ __forceinline__ float i8g64_deq_elem(const block_i8g64* __restrict__ b2, int d) {
    const block_i8g64* b = b2 + (d >> 7);
    const int j = d & 127;
    return (float)b->qs[j] * __half2float(b->s[j >> 6]);
}

// fd2 lane load: out[0..3] = dims lane*4..+3 (block 0), out[4..7] = the same
// offsets in block 1 -- matches fd2_ld8's float4 q-dot layout.
__device__ __forceinline__ void i8g64_ld8_lane(const block_i8g64* __restrict__ b2, int lane,
                                               float* out) {
    const int d0 = lane * 4;
    const float s0 = __half2float(b2[0].s[d0 >> 6]);
    const float s1 = __half2float(b2[1].s[d0 >> 6]);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        out[i] = (float)b2[0].qs[d0 + i] * s0;
        out[4 + i] = (float)b2[1].qs[d0 + i] * s1;
    }
}

// MMA-prefill staging: 8 dims starting at d8 (multiple of 8, within one
// 64-group by construction) into 4 half2s -- the turbo3_stage8_h2 shape.
__device__ __forceinline__ void i8g64_stage8_h2(const block_i8g64* __restrict__ b2, int d8,
                                                __half2* kd) {
    const block_i8g64* b = b2 + (d8 >> 7);
    const int j = d8 & 127;
    const float s = __half2float(b->s[j >> 6]);
#pragma unroll
    for (int i = 0; i < 4; i++)
        kd[i] = __floats2half2_rn((float)b->qs[j + 2 * i] * s,
                                  (float)b->qs[j + 2 * i + 1] * s);
}

} // namespace q27turbo
