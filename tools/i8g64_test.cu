// int8-g64 (ninfer KV profile) format validation: prove src/i8g64.cuh's device
// codec and every read path agree with an independent CPU oracle BEFORE the
// tail-study run. Plan: docs/plans/2026-08-15-ninfer-steals.md, phase 1.
//
// Stronger gate than the turbo tests can afford: absmax is order-independent
// (unlike turbo3/5's sum-based norms), and fp16 scale rounding, RNE, and the
// clamp are all deterministic -- so EVERY comparison here is bitwise-exact, no
// tie tolerance anywhere. The oracle implements ninfer's normative contract
// (their include/ninfer/ops/gqa_attention.h, recon 2026-08-15):
//   scale_bits = fp16_rne(absmax/127); inv = 1/fp32(scale_bits);
//   code = clamp(rne(x*inv), -127, 127); zero-scale group -> all zeros.
// The load-bearing subtlety under test: the reciprocal comes from the
// fp16-ROUNDED scale, not from the raw absmax/127.
//
// Build: make build/i8g64_test
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "../src/i8g64.cuh"
using namespace q27turbo;

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)

// ---- CPU oracle -------------------------------------------------------------
static void oracle_quant(const float* x, block_i8g64* dst) { // x[128] -> one block
    for (int g = 0; g < 2; ++g) {
        float amax = 0.f;
        for (int j = 0; j < 64; ++j) amax = fmaxf(amax, fabsf(x[g * 64 + j]));
        const __half sh = __float2half_rn(amax / 127.0f);
        const float sw = __half2float(sh);
        const float inv = sw > 0.f ? 1.0f / sw : 0.f;
        dst->s[g] = sh;
        for (int j = 0; j < 64; ++j) {
            const float v = x[g * 64 + j] * inv;
            // rintf under the default FE_TONEAREST == round-to-nearest-even,
            // the same mode as device __float2int_rn
            int q = (int)rintf(v);
            q = q < -127 ? -127 : q > 127 ? 127 : q;
            dst->qs[g * 64 + j] = (int8_t)(sw > 0.f ? q : 0);
        }
    }
}
static float oracle_deq(const block_i8g64* b2, int d) { // d in [0,256)
    const block_i8g64* b = b2 + (d >> 7);
    return (float)b->qs[d & 127] * __half2float(b->s[(d & 127) >> 6]);
}

// ---- device wrappers ----------------------------------------------------------
__global__ void k_quant(const float* __restrict__ x, block_i8g64* __restrict__ blocks) {
    __shared__ float red[128];
    i8g64_quant_group(x[blockIdx.x * 128 + threadIdx.x], blocks + blockIdx.x, threadIdx.x, red);
}
__global__ void k_deq(const block_i8g64* __restrict__ b, float* __restrict__ out, int rows) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    out[(size_t)r * 256 + threadIdx.x] = i8g64_deq_elem(b + r * 2, threadIdx.x);
}
__global__ void k_ld8(const block_i8g64* __restrict__ b, float* __restrict__ out, int rows) {
    const int r = blockIdx.x, lane = threadIdx.x; // 32 threads
    if (r >= rows) return;
    float kv[8];
    i8g64_ld8_lane(b + r * 2, lane, kv);
    // undo the fd2 lane layout: kv[0..3]=dims lane*4..+3, kv[4..7]=+128
    for (int i = 0; i < 4; i++) {
        out[(size_t)r * 256 + lane * 4 + i] = kv[i];
        out[(size_t)r * 256 + 128 + lane * 4 + i] = kv[4 + i];
    }
}
__global__ void k_stage8(const block_i8g64* __restrict__ b, __half* __restrict__ out, int rows) {
    const int r = blockIdx.x, t = threadIdx.x; // 32 threads: t*8 = d8
    if (r >= rows) return;
    __half2 kd[4];
    i8g64_stage8_h2(b + r * 2, t * 8, kd);
    for (int i = 0; i < 4; i++) {
        out[(size_t)r * 256 + t * 8 + 2 * i] = __low2half(kd[i]);
        out[(size_t)r * 256 + t * 8 + 2 * i + 1] = __high2half(kd[i]);
    }
}

int main() {
    const int ROWS = 4096; // 256-dim head rows = 2 blocks each
    const int BLOCKS = ROWS * 2;
    std::vector<float> hx(BLOCKS * 128);
    srand(64);
    // mix of regimes: normal values, huge, tiny, exact-tie fractions, zeros
    for (size_t i = 0; i < hx.size(); ++i) {
        const int r = rand();
        float v = ((r / (float)RAND_MAX) - 0.5f) * 8.f;
        if ((r % 97) == 0) v *= 1e4f;         // outlier spikes
        if ((r % 89) == 0) v *= 1e-6f;        // denormal-scale territory
        if ((r % 83) == 0) v = (float)(r % 255 - 127) / 2.f; // RNE tie bait
        hx[i] = v;
    }
    for (int j = 0; j < 128; ++j) hx[5 * 128 + j] = 0.f; // one all-zero block

    float* dx; block_i8g64* db;
    CK(cudaMalloc(&dx, hx.size() * 4));
    CK(cudaMalloc(&db, BLOCKS * sizeof(block_i8g64)));
    CK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    k_quant<<<BLOCKS, 128>>>(dx, db);
    CK(cudaGetLastError());

    std::vector<block_i8g64> hb(BLOCKS), ob(BLOCKS);
    CK(cudaMemcpy(hb.data(), db, BLOCKS * sizeof(block_i8g64), cudaMemcpyDeviceToHost));
    for (int b = 0; b < BLOCKS; ++b) oracle_quant(&hx[b * 128], &ob[b]);

    int bad_codes = 0, bad_scales = 0;
    for (int b = 0; b < BLOCKS; ++b) {
        if (memcmp(hb[b].qs, ob[b].qs, 128)) ++bad_codes;
        if (memcmp(hb[b].s, ob[b].s, 4)) ++bad_scales;
    }
    printf("quant: %d/%d blocks code-mismatched, %d scale-mismatched %s\n", bad_codes, BLOCKS,
           bad_scales, (bad_codes || bad_scales) ? "FAIL" : "PASS (bitwise)");

    // read paths: all three must reproduce oracle_deq exactly
    float* dout; __half* hout;
    CK(cudaMalloc(&dout, (size_t)ROWS * 256 * 4));
    CK(cudaMalloc(&hout, (size_t)ROWS * 256 * 2));
    std::vector<float> ro((size_t)ROWS * 256);
    std::vector<__half> rh((size_t)ROWS * 256);
    int bad_deq = 0, bad_ld8 = 0, bad_st8 = 0;

    k_deq<<<ROWS, 256>>>(db, dout, ROWS);
    CK(cudaMemcpy(ro.data(), dout, ro.size() * 4, cudaMemcpyDeviceToHost));
    for (int r = 0; r < ROWS; ++r)
        for (int d = 0; d < 256; ++d)
            if (ro[(size_t)r * 256 + d] != oracle_deq(&hb[r * 2], d)) ++bad_deq;

    k_ld8<<<ROWS, 32>>>(db, dout, ROWS);
    CK(cudaMemcpy(ro.data(), dout, ro.size() * 4, cudaMemcpyDeviceToHost));
    for (int r = 0; r < ROWS; ++r)
        for (int d = 0; d < 256; ++d)
            if (ro[(size_t)r * 256 + d] != oracle_deq(&hb[r * 2], d)) ++bad_ld8;

    k_stage8<<<ROWS, 32>>>(db, hout, ROWS);
    CK(cudaMemcpy(rh.data(), hout, rh.size() * 2, cudaMemcpyDeviceToHost));
    for (int r = 0; r < ROWS; ++r)
        for (int d = 0; d < 256; ++d) {
            const __half want = __float2half_rn(oracle_deq(&hb[r * 2], d));
            if (memcmp(&rh[(size_t)r * 256 + d], &want, 2)) ++bad_st8;
        }

    printf("deq_elem: %d bad %s\n", bad_deq, bad_deq ? "FAIL" : "PASS (bitwise)");
    printf("ld8_lane: %d bad %s\n", bad_ld8, bad_ld8 ? "FAIL" : "PASS (bitwise)");
    printf("stage8_h2: %d bad %s\n", bad_st8, bad_st8 ? "FAIL" : "PASS (bitwise)");

    const bool ok = !bad_codes && !bad_scales && !bad_deq && !bad_ld8 && !bad_st8;
    printf("%s\n", ok ? "ALL PASS" : "FAIL");
    return ok ? 0 : 1;
}
