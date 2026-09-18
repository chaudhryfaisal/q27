// q27 reference kernels: correct first, fast later (M2 replaces the GEMVs).
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

namespace q27k {

// Dequantize an entire tensor to f32 (validation / small tensors only).
void dequant_q4(const uint8_t* W, const __half* S, float* out, int64_t rows, int64_t cols,
                cudaStream_t st = 0);
void dequant_q8(const int8_t* W, const __half* S, float* out, int64_t rows, int64_t cols,
                cudaStream_t st = 0);

// Quantized activation vector (mmvq-style): per 32-element block, int8 values with
// f32 scale and integer block sum. Two byte orders are kept: natural (for Q8 weights)
// and even/odd split per 8 (matching Q4 nibble unpack for dp4a).
struct XQuant {
    int8_t* nat = nullptr;   // [cols]
    uint2* eo = nullptr;     // [cols/8]: .x = bytes {x0,x2,x4,x6}, .y = {x1,x3,x5,x7}
    float* scale = nullptr;  // [cols/32]
    int* isum = nullptr;     // [cols/32] sum of quantized values per block
    // group-64 requantization (prefill MMA GEMM only; matches the Q4_G64
    // weight group so two K=32 mma steps share one fp dequant step). Distinct
    // VALUES from nat (amax over 64), not a re-scaling -- decode lanes leave
    // these null and are untouched.
    int8_t* nat64 = nullptr; // [cols]
    float* s64 = nullptr;    // [cols/64]
};
XQuant xquant_alloc(int64_t max_cols, bool g64 = false);
void quantize_x(const float* x, int64_t cols, const XQuant& xq, cudaStream_t st = 0);
void quantize_x_g64(const float* x, int64_t cols, const XQuant& xq, cudaStream_t st = 0);

// y[r] = sum_c W[r,c] * x[c].  W quantized row-major, reduction along contiguous axis.
// Q4/Q8 use dp4a against the pre-quantized activation vector.
void gemv_q4(const uint8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
             int64_t cols, cudaStream_t st = 0);
void gemv_q8(const int8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
             int64_t cols, cudaStream_t st = 0);

// Batched: one weight pass, N quantized activation columns -> y[n][rows]
// (y column-major by batch: y + n*rows). N in 2..5. The speculative-verify core.
// ys: per-column output pointers (ys[n][row]); no post-split copies needed.
void gemv_q4_n(const uint8_t* W, const __half* S, const XQuant* xqs, int nbatch,
               float* const* ys, int64_t rows, int64_t cols, cudaStream_t st = 0);
void gemv_q8_n(const int8_t* W, const __half* S, const XQuant* xqs, int nbatch,
               float* const* ys, int64_t rows, int64_t cols, cudaStream_t st = 0);
void gemv_f16(const __half* W, const float* x, float* y, int64_t rows, int64_t cols,
              cudaStream_t st = 0);

// ---- T2_G128 (ternary, Bonsai 2 Phase 2) ----
// On disk (FORMAT.md): element i of a row in 2-bit field (i%4)*2 of byte i/4,
// code c -> (c-1)*scale, fp16 scale per 128. The device copy is INTERLEAVED
// per 16-element word so that ((w >> 2k) & 0x03030303) yields the same four
// elements the Q4 kernels' even/odd activation words carry: field 4b+0 holds
// e[2b], 4b+1 e[2b+1], 4b+2 e[8+2b], 4b+3 e[9+2b]. t2_interleave_device does
// that in place right after upload (each word independent, idempotent-free:
// apply exactly once). The dot is dp4a(codes, x) - sum(x) per 32-block.
void t2_interleave_device(uint8_t* W, uint64_t bytes, cudaStream_t st = 0);
void gemv_t2(const uint8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
             int64_t cols, cudaStream_t st = 0);
void gemv_t2_n(const uint8_t* W, const __half* S, const XQuant* xqs, int nbatch,
               float* const* ys, int64_t rows, int64_t cols, cudaStream_t st = 0);

// y = x * rsqrt(mean(x^2) + eps) * w      (single vector, n elements)
void rmsnorm(const float* x, const float* w, float* y, int n, float eps, cudaStream_t st = 0);

// out[i] = silu(gate[i]) * up[i]
void silu_mul(const float* gate, const float* up, float* out, int n, cudaStream_t st = 0);

// out[0..cols) = dequantized row *d_token of a Q8_G128 matrix (embedding lookup)
void embed_row_q8(const int8_t* W, const __half* S, const int* d_token, int64_t cols, float* out,
                  cudaStream_t st = 0);

// Grid-merged multi-token variants for the speculative round: identical
// per-token work distribution, tokens mapped to an extra grid dimension
// (1 launch vs ntok). ntok selects how many lanes are live (brace inits
// with fewer entries leave the rest null, unread); shallower widths fill a
// prefix. width-12 2026-07-10: 16 slots (verify lanes a..l = 12 live max;
// slots sized 16 so a future W=16 revisit is struct-free). Structs ride
// kernel params by value -- slots beyond ntok are never read.
struct P3 { float* p[16]; };
struct CP3 { const float* p[16]; };
struct IP3 { const int* p[16]; }; // int twin (draft-token lanes); lived in spec3.cuh until the sampled tail (blocks.cuh) needed it too
struct XQ3 { XQuant q[16]; };

void rmsnorm3(CP3 x, const float* w, P3 y, int n, float eps, cudaStream_t st = 0, int ntok = 3);
void add3(P3 x, CP3 y, int n, cudaStream_t st = 0, int ntok = 3);
void silu_mul3(P3 g, CP3 u, int n, cudaStream_t st = 0, int ntok = 3);
void quantize3(CP3 x, int64_t cols, const XQ3& xq, cudaStream_t st = 0, int ntok = 3);
// Fused rmsnorm3 + quantize3 of the result (bitwise those two launches; the
// verify forward quantizes every normed activation it produces).
void rmsnorm3q(CP3 x, const float* w, P3 y, const XQ3& xq, int n, float eps, cudaStream_t st = 0,
               int ntok = 3);

// ---- Bonsai 2 activation rotation (docs/plans/2026-09-18-bonsai2-ternary.md)
// The pack stores every projection in a rotated basis W' = W R^T with
// R = (1/32) H_1024 S per contiguous 1024-block of the INPUT dimension
// (H natural-order Sylvester Walsh-Hadamard, S a fixed +-1 diagonal, one
// sign vector per input width). Before such a matmul the activation gets
// fwd: y = (1/32) H (s * x); after an embedding lookup of a rotated row it
// gets inv: x = s * ((1/32) H y). In place, fp32, fixed operand order
// (bitwise deterministic, CPU-equal). width % 1024 == 0; signs[width].
void hadamard1024(float* x, const float* signs, int width, bool inv, cudaStream_t st = 0);
// T rows at row_stride floats apart (prefill).
void hadamard1024_rows(float* x, const float* signs, int width, int rows, long row_stride,
                       bool inv, cudaStream_t st = 0);
// Up to 16 lane vectors (speculative verify / multi-lane paths).
void hadamard1024_lanes(P3 x, const float* signs, int width, int nlanes, bool inv,
                        cudaStream_t st = 0);
// GDN value-head order: the engine's og is tiled [rep][nk][hd] (the GGUF
// convention); a Hadamard-folded ssm_out was folded in the training (grouped)
// order [nk][rep][hd], so permute before the rotation:
//   out[k*rep*hd + r*hd + h] = in[r*nk*hd + k*hd + h].
void gdn_v_tiled_to_grouped(const float* in, float* out, int hd, int nk, int rep,
                            cudaStream_t st = 0);
void gdn_v_tiled_to_grouped_rows(const float* in, float* out, int hd, int nk, int rep, int rows,
                                 long stride, cudaStream_t st = 0);
void gdn_v_tiled_to_grouped_lanes(CP3 in, P3 out, int hd, int nk, int rep, int nlanes,
                                  cudaStream_t st = 0);

} // namespace q27k
