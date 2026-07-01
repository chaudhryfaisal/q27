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

// y[r] = sum_c W[r,c] * x[c].  W quantized row-major, reduction along contiguous axis.
void gemv_q4(const uint8_t* W, const __half* S, const float* x, float* y, int64_t rows,
             int64_t cols, cudaStream_t st = 0);
void gemv_q8(const int8_t* W, const __half* S, const float* x, float* y, int64_t rows,
             int64_t cols, cudaStream_t st = 0);
void gemv_f16(const __half* W, const float* x, float* y, int64_t rows, int64_t cols,
              cudaStream_t st = 0);

// y = x * rsqrt(mean(x^2) + eps) * w      (single vector, n elements)
void rmsnorm(const float* x, const float* w, float* y, int n, float eps, cudaStream_t st = 0);

// out[i] = silu(gate[i]) * up[i]
void silu_mul(const float* gate, const float* up, float* out, int n, cudaStream_t st = 0);

// out[0..cols) = dequantized row `row` of a Q8_G128 matrix (embedding lookup)
void embed_row_q8(const int8_t* W, const __half* S, int64_t row, int64_t cols, float* out,
                  cudaStream_t st = 0);

} // namespace q27k
