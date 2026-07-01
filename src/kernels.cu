#include "cuda_common.h"
#include "kernels.cuh"

namespace q27k {

// ---------------- dequant ----------------

__global__ void k_dequant_q4(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                             float* __restrict__ out, int64_t rows, int64_t cols) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows * cols;
    if (idx >= n) return;
    int64_t r = idx / cols, c = idx % cols;
    uint8_t b = W[r * (cols / 2) + c / 2];
    int nib = (c & 1) ? (b >> 4) : (b & 0xF);
    float s = __half2float(S[r * (cols / 64) + c / 64]);
    out[idx] = (nib - 8) * s;
}

__global__ void k_dequant_q8(const int8_t* __restrict__ W, const __half* __restrict__ S,
                             float* __restrict__ out, int64_t rows, int64_t cols) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows * cols;
    if (idx >= n) return;
    int64_t r = idx / cols, c = idx % cols;
    float s = __half2float(S[r * (cols / 128) + c / 128]);
    out[idx] = (float)W[r * cols + c] * s;
}

void dequant_q4(const uint8_t* W, const __half* S, float* out, int64_t rows, int64_t cols,
                cudaStream_t st) {
    int64_t n = rows * cols;
    k_dequant_q4<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(W, S, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}
void dequant_q8(const int8_t* W, const __half* S, float* out, int64_t rows, int64_t cols,
                cudaStream_t st) {
    int64_t n = rows * cols;
    k_dequant_q8<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(W, S, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- GEMV (reference) ----------------
// One block per output row, 256 threads grid-stride the reduction axis.

template <int BLOCK>
__device__ __forceinline__ float block_reduce(float v) {
    __shared__ float sh[BLOCK];
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    return sh[0];
}

__global__ void k_gemv_q4(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                          const float* __restrict__ x, float* __restrict__ y, int64_t cols) {
    int64_t r = blockIdx.x;
    const uint8_t* wr = W + r * (cols / 2);
    const __half* sr = S + r * (cols / 64);
    float acc = 0.f;
    for (int64_t c = (int64_t)threadIdx.x * 2; c < cols; c += (int64_t)blockDim.x * 2) {
        uint8_t b = wr[c / 2];
        float s = __half2float(sr[c / 64]); // c even => c, c+1 share a group (64 is even)
        acc += ((int)(b & 0xF) - 8) * s * x[c];
        acc += ((int)(b >> 4) - 8) * s * x[c + 1];
    }
    float sum = block_reduce<256>(acc);
    if (threadIdx.x == 0) y[r] = sum;
}

__global__ void k_gemv_q8(const int8_t* __restrict__ W, const __half* __restrict__ S,
                          const float* __restrict__ x, float* __restrict__ y, int64_t cols) {
    int64_t r = blockIdx.x;
    const int8_t* wr = W + r * cols;
    const __half* sr = S + r * (cols / 128);
    float acc = 0.f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
        acc += (float)wr[c] * __half2float(sr[c / 128]) * x[c];
    float sum = block_reduce<256>(acc);
    if (threadIdx.x == 0) y[r] = sum;
}

__global__ void k_gemv_f16(const __half* __restrict__ W, const float* __restrict__ x,
                           float* __restrict__ y, int64_t cols) {
    int64_t r = blockIdx.x;
    const __half* wr = W + r * cols;
    float acc = 0.f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
        acc += __half2float(wr[c]) * x[c];
    float sum = block_reduce<256>(acc);
    if (threadIdx.x == 0) y[r] = sum;
}

void gemv_q4(const uint8_t* W, const __half* S, const float* x, float* y, int64_t rows,
             int64_t cols, cudaStream_t st) {
    k_gemv_q4<<<(unsigned)rows, 256, 0, st>>>(W, S, x, y, cols);
    CUDA_CHECK(cudaGetLastError());
}
void gemv_q8(const int8_t* W, const __half* S, const float* x, float* y, int64_t rows,
             int64_t cols, cudaStream_t st) {
    k_gemv_q8<<<(unsigned)rows, 256, 0, st>>>(W, S, x, y, cols);
    CUDA_CHECK(cudaGetLastError());
}
void gemv_f16(const __half* W, const float* x, float* y, int64_t rows, int64_t cols,
              cudaStream_t st) {
    k_gemv_f16<<<(unsigned)rows, 256, 0, st>>>(W, x, y, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- elementwise ----------------

__global__ void k_rmsnorm(const float* __restrict__ x, const float* __restrict__ w,
                          float* __restrict__ y, int n, float eps) {
    __shared__ float sh[256];
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += x[i] * x[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
}

__global__ void k_silu_mul(const float* __restrict__ g, const float* __restrict__ u,
                           float* __restrict__ o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = g[i];
    o[i] = (v / (1.f + expf(-v))) * u[i];
}

__global__ void k_embed_row_q8(const int8_t* __restrict__ W, const __half* __restrict__ S,
                               int64_t row, int64_t cols, float* __restrict__ out) {
    const int8_t* wr = W + row * cols;
    const __half* sr = S + row * (cols / 128);
    for (int64_t c = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; c < cols;
         c += (int64_t)gridDim.x * blockDim.x)
        out[c] = (float)wr[c] * __half2float(sr[c / 128]);
}

void rmsnorm(const float* x, const float* w, float* y, int n, float eps, cudaStream_t st) {
    k_rmsnorm<<<1, 256, 0, st>>>(x, w, y, n, eps);
    CUDA_CHECK(cudaGetLastError());
}
void silu_mul(const float* g, const float* u, float* o, int n, cudaStream_t st) {
    k_silu_mul<<<(n + 255) / 256, 256, 0, st>>>(g, u, o, n);
    CUDA_CHECK(cudaGetLastError());
}
void embed_row_q8(const int8_t* W, const __half* S, int64_t row, int64_t cols, float* out,
                  cudaStream_t st) {
    k_embed_row_q8<<<8, 256, 0, st>>>(W, S, row, cols, out);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace q27k
