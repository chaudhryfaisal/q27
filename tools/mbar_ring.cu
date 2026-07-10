// Phase-A day-1 risk gate (docs/plans/2026-07-09-prefill-async-rewrite.md):
// does a producer/consumer mbarrier ring actually decouple load and math
// warps on sm_120, or does cuda::pipeline degenerate into full-CTA barriers?
//
// Mimics the prefill-attn phase structure at its real sizes: stages of
// 32x272 fp8 "K" + 32x256 fp8 "V" (16.5 KB) streamed from an L2-busting
// global buffer; consumer warps run an FMA loop per stage sized like the
// QK^T+PV work. Variant SYNC = the current kernel's shape (all warps load,
// __syncthreads, all warps math). Variant ASYNC = 2 producer warps own
// cp.async via cuda::memcpy_async, 6 consumer warps only math, 3-stage
// pipeline. Identical total bytes + flops; wall-clock ratio is the gate.
//
// Build: /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120 tools/mbar_ring.cu -o build/mbar_ring
// GO bar: async >= 1.3x sync. Kill: <= 1.1x (library path emits CTA barriers).
#include <cooperative_groups.h>
#include <cuda/pipeline>

#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

constexpr int PP = 32, LDK = 272, HD = 256;
constexpr int STAGE_BYTES = PP * LDK + PP * HD; // 16.9 KB
constexpr int STAGES = 3;
constexpr int NTILES = 512;      // KV stream length per CTA
#ifndef MATH_ITERS
#define MATH_ITERS 340  // per stage per thread: ~QK^T+PV-ish latency
#endif

#define CK(x)                                                              \
    do {                                                                   \
        cudaError_t e_ = (x);                                              \
        if (e_ != cudaSuccess) {                                           \
            fprintf(stderr, "%s @%d\n", cudaGetErrorString(e_), __LINE__); \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

__device__ float math_over(const unsigned char* stage, int lane, float acc) {
    const float* f = (const float*)stage;
#pragma unroll 4
    for (int i = 0; i < MATH_ITERS; i++) {
        float v = f[(lane * 37 + i * 61) % (STAGE_BYTES / 4)];
        acc = fmaf(acc, 1.0000001f, v);
    }
    return acc;
}

// -------- SYNC baseline: current kernel's phase structure --------
__global__ void __launch_bounds__(256, 1) k_sync(const unsigned char* g, float* out) {
    extern __shared__ unsigned char smem[];
    unsigned char* stage = smem;
    float acc = 0.f;
    const size_t base = (size_t)blockIdx.x * NTILES * STAGE_BYTES;
    for (int t = 0; t < NTILES; t++) {
        __syncthreads();
        for (int i = threadIdx.x * 16; i < STAGE_BYTES; i += blockDim.x * 16)
            *(float4*)(stage + i) = *(const float4*)(g + base + (size_t)t * STAGE_BYTES + i);
        __syncthreads();
        acc = math_over(stage, threadIdx.x, acc);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// -------- ASYNC: 2 producer warps + 6 consumer warps, 3-stage ring --------
__global__ void __launch_bounds__(256, 1) k_async(const unsigned char* g, float* out) {
    extern __shared__ unsigned char smem[];
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> pss;
    auto block = cg::this_thread_block();
    const bool producer = threadIdx.x < 64; // warps 0-1
    auto pipe = cuda::make_pipeline(block, &pss,
                                    producer ? cuda::pipeline_role::producer
                                             : cuda::pipeline_role::consumer);
    const size_t base = (size_t)blockIdx.x * NTILES * STAGE_BYTES;
    float acc = 0.f;
    if (producer) {
        for (int t = 0; t < NTILES; t++) {
            pipe.producer_acquire();
            unsigned char* stage = smem + (t % STAGES) * STAGE_BYTES;
            for (int i = threadIdx.x * 16; i < STAGE_BYTES; i += 64 * 16)
                cuda::memcpy_async(stage + i, g + base + (size_t)t * STAGE_BYTES + i,
                                   cuda::aligned_size_t<16>(16), pipe);
            pipe.producer_commit();
        }
    } else {
        for (int t = 0; t < NTILES; t++) {
            pipe.consumer_wait();
            const unsigned char* stage = smem + (t % STAGES) * STAGE_BYTES;
            acc = math_over(stage, threadIdx.x - 64, acc);
            pipe.consumer_release();
        }
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <typename K>
static float run(K kern, const unsigned char* g, float* out, int blocks, size_t smem, int reps) {
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kern<<<blocks, 256, smem>>>(g, out); // warm
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int r = 0; r < reps; r++) kern<<<blocks, 256, smem>>>(g, out);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    return ms / reps;
}

int main() {
    const int blocks = 170; // one per SM: per-SM pipeline behavior is the question
    const size_t gbytes = (size_t)blocks * NTILES * STAGE_BYTES;
    unsigned char* g;
    float* out;
    CK(cudaMalloc(&g, gbytes));
    CK(cudaMemset(g, 0x3c, gbytes)); // ~0.011f pattern
    CK(cudaMalloc(&out, blocks * 256 * 4));
    const size_t sm_sync = STAGE_BYTES, sm_async = (size_t)STAGES * STAGE_BYTES;
    float ts = run(k_sync, g, out, blocks, sm_sync, 20);
    float ta = run(k_async, g, out, blocks, sm_async, 20);
    printf("sync  (CTA-barrier phases): %.3f ms\n", ts);
    printf("async (mbarrier ring x%d) : %.3f ms\n", STAGES, ta);
    printf("ratio: %.2fx  -> %s\n", ts / ta,
           ts / ta >= 1.3f ? "GO (>=1.3x)" : ts / ta > 1.1f ? "MARGINAL" : "KILL (<=1.1x)");
    return 0;
}
