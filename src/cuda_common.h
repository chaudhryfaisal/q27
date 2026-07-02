#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x)                                                                  \
    do {                                                                               \
        cudaError_t err_ = (x);                                                        \
        if (err_ != cudaSuccess) {                                                     \
            fprintf(stderr, "CUDA error: %s\n  at %s:%d: %s\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__, #x);                                           \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#ifdef __CUDACC__
#include <cuda_fp8.h>
// KV-cache element conversions (P2): fp16 default, fp8 E4M3 opt-in (Q27_KV=fp8).
// E4M3 store saturates to +-448; kvstats probe (2026-07-02, 8K wikitext tokens)
// measured K amax <= 21.8, V amax <= 118.6 across all 17 attention layers, so
// scale-free E4M3 has >=3.8x headroom and per-row scales buy nothing.
__device__ __forceinline__ float kv2f(__half x) { return __half2float(x); }
__device__ __forceinline__ float kv2f(__nv_fp8_e4m3 x) { return float(x); }
__device__ __forceinline__ __half kv2h(__half x) { return x; }
__device__ __forceinline__ __half kv2h(__nv_fp8_e4m3 x) {
    return __half(__nv_cvt_fp8_to_halfraw(x.__x, __NV_E4M3));
}
__device__ __forceinline__ void kv_set(__half& d, float x) { d = __float2half_rn(x); }
__device__ __forceinline__ void kv_set(__nv_fp8_e4m3& d, float x) { d = __nv_fp8_e4m3(x); }
#endif
