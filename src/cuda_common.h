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

// KV-cache format kind (Q27_KV): scalar fp16 (default) / fp8 E4M3 ("fp8") /
// turbo3 3-bit blocks ("turbo3", src/turbo3.cuh) / turbo3 V with plain fp16 K
// ("turbo3v" -- the GQA=6 escape hatch if turbo3-K craters, port spec risk
// section). Values 0/1 keep the old `bool fp8` call sites meaning-compatible
// (false->KV_F16, true->KV_FP8) where the parameter widened to int.
enum KvKind : int { KV_F16 = 0, KV_FP8 = 1, KV_T3 = 2, KV_T3V = 3 };

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

// Reports the __CUDA_ARCH__ the RESIDENT image was compiled for, as opposed
// to the physical device's compute capability (cudaDeviceGetAttribute).
// These two can disagree: CUDA's binary compatibility rule lets an sm_86
// cubin run on any sm_8x device with x>=6 (minor-version forward compat), so
// on a build that only carries sm_86 + sm_120 SASS, an sm_89 device (Ada)
// loads the sm_86 image. Any dispatch gated on `__CUDA_ARCH__ >= 890` (the
// e4m3 fp8-MMA kernels) is then compiled OUT for that image, even though
// cudaDeviceGetAttribute reports CC 8.9 and would say the opposite. Kernel
// dispatch decisions must gate on THIS value, never on the device attribute
// directly -- that class of bug was found and fixed once already in the
// prefill path (docs/BUILDLOG.md, 2026-07-09 "fp8 dispatch" finding) and
// then reintroduced in the newer fdmma verify-attention dispatch
// (src/spec3.cu), which checked cudaDeviceGetAttribute instead of this.
// Every arch-gated dispatch site should route through here so the fix can't
// regress a third time.
static __global__ void k_q27_arch_probe(int* out) {
#if defined(__CUDA_ARCH__)
    *out = __CUDA_ARCH__; // 860, 890, 1200, ...
#else
    *out = 0;
#endif
}

// Cached per-process -- the loaded image can't change at runtime.
static inline int q27_loaded_image_arch() {
    static int cached = -1;
    if (cached < 0) {
        int* d_arch = nullptr;
        int h_arch = 0;
        if (cudaMalloc(&d_arch, sizeof(int)) == cudaSuccess) {
            k_q27_arch_probe<<<1, 1>>>(d_arch);
            if (cudaMemcpy(&h_arch, d_arch, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
                h_arch = 0;
            cudaFree(d_arch);
        }
        cached = h_arch;
    }
    return cached;
}
#endif
