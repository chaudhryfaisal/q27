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
