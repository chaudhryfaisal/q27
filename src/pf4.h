// fp4 W4A4 prefill (Q27_PREFILL=fp4) -- host API. Kernels live in src/pf4.cu,
// which is the ONE translation unit compiled -gencode arch=compute_120a only
// (the block-scaled mxf4nvf4 MMA does not exist behind plain sm_120; see the
// Makefile MXF4FLAGS note). Binaries for other archs still link this TU --
// its kernels simply have no SASS there and pf4_on() refuses to route.
// Plan: docs/plans/2026-08-15-ninfer-steals.md phase 2.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace q27k {

// True when Q27_PREFILL=fp4 AND device 0 is sm_120 (re-read per call so tests
// can flip paths in-process, like prefill_use_mma; the arch probe is cached).
bool pf4_on();

// Instrument mode (Q27_PF4_INSTRUMENT=1, CLI gates only): pf4_on() plus the
// Q4 copies of sidecar-shadowed projections stay OFF the card and the decode
// graph is never built -- prefill-only instruments (--nll-long, --pf) at
// depths the dual-copy footprint cannot host. NEVER for serving.
bool pf4_instrument();

// Activation nvfp4 quantize (fp32 -> e2m1 codes + ue4m3 scale per 16) plus the
// block-scaled W4A4 GEMM. y[t*rows + row], fp32 -- the mmT contract.
// wc/ws: an FP4_G16 sidecar's data/scales (rows x cols, cols % 256 == 0,
// rows % 128 == 0 -- fatal otherwise; the repack include-list guarantees it).
// ac/as: caller-owned activation scratch sized >= ceil128(T)*cols/2 and /16.
void pf4_gemm_T(const void* wc, const void* ws, const float* xT, void* ac, void* as,
                float* y, int64_t rows, int64_t cols, int T, cudaStream_t st);

} // namespace q27k
