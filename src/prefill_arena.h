// M3a: the process-wide PREFILL ARENA (plan docs/plans/2026-08-16-m3-prefill-arena.md).
//
// Every Engine used to allocate its own ~0.73 GB of chunk-sized prefill
// scratch (the PF_T x {N_EMBD, GDN_CH, GDN_V, N_FFN, ...} staging buffers,
// the split-attention partials, the fp4 activation pair, the g64 quantized
// FFN staging, and the WY/split-K panels). At 7 slots that is ~5 GB of
// duplicate scratch -- HALF the measured ~1.48 GB/slot admission marginal --
// and only one copy is ever live, because prefill is serialized process-wide
// by the GpuGate (api_common.h): a request thread holds a Lease across its
// whole prefill and the conductor holds one across each decode round.
//
// THE HAZARD THIS TYPE EXISTS TO CLOSE. Sharing scratch across engines is
// exactly what the 2026-07-04 R1b-prereq entry made per-engine to fix: with
// round-granularity preemption, "two engines with chunks in flight would race
// one panel set across streams". The gate serializes ISSUE, but engines run
// on DIFFERENT streams, and a gate handover is only guaranteed drained when a
// yield actually fires (GpuGate::Lease documents an explicit exemption for
// work still in flight at release). So exclusivity alone is not enough: the
// arena must also guarantee the previous user's kernels have RETIRED before
// the next one writes. claim() does that -- on an owner change it
// synchronizes the previous owner's stream. One sync per prefill-owner change
// (prefills are 0.1-70 s) is unmeasurable; the alternative is a silent
// cross-stream stomp that reads as a quality regression, not a crash.
//
// NOT IN HERE, deliberately: anything a DECODE round touches. The fused round
// forks per-member work onto side streams, so a shared buffer there would
// race with no gate boundary between the racers. M1's commit Fold used to
// borrow the prefill oT for its <= W_MAX output rows; it now owns a small
// per-engine fold_o instead (engine.cuh), which is what let this arena become
// prefill-only. Keep it that way: if a new consumer is not strictly inside a
// prefill chunk, it does not belong in this struct.
#pragma once
#include <cstdint>
#include <cstdio>

#include "cuda_common.h"
#include "kernels.cuh"
#include "prefill.cuh"

namespace q27 {

// Sizes come from the engine's own constants; the header takes them as ctor
// args so this file stays free of the Engine's include order.
struct PrefillArena {
    // chunk staging (all PF_T rows)
    float *hT = nullptr, *x1T = nullptr, *yT = nullptr, *qkvT = nullptr, *convT = nullptr;
    float *zT = nullptr, *oT = nullptr, *ogT = nullptr, *qgT = nullptr, *kT = nullptr;
    float *vT = nullptr, *attnT = nullptr, *alphaT = nullptr, *betarT = nullptr;
    float *gT = nullptr, *betaT = nullptr, *ffnGT = nullptr, *ffnUT = nullptr;
    float *embT = nullptr, *ehnT = nullptr, *xmtpT = nullptr, *pf_part = nullptr;
    q27k::XQuant xqT{};
    uint8_t *pf4_ac = nullptr, *pf4_as = nullptr; // fp4 activation pair
    q27k::WyScratch wy_scratch;
    q27k::SplitKScratch splitk_ws;

    bool enabled() const { return hT != nullptr; }
    size_t bytes() const { return total_bytes; }

    // Allocate once, before any Engine exists (same discipline as KvPool:
    // never during serving -- graph capture under cudaStreamCaptureModeGlobal
    // makes allocation from any thread illegal).
    void init(int PF_T, int N_EMBD, int N_FFN, int N_HEAD, int N_KV, int HEAD_DIM,
              int GDN_CH, int GDN_V, int GDN_HEADS) {
        auto fal = [&](size_t n) {
            float* p;
            CUDA_CHECK(cudaMalloc((void**)&p, n * 4));
            total_bytes += n * 4;
            return p;
        };
        const size_t T = (size_t)PF_T;
        hT = fal(T * N_EMBD);   x1T = fal(T * N_EMBD);  yT = fal(T * N_EMBD);
        qkvT = fal(T * GDN_CH); convT = fal(T * GDN_CH);
        zT = fal(T * GDN_V);    oT = fal(T * GDN_V);    ogT = fal(T * GDN_V);
        qgT = fal(T * N_HEAD * 2 * HEAD_DIM);
        kT = fal(T * N_KV * HEAD_DIM);  vT = fal(T * N_KV * HEAD_DIM);
        attnT = fal(T * N_HEAD * HEAD_DIM);
        alphaT = fal(T * GDN_HEADS); betarT = fal(T * GDN_HEADS);
        gT = fal(T * GDN_HEADS);     betaT = fal(T * GDN_HEADS);
        ffnGT = fal(T * N_FFN);      ffnUT = fal(T * N_FFN);
        embT = fal(T * N_EMBD);      ehnT = fal(T * 2 * N_EMBD);
        xmtpT = fal(T * N_EMBD);
        pf_part = fal((size_t)N_HEAD * T * q27k::PF_SPLIT_MAX * 258);
        xqT = q27k::xquant_alloc(T * N_FFN, /*g64=*/true);
        {
            const size_t tp = (T + 127) & ~(size_t)127;
            CUDA_CHECK(cudaMalloc((void**)&pf4_ac, tp * N_FFN / 2));
            CUDA_CHECK(cudaMalloc((void**)&pf4_as, tp * N_FFN / 16));
            total_bytes += tp * N_FFN / 2 + tp * N_FFN / 16;
        }
        q27k::wy_scratch_reserve(&wy_scratch, PF_T);
        q27k::splitk_scratch_reserve(&splitk_ws);
    }

    // Hand the arena to `owner`, whose prefill work will run on `stm`. On an
    // owner CHANGE this drains the previous owner's stream: its last chunk's
    // kernels may still be reading these buffers (the gate's drained-handover
    // invariant has a documented in-flight exemption, and end-of-prefill
    // releases the Lease without a yield when nobody is queued). Same-owner
    // re-claims are free -- an engine is already ordered against itself.
    // Call from the prefill entry point, INSIDE the caller's gate lease.
    void claim(const void* owner, cudaStream_t stm) {
        if (last_owner && last_owner != owner)
            CUDA_CHECK(cudaStreamSynchronize(last_stm));
        last_owner = owner;
        last_stm = stm;
    }
    // An engine going away (or releasing its KV lineage) must not leave the
    // arena pointing at a dead stream.
    void forget(const void* owner) {
        if (last_owner == owner) { last_owner = nullptr; last_stm = nullptr; }
    }

private:
    const void* last_owner = nullptr;
    cudaStream_t last_stm = nullptr;
    size_t total_bytes = 0;
};

} // namespace q27
