# Plan: implementing the ninfer findings (2026-08-15)

What the ninfer recon (2026-08-14, BUILDLOG) established, in ROI order, and
what implementing each piece takes. ninfer is the q27 sibling engine
(5090-only, same checkpoints). One of its findings is already shipped: its
single XML-dialect tool parser corroborated our native-dialect fix, which
landed with per-model keying (00b6a7d..37caa07). Four remain.

Ground rule carried over from the engine roadmap: every phase has a
pre-declared bar and closes with a measurement, win or lose. No phase starts
before the one ahead of it reports, except phase 0, which is free.

## Phase 0 (hours): pin the sm_120a capability in-tree -- DONE 2026-08-15

Landed as `tools/microbench_mxf4.cu` + `build/microbench_mxf4` (sm_120a-only
target, trap documented in the Makefile). Ratio table in the BUILDLOG entry
of the same date: fp4 W4A4 clears the phase 2 gate at every measured point,
2.13x-2.97x vs the live W4A8-int path (gate was 1.3x), 2.5x at the PF_T=1024
production point, fp4 kernel at dense-peak TFLOPS. Phase 2 is GO on this
table.

The fp4 recon proof -- block-scaled mxf4nvf4 mma.sync executing on haight --
lives only in a session scratchpad and the BUILDLOG prose. The old NO-GO was
a toolchain trap (missing `-gencode arch=compute_120a,code=sm_120a`), and the
Makefile still carries only 86/89/120, so the trap would fire again today for
anyone adding an fp4 kernel.

- Land `tools/microbench_mxf4.cu` (the proof kernel, cleaned): fp4 W4A4
  block-scaled GEMM at the three prefill-relevant shapes (qkv/ffn projections
  at M = 512/2048/8192), timed against the current fp8/fp16 paths.
- Makefile: add the `compute_120a` gencode to THAT target only (the tri-arch
  serving binary stays as-is until a kernel needs it), with a comment naming
  the trap.
- Bar: none -- this is capability pinning, not a lever. Output is the
  measured GEMM ratio table that phase 2 decides on.

## Phase 1 (days): INT8-G64 KV arm in the tail study -- DONE 2026-08-15, NO-GO

Measured same day (BUILDLOG entry of the same date): 64 catastrophic at
25.6 KB/token vs turbo5k's 63 at 18.6 on the same binary -- identical tail
at 1.37x the bytes. The finding: groupwise-int8 K equals 5-bit WHT+Lloyd-Max
K on the catastrophic axis at 1.6x the K bytes, i.e. the rotation+centroid
stack is worth ~3 bits. No serving port; the arm stays in-tree as a
diagnostic kind. Steal-list item 1 CLOSED.

ninfer serves these checkpoints with INT8 KV in groups of 64. We now know
PPL is blind on the KV axis and the per-position tail study (`--nll-dump`,
64K fp16-referenced) is the instrument. turbo5k just shipped through exactly
this pipeline, so the harness exists end to end.

CORRECTED 2026-08-15, before the run -- the paragraph this replaces had two
factual errors. (1) There is no host-side sim path in the NLL harness; "the
turbo3v pattern" means a KvKind with zero NEW kernels (branches in the
shared store kernel, the fd2 template, and the prefill staging, force-routed
to fd2 by the dispatch guard). (2) "~18.0 KB/token" was an arithmetic slip:
int8 bytes = fp8 bytes, so both-sides INT8-G64 is ~37 KB/token (ninfer's own
accounting: 33,792 B over their 16 attn layers) -- MORE than fp8's 36.0, not
between turbo5k and fp8.

- The arm as built: K = INT8-G64, bit-faithful to ninfer's codec (64-dim
  groups, fp32 absmax, fp16_rne(absmax/127) scale, reciprocal from the
  ROUNDED scale, RNE, clamp [-127,127], no rotation); V = turbo3, the
  turbo3v/turbo5k denominator, keeping the row comparable on the K ladder
  (fp16 K = 52 catastrophic, 5-bit K = 65). 1056 B/token/layer K + 400 V =
  25.6 KB/token at the 18-pair accounting. src/i8g64.cuh + Q27_KV=int8g64,
  microtest tools/i8g64_test.cu (bitwise oracle).
- Bar, re-declared before the measurement: STRONG PASS = catastrophic <= 19
  (beats fp8 at 29% fewer bytes -- the original second clause). PASS =
  catastrophic < 65 (a new Pareto-frontier point: nothing exists between
  turbo5k 18.6/65 and fp8 36.0/19). NO-GO = catastrophic >= 65 (more bytes
  than turbo5k without beating its tail); the arm closes as a NO-GO entry
  and no serving port happens.
- Only on a pass: serving-grade port (fdmma/H16 legs, canonical + needle
  gates, throughput ladder). That is the expensive half; the study arm is
  the cheap gate in front of it.

## Phase 2 (1-2 weeks, gated on phase 0's table): fp4 W4A4 prefill

The falsified NO-GO reopens the biggest single-kernel lever. Prefill is the
compute-bound half, agentic traffic re-prefills on every cache miss, and
ninfer demonstrates fp4 MMA carrying real serving on this exact card.

- Gate (from phase 0): mxf4nvf4 GEMM must clear 1.3x over the current
  prefill inner path at M >= 512 on the real projection shapes. Below that,
  integration cost eats the win -- close as measured NO-GO.
- Stage A: weight-side. q4s tiers are already 4-bit but not in fp4
  block-scale layout; add a repack leg producing the mxf4 block format for
  the projection weights only (attention + FFN, NOT the SSM path -- the
  ssm_out lesson says that path's cancellation structure is fragile).
- Stage B: activation-side fp4 quantize + the prefill GEMM swap, opt-in
  behind `Q27_PF=fp4`. Decode stays untouched (memory-bound; fp4 buys
  nothing at M=1).
- Bars: canonical unchanged (greedy 128-tok walk is decode-side, must hold
  bitwise); prefill NLL delta within the fp8-KV envelope on the full battery
  INCLUDING long single-pass (chunked PPL is blind on recurrent-path damage);
  wall-clock prefill >= 1.25x on the 17K-27K prompts real agentic traffic
  shows. Any miss = NO-GO entry with the numbers.

## Phase 3 (redesign-scale, separate spec): single-engine batched decode

The concurrent ladder proved the C=8 gap is architectural: each q27 slot
carries ~8.2 GB of fixed engine stack, so a 32 GB card fits two engines,
while ninfer batches one engine and scales 2.88-5.67x at C=8 with marginal
cost ~KV. This is the only item that changes q27's shape rather than adding
a leg.

- First deliverable is a SPEC, not code: decompose the 8.2 GB (KV ctx,
  CUDA graphs, staging, logits, spec-decode state), identify what is
  per-sequence vs per-engine, and design the conductor around one engine
  owning N sequence states with batched MTP verify (the fp8/fp4 MMA paths
  get efficient exactly at batch width, which is ninfer's compounding trick).
- Bar for greenlighting implementation: the spec must show a credible path
  to C=4 in 32 GB with per-slot marginal cost under 2 GB + KV.
- Bar for the implementation itself: aggregate t/s ladder at C=1/2/4/8
  against the 2026-08-14 measured baseline (143.0 t/s C=1, 1.28x at 2), with
  single-stream regression capped at 3%.

## Parked: lm-head-draft shortlist

The recon flagged ninfer's lm-head-draft but did not extract the mechanism.
Before anything else: a half-day read of their draft path to write down what
it actually does and whether it composes with q27's MTP ladder (which is
already the documented win -- x1.73 on top of the same head). If it is an
alternative drafter rather than a compounding one, it dies here: the MTP
ladder closed with better numbers than any drafter swap has shown.

## Scheduling note

None of this preempts the 3.8 serving-recipe work in flight (recovery
campaign / suite3). Phase 0 and the phase 3 spec are background-friendly;
phases 1-2 want the 5090 for measurement windows and should queue behind the
current campaign.
