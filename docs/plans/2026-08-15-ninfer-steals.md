# Plan: implementing the ninfer findings (2026-08-15)

What the ninfer recon (2026-08-14, BUILDLOG) established, in ROI order, and
what implementing each piece takes. ninfer is the q27 sibling engine
(5090-only, same checkpoints). One of its findings is already shipped: its
single XML-dialect tool parser corroborated our native-dialect fix, which
landed with per-model keying (00b6a7d..37caa07). Four remain.

Ground rule carried over from the engine roadmap: every phase has a
pre-declared bar and closes with a measurement, win or lose. No phase starts
before the one ahead of it reports, except phase 0, which is free.

## Phase 0 (hours): pin the sm_120a capability in-tree

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

## Phase 1 (days): INT8-G64 KV arm in the tail study

ninfer serves these checkpoints with INT8 KV in groups of 64. We now know
PPL is blind on the KV axis and the per-position tail study (`--nll-dump`,
64K fp16-referenced) is the instrument. turbo5k just shipped through exactly
this pipeline, so the harness exists end to end.

- Add INT8-G64 as a STUDY-ONLY arm first: host-side quantize/dequantize in
  the NLL harness (the turbo3v pattern), no CUDA kernels. ~18.0 KB/token,
  between turbo5k (18.6) and fp8 (34).
- Bar, pre-declared: catastrophic-position count must beat turbo5k's 65 at
  comparable bytes, or beat fp8's 19 at meaningfully fewer bytes. Miss both
  and the arm closes as a NO-GO entry; nothing gets built.
- Only on a pass: port to a serving format (turbo5.cuh is the template --
  format struct, fd2 + H16 legs, prefill leg, canonical + needle gates).
  That is the expensive half; the study arm is the cheap gate in front of it.

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
