# Spec: single-engine batched decode (ninfer-steals phase 3)

Deliverable of docs/plans/2026-08-15-ninfer-steals.md phase 3. Pre-declared
greenlight bar: a credible path to C=4 in 32 GB with per-slot marginal cost
under 2 GB + KV. Verdict at the bottom: **GO, with margin**, as a staged
sequence whose first stage pays for itself under the CURRENT architecture.

Sources: full allocation audit of Engine::init, the conductor/fusion map,
and the per-kernel batchability survey (2026-08-15, three-track recon; the
ninfer contrast draws on their docs/maintainer/paged-kv-cache.md and
concurrent-inference-architecture.md, recon'd 2026-08-14/15).

## 0. The premise was wrong in our favor

The ladder headline "each slot costs ~8.2 GB of fixed engine stack" is the
ADMISSION ESTIMATOR, not a measurement (`server.cu:738-743`):
ENG_FIXED_BYTES = 6.611 GB of constants (0.89 base + 1.56 graph zoo + 2.04
GDN role sets + 2.20 co-residency fudge - 0.08) plus KV for a **32,768-token
window nobody asked for** -- `slot1_ctx` defaults to 32768 (`server.cu:288`)
and an explicit `--ctx` only propagates to slots 1+ in the auto branch
(`server.cu:666`). CONFIG BUG, filed in §6.

What two slots MEASURABLY consumed in the 2026-08-14 ladder: 11.16 GB total
= **~5.6 GB/slot** (explicit cudaMallocs 3.53 GB @16K + ~1.6-1.8 GB of graph
zoo + allocator slack). The estimator also double-charges the per-PROCESS
CUDA context (0.89 GB) per slot, carries a 2.2 GB fudge against ~1.0 GB of
real shareables, and sizes KV at 18 pairs where the engine allocates 17
(`engine.cuh:947` loop vs `server.cu:590` -- a 6% conservative bias).

## 1. Decomposition of a slot (ctx 16384, fp8 KV, W12 CC profile)

| block | size | scope under a shared engine |
|---|--:|---|
| KV: 17 attn pairs + MTP pair | 0.602 GB | per-sequence, irreducible (34 KB/tok) |
| GDN committed state: S[il] + conv_ring (role 0) | 0.157 GB | per-sequence, irreducible |
| GDN snapshot: S_snap + ring_snap | 0.150 GB | per-conversation (prefix cache) |
| **GDN spare roles: S_sp/ring_sp x (W_PLUMB-1)** | **1.646 GB** | **speculation-WIDTH state, not sequence state -- the lever** |
| prefill scratch (T-buffers, pf_part, xqT, splitk, wy, pf4) | 0.836 GB | per-engine (prefill already serialized behind the FIFO gate) |
| graph zoo: 254 execs at CC defaults (~6-7 MB each) | ~1.6 GB | per-engine once graphs stop baking per-engine pointers |
| decode/spec state (logits2, fd scratch, lanes, outcome) | ~0.08 GB | mostly per-engine; scratch/logits grow with K under batching |
| scalars, masks, misc | ~0.02 GB | trivia |
| CUDA context/modules (estimator's kEngBase) | 0.89 GB | per-PROCESS -- was never per-slot |

Irreducible per-sequence floor at 16K: **0.88 GB INCLUDING KV** (0.60 KV +
0.157 committed GDN + 0.150 snapshot + trivia). Everything else is either a
width knob (1.65 GB) or engine-level duplication (2.4+ GB).

## 2. What already exists (P0-P3 shipped more of phase 3 than the plan knew)

The batched weight sweep IS BUILT. `Conductor::fused_verify_round`
(`conductor.h:604-694`) runs ONE sweep at M = sum of granted widths across
members via `build_union_view` (`:399-455`), whose LaneView slots point into
each owning engine's lane buffers. `vgemm_verify` takes per-lane pointer
arrays and a lane count -- no position, no KV pointer, no per-lane state
(`vgemm.cu:27-213`) -- and its splitK workspace is already sized for all 16
lanes (`vgemm.cu:250-263`). `gemv_q4_n/q8_n` batch 2..16 lanes. Every
elementwise `*3` kernel takes 16-slot pointer structs with a lane grid dim.
Draft steps fuse across members (P2c, 1.31-1.35x); the whole round is
graph-captured under a shape key with an always-on pointer guard (P3,
1.41x). N-invariance is a shipped bitwise gate (tools/ninv_test.cu).

So phase 3 is NOT "build batched kernels." It is a MEMORY-OWNERSHIP
redesign: collapse N engine stacks into one engine + N sequence states, so
the already-batched round stops paying N x (role ring + graph zoo + scratch).

What genuinely assumes one sequence today:
- KV addressing: `kcache[ci]`/`vcache[ci]` are one contiguous max_ctx array
  per engine; `attn_decode3`/`kv_store3` take ONE kc/vc/max_ctx
  (`spec3.cuh:42-54, 87-89`).
- The GDN role rotation ring: `(role + perm) % W_MAX` over 13 full role
  sets per engine (`engine.cuh:955-980`), rotated by a per-engine host perm.
- The tails: `spec_verify_tail_sampled` still takes flat logits2 +
  d_draft* by name (`engine.cuh:1916-1930`); the greedy tail already went
  through the view in P1.
- The graph key: includes the ordered Engine* tuple + exact width vector
  (`conductor.h:1569-1576`) BECAUSE capture bakes engine pointers.

## 3. Target architecture

One `Engine` owns weights, prefill scratch, the graph zoo, and K
`SeqState`s:

```
struct SeqState {           // per admitted sequence
    KvBundle kv;            // paged: block-table row + reserved/mapped/valid extents
    float* S[47]; float* ring[47];        // committed GDN state (role 0)
    float* S_snap[47]; float* ring_snap[47];
    int d_pos, d_step, d_P, d_gen...;     // per-seq scalars (device)
    DraftState draft;       // margins, chain positions
    PfxState pfx;           // pinned staging + ckpt ring (host)
};
```

The four design moves, in dependency order:

**(a) GDN record-then-Fold (kills the rotation ring).** ninfer's shape:
verify RECORDS per-layer conv/K/V/gate rows into a fixed arena indexed by
batch row; after the host picks the accepted prefix, one Fold kernel commits
row b's state into lane b's committed S/ring. No per-speculative-position
role selector at all; the arena is [0,C) current + [C,2C) checkpoint. For
q27: replaces 13 role sets/engine (2.04 GB) with ~2 sets/lane (~0.31
GB/seq). The Fold is byte moves over accepted rows, not new arithmetic --
bitwise-preservable, gated by the ninv TWIN legs (the delta kernels carry
MIRROR WARNINGS, `blocks.cu:145-147, 211-212`; the `_t` table twins already
prove device-side role indirection). THIS STAGE PAYS UNDER THE CURRENT
ARCHITECTURE: engine-per-slot drops ~5.6 to ~4.1 GB/slot -- a third slot
fits today's 15.76 GB post-weight budget before any other work happens.

**(b) Paged KV + per-row lengths.** Replace per-engine contiguous KV with
one pool at page size 64 and per-sequence block-table rows carrying
reserved/mapped/valid extents (ninfer's three-extent discipline lets
speculative rounds write past the committed frontier and roll back by
moving a scalar). `attn_decode3`/`kv_store3` grow a per-row (base table,
length) instead of scalar kc/vc/max_ctx. Numerics identical -- the
indirection changes addressing, not arithmetic. Admission reserves full
prompt+output entitlement up front, FIFO waits instead of eviction.

**(c) Sequence-indexed round + unified tails.** The union view maps lanes
to (seq, lane) instead of (engine, lane); `spec_verify_tail_sampled` goes
through the view like the greedy tail; commit chains (`finish_round`,
outcome, h_next) become per-seq array walks. Attention decode stays
per-sequence BY DESIGN -- cross-sequence attention has zero shared reads
(design already ruled: 2026-07-14 design doc :44-47); the forked side
streams inside the captured round already cover it. GDN conv/delta gain a
sequence grid dim (48 blocks -> 48C; the one real occupancy win, and the
`_t` twins are the template).

**(d) Batch-shape graph keys.** With (a)+(b) removing baked engine pointers
(state reached via tables + selectors), the graph key collapses from
(Engine* tuple, exact width vector) -- a C! x W^C space that thrashes the
LRU -- to (exact B, width class, kv_kind), ninfer's discipline: never key
on request identity; membership changes select a different pre-instantiated
exec. Exact-B captures for B=1..C; B=1 stays the solo path, byte-identical.

## 4. The bar, arithmetically

Per-sequence marginal (16K ctx, fp8 KV): 0.602 KV + 0.157 committed GDN +
0.150 snapshot + ~0.31 fold arena + ~0.05 scratch/logits growth =
**~0.67 GB + KV** against the bar's "under 2 GB + KV" -- met with 3x margin.

C=4 @ 16K each: weights 15.46 + process context 0.89 + prefill scratch 0.84
+ batch-aware zoo ~2.0 (conservative: exact-B x width-class alphabet) +
conductor LRU 0.5 + misc 0.15 = ~19.8 GB shared; + 4 x 1.28 GB = **~25.0 GB
total. C=4 fits a 32 GB card with ~7 GB of headroom** -- enough that C=4 @
32K ctx (KV 1.14/seq) also fits (~27.6 GB). C=8 @ 16K lands ~30 GB:
memory-feasible but LANE-bound, not memory-bound -- W_PLUMB=16 gives 8
members a floor width of exactly 2 (`conductor.h:276-283`), degenerating
spec depth; raising W_PLUMB is an fdmma tile redesign, out of scope. C=8 is
therefore a stretch goal contingent on the lane ceiling, exactly as ninfer's
5.67x rides batch width q27 does not have. State the ladder target as C=4.

Implementation bar (unchanged from the plan): aggregate t/s ladder at
C=1/2/4(/8) vs the 2026-08-14 baseline (143.0 sampled C=1, 1.28x at 2),
single-stream regression <= 3% -- structurally protected: ConductorCore
k==1 falls through to the solo decode_step byte-for-byte, and stage (a)'s
Fold is gated bitwise before anything else lands.

## 5. Risks and their gates

- **Determinism contract.** Untrimmed lanes must stay bitwise vs solo; the
  union gemm_min family policy (`conductor.h:448-453`) must be re-derived
  for seq-indexed views. Gate: ninv TWIN legs + the canonical family.
- **Fold vs rotation numerics.** Fold must be pure state movement. Any
  reordering of the delta-rule arithmetic is a different format -- gate at
  bitwise, no tolerance class (P3 lesson, batch-p3-capture.md:80-88).
- **Paging read pattern.** Block-table indirection per 64-token page adds a
  dependent load to attention streaming; ninfer eats this cost at higher
  absolute t/s, but measure fd2 depth curves before/after on the 61K rig.
- **Host D2H gating stays outside capture** (the ~19 ms/round pageable-D2H
  absorption and the draft-phase host sync are non-capturable boundaries --
  batch-p3-capture.md:32-53).
- **Pinned host growth.** Ckpt rings (up to ~2.4 GB pinned/sequence at 16
  checkpoints) move per-sequence; at C=8 that is ~19 GB of pinned host --
  fine on 128 GB, but make ckpt_slots a per-seq budget knob.
- **Prefix-cache compat hash** gains the pool geometry; snapshot/restore
  paths must speak block tables (staging already per-slot pinned).

## 6. Immediate, independent of the redesign

1. **Config bug:** explicit `--ctx N --slots M` leaves slots 1+ at
   slot1_ctx=32768 (`server.cu:288` vs `:666,719`). Propagate explicit ctx.
2. **Estimator honesty:** charge kEngBase once per process, drop the
   co-residency fudge to the measured ~1.0 GB, size KV at 17 pairs + MTP
   explicitly. The 2026-08-14 ladder's admission would have read ~5.6-6
   GB/slot and the published "8.2 GB fixed" needs the BUILDLOG-corrected
   footnote (done in the phase-3 entry).

## 7. Staging and effort

- M0: §6 fixes + estimator recalibration. Hours. Ships alone.
- M1: GDN record-then-Fold. The hard, load-bearing stage (two ~80-line
  kernels + arena plumbing + bitwise gates). Pays at engine-per-slot
  immediately (~-1.5 GB/slot -> 3 slots today). Days-to-week.
- M2: paged KV pool + block tables + per-row lengths in attn/kv-store.
  Days-to-week; numerics-neutral, perf-gated on the depth rig.
- M3: SeqState extraction + seq-indexed union views + unified tails +
  shared zoo. The big refactor; the kernels are ready. 1-2 weeks.
- M4: batch-shape graph keys + exact-B captures. Days, after M3.
- M5 (optional): GDN seq-dim kernels (occupancy), attention stays per-seq.

Each stage lands with its own measurement against the ladder baseline; M1
and M2 are independently useful and independently revertible.

## Verdict

**GO.** The bar (C=4 in 32 GB, marginal < 2 GB + KV) is met on the audit's
numbers with 3x margin on the marginal and ~7 GB of headroom at C=4/16K --
and the plan's premise was too pessimistic: the batched compute half
already shipped in P0-P3, the "8.2 GB" was an estimator artifact of a
config bug plus double-charged process state, and the single hardest true
redesign item (the GDN rotation ring) has a proven-shape answer in ninfer's
record-then-Fold plus an in-tree indirection precedent (the `_t` twins).
The implementation is NOT greenlit as one unit: M1 must clear its bitwise
gates and show its -1.5 GB/slot before M2+ proceeds, per the ground rule
that every stage closes with a measurement.
