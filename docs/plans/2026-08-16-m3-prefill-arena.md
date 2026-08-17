# M3a: the shared prefill arena (2026-08-16)

Status: SHIPPED. Gates in BUILDLOG (i).

## Why this instead of the spec's M3

The batched-decode spec (docs/plans/2026-08-15-batched-decode-spec.md §3)
frames M3 as "SeqState extraction + seq-indexed union views + unified tails
+ shared zoo", 1-2 weeks, to stop paying N x (role ring + graph zoo +
scratch) per slot. Two of those three terms are already gone: M1 deleted the
role rotation ring, M1b collapsed the per-engine zoo to one exec per family.
A HEAD allocation census (4-agent recon + hand verification) puts the
remaining recoverable per-slot memory at:

| class | GB/slot | to recover |
|---|--:|---|
| B: prefill chunk scratch | 0.75-0.84 | one process-wide arena, no capture work |
| C: capture-baked decode state (zoo 0.13, record arena 0.043, fd scratch, mask pool, logits2, ...) | ~0.25 | the full SeqState + shared-zoo refactor |
| A: irreducible per-seq (committed GDN 0.157 + snapshot 0.157 + tables) | ~0.31 | not recoverable under current prefix-cache semantics |

So ~77% of the recoverable memory is class B and needs none of the refactor,
while class C costs the entire risk surface (per-seq graph indirection,
GraphKey redesign, admission rework) for ~0.25 GB/slot on sm_120 -- and
cannot raise the k <= 8 member ceiling (conductor.h:276) that now bounds the
ladder anyway. M3a takes class B; the rest is deferred with a bar (below).

## What ships

One `q27::PrefillArena` (src/prefill_arena.h) holding the PF_T=1024 chunk
staging set (22 float buffers), the split-attention partials, the fp4
activation pair, the g64 FFN staging, and the WY/split-K panels. Engines
point their existing members at it, so no prefill call site changed. Serving
passes one arena; every CLI/tool site passes nullptr and keeps its own
buffers bit-for-bit.

**The hazard and how it is handled.** The 2026-07-04 R1b-prereq entry made WY
panels per-engine precisely because "two engines with chunks in flight would
race one panel set across streams". The GpuGate serializes issue, but engines
run on different streams and `GpuGate::Lease` documents an exemption for work
still in flight at release -- whose rationale ("all target per-engine
buffers") stops holding once scratch is shared. `claim(owner, stm)`
synchronizes the PREVIOUS owner's stream on an ownership change, and must be
called PER CHUNK: ownership ping-pongs at the gate handovers, so a claim taken
once at prefill entry records the last CLAIMER while another engine is the
last WRITER (the first cut of this change had that bug; review HIGH).

Evidence, stated honestly: the drain is defence in depth, NOT a fix for a
demonstrated corruption. Every yield handover is already drained by the server
hook, leaving only the end-of-prefill lease release as an unguarded window,
and the race gate cannot make it bite -- it passes with the drain disabled
(`Q27_PF_ARENA_NODRAIN=1`) as well as with it. Kept because it is nearly free
and restores the premise the Lease exemption rests on.

**Prefill-only by contract.** Decode work must never touch the arena: fused
rounds fork per-member work onto side streams with no gate boundary between
the racers. M1's commit Fold had borrowed the prefill `oT` for its <= W_MAX
output rows (sound while that buffer was per-engine); it now owns a small
per-engine `fold_o`. That decoupling is what makes the arena prefill-only.

## Measured (q4s, fp8 KV, 16K ctx, --slots 8, 5090)

- arena 0.75 GB once per process; per-slot marginal **1.48 -> 0.64 GB**
- **8 slots** (the conductor member ceiling) vs 7; pool **2.40 -> 5.80 GB**
- prod shape (1 slot, auto-ctx): a WASH, as it must be with one engine --
  pool 12.87 GB without the arena vs 12.78 with, full 262144 window either way
- ladder: C=8 **406-420 t/s**, a new peak over C=7's 385.4; C=2 unchanged
- `Q27_PF_ARENA=0` reproduces pre-M3a exactly (7 slots, 2.40 GB, 3.02 GB free)
- arena footprint is MEASURED (cudaMemGetInfo delta), 0.85 GB -- an earlier
  hand-sum said 0.75 and missed xqT, the WY panels and the SM-sized split-K
  partials; `kEngArena` (the auto-ctx constant, the one estimator that runs
  before the allocation) is 0.84e9 to match

## Gate design note

Concurrent-vs-solo FULL-TEXT identity is not an invariant of this engine:
under concurrency the trim floors granted width and attention switches between
fd2 and fdmma at ntok >= 4, which are tolerance-class twins, so a near-tie
argmax can fork the trajectory. A control boot with the arena disabled fails
such a comparison too. `prefill_race.py` therefore compares only the FIRST
token -- produced by the prefill epilogue before any batched round -- which
isolates what the arena actually touches.

## Deferred: M3e (full SeqState + shared zoo)

Un-defer only when one of these is true: (a) C > 8 is un-parked with a priced
lane plan (today the ceiling is k <= 8, so more slots cannot buy throughput);
(b) sm_86/89 multi-slot serving matters -- the zoo is 0.43 GB/slot there, 3x
the sm_120 prize; (c) a measured >= 1 GB/config win is written down first.

Dead ends, priced and not to be re-explored: moving decode-side buffers into
the arena (side-stream races); cudaGraphExec parameter patching (reintroduces
the host-resolved pointer class M1 deleted); snapshot dedup (changes
per-lineage prefix-cache semantics).
