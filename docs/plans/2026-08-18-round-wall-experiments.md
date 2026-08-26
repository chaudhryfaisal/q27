# Round-wall experiments: the post-errata sequence

Written at the end of 2026-08-18 for a fresh session to execute. Everything
here descends from BUILDLOG entries (c)-(j) of that date; entry (j) is the
errata that reordered this list, and nothing below should be re-litigated
against the PRE-errata entries.

**Baseline (C=8, q4s, `--slots 8 --ctx 16384`, `Q27_KV=fp8 Q27_BATCH=1
Q27_PMIN=0.5`):** round 27.2 ms = phv 21.4 + fdraft 5.7 + rest 0.2;
tok/round 1.70; ~493 t/s aggregate. Day's shipped path: T4 draft ceiling
(96c90d9) + batched nucleus (b15fc1b) + GDN mix fusion (ffd3f8a), all
canonical-gated. Production traffic at C=3-6 additionally pays a measured
**2.9-7.8 ms/round of graph-capture tax** that uniform C=8 never sees.

The gap being chased: ninfer nvfp4 does 2.38 tok/round at an 18.9 ms round.
Quote the aggregate gap as **1.91x** (790/412.9); the per-stream 2.31x used a
stale denominator (errata E6). Their per-member slope (0.54 ms vs our ~1.4)
is structural and NOT reachable by this plan; the compounded honest target if
E1-E4 below land is **~640-710 t/s at C=8** plus a larger relative win at
C=3-6 from E2 alone.

## Instrument discipline (read before running anything)

The errata exist because instruments lied all day. Non-negotiable rules:

- **Judge wall on round/phv ms from `[req]` lines; judge tokens on
  tok/round; NEVER on ladder aggregate** (sampled acceptance swings +-10%
  between server instances). Discard the first ladder point after boot
  (~5% cold). Count requests per rung before slicing `[req]` lines.
- **`[req] rounds=` is the SHARED fused-round count.** Per-round means over a
  rung = sum(field)/sum(rounds) over that rung's requests only.
- **phv absorbs 20-28 ms of capture+instantiate on every gcache MISS round**
  (mean ~23; the old "2.4 ms" comment was 10x stale, now fixed). Before
  attributing ANY phv delta, sum the `[gcache] ... cap+inst=` walls per leg
  and subtract. This single omission produced the wrong T1 width verdict.
- **A/B hygiene:** rebuild every comparison binary at HEAD; design a null
  rung into the A/B (a rung where the knob provably cannot matter) and
  require it to read ~0.
- **nsys:** `--cuda-graph-trace=node`, delete stale `.sqlite`, SIGINT the
  server and wait for finalize, verify report mtime. Identical totals across
  different windows = stuck instrument. share x wall arithmetic is valid
  ONLY for serial-on-cstm work; side-stream exposure = union across members.
- **The canonical digest cannot judge depth/draft policy changes** -- it stays
  bitwise-exact while draft-side state degrades (verify is target-argmax).
  Depth experiments are judged on tok/round.

## E0 -- Fix the determinism anchors (BLOCKS EVERYTHING)

Two anchors are broken and every gate below leans on byte identity:

1. **Greedy f64e7c02 diverges to 8196e65e**, reproduced twice (08-16 w16 CLI,
   08-18 q27 CLI), both times on the FIRST run after a fresh build, ~1-in-13
   overall; 10/10 clean on repeats. The old "one divergent md5 is a rerun"
   rule is dead -- this reproduces. Hypothesis to test first: rebuild-then-
   single-run cycles (`make build/q27 && run once`, x20) vs 20 runs of a warm
   binary. If first-run-only, suspect JIT/module-load or uninitialized device
   state; `compute-sanitizer --tool initcheck` on the first run is the next
   step. Bar: either a mechanism + fix, or a documented pre-run warmup rule
   that drives the observed rate to 0/20.
2. **Sampled-seed anchor 900031e9 does not reproduce** -- a pre-08-18 binary
   returns 227a6b08 deterministically (recipe: `--tokens "760,6511,314,9338,369"
   -n 64 --temp 0.7 --top-p 0.95 --seed 42`, plus `--ctx 2048 --spec` per the
   recorded-anchor rule). Determine whether the recorded anchor's flags are
   wrong (the 07-13 gotcha: sampled anchors REQUIRE `--ctx 2048 --spec`) or
   the trajectory drifted across a legitimate change; then re-derive and
   publish the anchor with its exact command line.

Cost: half a day. Nothing else in this plan runs before E0 unless it is
explicitly judged on tok/round rather than byte identity.

## E1 -- Warm-cache width control (settles T1/08-16, cheap)

**Question.** Steady-state width cost at C=4 measured ~+0.75 ms/round (~3%)
against +2.3% tokens once capture stalls are subtracted -- a wash, not the
recorded -13.6%. Is width actually neutral-to-positive warm?

**Method.** w12 vs w16 binaries (BOTH rebuilt at HEAD), C=4, TWO ladder
passes per leg against the same server instance; score ONLY the second pass
(shape cache warm). Also compute the cold-pass phv minus summed cap+inst as a
cross-check -- the two methods must agree within ~0.5 ms/round or the
accounting is wrong. Null rung: C=8 must read ~0 delta (trim floors to 2
regardless of cap).

**Bars.** Warm w16 vs warm w12 at C=4: if tok/round gain >= wall cost (both
in %), the 08-16 "do NOT raise W_MAX" verdict is RETRACTED for warm-cache
serving and E5 (width policy) is unblocked. If wall cost still dominates by
>2x, the verdicts stand with the scope limit removed, and E5 dies. Either
way, amend BUILDLOG (g)/T1 with the warm number.

**Cost.** Two server boots, four ladder passes. An hour.

## E2 -- Kill the capture tax (pre-capture / graph-key slimming)

**Question.** C=3-6 production rounds pay 2.9-7.8 ms/round in capture stalls
today (35% miss at C=4 even with zero evictions -- shape-space exhaustion,
not LRU). Uniform C=8 has one shape and is immune, which is why the ladder
never showed it.

**Method, in escalating order of ambition:**
1. Ship `Q27_BATCH_GRAPH_CAP` default raise (64 -> auto-shrunk 512-class;
   the server already headroom-shrinks). Free, already measured +6.9% at
   C=4-w16.
2. Slim the graph key: the exec is keyed on the full granted-width vector;
   check which baked pointers actually differ across shapes with equal
   (k, sorted-gw, sfx, smp, kvk) -- if lane pointers can be indirected
   through a device table (the M2a block-table pattern), the shape space
   collapses from product-of-grants to ~175 canonical shapes.
3. Pre-capture the canonical shapes at startup (SGLang's pointer-swap
   SpecRuntimeState pattern: build every candidate up front, never capture
   mid-stream). Startup cost at ~23 ms/capture x 175 ~= 4 s, acceptable
   against server boot ~25 s; VRAM ~8 MB/exec is the real budget -- measure
   against the 4.06 GB free at ready.

**Bars.** C=4 and C=6 round-wall (capture-subtracted phv) unchanged, but
RAW phv drops by >= 2 ms/round at C=4; zero `cap+inst` events after warmup
in a 5-minute mixed-C soak. Null rung: C=8 unchanged. Canonical digest EXACT
(this must be pure choreography).

**Cost.** Step 1 free; step 2-3 a day-plus of conductor work. Do step 1
immediately regardless.

## E3 -- Depth-1 + draft-into-round-graph (fdraft 5.7 -> ~3)

**Question.** Post-T4 the ceiling is 2 steps (the "W draft ROWS must exist"
top-up invariant blocked 1); phs/round is ~1.8 and fdraft 5.7 ms. At depth 1
the margin gate buys ~0.06 ms/round -- vestigial -- and the whole draft phase
becomes unconditional, hence graph-capturable.

**Method.**
1. Derive the row invariant: read `draft_floor_topup` / `draft_and_gate` /
   the width-2 verify's lane reads / fold consumption (rec arena row count vs
   accepted n) and determine what a width-2 verify actually requires when
   only ONE draft step ran. If depth-1 is safe, A/B it behind an env flag
   first (`Q27_DRAFT_CEIL_MINUS1=1` style): fdraft should drop to ~2.9,
   round to ~24.4. **Judged on tok/round, not the digest** (see discipline).
   Kill: tok/round drops > 3% at C=8 (the verified position should be
   unaffected -- any drop means the invariant derivation was wrong).
2. If (1) lands, capture the single fused draft step + margin-free gating
   into the round graph (graph_round already hoists the draft_done waits;
   the per-step host sync exists only to read margins that no longer gate
   anything at C>=6). Watch dctl: pinning depth feeds degenerate
   observations into the adaptive ladder -- freeze dctl updates while the
   concurrency ceiling binds (the SGLang `if current_steps > 0` guard
   pattern), or C-drop behavior degrades.

**Bars.** C=8 round <= 24.5 ms with tok/round within 1% of baseline ->
~550-575 t/s. C=1/C=2 byte-identical (solo path untouched; ceiling
non-binding below k=5 -- null rungs). B8 assert must survive.

**Cost.** Derivation half a day; the graph capture change ~a day.

## E4 -- Cross-member batched mix, GDN AND attention (~4 ms exposed)

**Question.** cstm idles through the entire mix phase (errata E1: 0.000 ms
busy inside mix intervals); the exposed side-phase union is ~2 ms GDN +
~2 ms attention per round, paid as max-over-members at every one of 65
per-layer fork/joins. One launch per layer covering all members removes the
fork/join and fills the machine (delta: 8x48-block launches -> one
384-block launch).

**Method.** The nucleus_multi pattern one level up. Per layer, a conductor-
level batched launcher takes per-member pointer structs (the fused 3-kernel
chain from ffd3f8a is the substrate): `k_gdn_convnorm_b` and
`k_gdn_delta_all_b` with blockIdx.z = member; attention needs the per-lane
block-table indirection (`void* const* kt[W_PLUMB]` per member) flagged in
the 08-18 recon. Per-member vw rides in the struct; blocks branch on their
member's vw only (block-uniform). Bitwise per member by the same argument as
nucleus_multi/gdn_fuse_eq -- each block does exactly what its per-member
launch did; gate with a gdn_fuse_eq-style harness extended across members
with HETEROGENEOUS vw vectors, plus ninv, plus canonical.

Keep the per-member path for k<=2 (GEMV regime, bitwise contract) and solo.
Graph capture moves the mix nodes from side streams to cstm -- the union-view
guard fields do not change, but capture topology does: expect full re-capture
per shape, which is why E2 lands FIRST.

**Bars.** C=8 phv (capture-subtracted) drops >= 1.5 ms -> ~2.5 ms is the
ceiling estimate; kill if < 0.8 ms (the fork/join overhead estimate was
wrong). C=4 should gain MORE (less member-overlap today). Canonical EXACT,
gdn_fuse_eq-multi zero diffs, ninv pass.

**Cost.** The biggest item here -- multiple days, conductor + engine + two
kernels. Do not start before E2; do not start the attention half before the
GDN half A/Bs positive.

## E5 -- Width/token policy (gated on E1 warm result + E2)

**Question.** tok/round 1.70 vs ninfer's 2.38 is the largest remaining
factor. Uniform grant 3 bounds at ~2.22 tok/round (selection-biased upper
bound -- ~39% of lanes only want 2); realistic ~1.95-2.1.

**Method.** Only if E1 retracts the width NO-GO warm and E2 has killed the
capture tax: raise W_MAX toward W_PLUMB in the w16 build, grant uniform-3 at
C=8 via trim policy (uniform shapes keep the graph space small -- SGLang
deliberately pads to uniform for exactly this reason), measure vgemm column
cost at the POST-nucleus coefficient (~0.11-0.12 ms/col, not the stale
0.39). Note C=8 uniform-3 means union 24 > W_PLUMB=16: this requires the
W_PLUMB raise (p[16] structs, `6*W<=96` fdmma assert, vgemm NT tile, record
arena) -- price that refactor only after E1/E2/E4 all land, since E4 removes
the mix-side width cost that killed it before.

**Bars.** tok/round >= 1.95 at C=8 with round wall growth <= 2.5 ms ->
~640-710 compounded. Kill: tok/round < 1.85 or wall growth > 4 ms.

## E6 -- Kernel-local, anytime fillers

**CLOSED 2026-08-19 (n)+(o). cp.async NO-GO; shipped 16-byte staging and a
scale-load hoist for -4.2% cumulative, bit-identical. Also NO-GO on measurement:
depth-2 register prefetch, the z policy, fusing k_reduce_z, and ldmatrix.**
The "84% -> 94-96% SOL" premise was a unit mismatch: the two figures used
different SOL denominators (round_weight_cost's 1453 constant vs
microbench_mxf4's measured ~1690). Decomposed against a same-grid floor
(`tools/vgemm_e6`), the staging path is 0.25 ms off the floor and the exposed
cost is the MMA phase (0.72 ms) -- which is exactly where cp.async's Q4 leg
would have to move the nibble unpack. Shipped instead: uint4 staging,
-0.30 ms +- 0.01, bit-identical over all 401 weights at T=2/5/12/16.

- ~~**cp.async in k_vgemm**: 84% -> 94-96% SOL measured on identical shapes
  (T2's surviving fp4-arm yield); ~1.2 ms of the 10.3 ms sweep. Bitwise
  output (memory pipeline only), canonical-gated. ~Half a day.~~
- **PDL (programmatic dependent launch) on the serial cstm chains**: the
  13 us/layer inter-kernel gaps; ~0.5 ms, partially overlaps E4's win.

## What this plan refuses to do

- **Re-open fp4** in any form (BUILDLOG (c): byte-bound regime, 4.50 vs
  4.25 bpw; unaffected by all errata).
- **Port DSpark/DFlash** (e305621: ties the in-checkpoint MTP head for 2.5x
  the draft compute; op-point caveat recorded, C>=4 economics unchanged).
- **bf16 GDN state**: demoted -- ninfer keeps S fp32, so their slope is not
  bought there; a quality-gated format change is not worth it before the
  structural work above, and maybe not after.
- **Chase 18.9 ms**: ninfer's remaining edge is per-member slope all the way
  down; the honest compounded ceiling here is ~640-710. q27's actual
  strategic advantage (4.10x agentic prefill via reuse) was never in play
  today and needs no defense.

## Order

E0 -> E2.1 (free cap raise) -> E1 (hour) -> E3.1 (env-flag A/B) ->
E2.2-3 -> E3.2 -> E4 -> E5. E6 fills gaps. Each step amends its BUILDLOG
entry same-day, with the scope limit written into every verdict.
