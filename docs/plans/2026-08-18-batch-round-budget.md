# The C=8 round budget: where q27's 2.31x actually goes

T2 closed the fp4 line by measuring it (BUILDLOG 2026-08-18 (c)). Closing it
also removed the explanation everyone was using for ninfer's 790 t/s, so this
plan picks up the real question: **q27's C=8 round is 32.41 ms and only ~10.9 ms
of it is the shared weight sweep.** Everything here is about the other 21.5 ms
and about the tokens q27 declines to draft.

## What is already measured, and not up for re-litigation

Per-stream at C=8, nvfp4 against q4s is **2.31x**, and it factors exactly:

| term | ratio | source |
|---|--:|---|
| tokens per round | 2.3835 / 1.777 = **1.34x** | ninfer reqlogs; q27 `[req] dec=8192 rounds=4611` |
| round wall | 32.41 / 18.90 = **1.71x** | same |

Neither term is a weight-format question:

- **fp4 arithmetic is ruled out.** 9.5% of dense fp4 peak at M=16; nvfp4 is the
  larger format at 4.50 bpw against 4.25. T2.
- **Speculation depth is ruled out as the ninfer-internal explanation.** nvfp4
  and nint hold `draft_window=3` and ~0.46 acceptance at every rung and differ
  by 1.025x on tokens per round. Their 2.30x is all round wall, and it traces to
  `text_policy()` handing `AllowA4` only to `QType::NVFP4` and `A16Only` to
  everything else -- their int tier is locked out of its own batch kernel.
- **q27 does not have ninfer's int-tier problem.** `k_vgemm` runs at 84% of
  streaming SOL at M=16. There is no idle machine to reclaim.

So q27's 1.34x and 1.71x are two independent q27 problems. This plan measures
which is worth fixing before anything is built.

## Instrument discipline

`Q27_PHASE_STATS=1` already splits the per-request round wall into
`phd` (draft ms), `phv` (verify ms) and `phs` (draft steps) -- `src/server.cu:1189`.
`Q27_BATCH_DBG=1` gives the per-round `[bat] k=.. cap=.. want->granted` line.
Both are enough to answer T1 and T2 below with no new code and no profiler.
Take widths from `[bat]`, never from the nominal config: the C-sweep's "88% of
lanes at floor-2 from C=6" measured 100% at C=8.

---

## T1 -- Is the width trim actually costing tokens, and is width free?

> **RAN 2026-08-18. NEGATIVE -- width is not free, and the 1.34x is not
> available this way.** Confirms the 2026-08-16 C-sweep verdict (which this
> section originally failed to cite) and adds the mechanism plus one correction
> to it. At C=4, the rung where the cap actually binds:
>
> | leg | aggregate | tok/round | round wall | phv/round |
> |---|--:|--:|--:|--:|
> | w12 (shipped) | **287.8** | 2.1157 | 28.504 ms | 22.795 ms |
> | w16, gcache cap 64 | 232.6 (-19.2%) | 2.1501 (+1.6%) | 36.115 ms (+26.7%) | 30.478 ms (+33.7%) |
> | w16, gcache cap 463 | 248.7 (-13.6%) | -- | -- | -- |
>
> **Widening bought 1.6% more tokens and cost 13.6% of round wall.** The
> marginal union lane does not pay for itself, exactly as 2026-08-16 concluded
> at C=6/C=7 (-6.5%/-4.1%); the penalty is simply steeper at C=4 where the
> granted spread is widest.
>
> **The correction to that verdict: about a third of the measured penalty was a
> fixable cache artifact, not width.** Heterogeneous grants explode the
> fused-verify graph-variant space -- w12 produced 97 distinct `gw` shapes and
> ran 1011 hits / 83 misses / 19 evictions; w16 produced 192 against a 64-entry
> cache and ran 804 / 275 / **211 evictions**. Raising `Q27_BATCH_GRAPH_CAP` to
> 463 zeroed the evictions and recovered 232.6 -> 248.7. The remaining 13.6% is
> real per-lane cost. Note that even with zero evictions the hit rate is only
> 64.5%: with a combinatorial shape space and ~470 rounds, a third of rounds
> still see a shape for the first time. **Any future policy that widens grants
> heterogeneously has to price the graph-variant blowup, and the shipped 64-entry
> cap is too small the moment grants stop being uniform.**
>
> **C=8 was the control and it behaved exactly as predicted: 418.8 (w12) vs
> 419.6 (w16), a 0.2% null.** The cap cannot bind there -- `trim_widths` never
> cuts a lane at floor 2, so eight floored lanes overshoot to `W_PLUMB` either
> way. This is the instrument working: a knob that provably changes nothing at
> C=8 changed nothing at C=8.
>
> **Trap, logged because it nearly produced a published wrong answer:** the
> in-tree `build/q27-server-w16` was 2 days and 6 commits stale, and with it the
> C=8 control read **218.9 against w12's 417.6** -- a fake 48% regression at the
> rung where the policy is provably identical. The null at C=8 is what exposed
> it. **Rebuild every comparison binary at HEAD before believing an A/B**, and
> keep a rung in the design where the knob must do nothing.
>
> **Consequence for the plan.** T3 (raise W_PLUMB) is DEAD -- it was gated on
> width being nearly free and width is negative. That also collapses this plan
> onto one target: if q27 cannot afford wider drafts *because* its per-lane
> verify cost is steep, and its round wall also grows 1.75-2.23 ms/member
> against ninfer's 0.54-0.64, then the tokens/round term and the round-wall term
> are the SAME defect measured two ways. T2 below is the whole plan now.

**The question.** At C=8 every lane is granted width 2 (99.79% of 4806 lanes,
measured 2026-08-18) against wanted widths spread 2/3/4/5, mean ~3.3. q27 is
throttling speculation exactly where the shared weight sweep should make extra
lanes cheapest -- `k_vgemm` was built to be flat in W, which is the whole
argument for the trim being the wrong policy at batch.

**Why C=8 cannot be the test.** At C=8 the union is pinned at 16 by
construction: `trim_widths` never trims a lane already at floor 2
(`src/conductor.h:61`), so eight floored lanes overshoot the `W_MAX=12` cap and
stop at exactly `W_PLUMB`. Raising `Q27_W_MAX` changes nothing there. **C=8 is
W_PLUMB-bound, not W_MAX-bound**, and W_PLUMB is not a knob -- `static_assert(W_PLUMB <= 16)`
guards lane-pointer structs declared `p[16]` plus a `6*W <= 96` assert in
`k_attn_fdmma`.

**C=4 is the test.** Four members wanting ~3.3 each sum to ~13, over the cap of
12, so C=4 IS W_MAX-bound and the cap is a compile knob with a target already in
tree: `build/q27-server-w16` is `-DQ27_W_MAX=16` (`Makefile:202-206`). At
W_MAX=16 a C=4 union can reach 16 instead of 12, which is the same widening a
W_PLUMB raise would buy at C=8 -- bought for a rebuild instead of a refactor.

```bash
# A: shipped cap. B: cap raised. Same artifact, same everything else.
for BIN in q27-server q27-server-w16; do
  Q27_KV=fp8 Q27_BATCH=1 Q27_PMIN=0.5 Q27_BATCH_DBG=1 Q27_PHASE_STATS=1 \
    build/$BIN /mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp-q4s.q27 \
               /mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.tok \
               --slots 8 --ctx 16384 --port 8081 --host 127.0.0.1 &
  python3 bench/ladder/ladder.py <log> http://127.0.0.1:8081 4 1024
done
```

**Pre-declared bar.** Report tokens/round and round wall separately -- the
aggregate hides which one moved.

| outcome | reading | consequence |
|---|---|---|
| tok/round up >= 15%, round wall up <= 5% | width is nearly free at batch | the trim is the wrong policy; a W_PLUMB raise is justified and T3 scopes it |
| tok/round up, round wall up proportionally | per-lane work dominates | widening is not free; the trim is roughly right and the whole 1.34x is unavailable |
| tok/round flat | the cap was not binding at C=4 | re-read `[bat]`; the premise is wrong, stop |

**Cost.** Two ladder points, no code. An hour.

---

## T2 -- What is the 21.5 ms made of?

> **RAN 2026-08-18. PASSES the attribution bar at 98.2-99.3%, and the answer is
> the DRAFT PHASE, which the method as written could not have seen.**
>
> **The instrument had to be fixed first.** `phd` cannot price the draft phase
> at batch, by construction and by its own in-tree comment: the phase runs
> host-synced per step inside `Conductor::draft_widths`, BEFORE `ev_round_start`
> is recorded, so `phd` brackets only the unsynced tail. It reads 0.006 ms/round
> at C=8 while ~2.86 draft steps per member actually launch. The plan's
> `round_ms - phv - phd` model would have dumped the entire draft phase into an
> unexplained remainder and called it host gap. Fixed by bracketing
> `draft_widths()` with a host wall clock (`phfd`/`GenStats::fdraft_ms`) --
> exact precisely BECAUSE the phase host-syncs every step, and inert to
> arithmetic (a clock read and a counter).
>
> | rung | tok/rnd | round | fdraft | phd | phv | rest | accounted |
> |---|--:|--:|--:|--:|--:|--:|--:|
> | C=1 | 2.3273 | 16.641 | -- | -- | -- | -- | n/a |
> | C=2 | 2.1951 | 22.204 | 3.903 | 0.051 | 17.847 | 0.403 | 98.2% |
> | C=4 | 2.0419 | 28.488 | 5.362 | 0.002 | 22.888 | 0.236 | 99.2% |
> | C=8 | 1.7042 | 32.144 | **8.182** | 0.006 | 23.734 | 0.222 | **99.3%** |
>
> C=1 is unattributed by construction, not by failure: `ConductorCore::step`
> takes `if (k == 1) solo_round(...)` and never enters `draft_widths`/
> `fused_round`, so all three fused stamps are 0 there. Any future round-budget
> work at C=1 needs the solo path stamped separately.
>
> **The C=8 round is 73.8% fused verify and 25.5% DRAFT.** Per added member
> (C=2 -> C=8, 6 members): round **+1.657 ms**, of which phv **+0.981** (59%)
> and fdraft **+0.713** (43%). The host gap the plan expected to find is
> 0.222 ms and shrinking -- it is not the problem.
>
> **The finding: q27 drafts ~2.86 tokens per member per round and verifies 1.**
> `phs/round` is 2.86 at every fused rung while `trim_widths` grants width 2, and
> width W verifies W-1 drafted positions. **~65% of every drafted token is
> computed and thrown away**, and because the phase is host-synced per step it
> scales linearly with members instead of amortizing.
>
> **This closes the loop with T1.** T1: q27 cannot afford to USE more draft depth
> (+1.6% tokens for -13.6% wall). T2: it is PAYING for depth it does not use, to
> the tune of a quarter of the round. The two results are the same policy error
> seen from both ends -- the gate ladder picks depth per member without knowing
> `trim_widths` is about to crush the grant to floor-2.
>
> **Upper bound on the fix**, stated as a bound because it assumes zero fixed
> per-member cost in the phase: 8.182 ms / (8 members x 2.86 steps) = 0.357 ms
> per draft step. Drafting 1 step instead of 2.86 costs **zero accepted tokens**
> (only 1 drafted position is verified at width 2 regardless), and would take
> fdraft to ~2.86 ms, the round to ~26.8 ms, and the C=8 aggregate from 417 to
> **~500 t/s**. Unlike T1's widening, this trades nothing away -- it is waste
> removal. That makes T4 below the successor to this plan.

### Original method (superseded by the run above)

**The question.** 32.41 ms round, ~10.9 ms shared weight sweep. If the
remainder is per-round overhead, widening and fusing pay; if it is per-member
work, only the per-member work pays.

**Method.** `Q27_PHASE_STATS=1` at C=1/2/4/8 gives `phd`/`phv`/`phs` per
request. Fit `phv = a + b*k + c*M` over the fused rungs to separate the
per-member term from the per-union-column term, and take `round_ms - phv - phd`
as the unattributed remainder. No profiler, no new code.

The hypothesis to test, from reading the fused round: the per-member term is GDN
recurrent state (read-modify-written by `delta_step`, re-read by
`gdn_delta_chunk3`, folded again on commit) plus per-lane `fd2` KV, and the
per-round term is an ungraphed draft phase -- ninfer captures verify plus all
draft steps into **one** `cudaGraphLaunch`, q27 graphs only the verify.

**Pre-declared bar.** This is an attribution, not a gate: it passes if the three
terms account for >= 85% of the measured round wall at all four rungs. Below
that the model is wrong and the next step is a real profile
(`ncu`/`nsys` are not on PATH -- full path plus `sudo -n`, per the bench gotchas).

**Cost.** Four ladder points plus arithmetic. Half a day.

---

## T3 -- Raise W_PLUMB

> **DEAD, 2026-08-18.** Gated on T1 returning "width is nearly free". T1
> returned the opposite: +1.6% tokens for -13.6% wall at the rung where the cap
> binds, with C=8 a clean null. Raising `W_PLUMB` would buy q27 the right to
> grant widths it has now measured twice as not worth granting. The refactor it
> would have required -- `p[16]` lane-pointer structs, the `6*W <= 96` assert in
> `k_attn_fdmma`, `vgemm`'s `NT = W_PLUMB` tile, the per-width graph zoo, the
> record arena -- is not worth starting.
>
> Do not re-open this without first moving T2's per-member cost. If the marginal
> lane ever becomes cheap, width becomes worth revisiting; until then the
> ordering is fixed.

---

## What this plan refuses to do

Re-open fp4. T2 measured it at decode shapes and it lost on merits, and the
reason it lost -- decode is a byte count and nvfp4 is 5.9% more bytes -- does
not change with a wider union. If anything a wider union makes fp4 worse, since
the activation side grows while the weight side stays the same.

Also refuses to chase ninfer's 2.30x int-vs-fp4 delta as if it were a lesson for
q27. It is a lesson about `A16Only`, and q27's equivalent path is already at 84%
of SOL. The transferable finding from T2 is smaller and different: `k_vgemm`
stages through registers while a cp.async tile reached 94-96% of SOL on the same
shapes, so there is ~6% in q27's decode GEMM memory pipeline. That is worth
doing and is independent of everything above.

---

## T4 -- Draft to the grant you will actually get (opened by T2, 2026-08-18)

**The question.** The gate ladder picks each member's draft depth from its own
margin loop (`Q27_PMIN`, `gate_maxd`, dctl) with no knowledge that
`trim_widths` will crush the grant to floor-2 a moment later. At C=8 that costs
~65% of every drafted token and ~25% of the round.

**Method.** Feed the expected grant back into the draft ceiling: before the
margin loop, compute what `trim_widths` will grant given the current member set
(it is a pure function of `want[]`, `is_suffix[]`, `k`, `cap` and is already
CPU-testable in `tools/test_conductor.cpp`), and cap `md_used` at that. At C=8
with every lane floored this makes the ceiling 1 step. The prediction has to be
conservative -- granting MORE than predicted must stay legal, since a member
leaving mid-round changes k.

**Pre-declared bars.**
- C=8 aggregate strictly above 417 t/s, and `phfd`/round strictly below 8.182.
- **Accepted tokens per round must not fall.** At floor-2 only one drafted
  position is verified, so a correct implementation changes committed tokens by
  ZERO; any drop means the ceiling is cutting into verified positions and the
  change is wrong, not merely unprofitable.
- C=1 and k <= 2 untouched: canonical digest EXACT, solo bitwise contract
  intact. The trim never fires below the cap, so this must be a no-op there --
  and that is the control rung, per the T1 lesson.
- No regression at C=2/C=4 (the trim fires less there; a naive ceiling could
  over-cut).

**Risk to price first.** The margin loop may need steps it would now skip in
order to make its depth decision at all, and dctl's histograms feed the adaptive
ladder -- cutting steps changes what it learns. Check whether the ceiling can be
applied to LAUNCHES without changing the decision, or whether the decision
itself has to move.

**Cost.** A policy change inside `draft_widths` plus the existing CPU conductor
test. Days, not weeks, and T2 already built the instrument that scores it.
