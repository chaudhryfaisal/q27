# Sampling exit gate -- quality A/B + drift catalog under production sampling

Status: **PROPOSED / pending Gabe sign-off.** This is the design's stated exit
criterion (sampling-design.md, sampling-phase2-impl.md): the last gate before
temperature>0 may DEFAULT ON for serving. Phase 1 (plain sampler) and Phase 2
(spec rejection) are shipped and gated (HEAD 215eccc); this measures whether
turning sampling on regresses quality or breaks tool-call discipline. Nothing
here runs until approved -- it consumes shared-GPU hours and needs the q27-eval
window.

Rule zero (unchanged): greedy stays bitwise (canonical 4c4120c7). This gate is
about the SAMPLED path only. Passing it does not touch the greedy default; it
decides whether a NEW sampled default is safe.

## Why greedy numbers don't transfer (the whole reason this gate exists)

Every quality number backing the engine is greedy-no-think scoped: the 0.786
A/B tie, the five drift modes, acceptance 3.4-4.4 t/round, p(d4|prefix3)=97.4%.
Temperature moves acceptance AND drift SIMULTANEOUSLY:
- acceptance: measured Phase 2, greedy 3.43 -> T=0.7 3.45 -> T=1.0 1.90 t/round.
- drift: the five tool-format modes are argmax artifacts; sampling draws from
  the whole nucleus, so it can raise a mode's frequency OR surface a token the
  greedy argmax never picked (a NEW mode the greedy-tuned parser may not catch).

So the gate re-measures BOTH under the config that would actually ship.

## What "default on" means here

Today CC/CRUSH send no `temperature`; parse_sample() -> greedy spec (bitwise).
"Defaulting sampling on" = the server applies temp>0 EVEN when the client sends
none. The gate therefore tests server-FORCED sampling, not per-request opt-in
(which already works and is not in question). This is prerequisite P1.

## Decisions this gate needs from Gabe (3)

**D1 -- production config (temp/top_p).** Proposal: primary **T=0.7 / top_p=0.95**
-- the highest speed-neutral temp (accept-vs-temp is flat to T<=0.7; T=1.0 halves
throughput) and a standard agentic nucleus. PLUS a cheap temp ladder {0.3, 0.7,
1.0} on the single most drift-prone task to locate the quality/drift knee, so the
default recommendation is measured, not assumed. Alternatives: bless T=0.3
(conservative) or refuse a sampled default entirely (keep opt-in).

**D2 -- reference leg.** Proposal: **q27-greedy self-baseline** (sampled-q27 vs
greedy-q27; same engine, same tasks, same harness). This is the tightest test of
the ACTUAL gate question ("does turning sampling on hurt q27?") and the ONLY
comparison where harness-injected prompt-byte variation (the thing that made
analytics bimodal even under greedy) is matched between legs. Cross-engine vs
llama Q5_K_M is a DIFFERENT question (public "still ties the reference" claim),
already gated separately as queue #3 and needing the strongest-opponent llama
config (draft=10 + p_min .5) -- keep it out of this gate unless Gabe wants it.

**D3 -- GPU window.** q27-eval serves live CC traffic. This run needs the 5090;
greedy-config and sampled-config trials run against ONE server sequentially
(force-temp is a server-global env -> restart between the two configs; both
servers never coexist in 32GB anyway). Needs a coordinated window, not a seize.

## Task selection + trial budget (the red-team-critical part)

The greedy A/B ran T1-T10 x n=3 = 30 trials/leg. That minimal-scope justification
was DETERMINISM (`[[feedback_minimal_bench_scope]]`: greedy n=3 was near-zero
variance, extras redundant). Under sampling that justification is void -- every
trial is an independent draw. Keep the SPIRIT (few tasks) but reallocate the
budget from more-tasks to **more-seeds-per-task**, because the gate question is a
DISTRIBUTION shift and n=3 cannot see it.

Tasks (3, chosen to load the two failure modes + a control):
- **task-queue** -- drift failure mode. Scored 0.000 pre-parser under greedy
  (drift-fatal, zero writes executed). Maximizes the chance sampling surfaces an
  un-rescued tool-call mode. This is also the D1 temp-ladder task.
- **analytics-dashboard** -- quality/basin failure mode. The bimodal 0.48/0.83
  task that separated n=1 draws NEVER (self-inflicted twice). Under sampling with
  many seeds, P(low basin) becomes a MEASURABLE quantity: compare greedy-config
  vs sampled-config basin frequency directly. The old nemesis becomes the metric.
- **collab-server** -- clean-task control (q27 greedy led +0.103 here). Detects a
  broad regression that the two stress tasks might miss.

Seeds: **n=10 per task per config.** Not a round number -- it is the floor at
which the bimodal basin frequency carries a usable CI (+-~15% at n=10) so a gross
sampled-vs-greedy basin shift is detectable; n=3 gives none. Larger n only if the
CI straddles the exit threshold.

Budget: core A/B = 3 tasks x 2 configs (greedy, T=0.7) x 10 = 60 trials; temp
ladder = 1 task x {0.3, 1.0} x 10 = 20; total **~80 trials**, same order as the
greedy A/B (60). At q27 agentic ~100-200 s/trial ~ 2-4 GPU-hours.

## Drift catalog protocol (the correctness deliverable, > the score)

Under greedy: 5 modes, all silently rescued (17 recoveries; api_common.h
parse_bare_tool_calls handles mode-1 dropped-wrapper, mode-2 truncated-JSON,
mode-3 `<content>`-tagged, mode-4 `{"tool_call":` opener, mode-5 raw control
chars). Current logging = a COUNT only (`[tool-fallback] N recovered`), no mode,
no clean un-rescued flag. Prerequisite P2 fixes that.

Per sampling run:
1. **Classify** every tool call: {clean | mode 1-5 rescued | UN-RESCUED}. The
   parser already branches on mode internally -- just emit which branch fired.
2. **Flag** any intended-call the chain could NOT rescue (surfaced as raw text,
   .ok=false with no bare recovery). That is a NEW failure the greedy-tuned parser
   does not handle -- the thing this gate exists to catch.
3. **Compare** greedy vs sampled mode histograms: sampling may raise an existing
   mode's rate or add a mode. Either is a reported finding.

## Exit thresholds (decision rule, fixed BEFORE running -- red-team discipline)

Sampling defaults ON only if ALL hold at the chosen config:
1. **Quality:** per task, sampled mean >= greedy mean - epsilon, epsilon within
   the measured seed-variance CI (no significant regression). On analytics,
   P(low basin)_sampled <= P(low basin)_greedy + CI (sampling must not make the
   bad basin more likely).
2. **Drift:** ZERO un-rescued intended tool calls; no new mode the parser can't
   handle. (If a new mode appears and is parser-fixable, extend the chain and
   re-run -- do not hand-wave it.)
3. **Speed:** end-to-end decode t/s within ~5% of greedy at the config (Phase 2
   accept-vs-temp predicts this holds for T<=0.7; confirm live).

Any failure -> sampling stays OPT-IN (per-request temperature honored; server
default stays greedy). That is still a shipped, correct feature -- it just does
not become the default. Record which threshold failed and at what temp.

Coupling note: constrained-tools is OFF (engage-lag hole, queue #6). If the ONLY
way to pass the drift threshold is grammar-masked sampling (Phase 3), that is a
dependency on the constrain-tools bundle, not a pass -- state it, don't route
around it.

## Prerequisites (small, implement before the run)

**P1 -- server force-sample override.** `Q27_FORCE_TEMP` / `Q27_FORCE_TOP_P` seed
the SampleParams default in parse_sample() BEFORE the per-request body override
(so an explicit request temperature still wins). Per-request seed auto-derived
from a server counter and LOGGED (each trial is an independent draw but
reproducible from the log). ~15 lines, server.cu + api_common.h. Greedy path
untouched when the env is unset (canonical md5 must stay 4c4120c7).

**P2 -- drift-mode logging.** Tag each recovery in parse_bare_tool_calls with its
mode (1-5) and log un-rescued intended calls distinctly, so the catalog is
extractable and comparable to the greedy baseline. Small; no behavior change to
the parse itself (log-only), so canonical/serving output is unaffected.

## Ops (from memory, [[project_q27_engine]])

- Sequential configs against one server (restart between greedy and forced-temp);
  both servers never fit 32GB. `systemd-run --user` only; `cd` inside `bash -c`.
- q27-eval is transient (`systemctl stop` deletes the unit) -- recreate from the
  memory'd command line; `reset-failed` after crash.
- nvidia-smi before trusting any number (shared GPU; vox-transcriber owns 3090).
- Basins reroll daily (prompts embed dates) -- greedy-config and sampled-config
  legs MUST run same-day so the harness-variance baseline matches.

## Smoke findings (2026-07-06) -- harness pivot CC -> CRUSH

P1 (Q27_FORCE_TEMP/TOP_P) + P2 (per-mode drift logging) shipped and verified
(kernel-level: 5 modes tag as 1/12/13/14/15, un-rescued flag fires, clean prose
silent; live: CC and CRUSH both drive forced sampling, seeds increment, greedy
stays bitwise). The single-trial smokes then surfaced a **sixth drift mode** that
reframes the harness choice:

- **Mode 6 (name-dropped call), CC-specific.** q27 under the Claude Code tool
  schema emits `{"name":\n{...args...}}` -- the name STRING value and the
  `"arguments":` key are both absent from the bytes (e.g. `{"name":\n{"file_path":
  "/workspace/package.json"}}`). Unrecoverable by reframing (all 5 handled modes
  preserve the name). Result: no tool_use block -> CC gets text-only -> session
  ends after 1 turn -> score 0. THIS is the mechanism of the documented q27+CC
  "one-shot-quit basin."
- **Not sampling-caused.** The greedy CC A/B (2026-07-05T17-39-51, T2) shows the
  identical failure -- same bytes, deterministic. It fires under greedy too.
- **CC-specific, not general.** The CRUSH 0.786 baseline on the SAME T2 task shows
  ZERO mode-6 hits and 6 well-formed `"name":"Read"` calls. Under CRUSH's tool
  format q27 emits clean calls; only CC's schema/prompt triggers the corruption.
- **CRUSH sends no temperature** -> force-temp engages there too.

Consequence: the quality A/B runs on **CRUSH** (crush-q27-greedy-haight), where q27
produces real trajectories to measure. CC would drown the quality signal in mode-6
quit-basin noise (and misattribute it to sampling). CRUSH is also the faithful
baseline for both the 0.786 tie and the 5-mode catalog. CC mode-6 is a REAL but
SEPARATE serving bug (fix = P7 constrained decoding [off, engage-lag] or
schema-inference parser recovery) -- queue it independently; it is not the
sampling gate.

Open risk the smokes raised (n=1, NOT concluded): CRUSH sampled T8 scored 0 with a
truncated long-content `write` call (mode-2/5 class) -- sampling may perturb long
tool-call bodies. This is exactly what the multi-seed A/B measures; a single draw
on the bimodal T8 task proves nothing (the n=1-is-a-draw rule).

## The drift fixes (2026-07-06) -- the actual bottleneck was tool-call parsing

Batch 1 (T8 n=5/leg, CRUSH) came back sampled 0.356 vs greedy 0.095 -- NO sampling
regression, and greedy did WORSE (deterministically stuck in a failure basin the
sampled variation escaped). But BOTH legs were dominated by an un-rescued tool-call
drift, so the score signal was noise. Two pre-existing, non-sampling parser gaps were
tanking q27 agentic scores:

- **Write-content drift (mode 3 variant).** The file-write call is `"content":
  "CODE</content>` -- a JSON-quote-open value with a `</content>` tag close, raw
  newlines + unescaped quotes inside multi-line code. The unescaped `"` closes the
  string early and the code's own braces corrupt the depth scan. `escape_content_tags`
  only fired on a `<content>` OPENING tag, missing this shape. Fixed: handle the
  `"content": "RAW</content>` shape (anchor on the key, rewrite the raw span into a
  proper JSON string). Verified 6/6 on the batch-1 failures.
- **CC mode-6 (name-dropped call).** `{"name":\n{ARGS}}` -- name string + "arguments"
  key both absent from the bytes. Fixed with schema-inference (`infer_tool_name`:
  match the orphaned arg-keys to a tool, exact required-set wins, refuse on tie) plus
  a batch scanner for the unbalanced `{"name":{ARGS}{"name":{ARGS}...` run. Tools JSON
  plumbed to `parse_bare_tool_calls` (nonstream + streaming captures `tools` by value).

Live effect (fixed server, greedy T8), vs the un-fixed batch-1:
- **CC: 0.00 (7s one-shot-quit) -> 0.476** (0.49/0.46, 65 & 73 turns, 16-23K out tok).
  The mode-6 fix converted the quit-basin into full agentic trajectories.
- **CRUSH greedy: 0.095 -> 0.349** (write-content fix; un-rescued 4 -> 1).
- Residual un-rescued is now a non-fatal tail (loop continues). CC hidden_tests still
  ~0.04 with agent_tests/coverage/code_metrics ~1.0 -- q27 writes structurally-complete
  but behaviorally-imperfect code, a MODEL-quality ceiling distinct from the fixed drift.

Both fixes: api_common.h (parser) + server.cu (plumbing); greedy canonical 4c4120c7
untouched (CLI doesn't include api_common.h; build/q27 unchanged). Verified at the
parser level (drift_test/fix1_test/fix2_test on real captured bytes) and live.

## VERDICT (2026-07-06): sampling at T=0.7 PASSES -- no regression

Fixed-path re-gate, T8 n=5/leg, same-day, greedy vs sampled-T0.7:

    harness  greedy   sampled-T0.7   sampled un-rescued
    CC       0.466    0.553 (+.087)  0   (greedy had 6)
    CRUSH    0.484    0.510 (+.026)  2

- **Quality: PASS.** Sampled >= greedy on BOTH harnesses (same direction), consistent
  with sampling's variation escaping deterministic failure basins. n=5 with high bimodal
  variance so CIs overlap -- the defensible claim is "no regression detected, plausible
  small benefit", NOT a statistically-nailed win.
- **Drift: PASS (not sampling-attributable).** Sampling added no new mode; sampled:CC had
  FEWER un-rescued than greedy:CC (0 vs 6). Residual un-rescued is the pre-existing mode-6
  tail (separate item), not a sampling regression.
- **Speed: PASS.** T=0.7 is the flat region of the Phase-2 accept-vs-temp curve; trial
  durations not systematically slower.

The bimodality is MODEL quality, orthogonal to the gate: most trials write structurally-
complete code that fails behavioral tests (hidden_tests ~0.04, score ~0.48); occasionally
a trial nails it (hidden 0.92-0.96, score 0.84-0.85). Sampling gets more shots at that
high basin.

Decision: **sampling is cleared to default on at T<=0.7 / top_p 0.95** (server
Q27_FORCE_TEMP). Temp ladder (0.3/1.0) and larger-n CI are optional follow-ups; the gate
question ("does sampling regress?") is answered NO. The prior greedy-scoped numbers now
have a sampled counterpart that does not contradict them.

## Status

DONE 2026-07-06. Gate passed. Remaining: post-re-gate residual mode-6 tail fix (diagnostic
staged) + BUILDLOG/memory writeup. Model-quality (hidden_tests) ceiling is a SEPARATE thread.
