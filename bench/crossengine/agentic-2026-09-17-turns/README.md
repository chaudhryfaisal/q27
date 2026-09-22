# Why ninfer runs fewer Claude Code turns than q27 (issue #49), 2026-09-17

Question (issue #49): ninfer finishes SWE-bench-style Claude Code tasks in
about half the API turns q27 takes, with much shorter reasoning. Is the
shorter reasoning the cause, and where does it come from?

Short answer: the two engines behave the same on the same state once a
session is a few turns deep. They differ at the planning turn (turn 0 and the
turn or two after it), where ninfer thinks about 40% as much and goes
straight to the file it suspects instead of orienting (ls / git / find) and
reproducing first. Those first moves set the trajectory, and each engine then
continues in the style of its own history. Every engine-side mechanism we
could name on q27's side was tested and excluded (tokenizer, rendering,
effort line, drafter, sampler seed, output drops, error marker), and on
ninfer's side the rendering and the speculative verify are excluded too.
What is left is ninfer's forward pass on its NVFP4 / int8-KV artifact, and
the quantization ladder shows it is not bit width as such: llama.cpp at
Q4_K_M reasons as long as Q8_0 on the same prompt and q27's 4-bit tier a
little longer.

## Method

Everything here runs the SAME recorded Claude Code request bodies (Anthropic
`/v1/messages`, recorded from q27 sessions with `REQBODY_LOG`) or the same
raw rendered prompt against each engine, N seeds per state, and compares the
first action, the thinking length and the visible text before the tool call.
Bodies are real session content and stay out of the repo (scratchpad only).
Scripts: `state_replay.py` (Anthropic-path replay, saves the returned blocks),
`paired.py` / `classify.py` (paired per-state comparison and first-action
classes), `depth_think.py` (thinking by turn depth from retained transcripts),
`short_inst.py` (per-instance tool sequence), `recon_ninfer.py`
(ninfer-trajectory bodies reconstructed from transcripts, validated 0 diffs
against q27's recorded bodies on 6 instances), `raw_think.py` (raw prompts,
in `../agentic-2026-09-09-echo/`).

ninfer's `--request-log-jsonl` records no bodies and its Anthropic path
ignores a per-request `seed` (only the OpenAI chat path parses one), so
ninfer arms are unpaired samples; q27 arms are seeded.

## Results

### 1. Deep states: identical behaviour

Turns 8 / 12 / 16 of q27's recorded sessions (27 states x 6 seeds each):

| arm | orient | inspect | read | run | edit | end turn | think med | narrated |
|---|---|---|---|---|---|---|---|---|
| q27 (DFlash2 Q8) | 4% | 22% | 16% | 43% | 11% | 4% | 314 | 52% |
| ninfer (DFlash2, NVFP4) | 4% | 22% | 15% | 43% | 11% | 3% | 310 | 55% |

Post-edit states: run 68% vs 72%, end turn 12% vs 8%. Given the same
history ninfer does not edit sooner, verify less, or stop earlier.

### 2. Turn 0: ninfer plans shorter and jumps to the target

Same 12 turn-0 bodies, 24 seeds each:

| arm | inspect (grep/cat target) | orient (ls/git/find) | read tool | run | think med |
|---|---|---|---|---|---|
| q27 | 34% | 35% | 28% | 2% | 339 |
| ninfer | 62% | 16% | 20% | 1% | 130 |

Live transcripts agree (median thinking by depth, 12 instances per leg):
turn 0 q27 309 vs ninfer 166 / 113 (two ninfer legs); turn 1 79 vs 66 / 82;
depth 8+ overlapping. Narration (text before the call) differs at every
depth live (q27 50-83%, ninfer 17-42%) but NOT on identical states, and on
34 raw byte-identical mid-session prompts q27 / q27 q6 / llama.cpp Q8_0
narrate 67 / 66 / 64% with per-state all-or-nothing agreement: narration is
a property of the state the trajectory reached.

### 3. Where the turn count goes

ninfer finished 4 of 12 instances in 3-5 turns with the shape read, read,
edit, end (no reproduction; no test run at all on two of them). q27 took
15-25 turns on the same four, reproducing and running pytest repeatedly. On
the other 8 instances the two engines are within a few turns of each other.
All four short ninfer patches touch the gold file and read as plausible
fixes; the reporter's "breaks things" is not visible in our proxies.

### 4. Excluded on q27's side

| candidate | test | result |
|---|---|---|
| tokenizer tool tags (v0.11.4) | fixed 09-10, re-measured | -43% thinking/turn on q27, gap in turns remained |
| rendering: billing header dropped, schema keys sorted (ninfer's translate.cpp) | q27 on ninfer-style turn-0 renderings, 12 seeds | 316 / 292 (p=0.013) / 330 vs plain 339; ninfer 130; action mix unchanged |
| reasoning-effort line | both engines implement the Qwen3.8 template line; Claude Code sends `output_config.effort=medium` on every request | neither renders a line (q27 honours it per request; campaign also pinned medium) |
| DFlash2 drafter bias | ladder vs DFlash2, same binary, 8 states x 32 seeds; then a full ladder campaign leg | 222 vs 228 chars, 61 vs 59% narrated, every per-state p 0.27-0.96; leg: 19.9 turns/inst vs 21.4 / 21.2 on the DFlash2 legs (ninfer 14.0), gold 11/12, turn-0 thinking 261 |
| is_error marker (`[tool_error]` prefix) | ninfer on marked vs plain bodies | no effect; post-error behaviour equal on both engines |
| output drops | engine token counts vs returned blocks | exact on both (q27 -1 EOS, ninfer 0) |
| quantization tier | q27 default vs q6, raw prompts | equal (p=0.73); q27 = llama.cpp Q8_0 reference (per-state ratio 1.07, p=0.30) |

### 5. Trajectory-level legs (12 SWE-bench instances each, Claude Code, effort medium)

| leg | turns/inst | think K/inst | out tok/inst | gold | wall/inst |
|---|---|---|---|---|---|
| q27 DFlash2 Q8, seed 0 (09-10, two legs) | 21.2 / 21.4 | 27.5 / 29.7 | 11.2K / 11.7K | 11 / 10 of 12 | 75 / 70 s |
| q27 ladder, no drafter (09-17) | 19.9 | 32.5 | 12.2K | 11/12 | 92 s |
| q27 DFlash2 Q8, `Q27_SEED=random` (09-17) | 22.0 | 34.5 | 12.6K | 11/12 | 78 s |
| ninfer DFlash2 NVFP4 (09-07) | 14.0 | 15.5 | 6.2K | 11/12 | 36 s |

The four q27 legs cluster at 20-22 turns whatever the drafter or the seed;
ninfer sits at 14. Per instance the spread inside q27 is large (pytest-5809
3 to 8 turns, xarray-4094 29 to 56), so a 12-instance aggregate carries
about +-2 turns and single-instance comparisons say nothing.

### 6. Greedy turn 0: the difference survives argmax decoding

Same 12 turn-0 bodies, temperature 0 / top_k 1, two samples each (both
engines fully deterministic, 12/12 pairs identical):

| arm | think med | narrated | inspect | orient | read | p (thinking) |
|---|---|---|---|---|---|---|
| q27 | 292 | 100% | 50% | 33% | 17% | |
| ninfer | 136 | 17% | 75% | 8% | 17% | 0.0001 |

On 7 of the 12 prompts the argmax first call is the same tool with the same
or a near-identical argument on both engines; on 3 q27 orients (`git log`,
`ls`) where ninfer greps. What differs on every prompt is the amount of
thinking and whether any visible text precedes the call. q27's narration is
97-100% on every rendering variant of the same prompts (billing header
dropped, schema keys sorted, both); ninfer's is 36% sampled and 17% greedy.
With the sampler removed and the prompt held fixed, what is left is the
engine's forward pass.

### 7. ninfer's own histories: both engines converge again

19 states reconstructed at turns 8 / 12 / 16 of ninfer's recorded
trajectories, 6 seeds each:

| arm | orient | inspect | read | run | edit | no call | think med | narrated |
|---|---|---|---|---|---|---|---|---|
| q27 | 10% | 12% | 18% | 25% | 12% | 11% | 738 | 44% |
| ninfer | 8% | 11% | 26% | 28% | 8% | 11% | 518 | 44% |

Thinking p=0.39 pooled (per state q27 longer on 11, ninfer on 4, tie 4);
narration identical. Put next to section 1: on the same deep state the two
engines make the same moves whichever engine produced the history. The
trajectories diverge in the first turns and each engine then follows the
state it is in.

### 8. Quantization ladder on the planning turn: generic 4-5 bit does not shorten it

The 12 turn-0 prompts rendered raw (q27's `render_request`, byte-identical
across arms), 12 seeds each, `/v1/completions` on q27 and `/completion` on
llama.cpp:

| arm | think med | think mean | narrated | p vs Q8_0 |
|---|---|---|---|---|
| q27 default (Q4_G64 v2) | 304 | 355 | 50% | 0.001 (longer) |
| llama.cpp Q8_0 | 256 | 295 | 29% | |
| llama.cpp Q5_K_M | 244 | 267 | 100% | 0.56 |
| llama.cpp Q4_K_M (made today from the BF16) | 264 | 277 | 50% | 0.55 |

Thinking length at the planning turn does not move with the tier from 8 to
4 bits in llama.cpp, and q27's calibrated 4-bit tier sits slightly above
the 8-bit reference. "A ~5-bit artifact reasons shorter" is therefore not
a general property of the quantization; whatever ninfer's numerics do, it
is specific to its NVFP4 / int8-KV path. Narration on the raw prompts is a
knife-edge decision (29% to 100% between tiers of the same family), which
is consistent with ninfer's 17-36% being a numerics artifact and equally
consistent with the Q8_0 reference itself sitting at 29%.

Greedy (one argmax sample per prompt) on the same raw prompts: q27 280,
Q8_0 238, Q5_K_M 220, Q4_K_M 252 chars median; no arm narrated on any
prompt.

Caveat, then the correction: these raw prompts were 83 tokens short of the
served prompt. `tools/render_request` has no artifact to read
`general.name` from, so it rendered the Qwen3.6 JSON tool preamble
("return a JSON object ... inside <tool_call>") where the server renders
the trained XML preamble for 3.8. `count_tokens` bisect: no delta with no
tools, +83 with 1 or 28 tools; `Q27_TOOL_DIALECT=xml` reproduces the served
prompt byte for byte (25006 tokens). The mid-session raw prompts of
section 2 were built from q27's served system block and already carried
the XML preamble; only the turn-0 ladder above ran on the JSON one. The
tool now takes `--dialect` and infers xml from a qwen38 path, and the
ladder was re-run on the served prompt (next section).

### 9. ninfer without speculation: same terseness

Same 12 turn-0 bodies, 12 seeds, ninfer launched without `--spec` (there
is no "none" backend on its command line): thinking 159 vs 130 with
DFlash2, narration 33% vs 36%, inspect 64% vs 62%, orient 13% vs 16%. The
speculative verify is not what makes ninfer terse.

### 10. Quantization ladder on the served prompt (the clean run)

Same 12 turn-0 bodies rendered as the server renders them (XML preamble,
25006 tokens on turn0_1, byte-identical to the served prompt), 12 seeds
sampled and one greedy sample per arm:

| arm | sampled think med | narrated | p vs Q8_0 | greedy think med | greedy narrated |
|---|---|---|---|---|---|
| q27 default (Q4_G64 v2) | 304 | 94% | 0.010 (longer) | 281 | 100% |
| llama.cpp Q8_0 | 264 | 88% | | 240 | 100% |
| llama.cpp Q5_K_M | 242 | 85% | 0.48 | 220 | 100% |
| llama.cpp Q4_K_M | 266 | 88% | 0.73 | 220 | 100% |

Two things this settles. Thinking length at the planning turn does not
move from 8 to 4 bits in llama.cpp (Q4_K_M 266 vs Q8_0 264), and q27's
calibrated tier sits about 15% above the 8-bit reference rather than below
it, so bit width alone does not produce ninfer's 130. And on the trained
XML preamble every tier narrates 85-100% of the time, sampled or greedy,
which is what q27 does through the API (97-100%); ninfer renders the same
instruction block (its `kToolInstructions` is byte-for-byte q27's) and
narrates 17-36%. The JSON-preamble numbers of section 8 (0-50% narration)
were a prompt effect, not an engine one.

## Reading

Every engine-side mechanism on q27's side has been tested and excluded:
tokenizer, rendering (header, key order, effort line), drafter, seed,
sampler, output drops, error marker. On ninfer's side, speculation and the
rendering are excluded too. What remains is ninfer's forward pass on its
NVFP4 / int8-KV artifact: on an identical prompt under argmax decoding it
reasons about half as long at the planning turn, skips the visible text
before the call, and goes to the suspected file without orienting. Generic
4-5 bit quantization in llama.cpp does not do this, so it is something
about that artifact or those kernels specifically, and it is not visible
from outside ninfer. Everything after the first turns is state-following on
both engines.

The reporter's suggestion (expose a reasoning cutoff) would not reproduce
ninfer's behaviour: ninfer's thinking is not truncated (no replay sample on
either engine stopped on a budget; its log shows the budget unset), it is
shorter at the planning turn and then anchored by its own history. q27
already exposes the template's effort levels (`Q27_REASONING_EFFORT`,
`reasoning_effort` / `output_config.effort` per request) and
`--think-budget`; Claude Code's effort=medium is what both engines run at
today, and effort=low measured 18 turns vs 21 on 09-10.

Tooling fix that fell out: `tools/render_request` rendered the Qwen3.6
JSON tool preamble for 3.8 bodies unless `Q27_TOOL_DIALECT` was set (it has
no artifact to read `general.name` from). It now takes `--dialect` and
infers xml from a qwen38 path. The 08-22 flip-gate corpus was rendered
under the old default (JSON preamble, 83 tokens short of the served
prompt); its flip rates are numeric-stability measurements on a prompt the
server would not send, noted in `bench/flipgate/provenance.txt`.
- quantization ladder on the 12 turn-0 prompts, raw and byte-identical:
  q27 default, llama.cpp Q8_0 / Q5_K_M / Q4_K_M

## Reading

The reporter's suggestion (expose a reasoning cutoff) would not reproduce
ninfer's behaviour: ninfer's thinking is not truncated (0 of 162 deep-state
samples hit a budget), it is shorter at the planning turn and then anchored
by its own history. q27 already exposes the template's effort levels
(`Q27_REASONING_EFFORT`, `reasoning_effort` / `output_config.effort` per
request) and `--think-budget`; Claude Code's effort=medium is what both
engines run at today.
