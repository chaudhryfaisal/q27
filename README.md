# Quasar

A narrow inference engine for **Qwen3.6-27B-MTP and Qwen3.8-27B-MTP** (hybrid GDN+attention, trained-in MTP heads) and their fine-tunes on a single RTX 5090 (3090 and 4090/Ada also supported; Apple-silicon Metal backend for the q4s tier). One model family, one GPU, as fast as possible. In the spirit of [antirez/ds4](https://github.com/antirez/ds4)

## Why this is interesting

- **Fastest of the four engines tested, on this harness -- across two
  independent runs.** Mean wall per SWE-bench instance, same 12 tasks, one
  harness, unchanged competitor binaries:

  | leg | 2026-08-17 | 2026-08-19 |
  |---|--:|--:|
  | **q27** q5f | 46.8 s | **46.3 s** |
  | **q27** q4s | 48.5 s | 49.6 s |
  | llama.cpp | 71.4 s | 59.9 s |
  | vLLM | 84.5 s | 78.8 s |
  | ninfer NVFP4 | 96.8 s | 113.8 s |
  | ninfer int8 | 327.4 s | 266.2 s |

  Two runs is what it takes to see the honest shape: **q27 is the only leg that
  reproduced** (46.8 -> 46.3 s), while every competitor moved 7-19% on identical
  binaries. So the ordering is stable but the margins are not -- q27's edge over
  llama.cpp reads 1.53x one day and 1.29x the next. Quoting either as *the*
  number overstates the precision. n=1 per instance, trajectories diverge freely
  (turn counts range 1 to 56), legs ran sequentially, and sampling is each
  engine's own defaults rather than matched. Caveats in
  [FINDINGS.md](bench/crossengine/FINDINGS.md#6-caveats).

  q5f leads rather than the q4s canonical tier because q4s lost 3 of 12 instances
  to turn-1 quits in the 08-19 run (q5f: 0), which makes its mean cheap for the
  wrong reason. Those quits are model-side degeneration -- q27 logs the intended
  call it could not rescue, and all three were corrupted JSON rather than a
  dialect the parser refuses. The
  win is prefix reuse, not raw decode -- ninfer decodes *faster* than q27
  (250-261 t/s vs 215-223) and still takes 2-7x the wall time, because it
  re-prefills every turn. On a 24GB 3090: +19% decode at 2x the context over
  mainline llama.cpp (that run used the turbo3 default of the day at 262K; the
  default is now turbo5k, ~76% of that window for 43% fewer catastrophic KV
  positions -- `Q27_KV=turbo3` restores it). sglang 0.5.15 cannot load the
  model at all (BUILDLOG 2026-07-12).
  **Where q27 loses:** ninfer's NVFP4 tier peaks at 834 t/s aggregate at 8
  concurrent streams against q27's 531 (2026-08-19 re-run, both same session) --
  a 1.57x gap, down from 1.91x. Not an fp4-arithmetic win: their own
  `text_policy()` grants the batch kernel to NVFP4 only, so their int tier is
  locked out of it. Logged here at the same rate as the wins.
- **turbo3 3-bit KV cache** (capacity lever, not a quality-parity format --
  see the 2026-08-01 tail study: 6x fp8's catastrophic-position rate against an
  fp16 reference, at a dPPL of only +0.804%), symmetric K+V: 14.1 KB/token (14400 B,
  18-pair accounting), ~1% PPL, needle 6/6 at a 361K-token prompt,
  655K context allocatable on a 5090, two full 131K tenants at once,
  and a 24GB card promoted from a 32K box to a **262K box** (turbo3 is
  the Ampere serving default since v0.3.0; a bare w8 boot auto-sizes
  to the full native window). Ported from
  [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant);
  that fork refuses 3-bit K on this model class and caps 33% lower; I measured K costs +0.17%.
- **Native Anthropic Messages endpoint at Claude-Code grade**: thinking
  blocks, tool_use with input_json_delta streaming, exact
  `count_tokens`, anthropic-shaped context-limit errors, billing-header
  normalization so the prefix cache survives real CC turns, and a
  tolerant tool-call parser (nine cataloged drift modes) that ANY
  engine on this harness needs in order to score. One env var points
  Claude Code at it; OpenAI and Codex (Responses) shapes ride the same
  binary.
- **Self-speculation as the whole design**: trained-in MTP ladder +
  free suffix drafter through one shared-KV MMA verify -- 5.3-5.8
  accepted tokens per weight read on live traffic (231-246 t/s
  aggregate on a 5090).
- **Continuous batching on top of it** (serving default since
  2026-07-16): concurrent slots decode through ONE fused weight sweep,
  whole fused-verify rounds replayed as shape-keyed CUDA graphs --
  2-slot aggregate **1.41x** over round-interleaving, solo cost
  <=0.07%, byte-identity gated, zero config.
- **Receipts for everything**: bitwise canonical gates, negative
  results logged at the same rate as wins, and every number in this
  README traceable to a dated BUILDLOG entry.

**Baseline model (2026-07-09): vanilla Qwen3.6-27B-MTP** (`qwen36-27b-mtp`,
canonical md5 `a2982c51...`) -- the benchmark standard: bench rigs and gate
scripts default to it. Fine-tunes stay fully supported (`MODEL=`/`TOK=`/
`CANON_MD5=` env overrides; Qwopus3.6-27B-v2-MTP canonical `4c4120c7...`).
Pre-07-09 historical numbers (see BUILDLOG) were measured on Qwopus
unless noted. Qwen3.8 tiers carry their own canonicals (HF model
card); the benchmark standard remains vanilla 3.6 until a 3.8 bench
campaign replaces it.

## Quickstart

Requirements: an NVIDIA GPU with 24GB+ VRAM, CUDA toolkit 12.8+ at
`/usr/local/cuda`, and gcc. **12.8 is the floor for the default build on
every card, including a 3090**: `make` unconditionally emits `sm_120` and
links `build/pf4.o`, which is compiled for the arch-specific `sm_120a`
target, into every engine binary. (12.4 is only the floor for the sm_89
e4m3 MMA forms; older ptxas rejects those too.) To build on a 12.4
toolkit you would have to drop the sm_120 gencode from `NVCCFLAGS` and
the `build/pf4.o` dependency from the engine targets -- untested.
`make` builds ONE tri-arch binary (sm_86 +
sm_89 + sm_120: 3090, 4090/Ada, and 5090 class); arch dispatch is at
runtime -- fp8-KV and the e4m3 MMA paths need sm_89+, Ampere (sm_86/80)
runs the fp16-MMA verify (h16) and fp16/turbo3 KV. (v0.3.1+
release binaries include the sm_89 target; v0.3.0 did not.)

24GB cards (3090-class): build `make build/q27-server-w8` as well --
`Q27_W_MAX=8` shrinks the fixed VRAM stack so the server fits; the
default width-12 build OOMs at graph setup on 24GB. turbo5k (5-bit K +
3-bit V) is the default on Ampere since 2026-08-01 (no env needed) --
43% fewer catastrophic KV positions than turbo3 at 1.14x its per-round
decode; `Q27_KV=turbo3` trades that back for ~32% more context. Prefer
the q4s tier: its 2.27GB-smaller weight file goes straight to KV budget
(~56K tokens/GB at turbo5k rates, ~74K at turbo3's).
Cards with less than a true 24 GiB (A10-class cloud parts, ECC
reserve, decimal-GB VRAM) can also add `Q27_MAXD=4` (trims the graph
zoo ~280MB) and `Q27_SAMPLED=0` for greedy-only serving (skips the
sampled graph set, ~600MB on sm_86; temperature>0 requests get a 400).
Field-measured on a 22.6 GiB cloud A10 (issue #1, a31108a): q4s +
both knobs boots the FULL 262,144 native window; default weights
reach 102,400. The card must be otherwise idle: ~2.7GB of other
resident VRAM is the difference between boot and OOM.

3090 decode is power-sensitive: a fully-powered card (350-420W) runs
**~130 t/s** on short code-gen turns (q4s/w8, measured 126-150); a
200W-capped card gives roughly half that (issue #6). And `--ctx auto`
(or just omitting `--ctx`) sizes to measured free VRAM with an
arch-scaled safety margin -- ~254K on a 24GB Ampere card (q4s/turbo3)
with ~0.9 GB headroom, rather than sizing to the brim and OOMing at
`cudaGraphInstantiate` on cards that land below the 262K cap (issue #6).

Pick a quant first. Seven tiers, one repo
([signalnine/Qwen3.6-27B-MTP-q27](https://huggingface.co/signalnine/Qwen3.6-27B-MTP-q27))
-- all serve identically,
they trade decode speed for model quality:

| tier | file | GPU | pick it when |
|---|---|---|---|
| **default** (5.25 bpw) | `qwen36-27b-mtp.q27` | 24GB+ | the reference tier: bitwise canonical `a2982c51`, the most measured configuration |
| q4s (4.55 bpw) | `qwen36-27b-mtp-q4s.q27` | 24GB+ | max context on small cards; 2.27GB more KV budget, +5% decode, and wikitext PPL measures 0.26% BETTER than default (single Q4 lm_head + Q4 residual writers; error cancellation is real). Beats llama.cpp Q4_K_M by 4.2% PPL at 1.3GB smaller (matched protocol, 2026-07-22) |
| q5f (5.30 bpw) | `qwen36-27b-mtp-q5f.q27` | 24GB+ | **best quality that fits a 24GB card**: the q4s single-Q4 lm_head + `ffn_down` promoted to Q8. wikitext PPL **7.9491** -- beats q4s, default, and even q8, and matches q6 (7.9460) at 2.3GB less. Auto-ctx ~69K on a 3090 (vs q4s's ~315K), so it trades context for quality. HumanEval+ 30/30, LCB 23/30 (>= q4s). The FFN promotion stacks with the Q4-head cancellation where attn/ssm promotions don't |
| q6 (6.0 bpw) | `qwen36-27b-mtp-q6.q27` | 32GB | +0.35% PPL off Q5_K_M for ~5% slower decode (superseded by q6f at the same size) |
| q6f (6.11 bpw) | `qwen36-27b-mtp-q6f.q27` | 32GB | **the 32GB pick**: q4s's single-Q4 lm_head + `ffn_down`+`ffn_gate` at Q8 -- same recipe family as q5f, one more promotion. PPL **7.9189** beats q6 (7.9460) at the same size and nearly matches q6k at 2.25GB less; HumanEval+ 30/30 (q6: 29/30), LCB 22/30 (tied). Auto-ctx ~184K on a 5090 |
| q6k (6.8 bpw) | `qwen36-27b-mtp-q6k.q27` | 32GB | quality matching the best GGUFs of this model, ~10% slower decode |
| q8 (8.1 bpw) | `qwen36-27b-mtp-q8.q27` | 48GB+ | the near-lossless reference for big cards; PPL 7.9942 (better than default; q6/q6k's tuned promotions still edge it on wikitext -- error cancellation is non-monotonic in bits), and the acceptance-recovery tier: decode runs +26% over pure byte-scaling. Does not fit 32GB cards |

Task scores measure the same across tiers (q4s: fully validated --
PPL, suite, agentic NLL, needle, 18-run task dome, no deficit) -- the
quality tiers buy perplexity margin, not benchmark wins. When in doubt take the default.

### Qwen3.8-27B (v2 recipes)

Qwen3.8-27B (released 2026-08-14) is a repack-only port: every compile-time
constant matches (`docs/PORTING.md` predicted it from the config alone, and
the prediction held). The tiers are at
[signalnine/Qwen3.8-27B-MTP-q27](https://huggingface.co/signalnine/Qwen3.8-27B-MTP-q27)
and were **re-derived from a fresh per-tensor sensitivity sweep** (BUILDLOG
2026-08-14), NOT ported: the 3.6 recipes measurably do not transfer. The
q4-head error-cancellation family inverts, and the `ssm_out` Q8 promotion the
3.6 default/q6/q6k recipes carry is actively harmful at depth on 3.8 (+14%
dPPL at 24K single-pass vs +0.26% chunked -- it compounds through the GDN
recurrence, and chunked PPL cannot see it).

| tier | file | GB | wikitext PPL | HumanEval+ | needle |
|---|---|--:|--:|--:|---|
| q4s (v2) | `qwen38-27b-mtp-q4s.q27` | 15.70 | 7.3765 | 30/30 | 6/6 @ ~120K |
| **default (v2)** | `qwen38-27b-mtp.q27` | 17.00 | 7.3121 | 30/30 | 6/6 @ ~120K |
| q6 (v2) | `qwen38-27b-mtp-q6.q27` | 19.76 | 7.2233 | 28/30 | 6/6 @ ~100K |
| q6k (v2) | `qwen38-27b-mtp-q6k.q27` | 22.52 | 7.1718 | 29/30 | 6/6 @ ~40K |

All four are smaller AND lower-PPL than 3.6-recipe builds of the same
checkpoint. Per-tier canonicals are in the HF model card; note they gate
change, not identity (nearby tiers can share a 128-token greedy walk).

Serving 3.8 for agentic use: run with `--think` and the card sampler,
`--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0`. Full
19-task thunderdome suite under that recipe, one trial per task,
digest-verified load (BUILDLOG 2026-08-25): hidden-tests mean 0.928,
composite 0.895, 13 tasks at 1.000, nothing below 0.80. The 0.511 recorded
on 2026-08-15 for the same suite was measured with the think budget at its
16K default and greedy decoding, the two defaults the 08-22/08-23 entries
later found to be the problem; the tasks that flat-zeroed then
(task-queue, time-tracker, constraint-scheduler) score 1.000, 1.000 and
0.974 under the recipe. Single trials carry roughly +-0.1-0.15 of
trajectory noise on the volatile tasks, so read the per-task numbers as a
demonstrated ceiling, not a distribution. With thinking off the model runs
plausible-looking sessions that self-declare done and fail hidden tests.
The engine auto-selects 3.8's trained XML tool dialect from the artifact's
`general.name` and carries drift rescues (modes 14-22) for the forms the
model falls into under thinking.

```bash
# 1. tokenizer + your chosen tier from Hugging Face (Apache-2.0);
#    swap --include for the tier file you picked above
huggingface-cli download signalnine/Qwen3.6-27B-MTP-q27 \
  --include qwen36-27b-mtp.q27 qwen36-27b-mtp.tok CHECKSUMS.md5 \
  --local-dir models/qwen36-27b-mtp
# fine-tune variant: signalnine/Qwopus3.6-27B-v2-MTP-q27
# verify: (cd models/qwen36-27b-mtp && md5sum -c CHECKSUMS.md5 --ignore-missing)

# 2. build (CLI + server + test suites) -- or skip the toolchain and
#    grab prebuilt linux x86_64 binaries (sm_86/89/120 fatbin, CUDA
#    runtime statically linked; NEEDS NVIDIA driver r580+ -- on older
#    drivers build from source with your driver's toolkit, 12.4+ for
#    sm_89, 12.8+ for sm_120) from https://github.com/signalnine/q27/releases
git clone https://github.com/signalnine/q27 && cd q27
make

# 3. smoke test the CLI (should print 128 tokens; md5 of the output
#    line is the bitwise canonical a2982c51...)
./build/q27 ../models/qwen36-27b-mtp/qwen36-27b-mtp.q27 \
  --tokens "760,6511,314,9338,369" -n 128 --ctx 2048 --spec

# 4. serve -- zero config; defaults resolve the full measured stack
#    and --ctx auto-sizes to your VRAM (see Serving for escapes).
#    Binds 127.0.0.1 only. To reach it from other machines or from
#    containers (Claude Code in docker resolves the host via the
#    bridge, not loopback), opt in explicitly -- and set an API key
#    at the same time, since --host 0.0.0.0 with no key accepts
#    unauthenticated requests from anyone who can reach the port:
#      --host 0.0.0.0 --api-key <your-key>
./build/q27-server ../models/qwen36-27b-mtp/qwen36-27b-mtp.q27 \
  ../models/qwen36-27b-mtp/qwen36-27b-mtp.tok --port 8080
```

Sanity-check the server (native Anthropic Messages API; OpenAI
`/v1/chat/completions` and `/v1/completions` also served):

```bash
curl -s localhost:8080/v1/messages -H 'content-type: application/json' \
  -d '{"model":"q27","max_tokens":32,"messages":[{"role":"user","content":"say hi"}]}'
```

Point Claude Code at it:

```bash
export ANTHROPIC_BASE_URL="http://localhost:8080"
export ANTHROPIC_API_KEY="placeholder"
export ANTHROPIC_DEFAULT_OPUS_MODEL="q27"
export ANTHROPIC_DEFAULT_SONNET_MODEL="q27"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="q27"
claude
```

The server is single-model, so the model name in requests is accepted
as-is. Expect ~170-230 t/s decode on a 5090 depending on traffic shape
(see Reference numbers), warm multi-turn prefills served from the
prefix cache, and `count_tokens` + Anthropic-shaped context-limit
errors so Claude Code compacts correctly. Concurrent sessions
(`--slots N`) batch through one weight sweep by default -- 234-239 t/s
aggregate at 2 slots, zero config (see Serving).

## State of the engine (2026-07-30)

One binary serves Claude Code, Codex, and OpenAI clients at 231-246 t/s
aggregate live decode on a 5090 (90-116 t/s at 262K context on a 3090, turbo3 default),
with continuous batching ON by default: two concurrent slots decode
through one fused weight sweep plus shape-keyed CUDA-graph round replay
at **1.41x** aggregate (fp8 237.7 / turbo3 224.2 t/s), solo cost
<=0.07%, byte-identity gated. The 07-14..16 batching campaign got there
in three solo-neutral phases (FIFO 1.00x -> fused verify 1.21x -> fused
draft steps 1.31x -> graph replay 1.41x) and CLOSED on a measured P4
NO-GO; its physics triad: weight-BW-bound work wants FUSION,
state-latency-bound work wants OVERLAP, saturated work wants neither.
Single-stream, the 07-13 pass (k_vgemm flat-in-W verify GEMM, GEMV
occupancy retiers, GDN delta-step fusion) holds the short-bench suite
at 177.4 t/s. Full chronology, per-phase receipts, and every negative:
[docs/BUILDLOG.md](docs/BUILDLOG.md) (the 2026-07-13..16 entries).

Short cold prompts (<=128 tokens, the stateless single-shot / first-turn
case) get a prefill split-K that fills the SMs the small-output GEMMs
leave idle: ~8% faster prefill (server-measured, 62.4->57.4ms at 70
tokens), on by default since v0.3.3. The agentic NLL gate cleared it
(+0.018% on the full 154K CC corpus, worst segment +0.063%, vs a >+2%
bar); it auto-disables once the grid saturates, and every canonical
stays bitwise (the CLI eager-forwards, only the server path splits).
`Q27_GEMM_SPLITK=0` opts out. Warm agentic turns don't see it -- the
checkpoint already skips their prefill. Deep cold prefill (the saturated
large-T grid) gets a complementary lever since v0.3.4: an ntx M-minitile
GEMM that shares one activation `ldmatrix` load across two row-tiles
(+3.4% GEMM / ~6% cold-prefill wall, bitwise, sm_120; `Q27_PF_NTX=0`
opts out). fp4 is **not** a hardware dead end here -- that earlier verdict
was a toolchain trap, retracted 2026-08-14. The block-scaled fp4 MMA
(`mma.sync...kind::mxf4nvf4`) does exist on consumer Blackwell, but only
behind the arch-SPECIFIC `sm_120a` target; under the plain `sm_120`
gencode ptxas rejects the instruction and the failure is indistinguishable
from missing silicon. It runs, at 780-868 TFLOPS (`tools/microbench_mxf4`).
`tcgen05` really is absent. fp4 still loses at decode, for an unrelated
reason: nvfp4 spends 0.5 B of e2m1 plus one ue4m3 per 16 = 4.50 bpw
against Q4_G64's 0.5 plus one fp16 per 64 = 4.25, so it moves **1.0588x
the bytes for the same weights**, and decode here is a byte count, not a
FLOP count. Measured 1.016-1.085x end to end against a >=1.25x bar.

Two serving-side additions since: **P16 persistent prefix cache** (opt-in,
`--prefix-cache DIR`) survives a restart or a fresh conversation by writing the
P8 stable prefix to disk -- restart TTFT **8.15 s -> 1.20 s** on a 26,700-token
prompt, bitwise-identical continuations at up to 2 concurrent decode streams
(3+ streams default to the vgemm tolerance class, rel ~1e-6; `Q27_BATCH_GEMM=0`
restores bitwise at any concurrency), with an optional pinned host-RAM tier
above it (`--prefix-cache-ram-gb`). And a **reasoning-token budget**, ON by
default when the rendered prompt begins inside a `<think>` block: the block is
force-closed at half the request's `max_tokens` and the model still answers,
because an A/B over 240 scenarios found unbudgeted block mode truncating 28
times against inline's 0 while scoring level on raw accuracy. Later model-
generated reasoning is capped only when explicitly armed. See Serving for both.

### Reference numbers (v0.2.0, 2026-07-16, vanilla model, 5090)

Measured at `c0c5c5e` unless dated otherwise; full tables and history
in [docs/BUILDLOG.md](docs/BUILDLOG.md) and
[docs/BENCHMARKING.md](docs/BENCHMARKING.md).

- Short-bench suite **177.4 t/s** (fp16 stock CLI, 5-prompt mean;
  canonical `a2982c51` EXACT, stock clocks).
- 2-slot continuous-batching aggregate (`tools/batch_ab.sh`): fp8
  168.9 -> **237.7 t/s (1.41x)**, turbo3 158.5 -> **224.2 (1.41x)**
  over the FIFO baseline; solo regression <=0.07%. Zero-config spot
  check **234-239 t/s** aggregate (90.9% graph-cache hits).
- Decode @26K (server replay, fp8 basin): classic config 143.0 /
  **full default stack 176.3 t/s** (+23%). Echo ceiling (repetitive
  traffic, wide suffix): 26K zero-config server **400.6 t/s**.
- Live Claude-Code traffic (07-10, 9 scored task trials, 430
  requests): **231.3 t/s aggregate**, per-request median 225 / p75 277
  / peak 378 t/s. Qwopus fine-tune: **+5.7% decode at quality TIE**
  (246.5 vs 233.1 t/s aggregate, 5.65 vs 5.31 tok/rnd; matched 21-task
  sweep, 07-11).
- Prefill (fp8 batched TTFT): 8K 2.35s | 32K 10.4s | 128K 59.4s (~2200 t/s).
- Cross-engine: **+47% decode vs tuned llama.cpp's best config** (07-10
  protocol run; ~15 points of the 47 are bit-width, the rest is
  mechanism) and 202.7 vs 117.1 t/s vs vLLM on public SWE-bench agentic
  tasks -- see Benchmarks. llama's ngram spec still wins pure file
  re-emission, a mode mutually exclusive with its production draft-mtp
  config (decomposition: BUILDLOG 07-10/07-13).
- 3090 (24GB, turbo3 + h16, 07-12): **102.2 t/s median** live CC
  decode at **131K ctx** -- +19% decode at +60% context over mainline
  llama.cpp's strongest vanilla config.
- Weight tiers (07-12): q6k's matched-protocol PPL 7.9127 beats every
  measured GGUF of this model incl. unsloth's 26 GB flagship. On fp8 KV
  the quality tiers cost context (262144 / 192512 / 122880 auto-ctx for
  default/q6/q6k on a 32GB 5090, re-measured 07-17 under the calibrated
  auto-ctx) -- pair them with `Q27_KV=turbo3`, which keeps the full
  262144 window on every tier.
- q4s tier (07-16): 15.46 GB / 4.55 bpw. Paired-protocol PPL 8.0197
  vs default's 8.0409 (-0.26%, the third measured error-cancellation
  win), suite +5.2% (186.2 vs 177.0 t/s same-day) -- smaller, faster,
  AND lower perplexity. Exists for VRAM-starved cards: the 2.27 GB it
  returns is ~167K tokens of turbo3 KV budget. Field-measured on an
  A10 (22.6 GiB usable, issue #1): the arc ran 28,672 stock ->
  49,152 with `Q27_MAXD=4` -> 212,992 on q4s -> **262,144** (the full
  native window) at a31108a with the v0.3.0 capture gates
  (`Q27_SAMPLED=0`); default weights reach 102,400 on the same card.

### Carried state (pre-campaign, still in force)

- Width-12 verify (07-10): widths 9..12 belong to the suffix drafter
  (live agentic AL 10.6 on ~62% of decode); byte-identical at old
  widths by construction. W_MAX stays 12 -- W16 measured as a
  per-token LOSS, reopening only for file-re-emission traffic
  (`q27-server-w16`; BUILDLOG 07-13).
- The CLI stays fp16/reference so the bitwise canonicals hold; the
  server's fp8/mma stack is tolerance-class by policy (see Serving).
  Long-context validated on both compact KVs: needle 6/6 at 361K;
  allocation ceilings fp8 294,912 / turbo3 655,360 (W12, 5090).
- Tolerant tool-call parser: nine cataloged drift modes + the 07-11
  inference tie-break; every engine on this harness depends on it
  (strict parsing scores 0.000 on the hardest task class; BUILDLOG
  07-08). `--constrain-tools` stays opt-in (in-call cost 3.1x at depth).
- Multi-slot serving (`--slots N`): batching default since 07-16;
  `Q27_BATCH=0` restores R1b round-granularity time-slicing (the
  measured FIFO baseline). P8 stable-prefix snapshot (warm turns ~1.3s)
  + P9 same-session checkpoint ring own the prefill side.
- Measured NO-GOs with do-not-retry bars (deep-MTP ladder, GDN chunk,
  fdmma orchestration variants, W16 cap, P4 mixer co-residency,
  prefill FA2 relayout, draft-head shortlist): receipts in
  [docs/BUILDLOG.md](docs/BUILDLOG.md); parked levers in
  [docs/notes.md](docs/notes.md).

## Why this model is a good target

- Dense-ish 27B that fits entirely in 32 GB VRAM at 4-bit -- no expert offload, no DRAM scatter, none of the DSV4 pain
- MTP draft head trained into the checkpoint: self-speculation without a separate draft model
- Hybrid Gated-DeltaNet: near-O(1) memory per token for 48 of 65 layers.
  KV lives only in the 17 full-attention layers (16 + MTP, all global, no
  windowing): 72 KB/token fp16 (73728 B; the engine
  allocates 18 K/V pairs -- 17 attention layers + the MTP head). A dense 65-layer build would need ~68 GB
  @256K.
- Measured 5090 KV ceilings: fp16 ~180K | fp8 (36 KB/tok) **294,912** |
  turbo3 3-bit (14.1 KB/tok) **655,360** -- 2.5x the 262K native window.
  Auto-ctx caps at 262144 for fp8/turbo3, 131072 for fp16; explicit
  `--ctx` overrides. turbo3 position-bucket NLL is flat through 297K
  (tracks fp8 within +0.65-1.2% every bucket); the agentic quality gate
  closed PASS 07-16 (within +0.39% at CC depths on a real 154K CC
  transcript; shape-matched CC scores tie).
- The catch the per-token-memory napkin misses: attention KV is RESTORABLE state (any prefix row range replays for free) while GDN recurrent state is all-or-nothing per sequence -- you can only resume from a position you snapshotted. Hybrids make per-user context cheap but make context REUSE an engineering problem (prefix cache, mid-history divergence, multi-doc serving). That trade is where P8/checkpoint work lives; the measured cost of ignoring it was 7.9x wall-clock on agentic traffic (see build log P8/P9)
- The opponent, tuned honestly: llama.cpp's best measured config on this
  box is Q5_K_M + draft-mtp10 + p_min 0.5 (**~117 t/s @2K** single-stream;
  the win over stock is p_min, not draft depth -- swept 07-06). All
  cross-engine numbers use that config; see Reference numbers.

## Why paged-KV engines can't cache this model -- and how q27 turns that into wall time

vLLM's serving story is PagedAttention: KV memory is a global pool of
16-token blocks and the prefix cache shares blocks by content hash. On a
pure-attention transformer that is close to a free lunch -- attention KV
is an append-only, position-addressed log, so any cached prefix block
replays for free.

Hybrid GDN breaks the assumption the lunch depends on. 48 of this model's
65 layers carry no KV at all; their state is a dense recurrent summary
(128x128 per head + a conv ring) that REPLACES the token log. That state is
order-dependent and all-or-nothing: it cannot be paged, cannot be shared by
hash, and cannot be reconstructed from any cached block -- only replayed
from position 0 or restored from a snapshot you took yourself. A block
cache covers 17/65 layers; without the matching GDN state those blocks are
dead weight.

Measured consequence, 2026-08-17 four-engine run: **ninfer gets 0% reuse** on
real Claude-Code traffic -- 541 requests across two quant tiers, every one
logged `full_reset`, `computed_prefill_tokens == prompt_tokens` exactly. Not a
misconfiguration: their design admits exactly two resume offsets and their
maintainer docs list arbitrary longest-common-prefix reuse as an explicit
non-goal, for precisely the reason above. The cost is 97-327 s/instance against
q27's 47 s while *decoding faster* (250-261 t/s vs 215-223).

The other two engines do solve it, differently, and each pays somewhere:
llama.cpp reaches 93.9% by keeping recurrent-state checkpoints **per slot** at
~1047 MiB each -- which OOMs at 8 slots, capping it at 6 on a 32 GB card. vLLM
reaches 89.8%; note this is a change, its prefix cache measured 0% on this
architecture in the 07-15 run and has since been fixed upstream. Any claim that
hybrid GDN makes prefix reuse impossible is now falsified three ways; what
remains true is that it is not free, and the bill differs by design.

q27 treats the GDN summary as a first-class object instead of a cache miss:

- **P8 stable-prefix snapshot**: one device-side snapshot of all 48 GDN
  states at the last ChatML-stable boundary, plus split-encode at that
  boundary so tokenization itself is prefix-stable across turns.
- **P9 checkpoint ring**: pinned-host copies every 4096 tokens during
  prefill, so mid-history divergence rewinds to the nearest checkpoint
  instead of position 0.
- **P16 persistent prefix cache** (v0.6.0, opt-in): both tiers above die
  with the process, so a restart re-prefills everything. `--prefix-cache
  DIR` writes the GDN state plus the attention/MTP rows to a file keyed on
  the token prefix. Measured: restart TTFT **8.15 s -> 1.20 s** on a
  26,700-token prompt, bitwise identical output (at up to 2 concurrent
  decode streams; see the union-family note above). A second entry cut inside
  the system+tools block is shared across conversations, so a NEW
  conversation restores the part it has in common with an old one -- the
  thing a block cache would have given a pure-attention model for free.
- Attention KV needs neither: rows below the divergence point are
  append-only and stay valid in place.

A warm CC turn is therefore restore + suffix-only prefill -- real traffic
looks like `prompt=25473 hit=24136 pf=1337` in the `[req]` log, ~1.3 s
instead of a 10-20 s full re-prefill at p50 agentic depth. Measured over the
whole 08-17 agentic run: 88.7% of prompt tokens reused on q4s, 92.1% on q5f
(token-weighted, from `[gen] prefix_hit`), for an effective prefill rate of
24,224 and 31,314 tok/s against a cold 3,300-3,350.

That arithmetic, times every turn of a 30-90-turn trajectory, is the whole
wall-time story: **q27 46-50 s/instance vs ninfer 97-114 s** across two runs on
identical tasks, where
decode speed explains *none* of the gap -- ninfer decodes faster and still
loses by 2x. The continuous-batching stack (07-14..16) is independent of this
machinery and stacks on top: snapshots own prefill, batching owns decode.

The corollary cuts against spending more here: at an 8-11% miss rate, faster
prefill has little left to buy on this workload. See
`docs/plans/2026-08-17-prefill-performance.md`, which names the miss rate as
its own ROI gate.

## Architecture facts (ground truth from GGUF metadata)

| | |
|---|---|
| arch | `qwen35` (Qwen3-Next-style hybrid) |
| layers | 65 total: 48 Gated DeltaNet + 16 full attention (every 4th: 3,7,...,63) + 1 MTP layer (64, full attention) |
| hidden | 5120 |
| FFN | SwiGLU, intermediate 17408 |
| full attention | GQA 24Q/4KV, head_dim 256, QK-norm, gated output (attn_q packs Q+gate: 12288 = 2x6144) |
| DeltaNet blocks | attn_qkv [5120,10240] + attn_gate [5120,6144] + conv1d(k=4) + a/dt/alpha/beta (48-dim heads) + ssm_norm + ssm_out [6144,5120] |
| RoPE | partial, dim 64 of 256, sections [11,11,10,0] (M-RoPE; degenerates to standard for text-only), freq_base 1e7 |
| vocab | 248320, embeddings + lm_head untied |
| MTP | 1 nextn layer: eh_proj [10240->5120] combines (embedding, hidden) -> full attn + FFN -> shared lm_head |
| context | 262144 native |

Per-layer forward-pass semantics (extracted from llama.cpp source, not from
summaries): docs/SPEC.md.

## Performance model

Single-stream decode is weight-read-bound. The model: t/s = BW x eff /
bytes-per-step x accepted-tokens-per-round. Every number below is
measured.

| Card | DRAM | GEMV eff | Plain ceiling | Live agentic (measured) |
|---|---|---|---|---|
| 5090 (GDDR7) | 1.79 TB/s | 85-90% assumed, consistent with live rates | ~103-109 t/s @15.8 GB/step | **231-246 t/s** aggregate (vanilla/Qwopus, CC harness, 07-10/11) |
| 3090 (sm_86) | 936 GB/s | **81-90% ncu-MEASURED** (big FFN GEMVs ~90% DRAM SOL) | ~52 t/s | **102.2 t/s** median (turbo3+h16, 07-12) |

The efficiency assumption stopped being an assumption on 07-12: ncu on
the 3090 clocks the GEMV family at 81-90% of DRAM speed-of-light, and
the GEMV weight stream is 68% of the round. Plain decode's residual
~15% tail is GDN recurrence + ~140 launches/token; three attempts on it
(E4/E5/cp.async) came back negative, and the 3090 profile re-confirmed
there is nothing else material left in the kernels.

### Why self-speculation is the whole game at batch 1

Arithmetic-intensity framing (the same napkin datacenters use for the
opposite conclusion): the GPU offers hundreds of int8 ops per byte of
DRAM bandwidth, and batch-1 decode uses ~2 -- >99% of compute idles
while weights stream. Datacenters close that gap by batching USERS per
weight read; q27 batches WITH ITSELF: the width-12 verify amortizes one
weight read across the MTP ladder's drafts plus the suffix drafter's
free lanes. Live traffic runs 5.3-5.8 accepted tokens per round (echo
stretches hit 9-10.6), which is how 231 t/s clears a ~105 t/s plain
ceiling on the 5090 and 102 clears ~52 on the 3090. Corollary, twice
proven now: every decode win is (a) fewer bytes per step (quant policy,
fp8/turbo3 KV) or (b) more accepted positions per weight read (ladder,
suffix width) -- at batch 1 there is no third lever. The 07-14..16
continuous-batching campaign is lever (b) pointed across users --
concurrent slots SHARE the weight read (2-slot aggregate 1.41x) while
the per-stream arithmetic above still holds inside each fused round
(docs/multislot-throughput.md).

### Decode methodology (canonical, 2026-07-02)

These numbers are NOT interchangeable -- each answers a different question:

- **Short-bench suite** (SOTA-comparable): 5 fixed genre-diverse short
  prompts x 128 tokens, `--spec`, STOCK clocks -- `tools/shortbench_suite.sh`.
  **Current (v0.2.0, vanilla baseline): 177.4 t/s mean** (per-prompt
  170.7-185.5; series history in BUILDLOG). The per-prompt spread is
  trajectory/acceptance variance -- no single short prompt may carry a
  cross-engine number. It is also the param/launch-overhead CANARY: it
  caught the width-12 param-copy regression the depth gates missed.
- **Canonical prompt** (bitwise gate, NOT a benchmark): 128 tokens from the
  5-token canonical prompt -- vanilla baseline md5 `a2982c51...` (the
  standard; 144.2 t/s, 2.61 t/round at v0.2.0), Qwopus `4c4120c7...` for
  fine-tune gating. Held bitwise through every default-path kernel
  change since fd2 (the full list is in BUILDLOG). Tie-lottery
  sensitivity is why it gates bitwise identity and nothing else. It
  gates the CLI's reference defaults; the server's CC defaults are
  deliberately tolerance-class (fp8+mma) -- `Q27_PROFILE=ref` restores
  reference behavior there.
- **Tie-lottery methodology** (the project's most subtle measurement
  concept): tolerance-class numerics changes (fp8 paths, mma, split-count)
  re-roll greedy argmax ties -- **neutral in expectation, deterministic
  per build**. A quality flip on ONE benchmark basin is read via a basin
  MATRIX across tasks plus a re-roll on the next binary, never a single
  retrial (the mma case study is in BUILDLOG 07-10).
  Acceptance-sensitive decisions must name their basin; cross-BUILD
  text comparisons are invalid (same-binary legs only).
- **Depth numbers**: the current predictors are the Reference-numbers
  block (cctx 26K 143.0/176.3, live CC 213-246 aggregate); the 07-08-era
  61K/74K series and the 2K-soak long-generation series live in BUILDLOG
  with their cross-era bridges. Depth numbers, not 2K numbers, predict
  agentic wall time.

OC policy: headline + SOTA comparisons are reported STOCK (community numbers
aren't OC'd; sidesteps the non-ECC tail-risk conversation). +3000 stays a
supervised-bench option (+2.3% short-bench measured); the weight-checksum
tool (`--verify-weights`, `/health?verify=1`) exists for OC sessions. The
soft-error incident that set this policy is in the BUILDLOG appendix.

## Design decisions

- **Weights**: custom 4-bit symmetric groupwise (group 64, fp16 scales), packed for coalesced 128B warp loads, dequant fused into GEMV. Embeddings, lm_head, MTP layer, norms at 8-bit/f32. Repacked offline from the BF16 GGUF (container spec: docs/FORMAT.md).
- **KV cache**: fp16 by default; fp8 E4M3 is the server default since
  07-03 (`Q27_KV=fp8` on the CLI). Scale-free saturating conversion --
  measured amax sits 3.8x under the 448 E4M3 max, so per-row scales buy
  nothing. 36 KB/token, +11% decode @28.5K, PPL -0.05%, KL 3.4e-5. The
  CLI stays fp16 so the bitwise canonicals hold.
- **turbo5k 5-bit K + 3-bit V** (`Q27_KV=turbo5k`, src/turbo5.cuh) --
  **the Ampere default since 2026-08-01**. 32 Lloyd-Max centroids over the
  same WHT rotation turbo3 uses, 82-byte K blocks per 128 dims, 18.6
  KB/token. Exists because the 08-01 KV tail study found turbo3 carries
  6.0x fp8's catastrophic-position rate at a dPPL of only +0.804%: turbo5k
  cuts that 43% (114 -> 65 against an fp16 reference on a 64K agentic
  corpus). Measured on a 3090: 1.11-1.17x turbo3's per-round decode
  (~10-13% wall), 192512 auto-ctx vs turbo3's 253952 on the q4s tier,
  needle 6/6 @146K. Reads via fd2 and the fp16-tile H16 mma leg; it is
  deliberately NOT routed to the e4m3 fdmma leg, which would inflate its
  error 1.29x and give back a third of what the 5th bit bought.
  `Q27_KV=turbo3` trades the tail back for the deeper window.
- **turbo3 3-bit KV** (`Q27_KV=turbo3`; `turbo3v` = fp16-K diagnostic):
  WHT-rotated 50-byte blocks per 128 dims (ported from
  [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant),
  src/turbo3.cuh), 14.1 KB/token, full stack (decode, fdmma verify,
  batched prefill). PPL fp16 7.317 / fp8 7.327 / turbo3 7.381; K costs
  +0.17% -- the GQA=6 K-crater the source fork guards against does not
  exist on this model. Acceptance ties fp8 exactly on basin-matched
  replay.
- **MTP**: first-class. Draft + verify in one pipeline under a single CUDA graph. No separate draft context, no re-prefill.
- **Stack**: plain CUDA C++. No CUTLASS, no deps beyond CUDA runtime. Offline tools are Python: tools/repack.py (runs once; docs/FORMAT.md) and tools/gguf_to_hf.py (certified GGUF -> HF inversion, 866/866 tensors byte-exact, for cross-engine reference runs).
- **Serving**: OpenAI, Anthropic (Claude Code-grade), and OpenAI Responses (Codex-grade) shapes on one binary. Since 2026-07-03 the SERVER defaults to fp8 KV on sm_89+ (and on sm_86 to
turbo3 from v0.3.0, turbo5k since 2026-08-01) (--kv-fp16 or Q27_KV=fp16 opts out); the CLI keeps fp16 so decode canonicals stay bitwise.
- **Numerics contracts (batching)**: every fused lane family carries
  the ninv N-invariance gate -- **bitwise-when-untrimmed** -- plus its
  seam and twin legs; fused rounds run a UNION GEMM-family policy so
  batched numerics match solo; canonical + sampled-seed EXACT at every
  merge. The only text forks are the documented tolerance classes (A1
  suffix-trim, turbo3 concurrency tie re-rolls).
- **Two-tier batching guard**: user-EXPLICIT `Q27_BATCH=1` plus an
  incompatible env stays fail-fast FATAL; profile-DEFAULT plus an
  incompatible env auto-disables with one banner line and serves
  exactly as pre-batching (a default must never kill a
  formerly-working invocation).

## Security

The model chooses every byte q27's tool-call parser sees, so that parser is
attack surface. `docs/SECURITY.md` documents the trust boundaries (a parser
bug is GPU-host code execution; a *semantic* parser bug is client-sandbox
execution), why the vLLM `eval()` class does not reach this codebase (no
`eval`, no template engine), and the two defences: fuzzing for memory safety
(`make fuzz`, coverage-guided under ASan+UBSan) and per-shape adversarial
tests for the case where a model writing *about* a call must not make one.

## Serving

```
make build/q27-server
./build/q27-server model.q27 model.tok --port 8080
```

**Sampler: greedy by default, and that is a choice you may want to change.**
`--temp T` / `--top-p P` set the sampler used when a request OMITS the field
(the client still wins; `Q27_FORCE_TEMP`/`Q27_FORCE_TOP_P` still force). The
default is temperature 0 -- greedy argmax -- which is what every determinism
gate and byte-identity anchor in this repo assumes. It also means that a
client which sends no sampling fields gets greedy decoding: Claude Code sends
none, so agentic serving was greedy until you say otherwise, while llama.cpp
applies whatever the GGUF metadata carries (for Qwen3.8: temp 1.0, top_k 20,
top_p 0.95, min_p 0.05). Measured 2026-08-22 on a 4-task agentic suite, greedy
0.758, sampled 0.799, both defaults matched 0.847, llama.cpp 0.878 mean
hidden-test score. With both matched, no task separates the two engines
(p=0.199 to 0.898; pooled across 38 trials p=0.824); under the old defaults
three of four did (p=0.031, 0.031, 0.003). **For agentic serving of Qwen3.8, start the server with
`--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0`** -- the
model card's sampler, which is what llama.cpp applies from GGUF metadata.
(`--top-k`/`--min-p` added 2026-08-23; before that q27 had top_p only, so a
"matched sampler" comparison was not actually matched. They compose as logit
thresholds in llama.cpp's chain order and are gated against the CPU reference
at zero support mismatches. One documented difference: q27 filters the
temperature-scaled distribution, llama.cpp filters raw logits and applies
temperature last -- identical at T=1.0, the card value.) The third flag matters as much
as the first two: the default think budget is 50% of the client's
`max_tokens`, and Claude Code sends 64000, so reasoning is force-closed at
32,000 tokens. When that fires the model can emit EOS with no answer, and an
empty assistant turn ends the session -- measured 2026-08-23, 5 of 6
task-queue trials wrote no files at all and scored 0.000. `--think-budget 0`
removes the cap (llama.cpp has none, and its reasoning blocks run larger);
with it, every trial wrote files and the task mean went 0.334 -> 0.525
against llama.cpp's 0.549. The startup banner prints `sampler=greedy` or
`sampler=temp=1.00/top_p=0.95/top_k=20/min_p=0.05` so a log always records
which ran. The full 19-task suite under this recipe (2026-08-25) scored
0.928 hidden / 0.895 composite, against the 0.511 measured under the old
defaults.

**Defaults (2026-07-16) = the measured Claude-Code stack.** A bare server
serves the exact config every live trial and record number was earned on:
fp8 KV + `Q27_FD=mma` (e4m3 on sm_89+, fp16-MMA h16 on sm_80..88; fp8 KV
itself needs sm_89+; sm_86/Ampere defaulted to turbo3 from v0.3.0 and
to turbo5k since 2026-08-01 -- a bare w8 boot serves 192512 on a 24GB
3090 at the q4s tier, or 253952 under `Q27_KV=turbo3` (both measured on
the card 08-01) -- with
fp16 via `--kv-fp16`; sm_86/Ampere moved to turbo5k on 2026-08-01),
`Q27_PMIN=0.5`,
`Q27_MAXD=auto7`, suffix drafter at width 12, fast-head, no-think, phase
stats; `--ctx` auto-sizes the KV budget to free VRAM (capped at the
262144 native window for fp8/turbo3/turbo5k, 131072 fp16; single-slot).
Continuous batching is default since 2026-07-16 (`Q27_BATCH=1
Q27_BATCH_GRAPH=1`, graph-cache cap 64, shrunk to fit VRAM headroom;
`Q27_BATCH=0` disables; single-slot/solo traffic is byte-identical to
pre-batch). Every knob keeps its env/flag override (user env always
wins), `Q27_PROFILE=ref` restores the conservative reference behavior
(fp16, ungated, no suffix, fd2, no batching), and the **CLI binary keeps
reference defaults** so the bitwise canonical gates are untouched.
Escapes: `--kv-fp16 --no-fast-head --think --request-think --think-budget`, any individual `Q27_*`.

`--slots N` auto-sizes too (since 2026-07-18): with `--ctx` omitted the
free-VRAM budget is split across the N co-resident engines and every slot
gets the same computed window (logged `--ctx auto: <ctx> per slot`). Pass an
explicit `--ctx` to set slot 0 by hand and `--slot1-ctx` for the background
slots.

**Persistent prefix cache (P16).** Off by default; `--prefix-cache DIR` turns
it on. The P8 snapshot and P9 ring are process-lifetime, so restarting the
server (or starting a fresh conversation) re-prefills work already done.
This writes the recurrent state plus the attention/MTP rows for a prefix to
a file and restores it on a later boot. Measured on a 5090, 26,700-token
prompt: cold prefill 8.15 s, restart with a disk restore **1.20 s** (6.8x)
with the page cache cold, 0.72 s warm, output bitwise identical either way.

Two kinds of entry get written, both keyed on the token prefix itself:

- at the **P8 stable boundary**, which is what a later turn of the SAME
  conversation matches after a restart;
- at the last prefill-chunk boundary inside the **system+tools block**,
  which is what a DIFFERENT conversation matches -- Claude Code re-sends
  an identical 20-25K-token system prompt every session.

Lookup happens only after both RAM tiers miss. A candidate is chosen by a
filename hash but **verified by comparing the stored token vector element by
element** before any state is read, so a hash collision or a stale file can
never continue the wrong conversation. Entries carry a compat hash over the
model bytes, buffer geometry, and KV format; a mismatch is invisible rather
than coerced. Tuning: `--prefix-cache-max-gb 20` (LRU eviction),
`--prefix-cache-min 4096`, `--prefix-cache-max-tokens 32768` (also sizes the
pinned staging buffer, ~1.3 GB at fp8), `--prefix-cache-step 8192`.

On boot the most recent entries are read into the page cache on a background
thread, under the cover of the weight upload: that takes the first restore
after a restart from a 681 ms cold read to 36-39 ms. (`posix_fadvise
(WILLNEED)` was measured doing nothing at this size -- it is advisory and the
kernel declines a 1 GB readahead -- so the prefetch does a real chunked read.)

`--prefix-cache-ram-gb N` adds a pinned host-RAM tier above the disk. It is
**off by default on measured grounds**: a restore is alloc + read + import,
import (H2D) is a hard floor of ~38 ms/GB, and the boot prefetch already makes
the read 36-39 ms, so the tier makes the FIRST restore slightly slower (bigger
pinned slot) and saves 40-110 ms only on a repeat restore landing on a
different slot. Turn it on when the page cache is under pressure from other
work; otherwise leave it alone. Full numbers in
`docs/plans/2026-07-24-persistent-prefix-cache.md`.

Costs and limits: an entry is ~36 KiB/token at fp8 (~1.1 GB for 25K tokens,
~14 KiB/token at turbo3); persisting costs ~38 ms of D2H on the one turn
that pays it plus a background write; raw `/v1/completions` can restore but
never persists, and `/v1/responses` writes only the system-block entry (it
computes no stable boundary). It stores conversation content on disk in
plaintext -- see `docs/SECURITY-MODEL.md`.

**Auth.** Off by default -- loopback-only binding is the actual safety net
(see `docs/SECURITY-MODEL.md`); this is a convenience for the cases that
doc's own recommendation (put a real reverse proxy in front) is overkill
for, not a replacement for it under real multi-tenant/production exposure.
`--api-key KEY` (repeatable), `--api-key-file PATH` (one key per line, `#`
comments ignored), and `Q27_API_KEY` (env -- preferred in containers, where
CLI args are visible via `ps` but orchestrator-injected env vars are not)
all add keys; any of them is accepted. Every endpoint except `/health`
requires one once at least one key is configured. Both header conventions
work, so neither client family needs special handling:
`Authorization: Bearer <key>` (set via `OPENAI_API_KEY` for OpenAI-compatible
clients, or Codex's `env_key` in `~/.codex/config.toml`) or `x-api-key: <key>`
(set via `ANTHROPIC_API_KEY` for `claude` / Claude Code). Binding non-loopback
with no key configured prints a warning at boot but is not refused --
some deployments intentionally terminate auth at their own reverse proxy.

Behavior note (thinking): the default profile is no-think for speed -- it
prefills an empty `<think></think>` block so the model answers directly.
`--think` flips the server default the other way (prefills an open `<think>`
so the model reasons in a real block, closed with `</think>`, before it
answers).

Per-request thinking control is **opt-in behind `--request-think`**. Boot with
that flag and a request can override the server default in either direction, via
any of: `enable_thinking: <bool>` (OpenAI/Qwen top-level), nested
`chat_template_kwargs.enable_thinking` (llama.cpp/GLM), or
`thinking: {"type": "enabled"|"disabled"}` (Anthropic -- Claude Code's own
toggle). **Without `--request-think` (the default), those fields are ignored**
and thinking stays a boot decision (`--think`) -- so a benchmark or client that
sends `enable_thinking:True` (many harnesses do) can't silently flip a no-think
server into thinking mode. Thinking-on routes the reasoning trace to
`reasoning_content` (OpenAI) / a `thinking` content block (Anthropic), never
into the answer text.
Give a thinking request enough `max_tokens` to cover the trace **and** the
answer -- a tight budget is spent entirely on reasoning and truncates the
answer. When a client omits `max_tokens` the server defaults it to 8192
(unified across all three API shapes, clamped to the context window); a long
thinking trace wants more, set it explicitly.

**Reasoning budget (2026-07-30).** Tokens generated inside a prompt-seeded
`<think>` block are budgeted; on trip the block is force-closed and the model
answers with what it has. It is not a hard stop -- truncating the request
instead would reproduce the failure the budget exists to prevent. The default is
**half the request's `max_tokens`**, so the answer still fits after the close.
The prompt-seeded scope is intentional: standard thinking-enabled serving
starts inside the template-provided reasoning block, while a later model-
generated block is capped only when explicitly armed. `--think-budget N` with
`N>0` arms that later-block cap; `--think-budget 0` opts out. Requests can
likewise arm or override the budget, gated behind `--request-think`:
`thinking.budget_tokens` (Anthropic), `thinking_token_budget` (OpenAI/Qwen), or
`chat_template_kwargs.thinking_budget` (llama.cpp). Chat Completions and
Messages report a trip as `usage.reasoning_tokens` and
`usage.reasoning_budget_exceeded`; Responses reports the same fields under
`usage.output_tokens_details`.

Positive budget checks land after the accepted speculative round, so the
reasoning count may exceed the configured limit by that round's accepted width.
A natural close later in the same round ends normally rather than reporting a
budget trip.

Bounded sampled requests now retain normal speculative acceptance. The accepted
round is observed before host emission; if committing it would consume reserved
close/answer capacity, the engine retains only the prefix through the budget
trip, re-finishes that sampled state, and installs the forced close before the
next model decision. On the maintainer's matched RTX 5090 A/B
(`temperature=0.7`, `top_p=0.95`, `max_tokens=700`, n=2 per arm), bounded
throughput moved from 56.9 t/s on master to 147.9 t/s on this PR, while the
unbounded control held at 143.1 -> 143.5 t/s (inside noise). Both bounded PR
reps were exactly 147.9 t/s. The bounded arm used 275 rounds against 284
unbounded, so its slight lead reflects the budget closing the trajectory early,
not a faster kernel. This is a single-prompt gate, not a universal throughput
claim.


Why it defaults on: an 11-pack / 240-scenario A/B (BUILDLOG 2026-07-28, rescored
07-30) found unbudgeted block mode truncating **28 times against inline's 0**
while scoring level with it on raw accuracy -- 172 vs 173 of 220. The
unbudgeted block is the one configuration in that study that costs and returns
nothing.

A reasoning model handed zero reasoning budget over-refuses a narrow class of
borderline requests; mitigated 2026-07-13 by injecting a minimal default system
prompt when the client sends none (never fires for real Claude Code;
`Q27_BARE=1` opts out). For compliance-sensitive workloads default-on `--think`
remains the stronger lever (BUILDLOG 2026-07-13).

Three API shapes on one server:
- **OpenAI**: `/v1/chat/completions`, `/v1/completions` (text)
- **Anthropic**: `/v1/messages` -- native Messages API with thinking
  blocks, tool_use/tool_result, input_json_delta streaming, exact
  `/v1/messages/count_tokens`, an anthropic-shaped context-limit error
  (400) so Claude Code compacts instead of retry-looping, and cch
  billing-header normalization that keeps the prefix cache warm across
  CC turns. Claude Code-compatible:
  `ANTHROPIC_BASE_URL=http://host:8080 claude`
- **OpenAI Responses**: `/v1/responses` -- Codex CLI-compatible: function
  tools, `custom` freeform tools (apply_patch bridged through a
  one-string-param function), function_call/function_call_output history,
  reasoning items; event set verified against the codex-rs client source.

Codex config (`~/.codex/config.toml`):
```toml
model_provider = "q27"
model = "gpt-5-codex"

[model_providers.q27]
name = "q27 local"
base_url = "http://localhost:8080/v1"
wire_api = "responses"
```

Model tool protocol: tools rendered as JSON in the system `<tools>` block per
the qwen35 chat template; `<tool_call>` output parsed by a streaming splitter
(src/stream_split.h) that also routes `<think>`. The tolerant tool-call
parser recovers nine observed drift modes, logging each recovery for the
drift catalog. Greedy (spec decode) by default; `temperature>0` routes to
sampled SPEC decode -- top-p nucleus + Gumbel-max with rejection-sampled
spec acceptance at spec speed, seeded and reproducible, greedy left
bitwise-unchanged (docs/sampling-design.md). The exit-gate A/B passed
(docs/sampling-exit-gate.md), so the server can default sampling on for
clients that send no temperature via `Q27_FORCE_TEMP`/`Q27_FORCE_TOP_P`
(an explicit request temperature still wins). `--fast-head` trades output
exactness for ~7% more t/s.

Confidence-gated depth (P12 + P14): `Q27_PMIN=theta` caps verify width on
the drafter's top1-top2 margin; `Q27_DEXIT` stops drafting at the first
sub-theta margin. The adaptive 4..7 ladder lives in src/depthctl.h
(thresholds and knobs documented there); measured +2.7% geomean over
d4-gated, +4.2% over fixed-d5 on real-CC replay. Greedy output is
bitwise-identical under gating at every ceiling -- round segmentation
varies, tokens never do. The CLI and `Q27_PROFILE=ref` leave it all unset.

## Benchmarks

Four engines, one 5090, one harness, one accounting convention, same
Qwen3.6-27B-MTP weights. 12 pinned SWE-bench_Verified instances driven through
Claude Code; requests normalized across engines so each sees identical bytes;
timing taken client-side by a measurement proxy rather than from four different
log dialects (2026-08-17). What is normalized is the request payload -- SAMPLING
IS NOT MATCHED (each engine applies its own defaults when the client sends
none, which is the real-world configuration but not a controlled one), and
trajectories diverged between legs, so the reuse comparison is across different
conversation populations. n=1 per instance.

**Real agentic traffic** -- the workload the engine exists for:

| engine | decode | wall/inst | prefix reuse | gold |
|---|--:|--:|--:|--:|
| **q27** q5f (MTP + SuffixDraft, fused) | **223.4 t/s** | **47 s** | 92.1% | 10/12 |
| **q27** q4s | 215.5 t/s | 48 s | 88.7% | 9/12 |
| ninfer NVFP4 | 250.2 t/s | 97 s | **0%** | 11/12 |
| ninfer int8 | 261.5 t/s | 327 s † | **0%** | 11/12 |
| llama.cpp Q5_K_M + MTP | 120.5 t/s | 71 s | 93.9% | 11/12 |
| vLLM NVFP4 (no spec ‡) | 67.8 t/s | 84 s | 89.8% | 9/12 |

† 3 of 12 hit the harness's 700 s cap; its 9 uncapped instances averaged 203 s.
‡ vLLM's MTP path had to be disabled -- see the defect note below.

**Decode is not where engines differ most; prefix reuse is.** ninfer decodes
*faster* than q27 and still takes 2-7x the wall time, because it re-prefills
the whole conversation every turn: its design supports exactly two resume
offsets and lists arbitrary longest-common-prefix reuse as an explicit
non-goal, since Qwen3.6's GDN state cannot be rebuilt from a KV prefix. q27,
llama.cpp and vLLM all solve that; ninfer is the outlier.

**Concurrency ladder** -- aggregate decode t/s, distinct salted prompts, n=1.
Re-run 2026-08-19 on the same harness and the same protocol (8 slots / 16K),
because q27 gained ~19% at C=8 from work shipped after the original run and
leaving the old figure up understated its own engine by that much. **Competitor
binaries are byte-identical to the 08-17 run** -- ninfer built 08-15, llama.cpp
08-17, the same vLLM nightly image -- so this isolates q27's change rather than
measuring two moving targets:

| engine | C=1 | C=2 | C=4 | C=8 | 08-17 C=8 | slots at 16K |
|---|--:|--:|--:|--:|--:|--:|
| **q27** q4s | 141.3 | 229.7 | 352.3 | **530.6** | 412.9 | **8** |
| **q27** q5f | 134.7 | 173.8 | 299.9 | 509.7 | 385.9 | 8 |
| ninfer NVFP4 | 157.2 | 299.7 | 442.3 | **834.3** | 790.0 | 8 |
| ninfer int8 | 137.0 | 179.5 | 219.5 | 353.6 | 349.8 | 8 |
| vLLM NVFP4 | 67.4 | 121.4 | 215.7 | 438.7 | 398.8 | 6.53 |
| llama.cpp | 84.8 | 168.3 | 149.1 | 193.1 | 230.3 | **6** |

**The q27 rows run `Q27_DRAFT_CEIL1=1`, which is opt-in and OFF by default.**
Stated plainly because it is an asymmetry: q27 is tuned here and the other three
engines are at their shipped defaults. The flag is worth **+8.6%** at C=8 and
nothing anywhere else (q4s stock: 142.5 / 226.5 / 360.3 / **488.4**), because it
only binds where the width trim floors every lane -- so **q27 is +18.3% at C=8
on defaults alone**, and the tuned number is the ceiling rather than the
out-of-the-box figure.

Read the competitor deltas as the noise floor: on unchanged binaries they came
back +1.1%, +5.6%, +10.0% and -16.1%, so single rungs move ~10-16% between
sessions. Both q27 legs moved +28.5% and +32.1% in the same direction, which is
what makes them a result rather than a wobble. The one outlier inside q27's own
rows -- q5f at C=2, down 10.5% while its neighbours rose -- is that same
rung-level noise; its C=4 and C=8 track q4s.

**q27 still loses this half, by less.** ninfer's NVFP4 peaks at **1.57x**
q27's best (834.3 vs 530.6, same session), down from the 1.91x the 08-17 run
recorded -- the gap closed by q27 moving, not by ninfer slipping. *An earlier
draft of this section read that as a format win -- fp4 MMA engaging at batch
width -- because ninfer's own int8 tier peaks below q27. That attribution was
wrong, and their source says so: `text_policy()` hands `AllowA4` to
`QType::NVFP4` and `A16Only` to every other qtype, so **their int tier is locked
out of their own batch kernel by policy**. fp4 did not make nvfp4 fast; A16Only
made nint slow, and the within-engine comparison was never a format control. The
real gap is per-lane cost: their round wall is FLAT from C=1 to C=2 (15.04 ->
15.00 ms), which is a reclaimed-idle-machine story, not an arithmetic one.*
Client-side accounting agrees with q27's internal
`[req]` telemetry to within 0.1% at every rung, so these are comparable across
engines and to q27's own ladder history.

**Everyone pays for hybrid-GDN state; the bill lands in a different place.**
llama.cpp keeps recurrent state per slot at ~1047 MiB and OOMs at 8 slots
(measured ceiling 6). vLLM's KV pool holds 6.53 sequences at 16K. ninfer trades
reuse away entirely. q27 reaches 8 slots at 16K because M1 record-then-fold cut
its per-slot GDN cost -- that is what the 530.6 t/s rung is made of.

**Quality does not separate them.** benchlocal `--medium`, 5 packs / 75
deterministic-verifier scenarios, temp 0: q27 q5f **66**, llama.cpp **66**,
vLLM **66**, ninfer NVFP4 65, q27 q4s 64, ninfer int8 64 -- four engines and
five quant formats inside a 2-point band. No difference was DETECTED on these
75 scenarios -- at n=1 per scenario that is a null result, not a demonstration
of equivalence, and it does not rule out differences smaller than the band or
outside these packs. On this evidence the throughput numbers above
carry no quality asterisk.

**Defect found, filable upstream:** vLLM's MTP speculative path produces
*corrupted* generations under sustained agentic traffic on this checkpoint --
incoherent token soup returned as HTTP 200 after ~3 instances. Isolated over 4
instances x 3 configurations: spec ON = 1/4 gold with 3 sessions dead at turn 1;
spec OFF (prefix cache still on) = 3/4 gold with every session running 6-20
turns; spec OFF + cache OFF = 3/4 gold. One variable, and it is the speculative
path. vLLM is measured above in its best *working* configuration; no throughput
benchmark would have caught this.

Full methodology, fairness controls, the payload microbench, and reproduce
steps: [docs/BENCHMARKING.md](docs/BENCHMARKING.md). Harness, pinned task set,
and raw per-instance results: [bench/swebench/](bench/swebench/).

The 2026-08-17 four-engine run lives in
[bench/crossengine/](bench/crossengine/) -- harness, per-leg raw data, the
write-up in [FINDINGS.md](bench/crossengine/FINDINGS.md), and the two
attribution experiments the headline claims rest on (the vLLM MTP corruption
A/B and llama.cpp's slot-ceiling walk) under `isolation/`. The wall-time,
prefill and decode tables regenerate from the committed data with:

```bash
python3 bench/crossengine/harness/analyze.py bench/crossengine
```

**In the 08-17 archive the prefix-reuse column did not** -- it printed `n/a`
for both q27 and both ninfer legs, because that number comes from each engine's
own instrument and only llama.cpp's rode in the committed tap logs
(`cache_read_tok`, in `tap.llama.jsonl`). q27's reuse is parsed from
`[gen] prompt=N prefix_hit=M` server lines and ninfer's from its request log,
and neither file was archived.

**Fixed for the 08-19 re-run**, whose directory ships those server logs and
reqlogs (pure counters -- no prompt or completion text), so every leg's reuse
regenerates from a clone:

| leg | reuse, tok-weighted |
|---|--:|
| llama.cpp | 93.47% |
| **q27** q4s | **91.64%** |
| **q27** q5f | 89.91% |
| ninfer int8 / NVFP4 | 0.00% |
| vLLM | n/a (emits no cache field) |

vLLM stays `n/a` because it reports no per-request cache field at all, which is
an engine limitation rather than a missing file.

That section also documents the seven harness bugs the run hit, each of which
produced a plausible and wrong number first -- the worst of them said "ninfer
does not scale with concurrency", which is the opposite of true.

The earlier three-engine run (2026-07-14, q8 KV + greedy, llama `ngram-mod`
fork included) is superseded but not deleted -- it is the reason ngram-style
drafting is not in this engine: it added ~nothing on real coding while MTP
nearly doubled stock llama.cpp.

## Open items (2026-07-30)

- ~~**turbo3-vs-fp8 quality gate**~~ **CLOSED 2026-07-16, verdict PASS**
  (BUILDLOG "TURBO3 AGENTIC QUALITY GATE"): agentic-corpus NLL on a real
  154K-token CC transcript is within +0.39% of fp8 in every CC-depth
  bucket, and the shape-matched CC study (2x48K both legs, n=3/leg)
  ties/favors turbo3 -- the 07-15 band gap was the shape confound. No
  quality asterisk on turbo3; fp8 stays the sm_89+ CC serving default
  on speed alone -- turbo3 is the capacity lever there (2x96K vs 2x48K
  on 32GB) and, since v0.3.0, the outright default on Ampere.
- ~~**Saguaro-style 3090 off-path drafting**~~ **CLOSED 2026-07-24,
  verdict NO-GO** (BUILDLOG "off-path 3090 drafting"): the lever and the
  predictability are anti-correlated. Where the draft is expensive
  (prose/code, 2.0-2.7 ms of a ~16 ms round) the accepted count is
  predictable only 50-59% of the time even when conditioned on the gate's
  confidence cap; where it is perfectly predictable (echo) 96% of rounds
  are suffix-drafted and the draft is already free (0.12 ms/round).
  Realistic win +7-14% single-stream for a weeks-long dual-GPU pipeline,
  against ~90-130 t/s aggregate from simply running the second card as
  another server. The margin-conditioned predictor named as the one
  thing that could reopen it was then measured too: **58.7%**, not 80%.
  The drafter's own margin separates accepted from rejected steps at
  AUC 0.56-0.63, which bounds any predictor built on it -- self-reported
  confidence is not a proxy for agreement with the full model. Closed.
  Probes in-tree (`Q27_NJOINT=1`, `Q27_MPROBE=<file>`).
- ~~**Prefill follow-ons** (async producer/consumer + mbarrier rewrite of
  the prefill-attention kernel)~~ **CLOSED 2026-07-24, verdict NO-GO as
  filed** (BUILDLOG "prefill async/mbarrier"): the three shipped phases
  (cp.async +5.4%, fp8 QK^T +11.8%, fp8 PV +2.4%) already halved the stall
  profile it was filed against (14.23 -> 7.77 warp-cycles per issue). What
  binds now is occupancy -- 12.5%, 1 block/SM, limited INDEPENDENTLY by
  248 regs/thread and 70 KB smem -- and an async pipeline needs more smem
  stages, not fewer. Reaching 2 blocks/SM needs a 28% smem cut AND a 31%
  register cut, when the output accumulator alone is 128 registers.
  Reopens if long-context prefill (131K+) becomes a product goal: attention
  is 25.4% of prefill at 65K but ~40% at 131K. (The serial-threshold call shipped in v0.3.0 as
  `Q27_PF_BATCH_MIN` -- TTFT 350->31-33ms / 567->53-55ms; the CLI
  default is unchanged, so the canonical still holds.)
- ~~**Strict-parser zero-rescue config**~~ **CLOSED 2026-07-24, verdict
  NO-GO** (BUILDLOG "strict-parser grammar engage"): the configuration it
  unlocks is dominated. The 07-08 A/B prices it -- tolerant 0.837,
  strict+constrain 0.549 -- so the lever's best case is parity with the
  default that already works, at `--constrain-tools`' 3.1x in-call cost.
  The probe did find and fix **drift mode 13** (a wrapper-less call
  truncated inside an escape sequence: the repair's closing quote landed
  after a dangling `\`, escaping it and losing the whole call).
- **Graph-cache cap under churn**: live CC already draws 44+ keys vs
  the bench's 28; cap 64 swallows today's alphabet, revisit
  `Q27_BATCH_GRAPH_CAP` if multi-tenant composition churn widens it.
- **Does a budgeted think-block beat inline?** OPEN, not yet run. The 07-28
  A/B rescored (BUILDLOG 07-30) leaves block and inline level on raw accuracy
  (172 vs 173 of 220) while the block arm eats **28 truncations to inline's 0**.
  A budget recovering even part of those should put the block arm ahead --
  which no one on club-3090#741 has tested. The cheap version re-runs lcb-v6
  block with the budget on: its 13 failures are all `len(content)==0` runaways
  that reasoned 37-67 KB without answering, which is exactly what the cap
  converts. cli-40 is already NO-GO for this (07-30): at `max_tokens 1024` the
  cap relocates the spend and the answer meets the same ceiling.
- ~~**No intermediate K precision (5-6 bit band)**~~ **CLOSED 2026-08-01 by
  turbo5k** (`Q27_KV=turbo5k`, 18.6 KB/tok, now the Ampere default; see the KV
  bullet above and BUILDLOG 08-01 (b)-(f)). It landed at 65 catastrophic
  positions against turbo3's 114 and fp8's 19, recovering 79% of the
  turbo3 -> fp16-K gain at 43% of fp16-K's size. Two findings worth carrying
  forward: **PPL ranks turbo5k BACKWARDS** (+0.983% dPPL against turbo3's
  +0.804% while beating it on every tail metric, because 95-97% of both
  formats' absolute error cancels in the signed mean -- do not gate a KV
  format on PPL), and **e4m3 MMA staging is free for coarse quantizers and
  corrosive for fine ones** (+17.6% on turbo3's own error, +81% on turbo5's),
  which is why turbo5k reads through the fp16-tile H16 leg instead.
- **turbo5k's remaining tail is not a subset of turbo3's**: only 25 of its 65
  catastrophic positions are also turbo3's -- it fixes 89 of turbo3's 114 and
  introduces 40 new ones. Net -43%, but the two quantizers make different
  errors, so a workload that is fine on turbo3 is not guaranteed fine on
  turbo5k. Untested: whether TCQ (a different algorithm, not a bit-width
  change) would dominate both; filed as its own plan, not started.
- **`billing-header cch normalize`** fails in `test_tokenizer` on HEAD,
  independent of any 07-28/07-30 work (verified by stashing). Commit `5a81225`
  last touched that normalizer. Filed, not fixed.

Measured-and-parked levers (chunked-WY delta scan, AWQ-style scales, P11 split path, and friends):
[docs/notes.md](docs/notes.md).

## History

The full chronological record -- every DONE block with its numbers,
every negative result, the early milestones (M0..E6), the progress-log
table (43.4 -> 177.4 t/s), and the M6 prefill history -- lives in
[docs/BUILDLOG.md](docs/BUILDLOG.md). The BUILDLOG is the ledger, not
git history (commits sometimes batch a day's entries). Design docs and
phase plans: [docs/plans/](docs/plans/). Standing risk register, P10-A
status, and parked levers: [docs/notes.md](docs/notes.md). Multi-slot
throughput analysis (rewritten post-campaign):
[docs/multislot-throughput.md](docs/multislot-throughput.md). What a new
Qwen checkpoint has to match before it can load, and what a mismatch costs:
[docs/PORTING.md](docs/PORTING.md).

## License

MIT -- see [LICENSE](LICENSE).
