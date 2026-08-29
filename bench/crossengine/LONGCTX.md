# Cross-engine long-context decode on Qwen3.8-27B: decode survives on every engine, and vLLM's MTP-corruption verdict is stale

**TL;DR:** Apples-to-apples long-context bench of q27, vLLM (nightly), llama.cpp
and ninfer, all serving **Qwen3.8-27B-MTP** on the 5090, think-on, matched
sampler, same client-side instrument. Three results. (1) **Decode t/s does not
erode with context on any engine** -- flat to gently declining from 2K out to
51K. That is the hybrid linear+full attention (fixed-size GDN state), not an
engine feature. (2) The engines split on two axes that trade off: **q27 leads
decode** (~150-180 t/s, MTP+suffix), **vLLM leads cold prefill** (~2.5x faster
TTFT). (3) The 08-17 "vLLM MTP corrupts output, run spec OFF" finding **does
not reproduce on the current nightly** (v0.26.1rc1.dev1177) with the 3.8
modelopt-NVFP4 checkpoint: spec-on loads, is coherent, and is worth **+60%
decode** (78 -> ~115-138 t/s), narrowing q27's decode lead over vLLM from ~2x
to ~1.2-1.3x.

## Setup

One engine at a time on the 5090 (pinned by GPU UUID -- `--gpus device=0` is
the 3090 on this box), Qwen3.8-27B-MTP in each engine's native format,
thinking ON everywhere, sampler `temp 1.0 / top_p 0.95 / top_k 20 / min_p 0.05`
(see the min_p gotcha below), q27 loads digest-verified (`wsum b743d26b`).
Instrument: `harness/xengine_longctx.py` -- a client-side streaming driver
measuring TTFT and decode t/s from the SSE stream + usage, identical for every
engine. Two arms: (A) cold decode-vs-context sweep, unique prefix per request,
512 tokens generated; (B) a 12-turn incremental tool loop growing to ~29K
tokens, cache warm. Raw records: `longctx.*.jsonl` and `agentic.vllm38specON.*.jsonl` in this directory.

| leg | artifact | quant | spec |
|---|---|---|---|
| q27-def | qwen38-27b-mtp.q27 | ~5.25 bpw | MTP + suffix |
| q27-q4s | qwen38-27b-mtp-q4s.q27 | ~4.2 bpw | MTP + suffix |
| ninfer | qwen3_8_27b.ninfer (converted this run, 72 s) | int8 ~5.4 | MTP (`--spec mtp --lm-head-draft`) |
| llama.cpp | Qwen3.8-27B-MTP-Q5_K_M.gguf (`01818e49`) | ~5.5 | `--spec-type draft-mtp` |
| vLLM | qwen38-27b-nvfp4-rtx5090 (modelopt) | ~4.25 NVFP4 | both OFF and ON (see below) |

## Decode t/s vs context (arm A, median of 3)

```
      engine       spec |    ~0    3K    6K   13K   26K   51K
     q27-def    MTP+sfx |   145   153   179   178   166   164
     q27-q4s    MTP+sfx |   148   174   184   166   168   154
      ninfer        MTP |   156   126   148   147   148   135
 vllm spec-on       MTP |   115   133   138   133   116   117
vllm spec-off       --- |    79    79    78    78    76    74
     llama.cpp draft-mtp|    92    96    87    95    85    77
```

Every engine is flat-to-gently-declining. The tok/s spread within a leg is
MTP acceptance (content), not context; on q27's own `[req]` telemetry the
pure forward-pass rate declines only ~12% out to 50K (~23% to 100K). q27-q4s
== q27-def, so q27's lead is not a quant artifact.

## Prefill and the loop

Cold TTFT at 26K/51K: **vLLM 2.9/7.4 s**, q27 7.6/16.8, ninfer 8.7/19.0,
llama 9.1/20.5. vLLM's prefill is the fastest by ~2.5x; its weak axis was
decode, which spec-on mostly fixes.

In the incremental loop, **every engine reuses the prefix**: warm TTFT stays
in the 0.2-1.2 s band across 12 turns to ~29K for all four (q27 0.71->0.86,
ninfer 0.82->0.95, vLLM 0.18->0.58, llama 1.77->1.21). The earlier "ninfer/vLLM
get ~0% reuse" observations came from real agentic traffic (tool retries,
reordering); a clean prefix-extension loop is the favorable case and everyone
passes it.

## The stale verdict: vLLM MTP on this nightly

The 08-17 isolation run (Qwen3.6, older nightly) found vLLM's MTP path
returning corrupted generations under agentic load, so the cross-engine
harness pinned vLLM to spec-OFF. Re-tested here because the 3.8 artifact
carries the MTP head (15 `mtp.*` tensors) even though
`num_nextn_predict_layers` is unset in its config. Enable with:

```
--hf-overrides '{"num_nextn_predict_layers":1}' \
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The nightly logs "Detected MTP model. Sharing target model embedding/lm_head
weights with the draft model", and output is coherent (correct code, math,
prose) across the coherence probes, the sweep, and the loop. Decode goes
78 -> 115-138 t/s.

**Agentic re-run (same day):** 24 Claude Code SWE-bench sessions (12 think-on,
12 no-think) against spec-on, ~294K generated tokens at 118-141 t/s, 57-75%
MTP acceptance. No hard corruption: every session ran coherent multi-turn tool
work; the 08-17 failure signature (token soup, sessions dead at turn 1) did
not reproduce. Two real caveats remain. (a) **One of 24 sessions degenerated
into a repetition loop** (198K chars at a 0.03 unique-word ratio) -- a quality
flag that needs a spec-off control of the same run shape before vLLM+MTP is
cleared for serving. (b) Task completion (4/12 no-think vs the 08-17 spec-off
baseline's 9/12 on Qwen3.6) is dominated by a context wall, not generation:
Claude Code requests `max_tokens=64000` as a ceiling, vLLM validates
`prompt + max_tokens <= max_model_len`, so at `--max-model-len 106496` every
session 400s the moment its prompt crosses ~42.5K tokens. q27 and ninfer clamp
instead of rejecting. Raw session records in
`agentic.vllm38specON.*.jsonl` in this directory.

Two more walls hit standing this up: the Qwen3.8 chat template rejects Claude
Code's `reasoning effort: high` (it accepts only xhigh/medium/low -- patch the
template to alias `high`), and Claude Code's thinking is disabled for a
baseline-comparable run via the container env `MAX_THINKING_TOKENS=0`.

## Gotchas paid for

- **`min_p` silently kills vLLM streaming under spec-decode.** The server
  warns "min_p ... won't work with speculative decoding" and then streams
  **zero content chunks** (non-streaming requests still work). Drop `min_p`
  for any vLLM spec leg; that is the one sampler deviation in this run.
- The 3.8 vLLM artifact resolves as `Qwen3_5ForConditionalGeneration` (a VL
  arch); the `min_frames`/`max_frames` `[ERROR]` lines at startup are benign
  transformers docstring warnings, not failures.
- ninfer has a working 3.8 converter (`tools/convert/qwen3_8_27b`, CUDA,
  ~72 s for 18.2 GB).
- vLLM startup needs the real free-VRAM budget: a 6 GB TTS job sharing the
  card fails `gpu-memory-utilization 0.93` at init.
