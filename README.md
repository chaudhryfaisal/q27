# q27

A narrow inference engine for **Qwopus3.6-27B-v2-MTP** (Qwen3.6-27B hybrid + trained-in MTP heads) on a single RTX 5090. One model, one GPU, as fast as possible. In the spirit of [antirez/ds4](https://github.com/antirez/ds4).

## Why this model is a good target

- Dense-ish 27B that fits entirely in 32 GB VRAM at 4-bit -- no expert offload, no DRAM scatter, none of the DSV4 pain
- MTP draft head trained into the checkpoint: self-speculation without a separate draft model
- Hybrid Gated-DeltaNet architecture means near-O(1) memory per token for 48 of 65 layers. KV lives only in the 17 full-attention layers (16 + MTP, all **global**, no windowing): 68 KB/token at fp16 = ~4.3 GB @64K, ~8.5 GB @128K, ~17.8 GB @256K. A dense-attention 65-layer build would be ~68 GB @256K. The advertised 262K native does NOT fit on this card at fp16 alongside 16.75 GB of weights -- practical allocation ceiling is ~180K (fp8 KV, planned, would double that); correctness is only validated to 8K so far (see risk 5)
- Measured baseline to beat: llama.cpp (MTP-TurboQuant fork) at 106-127 t/s single-stream

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

## Performance model

5090 GDDR7 ~1.79 TB/s. Single-stream decode is weight-read-bound.

| Stage | Per-step read | Ceiling | With MTP (~1.9x measured) |
|---|---|---|---|
| llama.cpp Q5_K_M (62% BW eff) | 18.2 GB | 61.6 t/s measured | 106-127 t/s measured |
| q27 Q5-class, 85-90% eff | ~18 GB | ~88 t/s | ~165 |
| q27 custom 4-bit at 85-90% eff | ~14.8 GB | ~103-109 t/s | ~200-225 |
| q27 4-bit **measured** (2026-07-02, +4000 OC) | ~15.5 GB/step | **91.0 t/s plain** (~75% eff) | **188.9** (depth-3 spec, 2.07x) |

The original "~120 t/s ceiling" row implied ~99% BW efficiency and is retired.
Plain decode sits ~15% under the honest 85-90% ceiling; that tail is GDN
recurrence + ~140 small-kernel launches/token, and three attempts on it
(E4 launch geometry, E5 fusions, cp.async) all came back negative.

## Design decisions

- **Weights**: custom 4-bit symmetric groupwise (group 64, fp16 scales), packed for coalesced 128B warp loads, dequant fused into GEMV. Embeddings, lm_head, MTP layer, norms at 8-bit/f32. Repacked offline from the BF16 GGUF.
- **KV cache**: fp16 for the 17 attention layers (implemented; f32 originally). FP8 E4M3 is planned, not done -- it halves KV capacity cost again and cuts long-ctx decode bandwidth. DeltaNet recurrent state is tiny and stays f32.
- **MTP**: first-class. Draft + verify in one pipeline under a single CUDA graph. No separate draft context, no re-prefill.
- **Stack**: plain CUDA C++. No CUTLASS, no deps beyond CUDA runtime. Offline repack tool is Python (runs once).
- **Serving**: OpenAI, Anthropic (Claude Code-grade), and OpenAI Responses (Codex-grade) shapes on one binary.

## Milestones

- **M0** DONE -- repack tool: BF16 GGUF -> q27 4-bit format (policy v1.2)
- **M1** DONE -- correctness: greedy decode, output verified vs llama.cpp
- **M2** DONE -- dp4a GEMVs + CUDA-graph decode: 80.1 t/s plain
- **M3** DONE -- MTP speculative pipeline, lossless (token-identical):
  depth-2 drafting, batched verify, 3-perm cyclic state graphs. **146.0 t/s**
  (llama.cpp MTP fork on same model/GPU: 101.5). Stretch target was 165;
  verify-GEMV bandwidth floor makes the remaining gap ~1-2%/iteration work.
- **M4** DONE -- dual lm_head (Q4 draft / Q8 verify), grid merges, device-side
  round bookkeeping. `--fast-head` opt-in: **156.5 t/s**
- **M5** DONE -- HTTP serving: OpenAI + Anthropic + OpenAI Responses, exact
  byte-level BPE tokenizer (gated 21/21 vs llama-tokenize), tool calling
- **E6** DONE -- ungated depth-3 speculation: measured p(d3 | d1,d2 correct)
  = 83.7% offline (docs/E6-design.md), so the round always drafts 3 and
  batch-4-verifies {pending, d1, d2, d3}. 4 GDN buffers under a mod-4 role
  permutation, 4 captured graphs. 3.12 tok/round, **188.9 t/s** @2k
  (204.8 long-gen); 8000-token output bit-identical to depth-2. Also fixed
  two latent bugs found en route: flash-decode scratch under-allocation at
  ctx<4128, and missing ctx guard letting spec rounds write KV rows past
  max_ctx (silent corruption the prefix cache could then reuse).

## Serving

```
make build/q27-server
./build/q27-server model.q27 model.tok --port 8080 --ctx 8192 [--fast-head]
```

Three API shapes on one server:
- **OpenAI**: `/v1/chat/completions`, `/v1/completions` (text)
- **Anthropic**: `/v1/messages` -- native Messages API with thinking blocks
  (Qwopus `<think>` mapped to thinking/signature blocks), tool_use/tool_result,
  input_json_delta streaming. Claude Code-compatible:
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
(src/stream_split.h) that also routes `<think>`. Single slot (spec decode is
single-stream), greedy sampling. `--fast-head` trades output exactness for
~7% more t/s.

## Progress log (tg t/s, greedy, token-identical output verified each step)

| change | t/s |
|---|---|
| reference kernels e2e | 43.4 |
| dp4a int8-activation GEMVs | 58.8 |
| coalesced delta state + wide norms + multiblock argmax | 66.5 |
| CUDA-graph token replay, device-chained decode | 75.9 |
| delta_step i-parallel v2 | 80.1 |
| + speculative decode depth-1 (host-driven) | 84.2 |
| + direct-write batched GEMV | 92.2 |
| + parity-pair captured graphs | 109.3 |
| + depth-2 drafting (2.13 tok/round) | 107.3 |
| + grid-merged 3-token small kernels | 115.1 |
| + dual lm_head: Q4 drafts, Q8 verify (v1.3 repack) | 121.1 |
| steady state (128-token bench, 2.39 tok/round) | **133.5** |
| `--fast-head` opt-in (Q4 verify; output differs, coherent) | 143.0 |
| + full grid merges (l2/f16/gates/rope/kv/attn/sigmoid/embed x3) | 145.8 lossless / 156.5 fast |
| + device-side round bookkeeping (1 sync + 16B readback/round) | 146.0 lossless / 156.5 fast |
| E1: display compositor off GPU 0 (cosmic-comp/Xwayland stole ~10%) | **157.4** lossless / **168.5** fast |
| warp-cooperative decode attention (coalesced K/V) | **168.6** lossless @2k; 65.8 @8k ctx (~2x long-ctx) |
| flash-decode (split-K, K/V shared across GQA heads) | **173.1** @2k / **159.6** @8k ctx lossless; 178.1 fast |
| fp16 KV cache (attn + MTP) | 169.7 @2k / 159.7 @8k; halves KV bytes, -2.1GB @32k ctx |
| E2: GDDR7 mem offset +4000 (tools/mem_oc.py, volatile) | **176.6** lossless / **185** fast-head; prefill ~+6% |
| E6: ungated depth-3 speculation (3.12 tok/round; batch-4 verify) | **188.9** @2k (128-tok) / **204.8** long-gen; 8000-token output bit-identical to depth-2 |

Headline numbers from E2 onward include the +4000 GDDR7 offset (~+4%; stock
depth-3 ~181 est. from the E2 ratio). Caveat: consumer GDDR7 has no ECC, and
weights load once -- a bit flipped by a marginal OC during a long session is a
persistent silent error the token-identity gates cannot catch (they compare
against the same resident state). The +5000 512-token stock-identical soak was
a point-in-time check, not a long-session guarantee. For unattended multi-hour
serving, stock clocks are the conservative choice.

## Prefill (M6)

Batched prefill: 256-token chunks, smem-staged dp4a GEMM (16 rows/block share
one activation tile; per-lane accumulation order matches the serial GEMV
exactly, so prefill is bitwise-identical to the serial path -- gated on
identical continuations). GDN state scans sequentially inside one kernel with
S resident in shared memory; attention runs two-pass softmax in 32-token
sub-batches; MTP warm skips attention/FFN (only the K/V stores matter).

| prompt | serial | batched | speedup |
|---|---|---|---|
| 512 | 76 t/s | 567 t/s | 7.5x |
| 4096 | 53 t/s | 453 t/s | 8.5x |

**Prefix cache (M6.5)**: GDN state + conv rings snapshotted after prefill
(attention/MTP KV rows are append-only, so prefix rows stay valid); next
request LCP-matches the snapshot and prefills only the suffix. Claude Code
turn 2 on a 26.7k-token context: **1.3s** (26,670/26,693 tokens reused).
Unconditionally correct: any mismatch falls back to full prefill; warm-vs-cold
continuations gated identical.

Real-world (Claude Code `claude -p`, 26.7k-token system prompt):
| | TTFT |
|---|---|
| pre-M6 (serial prefill) | 15-min timeout, 0 tokens |
| M6 (batched) | 139s |
| + coalesced attention prefill | 90s |
| + GEMM tuning + FA-lite attention | 61s |
| turn 2+ with prefix cache | **1.3s** |

## Backlog

1. Prefill GEMM tuning: double-buffered staging, TB=48/64 via dynamic smem
   (fork reference: ~2,300-2,400 t/s; q27 at 583 @512 / 528 @4k / ~300 @26k)
2. FA-lite tiled attention prefill (smem K/V tiles shared across the 32-token
   sub-batch; kills the remaining quadratic degradation at long contexts)
3. Decode queue from the nsys research plan: E2 mem OC (user gate), E3
   instrumentation pack, E4 per-shape GEMV tuning (+6-8%), E5 grid-dim fusions
   (+4-6%), E6 confidence-gated depth-3, E7 draft-head diet (ceiling ~187)
4. Tensor-core prefill path (parity+ with the fork's cuBLAS GEMM prefill)

## Risk register

1. **Gated DeltaNet decode kernel** is the new risk center (was "simple dense" until we read the GGUF). llama.cpp's implementation is the semantic reference; validate per-layer.
2. 4-bit quality on a 27B: keep sensitive tensors high-bit, add importance-weighted scaling if PPL regresses > ~3% vs Q5_K_M. **STATUS: the PPL delta has never actually been measured** -- the threshold is unbacked, and the t/s win over Q5_K_M partly comes from reading 14.8 GB instead of 18.2, so the speed comparison is only honest next to a quality number. OPEN.
3. M-RoPE sections must match exactly or long-context quality silently degrades.
4. MTP acceptance rate must survive quantization (draft and verify disagreeing more = less speedup). STATUS: measured -- Q4 vs Q8 draft-head argmax agreement 98.1% (E3); depth-3 runtime acceptance 85.7%.
5. **Long-context correctness is untested.** All token-identity gates run at 2-8K. Risk 3 (M-RoPE) is precisely the failure mode that is correct at 2K and silently degraded at 128K. Needed before any context number is advertised as fact: needle/consistency check vs llama.cpp at >=64K. OPEN.
6. **Tensor-core prefill will break the bitwise gate.** Batched prefill is currently bit-identical to serial because the dp4a GEMM matches the serial per-lane accumulation order exactly. fp16/fp8 MMA changes accumulation order, so the identical-continuations gate stops working for that path. A tolerance gate (logit cosine / top-k agreement vs serial) must exist BEFORE that work starts, or the hardest kernel has no regression signal. Related fork-in-the-road: "plain CUDA, no CUTLASS" vs "parity with cuBLAS prefill (~2,300 t/s)" are in tension -- hand-rolled sm_120 wgmma at cuBLAS parity is CUTLASS-grade effort. Options: (a) scoped CUTLASS/cuBLASLt dep for the prefill GEMM only, (b) hand-rolled TC GEMM accepting less than parity, (c) keep dp4a prefill and lean on the prefix cache (cold first turn stays ~60s @26K; warm turns are already 1.3s). Decision pending.
