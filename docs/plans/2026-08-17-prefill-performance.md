# Prefill performance: where the 2.9x actually lives (2026-08-17)

Status: PLAN. No code. Supersedes the "prefill" framing in
docs/plans/2026-08-15-ninfer-steals.md phase 2 (that phase's NO-GO stands;
this doc says what is left after it).

## 1. The measured gap, and the control that interprets it

Single harness (club-3090 `bench_ab.sh`), our silicon, qwen3.6-27b both
sides, cache-busted fresh haystack per run
(`club-3090/results/ninfer-vs-q27-20260815`). Prompt-processing throughput:

| engine / format | 10K depth | 90K depth |
|---|--:|--:|
| q27 q4s | 3,537 tok/s | 2,643 tok/s |
| ninfer **nvfp4** | 10,418 (2.94x) | 5,177 (1.96x) |
| ninfer **int** | 3,051 (**0.86x**) | 2,367 (**0.90x**) |

The third row is the whole argument. Running the SAME engine on an integer
format, ninfer is SLOWER than q27 at both depths. The 2.9x is not engine
quality, not kernel maturity, not a vendor library: it is the weight format.
Within ninfer, nvfp4 vs int is 3.41x at 10K.

Second reading: the gap HALVES with depth (2.94x -> 1.96x). That is the
O(N^2) attention share growing, and no weight format touches attention. At
90K the two engines are mostly being compared on their attention kernels.

## 2. What the numbers RULE OUT (read this before proposing anything)

**(a) Batching prefill across sequences to raise M -- DEAD.** The obvious
idea after M3a (8 slots; concurrent prefills are gate-serialized, so
concatenating their chunks would raise M from 1024 and amortize weight
traffic, the P2a decode physics) buys NOTHING, because the GEMM is already
at its plateau. Measured 2026-08-17, `build/microbench_mxf4` on real q4s
projection weights:

| shape | M=512 | M=1024 | M=2048 | M=8192 |
|---|--:|--:|--:|--:|
| attn_q+gate | 280.3 | 313.3 | 314.9 | 322.0 |
| attn_output | 276.0 | 308.4 | 303.4 | 321.0 |
| ffn_gate | 319.5 | 319.7 | 319.6 | 318.4 |
| ffn_down | 283.5 | 315.5 | 292.5 | 302.7 |

TFLOPS, `q27k::gemm_q4_T`. **Flat from M=512 to M=8192** -- a 16x increase in
M moves throughput by ~2%. Prefill's PF_T=1024 chunk is already past the
efficiency knee. Do not build union-prefill for GEMM efficiency; there is no
GEMM efficiency to gain.

**(b) "ninfer is simply a better engine" -- DEAD.** See the int row above.

**(c) Dual-copy fp4 -- DEAD.** 31.8 GiB peak at ctx 2048 (phase 2). Any fp4
road must be a NATIVE single-copy tier.

**(d) Chunked PPL as the quality instrument -- DEAD, and actively
dangerous.** Chunked wiki PPL IMPROVED under fp4 (e2m1's 12:1 per-16 range
zeroing sub-amax/24 activations = sparsification, a regularizer on smooth
text and a wrecker on structured content). Chunked PPL alone would have
shipped a format that costs 561 catastrophic positions. Long single-pass NLL
is the instrument; this is the same lesson the KV axis taught on 2026-08-01,
from the other direction.

## 3. Decomposition: where q27's prefill wall actually goes

Derived from the phase-2 result, not directly profiled (flagged as derived).
The fp4 swap delivered 2.5x on its include set and moved the wall 1.237x at
17K. Solving `1/((1-f) + f/2.5) = 1.237` gives f ~= 0.32: the included GEMMs
are **~32% of the prefill wall**. The include set is 77% of GEMM flops
(GDN-input projections excluded on the ssm_out cancellation lesson), so ALL
projection GEMM is **~42%**, leaving **~58% in attention + delta-scan +
staging** at 17K -- and attention's share grows with N (measured 54% of
prefill at 128K, BUILDLOG 2026-07-07).

So: even a free, perfect GEMM caps out around 1.7x at 17K and less at depth.
That is why the format alone was never going to reproduce 2.9x here.

## 4. The three real levers

### Lever 1 -- prefill attention occupancy. Biggest, format-independent, no quality price.

`k_attn_prefill_mma` is latency-starved, not bandwidth or tensor bound
(docs/perf-attribution-prefill-attn.md, ncu):

| metric | value | reading |
|---|--:|---|
| achieved occupancy | **12.5%** (1 CTA/SM) | dual-limited: 248 regs/thread AND 84.48 KB smem |
| issue slots busy | **9.9%** | ~90% of issue slots EMPTY |
| DRAM throughput | 1.98% | nowhere near bandwidth |
| L2 hit | 95.6% | KV is L2-resident at depth |
| tensor pipe active | 33-35% | ncu: "should not be a bottleneck" |

This is the single largest addressable block in prefill and it costs nothing
in numerics. Getting 2 CTAs/SM needs BOTH a register cut (the `o[32][4]`
O-accumulator is 128 registers on its own) and smem halving -- the
from-the-smem-layout rewrite that blocked the fp8-MMA phase in 2026-07.
Staging Q as fp8 to free smem is the known candidate.

### Lever 2 -- the int GEMM's own headroom. Quality-free, bounded.

The int path plateaus at ~310-322 TFLOPS; the fp4 kernel on the same shapes
reaches 785-868. On Blackwell the fp4 dense tensor peak is 2x the fp8/int8
peak, so roughly **2x of that 2.5x is silicon, and the remainder (~25%) is
kernel headroom in `gemm_q4_T`** -- the fp4 kernel is a modern shape (128x128x256
tiles, cp.async double buffer, ldmatrix over swizzled smem, 8 warps) and the
int kernel is not. FIRST ACTION is a lookup, not a build: confirm the card's
dense int8/fp8 tensor peak, divide 310 by it. If the int path is at ~75% of
peak, this lever is worth ~1.3x on 42% of the wall (~1.1x overall) and is
probably not worth a rewrite on its own -- but the techniques are already
written and validated in `tools/microbench_mxf4.cu`, so the port is cheap.

### Lever 3 -- the format. The real 2x, and it carries a bill.

Only viable shape: a NATIVE fp4 tier -- single weight copy, fp4 GEMV for
decode, its own canonical md5. This is a tier project (repack + gates +
anchors), not a kernel patch. Prerequisite experiment, cheap and
informative: the **W4A8 attribution arm** (fp4 weights, int8 activations) to
split weight-grid damage from activation-sparsification damage. If the
damage is mostly activation-side, a W4-only variant could keep most of the
speed at a fraction of the quality cost. Phase 2 measured the combined
figure only: +3.734% dPPL / 561 catastrophic vs fp8-KV's +0.458% / 19.

## 5. Staging and bars

- **P0 (hours, no code):** confirm the int8/fp8 dense peak and compute the
  int GEMM's percent-of-peak. Decides whether Lever 2 exists. Also re-run
  `build/microbench_mxf4` if the driver moved.
- **P1 (days-to-week):** Lever 1, attention occupancy. Bar: 2 CTAs/SM
  achieved (ncu), prefill wall >= 1.25x at 17K-27K, and `--pf N` output
  IDENTICAL to the serial path (the canonical md5 gate does NOT cover this
  kernel -- NP=5 takes the serial prefill; use the `--pf 200 seq+32`
  identity pattern). No tolerance class: this is a pure scheduling change.
- **P2 (days, only if P0 says the headroom is real):** Lever 2, port the
  microbench's kernel shape into `gemm_q4_T`. Bar: >= 1.2x on the M=1024
  projection shapes, bitwise-identical output (int8 MMA reassociation must
  not move -- N-invariance is a shipped gate, `tools/ninv_test.cu`).
- **P3 (weeks, separate spec):** Lever 3, native fp4 tier, gated on the
  W4A8 attribution arm coming back favourable. Bars as phase 2, plus a
  single-copy VRAM bar and its own canonical anchor.

## 6. Does this matter? The honest ROI note

Prefill is the compute-bound half and agentic traffic re-prefills on every
cache miss -- but the P16 prefix cache turns the common case into a 1.20 s
restore instead of an 8.15 s re-prefill on a 26,700-token prompt. The real
exposure is cache MISSES, so the value of this work scales with miss rate,
not with prefill share. Before committing P1's week, measure the miss rate on
real traffic; if it is low, the whole plan is a lower priority than the
Qwen3.8 quality regression, and that is a legitimate outcome for this doc.

The decode side is already the strong half (q27 leads the same A/B's
narrative/code legs on wall, and the batched-decode arc closed at
406-425 t/s aggregate). Nothing here is needed to defend that.
