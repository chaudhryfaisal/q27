# fp4 viability: three tests, cheapest kill first

The 2026-08-15 W4A4 prefill arm was a NO-GO on all three bars (quality
+3.734% dPPL / 561 catastrophic, wall 1.19-1.24x against a 1.25x bar,
dual-copy VRAM 31.8/32.6 GiB). The 2026-08-17 four-engine run then measured
ninfer's NVFP4 tier at **790 t/s aggregate at C=8 against q27's 412.9**, and
established it as a format effect rather than a batching one -- their own int8
tier peaks *below* q27.

Those are not the same question. The NO-GO was about **prefill** quality and
VRAM. The 1.9x is **decode throughput at batch width**, where the fp4 MMA
engages and the economics differ. This plan is the smallest set of measurements
that decides whether any fp4 tier is worth building, ordered so the cheapest
test can kill the whole idea.

## Instrument discipline (read before running anything)

Every quality number here comes from `--nll-long` with `--nll-dump`, never from
`--nll-chunk` alone.

The phase-2 arm banked the reason: chunked wiki PPL **improved** under fp4
(7.77 -> 7.56, reproducible) while the long single-pass instrument showed the
damage. e2m1's 12:1 per-16 range clips activation outliers, which regularizes
smooth prose and wrecks structured text. Chunked PPL alone would have shipped a
format that produces 561 catastrophic positions. A bucket mean over a
high-confidence echo region dilutes a rare catastrophic position to nothing;
the per-position dump is the only instrument that sees it.

---

## T1 -- Is the fp4 weight grid survivable on its own?

**The question.** W4A4 quantizes weights *and* activations. The 561
catastrophic positions could come from either, and the two have completely
different consequences: bad weights kill every fp4 tier permanently, bad
activations leave fp4 weights viable with a W4A8 design.

Nothing measured so far separates them. This is the single highest-value
unknown, and every other fp4 question is downstream of it.

**Method, and why it needs no new kernel.** The obvious approach is to build a
W4A8 kernel, which is real work on the sm_120a MMA path. Skip it: round the
weights to the e2m1 grid (with the ue4m3 per-16 scale) and *store the rounded
values in an existing container tier*. The engine then runs its normal path at
full activation precision, so any damage is the weight grid alone.

A `--fp4-round` flag on `tools/repack.py` applied before the existing
quantizer. Then a differential against the same container unmodified, which
cancels the container's own error:

```bash
# A: container baseline (already published)
build/q27 /mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp-q8.q27 \
  --nll-long 65536 --nll-dump /tmp/base.nll

# B: same container, weights pre-rounded to the fp4 grid
tools/repack.py <bf16.gguf> /tmp/q8-fp4round.q27 --fp4-round
build/q27 /tmp/q8-fp4round.q27 --nll-long 65536 --nll-dump /tmp/fp4w.nll
```

Container choice is the one decision. q8 (28.4 GB) is the cleanest reference
but leaves little room for a 64K single-pass; drop to q6k or cap with
`--nll-max` if it does not fit. The differential holds either way -- both legs
carry the same container error.

**Pre-declared bar.** Weight-grid-only damage measured against the fp8-KV
envelope (+0.458% dPPL, ~19 catastrophic positions at 64K):

| outcome | reading | consequence |
|---|---|---|
| <= ~1.5x the envelope | grid is clean | activations are the problem; W4A8 becomes the design and T2 decides |
| >= ~50% of W4A4's 561 | grid is the problem | **every fp4 tier dies here**; stop, record, do not build T2 or T3 |
| between | mixed | both paths need work; re-scope before spending more |

**Cost.** One repack plus two NLL runs. No kernel work. Half a day.

---

## T2 -- Does fp4 actually win at DECODE shapes?

**The question.** Phase 0 cleared the GEMM gate at 2.13-2.97x, but it swept
M = 512..8192 -- prefill chunk widths. Batched decode at C=8 runs
M = sum of union widths, roughly **8-64 rows**. That is a different regime, and
fp4 MMA needs tokens >= 4-8 to engage at all. The 790 t/s number says ninfer
wins there; nothing here says q27's kernel would.

`tools/microbench_mxf4.cu` is tiled `BM=128`, built for the prefill point. At
M=8-64 that tile is mostly empty, so the existing binary cannot answer this
without a decode-shaped tile.

**Method.** Extend the microbench's sweep down to M = 4, 8, 16, 32, 64 with a
tile sized for it, and compare against the path those shapes take today
(`gemm_q4_T` / the union vgemm at k>=3, whichever the C-sweep verdict routes
to). Same weights, same K, same clocks.

```bash
make build/microbench_mxf4       # sm_120a-only target
build/microbench_mxf4 <q4s artifact>   # after the M-sweep extension
```

Take the real M distribution from a live C=8 ladder run rather than guessing it
-- `Q27_BATCH_DBG=1` prints the per-round union widths, and the C-sweep already
showed 88% of lanes sitting at floor-2 from C=6, so the distribution is
narrower than the nominal 8x16 suggests.

**Pre-declared bar.** >= **1.25x** over the current union GEMM at the measured
C=8 M distribution -- the same wall bar phase 2 was held to. Below that,
integration cannot pay for itself and the 790 t/s stays a ninfer property.

Report percent-of-peak alongside the ratio: fp4's dense peak is 2x fp8/int8 on
Blackwell, so roughly 2x of any win is silicon rather than kernel, and the
honest claim is the residual.

**Cost.** Microbenchmark only, no engine integration, no quality question.
A day, and it is independent of T1 -- run them in parallel if convenient.

---

## T3 -- Can ONE fp4 copy serve both phases?

Only if T1 and T2 both pass.

**The question.** The phase-2 arm needed fp4 weights for prefill *and* q4s
weights for decode, peaking at 31.8 of 32.6 GiB. That third bar failed on
arithmetic, not quality. ninfer avoids it by keeping one fp4 copy and running
W4A4 prefill plus W4A16 GEMV decode against the same bytes.

**Method.** Arithmetic first, then a load test -- this is engineering, not a
study. fp4 weights are ~10-11 GB against q4s's 15.46, so a single copy is
*cheaper* than what ships today. The real work is a decode path that reads the
fp4 container directly, which is exactly what T2 prices.

**Pre-declared bars.**
- Single-copy resident VRAM leaves >= the current q4s slot budget at 8 slots /
  16K, i.e. no regression in admitted slots.
- Decode quality from fp4 weights within the T1 envelope (same instrument).
- Canonical digest for the new tier derived and published before it is
  recommended, per the repo's standing rule.

**Cost.** Weeks, and only worth starting with T1 and T2 already green.

---

## What this plan explicitly refuses to do

Re-run the phase-2 W4A4 prefill arm hoping for a different answer. It failed
three bars with measurements, the code is in-tree and inert behind
`Q27_PREFILL=fp4`, and nothing since has changed the prefill economics --
`gemm_q4_T` being flat 280-322 TFLOPS from M=512 to M=8192 already killed the
follow-on idea of batching prefill across sequences to raise M.

The open question is decode at batch width. T1 and T2 answer it for roughly a
day and a half of work, and T1 alone can end it.
