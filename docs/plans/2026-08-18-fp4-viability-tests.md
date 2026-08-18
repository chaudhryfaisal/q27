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

> **RAN 2026-08-17. First read NO-GO at 347; that verdict is RETRACTED
> 2026-08-18.** The bar below is the fp8-KV envelope, which prices an 8-bit KV
> CACHE format -- and q27's own shipped q4s tier scores **455** on this same
> instrument, worse than fp4's 347 and worse than ninfer's 317. Against the only
> like-for-like control, the 4-bit weight format we actually ship, **fp4 is
> BETTER on every unsigned divergence metric.** The measurements below stand;
> the NO-GO does not, and T2/T3 are open again on their original speed and
> memory terms. Full retraction: BUILDLOG 2026-08-18 (b).
>
> The run record is BUILDLOG 2026-08-17. Three things this plan did not
> anticipate, kept because they change how the next fp4 question gets asked:
>
> 1. **The codec was miscalibrated, and the first answer was a false kill.**
>    `quant_nvfp4` -- and so every `--pf4` sidecar, and so the W4A4 arm this
>    plan is built on -- stores the ue4m3 block scale in ABSOLUTE units with no
>    per-tensor global scale, which puts ~100% of this checkpoint's blocks in
>    ue4m3's subnormal region. Canonical two-level NVFP4 cuts rel RMSE
>    0.1274 -> 0.0950 (oracle 0.0941) and moved the result 442 -> 347. The
>    verdict survived; it would not have been honest without the correction.
> 2. **e2m1 and Q4_G64 put their error in different places.** fp4 beats Q4_G64
>    on RMSE and zeroes half as many weights, yet is 2.06x worse on the top 1%
>    by magnitude: e2m1 spends resolution near zero and leaves gaps at the top.
>    Real and measured -- but it does NOT dominate, which is why fp4 still wins
>    the aggregate against q4s. This was written up as the mechanism of a kill;
>    it is a difference in error placement, not a disqualification.
> 3. **The decomposition premise below is false.** W4A4 is not weight damage
>    plus activation damage -- fp4 activations REPAIR positions fp4 weights
>    alone break. Quote the envelope ratio, not the "% of 561".
>
> Also corrected against the code, for anyone re-running this: `--nll-max` does
> NOT cap the `--nll-long` path (it gates the chunked path only,
> src/engine.cu:781); cap with a smaller `--nll-long`, which is min'd against
> `--ctx` at src/engine.cu:718. The commands below also omit the required
> `--nll <corpus>` and `--ctx`, and the corpus matters: both reference arms used
> `scratchpad/t3_quality/corpus/agentic_req0031.i32`, not wiki, which is the
> text where e2m1's clipping fakes a win.

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

> **Resolved, and the last sentence above is wrong.** q8 does not fit at any
> useful ctx (OOM on the 972 MiB logits buffer at 24576). More importantly the
> differential does NOT hold either way: Q8_G128 is transparent to the fp4 grid
> (+0.0002 rel RMSE) but Q4_G64 adds in quadrature, so a coarse container makes
> leg B "q4 of fp4" -- strictly worse than fp4, biasing a kill test toward a
> false kill, and it damages the reference leg that the `ref NLL < 0.1`
> confident mask is measured against. What ran: a purpose-built container with
> the five include-list projections at Q8_G128 and everything else q4s-like
> (`--q4-head --q8 '(attn_q|attn_output|ffn_gate|ffn_up|ffn_down)\.'`,
> 24.48 GB), full 65536 single pass, fp16 KV, no concessions.

**Pre-declared bar.** Weight-grid-only damage measured against the fp8-KV
envelope (+0.458% dPPL, ~19 catastrophic positions at 64K):

> **This bar is wrong and was the root cause of the retracted NO-GO.** It prices
> an 8-bit KV CACHE format; no 4-bit WEIGHT format of any shape lands near 19,
> and q27's shipped q4s measures 455 on it. A weight-format bar has to be a tier
> the repo is willing to ship. Do not re-run T2 against the number 19.

| outcome | reading | consequence |
|---|---|---|
| <= ~1.5x the envelope | grid is clean | activations are the problem; W4A8 becomes the design and T2 decides |
| >= ~50% of W4A4's 561 | grid is the problem | **every fp4 tier dies here**; stop, record, do not build T2 or T3 |
| between | mixed | both paths need work; re-scope before spending more |

**Cost.** One repack plus two NLL runs. No kernel work. Half a day.

---

## T2 -- Does fp4 actually win at DECODE shapes?

> **NOT RUN, and now UNBLOCKED.** It was skipped on T1's NO-GO, which is
> retracted -- fp4 measures better than the 4-bit weight format q27 ships, so
> the quality objection to a native fp4 tier is gone and this test is live again
> on its original terms. Before re-running it: the >= 1.25x wall bar below is
> fine, but do NOT carry over the number 19 as a quality envelope (see T1).

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

> **NOT RUN.** Was gated on T1, whose NO-GO is retracted; the VRAM arithmetic
> that motivated it (one fp4 copy at ~10-11 GB against q4s's 15.46) is unchanged
> and now has no quality objection standing against it.
>
> One thing worth carrying forward if the 790 t/s ever gets revisited: ninfer
> reaches it on NVFP4 and scores a quality TIE in the 2026-08-17 four-engine
> run, which T1 says should not be possible with e2m1 weights. Either their
> codec differs from what q27 measured, or a task-scored benchmark is blind to
> the failure mode a per-position instrument sees. That is a question about
> THEIR stack and the benchmark's sensitivity, not a reason to re-open this
> plan, and it should be settled by instrumenting ninfer rather than by
> building a q27 fp4 tier on the hope.
>
> **SETTLED 2026-08-18: their actual fp4 codes, scored through this same
> instrument, read 317** -- better than q27's own q4s at 455 on the identical
> reference, so the tie is two ordinary 4-bit weight quants being
> indistinguishable to a task rubric, not a blind benchmark (BUILDLOG
> 2026-08-18 (b)). Their codec is the same two-level
> NVFP4 with a bit-identical global scale; they do not quantize at all but
> consume a calibrated third-party artifact (HALO/GPTQ/block-output match, plus
> BF16 carve-outs on 8 of q27's 224 include-list tensors), and all of that buys
> **347 -> 317**. A tier carrying 317 catastrophic positions scores 65/75 on a
> task rubric. That GPTQ specifically -- the method that trades per-weight error
> for block-output fidelity -- fails to rescue it is the strongest evidence for
> T1's mechanism: the constraint is the grid's shape, and no encoder can place a
> code where the grid has none. 317 is a LOWER bound; they also run W4A4 at
> batch and fp4 the GDN path this include list excludes.

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
