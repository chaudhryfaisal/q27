# Verify weight path: flat-in-W MMA GEMM (`k_vgemm`), and the W16 reopen it unlocks

Synthesized 2026-07-13 from a 3-design / 3-adversarial-check workflow (designs biased
minimal-blast-radius / peak-perf / kernel-unification; each attacked by an independent
reader who re-measured on the box). Winner: the **perf** spine (`MR=32` warp-split tile),
with the numerics discipline of **minimal** and one framing steal from **unify**. Every
number below was **re-measured this session on the SERVED model** -- both design B and
design C benched the wrong container, and their headline arithmetic is void. Spike:
`/tmp/claude-1000/-home-gabe/cc438e33-d0d0-44d8-aabd-96a0f34ec18c/scratchpad/mmaC.cu`
(mmaB.cu + the two OOB/trim fixes + a corrected round replay).

---

## 0. Model correction, first, because everything downstream depends on it

The served model -- the one `tools/width_curve.sh:16` uses, the one the 07-13 W16 A/B and
the retier A/B were measured on -- is **`/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.q27`,
`quant_policy: v1.4`, `q8_extra: "(ssm_out|attn_output)\."`** (verified with `build/inspect`).
Not v1.3. Not `qwopus-27b-mtp-v13.q27`.

Consequences, all load-bearing:

| weight | shape | dtype | calls/round | bytes/round |
|---|---|---|---|---|
| `ffn_gate` / `ffn_up` / `ffn_down` | 17408x5120, 5120x17408 | Q4_G64 | 192 | 9.09 GB |
| `attn_qkv` | 10240x5120 | Q4_G64 | 48 | 1.34 GB |
| `attn_gate` | 6144x5120 | Q4_G64 | 48 | 0.80 GB |
| `attn_q` | 12288x5120 | Q4_G64 | 16 | 0.53 GB |
| **`ssm_out`** | 5120x6144 | **Q8_G128** | 48 | **1.53 GB** |
| **`attn_output`** | 5120x6144 | **Q8_G128** | 16 | **0.51 GB** |
| `attn_k` / `attn_v` | 1024x5120 | Q8_G128 | 32 | 0.17 GB |
| `output_q4` (head, `--fast-head`) | 248320x5120 | Q4_G64 | 1 | 0.68 GB |
| | | | **401** | **14.66 GB** |

**305 Q4 / 96 Q8.** The verify GEMM **needs a Q8 leg** -- 64 of those Q8 calls (`ssm_out`,
`attn_output`, rows=5120) are 2.05 GB/round = **14% of the round** and are GEMM-eligible.
Design C's "the verify GEMM never needs a Q8 leg" is false; the "GROUND TRUTH: engine"
dtype table is also wrong (it had the v1.4 policy but attributed Q8 to the wrong tensors).

---

## 1. Honest expectation FIRST

### 1a. What is measured

**Round weight path**, replaying the exact 401-call `mm5` sequence on the served model,
DRAM-cold, deterministic epilogue, reduce nodes timed inside, `attn_k`/`attn_v` left on the
GEMV (ms):

| W | GEMV (shipped, retiered) | **`k_vgemm` MR=32** | delta |
|---|---|---|---|
| 5 | **10.89** | 11.36 | +4.3% |
| 8 | 13.00 | **11.66** | -10.3% |
| 9 | 14.22 | **11.90** | -16.3% |
| 11 | 15.49 | **11.95** | -22.9% |
| **12** | **16.73** | **11.98** | **-28.4%** |
| 13 | 19.28 | **12.09** | -37.3% |
| 14 | 22.53 | **12.22** | -45.8% |
| **16** | **31.21** | **12.53** | **-59.9%** |

The GEMM is **flat**: 11.36 -> 12.53 across a 3.2x width range (+10%). The GEMV is not:
10.89 -> 31.21 (+187%). Per-lane marginal, W=14 to W=16: **GEMV 4.34 ms/lane, GEMM
0.155 ms/lane -- a factor of 28.**

**Non-weight round cost**, by subtraction against the BUILDLOG 07-13 width curve
(post-retier, `W_MAX=16` build, echo @28K):

| W | 5 | 8 | 12 | 13 | 14 | 16 |
|---|---|---|---|---|---|---|
| engine round (measured) | 16.02 | 19.65 | 24.33 | 27.21 | 30.47 | 39.52 |
| GEMV weights (replay) | 10.89 | 13.00 | 16.73 | 19.28 | 22.53 | 31.21 |
| **NonWeight** | **5.13** | **6.65** | **7.60** | **7.93** | **7.94** | **8.31** |

**NonWeight is NOT flat.** `NW ~= 5.0 + 0.21*W` for W >= 8. Designs B and C both claimed it
was flat and both were wrong -- the "flatness" was an artifact of subtracting a v1.3 weight
path from a v1.4 round. But the slope is **0.21 ms/lane against the GEMV's 4.3**, and that
is the whole story.

### 1b. What it predicts

`t/s = 1000 / (round/tok_per_round + 0.197)`. The 0.197 ms/tok of non-round overhead is
derived from the BUILDLOG curve itself (W=12: 2.3015 wall - 2.107 round = 0.1985; W=16:
2.7137 - 2.518 = 0.1957) and reproduces every published t/s to +/-0.3%.

| | round (ms) | tok/rnd | ms/tok | **t/s** | vs 433.4 |
|---|---|---|---|---|---|
| **today, W=12, GEMV** | 24.33 | 11.55 | 2.307 | **433.4** | -- |
| **P2: GEMM, W=12** | 19.58 | 11.55 | 1.892 | **528** | **+21.9%** |
| P3: GEMM, W=13 | 20.02 | 12.40 | 1.812 | 552 | +27.4% |
| P3: GEMM, W=14 | 20.16 | 14.00 | 1.637 | 611 | +41.0% |
| **P3: GEMM, W=16** | 20.84 | 15.87 | 1.510 | **662** | **+52.8%** |

(tok/rnd from the same curve; W>=13 tok/rnd is the ECHO payload's -- see 1d.)

**Engine wall**, by the composition law the retier established empirically
(`1/(1 - s + s*r)`, `s` = suffix share of the DECODE wall, `r` = new/old per-token in
suffix rounds):

| traffic | s | **P2 (GEMM @ W12)** | **P3 (+ W16 cap)** |
|---|---|---|---|
| novel prose (suffix never fires) | ~0% | **+0.0%** | **+0.0%** |
| real CC agentic (T8/T2) | 27-37% | **+5.1% to +7.1%** | **+10.3% to +14.6%** |
| file re-emission / echo | ~100% | **+21.9%** | **+52.8%** |

### 1c. Say the small number plainly

**P2 on its own is +5-7% on the traffic that matters. That does not justify a new kernel.**
The 07-13 GEMV retier bought the same +5-6% agentic for twenty lines and one constant. If
this plan stops at P2 it is a bad trade, and I will say so in the BUILDLOG rather than
quote the 21.9%.

**The reason to build is P3.** The flat weight path is the *only* thing that reopens the
cap, and the cap raise is where the +10-15% lives. Two further caveats, up front:

- **This is greedy-only.** `verify_sample_graph_w[6][W_MAX]` (`engine.cuh:319`) caps at
  W=5 and `spec_round_sampled` (`engine.cuh:1660+`) has **no suffix branch at all**. Under
  `temperature > 0` the suffix drafter never fires and this build is worth exactly **0%**.
  The 27-37% suffix wall share is a greedy-traffic measurement.
- **Novel prose is untouched by construction** (the ladder never reaches the GEMM). The
  224 t/s headline does not move. That is a feature -- it is also why the blast radius is
  small enough to be worth doing.

### 1d. What kills it

If the wide round is not weight-bound, everything above is fiction. **P0 is exactly that
experiment and it runs before a line of CUDA.** The falsifiable prediction: an `nsys` node
histogram of a live W=12 suffix round must show `k_gemv_q4_n` + `k_gemv_q8_n` totalling
**16.7 +/- 0.7 ms of a 24.3 ms round (69%)**. Precedent: the 2026-07-08 Task-0 nsys run
reproduced the P14 weight-stream model within 0.6% on the same instrument.

- **>= 15.5 ms** -> the model holds. Build.
- **12 to 15.5 ms** -> NW is bigger than the subtraction says; P2 shrinks to +2-4% agentic
  and P3 to +6-9%. Re-derive before building; probably still worth it *for P3 only*.
- **< 12 ms** -> the wide round is not weight-bound. **Stop.** The whole design is a
  footnote and the real bottleneck is somewhere in the 12+ ms of non-weight round that
  nobody has profiled at width 12.

---

## 2. The W16 REOPEN -- the actual argument

The 07-13 verdict was: W16 accepts **+33% more tokens per fire** (94% of fires still pin at
16) and **loses 15% t/s anyway** (434.5 -> 372.4), because the round cost grows faster than
the token yield. That verdict was correct **and it was a verdict about `k_gemv_q4_n`, not
about width.**

Reduce it to one ratio. **A cap raise from 12 to W pays iff**

```
tok(W) / tok(12)   >   round(W) / round(12)
```

| | round(16)/round(12) | required accept ratio | echo delivers | verdict |
|---|---|---|---|---|
| **today (GEMV)** | 39.52 / 24.33 = **1.624** | **+62% accepted-per-round** | 15.87/11.55 = **1.374** | **LOSES.** Exactly the measured -15%. |
| **with `k_vgemm`** | 20.84 / 19.58 = **1.064** | **+6.4%** | **1.374** | **WINS, with 5.8x of margin.** |

That single line is the project. The GEMM does not make W16 *better*; it drops the
**break-even accept ratio from 1.62x to 1.064x**, which converts a measured NO-GO into a
near-free bet. On the echo payload W16 clears the bar by 5.8x. On real CC traffic the
measured suffix accept-length is **9.4** (BUILDLOG 2026-07-10, 430 requests, "suffix AL 9.4
on 37%") -- so there the cap only has to lift AL from **9.4 to 10.0** to pay. That is a low
bar, but it is not zero, and it is **P3's gate**, not an assumption.

### What the optimal W becomes

Fit the two measured curves: `weight(W) ~= 10.8 + 0.106*W`, `NW(W) ~= 5.0 + 0.21*W`,
`tok(W) ~= 0.97*W` (cap-pinned on repetitive text). Then

```
ms/tok_wall(W) = (15.8 + 0.32*W) / (0.97*W) + 0.197  =  16.3/W + 0.52
```

Numerator ~flat, denominator linear: **`ms/tok` falls as ~1/W with no turnover.**
W=16 -> 1.55 (645 t/s), W=20 -> 1.34 (746), W=24 -> 1.20 (833), asymptote 0.52 (1900 t/s).

**There is no optimum at 16. Sixteen is a plumbing ceiling, not a minimum.** The binding
walls, all structural:

1. `W_PLUMB = 16` (`cuda_common.h:34`) -- every lane-pointer struct is `p[16]`.
2. `NT = 16` in the mma tile (= W_PLUMB). NT=32 doubles `acc[]` and will cost a CTA tier.
3. `k_attn_fdmma`'s `s_q` holds 16-rounded `6W` rows; at W=16 that is 96 rows / 52.1 KB, so
   **2 CTAs already do not co-reside (BUILDLOG 07-13)** and it runs at 1 CTA/SM. Past 16 it
   gets worse, and `FDMMA_CASE` only emits through 16.

So: **W=16 is the right target now.** W > 16 is a separate project, and its **first** gate
is not a kernel -- it is a real-traffic accept-length histogram showing a tail beyond 16.
With AL measured at 9.4 on live agentic traffic, I would not bet on one.

---

## 3. What's already there (no work)

- **The activation format.** `k_quantize_x3` (`kernels.cu:524-561`) **already writes
  `nat[b*32+lane]` and `scale[b]` on every decode lane, today** -- dead stores the dp4a GEMV
  never reads. The MMA kernel's XG32 leg consumes exactly those. **Zero new quantize
  passes, zero new buffers, zero new graph nodes, zero edits to `qx5`.** This is the single
  best finding in the whole workflow and it holds under measurement.
- **The dispatch hook.** `vw` (`engine.cuh:921`) is a plain host int read **only** inside
  `cudaStreamBeginCapture`. `if (vw >= gemm_min)` inside `mm5` is baked per captured graph.
  No new plumbing, no per-call `getenv`, no device query. Constraint 3 satisfied for free.
- **The bitwise boundary.** `gate_maxd` is hard-clamped to [4,7] (`engine.cuh:1226-1227`)
  -> ladder verify widths are 2..8. The suffix width is captured at exactly one site
  (`engine.cuh:1372`). `gemm_min = 9` touches that one capture and nothing else.
- **The lane-pointer-struct idiom.** `P3`/`CP3`/`XQ3` (`kernels.cuh:67-69`),
  `Q4Lanes`/`Q8Lanes` (`kernels.cu:268-279`) -- `XLanes`/`YLanes` are the same pattern and
  the same `__grid_constant__` cost (measured: none).
- **The W16 plumbing.** `W_PLUMB=16`, `-DQ27_W_MAX=16` build target, the derived spec3.cu
  literals, the fdmma stages/ns auto-selection, `build/q27-server-w16` -- all merged and
  gated on 07-13. P3 is a rebuild and a measurement, not a widening.
- **The warm round** (`engine.cuh:1291`) already executes at `vw = sfx_width()`, so the
  GEMM instantiations are warm before any capture.

---

## 4. What actually changes

### New: `src/vgemm.cuh` / `src/vgemm.cu`

`k_vgemm<MR=32, Q4IN, MODE>` -- a disciplined fork of `k_gemm_mma_T` (`prefill.cu:221-448`),
NOT a refactor of it (see 7.2).

```
MR=32  NT=16  KS=128  KG=2  KB=KG*KS=256  LDW=LDX=272  XGS=32  XSC=4
256 threads = 8 warps = WM(2 row tiles) x WN(2 token halves) x KG(2 K-groups)
each warp: one m16n8k32; the KG partials are summed in smem in FIXED WARP ORDER
grid = (1, ceil(rows/32), z)          // grid.x always 1: T <= 16 == NT
__launch_bounds__(256, 4)
MODE=0: store Y.y[tok][row]           // z == 1: head, deterministic, no workspace
MODE=1: store ws[(z*T+tok)*rows+row]  // z > 1, then k_reduce_z sums i=0..z-1 in order
Q4IN=false leg handles Q8_G128 (nws = MR, WLD doubled) -- 96 of the 401 calls
```

The intra-CTA K-split (KG=2) is what makes this the winning tile: it doubles the row-CTA
supply (MR 64->32) **and** halves the cross-CTA `z` needed, so the deterministic partials
never grow big enough to leave L2. That is why deterministic costs 0.4-0.7 ms here and not
the +38% the earlier audits feared.

### Changed: `src/engine.cuh` (5 edits)

| line | edit |
|---|---|
| `~115` (next to `logits2`) | `float* d_vgemm_ws = nullptr; int gemm_min = 9; int64_t gemm_min_rows = 4096;` |
| `~483` (next to `A((void**)&logits2, ...)`) | `A((void**)&d_vgemm_ws, q27k::vgemm_ws_bytes(model, W_PLUMB));` -- **computed by walking the weight list**, `max over w of z(w)*W_PLUMB*w.rows*4`. Assert at every capture that the shape fits. |
| ctor, **before** `build_spec_graphs` | `q27k::vgemm_init();` (currently a no-op: smem is 13.75 KB < the 48 KB default, so `cudaFuncSetAttribute` is not needed at all -- keep the call + a `static_assert(smem < 48*1024)` so a future tile bump cannot silently reintroduce the capture hazard) |
| **`939-952`, `mm5`** | the dispatch branch (below) |
| `~1250` (with `Q27_DEXIT`/`Q27_MAXD`/`Q27_SUFFIX*`) | `if (const char* e = getenv("Q27_GEMM_MIN")) gemm_min = atoi(e);` **plus the guardrail** `if (gate_maxd + 1 >= gemm_min) { fprintf(stderr, "q27: FATAL ladder width %d reaches the GEMM path -- the canonical bitwise gate no longer holds by construction\n", gate_maxd+1); abort(); }` |

```cpp
void mm5(const DevTensor& w, const std::array<float*, W_PLUMB>& ys_a) {
    if (vw >= gemm_min && w.rows >= gemm_min_rows) {   // both host ints, read at CAPTURE
        q27k::XLanes X{}; q27k::YLanes Y{};
        for (int i = 0; i < W_PLUMB; i++) {
            X.nat[i] = xq_L[i].nat;      // <- quantize3 already writes these, today
            X.xs[i]  = xq_L[i].scale;
            Y.y[i]   = ys_a[i];
        }
        q27k::vgemm_verify(w.data, (const __half*)w.scales, w.dtype == DType::Q4_G64,
                           X, Y, d_vgemm_ws, w.rows, w.cols, vw, stm);
        return;
    }
    /* ... existing gemv body, byte-for-byte unchanged ... */
}
```

`vgemm_verify` picks `z` from `(rows, cols)` on the host, picks `MODE` from `z == 1`, and
emits 1 or 2 nodes. **`z` and `spz` are capture-time host integers.** Two fixes that the
spike proved are mandatory (5.1, 5.2 below) live inside it.

### Changed: `Makefile`, `src/test_kernels.cu`

Add `src/vgemm.cu` to every link line (`q27`, `q27-server`, `q27-server-w8`,
`q27-server-w16`, `test_kernels`, `width_bench`). Add the vgemm unit (gate 3).

### Untouched

`src/kernels.cu` (`quantize3`, both GEMVs, all occupancy pins), **`src/prefill.cu`
entirely**, every lane allocation (`engine.cuh:493-621`), every downstream consumer.
`k_gemv_q4_n<9..16>` / `k_gemv_q8_n<9..16>` **stay compiled** -- `attn_k`/`attn_v` need
them and they are the A/B fallback until the tolerance campaign signs off.

---

## 5. Kernel budget, checked on the real binary

`ptxas -v` + `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, sm_120, live, this session.
Device measured on the box: **170 SMs, 65,536 regs/SM, 102,400 B smem/SM, 101,376 B/block
opt-in, 48 warps/SM, 24 blocks/SM.**

| instantiation | regs | stack | spill | dyn smem | **CTA/SM** |
|---|---|---|---|---|---|
| **MR=32 Q4 MODE=0** | **64** | **0** | **0/0** | 14,080 | **4** |
| **MR=32 Q4 MODE=1** | **64** | **0** | **0/0** | 14,080 | **4** |
| **MR=32 Q8 MODE=0** | **64** | **0** | **0/0** | 13,824 | **4** |
| **MR=32 Q8 MODE=1** | **64** | **0** | **0/0** | 13,824 | **4** |
| MR=64 Q4 MODE=0/1 | 64 | 0 | 0/0 | 12,288 | 4 |
| **MR=64 Q8 MODE=0** | 64 | **8 B** | **4/4** | 12,032 | 4 |
| **MR=64 Q8 MODE=1** | 64 | **8 B** | **8/8** | 12,032 | 4 |
| MR=16 (all four) | 64 | **8 B** | **4-8** | 18,432 | 4 |
| `k_reduce_z` | 40 | 0 | 0/0 | 0 | -- |

**MR=32 is the only tile that is spill-free on all four instantiations.** That is not a
preference; it is the constraint, and it is why the MR=64 spine both runner-up designs
proposed is rejected -- v1.4 puts 96 of the 401 calls on the Q8 leg.

- **registers: 64 x 256 = 16,384/CTA; 4 x 16,384 = 65,536 = exactly the SM budget.**
  Zero slack. One more live value -> 3 CTAs -> ~20% gone. This is constraint 4's trap
  relocated, and gate 4 is what catches it.
- smem: 4 x 14,080 = 56,320 <= 102,400. Smem would allow 7 CTAs. **Not binding.** And
  13.75 KB is under the 48 KB default limit, so `cudaFuncSetAttribute` is unnecessary --
  the `prefill.cu:476-481` lazy-`static bool attr` capture hazard is **deleted, not
  hoisted**.
- warps 32/48 (66.7% occ), blocks 4/24.

**CTA supply** (`grid = (1, rows/32, z)`; 170 SMs x 4 = **680 co-resident slots**),
`z = clamp(ceil(1400 / (rows/32)), 1, min(8, n_stages/4))`, then KB-aligned and trimmed:

| weight | rows | row-CTAs | z | total CTAs | waves | MODE |
|---|---|---|---|---|---|---|
| `ffn_gate`/`ffn_up` | 17408 | 544 | **3** | 1632 | 2.4 | 1 |
| `ffn_down` | 5120 | 160 | **8** | 1280 | 1.9 | 1 |
| `attn_qkv` | 10240 | 320 | **5** | 1600 | 2.4 | 1 |
| `attn_q` | 12288 | 384 | **4** | 1536 | 2.3 | 1 |
| `attn_gate` | 6144 | 192 | **7** | 1344 | 2.0 | 1 |
| `ssm_out`/`attn_output` (Q8) | 5120 | 160 | **8** | 1280 | 1.9 | 1 |
| **`output_q4` (head)** | 248320 | **7760** | **1** | 7760 | 11.4 | **0** |
| `attn_k`/`attn_v` (Q8) | 1024 | 32 | -- | -- | -- | **GEMV** |

Head: z=1, plain store into `logits2 + t*VOCAB` through `YLanes`, **no workspace, no reduce
node, no zeroing, deterministic.** The largest call in the round is also the easiest one.

**Workspace bound: `max(z * W_PLUMB * rows * 4)` = `ffn_gate` at 3 x 16 x 17408 x 4 =
3,342,336 B.** Compute it from the weight list at init; **do not hardcode 4 MB** -- the
`ctaTarget` knob at 2400 gives `ffn_gate` z=5 = 5.57 MB and silently corrupts the heap.

### 5.1 Fix that is not optional: KB-align `spz`

The spike prefetches `KG=2` stages per super-step but bounds only the MMA, not the load. On
any **odd** slice (`ffn_down` at z=8 -> spz=17; `attn_gate` at z=8 -> spz=5) the last
`load_stage` reads one stage past `s_end` -- and for the last z-slice, **past `cols`**.
`compute-sanitizer memcheck` catches it (invalid 4 B read, 1 B past a 2,176 B allocation =
a decode lane's x-scale array). Fix, two lines in the launcher:

```cpp
int spz = (n_stages + z - 1) / z;
spz = (spz + KG - 1) / KG * KG;     // KB-align: the slice can never straddle n_stages
z   = (n_stages + spz - 1) / spz;   // trim empty slices (or k_reduce_z sums uninit partials)
```

It is also **worth 1.6% at the round level** (12.17 -> 11.98 ms at W=12), because the odd
tail super-step was idling half the warp-groups. Both fixes are already in the corrected
spike and every number in this plan is post-fix.

### 5.2 `cols % (KG*KS) == 0` check

The bench fork dropped `prefill.cu:472-475`'s guard. Every decode `cols` (5120, 6144, 17408)
complies, but a silent tail-drop is a landmine. Re-add as a runtime abort at capture.

---

## 6. Numerics and determinism policy

| path | class | mechanism | measured |
|---|---|---|---|
| **ladder (W=2..8), all draft graphs, single-lane decode, ALL sampled rounds** | **BITWISE, by construction** | `gemm_min = 9 > gate_maxd + 1 <= 8` (`engine.cuh:1226-1227` hard clamp). Zero nodes added to those graphs, zero allocations moved, `quantize3` and both GEMVs untouched. The CLI never even *captures* a width>=9 graph (`suffix_on` is false without `Q27_SUFFIX`). | canonical md5 `a2982c5197c627551b27d76a0a94b220` EXACT |
| **suffix (W >= 9)** | **tolerance-class, run-to-run DETERMINISTIC** | Identical int8 activations (same `nat`/`scale`, same `quantize3`, same group-32 amax). int8xint8 products **bit-exact in int32** (max abs 8*127*32 = 32512, no overflow). The Q4 `-8` nibble bias moves from the GEMV's `isum` term to `__vsub4(..., 0x08080808u)` at the smem unpack -- algebraically identical. **Only the fp32 accumulation ORDER differs.** | Q4 rel **1.05e-06**; Q8 rel **2.80e-06**; **0 / 139,264 floats differ over 8 repeats** |

**No atomics, anywhere.** `atomicAdd` across `grid.z` is measurably nondeterministic
(11,082-20,355 of 61,440 floats differ run-to-run on `ffn_down` at z=8). The engine's greedy
path is run-to-run bitwise today and callers can depend on it. The deterministic two-pass
reduce costs 0.4-0.7 ms/round and buys that property back. **Not for sale at that price.**
This is a strictly stronger guarantee than the fdmma precedent (which is
deterministic-but-different).

**Own this, up front, in the BUILDLOG and the README:**

> **`Q27_SUFFIX=1` stops being byte-identical to `Q27_SUFFIX=0`.** Today that invariant is
> explicit (`engine.cuh:1474`, "emitted tokens stay greedy-identical regardless of proposal
> quality") and the A/B tooling leans on it. After this change, suffix rounds compute logits
> on a different numeric path, so an argmax near-tie can re-roll and the emitted text can
> differ. Magnitude: the vocab head's absolute logit error is 1.4e-06 against a typical
> top-2 gap of 0.236 (measured), so flips will be **rare** -- but rare is not never.
> Reclassify suffix-on greedy output as tolerance-class **now**, and update the comment at
> `engine.cuh:1474`. Do not discover this at a gate.

---

## 7. Design provenance

### 7.1 Taken

**From `perf` (the spine):**
- `k_mma_v2<MR=32, Q4IN, MODE>` with the **intra-CTA warp-group K-split** (KG=2). It is the
  only spill-free tile across Q4/Q8 x direct/partials, and it is **20% faster at the round
  level** than the MR=64 spine that both other designs proposed (11.98 vs 14.39 ms at W=12).
  The runner-ups converged on the wrong tile because neither measured a Q8 leg.
- The `Q4IN` template restored from `prefill.cu:221` (v1.4 makes it mandatory).
- `__launch_bounds__(256,4)` as load-bearing, plus the init-time occupancy assert.

**From `minimal` (the discipline):**
- **Rejecting XG64** -- argued first and hardest there, independently confirmed by the other
  two. XG32 reuses `nat`/`scale` that `quantize3` already emits: no `nat64`/`s64`, no
  `quantize3_g64`, no new buffer, no new graph node, no touch to `qx5`. (XG64 is *also*
  66 regs -> 3 CTA/SM, *and* rel 1e-2 vs the reference -- dead on all three axes.)
- **Deterministic partials + fixed-order reduce** over `atomicAdd`, and the argument for why
  run-to-run determinism is worth 0.7 ms.
- `gemm_min` read once in `build_spec_graphs` with the other env knobs; `Q27_GEMM_MIN=99` as
  the in-binary A/B off-switch; `GEMM_MIN_ROWS` excluding `attn_k`/`attn_v`.
- The ladder-reach guardrail (promoted from a warning to a hard abort).

**From `unify`:**
- The insistence on pricing the reduce nodes **inside** the measurement (0.41 ms/round for
  368 nodes in a captured graph -- measured, and inside the 11.98).
- The observation that prefill's NT=128 collapses below T~32 (547/320 GB/s at T=16) and,
  because prefill is not graph-captured, could pick NT at launch for free. **Filed as a
  separate future item (P4), explicitly NOT used to justify touching `prefill.cu` here.**
- The true DRAM roofline (~1695 GB/s asymptotic; a 47 MB kernel cannot ramp past ~1520).
  Our "84% of SOL" is really 76% of peak. Filed as P4; not used to inflate any number here.

**From the adversarial checks (all four kills incorporated):**
1. **KB-aligned `spz` + z-trim** (perf-critique KILL 1) -- a `compute-sanitizer`-demonstrated
   OOB on 112 of 369 GEMM calls per round. Fixed; also worth +1.6%.
2. **The model correction** (perf-critique KILL 2) -- v1.4, 305 Q4 / 96 Q8, 14.66 GB.
   **Everything in this plan is re-measured on it.**
3. **Workspace sized from the weight list and asserted** (perf-critique KILL 3;
   unify-critique defect 1 -- the head at z>1 would overrun by 7 MB).
4. **NonWeight is NOT flat** (perf-critique): `5.0 + 0.21*W`. Both designs' "flat NW"
   headline was a cross-model artifact, and one of them printed `NW(16) < NW(12)`, which is
   physically impossible.
5. **`gemm_min = 9`, not 11** (perf-critique): `tools/width_curve.sh` reaches W=9,10 via
   `Q27_MAXD=4 Q27_SUFFIX_W=W`. An 11-boundary would put a kernel-family discontinuity
   *inside the instrument used to gate this work*.
6. **The verify GEMM needs a Q8 leg** (unify-critique defect 4 + the model correction).
7. **This is greedy-only** (unify-critique) -- stated in 1c.
8. **`Q27_SUFFIX=1` is no longer byte-identical to `Q27_SUFFIX=0`** (unify-critique) --
   stated in section 6, not discovered at a gate.

### 7.2 Rejected, with reasons

| rejected | from | why |
|---|---|---|
| **The XQuant arena** (one `cudaMalloc` for all 16 lanes' `nat`/`scale`) | `unify` §3, `minimal` §8.1 fallback | **Measured unnecessary.** The spike hits **1292 GB/s** reading 16 *independently* `cudaMalloc`'d lanes through a `__grid_constant__` pointer struct. `tt` is warp-uniform in the staging loop, so the base-pointer load broadcasts from the constant bank at zero cost. Re-laying out the **ladder's** buffers for zero gain is blast radius bought for nothing. |
| **Refactoring `prefill.cu` into a shared `__device__` tile core** | `unify` §2 | Right in the abstract, wrong here. The unify-critique showed the refactor **cannot pass its own safety gate** (byte-identical prefill SASS): a runtime `ldx`, a dead `YLanes` in the param block, and the guarded `XLD` form each perturb NT=128 codegen -- and the sibling XG32 instantiation sits **3 registers from the 255 wall**. `prefill.cu` is a shipped, `--pf`-identity-gated path. Pay the ~200 lines of duplication. The two kernels want opposite occupancy pins (174 regs @ 1 CTA/SM vs 64 @ 4) and one `__global__` cannot carry both `__launch_bounds__` anyway. |
| **`atomicAdd` K-split** | the original bench (its only timed epilogue), `minimal` §8.3 fallback | Nondeterministic, measured. See section 6. |
| **MR=64** | `minimal`, `unify` | 20% slower at the round level **and it spills on the Q8 leg** (8 B stack), which v1.4 makes 96 of 401 calls. |
| **MR=16** | -- | Spills on all four instantiations. |
| **XG64 / `nat64` / `s64` / `quantize3_g64`** | the original brief; "GROUND TRUTH: engine" §6a | rel **1e-2** vs the dp4a reference. Argmax-flip territory, not tie-re-roll. Also 3 CTA/SM and +257 graph nodes/round. |
| **`gemm_min = 11`** | `minimal`, `unify` | See 7.1 item 5. |
| **Taking the GEMM down to the ladder** (W=8, where it already wins **10%** at the round level) | -- | **Declined deliberately.** It would trade the bitwise-by-construction canonical gate for ~10% of the weight path on rounds that are already the cheap ones, and it would force a full tolerance-class campaign on the **novel-prose decode path** (the 224 t/s headline). `gemm_min = 9` is a **safety line, not a performance crossover** -- say so plainly rather than implying 9 is where the kernel starts winning. Revisit only with its own plan and its own gates. |

---

## 8. Costs to measure, not assume

- **Graph zoo growth.** 368 of 401 `mm5` calls gain a `k_reduce_z` node -> the wide verify
  graph goes from ~401 weight nodes to **~769**, x `W_MAX` perms. Instantiation time and
  graph VRAM are **unpriced**. Measure boot time against the 4.0 s W12 baseline.
- **The reduce-node floor:** 368 nodes x ~1.1 us = **0.41 ms/round** in a captured graph
  (measured). Inside the 11.98. Killing it (last-CTA semaphore epilogue) is P4, not P1.
- **`attn_k`/`attn_v` stay on `gemv_q8_n` at N=vw.** 32 calls, 170 MB/round. `gemv_q8_n`
  has no `isum` term so it degrades gracefully (-18% from N=12 to N=16, vs the Q4 GEMV's
  -55%): ~0.37 ms/round at W=16. **Accept it, but do not claim "no superlinear term
  anywhere."**
- **fdmma at W>=14 runs at 1 CTA/SM** (BUILDLOG 07-13: `s_q` crosses 80 -> 96 rows, 2 x
  52.1 KB > 100 KB). At 28K that costs ~0.4 ms. **The entire width curve was taken at 28K,
  and fdmma's cost is context-dependent.** Re-run the width curve at 60K before shipping the
  cap raise (gate 8). If the fdmma term dominates at depth, P3's margin shrinks.
- **NW's 0.21 ms/lane slope** is what will eventually cap W. It is a subtraction with
  +/-0.3 ms of replay noise. **P0's node histogram measures it directly** -- take that,
  not the subtraction.
- **The MTP draft ladder.** NW(5) = 5.13 ms with only ~2 ms plausibly attributable to
  fdmma/GDN/elementwise at width 5. The remainder is most likely the depth-4 draft ladder
  **re-streaming the 675 MB Q4 head four times per round**. It is W-invariant, so it never
  shows up in a width curve -- and it would pay on **every** round including novel prose.
  P0's histogram prices it for free. If it is ~3 ms, that is the next universal lever and it
  is bigger than anything suffix-specific.

---

## 9. Phased build

### P0 -- the killer experiment. Zero code, ~1 hour. **Run it first.**

`nsys` a live W=12 suffix round; sum the `k_gemv_q4_n` + `k_gemv_q8_n` node durations.

```
sudo -n env CUDA_VISIBLE_DEVICES=0 nsys profile --capture-range=none \
  build/q27-server-w16 <model> <tok> --port 8081 --ctx 32768 --no-think --fast-head
# Q27_KV=fp8 Q27_FD=mma Q27_MAXD=4 Q27_PMIN=0.5 Q27_SUFFIX=1 Q27_SUFFIX_W=12 Q27_PHASE_STATS=1
# drive scratchpad/accept_payload_echo.json ; then:
nsys stats --report cuda_gpu_kern_sum
```

**Prediction: GEMV nodes = 16.7 +/- 0.7 ms of a 24.3 ms round (69%).** Kill criteria in
1d. Same run, for free: the full node histogram gives `NW(12)` directly (not by
subtraction), prices the fdmma / GDN / `argmax_masked` / draft-ladder split, and repeated at
W=16 gives `NW(16)`. **Do not skip this. The 07-13 W16 plan skipped its equivalent and
shipped a NO-GO.**

### P1 -- the kernel, standalone. No engine wiring. ~1 session.

`src/vgemm.{cu,cuh}`: `k_vgemm<32, Q4IN, MODE>` + `k_reduce_z` + the host z-policy, with
the KB-align/trim fixes (5.1), the `cols % 256` check (5.2), and `vgemm_ws_bytes()` walking
the weight list. Gates 3, 4, 6.

> **P0/P1/P2 DONE 2026-07-13 (same day).** P0 measured GEMV = 15.7 ms of a 21.4 ms
> round (BUILD). P1 shipped src/vgemm (gates 3/4/6 green: rel 1.9e-7, 64 regs / 0
> spill / 4 CTA/SM, sanitizer clean). P2 wired it at gemm_min=9: **canonical EXACT
> with GEMM on AND off** (the guardrail aborts if a ladder width ever reaches the
> GEMM), suffix round **24.33 -> 19.96 ms** (echo 427 -> 519 t/s, under the 20.5
> keep-bar), determinism gate byte-identical run-to-run, Q27_GEMM_MIN=99 byte-matches
> the pre-P2 binary, shortbench 174.7 (ladder untouched). P3 (the W16 cap reopen) is
> next.

### P2 -- engine wiring at W=12. `gemm_min = 9`. ~1 session.

The five `engine.cuh` edits. Ships with `Q27_GEMM_MIN=99` as the in-binary disable.
Gates 1, 2, 5, 7, 9, 10.
**Keep-bar: suffix round 24.33 -> <= 20.5 ms.** Above 21.5 and the NW-is-additive model is
wrong; stop and do not build P3.

### P3 -- the W16 REOPEN. The payoff. ~1 session.

Rebuild `-DQ27_W_MAX=16` (target exists), `Q27_SUFFIX_W=16`, re-run `tools/width_curve.sh`
at **both 28K and 60K**. Then the gate that actually decides it (gate 8): a real-traffic
accept-length histogram. Ship the cap iff `tok(16)/tok(12) > 1.07` on **live CC traffic**,
not on the echo payload.

### P4 -- what is left, priced, not scoped

- **Fuse `ffn_gate` + `ffn_up`** (same activation, same `cols`, adjacent calls, **128 of the
  401**). One kernel over a 34816-row range with two output bases doubles the streamed
  footprint per launch (47.3 -> 94.6 MB), which moves the size-matched DRAM ceiling
  1524 -> 1631 GB/s, *and* halves the reduce nodes on the biggest shape. Worth an afternoon.
- Last-CTA-semaphore epilogue: kills 368 reduce nodes (0.41 ms) and the L2 partial traffic.
- `attn_k`/`attn_v` onto a rows-aware tile (0.37 ms at W=16).
- **The MTP draft ladder** (see section 8) -- probably the biggest remaining lever and the
  only one that touches novel prose.

---

## 10. Gates

1. **Canonical `a2982c5197c627551b27d76a0a94b220` EXACT**, on the default build *and* the
   `-DQ27_W_MAX=16` build, `Q27_GEMM_MIN` at its default. **Structural, not hopeful** -- the
   CLI never captures a width>=9 graph.
2. **Graph-node identity on the ladder.** `cudaGraphGetNodes` count + kernel-name list for
   `verify_graph_w[2..8][*]` byte-identical before/after. Proves "zero nodes added to the
   ladder" instead of asserting it. Plus the runtime abort at `gate_maxd + 1 >= gemm_min`.
3. **`test_kernels`: the vgemm unit.** All 9 decode shapes x {Q4, Q8} x W = 2..16 x
   **all 16 lanes** (the old bench only ever checked lane 0), at z=1 and at production z.
   `rel < 1e-5` vs `gemv_q4_n`/`gemv_q8_n`; lanes >= vw provably untouched;
   **bitwise-identical across 8 repeats** (the determinism claim).
4. **`ptxas -v` + live occupancy, in CI.** All four `MR=32` instantiations: **<= 64 regs,
   0 stack, 0 spill, `cudaOccupancyMaxActiveBlocksPerMultiprocessor == 4`. Fail loud.**
   There are zero spare registers; one added live value silently costs a CTA tier and ~20%.
   This is the single most valuable line in the plan.
5. **`Q27_GEMM_MIN=99` A/B control.** Same binary, GEMM disabled -> must reproduce today's
   24.33 ms suffix round **and byte-identical output** on the suffix-heavy payload. Proves
   the plumbing is inert when off.
6. **`compute-sanitizer` memcheck + racecheck + initcheck** on a run that actually commits a
   wide suffix round. Non-negotiable: this is the gate that the perf design wrote and did
   not run, and it would have caught the OOB in 5.1.
7. **Perf keep-bar:** suffix `sfxm/sfxn` on the echo payload, **24.33 -> <= 20.5 ms** at
   W=12. Above 21.5 -> the model is wrong; stop before P3.
8. **P3's real gate (the one that decides the cap):** `Q27_SUFFIX_DBG=1` histogram of
   accepted-per-suffix-fire on **live CC traffic** (T8/T2, `tools/thunderdome_pin_ab.sh`),
   at W=12 and W=16, at 60K context. Ship the cap **iff** mean tok/fire rises by
   **> 6.4%**. Reference: real-traffic AL is **9.4** today (BUILDLOG 07-10), so the bar is
   AL 9.4 -> 10.0. Also re-run the width curve at 60K -- the fdmma 1-CTA cliff at W>=14 is
   context-dependent and the whole curve was taken at 28K.
9. **Tolerance-class campaign** (suffix output *will* change on ties): fixed-bytes paired
   replays (codegen / testgen / docs / echo -- compare quality, not bytes), the **basin
   matrix** A/B (fdmma precedent, BUILDLOG 2026-07-10), PPL, shortbench, thunderdome T8/T2.
10. **Determinism gate (new, and the reason we refused atomics):** run the same suffix
    payload twice on the same binary; byte-compare. Must be identical.
11. Novel-prose suite unchanged (regression guard -- the ladder path).

---

## 11. Effort / recommendation

**BUILD -- but in the stated order, and P0 first.**

- **P0**: 1 hour, no code, and it can kill the whole thing. It is also the only measurement
  that gives `NW(W)` directly instead of by subtraction, and it prices the draft ladder for
  free.
- **P1 + P2**: ~2 focused sessions. The kernel is a ~250-line fork with a measured budget
  (64 regs / 0 spill / 4 CTA/SM, all four instantiations), and the engine touch is **five
  edits, one of them a five-line branch in `mm5`**. The bitwise gate holds *by
  construction*, which is why the blast radius is small enough to justify.
- **P3**: ~1 session, and it is where the value is.

**The risk is not correctness; it is size.** P2 alone is +5-7% agentic, which the retier
already delivered for twenty lines. Do not sell it on that. **Sell it on the break-even
ratio: the cap raise needs a +62% accept jump today and a +6.4% jump after this** -- and on
the fact that the flat weight path removes the superlinear term that made "wider verify" a
dead axis for this engine. If P0 comes back at 50%, the honest answer is that the wide round
was never weight-bound, this plan is a footnote, and the next 12 ms are somewhere nobody has
profiled.