# Prefill-attn Phase 3: FA2-class relayout (2026-07-09)

Target: `k_attn_prefill_mma_fp8q` (the default fp8-KV prefill attention).
Phase-0 attribution (2026-07-07, still structurally true): 12.5% occupancy
(6 warps/SM), DUAL limiter registers(248/thread) + smem, Block Limit 1 by
`__launch_bounds__(192,1)`; stalls long_scoreboard 30% / math_throttle 28% /
barrier 15%. External validation of the prize: FlashRT's vendored FA2 runs
~2900 t/s average at 256K vs our 2206 at 128K.

## Why occupancy is register-pinned

CTA = 6 warps, one per GQA-6 q-head; each warp owns a full 16-token x
256-dim O tile = o[32][4] = 128 fp32 registers per thread. 192 threads x
248 regs = 47.6K regs -> Block Limit 1 (64K/SM). Halving the CTA without
cutting per-thread registers just repackages 6 warps/SM (2 CTAs x 3
warps): occupancy unchanged. The register cut IS the lever.

## Phase 3a shape (this build): token-split warps, fat CTA

- CTA = 384 threads (12 warps): 2 warps per q-head, each warp owns 8 of
  the 16 query tokens (per-thread O = 8x256/32 = 64 fp32).
- Register budget: ~150-170/thread -> 384 x 168 = 64.5K... enforced via
  `__launch_bounds__(384, 1)` + reg ceiling; target is 12 warps/SM (25%
  occupancy, 2x today) in ONE CTA, preserving the read-K/V-once GQA
  sharing (no duplicated KV traffic, unlike the split-CTA shape).
- smem: s_q 6x16x260 fp8 (25KB) + s_v 32x264 half (16.5KB) + s_kraw
  2x32x272 (17KB) + s_vraw 8KB = 66.4KB < 99KB (unchanged layout; the
  relayout is the WORK PARTITION, not the smem map -- rename pending).
- Softmax state (m/l) becomes per-warp over 8 rows instead of 16; the
  QK^T MMA fragment loop halves per warp; PV unchanged per row.
- If regs still pin below 12 warps: spill O rows 4..7 pairs to a small
  smem stash... NO -- kill criterion instead (below).

## Phase 3b (only if 3a pays): 4 warps/head (per-thread O = 32 regs),
24 warps/SM target; revisit cp.async depth and PP under the new balance.

## Gates

- Variant kernel behind `Q27_PF_FA2=1`, default OFF until gates pass.
  fp16 path, fp8q default kernel, and canonicals untouched by default.
- Correctness: --pf serial-vs-batched continuation identity under
  Q27_PF_XG=32 fp16 (n/a -- fp8 path only); primary gates = same
  tolerance class as fp8q's shipping gates: --pf continuation vs
  default-fp8q IDENTICAL-or-tolerance (greedy continuations), logit
  cosine >= 0.9999 + argmax MATCH at depth via --dump-logits on --pf,
  needle spot-check before default-on.
- Perf: --pf 32768 and 131072 TTFT, fp8 KV, vs default (59.4s @128K
  baseline, base model). GO bar: >= +10% @128K attn-heavy depths.
  ncu occupancy check: warps/SM must actually reach ~12 (if regs pin it
  lower and TTFT gain < +5%, kill and record).
- Kill criteria: reg pressure forces spills (sanitizer-visible slowdowns,
  local-memory traffic in ncu) that eat the occupancy win; or +<5% TTFT
  at 128K with occupancy doubled (would mean the stalls were not
  latency-hiding-limited -- record and stop).

## Non-goals

Do not touch: f16-MMA kernel, cp.async machinery semantics, split-KV
combine, GDN prefill, decode paths. No smem geometry changes in 3a.

---

## Phase 3a VERDICT (2026-07-09, same day): KILLED -- occupancy was a red herring

Built as designed: 384-thread CTA, warp-pair d-split, 144 regs (vs 254),
zero spills, LOGITS BITWISE IDENTICAL to fp8q (exact-math transform).
Occupancy doubled precisely (ncu: theoretical 25%, achieved 24.98%, 11.99
warps/SM). TTFT: 32K 10.48 -> 10.58s, 128K 59.7 -> 60.4s (-1%).

ncu tells why doubling warps bought nothing: Issued/scheduler 0.26 with No
Eligible 73.89% -- UNCHANGED from the 6-warp kernel's class. The added
warps are lockstep CLONES of their pair (same redundant QK^T, same
barriers): they stall on the same shared dependency chains at the same
cycles, adding zero eligible-warp diversity. The kernel is
BARRIER/DEPENDENCY-serialized, not latency-hiding-limited; per-tile
__syncthreads gates every warp through the same pipeline phases.

Consequence for Phase 3: the win FlashRT's FA2 demonstrates does not come
from occupancy -- it comes from the ASYNC STRUCTURE (software-pipelined
tiles, warp-specialized producer/consumer, no full-CTA barriers on the
hot path). That is the original "from-the-smem-layout rewrite" at full
scope (2-4 sessions), now with a sharper spec: the rewrite's target
metric is Eligible Warps Per Scheduler (0.44 today), not occupancy.

Do-not-retry within the current skeleton: register cuts, CTA repackaging,
and split-kk variants all preserve the barrier structure that binds.
Kernel + launcher branch REVERTED per kill protocol; this doc and the
BITWISE-transform technique (d-split warp pairs validated exact) are the
artifacts.
