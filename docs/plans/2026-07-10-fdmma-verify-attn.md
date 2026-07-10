# fdmma: fp8-MMA split-KV W-query verify attention (ceiling-8+ kernel)

Synthesized 2026-07-10 from a 3-design / 3-adversarial-check workflow (readers
over prefill.cu pv8 machinery, spec3 FD contract, budget arithmetic; designs
biased occupancy / mma-efficiency / ship-risk; checks on fragment math, byte
budgets, correctness edges). Winner: the mma-efficiency design, which survived
all three checks as-specified; fixes from the checks are folded in below.
Full agent outputs: workflow wf_946d364b-7bf journal.

## Why (verdict chain, all measured)

fd2 re-streams the full KV per verify lane and is DRAM-SATURATED at 61K W=8
(1GB in 795us = 1.26TB/s = 87% peak; one layer). Verify width marginal =
2.18ms/lane at 61K (ctx-dep component 1.36), attention-dominated and growing;
~27ms/round of attention projected at 128K W=8. Shared-KV floor is ~9x away
but needs ~240 TFLOP/s of scoring: above fp32 CUDA-core peak -- three scalar
prototypes measured 0.27-0.83x (tools/attn_fdw_bench.cu). Tensor-core fp8 MMA
is REQUIRED. GO bar: >=2x vs fd2 at 61K W=8, kernel+combine; predicted
4.3-5.0x (160-185us; 230us zero-overlap worst case).

## Locked shape

- One kernel `k_attn_fdmma`, fp8-e4m3 KV only, W(=ntok) 4..8. fp16 KV and
  W<4 stay on fd2. W=16 BLOCKED on CP3/P3/IP3 being `p[8]` structs + scratch
  ntok dim -- kernel design generalizes but plumbing must widen first (later).
- Q-quant in the kernel prologue (bare `__nv_fp8_e4m3` cast, NO scale --
  prefill precedent, cosine 0.99996). No pre-kernel, no extra graph node.
- NS = 128 = FD2_NS: scratch sizing (FD_MAXNS=128) and `k_attn_fd_combine`
  reused UNCHANGED. Any NS<=128 would also work (combine takes ns as arg)
  but 128 is the default; NS is NOT a runtime knob unless it threads through
  the combine launch too (split-brain hazard, occupancy-check O4).
- Grid dim3(NS=128, n_kv_heads=4) = 512 blocks; block 192 threads = 6 warps,
  `__launch_bounds__(192, 1)`. ~3 waves at 1 block/SM (family lands ~248
  regs; 12.5% occupancy is FINE -- barrier-serialized family, latency hiding
  = cp.async double-buffer, per FA2-3a).
- **Dense (head, lane) row packing**: global row r = j*W + t (j = gqa head
  0..5, t = lane). M = 6W live rows in a 96-row padded Q matrix; warp w owns
  rows 16w..16w+15 with the full 16x256 O in registers (o[32][4], the exact
  pv8 accumulator). Live warps = ceil(6W/16): W=4 -> 2, W=8 -> 3. Zero MMA
  padding waste at W=8 (W=4 pays 25% -- acceptable, W=4 sits at the fd2
  dispatch boundary). Dead warps do ALL staging (cp.async, s_vt transpose,
  s_q fill) and hit every barrier.
- KV tile PP=32 keys; QK = 8 k32 steps x 4 n8 subtiles; PV = 32 n8 dim-tiles
  x 1 k32 step; `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
  (mma_e4m3, prefill.cu:845; arch-probe gate for sm<890 stub).

## smem map (dynamic, fd_setattr pattern; 70,016B = 68.4KB, verified byte-exact)

| offset | buffer | shape | bytes |
|---|---|---|---|
| 0 | s_q e4m3 | [96][LDQ=260] | 24,960 |
| 24,960 | s_kraw e4m3 x2 | [2][32][LDK=272] | 17,408 |
| 42,368 | s_vraw e4m3 x2 | [2][32][256] | 16,384 |
| 58,752 | s_vt e4m3 | [256][32] | 8,192 |
| 66,944 | s_P e4m3 | [6][16][32] | 3,072 |

LDQ/LDK pads kill 8-way conflicts on u32 fragment loads (static_assert LD%4==0;
pad-tail poison test per prefill review finding #4).

## Geometry (the split contract -- risk #1)

Per lane t (t < W ONLY -- IP3 slots beyond ntok hold garbage, never deref):
seq_t = *pos.p[t]+1; chunk_t = ceil(seq_t/NS); lo_t = sp*chunk_t;
hi_t = min(seq_t, lo_t+chunk_t); live iff lo_t < seq_t (== fd2's skip rule
spec3.cu:333, == combine's used_t derivation). If NO lane live: return BEFORE
issuing any cp.async (pending-async-at-exit is UB -- occupancy-check O1).
Stream union of LIVE lanes only: p_beg = (min live lo_t) & ~31,
p_end = max live hi_t. Per-lane chunk_t can differ by 1 near seq = k*NS
(~5.5% of seqs at W=8): the union covers it (overhead sp+1 rows worst case,
~<1% amortized) and per-row two-sided masks carve each lane's exact
[lo,hi) window, keeping combine's per-lane geometry exactly satisfied.

Dead rows (r >= 6W): bounds lo=hi=0 (masks everything, NO signed-overflow --
simplicity's INT_MAX/INT_MIN variant is UB via `hi_r - p0`). Dead-row s_q
MUST be zero-filled up to 16*live_warps (mma-check M1: A-frags read up to
the warp's full 16 rows; unstaged smem = sanitizer initcheck fail).

## Mainloop (per tile: wait -> transpose -> prefetch -> MMA)

1. cpasync_wait_all; __syncthreads.
2. All 6 warps: transpose s_vraw[cur] -> s_vt[256][32]; __syncthreads.
3. Prefetch next tile into 1-cur (cpasync16, src_bytes=0 zero-fill past
   p_end); commit.
4. `if (warp >= live_warps) continue;` -- BOTH barriers precede this guard;
   any future edit adding a barrier below it deadlocks (checked fragile,
   comment in code).
5. QK: A = s_q rows (R0=16w+gid, R1=R0+8), B = s_kraw natural [key][LDK]
   col-major; s[4][4] f32.
6. Mask + online softmax: prefill.cu:1644-1693 block VERBATIM except the
   bound test becomes two-sided per row: mask iff p0+c < lo_R || p0+c >= hi_R
   (lo-side masking has NO in-tree precedent -- microtest surface #1).
   Masked lanes -> -FLT_MAX -> exact 0.f weights; m/l/o rescale idiom
   unchanged; scale applied to f32 scores, never folded into quant.
7. P relayout D-frag -> A-frag through s_P: four scalar e4m3 BYTE stores per
   row per subtile (donor prefill.cu:1703-1706; u32 stores are IMPOSSIBLE
   from the D fragment without cross-thread packing -- fragments-check
   finding), __syncwarp (same-warp ownership), u32 reads.
8. PV: A = s_P, B = s_vt (4 consecutive keys at fixed dim = 1 aligned
   lds.32); accumulate into o[32][4] across the whole chunk.

## Epilogue (-> FD_ST partials, combine unchanged)

Per live warp, rows {R0, R1}, skip r >= 6W and lanes with lo_t >= seq_t
(write NOTHING -- combine's max loop spec3.cu:256 reads EVERY slot sp < used_t
unguarded and scratch is never zeroed: one missed live-slot write poisons the
max with stale garbage; one extra write corrupts a neighbor. The written-slot
set must equal {(t,sp): sp < used_t} EXACTLY -- microtest asserts both
directions on NaN-poisoned scratch).
pair = t*(n_kv_heads*gqa) + kvh*gqa + j (spec3.cu:414); dst = part +
(pair*NS + sp)*FD_ST; tg==0 writes {m, l}; cols c = n*8+tg*2 (+1) write
unnormalized o. No atomics; fixed order -> bitwise run-to-run.

## Numerics + gates

- Tolerance-class vs fd2 (fp8 Q + fp8 P): per-head cosine > 0.999 vs fp64
  ref; parity vs fd2+combine before any bench number is trusted.
- Acceptance risk (bw24: fp8 numerics moved spec acceptance -8..+8pp
  silently): ships opt-in `Q27_FD=mma` (fd2 stays default; choice baked at
  graph capture like Q27_FD=v1); replay acceptance A/B (accept_ab rig) is
  the default-flip gate, not correctness alone. Canonical untouched.
- Fallback ladder if P-quant fails acceptance: fp8 QK + f16 PV via
  ldm_x2_trans (the fp8q variant, prefill.cu:1191) -- keeps everything but
  the PV8 step.
- compute-sanitizer memcheck + racecheck + initcheck clean (full path,
  sudo -n); cuobjdump reg count (family precedent 248-254; 255 cap; -O2
  only -- CUDA 13.x -O3 miscompiles sm_120).
- Day-1 perf gate: ncu Eligible-Warps on the 61K W=8 bench shape; if MMA
  utilization craters (3-live-warp schedule), the fallback packing is
  rows=lanes/warp=head (= the simplicity design, 2x padded MMA, still
  clears 2x on its own arithmetic).

## Microtest plan (tools/fdmma_test.cu, BEFORE the kernel; pv8_mma_test pattern)

1. QK fragment math standalone: dense-row s_q, per-row heterogeneous
   two-sided [lo,hi) masks -- lo inside tile, hi inside tile, empty
   intersection, full-dead rows -- vs CPU e4m3-exact reference; assert
   masked-exact-0 and l=0 sentinel behavior feeding combine's skip.
2. s_P relayout round-trip (byte stores, u32 reads) with dead rows, W in
   {4,8}.
3. Whole kernel single block, then full grid + REAL k_attn_fd_combine vs
   fp64 CPU reference AND vs fd2+combine: randomized (seq, W, sp); seq
   straddling {k*128-1, k*128, k*128+1} so per-lane chunk_t disagree; empty
   splits; W in {4,5,6,8} (6 exercises non-pow2 R%W); NaN-poisoned scratch
   with exact written-slot-set assertion; poisoned pos.p[t]/qp.p[t] for
   t >= W; bitwise repeat-run.
4. Bench leg in attn_fdw_bench.cu: fdmma vs fd2 at 26K/61K, W=4/8 -> GO bar.

## Integration (after microtest + bench GO)

attn_decode3 dispatch: `Q27_FD=mma` && fp8 && ntok>=4 -> fdmma, else fd2
(env read at launch = baked at graph capture per [vw][perm] graphs, same as
Q27_FD=v1 precedent). Scratch/combine/graph shapes untouched. Engine gates:
canonical EXACT (knob off AND on -- greedy tokens must not change if
tolerance holds... they CAN change (tie flips); therefore canonical applies
knob-OFF only, and knob-ON gates are needle + acceptance A/B + score-parity
per the fp8q default-on precedent), width-curve re-measure (phase stats),
then maxd7/8 economics re-run with the new curve.

## Predicted wall (61K, W=8, NS=128; arithmetic verified by checks)

DRAM: KV 125.8MB + partials round-trip 50.7MB (~40us, ~40-47% of the KV-only
floor) + Q ~L2-resident => ~140us @ 1.26TB/s. MMA: 12.1 GFLOP dense on <=3
schedulers => 48-80us at 150-250 TF realized. Wall = max x 1.15-1.3 =>
160-185us vs fd2 795us = 4.3-5.0x (worst case 3.5x; GO >=2x has 2x margin).
Attention/round at 61K W=8: 12.7ms -> ~2.6-3ms. At 128K the prize ~doubles.
