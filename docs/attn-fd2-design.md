# attn-fd2: register-accumulator flash-decode

2026-07-04 night. Follows the decode-at-depth attribution (BUILDLOG same
date): k_attn_fd is 99% of the +19.9ms/round cost 16K->61K, runs at ~5%
DRAM BW at both depths, and the NW 8->4 probe proved it latency-hiding-
bound -- occupancy is capped by the 55KB smem accumulator (1 block/SM, 8
resident warps). Fix = move accumulators to registers, vectorize the
byte-granular K/V loads, then re-fill the grid.

## Kernel (k_attn_fd2, replaces k_attn_fd as default; v1 kept as
## reference + Q27_FD=v1 fallback)

Grid (n_kv_heads=4, FD_NS, ntok in {1,5}) unchanged; block = NW*32
threads, NW a template param (tune 4 vs 8; probe favored 4).

Per-lane state: acc[6][8] registers -- lane owns dims
D(l) = {4l..4l+3, 128+4l..128+4l+3} (uchar4-aligned so K/V loads are two
uint32 per row instead of 16 single bytes; fp8->f32 convert per byte
after). m[6], l[6] per WARP as today (lane-uniform after shfl broadcast).

smem: s_q[6][256] (6KB) + s_merge[6][256] (6KB) + s_ml[NW][6][2]. ~12.3KB
total vs 55.3KB -- smem stops being the occupancy limiter; expected
~5 blocks/SM at NW=4 (reg-limited, est ~90 regs/lane) = 20+ resident
warps/SM vs 8 today.

Position loop per warp (stride NW over the split's range): load K,V rows
via uchar4 pairs, dot vs s_q over owned dims, shfl-reduce, online-softmax
update (m,l, acc FMA in REGISTERS -- this deletes the 96 smem RMWs per
position per warp that serialized v1).

Cross-warp merge (block epilogue): rescaled adds into s_merge, warps
SERIALIZED by barrier passes in warp order -- NOT smem atomics: atomics
reorder fp adds run-to-run and break bitwise run-to-run determinism,
which the transient-detection methodology relies on. NW barrier passes
of 6x256 adds, once per block -- noise. Partial layout {m, l, acc[256]}
and the combine kernel are UNTOUCHED; scratch sizing unchanged.

expf stays exact expf in v1 of this kernel (12/position/warp); __expf /
exp2f folding is a separately-measured lever later. __ldcs streaming
hints likewise deferred -- one lever at a time.

## Numerics + gate policy (per the g64 precedent)

Register/vector re-tiling reorders fp accumulation => decode output is
NOT bitwise vs v1; greedy argmax ties can flip trajectories (observed in
the probes). Policy, mirroring g64: the bitwise canonical is REPLACED on
the default path by (a) unit tolerance gate fd2-vs-v1 at rel 2e-6 across
seq {1,47,1024,16384,61440} x ntok {1,5} x {fp8,fp16}, in-contract
inputs; (b) run-to-run bitwise determinism of fd2 itself; (c) full-corpus
--nll within noise of 7.1889 (fp8) / 7.1928 (fp16); (d) --nll-long 160K
bucket flatness (depth-quality, cheaper and more sensitive than needle
for numerics deltas); (e) --pf 200 IDENTICAL under the new default
(serial and batched decode share the kernel); (f) acceptance parity:
t/round on the 61K workload within ~2%; (g) NEW canonical md5 derived in
the landing commit, old canonical retired to v1-fallback runs
(Q27_FD=v1 restores the old kernel bit-for-bit).

## Perf targets and kill criteria

61K ground truth (wikitext continuation, server, fp8): 47.2ms/round
today, attn 27.5ms. Target: attn <= 10ms/round (=> round <= ~30ms, ~100+
t/s at 61K); ceiling if BW-bound ~3-6ms. Short-bench guard: canonical
2K bench within noise of 177.5 t/s (empty-split overhead at small seq).
KILL the default-flip (keep kernel opt-in) if: 61K round improves <10%,
or short bench regresses >2%, or any numerics gate fails.
