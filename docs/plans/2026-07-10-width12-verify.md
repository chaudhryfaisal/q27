# Width-12 verify: widen the lane architecture 8 -> 12

Synthesized 2026-07-10 from a 4-reader inventory workflow (wf_d925cde8-1f5
journal has full file:line detail). Motivation: suffix drafter jammed at the
width-8 cap on live traffic (7.49/8 tok per fired round, 63% of decode;
uncapped AL ~10.7) + MTP ceiling 8 priced +4.9% on hot chains. fdmma verify
attention already handles W<=16 (M=6W dense rows <= 96; only its FCP3/FIP3
p[8] plumbing caps it).

## Decisions (from the inventory)

1. SCOPE = FULL widening, not suffix-only. The hoped-for scope cut is
   illusory: GDN role sets, perm modulus, graph zoo, p[] structs, prep/
   finish signatures, reserve math are all keyed on VERIFY WIDTH (commit
   count), not on who drafted. Full-vs-suffix delta is ~20KB/lane of MTP
   buffers + ladder arrays. Take it all; DECOUPLE POLICY instead: v1 keeps
   the MTP ladder at 4..7 and lets ONLY suffix rounds use widths 9..12
   (suffix proposals = 11). MTP ceilings 8..10 get priced later on the
   already-measured d10 chains before enabling.
2. WIDTH = 12, not 16. VRAM: +4 GDN role sets = +627MB per engine (x2
   slots ~= +1.26GB; fits the ~5GB eval-server headroom). W=16 costs
   +1.26GB/engine (+2.5GB total, too tight with 2 slots). W=12 proposals
   (11) capture ~all of suffix AL 10.7. Graph-zoo perm dimension 12 = 1.5x
   captures (vs 2x). Revisit 16 only if live AL at 11 is still jammed.

## Inventory highlights (full detail in the workflow journal)

- Per-lane marginal VRAM excl roles: ~4.66MB (activations 0.5 + logits2
  0.99 + FD scratch 3.17). +4 lanes = +18.6MB. Trivial next to roles.
- GDN roles: 3.27MB/role/layer x 48 layers = 156.9MB per role set; roles
  selected (role+perm)%8, allocated as S/S_spare1..7 + rings. 8->12 =
  +627MB/engine. gdn_state_bytes accounting + reset() must follow (NOTE:
  reset() memsets only spares 1-5 today -- pre-existing bug/stale, fix in
  passing; comments still say mod-7/mod-6 -- stale, actual mod-8).
- Perm invariant: modulus >= max commit n = W (roles 0..W-1 distinct);
  perm advances (n-1) mod M. Every perm value needs its own captured
  graph: all graph arrays [*][8] -> [*][12]; capture loops likewise.
- Signatures at param limits: prep_round (17 params) / finish_round (25)
  must convert to pointer-array structs (IP3-style) rather than grow.
  outcome -> W+2 = 14 ints. em[]/oc[]/h_sfx_prop/d_mask_ids/h_mask_ids5,
  gate hists, argmax_masked chain, logits2 offsets: all 8 -> 12.
- gemv_q4_n/q8_n: Q4Lanes arrays are [10]; N=9,11,12 NOT instantiated and
  N>8 has a documented register/occupancy cliff (N=8 pinned 4 CTAs, N=10
  natural 3). P1 must MEASURE N=12; contingencies: 2-pass 6+6 (weights
  stream twice -- bad) or the mma16 NT=16 MMA GEMM (tools/mma16_bench.cu,
  76% SOL, flat W2..16 -- the likely winner if N=12 gemv cliffs).
- Serial GDN chains at W=12: 2W launches/layer, ~0.32ms/lane device-serial.
  Chunk kernels (bitwise-validated, shelved): conv chunk 5-7x ships
  opportunistically; delta chunk only 1.29x (state-write-bound) -- the
  deferred-snapshot variant (final-S + n-step fixup) is the real fix if
  the W=12 width curve shows GDN dominating post-fdmma.
- ctx_round_reserve: max(gate_maxd, W_wide-1) + 2; 8 server call sites.
- MTP-KV holes: suffix-committed positions already skip MTP rows (known
  acceptance-dip-not-correctness); W=12 widens the stale window per burst
  -- keep the existing measure-first posture, telemetry already exists.
- fdmma: lift static_assert W<=8, widen FCP3/FIP3 to p[16] (kernel
  geometry verified 16-ready: LIVE_WARPS <= 6 at 192 threads).

## Phases

- P0 (mechanical, 2-3 sessions): p[16] struct family (IP3/CP3/P3/XQ3/
  FCP3/FIP3/Q4Lanes) + lane buffers i..l + role sets + perm mod 12 +
  graph arrays [12] + prep/finish struct-refactor + outcome 14 + reserve.
  GATES: canonical EXACT (widths <= 8 byte-identical -- pure plumbing;
  perm-modulus change means role addressing must be proven equivalent at
  n <= 8: same physical buffer sequence, new modulus never reached),
  test_kernels, sanitizer, replay byte-identity vs pre-widen binary.
- P1 (1 session): gemv N=12 instantiate + MEASURE vs mma16 contingency;
  argmax/logits chains to 12; capture verify_graph_w[9..13]; suffix
  decouple (propose 11, width 12) behind Q27_SUFFIX_W knob.
- P2 (1 session): fdmma_test at W=12; accept_ab replay A/B; live T8
  full-stack trial (does suffix AL reach ~10.7? wall delta?); width-curve
  re-measure at 12.
- P3 (optional): MTP ceilings 8..10 priced via ladder_price.py on the d10
  chains (data exists); GDN deferred-snapshot chunk if curve says so.

## Predicted value (to be re-measured at P2)

Suffix rounds 63% of decode at 7.49 -> ~10.5 tok/round removes ~25-30% of
suffix rounds ~= +10-15% wall on live CC traffic, on top of the 249 t/s
stack. MTP ceiling 8+ adds up to ~+5% on hot traffic later. Cost: +1.26GB
VRAM total, 1.5x graph captures at server start.
