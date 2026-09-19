# Bonsai 2 gates (2026-09-18)

Scripts behind BUILDLOG (as) and (at). They run servers as transient user
units, stop production (`q27-38`) on the 5090 for the duration and relaunch
it, and stop the transcribers for 3090 work. Output dir: `OUT` (default
`/tmp/bonsai2-gates`).

- `fused_gate_3090.sh` -- identity gate for fused multi-slot rounds on the
  pure-T2 pack: five server configs (conductor-free reference, 1-slot fused,
  2-slot fused, solo-pinned, FIFO), four greedy prompts as concurrent pairs
  and alone; every text must match the reference. sm_86 only: the 5090's
  greedy is not width-invariant (BUILDLOG 2026-09-07 (f), 2026-09-18 (as)).
- `ladder_5090.sh` -- the 08-14 concurrency ladder (8 slots / 16K, C=1/2/4/8,
  temp 0.6) for the fused path vs the solo-pinned control
  (`Q27_BONSAI_FUSED=0`).
- `d2ab_5090.sh` -- DFlash2 drafter pack A/B on the Bonsai target: exact-mode
  identity vs plain (an sm_86 instrument; see above) and vgemm timing per
  prompt. `MODEL`, `PACKB`, `RUNS`, `TAG`, `NORELAUNCH` env overrides.
