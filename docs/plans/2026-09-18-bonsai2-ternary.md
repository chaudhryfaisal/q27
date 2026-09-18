# Bonsai 2 27B (ternary Qwen3.8-27B) on the q27 CUDA engine

Started 2026-09-18. Source of truth for the model: PrismML's whitepaper
(`bonsai-2-27b-whitepaper.pdf` in PrismML-Eng/Bonsai-demo) and their
llama.cpp fork (`/mnt/ai/projects/prism-llama`, built with CUDA sm_120,
self-tests pass, reference smoke on CPU produces sane text). Artifact:
`/mnt/ai/models/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5.95 GB,
sha256 verified against the Hub).

## What the model is

- Qwen3.8-27B, architecture unchanged: 64 blocks (48 GDN + 16 full
  attention), 5120 hidden, 17408 FFN, 24/4 heads x 256, vocab 248320. NO
  MTP block (block_count 64). Same tokenizer and chat template as the
  Qwen3.8-27B-MTP artifact q27 serves.
- Every projection is ternary {-1,0,+1} x one FP16 scale per 128 columns,
  including token_embd and output: attn_qkv/attn_gate/ssm_out (GDN layers),
  attn_q/k/v/output (attention layers), ffn_gate/up/down. High precision
  stays on ssm_alpha/ssm_beta (BF16, 5120x48), conv1d, A_log, dt_bias, all
  norms (F32) -- 26.2M parameters.
- Weights are stored in a rotated basis: W' = W R^T with
  R = (1/sqrt(1024)) H_1024 S applied blockwise along the INPUT dimension
  (block 1024, S a fixed +-1 diagonal, one sign vector per input width:
  5120, 6144, 17408). Inference computes W'(R x): sign flip, then a
  normalized Sylvester Walsh-Hadamard per contiguous 1024-block of the
  activation, before every rotated matmul. token_embd rows are stored
  rotated and get the INVERSE after lookup. ssm_out's input (the GDN value
  path, 6144) uses a head-grouped permutation of the transform
  (`prism.hadamard.gdn_v_grouped`). The residual stream itself is NOT
  rotated; only matmul inputs are.
- GGUF packing PTQ1_0 (type 143): per 128-block {qs[24] five trits/byte,
  qh[2] four trits/byte, fp16 d}, 28 bytes, 1.75 bpw. PQ2_0 (142) is the
  2-bit-slot twin. Rotation metadata lives in `prism.hadamard.*` keys.

## Approach: correctness first on existing kernels, then the ternary kernels

Ternary values are exactly representable in q27's Q4_G64 and Q8_G128
containers when the group scale is taken verbatim (code -1/0/+1 with
scale = d), so the model can run bit-faithfully on the kernels that exist
today. That isolates the only new piece of math -- the activation rotation
-- and gives a parity gate against the fork before any new GEMV is written.

### Phase 1 -- run it (correctness)

1. `tools/repack.py`: read PTQ1_0 (forge type 143, 28-byte blocks) and
   PQ2_0 (142, 34 bytes); decode trits per the fork's element map; emit an
   EXACT Q4_G64 blob for every rotated matrix (codes -1/0/+1 -> nibbles
   7/8/9, Q4 scale per 64 = the 128-group's d duplicated), Q8_G128 exact
   for token_embd and output (codes as int8, scale = d), F16 for
   ssm_alpha/ssm_beta, F32 for the rest. Round-trip gate: dequantize the
   GGUF block per the fork's `dequantize_row_ptq1_0` and our blob per
   FORMAT.md, bit-equal. Meta: `quant_policy = "bonsai2-q4x-v1"` (exact
   ternary in Q4 containers), the `prism.hadamard.*` metadata verbatim
   (block size, transform, sign widths + values, rotated tensor list,
   inverse list, gdn_v_grouped), `qwen35.block_count = 64`, and
   `general.name` carrying "qwen38" so the XML dialect and 3.8 template
   rules key correctly (the GGUF says "Hf"). No MTP join.
2. Loader/engine metadata: `validate_arch` accepts block_count 64 with no
   nextn block (the manifest already treats blk.64 as optional); a
   `has_mtp` flag gates `mtp_warm_T` (engine.cuh:5331) and refuses the
   MTP ladder spec modes; DFlash2 stays available (the drafter reads the
   unrotated residual stream, so the pack is compatible in principle;
   acceptance is measured, not assumed).
3. Rotation kernels (new, fp32, in place): `hadamard1024(x, signs, n)` =
   sign flip then the 1024-point normalized butterfly (1/32 exact), and
   the inverse (butterfly then sign). Decode single-lane, multi-lane
   (verify width up to 8) and prefill (T rows) variants; signs in device
   memory keyed by width; the ssm_out grouped-permutation variant.
   Bitwise-deterministic operand order like `turbo3_butterfly128`.
4. Insertion points (engine.cuh, from the map in this plan's appendix):
   gdn_block xin -> a rotated COPY (ssm_alpha/beta read the raw xin);
   attn_block xin in place; ffn xin and ffn_g in place; attnout and og
   before attn_output/ssm_out; the head input after output_norm;
   embedding inverse after lookup (decode, multi-lane, prefill). The
   multi-lane verify path un-fuses `rmsnorm3q5` into norm -> rotate ->
   quant for this pack. Prefill uses the T-row variant before `qxT`.
5. Gates: (a) unit test of the rotation against a NumPy reference on
   random vectors (fwd, inv, inv(fwd) = x, grouped perm); (b) logit parity
   vs the fork on fixed token sequences (a small libllama tool dumps
   full logits; compare top-1 agreement and max |delta| over prompt
   positions, thresholds from the flip-gate history); (c) greedy canonical
   text agreement on the tier README prompts; (d) the wsum gate for the new
   artifact; (e) `make test` + the loader contract test updated.

### Phase 2 -- ternary decode (speed)

6. `gemv_t2` / `gemv_t2_n<N>` in kernels.cu: 2-bit codes, fp16 scale per
   128, dp4a against the int8 g32 activations like `k_gemv_q4` (codes
   -1/0/+1 are the nibble kernel with a different unpack; select-form on
   floats is the Metal design). Dispatch at mm / mm5 (fix the bare else) /
   mtp_mm / mtp_mm1 / vgemm; `__launch_bounds__` tiers. Target: a decode
   step that streams ~6 GB, bandwidth-bound.
7. T2 head + DFlash2 head enum (or keep the exact Q8 head, 1.3 GB, if the
   head GEMV is not on the critical path).
8. Artifact `bonsai2-t2-v1`: T2 for decode, exact Q4 shadow (`.q4x`
   sidecar, like `.pf4`) for prefill until Phase 3.

### Phase 3 -- ternary prefill, fused rotation

9. `bool Q4IN` -> a dtype enum through the MMA prefill path; T2 dequant
   into the int8/bf16 tiles; drop the shadow.
10. Fuse sign+WHT into the GEMV prologue if the standalone kernel shows
    in the round wall (the fork reports it as a top non-matmul cost at
    batch 1; q27's CUDA graphs already remove the launch part).
11. Tier README numbers (PPL, HumanEval+, needle), flip gate on the
    served-dialect corpus, DFlash2 acceptance, then the agentic campaign
    and the planning-turn probes.

## Log

- 09-18 10:20: converter + rotation kernels + decode wiring land. First run
  on the 3090: greedy text == fork, next-token logits vs fork corr 0.99997
  (max |d| 0.075, KL 2e-5), kernel tests bitwise vs CPU. The CLI's main
  path feeds the prompt PER TOKEN (step_with), so this validated decode
  only; the teacher-forced `--nll-long --flip-dump` path (prefill_chunk +
  head on T rows) gives NLL ~12.7 on the same tokens where the fork gives
  2.1 and the production model 2.5 -> the batched-prefill wiring has a
  defect. `--pfdbg 1` (serial vs batched layer-0 stage diff) is the
  localization tool; its MTP touches are now gated on has_mtp.
- 09-18 10:40: the defect was a build-ordering slip (the CLI head-row
  rotation was inserted after that build had started). Rebuilt: 384-position
  parity vs the fork top-1 0.9974, NLL 2.0792 vs 2.0796, mean|dNLL| 0.011.
  Serial-vs-batched prefill (--pfdbg 1/8) shows only g32/g64 activation
  quant noise. PHASE 1 GATE PASSED.
- 09-18 10:38: server fallback landed (plain_round: token-graph replay,
  first token = the prefill tail's argmax; sampled = sample_round;
  build_spec_graphs skips without an MTP block). Server smoke on the 3090:
  46 t/s plain, thinking on, greedy + sampled answers correct, prefix cache
  hits. PHASE 1 COMPLETE.
- Phase 2 in flight: gemv_t2 / gemv_t2_n (dp4a over 2-bit codes, value =
  code-1 so sum = dp4a - isum; scale per 128), device-side interleave at
  upload so the Q4 kernels' even/odd activation words apply unchanged,
  loader accepts T2 on CUDA (B1/T3 still refused), mm/mm5/mtp dispatch
  (mm5's bare else now fails loud), prefill reads the exact-Q4 `.q4x` shadow
  through TP(il, leaf), repack `--bonsai2-container t2`.

## Risks and open questions

- DFlash2 acceptance on the ternary target (drafter trained on BF16).
- The `gdn_v_grouped` permutation must match the fork exactly; the parity
  gate is the only proof.
- Quality on long-horizon agentic work is the model's (their SWE-bench
  Verified 60.8 vs 80.6), not the engine's; report it as such.
- Development on the 3090 (sm_86 w8 build, vox stopped during tests) to
  keep production on the 5090; 5090 only for Phase 2/3 measurements.

## Appendix: change surface (from the engine map, 2026-09-18)

- loader.cpp:105-119 `cuda_weight_dtype_supported` (Phase 2: T2 true);
  loader.cpp:141 token_embd pinned to Q8_G128 (keep in Phase 1);
  test_loader_contracts.cpp asserts the current rejection.
- engine.cuh:132-183 `validate_arch` (block_count 65 + nextn 1 today).
- engine.cuh dispatch: 1484 mm, 2048 mm5 (bare else), 1727 mtp_mm, 1752
  mtp_mm1, 3864 mmT; vgemm.cu:424/335; dflash2.h:99-111 + dflash2.cu:836.
- Insertion points: gdn_block 1504/1518, attn_block 1540/1576, ffn
  1581/1585, head 1621; multi-lane gdn_pre 2069, gdn_post 2160, attn_pre
  2174, attn_post 2229, ffn_pair 2244, verify driver 2322-2366
  (rmsnorm3q5 fused at 2343/2348/2360; conductor.h:700-742 mirrors it);
  prefill gdn_block_T 3882, attn_block_T 3902, ffn_T 3935, qxT 3831.
- Embedding: kernels.cu:665 (decode), spec3.cu:1331 (multi-lane),
  prefill.cu:1059 (T rows); seven hard-cast call sites.
- Existing WHT: prefill.cu:2670 `k_wht_T` / spec3.cu:620 `k_wht3` are
  128-point with fixed S1/S2 (KV codec) -- a template for the 1024-point
  kernel, not a drop-in.
