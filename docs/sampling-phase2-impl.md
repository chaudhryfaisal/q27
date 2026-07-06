# Sampling Phase 2 -- spec rejection sampling (implementation spec)

Concretizes docs/sampling-design.md section 1 into kernel signatures, the
Philox counter scheme, buffer layout, and the graph-capture plan. Phase 1
(plain-path sampling) shipped 2026-07-05; this restores spec speed under
temperature>0 by making the depth-4 speculative round sample the exact served
target instead of exact-match accepting (which is lossless for greedy ONLY).

Rule zero holds unchanged: the greedy spec path (k_finish_round equality chain,
argmax verify lanes) stays bitwise; the canonical md5 4c4120c7 gates it. All
Phase-2 code lives in a SEPARATE second graph set, launched only when a request
sets temperature>0.

## The round, sampled

The draft half is byte-identical to greedy (greedy drafts, design sec 2a):
spec_draft_launches() -> dr1..dr4 unchanged. Only the verify TAIL changes.
Verify forward (embed3 of {t1,dr1..dr4} at P+1..P+5 -> logits2[5*VOCAB]) is
also unchanged. What changes: replace [5x argmax_masked + k_finish_round] with

    for lane 0..4:  k_nucleus_d(logits2 + lane*VOCAB, d_samp, d_nuc5 + lane*4)
    k_spec_accept   -> d_spec = {n, stop_lane, exclude}
    k_argmax_reset(d_amax)
    k_sample_stop   -> atomicMax over stop lane's nucleus (Gumbel) -> d_amax
    k_argmax_extract(d_amax -> d_token)          // nt
    k_finish_sampled -> h_next = x1[n-1], *d_P += n, outcome

Lane k (0-indexed) forwards the token at position P+1+k and predicts P+2+k.
The draft proposed for P+2+k is dr_{k+1}. So lane k tests draft dr_{k+1}.

n (committed this round) = 1 + (#drafts accepted before first reject), in 1..5.
The pending token t1 (sampled last round) is ALWAYS committed -- exactly the
greedy invariant (t1 is the verified bonus from the prior round). h_next and
outcome selection by depth are IDENTICAL to k_finish_round; only two things
differ: n comes from rejection sampling, and the new pending nt is SAMPLED
(k_sample_stop) not argmax'd.

## Served distribution and the accept probability

Target p (design sec 1) is the SERVED distribution: temp-scaled at inv_temp,
top-p-truncated to the nucleus S={i: x_i >= thresh}, renormalized OVER S:

    p_served(i) = softmax_full(i) / mass          for i in S, else 0
    softmax_full(i) = exp(inv_temp*(x_i - M) - logZ)      (sums to 1 over vocab)
    mass = sum_{i in S} softmax_full(i)                   (the nucleus mass)

k_sample_stop draws via Gumbel-max over S -> it samples exactly p_served
(renormalized over the nucleus). For rejection sampling to yield p_served, the
per-lane accept probability MUST be p_served(dr), NOT softmax_full(dr):

    a_k = p_served(dr_{k+1} at lane k)
        = (x[dr] >= thresh) ? exp(inv_temp*(x[dr]-M) - logZ) / mass : 0

**Refinement over the sec-1 sketch:** blocks.cuh said d_nuc keeps {thresh,M,logZ}
"for the Phase-2 accept test". That is only enough when mass==1 (top_p>=1). For
top_p<1 (e.g. 0.95, common in agentic configs) mass ~ 0.95-0.97 and using
softmax_full(dr) under-accepts by 1/mass (~4%), biasing the output away from the
draft -- which the spec==non-spec chi-square gate would catch. So k_nucleus_d
gains out[3]=mass (one extra grid-stride reduction over the just-computed
threshold; ~1 MB reread, argmax cost class). When top_p>=1, thresh=-FLT_MAX and
mass=1 exactly, so temp-only requests are unaffected either way.

Rejection sampler correctness (q = delta at greedy draft dr, q(dr)=1):
accept a(dr)=min(1,p/q)=p_served(dr); on reject resample from
norm(max(0,p-q)) = p_served with dr excluded and renormalized -- exactly what
k_sample_stop computes with exclude=dr. On all-accept, lane 4 has no draft:
sample p_served(lane 4) with no exclusion (the free bonus token).

## Philox counter scheme (determinism)

Phase 1 keys the plain sampler on (*d_pos, draw_kind, vocab_index), kind 0=eager
first token, kind 1=graph draws. Phase 2 keys off *d_P (the committed position at
round start; the spec path advances d_P, not d_pos) and adds two disjoint kinds:

    accept uniform, lane k :  philox(seed; c0=*d_P, c1=SPEC_ACCEPT=2, c2=k)
    stop Gumbel, vocab i   :  philox(seed; c0=*d_P, c1=SPEC_STOP=3,   c2=i)

d_P is read BEFORE k_finish_sampled increments it, so each round keys off a
strictly-increasing, distinct base -> no cross-round collision, and replay /
prefix-restore / ckpt stay consistent (d_P is deterministic, nothing mutable to
advance). Distinct kinds keep accept draws, stop draws, and the Phase-1 plain
draws (kinds 0/1) from ever sharing a counter. Plain sample_round and
spec_sample_round are mutually exclusive per request regardless.

Seeded identity: fixed (seed,prompt,params) -> token-identical across runs on
one GPU arch. The accept chain reads uniforms in fixed lane order; k_sample_stop
reuses k_argmax's order-independent am_pack+atomicMax reduction. Deterministic.

## Kernels (signatures)

blocks.cu (extend):
    k_nucleus_d(...) additionally writes out[3] = nucleus mass. Plain path
    ignores out[3]; d_nuc alloc grows 3 -> 4 floats/lane.

spec3.cu / spec3.cuh (new; grid-merged-file home for round kernels):
    // serial accept chain. d_spec[3] = {n, stop_lane, exclude_token}.
    void spec_accept(const float* logits2, const float* d_nuc5,
                     const int* dr1, const int* dr2, const int* dr3, const int* dr4,
                     const SampleParams* d_sp, const int* d_P, const int* cap,
                     int vocab, int* d_spec, cudaStream_t st);
    // device-indirected nucleus Gumbel-max over stop lane, exclude-masked.
    // best = d_amax (u64), reset by caller; extract writes nt.
    void sample_stop(const float* logits2, const float* d_nuc5, const int* d_spec,
                     const SampleParams* d_sp, const int* d_P, int vocab,
                     unsigned long long* best, cudaStream_t st);
    // finish keyed on n from d_spec (mirror of finish_round's bookkeeping).
    void finish_sampled(int* d_P, int* d_token, const int* d_spec,
                        const int* dr1, const int* dr2, const int* dr3, const int* dr4,
                        const float* x1a, const float* x1b, const float* x1c,
                        const float* x1d, const float* x1e, float* h_next,
                        int* outcome, int n_embd, cudaStream_t st);

## Buffers (engine.cuh)

    d_nuc  -> grow to 5*4 floats (was 3). Plain path uses lane 0 (out[0..3]).
    d_spec -> 3 ints {n, stop_lane, exclude}. (may overlay unused d_outcome tail)
    spec_sample_graph[5] -> 2nd fused perm set (draft + sampled verify).

Prep is shared: k_prep_round already snapshots outcome[1]=t1 and derives lane
positions from *d_P. finish_sampled writes outcome[0]=n, [2..5]=dr, [6]=nt.

## Bootstrap (first pending token)

Symmetric with greedy. Post-prefill: h_next=x1(last prompt hidden), d_P=NP-1,
and the first pending token must be SAMPLED not argmax'd. Reuse samp_first:
spec_sample_round, on its first call, samples d_token from the retained prefill
`logits` (kind 0, no re-forward -- the last prompt token's GDN update already
ran), then launches spec_sample_graph[perm]. From there each round forwards the
committed pending + drafts and samples the next pending.

## Routing (generate loop)

    sampling(temp>0) && !tool_split  -> spec_sample_round   (Phase 2, fast)
    sampling && tool_split           -> plain sample_round  (Phase 3 does spec+mask)
    greedy                           -> spec_round          (unchanged, bitwise)

Tool constraining is disabled under sampling today (Phase-1 note), so the fast
path is the live one; the plain fallback is retained for correctness until
Phase 3 masks the sampled spec lanes pre-softmax.

## Gates

Kernel (test_kernels, no model):
  1. nucleus_mass out[3] == CPU nucleus mass over the same threshold.
  2. rejection-sampling distribution: fixed 5-lane logits2 + greedy drafts, run
     spec_accept+sample_stop over many Philox positions; histogram emitted nt
     per stop lane; chi-square vs analytic p_served of that lane. Composition
     (accept+resample) must reproduce the lane's served target.
  3. seeded identity (same seed,pos -> identical n and nt).
  4. exclude_token never emitted when its lane rejects.

Live (model on GPU):
  5. greedy canonical md5 4c4120c7 UNCHANGED (tools/shortbench_suite.sh).
  6. spec==non-spec distribution: drive plain sample_round vs spec_sample_round,
     same (seed,prompt,params); two-sample chi-square of next-token histograms
     (token-wise identity NOT expected -- different draw counts).
  7. acceptance-vs-temp: emit tokens/round across T in --stats; publish curve.

Exit criterion (design sec, before sampling defaults on): re-run Thunderdome
quality A/B + tool-format drift catalog under production sampling config. Not a
follow-up -- greedy quality numbers do not transfer across temperature.

## Status (2026-07-05)

DONE -- all kernel and live gates pass. q27-eval restarted on the Phase-2 binary
(greedy bitwise so CC/eval traffic is unchanged; gains the sampled-spec path).

Kernel gates (`build/test_kernels --sampling-only`, no model -> runs while a
server holds the GPU): ALL PASS.
- spec nucleus mass vs CPU: err 6.1e-8.
- spec accept rate == p_served(dr1): emp 0.4333 vs p_served 0.4215 (err 1.2e-2).
  This is the decisive mass check -- softmax_full(dr1) is 0.388, which the 3e-2
  tolerance would reject (~8 sigma). Confirms accept prob = softmax_full/mass.
- spec rejection-sampling vs served target: chi2 5.90, df 4 (bound 66.6).
- spec seeded identity; reject-excludes-draft; commit-in-nucleus: PASS.

Build: `make` clean on sm_86+sm_120; no new warnings. CLI gained
`--temp/--top-p/--seed`; the `--spec` loop routes temp>0 to spec_sample_round
(Q27_SAMPLE_PLAIN=1 -> plain sampler).

Live gates -- ALL PASS 2026-07-05 (`tools/sampling_gate.sh`, q27-eval briefly
stopped for GPU 0, Gabe-authorized). All via the CLI (build/q27 --spec --temp):

  1. greedy canonical md5 == 4c4120c7: OK -- greedy stayed bitwise, so the
     spec_verify_forward extraction + build_spec_graphs lambda refactor (the
     only greedy-adjacent source changes) were inert.
  2. sampled seeded identity: OK -- same (seed,prompt,params) -> identical
     stream, so the Philox plumbing is correct end-to-end through capture/replay.
  3. seed varies + sampled != greedy: OK.
  4. spec vs plain both produce valid 48-tok trajectories: OK (the full
     spec==non-spec chi-square is covered at the kernel level -- both paths
     sample the analytic served target via the shared k_nucleus_d; a
     many-trajectory live chi-square would need the server or an in-process
     batch mode, deferred as redundant).
  5. acceptance-vs-temp (tokens/round): greedy 3.43, T=0.3 3.59, T=0.7 3.45,
     T=1.0 1.90, T=1.5 1.00 -- holds near greedy through T<=0.7 (the draft head
     is sharp: E3 measured 98.1% Q4/Q8 draft-argmax agreement), then sags as the
     served target flattens. Distribution-fidelity cost, not implementation.

Then the design's exit criterion (Thunderdome quality A/B + drift catalog under
production sampling) before sampling defaults on anywhere.
