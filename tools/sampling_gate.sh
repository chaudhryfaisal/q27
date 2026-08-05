#!/usr/bin/env bash
# Sampling Phase 2 LIVE gates (docs/sampling-phase2-impl.md sec Gates).
# Runs on GPU 0 and loads the 17.7GB model, so free GPU 0 first (stop any
# resident q27-server). Kernel-level correctness is already proven by
# test_kernels --sampling-only; these confirm the end-to-end engine plumbing.
#
# Usage: tools/sampling_gate.sh [model.q27]
set -u
MODEL="${1:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.q27}"
BIN="$(dirname "$0")/../build/q27"
# baseline greedy canonical: vanilla qwen36-27b-mtp (benchmark standard
# 2026-07-09); other tiers/fine-tunes override via CANON_MD5= env --
#   q4s: f64e7c02252ca4c40cea62db662205e0
#   q5f: 683f7f4450ca4c60837abdb603ee3237  (Q4-head + ffn_down, 5.30bpw)
#   q6f: 2a4d22eafcde63e962bf2408605fe502  (Q4-head + ffn_down + ffn_gate, 6.11bpw)
#   Qwopus: 4c4120c7...
#
# THE CANONICAL IS PER-ARCHITECTURE. sm_86 codegen/ULP differences fork the
# tie-heavy trajectory exactly like a cross-build lottery, so the bitwise
# contract holds WITHIN an arch, not across them (BUILDLOG 2026-07-09):
#   sm_120 (5090):  a2982c5197c627551b27d76a0a94b220   <- the default below
#   sm_86  (3090, A40, and other GA10x): 6894254e3b1a184ee3802771ddd59c2b
#     ^ DERIVED LOCALLY 2026-08-05 on our own RTX 3090 with CANON_ARCH=sm86,
#       matching the A40 value reported in issue #7 exactly. Two sm_86 parts
#       with DIFFERENT SM counts (3090 82, A40 84) agree bit-for-bit, so the
#       canonical is SM-count-independent within an architecture -- consistent
#       with the CLI path taking fd2 at a fixed FD2_NS rather than any
#       SM-count-derived split.
# A non-Blackwell run that reports 6894254e has REPRODUCED its own canonical,
# it has not failed to reproduce this one. tools/shortbench_suite.sh learned
# this in 07-09 and grew BENCH_GPU=; this gate did not, and in 2026-08-04 that
# omission cost an outside contributor on an A40 a full gate run (q27 issue #7)
# because the comment above listed tier overrides and said nothing about arch.
# CANON_ARCH=sm86 selects the Ampere value; CANON_MD5= still overrides outright.
#
# CONTRIBUTORS ON OTHER SILICON: if no canonical is published for your arch, the
# correct gate is the SAME-DEVICE differential -- build unmodified upstream and
# your candidate on the one box and require byte-identical `generated:` lines.
# That is what per-arch means; it is not a weaker substitute.
case "${CANON_ARCH:-}" in
  sm86|ampere|3090|a40) CANON_DEFAULT=6894254e3b1a184ee3802771ddd59c2b ;;
  *)                    CANON_DEFAULT=a2982c5197c627551b27d76a0a94b220 ;;
esac
CANON_MD5="${CANON_MD5:-$CANON_DEFAULT}"
CANON_IDS="760,6511,314,9338,369"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
fail=0

gen() { # $1=outfile ; extra args after
  local out="$1"; shift
  "$BIN" "$MODEL" --tokens "$CANON_IDS" --ctx 2048 --spec "$@" >"$out" 2>>"$tmp/err.log"
  grep '^generated:' "$out"
}

echo "== gate 1: greedy canonical md5 (bitwise -- greedy path must be untouched)"
gen "$tmp/canon.out" -n 128 >/dev/null
md5="$(grep '^generated:' "$tmp/canon.out" | md5sum | cut -d' ' -f1)"
if [[ "$md5" == "$CANON_MD5" ]]; then echo "  OK  md5=$md5"
else echo "  FAIL md5=$md5 want $CANON_MD5" >&2; fail=1; fi

echo "== gate 2: sampled seeded identity (same seed -> identical stream)"
a="$(gen "$tmp/s1.out" -n 48 --temp 0.85 --top-p 0.95 --seed 42)"
b="$(gen "$tmp/s2.out" -n 48 --temp 0.85 --top-p 0.95 --seed 42)"
if [[ "$a" == "$b" && -n "$a" ]]; then echo "  OK  (identical across runs)"
else echo "  FAIL seeded runs differ" >&2; fail=1; fi

echo "== gate 3: seed varies + sampled != greedy (sanity)"
c="$(gen "$tmp/s3.out" -n 48 --temp 0.85 --top-p 0.95 --seed 7)"
g="$(grep '^generated:' "$tmp/canon.out" | head -c 400)"
[[ "$a" != "$c" ]] && echo "  OK  seed 42 != seed 7" || { echo "  FAIL seeds gave identical output" >&2; fail=1; }
[[ "$a" != "$g" ]] && echo "  OK  sampled != greedy" || { echo "  FAIL sampled == greedy" >&2; fail=1; }

echo "== gate 4: spec==plain trajectories both valid (full chi-square is kernel-proven)"
sp="$(gen "$tmp/spec.out" -n 48 --temp 0.85 --top-p 0.95 --seed 3)"
pl="$(Q27_SAMPLE_PLAIN=1 gen "$tmp/plain.out" -n 48 --temp 0.85 --top-p 0.95 --seed 3)"
ns="$(wc -w <<<"$sp")"; np="$(wc -w <<<"$pl")"   # word count includes the 'generated:' token
if [[ "$ns" -ge 40 && "$np" -ge 40 ]]; then echo "  OK  spec=$((ns-1)) tok, plain=$((np-1)) tok (both produced; distributions match by kernel gate)"
else echo "  FAIL a path under-produced (spec=$ns plain=$np words)" >&2; fail=1; fi

echo "== gate 5: acceptance-vs-temp (tokens/round should sag as T rises)"
for T in 0.0 0.3 0.7 1.0 1.5; do
  if [[ "$T" == "0.0" ]]; then
    "$BIN" "$MODEL" --tokens "$CANON_IDS" -n 96 --ctx 2048 --spec >"$tmp/t.out" 2>/dev/null
    tag="greedy"
  else
    "$BIN" "$MODEL" --tokens "$CANON_IDS" -n 96 --ctx 2048 --spec --temp "$T" --top-p 0.95 --seed 1 >"$tmp/t.out" 2>/dev/null
    tag="T=$T"
  fi
  tpr="$(sed -n 's/.*= [0-9.]* t\/s (\([0-9.]*\) tokens\/round.*/\1/p' "$tmp/t.out")"
  printf "  %-8s %s tokens/round\n" "$tag" "${tpr:-?}"
done

echo ""
[[ "$fail" == 0 ]] && echo "SAMPLING GATES: ALL PASS" || { echo "SAMPLING GATES: FAILED"; cat "$tmp/err.log" >&2; }
exit "$fail"
