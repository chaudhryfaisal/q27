#!/usr/bin/env bash

# Return the published canonical digest for one exact architecture/tier pair.
# CUDA entries hash the `generated:` line used by sampling_gate.sh. Metal
# entries hash the space-delimited token-id line emitted by --dump-token-ids.
canonical_md5_for() {
  local arch="$1" tier="$2"

  case "$arch" in
    sm120|sm_120|blackwell|5090) arch=sm120 ;;
    sm86|sm_86|ampere|3090|a40) arch=sm86 ;;
    metal-m4|apple-m4|m4) arch=metal-m4 ;;
  esac
  case "$tier" in
    default|vanilla|qwen36-27b-mtp) tier=default ;;
    q4s|q4s-v1) tier=q4s ;;
    q5f|q5f-v1) tier=q5f ;;
    q6f|q6f-v1) tier=q6f ;;
  esac

  # sm86:default was derived on an RTX 3090 (82 SMs) and independently matched
  # the A40 result (84 SMs) byte-for-byte, establishing SM-count independence
  # within sm_86. A non-Blackwell `6894254e...` result therefore reproduces its
  # own architecture's canonical rather than failing to reproduce Blackwell.
  # Unlisted pairs require a same-device upstream/candidate differential.
  # metal-m4:q4s was re-derived after the serving stack promoted the measured
  # reciprocal-multiply activation quantization and revised shader arithmetic.
  # The new trajectory was reproduced byte-for-byte on a base Apple M4 and an
  # independent 24 GB Apple M4 Pro using the same q4s artifact (artifact MD5
  # 7e5454e0c0ded717136ad3e42634ba25), preserving the shared family canonical.

  case "$arch:$tier" in
    sm120:default) printf '%s\n' a2982c5197c627551b27d76a0a94b220 ;;
    sm120:q4s)    printf '%s\n' f64e7c02252ca4c40cea62db662205e0 ;;
    sm120:q5f)    printf '%s\n' 683f7f4450ca4c60837abdb603ee3237 ;;
    sm120:q6f)    printf '%s\n' 2a4d22eafcde63e962bf2408605fe502 ;;
    sm86:default) printf '%s\n' 6894254e3b1a184ee3802771ddd59c2b ;;
    metal-m4:q4s) printf '%s\n' f301095522174bdb99f75ec840ad1389 ;;
    *) return 1 ;;
  esac
}

# Published sampled-seed anchors. The EXACT command is part of the anchor;
# every flag below is load-bearing. In particular `--spec` selects a
# different-but-valid sampled trajectory: omitting it produced a plausible
# WRONG md5 at v0.5.0 gating and again on 2026-08-18 (recorded as "anchor
# broken"; it never was -- see BUILDLOG 2026-08-18 (k)).
#   build/q27 <model> --tokens "760,6511,314,9338,369" --ctx 2048 --spec \
#     -n 64 --temp 0.7 --top-p 0.95 --seed 42 | grep '^generated:' | md5sum
sampled_md5_for() {
  local arch="$1" tier="$2"

  case "$arch" in
    sm120|sm_120|blackwell|5090) arch=sm120 ;;
    sm86|sm_86|ampere|3090|a40) arch=sm86 ;;
    metal-m4|apple-m4|m4) arch=metal-m4 ;;
  esac
  case "$tier" in
    default|vanilla|qwen36-27b-mtp) tier=default ;;
    q4s|q4s-v1) tier=q4s ;;
    q5f|q5f-v1) tier=q5f ;;
    q6f|q6f-v1) tier=q6f ;;
  esac

  case "$arch:$tier" in
    sm120:q4s) printf '%s\n' 900031e9b86df8f52493e6c1f4040c2e ;;
    *) return 1 ;;
  esac
}
