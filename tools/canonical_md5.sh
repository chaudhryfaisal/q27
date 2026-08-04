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

  case "$arch:$tier" in
    sm120:default) printf '%s\n' a2982c5197c627551b27d76a0a94b220 ;;
    sm120:q4s)    printf '%s\n' f64e7c02252ca4c40cea62db662205e0 ;;
    sm120:q5f)    printf '%s\n' 683f7f4450ca4c60837abdb603ee3237 ;;
    sm120:q6f)    printf '%s\n' 2a4d22eafcde63e962bf2408605fe502 ;;
    sm86:default) printf '%s\n' 6894254e3b1a184ee3802771ddd59c2b ;;
    metal-m4:q4s) printf '%s\n' 44310e1228cdd963f47ad8a2debb17d0 ;;
    *) return 1 ;;
  esac
}
