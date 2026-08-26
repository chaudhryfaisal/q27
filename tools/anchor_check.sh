#!/usr/bin/env bash
# Canonical-anchor check that DIAGNOSES its own failures.
#
# Why this exists: on 2026-08-16 and 2026-08-18 a greedy anchor run returned
# `8196e65e` instead of the canonical, twice, and neither sighting could be
# chased afterwards -- the only surviving record was an 8-character md5 prefix.
# Nobody had captured which binary, which artifact, which device, or which
# Q27_* latches were live. A 2026-08-18 investigation then failed to reproduce
# it in 44 controlled runs (BUILDLOG (l)), so the next sighting may be the last
# chance to catch the mechanism. On mismatch this script dumps everything that
# could plausibly select a different trajectory, THEN repeats to establish
# whether the divergence is sticky (a real config difference: wrong artifact,
# wrong tier, wrong device) or transient (the open non-determinism).
#
# Usage: tools/anchor_check.sh [model.q27] [-n REPEATS]
#   CANON_ARCH=sm120 CANON_TIER=q4s   (tier auto-detected from the path)
#   CANON_MD5 / SAMPLED_MD5           explicit escape hatches
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp-q4s.q27}"
[[ $# -gt 0 ]] && shift
REPEATS=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) REPEATS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

BIN="${Q27_BIN:-$HERE/../build/q27}"
CANON_IDS="760,6511,314,9338,369"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# shellcheck source=canonical_md5.sh
source "$HERE/canonical_md5.sh"

CANON_ARCH="${CANON_ARCH:-sm120}"
# Tier is a property of the ARTIFACT, and feeding the wrong one looks exactly
# like an engine regression (2026-08-16 gotcha). Derive it from the filename.
if [[ -z "${CANON_TIER:-}" ]]; then
  case "$MODEL" in
    *-q4s*) CANON_TIER=q4s ;; *-q5f*) CANON_TIER=q5f ;;
    *-q6f*) CANON_TIER=q6f ;; *) CANON_TIER=default ;;
  esac
fi
[[ -n "${CANON_MD5:-}" ]]   || CANON_MD5="$(canonical_md5_for "$CANON_ARCH" "$CANON_TIER")" || {
  echo "no published canonical for $CANON_ARCH:$CANON_TIER" >&2; exit 2; }
[[ -n "${SAMPLED_MD5:-}" ]] || SAMPLED_MD5="$(sampled_md5_for "$CANON_ARCH" "$CANON_TIER" 2>/dev/null || true)"

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

greedy_run() { # -> md5 on stdout, full output at $1
  "$BIN" "$MODEL" --tokens "$CANON_IDS" --ctx 2048 --spec -n 128 >"$1" 2>"$1.err"
  grep '^generated:' "$1" | md5sum | cut -d' ' -f1
}
sampled_run() {
  "$BIN" "$MODEL" --tokens "$CANON_IDS" --ctx 2048 --spec -n 64 \
        --temp 0.7 --top-p 0.95 --seed 42 >"$1" 2>"$1.err"
  grep '^generated:' "$1" | md5sum | cut -d' ' -f1
}

# Everything that could select a different trajectory. Printed ONLY on a
# mismatch, but gathered unconditionally so the record is of the failing run.
forensics() { # $1=label $2=got $3=want $4=outfile
  echo ""
  echo "################ ANCHOR MISMATCH -- $1 ################"
  echo "got  : $2"
  echo "want : $3"
  echo "-- record the FULL digest above; the 08-16/08-18 sightings only ever"
  echo "   preserved an 8-char prefix, which is why they were unchaseable."
  echo ""
  echo "[binary]   $BIN"
  echo "           md5=$(md5sum "$BIN" | cut -d' ' -f1) mtime=$(date -r "$BIN" '+%F %T')"
  echo "[artifact] $MODEL"
  echo "           size=$(stat -c%s "$MODEL") mtime=$(date -r "$MODEL" '+%F %T')"
  echo "           tier=$CANON_TIER arch=$CANON_ARCH (tier derived from filename)"
  local ck; ck="$(dirname "$MODEL")/CHECKSUMS.md5"
  if [[ -f "$ck" ]]; then
    echo "           CHECKSUMS.md5 says: $(grep -F "$(basename "$MODEL")" "$ck" || echo '(not listed)')"
    echo "           (verify with: md5sum $MODEL -- skipped here, it is a 15 GB read)"
  fi
  echo "[device]   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,clocks.sm,temperature.gpu,memory.used \
             --format=csv 2>/dev/null | sed 's/^/           /'
  echo "[co-resident]"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null | sed 's/^/           /'
  echo "[env]      (any Q27_* latch moves the run onto a different numeric path)"
  env | grep -E '^(Q27_|CUDA_)' | sed 's/^/           /' || echo "           (none set)"
  echo "[stderr]"
  sed 's/^/           /' "$4.err" | head -20
  echo "[generated]"
  grep '^generated:' "$4" | head -c 600 | sed 's/^/           /'
  echo ""
  echo "########################################################"
  echo ""
}

fail=0

echo "== greedy canonical ($CANON_ARCH:$CANON_TIER) x$REPEATS"
declare -A seen=()
for i in $(seq 1 "$REPEATS"); do
  md5="$(greedy_run "$tmp/g$i.out")"
  seen["$md5"]=$(( ${seen["$md5"]:-0} + 1 ))
  if [[ "$md5" == "$CANON_MD5" ]]; then
    echo "  [$i] OK  $md5"
  else
    echo "  [$i] MISMATCH $md5"
    forensics "greedy run $i" "$md5" "$CANON_MD5" "$tmp/g$i.out"
    fail=1
  fi
done
if [[ "$fail" == 1 ]]; then
  echo "== greedy distribution over $REPEATS runs (sticky config error vs transient):"
  for k in "${!seen[@]}"; do echo "     ${seen[$k]}x $k"; done
  echo "   ALL runs divergent  -> a CONFIG difference (artifact/tier/device/env), not the lottery."
  echo "   MIXED               -> the open 08-16/08-18 non-determinism. Save this log; it is the"
  echo "                          first capture with full context. See BUILDLOG (l)."
fi

if [[ -n "${SAMPLED_MD5:-}" ]]; then
  echo "== sampled-seed anchor (flags are PART of the anchor: --ctx 2048 --spec)"
  smd5="$(sampled_run "$tmp/s.out")"
  if [[ "$smd5" == "$SAMPLED_MD5" ]]; then echo "  OK  $smd5"
  else
    echo "  MISMATCH $smd5"
    forensics "sampled-seed" "$smd5" "$SAMPLED_MD5" "$tmp/s.out"
    echo "  NOTE: 227a6b08... is the NO---spec trajectory. If you got that, the recipe"
    echo "        lost its --spec (this exact trap fired at v0.5.0 and again 2026-08-18)."
    fail=1
  fi
else
  echo "== sampled-seed anchor: SKIP (none published for $CANON_ARCH:$CANON_TIER)"
fi

echo ""
[[ "$fail" == 0 ]] && echo "ANCHORS: ALL PASS" || echo "ANCHORS: FAILED"
exit "$fail"
