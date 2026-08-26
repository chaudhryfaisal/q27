#!/usr/bin/env bash
# Canonical-digest gate for the repack path (PR #23 follow-up).
#
# tools/test_repack_split{,_e2e}.py prove the split-GGUF plumbing on synthetic
# fixtures -- 15 tensors of a few KB each. They cannot prove that a REAL repack
# produces the artifact we publish, which is the discipline every other tier in
# this repo is held to (tools/shortbench_suite.sh, tools/metal_canonical_gate.sh).
# This closes that gap: repack a real source and compare the output MD5 against
# the published digest.
#
# Two modes, picked by what you point it at:
#
#   single  SRC_GGUF is one merged BF16 GGUF (the Qwen3.6 shape, and the
#           Qwen3.8 shape once someone has merged it locally).
#   split   SRC_GGUF is ggml-org's 64-block base and SRC_MTP_GGUF its 65-block
#           companion -- the path PR #23 added. A byte-identical result across
#           both modes is the strongest available proof the merge is correct,
#           because the merged file is what the published digest came from.
#
# Opt-in by design: the sources are 20-55 GB and a repack runs tens of minutes,
# so this is a release/CI gate, not a per-commit one. It SKIPS (exit 0) when the
# inputs are absent so it can sit in a pipeline unconditionally.
#
#   SRC_GGUF=/path/base.gguf SRC_MTP_GGUF=/path/mtp.gguf \
#   CANON_MD5=<published md5> [REPACK_ARGS="--q4-head"] \
#     tools/repack_canonical_gate.sh
set -uo pipefail

SRC_GGUF="${SRC_GGUF:-}"
SRC_MTP_GGUF="${SRC_MTP_GGUF:-}"
CANON_MD5="${CANON_MD5:-}"
REPACK_ARGS="${REPACK_ARGS:-}"
PYTHON="${PYTHON:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# REPACK_CMD exists so this gate is itself testable: the accompanying test
# points it at a spec-shrunk launcher over a tiny fixture, which exercises the
# real PASS / MISMATCH branches without a 20-55 GB source. Production leaves it
# unset and gets tools/repack.py.
REPACK_CMD="${REPACK_CMD:-$PYTHON $HERE/repack.py}"

if [[ -z "$SRC_GGUF" || -z "$CANON_MD5" ]]; then
  echo "repack canonical gate: SKIP (set SRC_GGUF and CANON_MD5 to run)"
  exit 0
fi
if [[ ! -r "$SRC_GGUF" ]]; then
  echo "repack canonical gate: SKIP (SRC_GGUF not readable: $SRC_GGUF)"
  exit 0
fi
if [[ -n "$SRC_MTP_GGUF" && ! -r "$SRC_MTP_GGUF" ]]; then
  echo "repack canonical gate: SKIP (SRC_MTP_GGUF not readable: $SRC_MTP_GGUF)"
  exit 0
fi

hash_file() {
  if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | cut -d' ' -f1
  elif command -v md5 >/dev/null 2>&1; then md5 -q "$1"
  else echo "md5 or md5sum is required" >&2; return 2; fi
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
out="$tmp/candidate.q27"

mode="single"
mtp_flag=()
if [[ -n "$SRC_MTP_GGUF" ]]; then
  mode="split"
  mtp_flag=(--mtp "$SRC_MTP_GGUF")
fi

echo "== repack canonical gate ($mode)"
echo "   src:   $SRC_GGUF"
[[ -n "$SRC_MTP_GGUF" ]] && echo "   mtp:   $SRC_MTP_GGUF"
echo "   want:  $CANON_MD5"

# shellcheck disable=SC2086
if ! $REPACK_CMD "$SRC_GGUF" "$out" "${mtp_flag[@]}" $REPACK_ARGS; then
  echo "repack canonical gate: FAIL (repack errored)" >&2
  exit 1
fi

got="$(hash_file "$out")" || exit 2
echo "   got:   $got"

if [[ "$got" == "$CANON_MD5" ]]; then
  echo "repack canonical gate: PASS ($mode)"
  exit 0
fi
echo "repack canonical gate: MD5 MISMATCH ($mode) -- got $got want $CANON_MD5" >&2
echo "  A mismatch here means the repacked artifact is NOT the published one." >&2
echo "  In split mode this is the check that would catch a bad MTP merge that" >&2
echo "  still satisfies every shape/manifest gate." >&2
exit 1
