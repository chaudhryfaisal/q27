#!/usr/bin/env bash
# phase_width_run.sh -- phv-by-width follow-up (Saguaro deep-ladder pricing).
# docs payload at MAXD auto/6/7 (byte-identical tokens per ceiling), phase stats on.
set -u
cd "$(dirname "$0")/.."
MODEL=${MODEL:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.q27}
TOK=${TOK:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.tok}
PORT=${PORT:-8208}
CTX=${CTX:-32768}
SRV=""
stop_server() { [ -n "$SRV" ] && kill "$SRV" 2>/dev/null && wait "$SRV" 2>/dev/null; SRV=""; }
trap stop_server EXIT

run_leg() { # $1=payload $2=maxd $3=tag
  local BODY=scratchpad/accept_payload_$1.json
  local LOG=/tmp/phase_width_$3.log
  Q27_PHASE_STATS=1 Q27_KV=fp8 Q27_PMIN=0.5 Q27_MAXD=$2 \
    build/q27-server "$MODEL" "$TOK" --port "$PORT" --ctx "$CTX" --no-think \
    --fast-head >"$LOG" 2>&1 &
  SRV=$!
  for i in $(seq 1 120); do
    curl -s -m 2 "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 2
  done
  for r in 1 2 3 4; do
    curl -s -m 600 "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
      --data-binary @"$BODY" >/dev/null
  done
  stop_server
  echo "=== leg $3 (payload=$1 maxd=$2) ==="
  grep "\[req\]" "$LOG" | sed 's/conv=[0-9a-f]*//'
}

run_leg docs auto docs_auto
run_leg docs 6 docs_m6
run_leg docs 7 docs_m7
echo ALL_LEGS_DONE
