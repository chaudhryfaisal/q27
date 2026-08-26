#!/usr/bin/env bash
# vLLM exposes prefix-cache counters ONLY on the live container's /metrics, and
# leg_stop removes the container. Snapshot them with timestamps so the agentic
# and quality windows can be separated after the fact.
OUT="${1:?}"; PORT="${2:-8160}"
while true; do
  ts=$(date +%s)
  m=$(curl -s -m 5 "http://127.0.0.1:$PORT/metrics" 2>/dev/null \
      | grep -E "^vllm:(prefix_cache_(queries|hits)_total|num_requests)" ) || true
  [ -n "$m" ] && echo "$ts|$(echo "$m" | tr '\n' ';')" >>"$OUT/vllm_metrics.log"
  sleep 20
done
