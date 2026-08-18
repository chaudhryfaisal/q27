#!/usr/bin/env bash
set -u
OUT=/mnt/ai/projects/club-3090/results/ninfer-vs-q27-20260817
bash "$OUT/harness/run_all.sh" "$OUT" ladder llama
bash "$OUT/harness/run_all.sh" "$OUT" agentic,ladder vllm
