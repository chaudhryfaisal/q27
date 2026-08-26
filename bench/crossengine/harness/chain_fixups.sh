#!/usr/bin/env bash
# Two arms had to be redone for harness reasons, not engine reasons:
#
#   llama ladder  -- OOM'd on the default 32 GDN recurrent-state checkpoints
#                    per slot at --parallel 8; now --ctx-checkpoints 4.
#   vllm agentic  -- Claude Code asks for a second model
#                    (claude-haiku-4-5-20251001) for background work; only the
#                    main id was aliased, so 10 of 12 instances 404'd at turn 1.
#                    The 2 that ran before hitting it both scored gold.
#
# vllm quality (66/75) and the four earlier legs stand and are skipped.
set -u
OUT=/mnt/ai/projects/club-3090/results/ninfer-vs-q27-20260817

while systemctl --user is-active ab-newlegs >/dev/null 2>&1; do sleep 15; done
sleep 10

mkdir -p "$OUT/void-vllm-agentic-modelalias" "$OUT/void-llama-ladder-oom"
mv "$OUT"/agentic.vllm.jsonl "$OUT"/agentic.vllm.log "$OUT"/tap.vllm.jsonl \
   "$OUT/void-vllm-agentic-modelalias/" 2>/dev/null
mv "$OUT"/ladder.llama.txt "$OUT"/tapladder.llama.jsonl \
   "$OUT/void-llama-ladder-oom/" 2>/dev/null
# keep the vllm prefix-cache counter trace with the run it belongs to
cp "$OUT"/vllm_metrics.log "$OUT/void-vllm-agentic-modelalias/" 2>/dev/null
: >"$OUT/vllm_metrics.log"

bash "$OUT/harness/run_all.sh" "$OUT" ladder llama
bash "$OUT/harness/run_all.sh" "$OUT" agentic vllm
