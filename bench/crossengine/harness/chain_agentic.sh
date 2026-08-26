#!/usr/bin/env bash
# Wait for the quality run to release the GPU, then re-run the agentic arm on
# all four legs under the full normalization (output_config + thinking stripped,
# chat_template_kwargs.enable_thinking translated).
#
# All four, not just the two ninfer legs: the q27 agentic data was collected
# before the thinking strip and the ctk translation existed, so it came from a
# different harness version. q27 provably ignores both fields, so its numbers
# should reproduce -- and if they do not, that is worth knowing before anything
# gets written down.
set -u
OUT=/mnt/ai/projects/club-3090/results/ninfer-vs-q27-20260817

# one engine owns the card at a time
while systemctl --user is-active ninfer-q27-quality >/dev/null 2>&1; do sleep 10; done
sleep 10

mkdir -p "$OUT/agentic-pre-thinkstrip"
mv "$OUT"/agentic.*.jsonl "$OUT"/agentic.*.log "$OUT"/tap.*.jsonl \
   "$OUT"/server.*.agentic.log "$OUT"/server.*.agentic.reqlog.jsonl \
   "$OUT/agentic-pre-thinkstrip/" 2>/dev/null

exec bash "$OUT/harness/run_all.sh" "$OUT" agentic q4s,nint,q5f,nvfp4
