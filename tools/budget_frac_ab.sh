#!/usr/bin/env bash
# budget_frac_ab.sh -- is 0.5 the right THINK_BUDGET_FRAC?
#
# THINK_BUDGET_FRAC splits a request's max_tokens between the <think> block and
# the answer that follows it. api_common.h says of the 0.5 default: "a judgement
# call, not a measurement", and that the club-3090 #765 agreement it cites
# compared two ABSOLUTE budgets rather than a fraction, so it is a magnitude
# sanity check rather than a reproduced result. The comment asks outright for a
# measured fraction to replace it.
#
# THE MECHANISM IS ALREADY ON RECORD. BUILDLOG 2026-08-15 attributes the 24K-arm
# damage to the budget rather than the effort line: "24K thinking leaves 8K per
# response, and the forced-close-plus-answer machinery reshapes every long-turn
# trajectory", and the tasks that fell hardest were the big-write ones. A LOWER
# fraction should therefore help exactly those tasks, by leaving more of the cap
# for the answer.
#
# TASKS chosen to test that, not to resample the suite. The 08-20 suite run at
# frac 0.5 scored plugin-marketplace 0.000 at 931K tokens and task-queue 0.182
# at 792K -- both big-write, both failing. monorepo-disaster is the null rung:
# 1507K tokens, also big-write, and it scored 1.000, so a fraction change that
# is merely disruptive should break it while one that helps should not.
#
# Arms run to completion in order, so an interrupted run still leaves whole arms
# comparable rather than a partial matrix.
set -uo pipefail

TD=/mnt/ai/projects/thunderdome
MODEL=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27
TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
Q27=/mnt/ai/projects/q27/build/q27-server
# docker bridge only: containers reach it, the LAN does not
HOST=172.17.0.1
PORT=8081
UNIT=q27-eval-frac
OUT="${1:?usage: budget_frac_ab.sh <outdir>}"
mkdir -p "$OUT"
MAP="$OUT/runmap.tsv"
: >"$MAP"

ARMS="${ARMS:-0.25 0.5 0.75}"
VOLATILE="${VOLATILE:-bench-plugin-marketplace bench-task-queue}"
STABLE="${STABLE:-bench-monorepo-disaster}"
NTRIAL="${NTRIAL:-3}"

boot() { # $1 = effort arm
  systemctl --user stop "$UNIT" 2>/dev/null
  # the unit is --collect, so the stop above fully reaps it before we rebind
  for _ in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done
  systemd-run --user --unit="$UNIT" --collect \
    --setenv=Q27_KV=fp8 --setenv=Q27_THINK_BUDGET_FRAC="$1" \
    -- "$Q27" "$MODEL" "$TOK" --port "$PORT" --host "$HOST" --ctx 131072 --think \
    >/dev/null 2>&1
  for _ in $(seq 1 180); do
    curl -s -m 3 -o /dev/null "http://$HOST:$PORT/health" && return 0
    sleep 5
  done
  echo "[$1] BOOT TIMEOUT" >&2; return 1
}

for ARM in $ARMS; do
  echo ""; echo "################ ARM frac=$ARM ################"
  boot "$ARM" || continue
  # prove the arm actually reached the process rather than trusting the launch
  PID=$(pgrep -f "q27-server.*--port $PORT" | head -1)
  echo "[$ARM] server pid=$PID env=$(tr '\0' '\n' < /proc/$PID/environ | grep -c '^Q27_THINK_BUDGET_FRAC='"$ARM"'$') (1 = confirmed)"

  for TASK in $VOLATILE $STABLE; do
    N=$NTRIAL; [ "$TASK" = "$STABLE" ] && N=1
    echo "[$ARM] $TASK x$N ..."
    BEFORE=$(ls -1 "$TD/results/runs" 2>/dev/null | tail -1)
    (cd "$TD" && ./thunderdome run --orchestrator claude-code-q27-haight \
        --task "$TASK" --trials "$N" >"$OUT/log.$ARM.$TASK.txt" 2>&1)
    AFTER=$(ls -1 "$TD/results/runs" | tail -1)
    [ "$AFTER" != "$BEFORE" ] && printf "%s\t%s\t%s\n" "$ARM" "$TASK" "$AFTER" >>"$MAP"
    tail -3 "$OUT/log.$ARM.$TASK.txt" | sed "s/^/[$ARM $TASK] /"
  done
done

systemctl --user stop "$UNIT" 2>/dev/null
echo ""; echo "################ DONE ################"
cat "$MAP"
