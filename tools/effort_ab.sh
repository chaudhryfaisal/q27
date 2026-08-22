#!/usr/bin/env bash
# effort_ab.sh -- does Qwen3.8's reasoning-effort render help or hurt?
#
# WHY THIS EXISTS. 3.8's chat template injects a reasoning-effort instruction at
# the head of the system block and defaults to xhigh; q27 shipped that render at
# 37caa07 on FIDELITY grounds (the checkpoint was trained expecting the line),
# never on a measured suite score. The two suite numbers on record confound the
# knob with the thinking budget:
#
#   effort OFF, 16K budget        hidden 0.511   <- the "3.8 regression" number
#   effort xhigh + 24K budget     hidden 0.315
#   effort xhigh + 16K (SHIPPED)  never measured
#
# So the shipped default is unmeasured, and xhigh-vs-medium-vs-low at a fixed
# budget has never been run at all. This runs it.
#
# DESIGN. Three arms differing in ONE server-side env var, everything else
# pinned. Task set follows the protocol verdict banked on 2026-08-15: "campaign
# comparisons need n=3 on the volatile tasks (the five fast tasks are stable and
# can stay n=1)", so three volatile tasks at n=3 plus one stable task at n=1 as
# a NULL RUNG that must read flat across arms. 21 tasks x 3 arms x n=3 would be
# ~190 runs; this is 30 and still resolves direction against the +-0.1-0.15
# basin noise the same entry records.
#
# SCORING. hidden = scores.hidden_tests where present, else scores.tests. NOT
# composite_score, and NOT scores.tests on greenfield tasks -- reading the
# visible column is exactly what produced the retracted 0.336.
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
UNIT=q27-eval-effort
OUT="${1:?usage: effort_ab.sh <outdir>}"
mkdir -p "$OUT"
# ABSOLUTE, because the thunderdome invocation below redirects from inside a
# subshell that has already cd'd to $TD -- a relative $OUT resolves against
# thunderdome's tree there and every log redirect fails with "No such file or
# directory" while the arms still appear to run.
OUT="$(cd "$OUT" && pwd)"
MAP="$OUT/runmap.tsv"
: >"$MAP"

ARMS="${ARMS:-xhigh medium low}"
VOLATILE="${VOLATILE:-bench-task-queue bench-time-tracker bench-constraint-scheduler}"
STABLE="${STABLE:-bench-financial-ledger}"
NTRIAL="${NTRIAL:-3}"

boot() { # $1 = effort arm
  systemctl --user stop "$UNIT" 2>/dev/null
  # the unit is --collect, so the stop above fully reaps it before we rebind
  for _ in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done
  # Q27_PRINT_WSUM: the 5090 silently corrupts a small fraction of model loads,
  # and this knob was INERT on q27-server until 2026-08-21 (it lived behind
  # own_weights in engine.cuh; the server takes the shared-weights path). Every
  # campaign before that ran unguarded. Print it per arm and compare across arms.
  systemd-run --user --unit="$UNIT" --collect \
    --setenv=Q27_KV=fp8 --setenv=Q27_PRINT_WSUM=1 --setenv=Q27_REASONING_EFFORT="$1" \
    -- "$Q27" "$MODEL" "$TOK" --port "$PORT" --host "$HOST" --ctx 131072 --think \
    >/dev/null 2>&1
  for _ in $(seq 1 180); do
    curl -s -m 3 -o /dev/null "http://$HOST:$PORT/health" && return 0
    sleep 5
  done
  echo "[$1] BOOT TIMEOUT" >&2; return 1
}

for ARM in $ARMS; do
  echo ""; echo "################ ARM effort=$ARM ################"
  boot "$ARM" || continue
  # prove the arm actually reached the process rather than trusting the launch.
  # $ARM, not $1: at this scope $1 is the script's outdir, and a path full of
  # slashes inside an s/// replacement is how the first version of this line
  # died with "unknown option to `s'".
  journalctl --user -u "$UNIT" --no-pager --since "-10min" 2>/dev/null \
    | grep -io "wsum:.*" | tail -1 | sed "s|^|[$ARM] |"
  PID=$(pgrep -f "q27-server.*--port $PORT" | head -1)
  echo "[$ARM] server pid=$PID env=$(tr '\0' '\n' < /proc/$PID/environ | grep -c '^Q27_REASONING_EFFORT='"$ARM"'$') (1 = confirmed)"

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
