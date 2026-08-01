#!/usr/bin/env bash
# accept_kv_ab.sh -- KV-format acceptance A/B at the CC serving point
# (PMIN=0.5, MAXD=auto7, profile suffix defaults): fp8-as-served (mma),
# fp8+fd2 (kernel-numerics control), turbo3 and turbo5k (both fd2 fallthrough).
#
# READ ms/rnd, NOT tps, when comparing KV FORMATS. A format change re-rolls the
# trajectory (different numerics -> different tokens -> different acceptance),
# so tps mixes kernel cost with a draft-acceptance lottery. ms/rnd is the
# acceptance-independent cost; tok/rnd is the acceptance; tps is their ratio.
# fp8fd2 is the control that separates "fd2 vs mma" from "format vs format" --
# turbo3/turbo5k have no mma leg by construction (see attn_decode3's guard).
#
# Derived from
# accept_ab.sh; same 1-cold + 3-warm replay protocol and [req] parser.
#
# For each payload (repro/code/testgen) x each depth leg (d4/d5/auto):
# fresh server, 1 cold prefill + 3 identical replays (P13 methodology,
# BUILDLOG:1655), greedy. Reports per leg: median warm decode t/s, tok/round,
# rounds (must be identical across replays -- greedy determinism), and the
# cumulative per-lane yields gla[j]/glf[j] from the final [req] line.
#
# Usage: bash tools/accept_ab.sh [PAYLOAD ...]   (default: all three)
# Env: MODEL, TOK, PORT, LEGS ("4 5 auto"), MAXTOK override.
# Needs the 5090 free; run tools/make_payloads.py first.

set -u
cd "$(dirname "$0")/.."
MODEL=${MODEL:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.q27}
TOK=${TOK:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.tok}
PORT=${PORT:-8199}
# BIN: the server binary. 24GB Ampere cards need build/q27-server-w8 (the
# default W12 build OOMs at graph setup there) -- see README.
BIN=${BIN:-build/q27-server}
CTX=${CTX:-32768}
LEGS=${LEGS:-"fp8 fp8fd2 turbo3 turbo5k"}
PAYLOADS=${*:-cctx cctx2 repro}
SRV=""

stop_server() { [ -n "$SRV" ] && kill "$SRV" 2>/dev/null && wait "$SRV" 2>/dev/null; SRV=""; }
trap stop_server EXIT

for pay in $PAYLOADS; do
  BODY=scratchpad/accept_payload_${pay}.json
  [ -f "$BODY" ] || { echo "missing $BODY (run tools/make_payloads.py)"; exit 1; }
  for leg in $LEGS; do
    LOG=$(mktemp /tmp/accept_ab.XXXXXX.log)
    case $leg in
      fp8)    kvenv="Q27_KV=fp8";;
      fp8fd2) kvenv="Q27_KV=fp8 Q27_FD=fd2";;
      turbo3) kvenv="Q27_KV=turbo3";;
      turbo5k) kvenv="Q27_KV=turbo5k";;
    esac
    env $kvenv Q27_PMIN=0.5 Q27_MAXD=auto7 \
      "$BIN" "$MODEL" "$TOK" --port "$PORT" --ctx "$CTX" --no-think \
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
    python3 - "$LOG" "$pay" "$leg" <<'PYEOF'
import re, statistics, sys
log, pay, leg = sys.argv[1:4]
reqs = [l for l in open(log) if "[req]" in l]
assert len(reqs) == 4, f"{pay}/{leg}: want 4 [req] lines, got {len(reqs)}"
def f(pat, l, cast=float):
    m = re.search(pat, l)
    return cast(m.group(1)) if m else None
warm = reqs[1:]
tps = [f(r" tps=([\d.]+)", l) for l in warm]
dec = [f(r" dec=(\d+)", l, int) for l in warm]
rnd = [f(r" rounds=(\d+)", l, int) for l in warm]
dms = [f(r" dec_ms=([\d.]+)", l) for l in warm]
det = "OK" if len(set(rnd)) == 1 and len(set(dec)) == 1 else f"NONDET rounds={rnd} dec={dec}"
prompt = f(r" prompt=(\d+)", reqs[0], int)
last = reqs[-1]
def vec(name, l):
    m = re.search(rf" {name}=([\d,]+)", l)
    return [int(x) for x in m.group(1).split(",")] if m else None
glf, gla, gch = vec("glf", last), vec("gla", last), vec("gch", last)
y = ["%.3f" % (a / fd) if fd else "--" for a, fd in zip(gla or [], glf or [])]
gated = sum(gch) if gch else 0
fired5 = "%.3f" % (glf[4] / gated) if glf and gated else "--"
if not rnd[0] or not dec[0]:
    sys.exit(f"{pay}/{leg}: dec={dec[0]} rounds={rnd[0]} prompt={prompt} -- "
             "zero-output leg (prompt>ctx returns 0 tokens on /v1/completions; "
             "or instant EOS). Fix the payload, this leg measured nothing.")
if dec[0] < 200:
    print(f"  WARNING: dec={dec[0]} < 200 (early EOS -- payload not open enough?)")
print(f"{pay:8s} {leg:7s} prompt={prompt} tps_med={statistics.median(tps):7.1f} "
      f"tok/rnd={dec[0]/rnd[0]:5.3f} ms/rnd={statistics.median(dms)/rnd[0]:6.2f} "
      f"rounds={rnd[0]} det={det} y1..5={','.join(y)} fired5={fired5}")
PYEOF
    rm -f "$LOG"
  done
done
