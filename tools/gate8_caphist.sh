#!/usr/bin/env bash
# Gate 8: does raising the suffix cap 12->16 actually accept MORE tokens per fire
# on LIVE CC traffic (not the echo best case)? Ship the cap iff the ratio > 1.07.
#
# Same W16 binary both legs; only Q27_SUFFIX_W differs (12 vs 16). Q27_SUFFIX_DBG
# emits per-fire accepted lengths; we take the mean over the whole run. Traffic is
# real Claude Code driving a thunderdome task through the local server.
set -u
BIN="${1:?usage: gate8_caphist.sh <w16-server-binary> <task>}"
TASK="${2:-T8}"
TD=/mnt/ai/projects/q27
MODEL=/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.q27
TOK=/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.tok
leg() { # $1=cap
  local CAP=$1
  systemctl --user stop q27-eval 2>/dev/null; sleep 2
  CUDA_VISIBLE_DEVICES=0 Q27_KV=fp8 Q27_FD=mma Q27_PMIN=0.5 Q27_MAXD=auto7 Q27_SUFFIX=1 \
    Q27_SUFFIX_W=$CAP Q27_SUFFIX_DBG=1 \
    "$BIN" "$MODEL" "$TOK" --port 8081 --host 0.0.0.0 > /tmp/g8_$CAP.log 2>&1 &
  local P=$!
  for _ in $(seq 180); do curl -s -m 1 http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 2; done
  ( cd /mnt/ai/projects/thunderdome && ./thunderdome run --orchestrator claude-code-q27-haight \
      --task "$TASK" --trials 1 >/tmp/g8_$CAP.harness 2>&1 )
  kill $P 2>/dev/null; wait $P 2>/dev/null
  # mean accepted-per-suffix-fire from the sfxdbg-oc trace (sfx_round=1 lines)
  python3 - "$CAP" <<'PY'
import sys, re
cap = sys.argv[1]
ns = []
for l in open(f"/tmp/g8_{cap}.log"):
    if "[sfxdbg-oc]" in l and "sfx_round=1" in l:
        m = re.search(r"n=(\d+)", l)
        if m: ns.append(int(m.group(1)))
if ns:
    import statistics as st
    pinned = sum(1 for x in ns if x >= int(cap))
    print(f"  cap={cap}: {len(ns)} fires, mean accepted {sum(ns)/len(ns):.2f}, "
          f"median {st.median(ns):.0f}, {100*pinned/len(ns):.0f}% pinned at {cap}")
    open(f"/tmp/g8_{cap}.mean","w").write(f"{sum(ns)/len(ns)}")
else:
    print(f"  cap={cap}: NO suffix fires captured")
PY
}
echo "=== gate 8: live-traffic accept-per-fire, cap 12 vs cap 16 (task $TASK) ==="
leg 12
leg 16
python3 - <<'PY'
try:
    t12=float(open("/tmp/g8_12.mean").read()); t16=float(open("/tmp/g8_16.mean").read())
    r=t16/t12
    print(f"\n  tok(16)/tok(12) = {t16:.2f}/{t12:.2f} = {r:.3f}")
    print(f"  GATE 8 BAR: ship iff > 1.07  ->  {'SHIP' if r>1.07 else 'DO NOT SHIP (cap raise does not pay on live traffic)'}")
except Exception as e:
    print("  (missing a leg's mean)", e)
PY
