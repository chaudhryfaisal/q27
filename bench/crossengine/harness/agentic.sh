#!/usr/bin/env bash
# Real-world agentic arm: drive Claude Code against a leg over the Anthropic API.
#
# Derived from q27's bench/swebench/run.sh (same 12 pinned SWE-bench_Verified
# instances, same task prompt, same gold-file overlap signal) with one change
# that makes it engine-agnostic: the original scrapes decode telemetry out of
# q27's journal `[req]` lines, which ninfer does not emit. Here Claude Code
# points at the tap, and the tap is the only source of timing -- one accounting
# convention for both engines.
#
#   usage: agentic.sh <leg> <tap_port> <outdir> [instance_filter]
set -u

SP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAN=/mnt/ai/projects/q27/bench/swebench/manifest.json
CACHE=${SWEBENCH_CACHE:-/mnt/ai/swebench-cache}
WORK=${SWEBENCH_WORK:-/mnt/ai/swebench-work}
IMG=${SWEBENCH_IMG:-thunderdome/claude-code:latest}

LEG="${1:?usage: agentic.sh <leg> <tap_port> <outdir> [filter]}"
TAP_PORT="${2:?}"
OUT="${3:?}"
FILTER="${4:-}"
RES="$OUT/agentic.$LEG.jsonl"
mkdir -p "$OUT"; : >"$RES"

case "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$TAP_PORT/health)" in
  200|401) ;;
  *) echo "leg $LEG not reachable through tap :$TAP_PORT" >&2; exit 1 ;;
esac

echo "=== agentic | leg=$LEG | tap=:$TAP_PORT ==="
mapfile -t IDS < <(python3 -c "import json;print('\n'.join(m['instance_id'] for m in json.load(open('$MAN'))))")

for IID in "${IDS[@]}"; do
  [ -n "$FILTER" ] && [ "$IID" != "$FILTER" ] && continue
  read -r REPO BASE REPOKEY < <(python3 -c "
import json
m=[x for x in json.load(open('$MAN')) if x['instance_id']=='$IID'][0]
print(m['repo'], m['base_commit'], m['repo'].replace('/','__'))")

  WS=$WORK/$LEG/$IID
  rm -rf "$WS"; mkdir -p "$WS/logs"
  git clone --quiet "$CACHE/$REPOKEY.git" "$WS/repo" 2>/dev/null
  git -C "$WS/repo" checkout -q "$BASE" 2>/dev/null || { echo "  $IID: checkout FAIL"; continue; }

  python3 -c "
import json
m=[x for x in json.load(open('$MAN')) if x['instance_id']=='$IID'][0]
open('$WS/task.txt','w').write(
'The repository \'%s\' is checked out at /workspace. Below is a GitHub issue.\n'
'Investigate and fix it by editing the source files under /workspace. Do NOT add\n'
'new test files and do NOT run the test suite -- just make the code changes needed.\n\n'
'ISSUE:\n%s\n' % (m['repo'], m['problem_statement']))"

  echo "[run] $LEG/$IID ($REPO) ..."
  # mark the tap log so this instance's requests can be attributed
  MARK=$(python3 -c "import time;print(time.time())")
  T0=$(date +%s)
  docker run --rm --user 1000:1000 --add-host host.docker.internal:host-gateway \
    -v "$WS/repo:/workspace" -w /workspace \
    -v "$WS/task.txt:/task.txt:ro" \
    -v "$WS/logs:/logs" -e HOME=/home/node \
    -e ANTHROPIC_BASE_URL=http://host.docker.internal:$TAP_PORT \
    -e ANTHROPIC_API_KEY=local -e ANTHROPIC_AUTH_TOKEN=local \
    --entrypoint bash "$IMG" \
    -c 'P=$(cat /task.txt); timeout 700 claude -p --output-format stream-json --verbose --dangerously-skip-permissions -- "$P" >/logs/out.jsonl 2>/logs/err.log; echo $? >/logs/exit' \
    >/dev/null 2>&1
  T1=$(date +%s); WALL=$((T1-T0))
  git -C "$WS/repo" diff >"$WS/patch.diff" 2>/dev/null
  echo "[run] $LEG/$IID done wall=${WALL}s"

  python3 - "$IID" "$REPO" "$WALL" "$WS" "$MAN" "$LEG" "$MARK" "$OUT/tap.$LEG.jsonl" >>"$RES" <<'PY'
import json,sys,re
iid,repo,wall,ws,man,leg,mark,taplog=sys.argv[1:9]
mark=float(mark)
m=[x for x in json.load(open(man)) if x['instance_id']==iid][0]
diff=open(f"{ws}/patch.diff").read()
files=sorted(set(re.findall(r'^\+\+\+ b/(.+)$', diff, re.M)))
gold=set(m['gold_files'])
diff_lines=sum(1 for l in diff.splitlines() if l[:1] in '+-' and not l.startswith(('+++','---')))
turns=out_tok=0; exit_reason='?'
try:
    for ln in open(f"{ws}/logs/out.jsonl"):
        try: d=json.loads(ln)
        except: continue
        if d.get('type')=='result':
            turns=d.get('num_turns',0); out_tok=(d.get('usage') or {}).get('output_tokens',0)
            exit_reason=d.get('subtype','?')
except Exception: pass
try: exitc=open(f"{ws}/logs/exit").read().strip()
except Exception: exitc='?'

# tap records belonging to this instance -- the engine-side truth
reqs=[]
try:
    for ln in open(taplog):
        try: d=json.loads(ln)
        except: continue
        if d.get('t_start',0)>=mark and d.get('status')==200 and d.get('out_tok'):
            reqs.append(d)
except Exception: pass
dec_tok=sum(r['out_tok'] for r in reqs)
dec_s=sum((r['wall_ms']-r['ttft_ms'])/1000.0 for r in reqs)
pre_tok=sum(r['in_tok'] for r in reqs)
pre_s=sum(r['ttft_ms']/1000.0 for r in reqs)
print(json.dumps({"leg":leg,"iid":iid,"repo":repo,"wall_s":int(wall),"turns":turns,
    "cc_out_tok":out_tok,"files_changed":files,"gold_files":sorted(gold),
    "gold_hit":bool(set(files)&gold),"diff_lines":diff_lines,"nonempty":bool(files),
    "exit":exitc,"result":exit_reason,
    "n_reqs":len(reqs),"tap_dec_tok":dec_tok,"tap_dec_s":round(dec_s,2),
    "tap_decode_tps":round(dec_tok/dec_s,1) if dec_s>0 else 0,
    "tap_prefill_tok":pre_tok,"tap_prefill_s":round(pre_s,2),
    "tap_prefill_tps":round(pre_tok/pre_s,1) if pre_s>0 else 0}))
PY
done

echo ""; echo "=== SUMMARY ($LEG) ==="
python3 - "$RES" <<'PY'
import json,sys,statistics as st
rows=[json.loads(l) for l in open(sys.argv[1])]
if not rows: print("  no instances"); raise SystemExit
n=len(rows); ne=sum(r['nonempty'] for r in rows); gh=sum(r['gold_hit'] for r in rows)
wall=sum(r['wall_s'] for r in rows)
dtok=sum(r['tap_dec_tok'] for r in rows); ds=sum(r['tap_dec_s'] for r in rows)
ptok=sum(r['tap_prefill_tok'] for r in rows); ps=sum(r['tap_prefill_s'] for r in rows)
tps=[r['tap_decode_tps'] for r in rows if r['tap_decode_tps']>0]
print(f"  instances: {n}  nonempty-diff: {ne}/{n}  edited-gold-file: {gh}/{n}")
print(f"  total wall: {wall}s ({wall/n:.0f}s/inst avg)")
print(f"  decode: {dtok/ds if ds else 0:.1f} t/s agg / {st.median(tps) if tps else 0:.1f} med  "
      f"({sum(r['n_reqs'] for r in rows)} reqs, {dtok} tok)")
print(f"  prefill: {ptok/ps if ps else 0:.1f} t/s agg  ({ptok} prompt tok over {ps:.0f}s TTFT)")
PY
echo "=== END ($LEG) ==="
