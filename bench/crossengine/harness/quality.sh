#!/usr/bin/env bash
# Correctness arm: deterministic verifier-scored packs, identical on every leg.
#
# --medium is 5 packs / 75 scenarios, no Docker, temp 0 (canonical). Greedy
# matters here: it is what makes the score a property of the weights+kernels
# rather than of the sampler, so a delta between legs is attributable.
#
# --no-thinking is forced rather than left to each pack's default so the
# pack-set is fixed across legs; q27 defaults to no-think and ninfer is booted
# with --no-thinking, so the request-level flag just closes the third door.
#
#   usage: quality.sh <leg> <port> <outdir>
set -u

LEG="${1:?usage: quality.sh <leg> <port> <outdir>}"
PORT="${2:?}"
OUT="${3:?}"
mkdir -p "$OUT"

BL=/mnt/ai/projects/benchlocal-cli
echo "=== quality | leg=$LEG | endpoint=:$PORT | --medium --no-thinking (temp 0) ==="

cd "$BL" || exit 1
# --save-json writes the machine-readable result directly; stdout stays the
# human report so a stray print can't corrupt the JSON.
python3 -c "
from benchlocal_cli.cli import main
import sys
sys.argv=['benchlocal-cli','run','--medium','--no-thinking',
          '--endpoint','http://127.0.0.1:$PORT',
          '--model','$LEG',
          '--save-json','$OUT/quality.$LEG.json',
          '--history-file','$OUT/history.csv']
main()
" >"$OUT/quality.$LEG.log" 2>&1
rc=$?

if [ ! -s "$OUT/quality.$LEG.json" ]; then
  echo "FAILED (rc=$rc) -- tail of log:" >&2
  tail -25 "$OUT/quality.$LEG.log" >&2
  exit 1
fi

python3 - "$OUT/quality.$LEG.json" "$LEG" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(f"  leg={sys.argv[2]}  mode={d.get('mode')}  thinking={d.get('thinking_enabled')}")
for p in d.get('packs',[]):
    if p.get('skipped'):
        print(f"    {p['pack_id']:<22} SKIPPED ({p.get('status')})"); continue
    print(f"    {p['pack_id']:<22} {p['passed']:>3}/{p['total']:<3} score={p['score']:.3f}")
t=d.get('totals') or {}
print(f"    {'TOTAL':<22} {t.get('passed')}/{t.get('total')} score={t.get('score')}")
PY

echo "=== quality done ($LEG) ==="
