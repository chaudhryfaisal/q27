#!/usr/bin/env python3
"""M3a gate: shared-prefill-arena race check.

The arena hands one set of chunk buffers to whichever engine is prefilling.
Engines run on DIFFERENT streams and the GpuGate hands over between chunks, so
a missing drain on the ownership change shows up as one prefill reading another
engine's chunk staging -- silently wrong hidden state, i.e. wrong TEXT, not a
crash. This fires a long (multi-chunk) prompt solo to get a reference, then the
same prompt on C concurrent slots, and demands byte-identical output.

Run the server with Q27_BATCH_GEMM=0: at k >= 3 the union weight sweep defaults
to vgemm (tolerance class), which would fail an identity check for reasons that
have nothing to do with the arena.

  python3 bench/ladder/prefill_race.py <url> <C> [prompt_tokens]
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8199"
C = int(sys.argv[2]) if len(sys.argv) > 2 else 4
PROMPT_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 6000

# ~3 chars/token of prose; PF_T=1024, so this spans ~6 chunks and yields
# between every one of them once slots contend.
with open("docs/BUILDLOG.md", encoding="utf-8", errors="replace") as f:
    corpus = f.read()
prompt = corpus[: PROMPT_TOKENS * 3]


def ask(_i):
    body = json.dumps({
        "model": "q27",
        "messages": [{"role": "user", "content":
                      prompt + "\n\nReply with exactly one short sentence."}],
        "max_tokens": 48,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["choices"][0]["message"]["content"]


ref = ask(0)
print(f"reference ({len(ref)} chars): {ref[:90]!r}")
with ThreadPoolExecutor(max_workers=C) as ex:
    outs = list(ex.map(ask, range(C)))
bad = [i for i, o in enumerate(outs) if o != ref]
for i in bad:
    print(f"  MISMATCH slot-run {i}: {outs[i][:90]!r}")
print(f"PREFILL RACE {'FAIL' if bad else 'PASS'}: {C - len(bad)}/{C} identical to solo")
sys.exit(1 if bad else 0)
