#!/usr/bin/env python3
"""M3a gate: shared-prefill-arena race check.

WHY max_tokens=1. The gate compares the FIRST emitted token only, because that
token is produced by the prefill epilogue (step_with on the request's own
stream) before any batched decode round exists. Comparing full responses does
NOT work and is not a bug in the engine: under concurrency the trim floors
granted width, attention switches between the fd2 and fdmma kernels at
ntok >= 4, and those are tolerance-class twins rather than bitwise ones -- so a
near-tie argmax can flip and fork the whole trajectory. Measured: a control
boot with the arena DISABLED (Q27_PF_ARENA=0) also "fails" a full-text
comparison, which is exactly how that confound was caught. First-token identity
isolates the prefill, which is the only thing the arena touches.

The arena hands one set of chunk buffers to whichever engine is prefilling.
Engines run on DIFFERENT streams and the GpuGate hands over BETWEEN CHUNKS, so
ownership ping-pongs mid-prefill; a missing drain on an ownership change shows
up as one prefill reading another engine's chunk staging -- silently wrong
hidden state, i.e. wrong TEXT, not a crash.

Each concurrent run gets a UNIQUE prompt (a per-run nonce in the first chunk),
for two reasons: identical prompts would let slot routing serve some runs from
the stable-prefix snapshot instead of cold-prefilling them (fewer real racers),
and a shared prefix would make the racy reads coincidentally correct. The gate
also ASSERTS that prefills actually interleaved, by requiring the server log to
show mid-prefill yields -- otherwise a run that happened to serialize passes
vacuously.

Sensitivity is verifiable, not assumed: Q27_PF_ARENA_NODRAIN=1 on the server
disables claim()'s drain, and this gate is expected to FAIL there.

Run the server with Q27_BATCH_GEMM=0: at k >= 3 the union weight sweep defaults
to vgemm (tolerance class), which would fail an identity check for reasons that
have nothing to do with the arena.

  python3 bench/ladder/prefill_race.py <server_log> <url> <C> [prompt_tokens]
"""
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

LOG = sys.argv[1]
URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8199"
C = int(sys.argv[3]) if len(sys.argv) > 3 else 4
PROMPT_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 6000

# ~3 chars/token of prose; PF_T=1024, so this spans several chunks and the gate
# hands over between every one of them once slots contend. Path is resolved
# from this file, not the cwd.
CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "docs", "BUILDLOG.md")
with open(CORPUS, encoding="utf-8", errors="replace") as f:
    body = f.read()[: PROMPT_TOKENS * 3]


def prompt_for(i):
    # The nonce leads, so run i's FIRST chunk already differs -- no snapshot or
    # prefix-cache entry from another run can cover it.
    return (f"Document revision {i}, unique build tag {i * 7919:08d}.\n" + body +
            "\n\nReply with exactly one short sentence about the text above.")


def ask(i):
    req = urllib.request.Request(
        URL + "/v1/chat/completions",
        data=json.dumps({
            "model": "q27",
            "messages": [{"role": "user", "content": prompt_for(i)}],
            "max_tokens": 1,
            "temperature": 0,
        }).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["choices"][0]["message"]["content"]  # 1 token


def yields_after(offset):
    """Mid-prefill yields recorded by the server since `offset`."""
    with open(LOG, "rb") as f:
        f.seek(offset)
        tail = f.read().decode("utf-8", "replace")
    return [int(m) for m in re.findall(r"yields=(\d+)", tail)]


# Reference: each run's prompt, alone, with nothing else on the GPU.
refs = [ask(i) for i in range(C)]
mark = os.path.getsize(LOG)
with ThreadPoolExecutor(max_workers=C) as ex:
    outs = list(ex.map(ask, range(C)))

bad = [i for i in range(C) if outs[i] != refs[i]]
for i in bad:
    print(f"  MISMATCH run {i}:\n    solo:  {refs[i][:100]!r}\n    conc:  {outs[i][:100]!r}")
ys = yields_after(mark)
interleaved = sum(1 for y in ys if y > 0)
print(f"  interleaving: {interleaved}/{len(ys)} concurrent requests yielded mid-prefill "
      f"(yields={ys})")
ok = not bad and interleaved >= 2
if not bad and interleaved < 2:
    print("  VACUOUS: prefills did not interleave -- the arena handoff was never exercised")
print(f"PREFILL RACE {'PASS' if ok else 'FAIL'}: {C - len(bad)}/{C} identical to solo")
sys.exit(0 if ok else 1)
