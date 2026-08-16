#!/usr/bin/env python3
# Concurrent-decode ladder client + accountant (the 2026-08-14 protocol,
# reconstructed for the M1 rerun -- the original rig was never committed).
#
# Protocol (BUILDLOG 2026-08-14 "CONCURRENT-DECODE LADDER vs ninfer"):
#   - server: vanilla q4s tier, Q27_KV=fp8 Q27_BATCH=1 Q27_PMIN=0.5,
#     --slots 8 --ctx 16384
#   - client: C concurrent chat requests, temp 0.6 / top-p 0.95 (no top-k /
#     presence -- q27 lacks both), max_tokens 8192, distinct salted prompts
#     per stream (no cross-request prefix reuse), n=1 per point
#   - aggregate = sum(dec) / union-of-decode-intervals, reconstructed from
#     the server's [req] telemetry: each request's decode interval is
#     [t - dec_ms, t] on the server clock (t is stamped at generate-return;
#     the cb_ms tail this folds in is the same bias the 08-14 numbers carry,
#     so the comparison stays apples-to-apples).
#
# Usage: ladder.py <server_log> <url> <C> [max_tokens]
#   Fires C requests, waits for all, then parses the [req] lines that
#   appeared in <server_log> AFTER the file offset captured at start.
import json
import sys
import re
import time
import threading
import urllib.request

LOG, URL, C = sys.argv[1], sys.argv[2], int(sys.argv[3])
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 8192

TOPICS = [
    "a deep-sea salvage crew discovering a pre-collapse research station",
    "a lighthouse keeper on a tidally locked planet",
    "a guild of clockmakers who repair timelines",
    "a caravan crossing a desert of powdered glass",
    "a botanist cataloguing the flora of a generation ship gone feral",
    "a cartographer mapping a city that rearranges itself nightly",
    "a foundry town powered by a captive storm",
    "an archivist translating the last library of a drowned continent",
]


def fire(i, results):
    salt = f"{time.time_ns() & 0xFFFFFF:06x}-{i}"
    prompt = (
        f"[stream {salt}] Write an extremely long serialized adventure novel "
        f"about {TOPICS[i % len(TOPICS)]}. Continue chapter after chapter "
        "without stopping, without a closing summary, and without asking "
        "whether to continue."
    )
    body = json.dumps({
        "model": "q27",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAXTOK,
        "temperature": 0.6,
        "top_p": 0.95,
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            usage = json.load(r).get("usage", {})
        results[i] = (usage.get("completion_tokens", 0), time.time() - t0, None)
    except Exception as e:  # noqa: BLE001 -- a failed stream must not kill the point
        results[i] = (0, time.time() - t0, str(e))


def main():
    with open(LOG, "rb") as f:
        f.seek(0, 2)
        start_off = f.tell()

    results = [None] * C
    threads = [threading.Thread(target=fire, args=(i, results)) for i in range(C)]
    t_wall0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t_wall0

    # server-side accounting from the [req] lines this point produced
    with open(LOG, "r", errors="replace") as f:
        f.seek(start_off)
        tail = f.read()
    reqs = []
    for m in re.finditer(
            r"\[req\] rid=(\d+).*?dec=(\d+) dec_ms=(\d+).*?tps=([0-9.]+).*? t=(\d+)", tail):
        rid, dec, dec_ms, tps, t_end = m.groups()
        dec, dec_ms, t_end = int(dec), float(dec_ms), float(t_end)
        reqs.append((int(rid), dec, dec_ms, float(tps), t_end - dec_ms, t_end))

    if len(reqs) != C:
        print(f"WARN: expected {C} [req] lines, parsed {len(reqs)}", file=sys.stderr)
    total_dec = sum(r[1] for r in reqs)
    ivs = sorted((r[4], r[5]) for r in reqs)
    union_ms = 0.0
    cur_a, cur_b = None, None
    for a, b in ivs:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                union_ms += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        union_ms += cur_b - cur_a

    agg = total_dec * 1000.0 / union_ms if union_ms > 0 else 0.0
    print(f"C={C} aggregate={agg:.1f} t/s  (sum_dec={total_dec}, "
          f"union={union_ms / 1000.0:.1f}s, client_wall={wall:.1f}s)")
    for rid, dec, dec_ms, tps, _a, _b in sorted(reqs):
        print(f"  rid={rid} dec={dec} dec_ms={dec_ms:.0f} per-req tps={tps:.1f}")
    for i, r in enumerate(results):
        if r and r[2]:
            print(f"  stream {i} FAILED: {r[2]}", file=sys.stderr)


if __name__ == "__main__":
    main()
