#!/usr/bin/env python3
"""Cross-engine concurrent-decode ladder.

Same protocol as q27's bench/ladder/ladder.py (2026-08-14): C concurrent
streams, distinct salted prompts so nothing shares a prefix, temp 0.6 /
top-p 0.95, n=1 per point, aggregate = sum(decoded tokens) / union of the
decode intervals.

The difference is where the intervals come from. q27's rig reconstructs them
from its own ``[req]`` telemetry; ninfer emits no such line, so a cross-engine
ladder cannot use it. This reads the tap log instead: each request's decode
interval is [t_start + ttft, t_start + wall], client-observed, identical
accounting on both engines.

That convention is not bit-identical to the server-side one -- it includes the
HTTP/SSE tail the server stamp excludes. Run with --calibrate against a q27 leg
to measure the offset before trusting cross-engine ratios; the q27 server log
is parsed alongside when supplied.

  usage: ladder_tap.py --url URL --tap TAPLOG --c N [--max-tokens 8192]
                       [--server-log Q27LOG]
"""
import argparse
import json
import os
import statistics as st
import sys
import threading
import time
import urllib.request

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


def fire(i, url, maxtok, results, model):
    salt = f"{time.time_ns() & 0xFFFFFF:06x}-{i}"
    prompt = (
        f"[stream {salt}] Write an extremely long serialized adventure novel "
        f"about {TOPICS[i % len(TOPICS)]}. Continue chapter after chapter "
        "without stopping, without a closing summary, and without asking "
        "whether to continue."
    )
    body = json.dumps({
        # ninfer validates this field and 400s on a mismatch; q27 ignores it.
        # Legs boot with --model-id <leg>, so the leg label works on both.
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": maxtok,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        # Drain the stream; the tap does the accounting. Reading it here (rather
        # than a non-streaming call) keeps the server in the same streaming code
        # path real clients use.
        with urllib.request.urlopen(req, timeout=3600) as r:
            for _ in r:
                pass
        results[i] = (time.time() - t0, None)
    except Exception as e:  # noqa: BLE001 -- one dead stream must not kill the point
        results[i] = (time.time() - t0, str(e))


def union_ms(intervals):
    total, cur_a, cur_b = 0.0, None, None
    for a, b in sorted(intervals):
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total


def parse_server_log(path, since):
    """q27 [req] telemetry, for calibrating the client convention."""
    import re
    if not path or not os.path.exists(path):
        return None
    ivs, tok = [], 0
    with open(path, errors="replace") as f:
        f.seek(since)
        for m in re.finditer(
                r"\[req\] rid=(\d+).*?dec=(\d+) dec_ms=(\d+).*?tps=([0-9.]+).*? t=(\d+)",
                f.read()):
            dec, dec_ms, t_end = int(m.group(2)), float(m.group(3)), float(m.group(5))
            ivs.append((t_end - dec_ms, t_end)); tok += dec
    if not ivs:
        return None
    u = union_ms(ivs)
    return {"reqs": len(ivs), "tok": tok, "agg": tok * 1000.0 / u if u else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--tap", required=True)
    ap.add_argument("--c", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--model", default="", help="request model field; "
                    "defaults to --label (legs boot with --model-id <leg>)")
    a = ap.parse_args()

    tap_off = os.path.getsize(a.tap) if os.path.exists(a.tap) else 0
    srv_off = os.path.getsize(a.server_log) if (a.server_log and os.path.exists(a.server_log)) else 0

    results = [None] * a.c
    threads = [threading.Thread(target=fire,
                                args=(i, a.url, a.max_tokens, results,
                                      a.model or a.label or "bench"))
               for i in range(a.c)]
    t_wall0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t_wall0
    time.sleep(0.5)  # let the tap flush its last records

    recs = []
    with open(a.tap, errors="replace") as f:
        f.seek(tap_off)
        for ln in f:
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if d.get("status") == 200 and d.get("out_tok"):
                recs.append(d)

    if len(recs) != a.c:
        print(f"WARN: expected {a.c} tap records, got {len(recs)}", file=sys.stderr)
    if not recs:
        print(f"C={a.c} FAILED: no tap records")
        return

    ivs = [(r["t_start"] * 1000 + r["ttft_ms"], r["t_start"] * 1000 + r["wall_ms"])
           for r in recs]
    tok = sum(r["out_tok"] for r in recs)
    u = union_ms(ivs)
    agg = tok * 1000.0 / u if u else 0.0
    per = [r["decode_tps"] for r in recs]
    ttfts = [r["ttft_ms"] for r in recs]

    print(f"C={a.c} {a.label} aggregate={agg:.1f} t/s  "
          f"(sum_dec={tok}, union={u/1000.0:.1f}s, client_wall={wall:.1f}s)")
    print(f"   per-stream tps: median={st.median(per):.1f} "
          f"min={min(per):.1f} max={max(per):.1f}")
    print(f"   ttft_ms: median={st.median(ttfts):.0f} max={max(ttfts):.0f}")

    srv = parse_server_log(a.server_log, srv_off)
    if srv:
        skew = (agg / srv["agg"] - 1) * 100 if srv["agg"] else 0
        print(f"   [calib] server-side convention: {srv['agg']:.1f} t/s "
              f"({srv['reqs']} reqs, {srv['tok']} tok)  client is {skew:+.1f}%")

    for i, r in enumerate(results):
        if r and r[1]:
            print(f"   stream {i} FAILED: {r[1]}", file=sys.stderr)

    print("JSON " + json.dumps({
        "c": a.c, "label": a.label, "aggregate_tps": round(agg, 1),
        "sum_dec": tok, "union_s": round(u / 1000.0, 1),
        "client_wall_s": round(wall, 1),
        "per_stream_median": round(st.median(per), 1),
        "ttft_ms_median": round(st.median(ttfts), 1),
        "n_streams": len(recs),
        "server_side_agg": round(srv["agg"], 1) if srv else None,
    }))


if __name__ == "__main__":
    main()
