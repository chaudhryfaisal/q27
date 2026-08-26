#!/usr/bin/env python3
"""Regression test for the tap's TTFT/decode split.

Stands up a fake engine that emits 10 SSE deltas 50 ms apart -- a total body
well under one 8 KB read. With HTTPResponse.read(8192) the whole stream lands
in a single block at EOF, TTFT swallows the decode, and decode_tps explodes.
With read1 the timestamps are real.

Asserts the shape, not exact times: ttft near the first delta, wall near the
last, decode rate in a sane band.
"""
import json
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

N_DELTAS = 10
GAP = 0.05
PRE = 0.20  # simulated prefill before the first token


class Fake(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def ev(obj):
            b = b"data: " + json.dumps(obj).encode() + b"\n\n"
            self.wfile.write(b"%x\r\n" % len(b) + b + b"\r\n")
            self.wfile.flush()

        time.sleep(PRE)
        ev({"type": "message_start",
            "message": {"usage": {"input_tokens": 100, "output_tokens": 0}}})
        for i in range(N_DELTAS):
            ev({"type": "content_block_delta",
                "delta": {"type": "text_delta", "text": f"tok{i} "}})
            time.sleep(GAP)
        ev({"type": "message_delta", "usage": {"output_tokens": N_DELTAS}})
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def main():
    srv = HTTPServer(("127.0.0.1", 8577), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    log = "/tmp/claude-1000/-home-gabe/720f8f0a-ec38-448d-b159-d6b56516fbc7/scratchpad/ab/test_tap.jsonl"
    open(log, "w").close()
    tap = subprocess.Popen(
        [sys.executable,
         "/tmp/claude-1000/-home-gabe/720f8f0a-ec38-448d-b159-d6b56516fbc7/scratchpad/ab/tapproxy.py",
         "--listen", "8578", "--upstream", "127.0.0.1:8577", "--log", log],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8578/v1/messages",
            data=json.dumps({"messages": []}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            for _ in r:
                pass
        time.sleep(0.5)
        rec = json.loads(open(log).read().strip().splitlines()[-1])
    finally:
        tap.kill()
        srv.shutdown()

    body_bytes = "measured"
    exp_ttft = PRE * 1000
    exp_wall = (PRE + N_DELTAS * GAP) * 1000
    print(f"  record: {json.dumps(rec)}")
    print(f"  expect ttft ~{exp_ttft:.0f}ms, wall ~{exp_wall:.0f}ms, "
          f"decode ~{N_DELTAS / (N_DELTAS * GAP):.0f} t/s")

    ok = True
    if not (exp_ttft * 0.5 < rec["ttft_ms"] < exp_ttft * 2.5):
        print(f"  FAIL ttft_ms={rec['ttft_ms']} not near {exp_ttft}"); ok = False
    if not (exp_wall * 0.7 < rec["wall_ms"] < exp_wall * 1.6):
        print(f"  FAIL wall_ms={rec['wall_ms']} not near {exp_wall}"); ok = False
    dec = rec["wall_ms"] - rec["ttft_ms"]
    if dec < 100:
        print(f"  FAIL decode window {dec:.2f}ms collapsed -- read() buffering"); ok = False
    if not (5 < rec["decode_tps"] < 80):
        print(f"  FAIL decode_tps={rec['decode_tps']} outside sane band"); ok = False
    if rec["out_tok"] != N_DELTAS:
        print(f"  FAIL out_tok={rec['out_tok']} != {N_DELTAS}"); ok = False

    print("  PASS" if ok else "  FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
