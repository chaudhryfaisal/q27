#!/usr/bin/env python3
"""Engine-independent measurement tap.

Sits between the client (Claude Code, benchlocal, the ladder) and whichever
engine is under test, and records per-request timing to JSONL. Both engines
self-report throughput in their own log dialect and with their own definition
of "decode"; scraping those makes the A/B depend on two different accounting
conventions. This records ONE convention, client-observed, for both.

Streams SSE through byte-for-byte with no buffering, so TTFT stays honest --
the recorded first-token time is when the first content byte crossed the wire.

  usage: tapproxy.py --listen 8081 --upstream 127.0.0.1:8090 --log run.jsonl

Per-request record:
  t_start        wall clock at request receipt (epoch seconds)
  path           upstream path
  ttft_ms        request send -> first content delta byte
  wall_ms        request send -> upstream stream close
  in_tok         prompt/input tokens as reported in the response body
  out_tok        completion/output tokens as reported in the response body
  cache_read_tok cache_read_input_tokens when the engine reports it (Anthropic
                 dialect); null when absent. Drives the prefix-cache miss rate.
  decode_tps     out_tok / (wall - ttft), the number to compare across engines
  status         upstream HTTP status
"""
import argparse
import http.client
import json
import re
import socket
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler

LOCK = threading.Lock()
LOGF = None


def emit(rec):
    if LOGF is None:
        return
    with LOCK:
        LOGF.write(json.dumps(rec) + "\n")
        LOGF.flush()


class Tally:
    """Accumulates token counts + first-content time out of a streaming body.

    Handles both server dialects in one pass because the A/B drives the same
    proxy from Claude Code (Anthropic /v1/messages) and from benchlocal and the
    ladder (OpenAI /v1/chat/completions). Non-streaming JSON bodies fall through
    to a whole-body parse at close.
    """

    def __init__(self):
        self.ttft = None
        self.in_tok = 0
        self.out_tok = 0
        self.cache_read = None
        self.buf = b""
        self.saw_sse = False

    def feed(self, chunk, t0):
        self.buf += chunk
        # Keep the tail; SSE events are newline-delimited.
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            self.saw_sse = True
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                d = json.loads(payload)
            except Exception:
                continue
            self._event(d, t0)
        # Cap the carry so a non-SSE body can't grow unbounded here; the
        # whole-body path re-reads from its own buffer.
        if len(self.buf) > 1 << 20:
            self.buf = self.buf[-(1 << 16):]

    def _event(self, d, t0):
        typ = d.get("type")
        # ---- Anthropic Messages streaming ----
        if typ == "message_start":
            u = (d.get("message") or {}).get("usage") or {}
            self.in_tok = u.get("input_tokens", self.in_tok) or self.in_tok
            if u.get("cache_read_input_tokens") is not None:
                self.cache_read = u.get("cache_read_input_tokens")
        elif typ == "content_block_delta":
            if self.ttft is None:
                self.ttft = time.time() - t0
        elif typ == "message_delta":
            u = d.get("usage") or {}
            self.out_tok = u.get("output_tokens", self.out_tok) or self.out_tok
        # ---- OpenAI chat.completion streaming ----
        elif "choices" in d or "usage" in d:
            ch = d.get("choices") or []
            if ch:
                delta = ch[0].get("delta") or {}
                if (delta.get("content") or delta.get("reasoning_content")) and self.ttft is None:
                    self.ttft = time.time() - t0
            u = d.get("usage")
            if u:
                self.in_tok = u.get("prompt_tokens", self.in_tok) or self.in_tok
                self.out_tok = u.get("completion_tokens", self.out_tok) or self.out_tok

    def finish_nonstream(self, body):
        if self.saw_sse:
            return
        try:
            d = json.loads(body)
        except Exception:
            return
        u = d.get("usage") or {}
        self.in_tok = u.get("prompt_tokens") or u.get("input_tokens") or 0
        self.out_tok = u.get("completion_tokens") or u.get("output_tokens") or 0
        if u.get("cache_read_input_tokens") is not None:
            self.cache_read = u.get("cache_read_input_tokens")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "tapproxy"

    def log_message(self, *a):  # silence per-request stderr noise
        pass

    def _proxy(self, method):
        t0 = time.time()
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""

        # Cross-engine request normalization. Claude Code 2.1.170 sends
        # output_config={"effort":"high"}; q27 ignores unknown top-level knobs,
        # ninfer's Qwen3.6 chat template hard-rejects that one with a 400. The
        # A/B is only meaningful if both engines process the SAME request, so
        # the field is dropped here, uniformly, on every leg -- rather than
        # letting one engine silently discard what the other refuses.
        stripped = []
        if (self.server.strip_fields or self.server.translate) and body[:1] == b"{":
            try:
                obj = json.loads(body)
                for f in self.server.strip_fields:
                    if f in obj:
                        obj.pop(f)
                        stripped.append(f)
                # benchlocal's --no-thinking sends
                # chat_template_kwargs.enable_thinking; ninfer rejects that key
                # with chat_template_option_not_supported and wants the flag
                # TOP-LEVEL, while q27 ignores both. MOVE rather than drop it:
                # dropping would leave ninfer in its default thinking mode while
                # q27 stayed no-think, which is precisely the asymmetry the
                # normalization exists to prevent.
                if self.server.translate:
                    ctk = obj.get("chat_template_kwargs")
                    if isinstance(ctk, dict) and "enable_thinking" in ctk:
                        obj["enable_thinking"] = ctk.pop("enable_thinking")
                        if not ctk:
                            obj.pop("chat_template_kwargs")
                        stripped.append("ctk.enable_thinking->enable_thinking")
                if stripped:
                    body = json.dumps(obj).encode()
            except Exception:
                pass

        host, port = self.server.upstream
        try:
            conn = http.client.HTTPConnection(host, port, timeout=1800)
            hdrs = {k: v for k, v in self.headers.items()
                    if k.lower() not in ("host", "connection", "content-length",
                                         "accept-encoding")}
            hdrs["Content-Length"] = str(len(body))
            hdrs["Accept-Encoding"] = "identity"
            hdrs["Connection"] = "close"
            conn.request(method, self.path, body=body, headers=hdrs)
            resp = conn.getresponse()
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Length", "0")
            self.end_headers()
            emit({"t_start": t0, "path": self.path, "status": 502,
                  "error": str(e), "wall_ms": (time.time() - t0) * 1000})
            return

        self.send_response(resp.status)
        passthru = []
        for k, v in resp.getheaders():
            if k.lower() in ("connection", "transfer-encoding", "content-length"):
                continue
            self.send_header(k, v)
            passthru.append(k.lower())
        # Stream downstream with chunked framing so the client sees deltas as
        # they arrive; a Content-Length would force us to buffer the whole body
        # and destroy the TTFT we are here to measure.
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()

        tally = Tally()
        raw = bytearray()
        try:
            while True:
                # read1, NOT read: HTTPResponse.read(n) blocks until it has n
                # bytes or EOF, so any response smaller than the buffer arrives
                # as one block at end-of-stream -- TTFT then absorbs the whole
                # decode and the measured decode rate goes to infinity. read1
                # returns as soon as any bytes are available, which is the only
                # way the first-token timestamp means anything.
                chunk = resp.read1(8192)
                if not chunk:
                    break
                tally.feed(chunk, t0)
                if len(raw) < (1 << 20):
                    raw += chunk
                self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

        tally.finish_nonstream(bytes(raw))
        wall = time.time() - t0
        ttft = tally.ttft if tally.ttft is not None else wall
        dec = max(wall - ttft, 1e-6)
        # On a rejection, the status code alone says nothing actionable -- keep
        # the engine's own error text and the offending request so the cause is
        # diagnosable without re-running the whole leg.
        err = None
        if resp.status >= 400:
            # Rejections are rare; keep the whole request. The offending field
            # is often a top-level knob that sorts AFTER the (huge) messages
            # array, so a head-truncated capture reliably misses it.
            err = {"resp": bytes(raw)[:2000].decode("utf-8", "replace"),
                   "req": body.decode("utf-8", "replace")}
        emit({
            "t_start": t0,
            "path": self.path,
            "status": resp.status,
            "ttft_ms": round(ttft * 1000, 2),
            "wall_ms": round(wall * 1000, 2),
            "in_tok": tally.in_tok,
            "out_tok": tally.out_tok,
            "cache_read_tok": tally.cache_read,
            "decode_tps": round(tally.out_tok / dec, 2) if tally.out_tok else 0.0,
            "prefill_tps": round(tally.in_tok / ttft, 2) if (tally.in_tok and ttft > 0) else 0.0,
            **({"stripped": stripped} if stripped else {}),
            **({"error": err} if err else {}),
        })

    def do_POST(self):
        self._proxy("POST")

    def do_GET(self):
        self._proxy("GET")


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, required=True)
    ap.add_argument("--upstream", required=True, help="host:port")
    ap.add_argument("--log", required=True)
    ap.add_argument("--translate-thinking", action="store_true",
                    help="move chat_template_kwargs.enable_thinking to a "
                         "top-level enable_thinking field")
    ap.add_argument("--strip-fields", default="",
                    help="CSV of top-level request fields to drop before "
                         "forwarding, applied identically on every leg")
    a = ap.parse_args()
    host, port = a.upstream.split(":")
    LOGF = open(a.log, "a")
    srv = Server(("0.0.0.0", a.listen), Handler)
    srv.upstream = (host, int(port))
    srv.strip_fields = [x.strip() for x in a.strip_fields.split(",") if x.strip()]
    srv.translate = a.translate_thinking
    print(f"[tap] :{a.listen} -> {a.upstream}  log={a.log}  "
          f"strip={srv.strip_fields or 'none'}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
