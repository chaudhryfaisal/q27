#!/usr/bin/env python3
"""Exercise auth and truncated Responses lifecycles on the production server."""

import argparse
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time

API_KEY = "q27-production-gate-key"
CODEX_HEADERS = (
    ("none", {}),
    ("legacy", {"x-codex-installation-id": "q27-gate"}),
    ("current", {"x-codex-turn-metadata": "{}"}),
)


def fail(message):
    raise RuntimeError(message)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(port, method, path, body=None, headers=None, timeout=300):
    payload = None if body is None else json.dumps(body).encode()
    request_headers = dict(headers or {})
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
        request_headers["Content-Length"] = str(len(payload))
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(method, path, body=payload, headers=request_headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def require_status(label, actual, expected):
    if actual != expected:
        fail(f"{label}: expected HTTP {expected}, got {actual}")


def parse_json(label, payload):
    try:
        return json.loads(payload)
    except Exception as error:
        fail(f"{label}: invalid JSON: {error}: {payload[:500]!r}")


def parse_sse(label, payload):
    events = []
    for line in payload.decode("utf-8").splitlines():
        if line.startswith("data: "):
            events.append(parse_json(label, line[6:]))
    if not events:
        fail(f"{label}: no SSE data events: {payload[:500]!r}")
    return events


def assert_incomplete_response(label, response, reasoning_limited):
    if response.get("status") != "incomplete":
        fail(f"{label}: terminal status is not incomplete: {response!r}")
    if response.get("incomplete_details") != {"reason": "max_output_tokens"}:
        fail(f"{label}: missing max_output_tokens details: {response!r}")
    usage = response.get("usage", {})
    details = usage.get("output_tokens_details", {})
    if details.get("reasoning_budget_exceeded") is not reasoning_limited:
        fail(f"{label}: wrong reasoning budget state: {details!r}")
    output = response.get("output")
    if not isinstance(output, list) or not output:
        fail(f"{label}: expected at least one output item: {response!r}")
    statuses = [item.get("status") for item in output if isinstance(item, dict)]
    if "incomplete" not in statuses:
        fail(f"{label}: no incomplete output item: {statuses!r}")


def run_response_case(port, header_label, codex_headers, stream, reasoning_limited):
    label = f"{header_label}/{'stream' if stream else 'nonstream'}/{'reasoning' if reasoning_limited else 'token'}"
    headers = {"Authorization": f"Bearer {API_KEY}", **codex_headers}
    input_text = (
        "Reply with exactly these eight words: alpha beta gamma delta epsilon zeta eta theta."
        if reasoning_limited
        else "Reply with the words hello world and nothing else."
    )
    body = {
        "input": input_text,
        "stream": stream,
        "max_output_tokens": 8 if reasoning_limited else 1,
        "enable_thinking": reasoning_limited,
    }
    if reasoning_limited:
        body["thinking_token_budget"] = 0
    status, response_headers, payload = request(
        port, "POST", "/v1/responses", body, headers
    )
    require_status(label, status, 200)
    if stream:
        content_type = response_headers.get("Content-Type", "")
        if not content_type.startswith("text/event-stream"):
            fail(f"{label}: wrong content type {content_type!r}")
        events = parse_sse(label, payload)
        terminal = events[-1]
        if terminal.get("type") != "response.incomplete":
            fail(f"{label}: wrong terminal event: {terminal!r}")
        assert_incomplete_response(label, terminal.get("response", {}), reasoning_limited)
        done_items = [
            event.get("item", {})
            for event in events
            if event.get("type") == "response.output_item.done"
        ]
        if not done_items or "incomplete" not in [item.get("status") for item in done_items]:
            fail(f"{label}: stream has no incomplete output_item.done: {done_items!r}")
    else:
        assert_incomplete_response(label, parse_json(label, payload), reasoning_limited)
    print(f"{label}: PASS")


def tool_body(max_output_tokens):
    return {
        "input": "Call the echo tool with text hello world. Do not answer directly.",
        "stream": True,
        "max_output_tokens": max_output_tokens,
        "enable_thinking": False,
        "tools": [{
            "type": "function",
            "name": "echo",
            "description": "Echo text",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        }],
        "tool_choice": "required",
    }


def require_balanced_output_items(label, events):
    added = {
        event.get("item", {}).get("id")
        for event in events
        if event.get("type") == "response.output_item.added"
    }
    done = {
        event.get("item", {}).get("id")
        for event in events
        if event.get("type") == "response.output_item.done"
    }
    dangling = added - done
    if dangling:
        fail(f"{label}: output items left in progress: {sorted(dangling)!r}")


def run_tool_stream_cases(port):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "x-codex-turn-metadata": "{}",
    }

    label = "tool/stream/truncated"
    status, response_headers, payload = request(
        port, "POST", "/v1/responses", tool_body(12), headers
    )
    require_status(label, status, 200)
    if not response_headers.get("Content-Type", "").startswith("text/event-stream"):
        fail(f"{label}: wrong content type {response_headers!r}")
    events = parse_sse(label, payload)
    terminal = events[-1]
    error = terminal.get("response", {}).get("error", {})
    if (terminal.get("type") != "response.failed" or
            error.get("code") != "invalid_model_output"):
        fail(f"{label}: unexpected terminal event: {terminal!r}")
    call_item_events = [
        event for event in events
        if (event.get("type") in ("response.output_item.added",
                                  "response.output_item.done") and
            event.get("item", {}).get("type") in ("function_call",
                                                    "custom_tool_call"))
    ]
    if call_item_events:
        fail(f"{label}: incomplete call was advertised: {call_item_events!r}")
    print(f"{label}: PASS")

    label = "tool/stream/completed"
    status, response_headers, payload = request(
        port, "POST", "/v1/responses", tool_body(64), headers
    )
    require_status(label, status, 200)
    events = parse_sse(label, payload)
    types = [event.get("type") for event in events]
    required = (
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    )
    if any(event_type not in types for event_type in required):
        fail(f"{label}: incomplete function-call lifecycle: {types!r}")
    positions = [types.index(event_type) for event_type in required]
    if positions != sorted(positions):
        fail(f"{label}: out-of-order function-call lifecycle: {types!r}")
    done_items = [
        event.get("item", {}) for event in events
        if event.get("type") == "response.output_item.done"
    ]
    if (not done_items or done_items[-1].get("status") != "completed" or
            done_items[-1].get("name") != "echo"):
        fail(f"{label}: wrong completed function call: {done_items!r}")
    arguments = parse_json(label, done_items[-1].get("arguments", ""))
    if arguments != {"text": "hello world"}:
        fail(f"{label}: wrong function arguments: {arguments!r}")
    require_balanced_output_items(label, events)
    print(f"{label}: PASS")


def wait_ready(process, port, log_path):
    deadline = time.monotonic() + 300
    last_error = "not started"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            status, _, _ = request(port, "GET", "/health", timeout=2)
            if status == 200:
                return
            last_error = f"HTTP {status}"
        except Exception as error:
            last_error = str(error)
        time.sleep(0.5)
    log_tail = Path(log_path).read_text(errors="replace")[-4000:]
    fail(f"server failed readiness ({last_error}); log tail:\n{log_tail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()

    server = Path(args.server).resolve()
    model = Path(args.model).resolve()
    tokenizer = Path(args.tokenizer).resolve()
    for label, path in (("server", server), ("model", model), ("tokenizer", tokenizer)):
        if not path.is_file():
            fail(f"{label} does not exist: {path}")

    port = free_port()
    with tempfile.NamedTemporaryFile(prefix="q27-responses-gate-", suffix=".log", delete=False) as log:
        log_path = log.name
        process = subprocess.Popen(
            [
                str(server), str(model), str(tokenizer),
                "--host", "127.0.0.1", "--port", str(port),
                "--ctx", "4096", "--slots", "1", "--request-think",
                "--api-key", API_KEY,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        wait_ready(process, port, log_path)
        status, _, payload = request(port, "GET", "/health")
        require_status("health exemption", status, 200)
        if parse_json("health exemption", payload).get("status") != "ok":
            fail(f"health exemption: unexpected body {payload!r}")

        for label, headers, expected in (
            ("models missing key", {}, 401),
            ("models wrong key", {"Authorization": "Bearer wrong"}, 401),
            ("models bearer key", {"Authorization": f"Bearer {API_KEY}"}, 200),
            ("models x-api-key", {"x-api-key": API_KEY}, 200),
        ):
            status, _, _ = request(port, "GET", "/v1/models", headers=headers)
            require_status(label, status, expected)
            print(f"{label}: PASS")

        rejected_body = {"input": "This must not reach generation.", "max_output_tokens": 1}
        for label, headers in (
            ("responses missing key", {}),
            ("responses wrong key", {"x-api-key": "wrong"}),
        ):
            status, _, _ = request(port, "POST", "/v1/responses", rejected_body, headers)
            require_status(label, status, 401)
            print(f"{label}: PASS")

        for header_label, codex_headers in CODEX_HEADERS:
            for stream in (False, True):
                for reasoning_limited in (False, True):
                    run_response_case(
                        port, header_label, codex_headers, stream, reasoning_limited
                    )
        run_tool_stream_cases(port)
        print("all production Responses integration tests passed")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        Path(log_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
