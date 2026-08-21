#!/usr/bin/env python3
"""Drive a lot of tool calls at a q27 server and collect every dialect it emits.

WHY. Four spellings of the `<function...>` opener turned up inside two days, and
every one was found the same way: a session died, someone read the raw assistant
text, a mode got added. That is a reporting pipeline, not a test. It only sees
dialects that happen to break something a person was watching.

This asks the question directly instead. Declare a realistic tool schema, send
many prompts that require tool use, and record what comes back. A response that
carries a `tool_use` block parsed. A response that carries dialect markup as
plain TEXT did not, and is either a mode the parser is missing or a shape it
refuses on purpose -- both worth seeing, and neither shows up in a score.

The prompts deliberately spread across the shapes that have bitten before: long
code arguments (drift mode 11), several calls in one turn (mode 14's batch),
file edits with old/new string pairs (issue #24), and plain single calls.

usage: tool_dialect_survey.py --url http://127.0.0.1:8085 [--n 40] [--out DIR]
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter

# Claude Code's actual shapes, near enough that inference behaves the same way.
TOOLS = [
    {"name": "Bash", "description": "Run a shell command",
     "input_schema": {"type": "object", "properties": {
         "command": {"type": "string"}, "description": {"type": "string"}},
         "required": ["command"]}},
    {"name": "Read", "description": "Read a file",
     "input_schema": {"type": "object", "properties": {
         "file_path": {"type": "string"}, "limit": {"type": "integer"}},
         "required": ["file_path"]}},
    {"name": "Write", "description": "Write a file",
     "input_schema": {"type": "object", "properties": {
         "file_path": {"type": "string"}, "content": {"type": "string"}},
         "required": ["file_path", "content"]}},
    {"name": "Edit", "description": "Replace a string in a file",
     "input_schema": {"type": "object", "properties": {
         "file_path": {"type": "string"}, "old_string": {"type": "string"},
         "new_string": {"type": "string"}}, "required": ["file_path", "old_string", "new_string"]}},
    {"name": "Grep", "description": "Search file contents",
     "input_schema": {"type": "object", "properties": {
         "pattern": {"type": "string"}, "path": {"type": "string"}},
         "required": ["pattern"]}},
]

PROMPTS = [
    # single plain calls
    "List the files in /workspace. Use your tools.",
    "Read /workspace/package.json.",
    "Search /workspace for the string TODO.",
    "Show me the git status of /workspace.",
    "Count the lines in /workspace/src/index.ts.",
    # long code arguments -- the mode 11 shape
    "Write /workspace/src/server.ts containing a complete Express server with "
    "three routes, error handling middleware, and JSDoc comments on every function.",
    "Write /workspace/src/parser.py implementing a recursive descent parser for "
    "arithmetic expressions, with docstrings and inline comments.",
    "Create /workspace/config.yaml with a full CI pipeline definition: build, "
    "test, lint and deploy stages, each with several steps.",
    # edits -- the issue #24 shape
    "In /workspace/main.go, replace the import block with one that also imports "
    "bytes and compress/gzip.",
    "Edit /workspace/src/app.js to change the port from 3000 to 8080.",
    # several calls in one turn -- the mode 14 batch shape
    "Read both /workspace/a.txt and /workspace/b.txt, then tell me which is longer.",
    "Check whether /workspace/src, /workspace/test and /workspace/docs exist.",
    "Read /workspace/package.json, then search the src directory for its main entry point.",
    # awkward arguments
    "Run a shell command that prints a JSON object with nested quotes in it.",
    "Write /workspace/note.md whose content contains the literal text </parameter> "
    "and <function=Bash> inside a fenced code block.",
    "Search for the regex ^\\s*<function=[A-Za-z]+> across /workspace.",
]

MARKUP = re.compile(r"<function|<tool_call|<tool_name>|<parameter=|<invoke|<name>")

# Markup inside a fenced block or a backtick span is the model WRITING ABOUT the
# dialect, not emitting it, and the server is right to leave it as text -- the
# StreamSplitter makes the same distinction (display_text_context_is_executable).
# Found the hard way on 2026-08-20: capture 015 of the first survey run was a
# prose report whose markdown table quoted `^\s*<function=[A-Za-z]+>` from the
# prompt six times. It was counted as a dialect miss, inflating the failure
# rate, and it was the ONLY capture of the six that the parser had handled
# correctly all along. Strip the inert spans before matching.
FENCE = re.compile(r"```.*?```|~~~.*?~~~", re.S)
CODESPAN = re.compile(r"`[^`\n]*`")


def executable_text(text):
    """The subset of a response where dialect markup would be a real call."""
    return CODESPAN.sub("", FENCE.sub("", text))


def post(url, body, timeout=600):
    req = urllib.request.Request(url + "/v1/messages",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json",
                                          "x-api-key": "survey",
                                          "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=len(PROMPTS))
    ap.add_argument("--out", default="scratchpad/dialect-survey")
    ap.add_argument("--max-tokens", type=int, default=2048)
    # Single-shot prompts are the easy case and they all pass. Every dialect
    # found so far came out of a LONG session: context grows, tool results pile
    # up, and the model drifts off the trained format somewhere past turn 5.
    # --turns replays each prompt as a conversation, feeding a synthetic
    # tool_result back for every call so the transcript keeps growing.
    ap.add_argument("--turns", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    stats = Counter()
    unparsed = []
    for i in range(a.n):
        prompt = PROMPTS[i % len(PROMPTS)]
        msgs = [{"role": "user", "content": prompt}]
        for turn in range(a.turns):
            body = {"model": "q27", "max_tokens": a.max_tokens, "tools": TOOLS,
                    "messages": msgs}
            try:
                r = post(a.url, body)
            except Exception as e:  # a survey should not die on one bad response
                stats["request_error"] += 1
                print(f"  [{i:3}.{turn}] request error: {e}", flush=True)
                break
            blocks = r.get("content") or []
            kinds = [b.get("type") for b in blocks]
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            calls = [b for b in blocks if b.get("type") == "tool_use"]
            stop = r.get("stop_reason")
            if calls:
                stats["tool_use"] += 1
                msgs.append({"role": "assistant", "content": blocks})
                # Synthetic results keep the conversation growing without a
                # sandbox. Plausible-looking output matters: a bare "ok" makes
                # the model wrap up early and the transcript never gets long.
                msgs.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": c.get("id", "x"),
                     "content": "total 12\ndrwxr-xr-x 3 u u 4096 src\n-rw-r--r-- 1 u u 220 index.ts\n"}
                    for c in calls]})
                continue
            if MARKUP.search(executable_text(text)):
                stats["UNPARSED"] += 1
                # A truncated response is a DIFFERENT finding: the server
                # deliberately refuses to execute a half-written call, so the
                # markup landing in text is the guard working, not a missing
                # mode. Counted separately or the survey inflates its own
                # failure rate with its own max_tokens.
                if stop == "max_tokens":
                    stats["truncated"] += 1
                    print(f"  [{i:3}.{turn}] truncated (max_tokens) -- not a dialect miss",
                          flush=True)
                    break
                # Block STRUCTURE matters as much as the bytes: the server
                # parses text as it flushes, so an emission interrupted by a
                # channel switch reaches the parser in fragments that this
                # concatenation would silently rejoin.
                fn = os.path.join(a.out, f"unparsed.{i:03}.t{turn}.txt")
                open(fn, "w").write(prompt + "\n\n===BLOCKS===\n" +
                                    json.dumps([{"type": b.get("type"),
                                                 "len": len(b.get("text", "") or "")}
                                                for b in blocks]) +
                                    "\n\n===RESPONSE===\n" + text)
                # classification used the stripped text; the EXCERPT offset
                # has to come from the original or the slice below is shifted.
                m = MARKUP.search(text)
                print(f"  [{i:3}.{turn}] UNPARSED -> {fn}", flush=True)
                print(f"        {text[max(0, m.start()-40):m.start()+140]!r}", flush=True)
                unparsed.append(text)
            else:
                stats["text_only"] += 1
            break

    print("\n=== survey ===")
    for k, v in stats.most_common():
        print(f"  {k:16} {v}")
    if unparsed:
        print(f"\n{len(unparsed)} unparsed emission(s) written to {a.out}/")
    return 1 if stats["UNPARSED"] else 0


if __name__ == "__main__":
    sys.exit(main())
