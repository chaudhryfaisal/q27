#!/usr/bin/env python3
"""Tool-call success rate, measured identically on any engine's transcripts.

WHY NOT TASK SCORE. bench-task-queue scores 0.000 whether we drop fourteen calls
or the model simply cannot write a task queue, and at n=3 with bimodal outcomes
the mean moves on one trial flipping. Steering parity by hidden-test score means
steering by noise. This counts the thing the parser actually controls.

THE METRIC. Per assistant turn, from the client's own stream-json transcript, so
it is engine-agnostic and needs no server-side cooperation:

    executed   turn carried >=1 tool_use block
    MISSED     turn carried dialect markup in its TEXT and no tool_use --
               the model tried to call something and the bytes reached the user
    prose      neither: an ordinary answer

    success = executed / (executed + MISSED)

Markup inside a fenced block or a backtick span does NOT count: that is the
model writing ABOUT the dialect, and the server is right to leave it as text.
Skipping that check inflated a survey's failure rate on 2026-08-20 (a prose
report quoting the dialect six times read as six misses), so it is applied here
from the start.

A turn that stops on max_tokens mid-call is counted separately. It is a real
loss but it is truncation, not a parse failure, and folding the two together
hides which one is moving.

A miss is not automatically a parser gap. Some emissions cannot be recovered by
anything -- a call truncated mid-value, a `<parameter=` with no name -- and
executing them would mean inventing content. Pass --dump DIR to write every
missed turn out, then replay them through the current parser with
build/replay_missed_calls to split "we could fix this" from "we should not".

usage: toolcall_parity.py <run-dir> [<run-dir> ...] [--dump DIR]
"""
import json
import os
import re
import sys
from collections import Counter

MARKUP = re.compile(r"<function|<tool_call|<tool_name>|<parameter=|<invoke|<name>")
FENCE = re.compile(r"```.*?```|~~~.*?~~~", re.S)
CODESPAN = re.compile(r"`[^`\n]*`")


def executable_text(text):
    """The subset of a response where dialect markup would be a real call."""
    return CODESPAN.sub("", FENCE.sub("", text))


DUMP = {"dir": None, "n": 0}


def dump_miss(text, who):
    if not DUMP["dir"]:
        return
    os.makedirs(DUMP["dir"], exist_ok=True)
    fn = os.path.join(DUMP["dir"], "miss.%03d.txt" % DUMP["n"])
    open(fn, "w").write(who + "\n===\n" + text)
    DUMP["n"] += 1


def walk_transcript(path, st):
    """Classify every assistant turn in one trial."""
    if not os.path.exists(path):
        return
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "assistant":
            continue
        msg = d.get("message") or {}
        content = msg.get("content") or []
        if not isinstance(content, list):
            continue
        kinds = [c.get("type") for c in content]
        text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
        st["turns"] += 1
        if "tool_use" in kinds:
            st["executed"] += 1
            st["calls"] += sum(1 for k in kinds if k == "tool_use")
        elif MARKUP.search(executable_text(text)):
            # truncation is a different loss from a parse failure
            if msg.get("stop_reason") == "max_tokens":
                st["truncated"] += 1
            else:
                st["missed"] += 1
                dump_miss(text, path)
        else:
            st["prose"] += 1


def score_run(run_dir):
    """-> {orchestrator: Counter} for one thunderdome run directory."""
    out = {}
    trials = os.path.join(run_dir, "trials")
    if not os.path.isdir(trials):
        return out
    for orch in sorted(os.listdir(trials)):
        st = out.setdefault(orch, Counter())
        for task in sorted(os.listdir(os.path.join(trials, orch))):
            tdir = os.path.join(trials, orch, task)
            if not os.path.isdir(tdir):
                continue
            for trial in sorted(os.listdir(tdir)):
                st["trials"] += 1
                walk_transcript(
                    os.path.join(tdir, trial, "workspace", ".thunderdome-output.jsonl"), st)
    return out


def main():
    argv = sys.argv[1:]
    if "--dump" in argv:
        i = argv.index("--dump")
        DUMP["dir"] = argv[i + 1]
        del argv[i:i + 2]
    dirs = [a for a in argv if not a.startswith("-")]
    if not dirs:
        sys.exit(__doc__)
    totals = {}
    for d in dirs:
        for orch, st in score_run(d).items():
            totals.setdefault(orch, Counter()).update(st)
    print(f"{'orchestrator':34}{'trials':>7}{'turns':>7}{'exec':>7}{'calls':>7}"
          f"{'MISS':>6}{'trunc':>6}{'prose':>7}{'success':>9}")
    print("-" * 90)
    for orch, st in sorted(totals.items()):
        denom = st["executed"] + st["missed"]
        rate = (st["executed"] / denom) if denom else float("nan")
        print(f"{orch:34}{st['trials']:>7}{st['turns']:>7}{st['executed']:>7}{st['calls']:>7}"
              f"{st['missed']:>6}{st['truncated']:>6}{st['prose']:>7}{rate:>9.4f}")
    if DUMP["dir"]:
        print(f"\n{DUMP['n']} missed turn(s) written to {DUMP['dir']}/ -- replay them with:")
        print(f"  make build/replay_missed_calls && "
              f"./build/replay_missed_calls <tools.json> {DUMP['dir']}/miss.*.txt")


if __name__ == "__main__":
    main()
