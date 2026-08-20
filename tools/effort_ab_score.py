#!/usr/bin/env python3
"""Score the reasoning-effort A/B from thunderdome trial metadata.

The ONE rule that matters, and the one that was gotten wrong before (BUILDLOG
2026-08-15, the retracted 0.336): thunderdome's meta.json splits per task class.
Real-repo tasks carry `scores.tests` alone; greenfield tasks carry `tests`
(VISIBLE, what the agent's own suite says) and `hidden_tests` (the actual
discriminator) as separate keys. So:

    hidden = scores.hidden_tests if present else scores.tests

NOT composite_score -- that folds in static analysis and code metrics, and it
is what the CLI prints, which is why it is easy to quote by accident.

Also reports mean tokens per trial, because it separates two failure modes that
score identically: a task that was attempted and got it wrong (high tokens, low
hidden) versus one the model declined to attempt (low tokens, low hidden). The
xhigh arm produced both.

usage: effort_ab_score.py <outdir>   # outdir holds runmap.tsv
"""
import json
import os
import re
import sys
from collections import defaultdict

TD = "/mnt/ai/projects/thunderdome/results/runs"

# A session ends when Claude Code gets an assistant turn carrying no tool_use
# block -- but that is ALSO how every normal completion ends, so it cannot be
# the discriminator on its own. Two earlier attempts at this metric were wrong:
#
#   "< 60K tokens"            measures "stopped early", not "stopped on a parse
#                             failure". The canonical pre-fix example ran six
#                             turns and made two successful tool calls first.
#   any dialect markup        flags trials that scored 1.000 -- a summary turn
#     in the final turn       can legitimately contain <parameter= or <function.
#
# What actually identifies the failure is markup the OLD parser REFUSED, in a
# final turn that delivered no tool_use: the model issued a call, the parser
# would not convert it, and the loop fell out. Post-fix those same bytes parse,
# so they never reach the text channel and the count goes to zero -- which is
# exactly the prediction the fix is judged on.
REFUSED = [
    re.compile(r'<function\s+name\s*='),                 # mode 19: attribute opener
    re.compile(r'<tool_calls>\s*\n\s*<tool_name>\s*\n\s*<parameter='),  # mode 20: nameless
]


def stopped_on_unparsed(stream_path):
    """True iff the last assistant turn had no tool_use and shows dialect markup."""
    if not os.path.exists(stream_path):
        return False
    last_kinds, last_text = None, ""
    for line in open(stream_path, errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "assistant":
            continue
        content = d.get("message", {}).get("content", [])
        last_kinds = [c.get("type") for c in content]
        last_text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
    if last_kinds is None:
        return False
    if "tool_use" in last_kinds:
        return False
    return any(r.search(last_text) for r in REFUSED)


def hidden_of(meta):
    s = meta.get("scores") or {}
    return s["hidden_tests"] if "hidden_tests" in s else s.get("tests")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    mapping = os.path.join(out, "runmap.tsv")
    if not os.path.exists(mapping):
        sys.exit(f"no runmap.tsv in {out}")

    # (arm, task) -> list of (hidden, tokens)
    cells = defaultdict(list)
    arms, tasks = [], []
    for line in open(mapping):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        arm, task, rundir = parts
        if arm not in arms:
            arms.append(arm)
        if task not in tasks:
            tasks.append(task)
        base = os.path.join(TD, rundir, "trials", "claude-code-q27-haight", task)
        if not os.path.isdir(base):
            continue
        for trial in sorted(os.listdir(base)):
            mp = os.path.join(base, trial, "meta.json")
            if not os.path.exists(mp):
                continue
            m = json.load(open(mp))
            h = hidden_of(m)
            if h is None:
                continue
            stream = os.path.join(base, trial, "workspace", ".thunderdome-output.jsonl")
            cells[(arm, task)].append((float(h), int(m.get("total_tokens") or 0),
                                       stopped_on_unparsed(stream)))

    print(f"{'task':30}" + "".join(f"{a:>22}" for a in arms))
    print(f"{'':30}" + "".join(f"{'hidden   tok   n':>22}" for _ in arms))
    print("-" * (30 + 22 * len(arms)))
    arm_means = defaultdict(list)
    for t in tasks:
        row = f"{t:30}"
        for a in arms:
            v = cells.get((a, t), [])
            if not v:
                row += f"{'--':>22}"
                continue
            hs = [x[0] for x in v]
            tk = sum(x[1] for x in v) / len(v)
            arm_means[a].append(sum(hs) / len(hs))
            row += f"{sum(hs)/len(hs):>10.3f}{str(round(tk/1000))+'k':>7}{len(hs):>4}"
        print(row)
    print("-" * (30 + 22 * len(arms)))
    row = f"{'MEAN hidden (all tasks)':30}"
    for a in arms:
        m = arm_means[a]
        row += f"{(sum(m)/len(m) if m else 0):>10.3f}{'':>12}"
    print(row)

    # the number the fix is judged on
    print()
    for a in arms:
        tot = sum(len(cells.get((a, t), [])) for t in tasks)
        bad = sum(1 for t in tasks for x in cells.get((a, t), []) if x[2])
        print(f"  {a:8} stopped on an unparsed call: {bad}/{tot}")

    print("\nper-trial detail (hidden, tokens, * = stopped on unparsed call):")
    for t in tasks:
        print(f"  {t}")
        for a in arms:
            v = cells.get((a, t), [])
            if v:
                print(f"    {a:8} " + "  ".join(
                    f"{h:.3f}/{tk//1000}k{'*' if bad else ''}" for h, tk, bad in v))


if __name__ == "__main__":
    main()
