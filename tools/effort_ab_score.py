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
import sys
from collections import defaultdict

TD = "/mnt/ai/projects/thunderdome/results/runs"


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
            cells[(arm, task)].append((float(h), int(m.get("total_tokens") or 0)))

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

    print("\nper-trial detail (hidden, tokens):")
    for t in tasks:
        print(f"  {t}")
        for a in arms:
            v = cells.get((a, t), [])
            if v:
                print(f"    {a:8} " + "  ".join(f"{h:.3f}/{tk//1000}k" for h, tk in v))


if __name__ == "__main__":
    main()
