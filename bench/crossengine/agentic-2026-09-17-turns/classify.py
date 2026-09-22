#!/usr/bin/env python3
"""Classify first actions from state_replay.py outputs and compare arms.
usage: classify.py <label=path.jsonl> ..."""
import json, sys, re
from collections import Counter, defaultdict
RUN = re.compile(r"\b(python3?|pytest|pip3?|py\.test|tox)\b")
ORIENT = re.compile(r"^\s*(ls|git\s+(log|status|show|diff)|find\s|tree|pwd|cat\s+\S*(README|setup|pyproject))")
def cls(r):
    t, a = r["tool"], r["arg"] or ""
    if t is None: return "no call"
    if t == "Bash":
        if RUN.search(a): return "run code"
        if re.match(r"\s*(pip|apt|conda|uv|poetry)\b", a): return "install"
        if ORIENT.search(a): return "orient (ls/git/find)"
        if re.search(r"\b(grep|rg|ugrep|sed -n|head|cat|awk)\b", a): return "inspect (grep/cat)"
        return "other bash"
    if t in ("Read", "Grep", "Glob"): return "read/grep tool"
    if t in ("Edit", "Write", "MultiEdit"): return "edit"
    if t.startswith("Web"): return "web"
    return t
arms = {}
for a in sys.argv[1:]:
    k, p = a.split("=", 1); arms[k] = [json.loads(l) for l in open(p) if l.strip()]
cats = sorted({cls(r) for rs in arms.values() for r in rs})
print(f"{'arm':14s} {'n':>4s} " + " ".join(f"{c[:16]:>16s}" for c in cats) + "   think med")
for k, rs in arms.items():
    c = Counter(cls(r) for r in rs); n = len(rs); th = sorted(r["think"] for r in rs)
    print(f"{k:14s} {n:4d} " + " ".join(f"{c[x]/n:15.0%} " for x in cats) + f"   {th[n//2] if th else 0}")
# per-state paired view for arms sharing states
if len(arms) >= 2:
    ks = list(arms)
    by = {k: defaultdict(Counter) for k in ks}
    for k in ks:
        for r in arms[k]: by[k][r["state"].replace(".plain", "").replace(".marked", "")][cls(r)] += 1
    common = sorted(set.intersection(*(set(by[k]) for k in ks)))
    if common:
        print(f"\nper state, share 'run code' / 'edit' per arm ({len(common)} states):")
        for s in common:
            print(f"  {s:18s} " + "  ".join(f"{k}: run {by[k][s]['run code']}/{sum(by[k][s].values())} edit {by[k][s]['edit']}" for k in ks))
