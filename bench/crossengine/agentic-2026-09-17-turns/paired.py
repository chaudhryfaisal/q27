#!/usr/bin/env python3
"""Paired comparison of state_replay arms over common states: per-arm action
mix, thinking, narration (from .content.jsonl when present), prompt tokens,
and a per-state table. usage: paired.py <label=path.jsonl> ..."""
import json, os, sys, re, statistics as st
from collections import Counter, defaultdict
RUN = re.compile(r"\b(python3?|pytest|pip3?|py\.test|tox)\b"); ORIENT = re.compile(r"^\s*(ls|git\s+(log|status|show|diff)|find\s|tree|pwd)")
def cls(r):
    t, a = r["tool"], r["arg"] or ""
    if t is None: return "no call"
    if t == "Bash":
        if RUN.search(a): return "run"
        if re.match(r"\s*(pip|apt|conda|uv|poetry)\b", a): return "install"
        if ORIENT.search(a): return "orient"
        return "inspect"
    if t in ("Read", "Grep", "Glob"): return "read"
    if t in ("Edit", "Write", "MultiEdit"): return "edit"
    return "other"
arms = {}
for a in sys.argv[1:]:
    k, p = a.split("=", 1); rows = [json.loads(l) for l in open(p) if l.strip()]
    cp = p.replace(".jsonl", ".content.jsonl"); content = {}
    if os.path.exists(cp):
        for l in open(cp):
            if l.strip(): c = json.loads(l); content[(c["state"], c["seed"])] = c
    for r in rows:
        r["key"] = r["state"].replace(".plain", "").replace(".marked", "")
        c = content.get((r["state"], r["seed"]))
        r["narr"] = (any(x.get("type") == "text" and x.get("text", "").strip() for x in c["content"]) if c else None)
        r["ptok"] = c["usage"].get("input_tokens") if c else None
    arms[k] = rows
common = set.intersection(*(set(r["key"] for r in rs) for rs in arms.values()))
print(f"{len(common)} common states")
cats = ["orient", "inspect", "read", "run", "install", "edit", "no call", "other"]
print(f"{'arm':14s} {'n':>4s} " + " ".join(f"{c:>8s}" for c in cats) + f" {'think':>6s} {'narr':>5s} {'ptok':>7s}")
for k, rs in arms.items():
    rs = [r for r in rs if r["key"] in common]; n = len(rs); c = Counter(cls(r) for r in rs)
    nr = [r["narr"] for r in rs if r["narr"] is not None and r["tool"]]
    pt = [r["ptok"] for r in rs if r["ptok"]]
    print(f"{k:14s} {n:4d} " + " ".join(f"{c[x]/n:8.0%}" for x in cats) + f" {st.median(r['think'] for r in rs):6.0f} {(sum(nr)/len(nr) if nr else float('nan')):5.0%} {st.median(pt) if pt else 0:7.0f}")
if len(arms) >= 2:
    ks = list(arms)
    print("\nper state: think median | run+edit share, per arm")
    for s in sorted(common):
        cells = []
        for k in ks:
            rs = [r for r in arms[k] if r["key"] == s]
            cells.append(f"{k}: {st.median(r['think'] for r in rs):5.0f} | run {sum(cls(r)=='run' for r in rs)}/{len(rs)} edit {sum(cls(r)=='edit' for r in rs)}")
        print(f"  {s:18s} " + "   ".join(cells))
