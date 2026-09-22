#!/usr/bin/env python3
"""Thinking chars per API turn by turn DEPTH, per leg, from retained Claude Code
transcripts (/mnt/ai/swebench-work/<leg>/<iid>/logs/out.jsonl). Median over
all turns at that depth across instances. usage: depth_think.py leg..."""
import json, glob, sys, statistics as st
W="/mnt/ai/swebench-work"
def transcript(path):
    msgs={}; order=[]
    for ln in open(path, errors="replace"):
        try: d=json.loads(ln)
        except: continue
        if d.get("type")!="assistant": continue
        m=d["message"]; mid=m.get("id")
        if mid not in msgs: msgs[mid]=dict(think=0,text=0,tools=0); order.append(mid)
        for b in m.get("content",[]):
            t=b.get("type")
            if t=="thinking": msgs[mid]["think"]+=len(b.get("thinking",""))
            elif t=="text": msgs[mid]["text"]+=len(b.get("text",""))
            elif t=="tool_use": msgs[mid]["tools"]+=1
    return [msgs[m] for m in order]
buckets=[(0,0),(1,1),(2,2),(3,4),(5,7),(8,11),(12,15),(16,23),(24,99)]
for leg in sys.argv[1:]:
    per={b:[] for b in buckets}; narr={b:[] for b in buckets}; n=0
    for f in sorted(glob.glob(f"{W}/{leg}/*/logs/out.jsonl")):
        T=transcript(f); n+=1
        for i,t in enumerate(T):
            for b in buckets:
                if b[0]<=i<=b[1]: per[b].append(t["think"]); narr[b].append(1 if t["text"]>0 and t["tools"]>0 else 0)
    print(f"== {leg} ({n} instances)")
    print("  depth      n  think med  think mean  narrated")
    for b in buckets:
        v=per[b]
        if not v: continue
        print(f"  {b[0]:2d}-{b[1]:<2d} {len(v):5d} {st.median(v):9.0f} {st.mean(v):10.0f} {sum(narr[b])/len(v):8.0%}")
