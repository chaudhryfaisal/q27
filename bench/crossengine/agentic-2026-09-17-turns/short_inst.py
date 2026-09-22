#!/usr/bin/env python3
"""Per-instance trajectory shape from retained transcripts: turns, per-turn tool
names (abbrev) + thinking chars, gold. usage: short_inst.py leg iid..."""
import json, sys, glob
W="/mnt/ai/swebench-work"
AB={"Bash":"B","Read":"R","Edit":"E","Write":"W","Grep":"G","Glob":"L","MultiEdit":"M","TodoWrite":"T","Task":"A","WebFetch":"F","NotebookEdit":"N"}
def transcript(path):
    msgs={}; order=[]
    for ln in open(path, errors="replace"):
        try: d=json.loads(ln)
        except: continue
        if d.get("type")!="assistant": continue
        m=d["message"]; mid=m.get("id")
        if mid not in msgs: msgs[mid]=dict(think=0,text=0,tools=[]); order.append(mid)
        for b in m.get("content",[]):
            t=b.get("type")
            if t=="thinking": msgs[mid]["think"]+=len(b.get("thinking",""))
            elif t=="text": msgs[mid]["text"]+=len(b.get("text",""))
            elif t=="tool_use":
                i=b.get("input",{}); arg=str(i.get("command") or i.get("file_path") or i.get("pattern") or "")
                msgs[mid]["tools"].append(AB.get(b["name"],b["name"][:2])+("py" if b["name"]=="Bash" and ("python" in arg or "pytest" in arg) else ""))
    return [msgs[m] for m in order]
leg=sys.argv[1]
for iid in sys.argv[2:]:
    f=f"{W}/{leg}/{iid}/logs/out.jsonl"
    try: T=transcript(f)
    except FileNotFoundError: print(f"{leg} {iid}: missing"); continue
    seq=" ".join(f"{'+'.join(t['tools']) or 'end'}:{t['think']}" for t in T)
    print(f"{leg:9s} {iid:26s} turns={len(T):2d} | {seq}")
