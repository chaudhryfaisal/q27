#!/usr/bin/env python3
"""Reconstruct ninfer-trajectory deep states (Anthropic bodies) from the retained
Claude Code transcripts: q27's recorded turn-0 body for the same task supplies
system/tools/first user message; the ninfer transcript supplies the assistant
messages (thinking+signature, text, tool_use) and tool_result user messages.
Writes ndeep_<k>_p<N>[_edited].json + ndeep_select.txt. Self-check: the same
reconstruction from a q27tok transcript vs q27's actually recorded body."""
import json, glob, os, sys, re
D = os.path.dirname(os.path.abspath(__file__)); W = "/mnt/ai/swebench-work"
DEPTHS = (8, 12, 16)
def transcript(path):
    """Assistant blocks grouped by message id (parallel tool calls stream as
    separate records, interleaved with their results); consecutive tool_result
    records form one user message."""
    msgs = []; cur_user = None; by_id = {}
    for ln in open(path, errors="replace"):
        try: d = json.loads(ln)
        except: continue
        t = d.get("type")
        if t == "assistant":
            m = d["message"]; aid = m.get("id")
            if aid not in by_id:
                if cur_user: msgs.append(cur_user); cur_user = None
                by_id[aid] = {"role": "assistant", "content": []}; msgs.append(by_id[aid])
            tgt = by_id[aid]
            for b in m.get("content", []):
                if b.get("type") == "thinking": tgt["content"].append({"type": "thinking", "thinking": b.get("thinking", ""), "signature": b.get("signature", "")})
                elif b.get("type") == "text": tgt["content"].append({"type": "text", "text": b.get("text", "")})
                elif b.get("type") == "tool_use": tgt["content"].append({"type": "tool_use", "id": b["id"], "name": b["name"], "input": b.get("input", {})})
        elif t == "user":
            m = d["message"]; c = m.get("content")
            if cur_user is None: cur_user = {"role": "user", "content": []}
            if isinstance(c, list):
                for b in c:
                    if b.get("type") == "tool_result":
                        r = {"type": "tool_result", "tool_use_id": b["tool_use_id"], "content": b.get("content", "")}
                        if b.get("is_error"): r["is_error"] = True
                        cur_user["content"].append(r)
                    elif b.get("type") == "text": cur_user["content"].append({"type": "text", "text": b["text"]})
            elif isinstance(c, str): cur_user["content"].append({"type": "text", "text": c})
    if cur_user: msgs.append(cur_user)
    return msgs
REC = None
def skills_msg(iid):
    """Claude Code's mid-conversation system message (skills list), copied from a
    recorded q27 body of the same instance."""
    global REC
    if REC is None: REC = [json.loads(l)["body"] for l in open(os.environ.get("Q27_REQBODIES", "req.q27tok.jsonl"))]
    key = norm(open(f"{W}/q27tok/{iid}/task.txt").read())[:300]
    for b in REC:
        if isinstance(b, str): b = json.loads(b)
        ms = b.get("messages") or []
        if len(ms) > 1 and ms[1].get("role") == "system" and key in norm(first_user_text(b)): return ms[1]
    return None
def first_user_text(body):
    c = body["messages"][0]["content"]
    return "\n".join(x.get("text", "") for x in c) if isinstance(c, list) else c
def norm(s): return re.sub(r"\s+", " ", s).strip()
t0 = {}
for f in sorted(glob.glob(os.path.join(D, "turn0_*.json"))):
    if re.search(r"turn0_\d+\.json$", f): t0[f] = json.load(open(f))
def match_body(task_text):
    k = norm(task_text)[:300]
    hits = [f for f, b in t0.items() if k in norm(first_user_text(b))]
    return hits[0] if len(hits) == 1 else None
def recorded_body(iid, depth):
    global REC
    if REC is None: REC = [json.loads(l)["body"] for l in open(os.environ.get("Q27_REQBODIES", "req.q27tok.jsonl"))]
    key = norm(open(f"{W}/q27tok/{iid}/task.txt").read())[:300]
    for b in REC:
        if isinstance(b, str): b = json.loads(b)
        ms = b.get("messages") or []
        if "tools" in b and sum(1 for m in ms if m["role"] == "assistant") == depth and key in norm(first_user_text(b)): return b
    return None
def build(leg, iid, depth):
    task = open(f"{W}/{leg}/{iid}/task.txt").read()
    bf = match_body(task)
    if not bf: return None, "no turn0 body"
    base = json.loads(json.dumps(t0[bf]))
    T = transcript(f"{W}/{leg}/{iid}/logs/out.jsonl")
    a_idx = [i for i, m in enumerate(T) if m["role"] == "assistant"]
    if len(a_idx) < depth + 1: return None, f"only {len(a_idx)} assistant msgs"
    cut = a_idx[depth]  # messages before the (depth)th assistant message (0-based)
    hist = T[:cut]
    if not hist or hist[-1]["role"] != "user": return None, "history does not end in a user message"
    if any(m["role"] == "assistant" and not any(b["type"] == "tool_use" for b in m["content"]) for m in hist): return None, "an end-turn assistant message in history"
    sysm = skills_msg(iid)
    if sysm is None: return None, "no skills system message on record"
    msgs = [base["messages"][0], sysm] + hist
    # Claude Code's later mid-conversation system messages ("task tools haven't
    # been used recently", ...): copy them from q27's recorded body of the same
    # instance at the same depth, at the same message indices.
    rb = recorded_body(iid, depth)
    if rb is not None:
        for i, m in enumerate(rb["messages"]):
            if i > 1 and m.get("role") == "system" and i <= len(msgs): msgs.insert(i, m)
    base["messages"] = msgs
    edited = any(b.get("type") == "tool_use" and b["name"] in ("Edit", "Write", "MultiEdit", "NotebookEdit") for m in hist if m["role"] == "assistant" for b in m["content"])
    return base, ("edited" if edited else "")
if __name__ == "__main__":
    leg = sys.argv[1] if len(sys.argv) > 1 else "ninferd2"
    sel = []; k = 0
    for d in sorted(glob.glob(f"{W}/{leg}/*/")):
        iid = d.rstrip("/").split("/")[-1]; k += 1
        for depth in DEPTHS:
            body, tag = build(leg, iid, depth)
            if body is None: print(f"  {iid} p{depth}: skip ({tag})"); continue
            name = f"ndeep_{k}_p{depth}" + ("_edited" if tag else "")
            json.dump(body, open(os.path.join(D, name + ".json"), "w")); sel.append(name)
            print(f"  {iid} p{depth}: {name} ({len(body['messages'])} messages)")
    open(os.path.join(D, "ndeep_select.txt"), "w").write("\n".join(sel) + "\n"); print(len(sel), "states")
    # self-check on q27tok: reconstruction vs the recorded body at the same depth
    print("== self-check q27tok")
    rec = [json.loads(l) for l in open(os.environ.get("Q27_REQBODIES", "req.q27tok.jsonl"))]
    checked = 0
    for d in sorted(glob.glob(f"{W}/q27tok/*/"))[:6]:
        iid = d.rstrip("/").split("/")[-1]
        body, tag = build("q27tok", iid, 8)
        if body is None: continue
        key = norm(open(f"{W}/q27tok/{iid}/task.txt").read())[:300]
        cands = []
        for r in rec:
            b = r["body"] if isinstance(r["body"], dict) else json.loads(r["body"])
            if not b.get("messages") or "tools" not in b: continue
            if sum(1 for m in b["messages"] if m["role"] == "assistant") != 8: continue
            if key in norm(first_user_text(b)): cands.append(b)
        if not cands: print(f"  {iid}: no recorded body at depth 8"); continue
        b = cands[0]; checked += 1
        same_n = len(b["messages"]) == len(body["messages"])
        diffs = []
        for i, (x, y) in enumerate(zip(b["messages"], body["messages"])):
            tx = [(z.get("type"), norm(z.get("text", z.get("thinking", "")) if z.get("type") != "tool_result" else norm(json.dumps(z.get("content"))))[:60]) for z in x["content"]] if isinstance(x["content"], list) else [("str", norm(x["content"])[:60])]
            ty = [(z.get("type"), norm(z.get("text", z.get("thinking", "")) if z.get("type") != "tool_result" else norm(json.dumps(z.get("content"))))[:60]) for z in y["content"]] if isinstance(y["content"], list) else [("str", norm(y["content"])[:60])]
            if tx != ty: diffs.append((i, x["role"], [t for t, _ in tx], [t for t, _ in ty], tx[:2], ty[:2]))
        print(f"  {iid}: recorded {len(b['messages'])} msgs vs recon {len(body['messages'])}; msg-level diffs {len(diffs)}")
        for df in diffs[:3]: print("     ", df)
    print("checked", checked)
