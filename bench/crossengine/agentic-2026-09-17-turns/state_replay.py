#!/usr/bin/env python3
"""Replay recorded /v1/messages bodies with seeds 1..N on any engine and record
the FIRST tool call's name + primary argument (command / file_path / pattern),
thinking chars, output tokens. Optionally strip is_error flags (--strip-error)
so an engine that marks errors itself (ninfer) renders them plain.
usage: state_replay.py <base> <label> <dir> <name...> [--n N] [--strip-error]"""
import json, os, sys, time, urllib.request
args = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 4
strip = "--strip-error" in sys.argv
base, label, d = args[:3]; names = args[3:]
if "--n" in sys.argv: names = [x for x in names if x != str(N)]
out = open(os.path.join(d, f"{label}.jsonl"), "w")
content_out = open(os.path.join(d, f"{label}.content.jsonl"), "w")  # full blocks + usage; session content, local only
for name in names:
    body = json.load(open(os.path.join(d, name + ".json")))
    body["stream"] = False; body["max_tokens"] = 8192
    body.setdefault("temperature", 1.0); body.setdefault("top_p", 0.95); body.setdefault("top_k", 20)
    # SR_TEMP / SR_TOPK / SR_TOPP / SR_MAXTOK override the sampler (greedy arm: SR_TEMP=0 SR_TOPK=1)
    if os.environ.get("SR_TEMP"): body["temperature"] = float(os.environ["SR_TEMP"])
    if os.environ.get("SR_TOPK"): body["top_k"] = int(os.environ["SR_TOPK"])
    if os.environ.get("SR_TOPP"): body["top_p"] = float(os.environ["SR_TOPP"])
    if os.environ.get("SR_MAXTOK"): body["max_tokens"] = int(os.environ["SR_MAXTOK"])
    if strip:
        for m in body["messages"]:
            if m["role"] == "user" and not isinstance(m["content"], str):
                for x in m["content"]:
                    if x.get("type") == "tool_result": x.pop("is_error", None)
    for seed in range(1, N + 1):
        b = dict(body); b["seed"] = seed
        req = urllib.request.Request(base.rstrip("/") + "/v1/messages", data=json.dumps(b).encode(),
                                     headers={"content-type": "application/json", "x-api-key": "local", "anthropic-version": "2023-06-01"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1800) as r: resp = json.load(r)
        except urllib.error.HTTPError as e:
            print(f"{name} seed {seed}: HTTP {e.code} {e.read()[:200]}", flush=True); continue
        c = resp.get("content", [])
        calls = [x for x in c if x.get("type") == "tool_use"]
        first = calls[0] if calls else None
        arg = ""
        if first:
            i = first.get("input", {}); arg = str(i.get("command") or i.get("file_path") or i.get("pattern") or i.get("url") or "")[:160]
        row = dict(state=name, seed=seed, tool=first["name"] if first else None, arg=arg, ncalls=len(calls),
                   think=sum(len(x.get("thinking", "")) for x in c if x.get("type") == "thinking"),
                   out=resp.get("usage", {}).get("output_tokens"), stop=resp.get("stop_reason"), secs=round(time.time() - t0, 1))
        out.write(json.dumps(row) + "\n"); out.flush()
        content_out.write(json.dumps({"state": name, "seed": seed, "content": c, "usage": resp.get("usage", {}), "stop": resp.get("stop_reason")}) + "\n"); content_out.flush()
    print(f"  {name}: done", flush=True)
print(f"== {label}: done")
