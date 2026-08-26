#!/usr/bin/env bash
# Regenerate tools/golden/qwen38_tools_request.prompt from a RUNNING llama-server
# (CPU only, -ngl 0) via /apply-template: this is what llama.cpp feeds the model,
# not a jinja2 approximation. Run from the repo root with a free ~20 GB of RAM.
# CPU-only llama-server (-ngl 0) purely to call /apply-template: minja's exact
# rendering of a tools request, the ground truth for what llama.cpp feeds the
# model. No VRAM, so it runs alongside the fp16 arm.
set -uo pipefail
S="${1:?usage: capture_template_golden.sh <outdir>}"
LC=/mnt/ai/projects/llama.cpp/build/bin/llama-server
GG=/mnt/ai/models/qwen38-27b-mtp-gguf/Qwen3.8-27B-MTP-Q5_K_M.gguf
TPL=/mnt/ai/projects/q27/bench/crossengine/harness/qwen38_sysinline.jinja
$LC -m $GG --port 8193 --host 127.0.0.1 -ngl 0 --jinja --chat-template-file $TPL --reasoning off -c 4096 --no-warmup >$S/minja_srv.log 2>&1 &
P=$!
for i in $(seq 1 120); do [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8193/health)" = 200 ] && break; sleep 5; done
python3 - <<'PY'
import json,urllib.request
S="${1:?usage: capture_template_golden.sh <outdir>}"
req=json.load(open(f"{S}/req.json"))   # the canonical Anthropic-shaped request from the earlier diff
msgs=[{"role":"system","content":req["system"]}]
for m in req["messages"]:
    c=m["content"]
    if isinstance(c,str): msgs.append({"role":m["role"],"content":c}); continue
    tcs=[];text="";think=""
    for b in c:
        if b["type"]=="text": text+=b["text"]
        elif b["type"]=="thinking": think+=b["thinking"]
        elif b["type"]=="tool_use": tcs.append({"type":"function","id":b["id"],"function":{"name":b["name"],"arguments":json.dumps(b["input"])}})
        elif b["type"]=="tool_result": msgs.append({"role":"tool","content":b["content"],"tool_call_id":b["tool_use_id"]})
    if m["role"]=="assistant":
        e={"role":"assistant","content":text}
        if think: e["reasoning_content"]=think
        if tcs: e["tool_calls"]=tcs
        msgs.append(e)
tools=[{"type":"function","function":{"name":t["name"],"description":t["description"],"parameters":t["input_schema"]}} for t in req["tools"]]
body={"messages":msgs,"tools":tools,"chat_template_kwargs":{"enable_thinking":True}}
r=urllib.request.Request("http://127.0.0.1:8193/apply-template",data=json.dumps(body).encode(),headers={"content-type":"application/json"})
out=json.loads(urllib.request.urlopen(r,timeout=120).read())
p=out.get("prompt") if isinstance(out,dict) else str(out)
open(f"{S}/prompt_minja.txt","w").write(p)
print("minja golden:",len(p),"chars; keys:",list(out)[:4] if isinstance(out,dict) else type(out))
PY
kill $P; wait $P 2>/dev/null
echo "golden written to $S/qwen38_tools_request.prompt -- copy into tools/golden/ if the template or fixture changed"
