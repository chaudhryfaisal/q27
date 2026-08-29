#!/usr/bin/env python3
"""Engine-agnostic long-context decode bench (client-side, same instrument for
every OpenAI-compatible engine). Two arms:
  A) decode vs context: cold unique prefix at increasing lengths, fixed gen.
  B) incremental tool loop: a growing multi-turn conversation, warm each turn.
Matched serving sampler, thinking ON. Measures TTFT and decode t/s from the
token stream + usage; no engine-specific logs, so q27/llama/vLLM/ninfer are
compared on identical footing."""
import requests, time, json, os, statistics, sys, argparse

SAMPLER=dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.05)

def filler(approx_tokens, nonce):
    unit=("def h_{n}(recs,cfg):  # {x}\n    o=[]\n    for i,r in enumerate(recs):\n"
          "        if not ok(r,cfg.s_{n}): continue\n        o.append(tx(r,{n}.0,i))\n    return o\n"
          "[2026-08-27T0{m}:{nn}:12Z] INFO w-{n} did {n}00 recs in {n}.4s {x}\n\n")
    s=[];n=0;c=0
    while c<approx_tokens*3:
        s.append(unit.format(n=(n%97)+1,nn=(n%59)+1,m=(n%9),x=nonce)); c+=len(s[-1]); n+=1
    return "".join(s)

GEN=("Ignore the material above. Output a numbered list, one item per line, of 50 "
     "distinct two-word English noun phrases about weather. Nothing else.")

def call(base, model, messages, max_tokens, extra_sampler=None):
    body=dict(model=model, messages=messages, max_tokens=max_tokens, stream=True,
              stream_options={"include_usage":True}, **SAMPLER)
    if extra_sampler: body.update(extra_sampler)
    t0=time.time(); t_first=None; n_delta=0; prompt=None; comp=None; err=None
    try:
        with requests.post(base+"/v1/chat/completions", json=body, stream=True, timeout=900) as r:
            if r.status_code!=200:
                return dict(error=f"HTTP {r.status_code}: {r.text[:200]}")
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "): continue
                d=line[6:]
                if d==b"[DONE]": break
                try: j=json.loads(d)
                except: continue
                chs=j.get("choices") or []
                delta=(chs[0].get("delta") if chs else {}) or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    if t_first is None: t_first=time.time()
                    n_delta+=1
                if j.get("usage"):
                    prompt=j["usage"].get("prompt_tokens"); comp=j["usage"].get("completion_tokens")
    except Exception as e:
        return dict(error=str(e)[:200])
    t_last=time.time()
    if t_first is None: t_first=t_last
    gen=comp if comp else n_delta
    ttft=t_first-t0; dect=max(t_last-t_first,1e-6)
    return dict(prompt=prompt, gen=gen, ttft=ttft, dec_s=dect,
                dtps=(gen-1)/dect if gen and gen>1 else 0.0, usage_tokens=comp is not None)

def bench_A(base, model, label, trials=3):
    out=[]
    print(f"[{label}] BENCH A: decode vs context (cold, unique)")
    print(f"{'ctx':>8} {'tps':>7} {'ttft_s':>8} {'gen':>5} {'trials':>6}")
    for tgt in [0,2000,4000,8000,16000,32000]:
        rows=[]
        for k in range(trials):
            nonce=os.urandom(5).hex()
            msgs=[]
            if tgt:
                msgs=[{"role":"user","content":"logs (ref "+nonce+"):\n\n"+filler(tgt,nonce)},
                      {"role":"assistant","content":"read."}]
            msgs.append({"role":"user","content":GEN+" ("+nonce+")"})
            r=call(base,model,msgs,512)
            if r.get("error"): print("  ERR",tgt,r["error"]); continue
            rows.append(r)
        if not rows: continue
        med=lambda k: statistics.median(x[k] for x in rows)
        ctx=statistics.median(x["prompt"] for x in rows if x["prompt"]) if any(x["prompt"] for x in rows) else tgt
        rec=dict(arm="A",label=label,ctx=int(ctx),tps=round(med("dtps"),1),
                 ttft=round(med("ttft"),3),gen=int(med("gen")),n=len(rows))
        out.append(rec); print(f"{rec['ctx']:>8} {rec['tps']:>7} {rec['ttft']:>8} {rec['gen']:>5} {rec['n']:>6}")
    return out

def bench_B(base, model, label, turns=12):
    out=[]
    print(f"[{label}] BENCH B: incremental tool loop (warm)")
    print(f"{'turn':>4} {'cum_ctx':>8} {'ttft_s':>8} {'tps':>7}")
    conv=[{"role":"system","content":"You are a coding agent. Use tools, then summarize."},
          {"role":"user","content":"Review this repo. I will feed you files; after each, list 3 findings."}]
    for t in range(1,turns+1):
        nonce=os.urandom(4).hex()
        conv.append({"role":"user","content":f"[file src/module_{t}.py]\n\n"+filler(1500,nonce)})
        r=call(base,model,conv,300)
        if r.get("error"): print("  ERR",t,r["error"]); break
        conv.append({"role":"assistant","content":(r.get("out") or "noted.")[:500]})
        rec=dict(arm="B",label=label,turn=t,ctx=r["prompt"],ttft=round(r["ttft"],3),tps=round(r["dtps"],1))
        out.append(rec); print(f"{t:>4} {rec['ctx'] or 0:>8} {rec['ttft']:>8} {rec['tps']:>7}")
    return out

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True); ap.add_argument("--model",required=True)
    ap.add_argument("--label",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--arms",default="AB")
    a=ap.parse_args()
    recs=[]
    if "A" in a.arms: recs+=bench_A(a.base,a.model,a.label)
    if "B" in a.arms: recs+=bench_B(a.base,a.model,a.label)
    with open(a.out,"w") as f:
        for r in recs: f.write(json.dumps(r)+"\n")
    print(f"[{a.label}] wrote {len(recs)} records -> {a.out}")
