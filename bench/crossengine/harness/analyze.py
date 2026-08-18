#!/usr/bin/env python3
"""Build the head-to-head report from whatever arms have finished.

Safe to run mid-run: every arm is optional and missing legs are reported as
missing rather than silently dropped.

  usage: analyze.py <outdir>
"""
import json
import os
import re
import statistics as st
import sys

LEGS = ["q4s", "nint", "q5f", "nvfp4", "llama", "vllm"]
ENGINE = {"q4s": "q27", "q5f": "q27", "nint": "ninfer", "nvfp4": "ninfer",
          "llama": "llama", "vllm": "vllm"}
DESC = {
    "q4s":   "q27 q4s        15.46 GB  canonical anchor",
    "nint":  "ninfer int     17.50 GB  format control",
    "q5f":   "q27 q5f        18.22 GB  bpw-matched to nvfp4",
    "nvfp4": "ninfer nvfp4   18.32 GB  their headline",
    "llama": "llama.cpp      18.19 GB  Q5_K_M, master 01818e49, q8_0 KV",
    "vllm":  "vLLM NVFP4     ~23 GB    unsloth fixed-MTP, nightly 0.27.2rc1",
}
# A cold prefill on this box runs a few thousand tok/s; a prefix-cache hit
# reports an order of magnitude more because TTFT collapses while in_tok does
# not. 15k separates the two populations with a wide margin on both sides.
CACHE_HIT_TPS = 15000


def load_jsonl(p):
    if not os.path.exists(p):
        return []
    out = []
    for ln in open(p, errors="replace"):
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


def h(t):
    print(f"\n{'='*78}\n{t}\n{'='*78}")


def agentic(out):
    h("AGENTIC -- 12 SWE-bench_Verified instances driven by Claude Code")
    print(f"{'leg':<7} {'inst':>5} {'nonempty':>9} {'gold':>6} {'quit@1':>7} "
          f"{'wall/inst':>10} {'decode':>9} {'reqs':>6}")
    rows = {}
    for leg in LEGS:
        r = load_jsonl(f"{out}/agentic.{leg}.jsonl")
        if not r:
            print(f"{leg:<7} {'-- not run --':>40}")
            continue
        n = len(r)
        dtok = sum(x["tap_dec_tok"] for x in r)
        ds = sum(x["tap_dec_s"] for x in r)
        quit1 = sum(1 for x in r if x["turns"] <= 1)
        rows[leg] = {
            "n": n, "nonempty": sum(x["nonempty"] for x in r),
            "gold": sum(x["gold_hit"] for x in r), "quit1": quit1,
            "wall": sum(x["wall_s"] for x in r) / n,
            "decode": dtok / ds if ds else 0,
            "reqs": sum(x["n_reqs"] for x in r),
        }
        v = rows[leg]
        print(f"{leg:<7} {v['n']:>5} {v['nonempty']:>9} {v['gold']:>6} "
              f"{v['quit1']:>7} {v['wall']:>9.0f}s {v['decode']:>8.1f} {v['reqs']:>6}")
    print("\n  decode = tokens / summed decode windows, client-observed via the tap")
    print("  quit@1 = agent stopped after one turn (see TOOL-NAMING note below)")
    return rows


def reuse_stats(out, leg):
    """Token-weighted prefix reuse, from whichever instrument the engine offers.

    The four engines do not report this the same way and the denominators are
    NOT interchangeable:
      q27     server log `[gen] prompt=N prefix_hit=M`; prompt is the FULL
              prompt and the hit is counted inside it.
      ninfer  reqlog result.prefix_cache_hit_tokens / result.prompt_tokens.
      llama   Anthropic usage cache_read_input_tokens, where input_tokens counts
              ONLY the new tokens -- so the full prompt is in_tok + cache_read.
      vllm    same field if it emits one; absent otherwise.
    Returns (hit_tokens, total_prompt_tokens) or None.
    """
    eng = ENGINE[leg]
    if eng == "q27":
        p = f"{out}/server.{leg}.agentic.log"
        if not os.path.exists(p):
            return None
        H = P = 0
        for m in re.finditer(r"\[gen\] prompt=(\d+) prefix_hit=(\d+)",
                             open(p, errors="replace").read()):
            P += int(m.group(1)); H += int(m.group(2))
        return (H, P) if P else None
    if eng == "ninfer":
        p = f"{out}/server.{leg}.agentic.reqlog.jsonl"
        if not os.path.exists(p):
            return None
        H = P = 0
        for r in load_jsonl(p):
            if r.get("event") != "request_done":
                continue
            res = r.get("result") or {}
            H += res.get("prefix_cache_hit_tokens", 0)
            P += res.get("prompt_tokens", 0)
        return (H, P) if P else None
    # llama / vllm: client-observed Anthropic usage through the tap
    recs = [r for r in load_jsonl(f"{out}/tap.{leg}.jsonl")
            if r.get("status") == 200 and r.get("cache_read_tok") is not None]
    if not recs:
        return None
    H = sum(r["cache_read_tok"] for r in recs)
    P = sum(r["in_tok"] + r["cache_read_tok"] for r in recs)
    return (H, P) if P else None


def prefill_and_cache(out):
    h("PREFILL + PREFIX-CACHE MISS RATE (real agentic traffic)")
    print(f"{'leg':<7} {'reqs':>6} {'reuse tok-wtd':>14} {'effective tok/s':>16} "
          f"{'prompt tok':>12}")
    for leg in LEGS:
        recs = [r for r in load_jsonl(f"{out}/tap.{leg}.jsonl")
                if r.get("status") == 200 and r.get("in_tok")]
        if not recs:
            print(f"{leg:<7} {'-- not run --':>40}")
            continue
        rs = reuse_stats(out, leg)
        # effective prefill = every prompt token the client sent / all TTFT.
        # For llama/vllm the client-visible prompt is in_tok + cache_read.
        tot = sum(r["in_tok"] + (r.get("cache_read_tok") or 0) for r in recs)
        ttft = sum(r["ttft_ms"] for r in recs) / 1000.0
        rtxt = f"{rs[0]/rs[1]*100:.2f}%" if rs else "n/a"
        print(f"{leg:<7} {len(recs):>6} {rtxt:>14} "
              f"{tot/ttft if ttft else 0:>16.0f} {tot:>12}")
    print("\n  reuse comes from each engine's OWN instrument -- q27 [gen] prefix_hit,")
    print("  ninfer reqlog prefix_cache_hit_tokens, llama/vllm Anthropic")
    print("  cache_read_input_tokens. Denominators differ; see reuse_stats().")
    print("  cache MISS rate = 100% - cache hit. This is the number BUILDLOG names")
    print("  as the gate on the prefill-performance plan's ROI.")


def ladder(out):
    h("CONCURRENCY LADDER -- aggregate decode t/s")
    data = {}
    for leg in LEGS:
        p = f"{out}/ladder.{leg}.txt"
        if not os.path.exists(p):
            continue
        pts = {}
        for ln in open(p, errors="replace"):
            if ln.startswith("JSON "):
                try:
                    d = json.loads(ln[5:])
                    pts[d["c"]] = d
                except Exception:
                    pass
        if pts:
            data[leg] = pts
    if not data:
        print("  -- not run --")
        return
    print(f"{'leg':<7} " + "".join(f"{'C='+str(c):>10}" for c in (1, 2, 4, 8))
          + f"{'peak':>9} {'scaling':>9}")
    for leg in LEGS:
        if leg not in data:
            print(f"{leg:<7} {'-- not run --':>40}")
            continue
        p = data[leg]
        cells = "".join(f"{p[c]['aggregate_tps']:>10.1f}" if c in p else f"{'-':>10}"
                        for c in (1, 2, 4, 8))
        vals = [p[c]["aggregate_tps"] for c in (1, 2, 4, 8) if c in p]
        base = p.get(1, {}).get("aggregate_tps")
        peak = max(vals) if vals else 0
        sc = f"{peak/base:.2f}x" if base else "-"
        print(f"{leg:<7} {cells}{peak:>9.1f} {sc:>9}")
    # the calibration line matters: it ties this client-side convention to the
    # server-side one the 08-14/08-16 ladder history was recorded in
    print()
    for leg in LEGS:
        if leg not in data:
            continue
        for c, d in sorted(data[leg].items()):
            if d.get("server_side_agg"):
                skew = (d["aggregate_tps"] / d["server_side_agg"] - 1) * 100
                print(f"  [calib] {leg} C={c}: client {d['aggregate_tps']:.1f} vs "
                      f"server {d['server_side_agg']:.1f} ({skew:+.1f}%)")


def quality(out):
    h("QUALITY -- benchlocal --medium, 5 packs / 75 scenarios, temp 0, no-think")
    got = {}
    for leg in LEGS:
        p = f"{out}/quality.{leg}.json"
        if os.path.exists(p):
            try:
                got[leg] = json.load(open(p))
            except Exception:
                pass
    if not got:
        print("  -- not run --")
        return
    packs = []
    for d in got.values():
        for pk in d.get("packs", []):
            if pk["pack_id"] not in packs:
                packs.append(pk["pack_id"])
    print(f"{'pack':<24}" + "".join(f"{l:>12}" for l in LEGS if l in got))
    for pid in packs:
        row = f"{pid:<24}"
        for leg in LEGS:
            if leg not in got:
                continue
            pk = next((x for x in got[leg].get("packs", []) if x["pack_id"] == pid), None)
            row += f"{(str(pk['passed'])+'/'+str(pk['total'])) if pk else '-':>12}"
        print(row)
    row = f"{'TOTAL':<24}"
    for leg in LEGS:
        if leg not in got:
            continue
        t = got[leg].get("totals") or {}
        row += f"{str(t.get('passed'))+'/'+str(t.get('total')):>12}"
    print(row)
    print("\n  q5f vs nvfp4 is the bpw-matched pair -- the only comparison here")
    print("  where a quality delta is attributable to engine/kernel rather than bits.")


def tool_naming(out):
    """Quantify the confound: agent stopped at turn 1 with a tool call in text."""
    h("TOOL-NAMING CONFOUND")
    work = os.environ.get("SWEBENCH_WORK", "/mnt/ai/swebench-work")
    any_found = False
    for leg in LEGS:
        rows = load_jsonl(f"{out}/agentic.{leg}.jsonl")
        if not rows:
            continue
        any_found = True
        names = {}
        for r in rows:
            if r["turns"] > 1:
                continue
            p = f"{work}/{leg}/{r['iid']}/logs/out.jsonl"
            if not os.path.exists(p):
                continue
            for ln in open(p, errors="replace"):
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                if d.get("type") != "assistant":
                    continue
                for c in d["message"].get("content", []):
                    if c.get("type") == "text":
                        for m in re.finditer(r'"name"\s*:\s*"([A-Za-z_]+)"', c["text"]):
                            names[m.group(1)] = names.get(m.group(1), 0) + 1
        q = sum(1 for r in rows if r["turns"] <= 1)
        print(f"  {leg:<7} quit@1: {q}/{len(rows)}   "
              f"unparsed tool names in text: {dict(sorted(names.items(), key=lambda x:-x[1]))}")
    if not any_found:
        print("  -- not run --")
    print("\n  These are tool calls the model emitted as TEXT that no registry")
    print("  recognized, so the agent saw prose and stopped. It is a property of")
    print("  the vanilla 3.6 checkpoint + this Claude Code build, not of either")
    print("  engine -- but it caps task success on BOTH sides, so gold-hit rates")
    print("  here are not comparable to the 2026-07-15 baseline (11/12).")


def baseline_diff(out):
    """Per-instance diff against the 2026-07-14 three-engine run.

    That baseline is NOT config-matched to this one and must not be read as a
    regression test: BUILDLOG 2026-07-14 records it as greedy, q8 KV, and a
    ~5.25 bpw q27 tier, against q4s at 4.6 bpw and fp8 KV sampled here. The
    q5f leg is the one that lands near the baseline's bitrate, so a q4s-only
    gap that q5f closes is a QUANT effect, not an engine one.
    """
    h("vs 2026-07-14 BASELINE (not config-matched -- see note)")
    bp = "/mnt/ai/projects/q27/bench/swebench/results.q27.jsonl"
    if not os.path.exists(bp):
        print("  baseline file missing")
        return
    base = {r["iid"]: r for r in load_jsonl(bp)}
    for leg in LEGS:
        if ENGINE[leg] != "q27":
            continue
        rows = load_jsonl(f"{out}/agentic.{leg}.jsonl")
        if not rows:
            continue
        print(f"\n  {leg}:")
        print(f"    {'instance':<26} {'base turns/gold':>16} {'now turns/gold':>16}  delta")
        for r in rows:
            b = base.get(r["iid"])
            if not b:
                continue
            bs = f"{b['turns']}/{'Y' if b['gold_hit'] else 'N'}"
            ns = f"{r['turns']}/{'Y' if r['gold_hit'] else 'N'}"
            d = ""
            if b["gold_hit"] and not r["gold_hit"]:
                d = "  LOST"
            elif r["gold_hit"] and not b["gold_hit"]:
                d = "  GAINED"
            print(f"    {r['iid']:<26} {bs:>16} {ns:>16}{d}")
    print("\n  baseline headline: 202.7 t/s agg, 47 s/inst, 11/12 gold")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"HEAD-TO-HEAD  ninfer vs q27   ({out})")
    mp = f"{out}/manifest.txt"
    if os.path.exists(mp):
        print(open(mp).read().strip())
    print("\nlegs:")
    for l in LEGS:
        print(f"  {l:<7} {DESC[l]}")
    agentic(out)
    prefill_and_cache(out)
    ladder(out)
    quality(out)
    tool_naming(out)
    baseline_diff(out)


if __name__ == "__main__":
    main()
