#!/usr/bin/env python3
"""Decide whether a new Qwen checkpoint is a repack job or an engine job.

Reads the architecture constants OUT OF THE SOURCE -- src/engine.cuh and
src/metal/metal_engine.h -- so this tool cannot drift from the engines the way
a restated table in a doc can. It also cross-checks the two tables against each
other, which is the check docs/PORTING.md calls for and nothing else enforces:
they are separate declarations, each validated against the artifact
independently, so a port that updates one and forgets the other produces a
working backend and a backend that refuses to load rather than a compile error.

  tools/check_checkpoint.py Qwen/Qwen3.8-27B          # Hub repo id
  tools/check_checkpoint.py path/to/config.json       # local file
  tools/check_checkpoint.py --tables-only             # just CUDA vs Metal

Exit 0 means every constant matches and the shape constraints hold, i.e. a
repack job. Exit 1 means an engine change is needed; the rows say which.
Exit 2 means the config could not be read.

This answers only the question the constants can answer. A checkpoint that
passes can still behave differently; see docs/PORTING.md.
"""
import json
import os
import re
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUDA_H = os.path.join(ROOT, "src", "engine.cuh")
METAL_H = os.path.join(ROOT, "src", "metal", "metal_engine.h")

# constant -> how to pull the expected value out of a text_config subtree.
# Mirrors the table in docs/PORTING.md; keep the two in step.
FIELDS = {
    "N_LAYER":   ("num_hidden_layers",   lambda c: c.get("num_hidden_layers")),
    "N_EMBD":    ("hidden_size",         lambda c: c.get("hidden_size")),
    "N_FFN":     ("intermediate_size",   lambda c: c.get("intermediate_size")),
    "N_HEAD":    ("num_attention_heads", lambda c: c.get("num_attention_heads")),
    "N_KV":      ("num_key_value_heads", lambda c: c.get("num_key_value_heads")),
    "HEAD_DIM":  ("head_dim",            lambda c: c.get("head_dim")),
    "N_ROT":     ("head_dim x partial_rotary_factor",
                  lambda c: (int(c["head_dim"] * _prf(c))
                             if c.get("head_dim") is not None else None)),
    "FREQ_BASE": ("rope_theta",          lambda c: _rope(c, "rope_theta")),
    "EPS":       ("rms_norm_eps",        lambda c: c.get("rms_norm_eps")),
    "GDN_HEADS": ("linear_num_value_heads", lambda c: c.get("linear_num_value_heads")),
    "GDN_DIM":   ("linear_value_head_dim",  lambda c: c.get("linear_value_head_dim")),
    "GDN_V":     ("linear_num_value_heads x linear_value_head_dim",
                  lambda c: _mul(c.get("linear_num_value_heads"),
                                 c.get("linear_value_head_dim"))),
    "GDN_CH":    ("(linear_num_key_heads x 2 + linear_num_value_heads) x dim",
                  lambda c: _gdn_ch(c)),
    "VOCAB":     ("vocab_size",          lambda c: c.get("vocab_size")),
}


def _prf(c):
    rp = c.get("rope_parameters") or {}
    return c.get("partial_rotary_factor", rp.get("partial_rotary_factor", 1.0))


def _rope(c, key):
    rp = c.get("rope_parameters") or {}
    return c.get(key, rp.get(key))


def _mul(a, b):
    return None if a is None or b is None else a * b


def _gdn_ch(c):
    k, v, d = (c.get("linear_num_key_heads"), c.get("linear_num_value_heads"),
               c.get("linear_value_head_dim"))
    return None if None in (k, v, d) else (2 * k + v) * d


def parse_constants(path, pattern):
    """Pull `name = value` pairs out of the constexpr declarations."""
    text = open(path).read()
    out = {}
    for decl in re.finditer(pattern, text):
        for name, value in re.findall(r"(\w+)\s*=\s*([0-9.eE+-]+f?)", decl.group(0)):
            v = value.rstrip("f")
            out[name] = float(v) if any(ch in v for ch in ".eE") else int(v)
    return out


def load_config(source):
    if os.path.exists(source):
        return json.load(open(source)), source
    url = f"https://huggingface.co/{source}/raw/main/config.json"
    try:
        return json.load(urllib.request.urlopen(url, timeout=30)), url
    except urllib.error.HTTPError as e:
        print(f"could not read {url}: HTTP {e.code}", file=sys.stderr)
        # The Hub answers 401 rather than 404 for a repo that does not exist,
        # so it does not leak whether a private one does. An unreleased model
        # therefore looks like an auth failure, which is the common case here.
        if e.code in (401, 403, 404):
            print("  (not released yet, does not exist, or is gated)", file=sys.stderr)
        sys.exit(2)
    except Exception as e:  # network, DNS, timeout
        print(f"could not read {url}: {e}", file=sys.stderr)
        sys.exit(2)


def close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
    return a == b


def check_tables(cuda, metal):
    print("== constant tables: src/engine.cuh vs src/metal/metal_engine.h ==")
    bad = 0
    for name in FIELDS:
        a, b = cuda.get(name), metal.get(name)
        if b is None:
            print(f"  {name:<10} {a!s:>10}   (Metal does not declare it)")
            continue
        ok = close(a, b)
        bad += not ok
        print(f"  {name:<10} {a!s:>10} {b!s:>10}   {'ok' if ok else 'MISMATCH'}")
    print(f"  -> {'tables agree' if not bad else f'{bad} MISMATCHED -- fix before porting'}\n")
    return bad


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    tables_only = "--tables-only" in sys.argv[1:]
    if not args and not tables_only:
        print(__doc__)
        return 2

    cuda = parse_constants(CUDA_H, r"static constexpr (?:int|float) [^;]+;")
    metal = parse_constants(METAL_H, r"static constexpr (?:uint32_t|float) [^;]+;")
    bad = check_tables(cuda, metal)
    if tables_only:
        return 1 if bad else 0

    raw, where = load_config(args[0])
    cfg = raw.get("text_config", raw)
    print(f"== {args[0]} ==")
    print(f"   {where}")
    arch = ", ".join(raw.get("architectures") or ["?"])
    print(f"   architectures: {arch}   model_type: {raw.get('model_type','?')}\n")

    print(f"   {'constant':<10} {'engine':>10} {'checkpoint':>12}   field")
    fails = []
    for name, (field, get) in FIELDS.items():
        want = cuda.get(name)
        try:
            got = get(cfg)
        except Exception:
            got = None
        if got is None:
            fails.append((name, "field absent from config"))
            print(f"   {name:<10} {want!s:>10} {'--':>12}   {field}  MISSING")
            continue
        ok = close(want, got)
        if not ok:
            fails.append((name, f"config says {got}, engine is {want}"))
        print(f"   {name:<10} {want!s:>10} {got!s:>12}   {field}"
              f"{'' if ok else '  <-- MISMATCH'}")

    print("\n== constraints that are not single constants ==")
    notes = []
    for label, value in (("hidden_size", cfg.get("hidden_size")),
                         ("intermediate_size", cfg.get("intermediate_size"))):
        if value is None:
            continue
        ok = value % 256 == 0
        notes.append(ok)
        print(f"   {label} {value} divisible by VG_KB 256: "
              f"{'yes' if ok else 'NO -- vgemm.cu aborts'}")
    hd = cfg.get("head_dim")
    if hd is not None:
        ok = hd % 128 == 0
        notes.append(ok)
        print(f"   head_dim {hd} divisible by the 128-element WHT group: "
              f"{'yes' if ok else 'NO -- breaks turbo3 and turbo5'}")
    moe = any(k in cfg for k in ("num_experts", "num_local_experts",
                                 "n_routed_experts", "moe_intermediate_size"))
    print(f"   mixture of experts: {'YES -- a different engine, not a port' if moe else 'no'}")
    mtp = cfg.get("mtp_num_hidden_layers")
    print(f"   MTP layers: {mtp if mtp is not None else 'absent -- decodes, but loses the draft ladder'}")

    print()
    if bad:
        print("VERDICT: fix the CUDA/Metal table mismatch first.")
        return 1
    if moe or fails or not all(notes):
        print("VERDICT: ENGINE CHANGE NEEDED")
        for name, why in fails:
            print(f"  - {name}: {why}")
        if moe:
            print("  - mixture of experts: no expert routing exists in q27")
        print("  See docs/PORTING.md for what each one costs.")
        return 1
    print("VERDICT: REPACK JOB -- every constant matches and the shape "
          "constraints hold.\n         Re-derive the canonical md5 after "
          "repacking; it is checkpoint-specific.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
