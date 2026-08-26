#!/usr/bin/env python3
"""Top-1 flip gate: compare two --flip-dump captures of the SAME token stream.

Why this and not perplexity. A mean NLL over wikitext is a mean, and agentic
serving does not fail on the mean -- it fails when one token inside a tool
call comes out different and the wrong command executes. thr3e's level1techs
posts (2026-08) make the point with full-logit captures: divergence between
runtimes clusters at tool calls, and a corpus average hides it. This measures
what those posts measure -- top-1 agreement against a reference, plus the KL
divergence of the distributions -- on our own transcripts, split by region.

Both runs must teacher-force the same stream, so a disagreement is the
runtime's, not the sampler's: same tokens in, only the top-1 choice compared.

    build/q27 MODEL --nll stream.i32 --flip-dump ref.bin --flip-positions p.i32
    build/q27 MODEL --nll stream.i32 --flip-dump cand.bin --flip-positions p.i32
    tools/flip_gate.py ref.bin cand.bin --regions regions.tsv

KLD is computed over the union of the two top-K sets with the captured
log-sum-exp as the normalizer, so it is a TRUNCATED KLD: it accounts for the
mass both runs agree is negligible only through that normalizer. At K=16 on
these distributions the missing tail is small, but the number is a lower
bound, and it is reported as one.
"""
import struct
import sys

MAGIC = b"Q27FLIP1"


def read_dump(path):
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:8] != MAGIC:
        sys.exit(f"{path}: not a flip dump (bad magic)")
    k, n, vocab, N = struct.unpack_from("<iiii", blob, 8)
    rec = 16 + 8 * k
    off = 24
    out = {}
    while off + rec <= len(blob):
        pos, tgt, nll, lse = struct.unpack_from("<iiff", blob, off)
        top = struct.unpack_from("<" + "if" * k, blob, off + 16)
        ids = top[0::2]
        lg = top[1::2]
        out[pos] = (tgt, nll, lse, ids, lg)
        off += rec
    return {"k": k, "n": n, "vocab": vocab, "N": N, "rows": out}


def kld(a, b):
    """KL(a || b) over the union of the two top-K sets, in nats."""
    import math
    _, _, lse_a, ids_a, lg_a = a
    _, _, lse_b, ids_b, lg_b = b
    pa = dict(zip(ids_a, lg_a))
    pb = dict(zip(ids_b, lg_b))
    total = 0.0
    for t in set(ids_a) | set(ids_b):
        # a token outside a run's top-K is below that run's K-th logit; use
        # the K-th as the (generous) bound so the term cannot blow up
        la = pa.get(t, min(lg_a))
        lb = pb.get(t, min(lg_b))
        p = math.exp(la - lse_a)
        total += p * ((la - lse_a) - (lb - lse_b))
    return max(total, 0.0)


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    regions = None
    if "--regions" in sys.argv:
        regions = sys.argv[sys.argv.index("--regions") + 1]
    if len(args) < 2:
        sys.exit(__doc__)
    ref, cand = read_dump(args[0]), read_dump(args[1])
    for key in ("k", "vocab", "N"):
        if ref[key] != cand[key]:
            sys.exit(f"capture mismatch on {key}: {ref[key]} vs {cand[key]}")
    cls = {}
    if regions:
        with open(regions) as f:
            next(f)
            for line in f:
                p, c = line.split("\t")[:2]
                cls[int(p)] = c
    shared = sorted(set(ref["rows"]) & set(cand["rows"]))
    if not shared:
        sys.exit("no positions in common")
    if len(shared) != len(ref["rows"]) or len(shared) != len(cand["rows"]):
        print(f"note: comparing {len(shared)} shared positions "
              f"(ref {len(ref['rows'])}, cand {len(cand['rows'])})")
    buckets = {}
    for pos in shared:
        a, b = ref["rows"][pos], cand["rows"][pos]
        if a[0] != b[0]:
            sys.exit(f"position {pos}: different target token -- not the same stream")
        flip = a[3][0] != b[3][0]
        d = kld(a, b)
        for name in ("all", cls.get(pos, "?")):
            st = buckets.setdefault(name, {"n": 0, "flip": 0, "kld": [], "dnll": 0.0})
            st["n"] += 1
            st["flip"] += flip
            st["kld"].append(d)
            st["dnll"] += b[1] - a[1]
    order = ["all", "out", "tool", "think", "prompt", "?"]
    print(f"{args[0]}  ->  {args[1]}")
    print(f"{'region':>8}{'n':>8}{'flips':>7}{'rate':>9}"
          f"{'KLD p50':>11}{'KLD p95':>11}{'KLD p99':>11}{'dNLL':>10}")
    print("-" * 75)
    for name in order:
        if name not in buckets:
            continue
        st = buckets[name]
        rate = st["flip"] / st["n"] if st["n"] else float("nan")
        print(f"{name:>8}{st['n']:>8}{st['flip']:>7}{rate:>8.3%}"
              f"{pct(st['kld'], .50):>11.6f}{pct(st['kld'], .95):>11.6f}"
              f"{pct(st['kld'], .99):>11.6f}{st['dnll'] / st['n']:>+10.5f}")
    if "tool" in buckets and "out" in buckets:
        t = buckets["tool"]["flip"] / max(1, buckets["tool"]["n"])
        o = buckets["out"]["flip"] / max(1, buckets["out"]["n"])
        if o > 0:
            print(f"\ntool-call positions flip {t / o:.2f}x the rate of ordinary output")
    print("\n(KLD is truncated to the captured top-K: a lower bound.)")


if __name__ == "__main__":
    main()
