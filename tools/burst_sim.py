#!/usr/bin/env python3
"""maxd6 GO-IF rerun: exact offline round simulation from --burst-stats CSVs.

Every burst CSV row holds the 10-deep MTP draft chain + margins at one free
position of the greedy trajectory. Spec decode is bitwise-identical to the
serial path, so a gated round starting at position q draws EXACTLY these
drafts/margins -- walking the sequence round-by-round (jump n per round)
reproduces served behavior at ANY ceiling 4..10 and any theta, killing the
per-position-vs-round discount that inflated the old brief's raw chain stats.

Validation: MD=4/5 tok/round must reproduce the measured accept_ab legs.
Economics: fit ms/round = base + A*drafts_launched + B*verify_width by least
squares on the 10 measured (payload x d4/d5) legs, then project d6/d7.
Width-7+ lane cost is an EXTRAPOLATION (no width-7 graph exists) -- flagged.
"""
import csv
import sys

THETA = 0.5
SD = 10

# measured accept_ab legs (scratchpad/accept_ab_run3.log + accept_ab_61k.log):
# payload -> {md: (tps_med, ms_per_round)}
MEASURED = {
    "echo":    {4: (151.6, 21.37), 5: (155.7, 22.52)},
    "docs":    {4: (168.1, 21.15), 5: (177.5, 22.20)},
    "codegen": {4: (163.9, 21.40), 5: (164.2, 22.27)},
    "testgen": {4: (162.0, 21.64), 5: (168.3, 23.05)},
    "docs61k": {4: (114.4, 23.56), 5: (112.5, 25.29)},
    # cctx: real CC transcript payload (bench-legacy-feature session), CLI legs
    # (serial prefill; ms = decode_ms/rounds). sat5 0.714 vs live-T8 0.652.
    "cctx":    {4: (204.14, 22.34), 5: (218.51, 24.19)},
}

def load(payload):
    rows = {}
    with open(f"scratchpad/burst_{payload}.csv") as f:
        for r in csv.reader(f):
            if r[0] == "q":
                continue
            q = int(r[0])
            d = [int(r[1 + 2 * k]) for k in range(SD)]
            m = [float(r[2 + 2 * k]) for k in range(SD)]
            rows[q] = (d, m)
    seq = [int(x) for x in open(f"scratchpad/burst_{payload}.csv.seq").read().split()]
    return rows, seq

def simulate(rows, seq, md, theta=THETA, budget=256):
    q = min(rows)
    tokens = rounds = drafts_launched = width_sum = sat = 0
    fired = [0] * (SD + 1)
    acc = [0] * (SD + 1)
    while tokens < budget and q in rows:
        d, m = rows[q]
        cap = 0
        while cap < md and m[cap] >= theta:
            cap += 1
        W = max(cap + 1, 2)
        # dexit: drafts 0..cap ran (the sub-theta one stopped the loop, but it
        # ran); at cap==md no failing draft. Width-floor top-up at cap==0.
        launched = min(cap + 1, md)
        launched = max(launched, min(W, md))
        accepts = 0
        while accepts < W - 1 and q + 1 + accepts < len(seq) and d[accepts] == seq[q + 1 + accepts]:
            accepts += 1
        n = accepts + 1
        for j in range(1, cap + 1):
            fired[j] += 1
            if n >= j + 1:
                acc[j] += 1
        sat = sat + 1 if n == md + 1 else sat
        tokens += n
        rounds += 1
        drafts_launched += launched
        width_sum += W
        q += n
    return dict(tokens=tokens, rounds=rounds, tpr=tokens / rounds,
                drafts=drafts_launched / rounds, width=width_sum / rounds,
                sat=sat / rounds,
                fired=[f / rounds for f in fired],
                y=[(acc[j] / fired[j] if fired[j] else None) for j in range(SD + 1)])

def main():
    payloads = sys.argv[1:] or list(MEASURED)
    sims = {}  # (payload, md) -> sim
    print("== validation: simulated vs measured tok/round (d4, d5) ==")
    for p in payloads:
        rows, seq = load(p)
        for md in (4, 5, 6, 7, 8):
            sims[p, md] = simulate(rows, seq, md)
        s4, s5 = sims[p, 4], sims[p, 5]
        print(f"{p:8s} d4 sim {s4['tpr']:5.3f}  d5 sim {s5['tpr']:5.3f}  "
              f"(rounds {s4['rounds']}/{s5['rounds']})  "
              f"y5(d5) sim {s5['y'][5] if s5['y'][5] is not None else float('nan'):.3f}")
    # Per-payload marginal cost: the measured d4->d5 ms delta divided by the
    # simulated deep-work delta (drafts + verify lanes, equal-cost units --
    # maxd6 brief measured ~1.0 ms/draft-step and ~0.9-1.2 ms/lane; the fd2
    # width increment DECAYS with width, so extrapolating flat is mildly
    # conservative for d6+).
    print("\n== projections (theta 0.5; per-payload kappa from measured d4->d5 delta;")
    print("   d6+ lane cost EXTRAPOLATED -- no width-7 graph exists) ==")
    print(f"{'payload':8s} {'md':>3s} {'tok/rnd':>7s} {'ms/rnd':>7s} {'t/s':>6s} "
          f"{'vs d5':>6s} {'fired6':>6s} {'y6':>5s} {'fired7':>6s} {'y7':>5s}")
    for p in payloads:
        ms4, ms5 = MEASURED[p][4][1], MEASURED[p][5][1]
        work = {md: sims[p, md]["drafts"] + sims[p, md]["width"] for md in (4, 5, 6, 7, 8)}
        kappa = (ms5 - ms4) / (work[5] - work[4])
        base = sims[p, 5]["tpr"] / ms5 * 1000  # sim tpr over measured ms: trajectory bias cancels in ratios
        for md in (5, 6, 7, 8):
            s = sims[p, md]
            ms = ms5 + kappa * (work[md] - work[5])
            tps = s["tpr"] / ms * 1000
            f6 = s["fired"][6] if md >= 6 else 0.0
            y6 = s["y"][6] if md >= 6 and s["y"][6] is not None else float("nan")
            f7 = s["fired"][7] if md >= 7 else 0.0
            y7 = s["y"][7] if md >= 7 and s["y"][7] is not None else float("nan")
            print(f"{p:8s} d{md:<2d} {s['tpr']:7.3f} {ms:7.2f} {tps:6.1f} "
                  f"{(tps/base-1)*100:+5.1f}% {f6:6.3f} {y6:5.3f} {f7:6.3f} {y7:5.3f} sat={s['sat']:.3f}")
        print(f"{'':8s} (kappa {kappa:.2f} ms/work-unit; measured d5 {MEASURED[p][5][0]} t/s @ {ms5} ms/rnd)")

if __name__ == "__main__":
    main()
