#pragma once
// Adaptive draft-depth ceiling (Q27_MAXD=auto): P13's 4<->5 controller,
// extended 2026-07-08 to a 4..6 ladder per the maxd6 GO verdict
// (docs/maxd6-decision.md). CPU-tested: tools/test_depthctl.cpp.
//
// Per-level EMAs from REALIZED acceptance: at the live level k,
//   sat[k] = EMA(n reached k+1), unconditional  -- promote k->k+1 at >= hi
//   yld[k] = EMA(n reached k+1 | top lane FIRED, cap >= k)
//                                              -- demote k->k-1 at < lo
// lo = 0.35 is the measured d5 win/loss crossover in conditional-yield units
// (BUILDLOG 2026-07-08 accept-gate Phase 0); the level-6 breakeven has the
// same structure (maxd6-decision.md rerun). hi = 0.50 revalidated 3x.
//
// Entering a level (either direction) resets its sat and seeds its yield just
// above the demote line: every stint gets a bounded grace window
// (~ln(2)/ema_a all-miss fired rounds) and must re-earn the next promote --
// prevents stale-EMA thrash cascades (6->5->4). At k_max=5 this is
// behavior-identical to the shipped P13+Phase-1 controller (sat[5]/yld[4]
// were never read there).
//
// The ceiling changes round grouping / draft depth / verify width only --
// never the emitted sequence (greedy is width-invariant).
struct DepthCtl {
    int k_min = 4, k_max = 5; // ladder range; engine raises k_max to 6 under
                              // Q27_MAXD=auto with dexit on (gate_maxd 6)
    int cur = 4;              // live ceiling (starts shallow)
    float sat[7] = {};        // [level]; only the live level's entries update
    float yld[7] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    float ema_a = 1.f / 16.f; // EMA weight (~11-round half-life)
    float hi = 0.50f;         // promote bar on sat (Q27_MAXD_HI)
    float lo = 0.35f;         // demote bar on conditional yield (Q27_MAXD_LO)
    long rounds[7] = {};      // gated rounds run at each level
    long promotes = 0, demotes = 0;

    void enter(int k) {
        sat[k] = 0.f;
        yld[k] = 2.f * lo > 1.f ? 1.f : 2.f * lo;
    }

    // Fold one gated greedy round into the ceiling. md = ceiling the round
    // drafted under (<0 = not a gated round: no-op), cap = this round's
    // margin-run depth, n = tokens committed (1..md+1).
    void update(int md, int cap, int n) {
        if (md < 0) return;
        if (md > 6) md = 6;
        rounds[md]++;
        float hit = (n >= md + 1) ? 1.f : 0.f;
        sat[md] += ema_a * (hit - sat[md]);
        if (md < k_max && sat[md] >= hi) {
            cur = md + 1;
            enter(cur);
            promotes++;
            return; // this round's evidence went into the promote
        }
        // yield evidence is CONDITIONAL on the top lane firing (accept-gate
        // Phase 1): unfired rounds say nothing about the deep lane and, under
        // early-exit, barely pay for it.
        if (md > k_min && cap >= md) {
            yld[md] += ema_a * (hit - yld[md]);
            if (yld[md] < lo) {
                cur = md - 1;
                enter(cur);
                demotes++;
            }
        }
    }
};
