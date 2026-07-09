#pragma once
// P13 adaptive draft-depth ceiling (Q27_MAXD=auto), extracted from engine.cuh
// for CPU testability (accept-gate plan Task 5; tests: tools/test_depthctl.cpp).
//
// Floats the per-stream ceiling between 4 and 5 from REALIZED acceptance:
// promote 4->5 when depth-4 rounds saturate the ceiling often enough
// (sat_ema >= hi), demote 5->4 when the 5th lane stops paying
// (yield_ema < lo). On promote, yield_ema is seeded just above the demote
// line so depth-5 gets a bounded grace window (~1/ema_a rounds) to prove
// itself. The ceiling changes round grouping / draft depth / verify width
// only -- never the emitted sequence (greedy is width-invariant).
//
// Host-side control logic only: no CUDA types, no engine state. The engine
// guards calls with (maxd_auto && md_used >= 0) semantics via update()'s
// own md_used check.
struct DepthCtl {
    int cur = 4;              // live ceiling (starts shallow)
    float sat_ema = 0.f;      // depth-4: EMA of (n reached ceiling)
    float yield_ema = 1.f;    // depth-5: EMA of (5th lane accepted)
    float ema_a = 1.f / 16.f; // EMA weight (~11-round half-life)
    float hi = 0.50f;         // promote 4->5 when sat_ema >= hi (Q27_MAXD_HI)
    float lo = 0.10f;         // demote 5->4 when yield_ema < lo (Q27_MAXD_LO)
    long rounds4 = 0, rounds5 = 0;  // gated rounds run at each ceiling
    long promotes = 0, demotes = 0; // 4->5 / 5->4 transitions

    // Fold one gated greedy round into the ceiling. md_used = ceiling the
    // round drafted under (<0 = not a gated round: no-op), gate_cap = this
    // round's margin-run depth (unused pre-Phase-1; the acceptance-gate
    // conditional-yield change consumes it), n = tokens committed (1..6).
    void update(int md_used, int gate_cap, int n) {
        (void)gate_cap;
        if (md_used < 0) return;
        if (md_used < 5) {
            rounds4++;
            float hit = (n >= md_used + 1) ? 1.f : 0.f;
            sat_ema += ema_a * (hit - sat_ema);
            if (sat_ema >= hi) { cur = 5; yield_ema = 2.f * lo; promotes++; }
        } else {
            rounds5++;
            float hit = (n >= 6) ? 1.f : 0.f;
            yield_ema += ema_a * (hit - yield_ema);
            if (yield_ema < lo) { cur = 4; sat_ema = 0.f; demotes++; }
        }
    }
};
