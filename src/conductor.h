#pragma once
// Continuous-batching P1 (design doc docs/plans/2026-07-14-continuous-batching
// -design.md): host-side conductor pieces. This file starts with the LANE
// POLICY only (design Decisions item 3): per-engine gates + cap -- each engine
// keeps its own P12-gated width and suffix width; when the union of requested
// widths exceeds the fused round's cap (W_MAX), trim the WIDEST requesters
// first, suffix lanes before gated lanes. Rationale: the widest lane paid the
// least per marginal slot (deep speculative lanes are the cheapest to lose),
// and suffix lanes are opportunistic re-emission bets while gated lanes carry
// depthctl-earned evidence -- so at equal width the bet yields before the
// earned width does.
//
// The Conductor itself (registry, round loop, token queues) lands in a later
// task and calls a THIN ENGINE-OWNED surface only -- solo_view()/pre/mix/post/
// tails/set_granted_width -- no friend access, no raw member reaches from this
// header (consensus addendum A4). Pure host code: deterministic, no
// allocation, no CUDA includes; CPU-tested in tools/test_conductor.cpp.

namespace q27 {

// Trim requested verify widths want[0..k) in place until sum(want) <= cap.
// One deterministic victim per step: the current widest lane; ties broken
// suffix-first, then higher slot index. Floor 2 -- never decrement a lane
// below 2 (engine floor: no width-1 gemv); a lane already under the floor is
// never a victim and never raised. If every lane sits at/below the floor and
// the sum still exceeds cap, return with sum > cap rather than loop -- an
// unsatisfiable cap is the caller's admission-control problem. k <= 1 returns
// before touching either array (solo bypasses fusion; is_suffix may be null).
inline void trim_widths(int* want, const bool* is_suffix, int k, int cap) {
    if (k <= 1) return;
    int sum = 0;
    for (int i = 0; i < k; i++) sum += want[i];
    while (sum > cap) {
        int pick = -1; // widest trimmable; ascending scan => later equal
                       // candidates of the same class win (higher-index rule)
        for (int i = 0; i < k; i++) {
            if (want[i] <= 2) continue; // at/under floor: never a victim
            // width tie: i wins unless it would demote a suffix pick to a
            // gated one (sfx beats gated; same class -> later slot, i > pick)
            if (pick < 0 || want[i] > want[pick] ||
                (want[i] == want[pick] && (is_suffix[i] || !is_suffix[pick])))
                pick = i;
        }
        if (pick < 0) return; // all floored: cap unsatisfiable, stop
        want[pick]--;
        sum--;
    }
}

} // namespace q27
