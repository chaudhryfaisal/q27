// CPU unit tests for src/conductor.h -- continuous-batching P1 trim policy
// (plan 2026-07-14 Task 6). No CUDA; pins q27::trim_widths() so the fused
// round's width arbitration is reviewable in isolation before the Conductor
// itself lands (Task 9).
//
//   g++ -O2 -std=c++17 -Wall -Wextra -I src tools/test_conductor.cpp
//     -o build/test_conductor && build/test_conductor   (one line)
//
// Semantics pinned here (design doc Decisions item 3 + plan Task 6 DEFINE):
//   - repeatedly decrement the CURRENT WIDEST lane-count until sum <= cap
//   - ties: suffix lanes first, then higher slot index (one deterministic
//     victim per step -- same inputs always trim identically)
//   - floor 2: a lane is never decremented below 2 (no width-1 gemv); if all
//     trimmable lanes sit at/below floor the cap is unsatisfiable and the
//     function returns with sum > cap (caller's problem, not an infinite loop)
//   - k <= 1 returns immediately (solo bypasses fusion; is_suffix may be null)
#include "../src/conductor.h"

#include <cstdio>

static int fails = 0;
#define CHECK(cond, name)                                          \
    do {                                                           \
        bool ok = (cond);                                          \
        printf("  %-58s %s\n", name, ok ? "PASS" : "FAIL");        \
        if (!ok) fails++;                                          \
    } while (0)

static bool eq(const int* a, const int* b, int k) {
    for (int i = 0; i < k; i++)
        if (a[i] != b[i]) return false;
    return true;
}

int main() {
    { // fits: under cap, untouched
        int w[] = {4, 5};
        const bool s[] = {false, false};
        q27::trim_widths(w, s, 2, 12);
        int e[] = {4, 5};
        CHECK(eq(w, e, 2), "fits: {4,5} cap 12 unchanged");
    }
    { // exactly at cap: no trim (boundary of the sum<=cap loop condition)
        int w[] = {6, 6};
        const bool s[] = {false, false};
        q27::trim_widths(w, s, 2, 12);
        int e[] = {6, 6};
        CHECK(eq(w, e, 2), "boundary: {6,6} cap 12 unchanged (sum==cap)");
    }
    { // overflow trims widest first; the {8,7}->{7,7} tie then goes to the
      // higher slot index (neither suffix), landing {6,6} not {5,7}
        int w[] = {8, 7};
        const bool s[] = {false, false};
        q27::trim_widths(w, s, 2, 12);
        int e[] = {6, 6};
        CHECK(eq(w, e, 2), "widest-first: {8,7} cap 12 -> {6,6}");
    }
    { // suffix absorbs all trim before the gated lane loses any (plan case:
      // the suffix lane is the widest at every step until they meet)
        int w[] = {12, 6};
        const bool s[] = {true, false};
        q27::trim_widths(w, s, 2, 12);
        int e[] = {6, 6};
        CHECK(eq(w, e, 2), "suffix-first: {12(sfx),6} cap 12 -> {6,6}");
    }
    { // equal-width tie between a gated and a suffix lane: suffix loses first
      // even though it has the LOWER slot index (suffix rank beats index rank)
        int w[] = {8, 8};
        const bool s[] = {true, false};
        q27::trim_widths(w, s, 2, 15);
        int e[] = {7, 8};
        CHECK(eq(w, e, 2), "tie rule: sfx before gated: {8(sfx),8} cap 15 -> {7,8}");
    }
    { // floor 2: k=4 all-suffix round-robins down from the highest index and
      // stops exactly at cap, well above the floor
        int w[] = {12, 12, 12, 12};
        const bool s[] = {true, true, true, true};
        q27::trim_widths(w, s, 4, 16);
        int e[] = {4, 4, 4, 4};
        CHECK(eq(w, e, 4), "floor path: all-sfx {12,12,12,12} cap 16 -> {4,4,4,4}");
    }
    { // k=1 never trims (solo bypasses fusion anyway); is_suffix legal as null
      // because the function must return before reading it
        int w[] = {16};
        q27::trim_widths(w, nullptr, 1, 12);
        CHECK(w[0] == 16, "k=1: {16} cap 12 untouched (nullptr is_suffix ok)");
    }
    // ---- extra edges (load-bearing for the Task 9 round loop) ----
    { // k=0: no lanes, no reads, no crash
        q27::trim_widths(nullptr, nullptr, 0, 12);
        CHECK(true, "k=0: no-op, no deref");
    }
    { // cap already violated by floors: nothing trimmable -> must TERMINATE
      // and leave the floors intact (sum>cap is the caller's problem)
        int w[] = {2, 2, 2};
        const bool s[] = {false, false, false};
        q27::trim_widths(w, s, 3, 4);
        int e[] = {2, 2, 2};
        CHECK(eq(w, e, 3), "unsatisfiable: {2,2,2} cap 4 terminates unchanged");
    }
    { // partial floor: the one trimmable lane drops to 2, then we stop even
      // though sum (6) still exceeds cap (4) -- floor beats cap
        int w[] = {2, 5, 2};
        const bool s[] = {false, false, false};
        q27::trim_widths(w, s, 3, 4);
        int e[] = {2, 2, 2};
        CHECK(eq(w, e, 3), "floor beats cap: {2,5,2} cap 4 -> {2,2,2}");
    }
    { // sub-floor lane is never a victim and never raised: floor means "do not
      // decrement below 2", not "clamp up to 2"
        int w[] = {1, 9};
        const bool s[] = {false, false};
        q27::trim_widths(w, s, 2, 6);
        int e[] = {1, 5};
        CHECK(eq(w, e, 2), "sub-floor lane untouched: {1,9} cap 6 -> {1,5}");
    }
    { // all-equal, no suffix: deterministic round-robin from the highest index
        int w[] = {5, 5, 5, 5};
        const bool s[] = {false, false, false, false};
        q27::trim_widths(w, s, 4, 18);
        int e[] = {5, 5, 4, 4};
        CHECK(eq(w, e, 4), "all-equal tie determinism: {5,5,5,5} cap 18 -> {5,5,4,4}");
    }
    { // all-equal, mixed classes: BOTH suffix lanes lose before any gated lane
      // does, highest-index suffix first
        int w[] = {5, 5, 5, 5};
        const bool s[] = {false, true, false, true};
        q27::trim_widths(w, s, 4, 18);
        int e[] = {5, 4, 5, 4};
        CHECK(eq(w, e, 4), "mixed tie: sfx lanes {1,3} absorb {5,5,5,5} cap 18 -> {5,4,5,4}");
    }
    { // widest-first is the PRIMARY key: a wide gated lane trims before a
      // narrower suffix lane (suffix rank only breaks ties)
        int w[] = {6, 10};
        const bool s[] = {true, false};
        q27::trim_widths(w, s, 2, 14);
        int e[] = {6, 8};
        CHECK(eq(w, e, 2), "width beats class: {6(sfx),10} cap 14 -> {6,8}");
    }
    printf(fails ? "test_conductor: %d FAILED\n" : "test_conductor: ALL PASS\n", fails);
    return fails ? 1 : 0;
}
