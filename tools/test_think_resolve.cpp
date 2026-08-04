// CPU unit test for resolve_think (api_common.h): per-request thinking opt-in.
// The server profile sets the DEFAULT (server_default = !no_think_srv); an
// explicit request field overrides it in either direction, across all three
// client conventions. Malformed fields must be ignored, never thrown.
//
// Build+run (no CUDA): g++ -std=c++17 -I src tools/test_think_resolve.cpp -o build/test_think_resolve && ./build/test_think_resolve
#include "api_common.h"
#include <cstdio>

using json = nlohmann::json;

static int failures = 0;
static void ok(bool c, const char* n) {
    printf("  %s %s\n", c ? "PASS" : "FAIL", n);
    if (!c) failures++;
}


int main() {
    // existing cases exercise the honoring path (allow_request=true, i.e. the
    // server booted with --request-think); a local wrapper supplies that arg.
    auto resolve_think = [](const json& b, bool sd) { return q27::resolve_think(b, sd, true); };

    // --- GATING: without --request-think (allow_request=false) the request's
    // thinking fields are IGNORED and the server default stands (the fix for
    // benchmarks that send enable_thinking:True flipping a no-think server). ---
    ok(q27::resolve_think(json{{"enable_thinking", true}}, false, false) == false,
       "gated: enable_thinking:true IGNORED when allow_request=false -> server default (no-think)");
    ok(q27::resolve_think(json{{"enable_thinking", false}}, true, false) == true,
       "gated: enable_thinking:false IGNORED when allow_request=false -> server default (think)");
    ok(q27::resolve_think(json{{"thinking", {{"type", "enabled"}}}}, false, false) == false,
       "gated: anthropic thinking IGNORED when allow_request=false");

    // --- server default honored when the request says nothing ---
    ok(resolve_think(json::object(), false) == false, "no-think server + silent -> no-think");
    ok(resolve_think(json::object(), true) == true, "--think server + silent -> think");

    // --- OpenAI/Qwen top-level enable_thinking overrides in BOTH directions ---
    ok(resolve_think(json{{"enable_thinking", true}}, false) == true,
       "no-think server + enable_thinking:true -> think (THE opt-in)");
    ok(resolve_think(json{{"enable_thinking", false}}, true) == false,
       "--think server + enable_thinking:false -> no-think");

    // --- llama.cpp/GLM nested chat_template_kwargs.enable_thinking ---
    ok(resolve_think(json{{"chat_template_kwargs", {{"enable_thinking", true}}}}, false) == true,
       "nested kwargs true -> think");
    ok(resolve_think(json{{"chat_template_kwargs", {{"enable_thinking", false}}}}, true) == false,
       "nested kwargs false -> no-think");

    // --- Anthropic native thinking field (what Claude Code's toggle emits) ---
    ok(resolve_think(json{{"thinking", {{"type", "enabled"}}}}, false) == true,
       "anthropic thinking enabled -> think");
    ok(resolve_think(json{{"thinking", {{"type", "disabled"}}}}, true) == false,
       "anthropic thinking disabled -> no-think");
    ok(resolve_think(json{{"thinking", {{"type", "enabled"}, {"budget_tokens", 2000}}}}, false) == true,
       "anthropic thinking enabled + budget_tokens ignored -> think");

    // --- malformed / wrong-typed fields: never throw, leave the default in force ---
    ok(resolve_think(json{{"enable_thinking", "yes"}}, false) == false,
       "string enable_thinking ignored -> default");
    ok(resolve_think(json{{"enable_thinking", 1}}, true) == true,
       "int enable_thinking ignored -> default");
    ok(resolve_think(json{{"thinking", "enabled"}}, false) == false,
       "non-object thinking ignored -> default");
    ok(resolve_think(json{{"thinking", {{"type", 3}}}}, true) == true,
       "non-string thinking.type ignored -> default");
    ok(resolve_think(json{{"chat_template_kwargs", "x"}}, false) == false,
       "non-object chat_template_kwargs ignored -> default");
    ok(resolve_think(json{{"thinking", {{"type", "bogus"}}}}, false) == false,
       "unknown thinking.type leaves default (false)");
    ok(resolve_think(json{{"thinking", {{"type", "bogus"}}}}, true) == true,
       "unknown thinking.type leaves default (true)");

    // --- BUDGET (2026-07-28). Every convention that can enable thinking must
    // also be able to bound it; an accepted-and-inert field reads as working
    // and is worse than an absent one (club-3090 #741). ---
    auto bud = [](const json& b, int server_budget = -1, bool allow = true) {
        return q27::resolve_think_cfg(b, true, allow, server_budget).budget;
    };
    ok(bud(json::object()) == -1, "budget: silent request -> unbounded (-1)");
    ok(bud(json::object(), 4096) == 4096, "budget: silent request -> server budget");
    ok(bud(json{{"thinking", {{"type", "enabled"}, {"budget_tokens", 512}}}}) == 512,
       "budget: anthropic thinking.budget_tokens honored");
    ok(bud(json{{"thinking_token_budget", 256}}) == 256,
       "budget: openai/qwen thinking_token_budget honored");
    ok(bud(json{{"chat_template_kwargs", {{"thinking_budget", 128}}}}) == 128,
       "budget: llama.cpp chat_template_kwargs.thinking_budget honored");
    ok(bud(json{{"thinking", {{"budget_tokens", 512}}}}, 4096, false) == 4096,
       "budget: GATED -- request budget IGNORED without --request-think");
    ok(bud(json{{"thinking", {{"budget_tokens", 0}}}}) == 0,
       "budget: explicit 0 is honored (not treated as unset)");
    ok(bud(json{{"thinking", {{"budget_tokens", "512"}}}}, 4096) == 4096,
       "budget: non-integer budget ignored -> server budget");
    ok(bud(json{{"thinking", {{"budget_tokens", -1}}}}, 4096) == -1,
       "budget: negative request budget means unbounded, overriding server");
    ok(bud(json{{"thinking_token_budget", 2147483648ULL}}, 4096) == 4096,
       "budget: unsigned value above INT_MAX ignored -> server budget");
    ok(bud(json{{"thinking", {{"budget_tokens", -2147483649LL}}}}, 4096) == 4096,
       "budget: signed value below INT_MIN ignored -> server budget");
    auto explicit_unbounded = q27::resolve_think_cfg(
        json{{"thinking", {{"budget_tokens", -1}}}}, true, true, 4096);
    ok(explicit_unbounded.budget_set,
       "budget: explicit negative override remains distinguishable from default");
    ok(q27::think_budget_for_request(true, explicit_unbounded, 64, 128) == -1,
       "budget: explicit unbounded request overrides absolute server flag");
    auto silent_budget = q27::resolve_think_cfg(json::object(), true, true, -1);
    ok(!silent_budget.budget_set,
       "budget: absent request budget keeps server default eligible");
    auto bounded_limits = q27::resolve_think_decode_limits(
        8, 20, 8, 5, 2, true, q27::ThinkCfg{true, 0, true}, -1);
    ok(bounded_limits.n_max == 6,
       "budget: bounded close ids reserve decoder context outside public cap");
    auto unbounded_limits = q27::resolve_think_decode_limits(
        8, 20, 8, 5, 2, true, q27::ThinkCfg{true, -1, true}, -1);
    ok(unbounded_limits.n_max == 8,
       "budget: explicit unbounded span needs no forced-close reserve");
    auto default_limits = q27::resolve_think_decode_limits(
        8, 20, 8, 5, 2, true, q27::ThinkCfg{true, -1, false}, -1);
    ok(default_limits.n_max == 6 && default_limits.budget == 3,
       "budget: fractional default recomputes against context-clamped public cap");
    auto unattainable_limits = q27::resolve_think_decode_limits(
        8, 20, 8, 5, 2, true, q27::ThinkCfg{true, 64, true}, -1);
    ok(unattainable_limits.context_ok && unattainable_limits.n_max == 8 &&
           unattainable_limits.budget == -1,
       "budget: unattainable absolute cap reserves nothing and disables enforcement");
    auto impossible_limits = q27::resolve_think_decode_limits(
        8, 14, 8, 5, 2, true, q27::ThinkCfg{true, 0, true}, -1);
    ok(!impossible_limits.context_ok,
       "budget: context unable to fit close plus answer is rejected");
    auto fixed_point_limits = q27::resolve_think_decode_limits(
        8, 16, 8, 5, 2, true, q27::ThinkCfg{true, -1, false}, -1);
    ok(fixed_point_limits.context_ok && fixed_point_limits.n_max == 2 &&
           fixed_point_limits.budget == 1,
       "budget: fractional cap is solved after reserving close tokens");
    ok(q27::max_prompt_for_think_decode(
           8, 16, 5, 11, 10, 2, true, q27::ThinkCfg{true, -1, false}, -1) == 9,
       "budget: reported fractional prompt ceiling matches fixed-point admission");
    ok(q27::max_prompt_for_think_decode(
           8, 16, 5, 11, 12, 2, true, q27::ThinkCfg{true, -1, false}, 0) == 11,
       "budget: unbounded prompt ceiling keeps the ordinary context limit");
    ok(q27::max_prompt_for_think_decode(
           8, 16, 5, 11, 7, 2, true, q27::ThinkCfg{true, 4, true}, -1) == 5,
       "budget: nonmonotonic admission reports a lower compaction target");
    auto zero_output_too_large = q27::resolve_think_decode_limits(
        0, 12, 8, 5, 2, false, q27::ThinkCfg{}, -1);
    ok(!zero_output_too_large.context_ok,
       "budget: zero-output request still rejects a prompt with no decode reserve");
    auto zero_output_fits = q27::resolve_think_decode_limits(
        0, 13, 8, 5, 2, false, q27::ThinkCfg{}, -1);
    ok(zero_output_fits.context_ok && zero_output_fits.n_max == 0,
       "budget: zero-output request may route when the prompt reserve fits");
    // enabling and bounding are independent: a request may bound a block it
    // did not itself turn on.
    ok(q27::resolve_think_cfg(json{{"thinking_token_budget", 64}}, true, true, -1).enabled == true,
       "budget: bounding alone does not disable thinking");
    ok(q27::resolve_think_cfg(json{{"enable_thinking", false}, {"thinking_token_budget", 64}},
                              true, true, -1).enabled == false,
       "budget: bounding alone does not re-enable a disabled block");
    const auto disabled_bounded = q27::resolve_think_cfg(
        json{{"enable_thinking", false}, {"thinking_token_budget", 32}},
        true, true, -1);
    const auto disabled_limits = q27::resolve_think_decode_limits(
        8, 20, 8, 5, 2, false, disabled_bounded, -1);
    ok(disabled_limits.context_ok && disabled_limits.n_max == 8 &&
           disabled_limits.budget == -1,
       "budget: explicit thinking disable reserves no close tokens and enforces no cap");
    ok(q27::think_budget_for_request(false, q27::ThinkCfg{}, -1, 8) == -1,
       "default: no-think request preserves sampled speculation");
    ok(q27::think_budget_for_request(false, q27::ThinkCfg{false, 3, true}, 0, 8) == 3,
       "default: explicit request budget applies without prompt-seeded thinking");
    ok(q27::think_budget_for_request(false, q27::ThinkCfg{}, 3, 8) == 3,
       "default: explicit server budget applies without prompt-seeded thinking");

    const auto active_two_token_limits = q27::resolve_think_decode_limits(
        2, 20, 8, 5, 2, true, q27::ThinkCfg{}, -1);
    ok(active_two_token_limits.context_ok && active_two_token_limits.n_max == 2 &&
           active_two_token_limits.budget == 1,
       "budget: prompt-seeded think reserves one answer token");
    const auto spontaneous_two_token_limits = q27::resolve_think_decode_limits(
        2, 20, 8, 5, 2, false, q27::ThinkCfg{}, -1);
    ok(spontaneous_two_token_limits.context_ok &&
           spontaneous_two_token_limits.n_max == 2 &&
           spontaneous_two_token_limits.budget == -1,
       "budget: short spontaneous request falls back to ordinary length stop");
    const auto spontaneous_one_token_limits = q27::resolve_think_decode_limits(
        1, 20, 8, 5, 2, false, q27::ThinkCfg{}, -1);
    ok(spontaneous_one_token_limits.context_ok &&
           spontaneous_one_token_limits.n_max == 1 &&
           spontaneous_one_token_limits.budget == -1,
       "budget: one-token non-thinking request is not a context overflow");
    const auto spontaneous_three_token_limits = q27::resolve_think_decode_limits(
        3, 20, 8, 5, 2, false, q27::ThinkCfg{}, -1);
    ok(spontaneous_three_token_limits.context_ok &&
           spontaneous_three_token_limits.n_max == 3 &&
           spontaneous_three_token_limits.budget == -1,
       "budget: default no-think request keeps ordinary decode width");
    const auto spontaneous_zero_budget_limits = q27::resolve_think_decode_limits(
        2, 20, 8, 5, 2, false, q27::ThinkCfg{true, 0, true}, -1);
    ok(spontaneous_zero_budget_limits.context_ok &&
           spontaneous_zero_budget_limits.n_max == 2 &&
           spontaneous_zero_budget_limits.budget == 0,
       "budget: explicit zero spontaneous budget keeps opener and answer capacity");

    // --- server default: fraction of max_tokens, absolute, or opt-out ---
    ok(q27::think_budget_default(-1, 8192) == 4096, "default: <0 flag -> half of max_tokens");
    ok(q27::think_budget_default(0, 8192) == -1, "default: flag 0 -> unbounded (opt out)");
    ok(q27::think_budget_default(1234, 8192) == 1234, "default: flag >0 -> absolute");
    ok(q27::think_budget_default(-1, 0) == -1, "default: no max_tokens -> unbounded");

    // --- round-boundary enforcement: overshoot, natural close, and re-entry ---
    using Action = q27::ThinkBudgetAction;
    {
        q27::ThinkBudgetState state(-1);
        ok(state.start() == Action::NONE, "enforce: unbounded start is inert");
        state.observe(q27::StreamSplitter::THINK, q27::StreamSplitter::THINK);
        ok(state.finish_round(q27::StreamSplitter::THINK) == Action::NONE &&
               state.used == 1 && !state.tripped,
           "enforce: unbounded counts usage without forcing a close");
    }
    {
        q27::ThinkBudgetState state(0);
        ok(state.start() == Action::FORCE_RESERVED && state.used == 0 &&
               state.tripped && state.transition_pending && !state.reserved_close,
           "enforce: zero budget requests the reserved close before decoding");
    }
    {
        q27::ThinkBudgetState state(0);
        ok(state.start(q27::StreamSplitter::TEXT) == Action::NONE && !state.tripped,
           "enforce: zero budget stays armed outside a think span");
        state.observe(q27::StreamSplitter::TEXT, q27::StreamSplitter::THINK);
        ok(state.finish_round(q27::StreamSplitter::THINK) == Action::FORCE_RESERVED &&
               state.tripped,
           "enforce: zero budget closes a later spontaneous think span");
    }
    {
        q27::ThinkBudgetState state(2);
        ok(state.start() == Action::NONE, "enforce: positive budget starts normally");
        state.observe(q27::StreamSplitter::THINK, q27::StreamSplitter::THINK);
        state.observe(q27::StreamSplitter::THINK, q27::StreamSplitter::THINK);
        state.observe(q27::StreamSplitter::THINK, q27::StreamSplitter::THINK);
        ok(state.finish_round(q27::StreamSplitter::THINK) == Action::FORCE_RESERVED &&
               state.used == 3 && state.tripped && state.transition_pending,
           "enforce: fused round may overshoot but uses its retained token count");
        ok(state.finish_round(q27::StreamSplitter::THINK) == Action::NONE,
           "enforce: a pending transition is requested only once");
        state.observe(q27::StreamSplitter::THINK, q27::StreamSplitter::TEXT, true);
        ok(state.tripped && !state.transition_pending && state.used == 3,
           "enforce: forced close exits think without adding reasoning usage");
        state.observe(q27::StreamSplitter::TEXT, q27::StreamSplitter::THINK);
        ok(state.finish_round(q27::StreamSplitter::THINK) == Action::FORCE_PUBLIC,
           "enforce: later re-entry borrows close capacity from public tokens");
    }
    {
        q27::ThinkBudgetState state(1);
        state.observe(q27::StreamSplitter::THINK, q27::StreamSplitter::THINK);
        state.observe(q27::StreamSplitter::THINK, q27::StreamSplitter::TEXT);
        ok(state.finish_round(q27::StreamSplitter::TEXT) == Action::NONE &&
               state.used == 2 && !state.tripped && state.reserved_close,
           "enforce: natural close later in the same round suppresses forcing");
        state.observe(q27::StreamSplitter::TEXT, q27::StreamSplitter::THINK);
        ok(state.finish_round(q27::StreamSplitter::THINK) == Action::FORCE_RESERVED,
           "enforce: unused reserved close remains available for later re-entry");
    }

    printf(failures ? "\nTHINK-RESOLVE: %d FAIL\n" : "\nTHINK-RESOLVE: all pass\n", failures);
    return failures ? 1 : 0;
}
