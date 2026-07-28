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
    // enabling and bounding are independent: a request may bound a block it
    // did not itself turn on.
    ok(q27::resolve_think_cfg(json{{"thinking_token_budget", 64}}, true, true, -1).enabled == true,
       "budget: bounding alone does not disable thinking");
    ok(q27::resolve_think_cfg(json{{"enable_thinking", false}, {"thinking_token_budget", 64}},
                              true, true, -1).enabled == false,
       "budget: bounding alone does not re-enable a disabled block");

    // --- server default: fraction of max_tokens, absolute, or opt-out ---
    ok(q27::think_budget_default(-1, 8192) == 4096, "default: <0 flag -> half of max_tokens");
    ok(q27::think_budget_default(0, 8192) == -1, "default: flag 0 -> unbounded (opt out)");
    ok(q27::think_budget_default(1234, 8192) == 1234, "default: flag >0 -> absolute");
    ok(q27::think_budget_default(-1, 0) == -1, "default: no max_tokens -> unbounded");

    printf(failures ? "\nTHINK-RESOLVE: %d FAIL\n" : "\nTHINK-RESOLVE: all pass\n", failures);
    return failures ? 1 : 0;
}
