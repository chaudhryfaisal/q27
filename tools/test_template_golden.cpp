// Golden test: q27's renderer must produce, byte for byte, what llama.cpp's
// minja produces from the Qwen3.8 chat template for the same request.
//
// WHY. On 2026-08-22 the two renderers were shown byte-identical for a
// tool-free request, and every remaining difference in an agentic loop lived in
// the tools preamble: nlohmann::json sorts keys so the model saw
// {"function":{"description",...,"name",...},"type"} where the template (and
// the checkpoint's training data) has {"type": "function", "function":
// {"name", "description", "parameters"}}; the instruction text was a
// paraphrase that dropped the template's own "must be nested within
// <tool_call>" reminder -- the rule every drift shape of that week broke; and
// tool results kept a trailing newline the template trims.
//
// The golden is NOT a jinja2 render. It was captured from a running
// llama-server via /apply-template (commit log has the invocation), so it is
// what llama.cpp actually feeds the model. Regenerate it the same way if the
// template or the fixture changes; never edit it by hand.
#include "api_common.h"
#include <cstdio>
#include <fstream>
#include <sstream>

using json = nlohmann::json;

static std::string slurp(const char* p) {
    std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str();
}


static int fails = 0;
static void ok(bool c, const char* what) { printf("  %-66s %s\n", what, c ? "PASS" : "FAIL"); if (!c) fails++; }

// Boundaries the golden does not exercise.
static void boundaries() {
    using q27::anthropic_tools_decl;
    // spacing + escaping must match Python json.dumps(ensure_ascii=False): the
    // expected string below was produced by exactly that call.
    {
        nlohmann::ordered_json v = nlohmann::ordered_json::parse(R"JSON({"type": "function", "function": {"name": "Écrire", "description": "tab\there \"q\" / slash \\ back é 中", "parameters": {"type": "object", "properties": {"z": {"type": "string", "enum": ["a", "b"]}, "a": {"type": "integer"}}, "required": ["z"], "x": [1, 2.5, true, null, {}]}}})JSON");
        ok(q27::ordered_dump_spaced(v) == R"JSON({"type": "function", "function": {"name": "Écrire", "description": "tab\there \"q\" / slash \\ back é 中", "parameters": {"type": "object", "properties": {"z": {"type": "string", "enum": ["a", "b"]}, "a": {"type": "integer"}}, "required": ["z"], "x": [1, 2.5, true, null, {}]}}})JSON",
           "ordered_dump_spaced: unicode, escapes, nested, mixed scalars == json.dumps");
    }
    // malformed entries are skipped the way anthropic_tools_json skips them
    {
        const std::string raw = R"({"tools":[{"description":"nameless"},{"name":123},{"name":""},
            {"name":"ok","description":7,"input_schema":"bad"},{"name":"z2","input_schema":{"b":1,"a":2}}]})";
        const std::string d = anthropic_tools_decl(raw);
        ok(d.find("nameless") == std::string::npos && d.find("123") == std::string::npos,
           "decl: nameless / non-string / empty names are skipped");
        ok(d.find(R"({"type": "function", "function": {"name": "ok", "description": "", "parameters": {}}})") != std::string::npos,
           "decl: bad description -> \"\", bad input_schema -> {}");
        ok(d.find(R"("parameters": {"b": 1, "a": 2})") != std::string::npos,
           "decl: client key order preserved (b before a)");
    }
    // tool_choice subset: keep restricts without reordering
    {
        const std::string raw = R"({"tools":[{"name":"A"},{"name":"B"},{"name":"C"}]})";
        std::vector<std::string> keep = {"C", "A"};
        const std::string d = anthropic_tools_decl(raw, &keep);
        ok(d.find("\"B\"") == std::string::npos && d.find("\"A\"") < d.find("\"C\""),
           "decl: keep filters to the selection and keeps client order");
    }
    // control characters in a description are stripped like the legacy dump
    {
        const std::string raw = "{\"tools\":[{\"name\":\"n\",\"description\":\"a\\u0007b\"}]}";
        const std::string d = anthropic_tools_decl(raw);
        ok(d.find('\x07') == std::string::npos, "decl: control bytes stripped (strip_ctrl)");
    }
    // garbage in -> empty out, never a throw
    ok(anthropic_tools_decl("not json").empty() && anthropic_tools_decl("{}").empty() &&
           anthropic_tools_decl(R"({"tools":"x"})").empty(),
       "decl: unparseable / no tools / wrong type -> empty");
    // legacy fallback: no decl -> the sorted dump still renders the block
    {
        json tools = json::array({{{"type","function"},{"function",{{"name","Read"},{"description","d"},{"parameters",json::object()}}}}});
        const std::string pre = q27::tools_preamble(tools);
        ok(pre.find("\"name\":\"Read\"") != std::string::npos && pre.find("<tools>") != std::string::npos,
           "tools_preamble: empty decl falls back to the sorted dump");
    }
    // tool results are trimmed like the template's |trim
    ok(q27::tool_response_text("index.ts\n") == "<tool_response>\nindex.ts\n</tool_response>" &&
           q27::tool_response_text("  x  ") == "<tool_response>\nx\n</tool_response>" &&
           q27::tool_response_text("\n\n") == "<tool_response>\n\n</tool_response>",
       "tool_response_text: trailing/leading whitespace trimmed, empty stays empty");
}

int main() {
    boundaries();
    if (fails) { printf("template golden: %d boundary FAILURE(S)\n", fails); return 1; }
    const std::string raw = slurp("tools/golden/qwen38_tools_request.anthropic.json");
    const std::string want = slurp("tools/golden/qwen38_tools_request.prompt");
    if (raw.empty() || want.empty()) { fprintf(stderr, "fixture missing (run from repo root)\n"); return 2; }

    json body = json::parse(raw);
    q27::tool_dialect_xml_default() = true;                 // a Qwen3.8 checkpoint boots XML
    q27::TemplateOpts opts = q27::template_opts_from_body(body);
    opts.tools_decl = q27::anthropic_tools_decl(raw);        // ordered, minja-spaced
    const json tools = q27::anthropic_tools_json(body);
    const std::string got = q27::chatml_prompt(q27::anthropic_msgs(body), tools, /*think=*/true,
                                               nullptr, nullptr, {}, nullptr, &opts);
    if (got == want) { printf("template golden: PASS (%zu bytes)\n", got.size()); return 0; }
    // first differing byte, with context, so a failure is diagnosable
    size_t i = 0; while (i < got.size() && i < want.size() && got[i] == want[i]) i++;
    printf("template golden: FAIL at byte %zu (got %zu bytes, want %zu)\n", i, got.size(), want.size());
    auto show = [&](const char* tag, const std::string& s) {
        size_t a = i > 60 ? i - 60 : 0, b = std::min(s.size(), i + 80);
        printf("  %s: %s\n", tag, json(s.substr(a, b - a)).dump().c_str());
    };
    show("got ", got); show("want", want);
    return 1;
}
