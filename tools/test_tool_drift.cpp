// CPU-only regression test for the bare-tool-call drift recoveries in
// api_common.h (parse_bare_tool_calls). Covers the modes that have bitten
// real Claude Code sessions:
//   mode 10 -- dropped `{"name": "` opener (issue: flask-5014 early quit)
//   mode 11 -- raw code-body string value, unescaped inner quotes (issue #4)
// plus the negatives (prose must not false-recover, well-formed calls take
// the normal path).
//
// Build + run (no CUDA needed):
//   g++ -std=c++17 -I src tools/test_tool_drift.cpp -o build/test_tool_drift && ./build/test_tool_drift
#include "api_common.h"
#include <cstdio>
#include <string>

using json = nlohmann::json;

static int failures = 0;
static void ok(bool cond, const char* name) {
    printf("  %s %s\n", cond ? "PASS" : "FAIL", name);
    if (!cond) failures++;
}

static json tool(const char* name, std::vector<std::pair<std::string, bool>> params) {
    // params: (key, is_required); all typed string for these tests
    json props = json::object(), req = json::array();
    for (auto& p : params) {
        props[p.first] = {{"type", "string"}};
        if (p.second) req.push_back(p.first);
    }
    return {{"type", "function"},
            {"function",
             {{"name", name},
              {"parameters", {{"type", "object"}, {"properties", props}, {"required", req}}}}}};
}

// Drift mode 13 (2026-07-24, found live by the grammar-engage probe): a
// wrapper-less call truncated INSIDE an escape sequence. The repair used to
// append the closing quote straight after a dangling backslash, which escaped
// it -- string still open, object never parsed, call UN-RESCUED. The real
// payload was a Write whose markdown content was cut while writing an escaped
// JSON example (`\\"role\\": \\"assistant\\",\\`).
// Drift mode 14 (2026-08-14, captured live from a thunderdome run on the
// Qwen3.8-27B q5f repack, bench-time-tracker trial-1). The model was told to
// emit `<tool_call>\n{"name":..., "arguments":{...}}\n</tool_call>` by
// tools_preamble and DID so correctly on turn 1, then drifted on turn 2 into
// its chat-template's XML dialect with no <tool_call> wrapper at all:
//
//   <tool_name>Read</tool_name>
//   <parameter=file_path>/workspace/tests/index.test.ts</parameter>
//   <tool_name>Bash</tool_name>
//   <parameter=arguments>
//   {"command":"...","description":"..."}}
//
// Three things make this its own mode rather than a variant of 10/11:
//   1. TWO calls are concatenated in one assistant block.
//   2. The conventions DISAGREE between them -- Read uses a scalar
//      <parameter=KEY>VALUE</parameter>, Bash uses <parameter=arguments>
//      wrapping a whole JSON object and never closes the tag.
//   3. The Bash payload carries one unbalanced trailing brace.
//
// Claude Code saw plain text, made no tool call, and the trial ended at
// num_turns=2 scoring 0.000 -- the same shape as the pre-fix 3.6 agentic
// ceiling, which was parser-bound rather than quality-bound.
//
// NOT YET RESCUED. This fixture pins the observed bytes and the CURRENT
// behaviour so a future parser change has a real target and a regression
// witness. Flip `expect_rescued` to true in the same commit that teaches
// parse_bare_tool_calls this dialect.
// Drift mode 15 (2026-08-14, Qwen3.8-27B): surfaced by re-running the same
// thunderdome task once mode 14 stopped the earlier stall. An unclosed <name>
// pseudo-tag, then a bare identifier and JSON args with no opening brace or
// quote, plus the same trailing unbalanced brace seen in mode 14.
static void test_mode15_name_tag_bare_args() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"Bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"}},"required":["command"]}}}])");
    // Verbatim bytes from the captured transcript.
    std::string t =
        "<name>Bash, \"arguments\": {\"command\":\"ls /workspace/tests /workspace/src && cat "
        "/workspace/.eslintrc.cjs /workspace/vitest.config.ts\",\"description\":\"List tests "
        "and src, show eslint and vitest config\"}}";
    std::string pre;
    auto v = q27::parse_bare_tool_calls(t, &pre, &tools);
    ok(v.size() == 1 && v[0].ok && v[0].name == "Bash",
       "mode15: <name> pseudo-tag with bare args recovered");
    if (v.size() == 1)
        ok(v[0].arguments.value("description", std::string()) ==
               "List tests and src, show eslint and vitest config",
           "mode15: trailing unbalanced brace tolerated, args intact");
    // An undeclared name must NOT be rescued.
    std::string bad = "<name>NotATool, \"arguments\": {\"x\":1}}";
    auto v2 = q27::parse_bare_tool_calls(bad, &pre, &tools);
    ok(v2.empty(), "mode15: undeclared name is rejected");
}

// Drift mode 16 and the mode-15 attribute variant (2026-08-14), both harvested
// from the same batch of Qwen3.8 thunderdome runs. Also pins the one form that
// must NEVER be rescued.
static void test_mode16_and_15_variants() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"Bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"}},"required":["command"]}}},{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}}])");
    std::string pre;

    // mode 15 variant: attribute spelling <name="Read", ...
    std::string attr = "<name=\"Read\", \"arguments\": {\"file_path\": \"/workspace/tests/tracker.test.ts\"}}";
    auto v1 = q27::parse_bare_tool_calls(attr, &pre, &tools);
    ok(v1.size() == 1 && v1[0].ok && v1[0].name == "Read" &&
           v1[0].arguments.value("file_path", std::string()) ==
               "/workspace/tests/tracker.test.ts",
       "mode15-attr: <name=\"X\" spelling recovered");

    // mode 16: correct JSON, wrong wrapper.
    std::string fn = "I'll start by exploring the workspace.\n\n<function>\n"
        "{\"name\": \"Bash\", \"arguments\": {\"command\": \"ls -la /workspace\", "
        "\"description\": \"List workspace\"}}";
    auto v2 = q27::parse_bare_tool_calls(fn, &pre, &tools);
    ok(v2.size() == 1 && v2[0].ok && v2[0].name == "Bash" &&
           v2[0].arguments.value("command", std::string()) == "ls -la /workspace",
       "mode16: <function>-wrapped JSON recovered");
    ok(pre.find("exploring the workspace") != std::string::npos,
       "mode16: prose before the wrapper is preserved as prefix");

    // MUST NOT RESCUE: a hallucinated tool RESULT, not a call. Rescuing this
    // would feed invented command output back as though a tool had run.
    std::string halluc =
        "I'll start by exploring the codebase.\n\n<tool_calls>\n<result>\n"
        "<name>Bash</name>\n<output>total 40\ndrwxr-xr-x 1 node node 4096 .\n</output>\n"
        "</result>\n</tool_calls>";
    auto v3 = q27::parse_bare_tool_calls(halluc, &pre, &tools);
    ok(v3.empty(), "hallucinated <result>/<output> block is NOT rescued as a call");
}

// Native XML dialect (2026-08-14): the format Qwen3.8's chat template trains.
// Not a drift mode -- the first-class wrapped-body format, parsed by
// parse_tool_call via parse_native_xml_call. Round-trips the template's own
// example shape, typed values, multi-line values, zero-parameter calls, and
// refuses a truncated parameter rather than guessing.
// Per-model dialect default (keyed on general.name, normalized because
// conversion mangles it -- the real 3.8 artifact says "Qwen38 27b Hf").
// Think-mode drift (2026-08-14): bare native dialect without the wrapper, and
// the JSON-head/XML-params chimera that killed bench-time-tracker at turn 4.
// Full-suite capture (2026-08-15): <function=NAME> opener with mode-14-style
// <parameter=arguments> wrapping the whole JSON, streamed with NO closers.
// Fell between mode 14 (wrong opener) and the bare-native rescue (demanded
// closers) and zeroed the first four suite tasks.
static void test_function_arguments_unterminated() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"Bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"}},"required":["command"]}}}])");
    std::string pre;
    auto v = q27::parse_bare_tool_calls(
        "\n\n<function=Bash>\n<parameter=arguments>\n"
        "{\"command\":\"cat /workspace/schema.sql /workspace/TASK.md\","
        "\"description\":\"Show schema and task\"}",
        &pre, &tools);
    ok(v.size() == 1 && v[0].ok && v[0].name == "Bash" &&
           v[0].arguments.value("command", std::string()).rfind("cat /workspace", 0) == 0 &&
           v[0].arguments.value("description", std::string()) == "Show schema and task" &&
           !v[0].arguments.contains("arguments"),
       "suite-form: unterminated <parameter=arguments> merged, not nested");
    // an unterminated NON-final parameter (another <parameter= follows) still refuses
    q27::ToolCall t2;
    ok(!q27::parse_native_xml_call(
           "<function=Bash>\n<parameter=command>\nls\n<parameter=description>\nx\n</parameter>\n</function>", t2),
       "suite-form: unterminated mid-stream parameter still refused");
}

static void test_think_mode_drift() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"Write","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}},{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}}])");
    std::string pre;
    // bare <function=NAME>, no <tool_call> wrapper (task-queue stray blocks)
    auto v1 = q27::parse_bare_tool_calls(
        "\n<function=Read>\n<parameter=file_path>\n/workspace/tests/phase-06.test.ts\n</parameter>\n</function>\n",
        &pre, &tools);
    ok(v1.size() == 1 && v1[0].ok && v1[0].name == "Read" &&
           v1[0].arguments.value("file_path", std::string()) ==
               "/workspace/tests/phase-06.test.ts",
       "bare native dialect without wrapper recovered");
    // the chimera, verbatim shape from the archived transcript
    auto v2 = q27::parse_bare_tool_calls(
        "The test suite is clear. Writing the implementation:\n\n"
        "{\"name\": \"Write\",\n<parameter=file_path>\n/workspace/src/index.ts\n</parameter>\n"
        "<parameter=content>\nimport { existsSync, mkdirSync } from 'fs';\nconst x = 1;\n</parameter>",
        &pre, &tools);
    ok(v2.size() == 1 && v2[0].ok && v2[0].name == "Write" &&
           v2[0].arguments.value("file_path", std::string()) == "/workspace/src/index.ts" &&
           v2[0].arguments.value("content", std::string()).rfind("import { existsSync", 0) == 0,
       "mode17: json-head/xml-params chimera recovered");
    ok(pre.find("Writing the implementation") != std::string::npos,
       "mode17: prose before the chimera preserved as prefix");
    // an UNDECLARED chimera name stays text
    auto v3 = q27::parse_bare_tool_calls(
        "{\"name\": \"NotATool\",\n<parameter=x>\n1\n</parameter>", &pre, &tools);
    ok(v3.empty(), "mode17: undeclared chimera name is not rescued");
}

static void test_dialect_default_keying() {
    unsetenv("Q27_TOOL_DIALECT");
    auto meta = [](const char* n) { return std::string("{\"general.name\": \"") + n + "\"}"; };
    q27::set_tool_dialect_for_model(meta("Qwen38 27b Hf"));
    ok(q27::tool_dialect_xml(), "dialect: mangled 3.8 name selects xml");
    q27::set_tool_dialect_for_model(meta("Qwen3.8-27B"));
    ok(q27::tool_dialect_xml(), "dialect: clean 3.8 name selects xml");
    q27::set_tool_dialect_for_model(meta("Qwen3.6-27B"));
    ok(!q27::tool_dialect_xml(), "dialect: 3.6 stays json");
    q27::set_tool_dialect_for_model(meta("Qwopus3.6 27B v2"));
    ok(!q27::tool_dialect_xml(), "dialect: qwopus fine-tune stays json");
    // env overrides win in BOTH directions
    setenv("Q27_TOOL_DIALECT", "json", 1);
    q27::set_tool_dialect_for_model(meta("Qwen3.8-27B"));
    ok(!q27::tool_dialect_xml(), "dialect: env json overrides a 3.8 model");
    setenv("Q27_TOOL_DIALECT", "xml", 1);
    q27::set_tool_dialect_for_model(meta("Qwen3.6-27B"));
    ok(q27::tool_dialect_xml(), "dialect: env xml overrides a 3.6 model");
    unsetenv("Q27_TOOL_DIALECT");
    q27::tool_dialect_xml_default() = false;   // leave global state clean
}

static void test_native_xml_dialect() {
    q27::ToolCall tc;
    ok(q27::parse_native_xml_call(
           "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
           "<parameter=units>\nmetric\n</parameter>\n</function>\n", tc) &&
           tc.ok && tc.name == "get_weather" &&
           tc.arguments.value("city", std::string()) == "Paris" &&
           tc.arguments.value("units", std::string()) == "metric",
       "native-xml: two scalar parameters round-trip");
    q27::ToolCall t2;
    ok(q27::parse_native_xml_call(
           "<function=Write>\n<parameter=content>\nline one\nline two\n</parameter>\n"
           "<parameter=count>\n3\n</parameter>\n<parameter=force>\ntrue\n</parameter>\n"
           "</function>", t2) &&
           t2.arguments.value("content", std::string()) == "line one\nline two" &&
           t2.arguments.value("count", 0) == 3 && t2.arguments.value("force", false) == true,
       "native-xml: multi-line string, tojson-typed number and bool");
    q27::ToolCall t3;
    ok(q27::parse_native_xml_call("<function=list_files>\n</function>", t3) &&
           t3.ok && t3.arguments.empty(),
       "native-xml: zero-parameter call is legal");
    q27::ToolCall t4;
    ok(q27::parse_native_xml_call(
           "<function=Write>\n<parameter=content>\ntruncated with no closer", t4) &&
           t4.arguments.value("content", std::string()) == "truncated with no closer",
       "native-xml: unterminated FINAL parameter closes at EOF (2026-08-15 leniency)");
    q27::ToolCall t5;
    ok(!q27::parse_native_xml_call("plain text, no dialect", t5),
       "native-xml: non-dialect text is not consumed");
}

static void test_mode14_tool_name_xml_dialect() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}},{"type":"function","function":{"name":"Bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"}},"required":["command"]}}}])");
    // Verbatim bytes from the captured transcript.
    std::string t =
        "<tool_name>Read</tool_name>\n"
        "<parameter=file_path>/workspace/tests/index.test.ts</parameter>\n"
        "<tool_name>Bash</tool_name>\n"
        "<parameter=arguments>\n"
        "{\"command\":\"ls /workspace/src /workspace/tests && cat /workspace/.eslintrc.cjs "
        "/workspace/vitest.config.ts\",\"description\":\"List src/tests and show eslint and "
        "vitest configs\"}}";
    const bool expect_rescued = true;    // rescued as of the mode-14 parser path
    std::string pre;
    auto v = q27::parse_bare_tool_calls(t, &pre, &tools);
    ok((!v.empty()) == expect_rescued,
       expect_rescued ? "mode14: <tool_name>/<parameter=> dialect rescued"
                      : "mode14: <tool_name>/<parameter=> dialect UN-RESCUED (documented)");
    // Both calls must come back, with the scalar parameter AND the JSON-object
    // parameter each mapped correctly, and the trailing brace tolerated.
    ok(v.size() == 2, "mode14: both concatenated calls recovered");
    if (v.size() == 2) {
        ok(v[0].ok && v[0].name == "Read" &&
               v[0].arguments.value("file_path", std::string()) ==
                   "/workspace/tests/index.test.ts",
           "mode14: scalar <parameter=file_path> mapped");
        ok(v[1].ok && v[1].name == "Bash" &&
               v[1].arguments.value("command", std::string()).rfind("ls /workspace/src", 0) == 0 &&
               v[1].arguments.value("description", std::string()) ==
                   "List src/tests and show eslint and vitest configs",
           "mode14: <parameter=arguments> JSON object merged, trailing brace tolerated");
    }
}

static void test_mode13_truncated_mid_escape() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"Write","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}}])");
    // trailing dangling backslash
    std::string t1 = "Let me write that.\n{\"name\": \"Write\", \"arguments\": {\"file_path\": \"g.md\", "
                     "\"content\": \"# Guide\\n\\n```json\\n{\\n  \\\"role\\\": \\\"assistant\\\",\\";
    std::string pre;
    auto v1 = q27::parse_bare_tool_calls(t1, &pre, &tools);
    ok(v1.size() == 1 && v1[0].ok && v1[0].name == "Write",
       "mode13: truncated at a dangling backslash");
    // partial \uXXXX at the cut
    std::string t2 = "{\"name\": \"Write\", \"arguments\": {\"file_path\": \"g.md\", \"content\": \"caf\\u00";
    auto v2 = q27::parse_bare_tool_calls(t2, &pre, &tools);
    ok(v2.size() == 1 && v2[0].ok && v2[0].name == "Write",
       "mode13: truncated inside a partial \\uXXXX");
    // a COMPLETE escape at the cut must still round-trip (no over-trim)
    std::string t3 = "{\"name\": \"Write\", \"arguments\": {\"file_path\": \"g.md\", \"content\": \"caf\\u00e9";
    auto v3 = q27::parse_bare_tool_calls(t3, &pre, &tools);
    ok(v3.size() == 1 && v3[0].ok &&
           v3[0].arguments.value("content", std::string()) == "caf\u00e9",
       "mode13: a COMPLETE escape at the cut is not over-trimmed");
}

int main() {
    test_mode13_truncated_mid_escape();
    test_function_arguments_unterminated();
    test_think_mode_drift();
    test_dialect_default_keying();
    test_native_xml_dialect();
    test_mode14_tool_name_xml_dialect();
    test_mode15_name_tag_bare_args();
    test_mode16_and_15_variants();
    json tools = json::array();
    tools.push_back(tool("Write", {{"content", true}, {"file_path", true}}));
    tools.push_back(tool("Read", {{"file_path", true}}));

    auto call = [&](const std::string& txt) {
        std::string pre;
        return q27::parse_bare_tool_calls(txt, &pre, &tools);
    };

    // mode 10: dropped `{"name": "` opener
    {
        auto v = call("prose.\n\nRead\", \"file_path\": \"/x/y.py\"}");
        ok(v.size() == 1 && v[0].name == "Read" &&
               v[0].arguments.value("file_path", std::string()) == "/x/y.py",
           "mode10 dropped-opener");
    }
    // mode 11: raw code, unescaped inner quotes, content last
    {
        auto v = call("{\"name\": \"Write\", \"arguments\": {\"content\": \"package main\n"
                      "import \"fmt\"\nfunc main(){ fmt.Println(\"hi\") }\n\"}}");
        ok(v.size() == 1 && v[0].name == "Write" &&
               v[0].arguments.value("content", std::string()).find("fmt.Println") !=
                   std::string::npos,
           "mode11 raw-content-last");
    }
    // mode 11: inner []string{"a","b"} + a scalar AFTER content
    {
        auto v = call("{\"name\": \"Write\", \"arguments\": {\"content\": \"a := []string{\"x\", "
                      "\"y\"}\nfmt.Println(a)\n\", \"file_path\": \"m.go\"}}");
        ok(v.size() == 1 && v[0].name == "Write" &&
               v[0].arguments.value("file_path", std::string()) == "m.go" &&
               v[0].arguments.value("content", std::string()).find("[]string") !=
                   std::string::npos,
           "mode11 inner-braces + scalar-after");
    }
    // mode 11: scalar BEFORE content
    {
        auto v = call("{\"name\": \"Write\", \"arguments\": {\"file_path\": \"m.go\", \"content\": "
                      "\"func f(){ s := \"hi\" }\n\"}}");
        ok(v.size() == 1 && v[0].arguments.value("file_path", std::string()) == "m.go" &&
               !v[0].arguments.value("content", std::string()).empty(),
           "mode11 scalar-before-content");
    }
    // mode 11 refinement (issue #4, 2026-07-20, @chaudhryfaisal): content is
    // MOSTLY-escaped JSON (\n \" all escaped) with ONE sparse escape error
    // (\"x" -- bare closing quote) AND a trailing </tool_call>. json().dump()
    // would double-escape the already-escaped body, and the trailing tag broke
    // the reconstruction's parse -> UN-RESCUED. minimal-escape + first-balanced-
    // object recover it with the CORRECT content ("fmt" quotes preserved).
    {
        auto v = call("{\"name\": \"Write\", \"arguments\": {\"content\":\"package main\\n"
                      "import \\\"fmt\\\"\\nvar s = \\\"x\"\\n\",\"file_path\":\"m.go\"}}\n"
                      "</tool_call>");
        ok(v.size() == 1 && v[0].name == "Write" &&
               v[0].arguments.value("file_path", std::string()) == "m.go" &&
               v[0].arguments.value("content", std::string()).find("\"fmt\"") != std::string::npos,
           "mode11 mostly-escaped content + trailing tag");
    }
    // mode 12: unquoted tool-name value {"name": Read, "arguments": {...}}
    // (club-3090 cli-40: the model emitted {"name": bash, ...} and the whole
    // call went UN-RESCUED -> agent turn stopped at turnsUsed=0).
    {
        auto v = call("{\"name\": Read, \"arguments\": {\"file_path\": \"/x/y.py\"}}");
        ok(v.size() == 1 && v[0].name == "Read" &&
               v[0].arguments.value("file_path", std::string()) == "/x/y.py",
           "mode12 unquoted-name");
    }
    // mode 12b: dropped OPENING quote of the name value -> {"name": Read", ...}
    // (thunderdome 2026-07-20: model emitted {"name": read", ...} -- bareword +
    // stray closing quote; naive quoting would make "Read"" (invalid)).
    {
        auto v = call("{\"name\": Read\", \"arguments\": {\"file_path\": \"/x/y.py\"}}");
        ok(v.size() == 1 && v[0].name == "Read" &&
               v[0].arguments.value("file_path", std::string()) == "/x/y.py",
           "mode12b dropped-opening-quote");
    }
    // mode 12 negative: an unquoted name that is NOT a registered tool must be
    // left untouched (never quote arbitrary barewords).
    {
        auto v = call("{\"name\": notatool, \"arguments\": {\"file_path\": \"/x\"}}");
        ok(v.empty(), "mode12 unknown unquoted-name rejected");
    }
    // negative: well-formed call recovers via the normal path (not a drift mode)
    {
        auto v =
            call("{\"name\": \"Write\", \"arguments\": {\"file_path\": \"a.txt\", \"content\": "
                 "\"hello\"}}");
        ok(v.size() == 1 && v[0].arguments.value("content", std::string()) == "hello",
           "wellformed via normal path");
    }
    // negative: prose JSON with an unregistered "name" must NOT recover
    {
        auto v = call("config: {\"name\": \"my-app\", \"version\": \"1.0\"} shipped.");
        ok(v.empty(), "prose-unknown-name rejected");
    }
    // fence-skip (thunderdome 2026-07-20): a COMPLETE, well-formed call inside a
    // ```fenced``` block is a displayed example / echoed injection, NOT a call
    // the model is making -> must not recover (prose-to-execution guard).
    {
        auto v = call("Here is how it works:\n```json\n{\"name\": \"Read\", \"arguments\": "
                      "{\"file_path\": \"/etc/passwd\"}}\n```\nThat is the format.");
        ok(v.empty(), "fence-skip: fenced example not recovered");
    }
    // but a write whose CONTENT contains fences still recovers (its ``` are
    // after the call's opener, so the guard -- which looks only before -- ignores them)
    {
        auto v = call("{\"name\": \"Write\", \"arguments\": {\"content\": \"# doc\\n```go\\nx := 1\\n"
                      "```\\n\", \"file_path\": \"/x.md\"}}");
        ok(v.size() == 1 && v[0].name == "Write" &&
               v[0].arguments.value("file_path", std::string()) == "/x.md",
           "fence-skip: write w/ fenced content still recovers");
    }

    printf(failures ? "\nDRIFT TESTS: %d FAIL\n" : "\nDRIFT TESTS: all pass\n", failures);
    return failures ? 1 : 0;
}
