// CPU unit tests for src/drift_capture.h: the capture-time redaction that
// makes Q27_DRIFT_CORPUS safe to commit from a public repo. Pure header
// logic, no CUDA/engine dependency.
//
// Build+run: g++ -std=c++17 -I src tools/test_drift_capture.cpp -o build/test_drift_capture && ./build/test_drift_capture
#include "drift_capture.h"
#include <cstdio>
#include <cstring>
static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("  FAIL %s:%d %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

static void test_redacts_xml_values() {
    const std::string in =
        "<tool_call>\n<function=Read>\n<parameter=file_path>\n/home/gabe/secret.txt\n"
        "</parameter>\n</function>\n</tool_call>";
    const std::string out = q27::redact_drift(in);
    // framing and keys survive verbatim
    CHECK(out.find("<function=Read>") != std::string::npos);
    CHECK(out.find("<parameter=file_path>") != std::string::npos);
    // the value does not
    CHECK(out.find("/home/gabe/secret.txt") == std::string::npos);
    CHECK(out.find("PATH_1") != std::string::npos);
}

static void test_preserves_dialect_inside_values() {
    // a closer inside a value is STRUCTURE -- it must survive redaction,
    // because it is the shape that broke the closer-bounding fix
    const std::string in =
        "<function=Write>\n<parameter=content>\nfoo </function> bar\n</parameter>\n</function>";
    const std::string out = q27::redact_drift(in);
    CHECK(out.find("</function> ") != std::string::npos);
}

static void test_redacts_json_string_values() {
    const std::string in = R"({"name":"Read","arguments":{"file_path":"/etc/shadow"}})";
    const std::string out = q27::redact_drift(in);
    CHECK(out.find("\"name\":\"Read\"") != std::string::npos);   // tool name survives
    CHECK(out.find("\"file_path\"") != std::string::npos);        // key survives
    CHECK(out.find("/etc/shadow") == std::string::npos);          // value does not
}

static void test_fences_inside_a_string_value_stay_inside_it() {
    // a README written through a JSON call: fences are kept (the display
    // lexer reads them anywhere) but the string must not be cut at the first
    // backtick, or the rest of the content is scanned as loose JSON
    const std::string in =
        "{\"name\":\"Write\",\"arguments\":{\"content\":\"# Title\\n```py\\nx = secret\\n```\\nend\",\"file_path\":\"/w/README.md\"}}";
    const std::string out = q27::redact_drift(in);
    CHECK(out.find("secret") == std::string::npos);
    CHECK(out.find("Title") == std::string::npos);
    CHECK(out.find("```") != std::string::npos);
    CHECK(out.find("\",\"file_path\":\"PATH_") != std::string::npos);   // the string closed where it should
    CHECK(out.find("README") == std::string::npos);
    // and a fence inside an XML value: one run per stretch, no leak
    const std::string xml = "<parameter=content>\n# T\n```\nx = secret\n```\n</parameter>";
    const std::string xo = q27::redact_drift(xml);
    CHECK(xo.find("secret") == std::string::npos);
    CHECK(xo.find("</parameter>") != std::string::npos);
}

static void test_json_literals_stay_valid_json() {
    // the corpus is replayed through the parser (Phase 2); a bare boolean
    // or number redacted to TEXT_n is no longer valid JSON and the strict
    // parser refuses a call the live server accepted
    const std::string in =
        R"({"name":"Edit","arguments":{"file_path":"/w/a.py","old_string":"x","new_string":"y","replace_all":true,"timeout":120000,"n":null,"v":false}})";
    const std::string out = q27::redact_drift(in);
    CHECK(out.find("\"replace_all\":true") != std::string::npos);
    CHECK(out.find("\"timeout\":0") != std::string::npos);
    CHECK(out.find("\"n\":null") != std::string::npos);
    CHECK(out.find("\"v\":false") != std::string::npos);
    CHECK(out.find("120000") == std::string::npos);
    CHECK(nlohmann::json::accept(out));   // still JSON
    // a bare word that is not a literal is still redacted
    CHECK(q27::redact_drift(R"({"name":"Edit","arguments":{"x":secret}})").find("secret") == std::string::npos);
}

static void test_novel_tags_survive_as_structure() {
    // the first novel shape real traffic produced: tags the catalogue never
    // named. The element names are the shape; the value is still redacted
    // and the tool name is kept only because it is declared.
    const q27::DriftNames names = {"Read", "Bash"};
    const std::string in =
        "\n\n<tool_use>\n<tool>\n<parameter_name>\n<parameter_name>Read\n</parameter>\n"
        "<parameter=file_path>\n/workspace/pylint/lint/run.py\n";
    const std::string out = q27::redact_drift(in, &names);
    CHECK(out.find("<tool_use>\n<tool>\n<parameter_name>\n<parameter_name>Read\n</parameter>") != std::string::npos);
    CHECK(out.find("/workspace") == std::string::npos);
    CHECK(out.find("PATH_") != std::string::npos);
    // an undeclared name after a name tag is still a placeholder
    CHECK(q27::redact_drift("<parameter_name>Alice\n</parameter>", &names).find("Alice") == std::string::npos);
    // a tag with attributes is content, and so is a comparison in code
    CHECK(q27::redact_drift("<parameter=content>\n<div class=\"secret\">x</div>\n</parameter>", &names).find("secret") == std::string::npos);
    CHECK(q27::redact_drift("<parameter=command>\nif (a < secret) run\n</parameter>", &names).find("secret") == std::string::npos);
}

static void test_placeholder_type_from_key() {
    const std::string in =
        "<function=Bash>\n<parameter=command>\nrm -rf /\n</parameter>\n"
        "<parameter=description>\nwipe\n</parameter>\n</function>";
    const std::string out = q27::redact_drift(in);
    CHECK(out.find("CODE_1") != std::string::npos);   // command -> CODE
    CHECK(out.find("TEXT_2") != std::string::npos);   // description -> TEXT
}

static void test_length_class_recorded() {
    std::string big = "<function=Write>\n<parameter=content>\n" + std::string(5000, 'x') +
                      "\n</parameter>\n</function>";
    const std::string out = q27::redact_drift(big);
    CHECK(out.find("TEXT_1:big") != std::string::npos);  // >4KB marked
    CHECK(out.size() < 500);                             // and not carried
}

static void test_dropped_opener_keeps_keys() {
    // mode 10: the leading `{"` is gone, so every later quote is off by one.
    // The keys must still read as keys or the shape collapses into prose.
    const std::string in = "name\": \"Read\", \"arguments\": {\"file_path\": \"/w/x\"}}";
    const std::string out = q27::redact_drift(in);
    CHECK(out.find("name\": \"Read\"") != std::string::npos);
    CHECK(out.find("\"file_path\": \"PATH_1\"") != std::string::npos);
    CHECK(out.find("/w/x") == std::string::npos);
    // the other dropped-opener shape: `{"name": "` is gone and the turn opens
    // on the tool name itself (tools/fuzz_seeds/mode10_dropped_opener)
    const std::string in2 = "Read\", \"file_path\": \"/a\"}";
    const std::string out2 = q27::redact_drift(in2);
    CHECK(out2 == "Read\", \"file_path\": \"PATH_1\"}");
    // and with a declared set, an undeclared leading identifier is not a name
    const q27::DriftNames names = {"Read", "Write"};
    const std::string out3 = q27::redact_drift("Alice\", \"file_path\": \"/a\"}", &names);
    CHECK(out3.find("Alice") == std::string::npos);
    CHECK(out3.find("NAME_1\", \"file_path\"") != std::string::npos);
}

static void test_no_value_bytes_survive() {
    // every distinctive value token must be absent from the output
    const char* secrets[] = {"sk-ant-api03-XXXX", "/home/gabe/.ssh/id_ed25519",
                             "hunter2", "AKIAIOSFODNN7EXAMPLE"};
    for (const char* s : secrets) {
        const std::string in = std::string("<function=Bash>\n<parameter=command>\necho ") +
                               s + "\n</parameter>\n</function>";
        const std::string out = q27::redact_drift(in);
        CHECK(out.find(s) == std::string::npos);
        // and no 8-byte run of it either, in case of partial copy
        for (size_t i = 0; i + 8 <= strlen(s); i++)
            CHECK(out.find(std::string(s + i, 8)) == std::string::npos);
    }
}

static bool leaks(const std::string& out, const std::string& s) {
    if (out.find(s) != std::string::npos) return true;
    for (size_t i = 0; i + 8 <= s.size(); i++)
        if (out.find(s.substr(i, 8)) != std::string::npos) return true;
    return false;
}

// The same property over every channel the redactor knows: JSON values, prose,
// a mode-11 raw value whose inner quote desyncs the string scanner, tool-name
// positions (gated on the declared set), reasoning, fences, hallucinated
// results, and a value with dialect markup inside it.
static void test_no_value_bytes_survive_any_channel() {
    const q27::DriftNames names = {"Read", "Write", "Bash"};
    const char* secrets[] = {"sk-ant-api03-XXXX", "/home/gabe/.ssh/id_ed25519",
                             "hunter2", "AKIAIOSFODNN7EXAMPLE", "def leak(): pass"};
    for (const char* sc : secrets) {
        const std::string s = sc;
        const std::string shapes[] = {
            "{\"name\":\"Bash\",\"arguments\":{\"command\":\"echo " + s + "\"}}",
            "Now I will use " + s + " to authenticate.\n<tool_call>{\"name\":\"Read\","
                "\"arguments\":{\"file_path\":\"" + s + "\"}}</tool_call>\nDone with " + s,
            "{\"name\":\"Write\",\"arguments\":{\"file_path\":\"/w/c.json\",\"content\":\"x = \"" +
                s + "\"\n\"}}",
            "{\"name\":\"Write\",\"arguments\":{\"content\":\"{\"token\": \"" + s + "\"}\"}}",
            "{\"name\":\"" + s + "\",\"arguments\":{}}",
            "<function=" + s + ">\n<parameter=x>\n1\n</parameter>\n</function>",
            "<tool_name>" + s + "</tool_name>\n<parameter=arguments>\n{\"a\":\"b\"}",
            s + "\", \"file_path\": \"/a\"}",
            "<think>\nremember " + s + "\n</think>\n{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/a\"}}",
            "```\n" + s + "\n```\nuse `" + s + "` now",
            "<function=Write>\n<parameter=content>\n" + s + " </function> " + s + "\n</parameter>\n</function>",
            "<result>\n<output>\n" + s + "\n</output>\n</result>",
            "<parameter=file_path>" + s + "</parameter>",   // no opener at all
            "<parameter=" + s,                               // truncated opener
            "\"" + s,                                        // unterminated string
        };
        for (const auto& in : shapes) {
            const std::string out = q27::redact_drift(in, &names);
            if (leaks(out, s)) {
                printf("  LEAK of %s in: %s\n    -> %s\n", sc, in.c_str(), out.c_str());
                fails++;
            }
        }
    }
}

static void test_shape_hash_ignores_values_not_structure() {
    const std::string a = "<function=Read>\n<parameter=file_path>\n/a\n</parameter>\n</function>";
    const std::string b = "<function=Read>\n<parameter=file_path>\n/b\n</parameter>\n</function>";
    const std::string c = "<function=Read>\n<parameter=file_path>\n/a\n</function>";  // closer dropped
    CHECK(q27::shape_hash(q27::redact_drift(a)) == q27::shape_hash(q27::redact_drift(b)));
    CHECK(q27::shape_hash(q27::redact_drift(a)) != q27::shape_hash(q27::redact_drift(c)));
    // stable across runs and builds: a dedup key has to mean the same thing
    // in tomorrow's corpus_dedup.py as in today's capture
    CHECK(q27::shape_hash("") == 0xcbf29ce484222325ull);           // FNV-1a offset basis
    CHECK(q27::shape_hash("a") == 0xaf63dc4c8601ec8cull);          // FNV-1a("a")
    // the key is the skeleton: size class and prose line count are not shape
    CHECK(q27::shape_key("<parameter=content>\nTEXT_1:big\n</parameter>") ==
          "<parameter=content>\nTEXT\n</parameter>");
    CHECK(q27::shape_hash("PROSE_1:ml\n<tool_call>") == q27::shape_hash("PROSE_1\n<tool_call>"));
    CHECK(q27::shape_hash("TEXT_1:big") == q27::shape_hash("TEXT_1"));
    CHECK(q27::shape_hash("PATH_1 PATH_2") == q27::shape_hash("PATH_1 PATH_9"));
    // but a raw newline inside a value is (mode 5 / 11)
    CHECK(q27::shape_hash("{\"content\":\"TEXT_1:ml\"}") != q27::shape_hash("{\"content\":\"TEXT_1\"}"));
    // and a preamble is (prefix handling)
    CHECK(q27::shape_hash("PROSE_1\n<tool_call>") != q27::shape_hash("<tool_call>"));
    // an ordinary identifier that merely looks like a placeholder is untouched
    CHECK(q27::shape_key("\"MY_KEY_2\"") == "\"MY_KEY_2\"");
    // indentation of a value's first line is not shape; the newline is
    CHECK(q27::shape_hash("<parameter=new_string>\n      CODE_2:ml\n</parameter>") ==
          q27::shape_hash("<parameter=new_string>\n  CODE_2:ml\n</parameter>"));
    CHECK(q27::shape_hash("<parameter=new_string>\nCODE_2\n</parameter>") !=
          q27::shape_hash("<parameter=new_string> CODE_2 </parameter>"));
    // inline code spans inside a value are content: the placeholders they
    // split merge, and a README with nine mentions is the shape of one with two
    CHECK(q27::shape_key("TEXT_1:ml `TEXT_2` TEXT_3 `TEXT_4` TEXT_5:ml") == "TEXT:ml");
    CHECK(q27::shape_key("CODE_2:ml`CODE_3`CODE_4:ml`CODE_5`CODE_6:ml") == "CODE:ml");
    CHECK(q27::shape_key("TEXT_1 `TEXT_2` TEXT_3") == "TEXT");
    // but a block fence is display context and stays
    CHECK(q27::shape_key("PROSE_1\n```\nTEXT_2\n```\nPROSE_3") == "PROSE\n```\nTEXT\n```\nPROSE");
    // and different types do not merge
    CHECK(q27::shape_key("PATH_1 TEXT_2") == "PATH TEXT");
}

static std::string hex16(uint64_t h) {
    char b[17]; snprintf(b, sizeof b, "%016llx", (unsigned long long)h); return b;
}

static void test_record_format() {
    using json = nlohmann::json;
    const std::string text =
        "<tool_call>\n<function=Read>\n<parameter=file_path>\n/x/y\n</parameter>\n</function>\n</tool_call>";
    q27::DriftContext ctx;
    const std::string line = q27::format_drift_record(text, "recovered:1", ctx, nullptr, 0);
    CHECK(line.find('\n') == std::string::npos);                 // one JSONL line
    CHECK(line.find("/x/y") == std::string::npos);               // no raw value on the line
    json j = json::parse(line);
    const std::string redacted = q27::redact_drift(text);
    CHECK(j["redacted"] == redacted);                            // exactly redact_drift's output
    CHECK(j["id"] == hex16(q27::shape_hash(redacted)));
    CHECK(j["shape"] == "");                                     // labelled later
    CHECK(j["outcome"] == "recovered:1");
    CHECK(j["bytes"] == text.size());
    CHECK(j["ts"] == "1970-01-01T00:00:00Z");
    CHECK(j["tags"] == json::array({"xml"}));                    // wrapped, not in think
    CHECK(json::parse(j.dump()) == j);                           // round-trips

    ctx.in_think = true;
    json j2 = json::parse(q27::format_drift_record(
        R"({"name":"Read","arguments":{"file_path":"/a"}})", "unrescued", ctx, nullptr, 0));
    CHECK(j2["tags"] == json::array({"json", "no_wrapper", "in_think"}));
    // pretty-printed JSON is still json
    CHECK(q27::drift_tags("{ \"name\" : \"Read\" }", q27::DriftContext{}) == std::vector<std::string>({"json", "no_wrapper"}));
    CHECK(q27::drift_tags("the word name: here", q27::DriftContext{}) == std::vector<std::string>({"no_wrapper"}));

    // a TOOL-segment body reaches the parser with its wrapper already stripped
    // by the splitter; the caller says so and it must not read as no_wrapper
    q27::DriftContext wrapped; wrapped.wrapped = true;
    json j3 = json::parse(q27::format_drift_record(
        "<function=Read>\n<parameter=file_path>\n/a\n</parameter>\n</function>", "strict", wrapped, nullptr, 0));
    CHECK(j3["tags"] == json::array({"xml"}));

    // the declared set reaches the redactor
    const q27::DriftNames names = {"Read"};
    json j4 = json::parse(q27::format_drift_record(
        R"({"name":"Alice","arguments":{}})", "unrescued", ctx, &names, 0));
    CHECK(j4["redacted"].get<std::string>().find("Alice") == std::string::npos);
}

static void test_write_appends_jsonl() {
    const char* path = "build/test_drift_capture.jsonl";
    remove(path);
    q27::DriftContext ctx;
    q27::write_drift_record(path, "<function=Read>\n<parameter=file_path>\n/a\n</parameter>\n</function>", "recovered:1", ctx, nullptr);
    q27::write_drift_record(path, R"({"name":"Bash","arguments":{"command":"ls"}})", "strict", ctx, nullptr);
    q27::write_drift_record(path, "", "strict", ctx, nullptr);   // empty text: no record
    FILE* f = fopen(path, "rb");
    CHECK(f != nullptr);
    if (!f) return;
    std::string all; char buf[4096]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) all.append(buf, n);
    fclose(f);
    size_t lines = 0;
    for (size_t at = 0; (at = all.find('\n', at)) != std::string::npos; at++) lines++;
    CHECK(lines == 2);
    CHECK(all.back() == '\n');
    const std::string first = all.substr(0, all.find('\n'));
    CHECK(nlohmann::json::parse(first)["outcome"] == "recovered:1");
    remove(path);
}

int main() {
    test_record_format();
    test_write_appends_jsonl();
    test_shape_hash_ignores_values_not_structure();
    test_no_value_bytes_survive();
    test_no_value_bytes_survive_any_channel();
    test_dropped_opener_keeps_keys();
    test_redacts_xml_values();
    test_preserves_dialect_inside_values();
    test_redacts_json_string_values();
    test_placeholder_type_from_key();
    test_novel_tags_survive_as_structure();
    test_json_literals_stay_valid_json();
    test_fences_inside_a_string_value_stay_inside_it();
    test_length_class_recorded();
    if (fails) { printf("%d FAILURE(S)\n", fails); return 1; }
    printf("DRIFT CAPTURE: all pass\n");
    return 0;
}
