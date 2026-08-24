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
}

int main() {
    test_shape_hash_ignores_values_not_structure();
    test_no_value_bytes_survive();
    test_no_value_bytes_survive_any_channel();
    test_dropped_opener_keeps_keys();
    test_redacts_xml_values();
    test_preserves_dialect_inside_values();
    test_redacts_json_string_values();
    test_placeholder_type_from_key();
    test_length_class_recorded();
    if (fails) { printf("%d FAILURE(S)\n", fails); return 1; }
    printf("DRIFT CAPTURE: all pass\n");
    return 0;
}
