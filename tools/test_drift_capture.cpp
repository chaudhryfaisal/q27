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

int main() {
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
