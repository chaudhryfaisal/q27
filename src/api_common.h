// Shared prompt construction + tool-call parsing for the q27 API endpoints.
// Qwopus (qwen35) tool protocol, from the GGUF chat template:
//   system preamble lists tools as JSON inside <tools>...</tools>
//   model emits  <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>
//   results go back as user content wrapped in <tool_response>...</tool_response>
#pragma once
#include <string>
#include <vector>

#include "../third_party/json.hpp"
#include "stream_split.h"

namespace q27 {
using json = nlohmann::json;

struct Msg {
    std::string role;     // system | user | assistant
    std::string content;  // flattened text (think blocks already reconstructed)
};

// Tools preamble, verbatim structure from the chat template. `tools` entries
// must already be in {"type":"function","function":{...}} shape.
inline std::string tools_preamble(const json& tools) {
    std::string s = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (auto& t : tools) s += "\n" + t.dump();
    s += "\n</tools>\n\nFor each function call, return a JSON object with the function name "
         "and arguments inside <tool_call></tool_call> tags:\n<tool_call>\n{\"name\": "
         "<function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\n<IMPORTANT>\n"
         "- Required parameters MUST be specified.\n- You may provide optional reasoning "
         "before the function call, but never after it.\n- If no function call is needed, "
         "answer normally and do not mention the tool interface.\n</IMPORTANT>";
    return s;
}

// Build the full ChatML prompt string. If tools are present they are merged
// into the (first) system message per the template's merged_system behavior.
inline std::string chatml_prompt(const std::vector<Msg>& msgs, const json& tools) {
    std::string p;
    size_t start = 0;
    std::string sys;
    if (!msgs.empty() && msgs[0].role == "system") { sys = msgs[0].content; start = 1; }
    if (tools.is_array() && !tools.empty()) {
        p += "<|im_start|>system\n" + tools_preamble(tools);
        if (!sys.empty()) p += "\n\n" + sys;
        p += "<|im_end|>\n";
    } else if (!sys.empty()) {
        p += "<|im_start|>system\n" + sys + "<|im_end|>\n";
    }
    for (size_t i = start; i < msgs.size(); i++)
        p += "<|im_start|>" + msgs[i].role + "\n" + msgs[i].content + "<|im_end|>\n";
    p += "<|im_start|>assistant\n";
    return p;
}

inline std::string tool_call_text(const std::string& name, const json& args) {
    return "<tool_call>\n{\"name\": \"" + name + "\", \"arguments\": " + args.dump() +
           "}\n</tool_call>";
}

inline std::string tool_response_text(const std::string& out) {
    return "<tool_response>\n" + out + "\n</tool_response>";
}

// Parsed model tool call. `ok` false if the JSON was malformed (raw kept).
struct ToolCall {
    bool ok = false;
    std::string name;
    json arguments;
    std::string raw;
};

inline ToolCall parse_tool_call(const std::string& seg) {
    ToolCall tc;
    tc.raw = seg;
    try {
        json j = json::parse(seg);
        tc.name = j.value("name", std::string());
        tc.arguments = j.contains("arguments") ? j["arguments"] : json::object();
        if (tc.arguments.is_string()) // some models double-encode
            tc.arguments = json::parse(tc.arguments.get<std::string>());
        tc.ok = !tc.name.empty();
    } catch (...) { tc.ok = false; }
    return tc;
}

inline std::string strip_ws2(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

} // namespace q27
