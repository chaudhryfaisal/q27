// Fuzz the tool-call parser with MODEL-CONTROLLED bytes. See docs/SECURITY.md.
// Boyd Kane's threat model: the LLM chooses every byte the inference engine's
// parser sees, so that parser is attack surface. q27 has no eval() (the
// CVE-2025-9141 class does not apply -- it emits structured calls that a
// separate sandboxed client executes), but it has ~109 substr() calls and 72
// span computations over strings the model authored, and a 22-mode recovery
// chain designed to be maximally forgiving. Forgiving parsers fabricate.
// Targets every entry point the server reaches with generated text.
#include "api_common.h"
#include <cstdint>
#include <string>
using json = nlohmann::json;

static const json& tools() {
    static json t = [] {
        json body = json::parse(R"({"tools":[
          {"name":"Read","input_schema":{"type":"object","properties":{"file_path":{"type":"string"}}}},
          {"name":"Bash","input_schema":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"}}}},
          {"name":"Write","input_schema":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}}}}]})");
        return q27::anthropic_tools_json(body);
    }();
    return t;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size > 64u << 10) return 0;
    const std::string s((const char*)data, size);
    std::string prefix, residual;
    // 1. the bare/drift chain (all 22 modes)
    for (bool repair : {false, true}) {
        auto calls = q27::parse_bare_tool_calls(s, &prefix, &tools(), true, repair, &residual);
        for (const auto& c : calls) {
            // spans must be sane: the server does raw.substr(cursor, begin-cursor)
            if (c.source_begin != std::string::npos) {
                if (c.source_begin > s.size() || c.source_end > s.size() ||
                    c.source_end < c.source_begin) __builtin_trap();
            }
        }
    }
    // 2. the strict wrapped parser
    q27::parse_tool_call(q27::strip_ws2(s));
    // 3. the splitter, then the ordered resolver (the real handler path)
    {
        q27::StreamSplitter sp;
        std::vector<std::pair<q27::StreamSplitter::Chan, std::string>> segs;
        for (size_t i = 0; i < s.size(); i += 7)
            for (auto& x : sp.feed(s.substr(i, 7))) {
                if (!segs.empty() && segs.back().first == x.first) segs.back().second += x.second;
                else segs.push_back(x);
            }
        for (auto& x : sp.flush()) {
            if (!segs.empty() && segs.back().first == x.first) segs.back().second += x.second;
            else segs.push_back(x);
        }
        q27::resolve_ordered_tool_segments(segs, &tools(), true,
                                           [](const std::string&, size_t) { return true; });
    }
    // 4. the streaming holdback (TEXT and THINK)
    {
        std::set<std::string> names{"Read", "Bash", "Write"};
        q27::BareToolTextHoldback hb;
        std::string vis;
        auto emit = [&](const std::string& t) { vis += t; };
        auto classify = [&](const std::string& src, bool rep, auto&& visible) {
            std::string p, r;
            auto cs = q27::parse_bare_tool_calls(src, &p, &tools(), true, rep, &r);
            if (cs.empty()) return q27::BareToolCandidateResult{};
            size_t cur = 0;
            for (auto& c : cs) {
                if (c.source_begin == std::string::npos || c.source_begin < cur ||
                    c.source_end > src.size()) break;
                visible(src.substr(cur, c.source_begin - cur));
                cur = c.source_end;
            }
            visible(src.substr(cur));
            return q27::BareToolCandidateResult{true, true};
        };
        for (size_t i = 0; i < s.size(); i += 5) hb.route(s.substr(i, 5), names, emit, classify);
        hb.finish(true, names, emit, classify);
    }
    // 5. the unclosed-tail recovery
    q27::recover_unclosed_tool_tail(s, &tools(), [](const std::string&) {},
                                    [](const q27::ToolCall&) { return true; });
    return 0;
}
