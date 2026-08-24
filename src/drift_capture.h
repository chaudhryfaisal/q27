// Drift-corpus capture: redaction, shape hashing and record formatting for
// Q27_DRIFT_CORPUS. Pure functions, no CUDA, unit-tested by
// tools/test_drift_capture.cpp. Design: docs/plans/2026-08-24-parser-normalize-then-parse-design.md
//
// WHY THIS EXISTS. q27 is a public repo and the corpus is real Claude Code
// session content: file paths, source, tool arguments, secrets echoed into
// shell commands. Nothing scrubs the corpus later, so redaction has to happen
// here, before a byte reaches disk.
//
// WHAT SURVIVES. The parser cares about structure, not values, so structure
// is kept verbatim and everything else becomes a typed placeholder:
//
//   kept      dialect framing (<tool_call>, <function=NAME>, <parameter=KEY>,
//             </parameter>, </function>, <tool_name>, <think>, <result>...),
//             JSON punctuation and keys, declared tool names, markdown fences
//             and backticks (the display-context rules hang off them), and the
//             whitespace around a value (the dialect parser is newline-shaped).
//   replaced  every other byte run, as TYPE_N where TYPE is PATH / CODE / TEXT
//             (from the key the value sits under), PROSE (bytes outside any
//             call structure) or NAME (a tool name that is not declared).
//             N is 1-based, in order of appearance. `:ml` marks a multi-line
//             value, `:big` one over 4 KB; neither carries the bytes.
//
// The one exception that matters: dialect markup INSIDE a value is structure.
// A `</function>` sitting in a content value is the exact shape that broke the
// closer-bounding fix, so it is passed through and the value continues around
// it. Only a definite boundary (</parameter>, a new <parameter= or <function=,
// the wrapper tokens) ends a value.
//
// Prose is redacted too, even though the plan only names values: a visible
// turn routinely says "I'll now edit /home/x/y.py", and the definition of done
// is that no path, code or content byte from the session appears in a record.
// Over-redaction is the safe direction; nothing the parser needs lives in prose.
//
// Residual, by design: JSON *keys* are kept, so a raw unescaped value that is
// itself JSON (mode 11 writing a config file) shows its keys. Values under
// those keys are still replaced. Tool names are kept only when declared;
// when no declared set is supplied (unit tests) any identifier-like name is
// kept.
#pragma once

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "../third_party/json.hpp"

namespace q27 {

using DriftNames = std::set<std::string>;

namespace drift_detail {

inline bool is_ws(char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }
inline bool is_name_char(char c) {
    return std::isalnum((unsigned char)c) || c == '_' || c == '-' || c == '.' || c == ':';
}
inline std::string lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}
inline bool ends_with(const std::string& s, const char* suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// Placeholder type from the key a value sits under. Keys are schema, so this
// is a hint for the labeller, not a security boundary: an unknown key is TEXT.
inline const char* type_for_key(const std::string& key_in) {
    const std::string k = lower(key_in);
    if (k == "path" || ends_with(k, "path") || k == "cwd" || k == "directory") return "PATH";
    if (k == "command" || k == "cmd" || k == "code" || k == "script" ||
        k == "old_string" || k == "new_string" || ends_with(k, "_command")) return "CODE";
    return "TEXT";
}

inline bool identifier_like(const std::string& s) {
    if (s.empty() || s.size() > 64) return false;
    for (char c : s) if (!is_name_char(c)) return false;
    return true;
}

// What may survive as a JSON key: a schema-shaped identifier. Narrower than a
// tool name on purpose -- a key is kept verbatim with no declared set to check
// it against, so the shape itself is the only gate. No '-', '.', ':' means a
// `sk-ant-...` token or a path can never pass as one.
inline bool key_like(const std::string& s) {
    if (s.empty() || s.size() > 64) return false;
    if (!(std::isalpha((unsigned char)s[0]) || s[0] == '_')) return false;
    for (char c : s) if (!(std::isalnum((unsigned char)c) || c == '_')) return false;
    return true;
}

// A tool name survives only when it is one the request declared (case-
// insensitive, so a drifted `read` for `Read` still shows as a name). With no
// declared set, identifier-like is enough -- that is the unit-test contract.
inline bool name_allowed(const std::string& s, const DriftNames* names) {
    if (!identifier_like(s)) return false;
    if (!names) return true;
    if (names->count(s)) return true;
    const std::string ls = lower(s);
    for (const auto& n : *names) if (lower(n) == ls) return true;
    return false;
}

// Tags kept verbatim as <tag> / </tag>. Everything the drift catalogue has
// ever keyed off, plus the hallucinated-result tags the parser refuses on.
inline bool known_tag(const std::string& t) {
    static const char* const kTags[] = {
        "tool_call", "tool_calls", "function", "parameter", "tool_name", "think",
        "result", "output", "content", "name", "arguments", "invoke", "tool"};
    for (const char* k : kTags) if (t == k) return true;
    return false;
}

struct Redactor {
    const std::string& s;
    const DriftNames* names;
    std::string out;
    size_t counter = 0;

    std::string run;          // non-token bytes waiting for a placeholder
    std::string xml_key;      // non-empty while inside an XML parameter value
    std::string json_key;     // last JSON key seen; types the value after it
    bool name_next = false;   // the next value is a tool name
    bool last_was_string = false;
    int json_depth = 0;

    Redactor(const std::string& in, const DriftNames* n) : s(in), names(n) {
        out.reserve(in.size() < 4096 ? in.size() : 4096);
    }

    bool in_xml_value() const { return !xml_key.empty(); }
    // JSON structure is recognised outside XML values, and inside the one XML
    // value that carries JSON by convention (mode 14: <parameter=arguments>).
    bool json_active() const { return !in_xml_value() || lower(xml_key) == "arguments"; }

    const char* run_type() const {
        if (name_next) return "NAME";
        if (in_xml_value() && !(lower(xml_key) == "arguments" && !json_key.empty()))
            return type_for_key(xml_key);
        if (!json_key.empty()) return type_for_key(json_key);
        return json_depth > 0 ? "TEXT" : "PROSE";
    }

    void placeholder(const std::string& core, const char* type) {
        if (std::strcmp(type, "NAME") == 0 && name_allowed(core, names)) { out += core; return; }
        counter++;
        out += type;
        out += '_';
        out += std::to_string(counter);
        if (core.size() > 4096) out += ":big";
        else if (core.find('\n') != std::string::npos) out += ":ml";
    }

    // Emit the pending run: leading/trailing whitespace verbatim (the value's
    // shape), the core as one placeholder.
    void flush() {
        if (run.empty()) return;
        size_t b = 0, e = run.size();
        while (b < e && is_ws(run[b])) b++;
        while (e > b && is_ws(run[e - 1])) e--;
        out.append(run, 0, b);
        if (e > b) {
            placeholder(run.substr(b, e - b), run_type());
            name_next = false;
            last_was_string = false;
        }
        out.append(run, e, std::string::npos);
        run.clear();
    }

    void token(const std::string& t) {
        flush();
        out += t;
        last_was_string = false;
    }

    // Does a dialect token or fence start at i? Used to bound JSON strings so
    // an unterminated quote cannot swallow the call that follows it.
    bool hard_boundary_at(size_t i) const {
        if (s[i] == '`') return true;
        if (s[i] != '<') return false;
        static const char* const kStarts[] = {"<tool_call", "</tool_call", "<function", "</function",
                                              "<parameter", "</parameter", "<tool_name", "</tool_name",
                                              "<think>", "</think>"};
        for (const char* k : kStarts)
            if (s.compare(i, std::strlen(k), k) == 0) return true;
        return false;
    }

    // <tag>, </tag>, <function=NAME>, <parameter=KEY>. Returns bytes consumed
    // (0 = not a tag we know; the '<' is ordinary content).
    size_t try_tag(size_t i) {
        size_t j = i + 1;
        const bool closer = j < s.size() && s[j] == '/';
        if (closer) j++;
        const size_t name_begin = j;
        while (j < s.size() && (std::islower((unsigned char)s[j]) || s[j] == '_')) j++;
        const std::string tag = s.substr(name_begin, j - name_begin);
        if (tag.empty() || !known_tag(tag)) return 0;
        if (j < s.size() && s[j] == '>') {
            token(s.substr(i, j + 1 - i));
            if (tag == "parameter" && closer) xml_key.clear();
            else if (tag == "tool_call" || tag == "tool_calls" || tag == "tool_name" ||
                     tag == "think" || tag == "invoke" || tag == "tool")
                xml_key.clear();
            // </function> deliberately leaves xml_key alone: a closer inside a
            // value is the shape we are here to record, and the value continues.
            name_next = !closer && (tag == "tool_name" || tag == "name");
            return j + 1 - i;
        }
        if (!closer && s[j] == '=' && (tag == "function" || tag == "parameter")) {
            size_t k = j + 1;
            while (k < s.size() && k - (j + 1) < 64 && s[k] != '>' && !is_ws(s[k])) k++;
            const std::string ident = s.substr(j + 1, k - (j + 1));
            const bool closed = k < s.size() && s[k] == '>';
            token(s.substr(i, j + 1 - i));            // "<function=" / "<parameter="
            if (tag == "function") {
                xml_key.clear();
                if (!ident.empty()) placeholder(ident, "NAME");
            } else if (closed) {
                out += ident;                          // keys are schema
                xml_key = ident.empty() ? std::string("_") : ident;
                json_key.clear();
            } else {
                // No '>' -- the model was cut off, or never closed the opener.
                // Whatever follows '=' is not a key we can vouch for, so it is
                // value bytes: the leak gate has a case for exactly this.
                xml_key = "_";
                json_key.clear();
                run = ident;
            }
            if (closed) { out += '>'; return k + 1 - i; }
            return k - i;
        }
        return 0;
    }

    // A JSON string starting at the quote s[i]. Keys (followed by ':') are kept;
    // values become placeholders; the quotes themselves are structure.
    size_t scan_string(size_t i) {
        size_t j = i + 1;
        bool terminated = false;
        while (j < s.size()) {
            if (s[j] == '\\' && j + 1 < s.size()) { j += 2; continue; }
            if (s[j] == '"') { terminated = true; break; }
            if (hard_boundary_at(j)) break;
            j++;
        }
        const std::string inner = s.substr(i + 1, j - (i + 1));
        size_t after = terminated ? j + 1 : j;
        size_t p = after;
        while (p < s.size() && is_ws(s[p])) p++;
        const bool is_key = terminated && p < s.size() && s[p] == ':' && key_like(inner);
        flush();
        out += '"';
        if (is_key) {
            out += inner;
            json_key = inner;
            name_next = lower(inner) == "name";
        } else {
            bool all_ws = true;
            for (char c : inner) if (!is_ws(c)) { all_ws = false; break; }
            if (all_ws) out += inner;
            else {
                placeholder(inner, name_next ? "NAME" : run_type());
                name_next = false;
            }
        }
        if (terminated) out += '"';
        last_was_string = true;
        return after - i;
    }

    // Mode 10 (dropped opener): the turn starts `Read", "file_path": ...` or
    // `name": "Read", ...` because the model lost the leading `{"name": "` or
    // `{"`. Read literally, every quote after that is off by one and the whole
    // call collapses into prose. An identifier run sitting directly against
    // `":` is a key missing its opening quote, and one against `",` is the
    // tool name missing its opening quote; keep them as such, so the shape
    // that mode exists for stays visible. The name still goes through the
    // declared-name rule, so an undeclared identifier becomes NAME_n.
    size_t try_bare_key(size_t i) {
        if (run.empty() || is_ws(run.back())) return 0;
        size_t b = 0;
        while (b < run.size() && is_ws(run[b])) b++;
        const std::string core = run.substr(b);
        if (!identifier_like(core)) return 0;
        size_t p = i + 1;
        while (p < s.size() && is_ws(s[p])) p++;
        if (p >= s.size() || (s[p] != ':' && s[p] != ',')) return 0;
        if (s[p] == ':' && !key_like(core)) return 0;
        out.append(run, 0, b);
        if (s[p] == ':') {
            out += core;
            json_key = core;
            name_next = lower(core) == "name";
        } else {
            placeholder(core, "NAME");
            json_key.clear();
            name_next = false;
        }
        run.clear();
        out += '"';
        last_was_string = true;
        return 1;
    }

    std::string operator()() {
        size_t i = 0;
        while (i < s.size()) {
            const char c = s[i];
            if (c == '<') {
                if (size_t n = try_tag(i)) { i += n; continue; }
            } else if (c == '`') {
                size_t j = i;
                while (j < s.size() && s[j] == '`') j++;
                token(s.substr(i, j - i));
                i = j;
                continue;
            } else if (json_active()) {
                if (c == '{' || c == '[') { token(std::string(1, c)); json_depth++; i++; continue; }
                if (c == '}' || c == ']') {
                    token(std::string(1, c));
                    if (json_depth > 0) json_depth--;
                    if (json_depth == 0) json_key.clear();
                    i++; continue;
                }
                if ((c == ':' || c == ',') && (json_depth > 0 || last_was_string)) {
                    const bool keep_name = name_next && c == ':';
                    token(std::string(1, c));
                    name_next = keep_name;
                    i++; continue;
                }
                if (c == '"') {
                    if (size_t n = try_bare_key(i)) { i += n; continue; }
                    i += scan_string(i);
                    continue;
                }
            }
            run += c;
            i++;
        }
        flush();
        return std::move(out);
    }
};

}  // namespace drift_detail

// Redact one model turn for the corpus: framing, keys and declared tool names
// verbatim, every value/prose run a typed placeholder. See the file comment.
inline std::string redact_drift(const std::string& in, const DriftNames* names = nullptr) {
    return drift_detail::Redactor(in, names)();
}

// Dedup key for the corpus: FNV-1a over the redacted text. Values are already
// placeholders, so two turns with the same markup skeleton hash alike and a
// dropped closer does not. A hash, not a security primitive.
inline uint64_t shape_hash(const std::string& redacted) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (unsigned char c : redacted) { h ^= c; h *= 0x100000001b3ull; }
    return h;
}

}  // namespace q27
