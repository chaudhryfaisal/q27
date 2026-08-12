// Model-free streaming helpers for the Metal server: UTF-8 boundary gating,
// stop-sequence holdback, and SSE event framing. Kept separate from the
// engine so the SSE shapes and stop logic have unit coverage that runs
// without the model artifact (build/test_metal_stream). The event shapes and
// the UTF-8 gate mirror the CUDA reference server (src/server.cu,
// src/api_common.h) so Metal and CUDA are wire-compatible.
#pragma once

#include "../../third_party/json.hpp"
// Utf8Gate now comes from the shared header directly (it was a verbatim
// copy here until the 2026-07-17 agentic-parity round, which made the
// Metal server consume api_common.h wholesale — one gate, both servers).
#include "../api_common.h"

#include <algorithm>
#include <array>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace q27 {

// Stop-sequence holdback for streaming text. Implements the OpenAI `stop` /
// Anthropic `stop_sequences` contract for an incremental byte stream: text up
// to (but not including) the first stop occurrence is emitted, the stop
// sequence itself and everything after it is suppressed, and generation stops.
// Because pieces arrive incrementally, a trailing substring of the emitted
// text that could still grow into a stop sequence is held back until it is
// resolved (completed -> stop, or diverged -> released). This is the standard
// partial-prefix holdback; without it a stop sequence split across two tokens
// would leak its first half to the client before the match fires.
inline constexpr size_t kMaxOpenAIStopSequences=4;
inline constexpr size_t kMaxStopSequenceTotalBytes=64u<<10;

inline void validate_stop_sequences(const std::vector<std::string>& stops,
                                    bool openai_count_limit=false) {
    if(openai_count_limit && stops.size()>kMaxOpenAIStopSequences)
        throw std::invalid_argument("too many stop sequences");
    size_t total=0;
    for(const std::string& stop:stops) {
        if(stop.size()>kMaxStopSequenceTotalBytes-total)
            throw std::invalid_argument("stop sequences too large");
        total+=stop.size();
    }
}

struct StopBuffer {
    struct Node {
        int edge=-1;
        uint32_t fail=0;
        uint32_t depth=0;
        int terminal=-1;
        int output=-1;
    };
    struct Edge {
        uint32_t child=0;
        int next=-1;
        unsigned char byte=0;
    };
    static_assert(sizeof(Node)<=24,"stop trie nodes must remain compact");
    static_assert(sizeof(Edge)<=12,"stop trie edges must remain compact");

    std::vector<std::string> stops;
    std::string pend;
    size_t pend_head=0;
    int matched = -1; // index into `stops` once a full match fires, else -1
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::array<int,256> root_next;
    uint32_t state=0;
    size_t candidate_pos=std::string::npos;
    int candidate_idx=-1;

    int child_index(uint32_t node,unsigned char c) const {
        if(node==0) return root_next[c];
        for(int edge=nodes[node].edge;edge>=0;edge=edges[(size_t)edge].next)
            if(edges[(size_t)edge].byte==c) return (int)edges[(size_t)edge].child;
        return -1;
    }

    uint32_t ensure_child(uint32_t node,unsigned char c) {
        const int found=child_index(node,c);
        if(found>=0) return (uint32_t)found;
        const uint32_t child=(uint32_t)nodes.size();
        nodes.emplace_back();
        nodes[child].depth=nodes[node].depth+1;
        edges.push_back({child,nodes[node].edge,c});
        nodes[node].edge=(int)edges.size()-1;
        if(node==0) root_next[c]=(int)child;
        return child;
    }

    explicit StopBuffer(std::vector<std::string> seqs = {}) : stops(std::move(seqs)) {
        // Drop empty stop strings: they would "match" at every position.
        stops.erase(std::remove(stops.begin(), stops.end(), std::string()), stops.end());
        validate_stop_sequences(stops);
        root_next.fill(-1);
        nodes.emplace_back();
        for(size_t si=0;si<stops.size();si++) {
            uint32_t node=0;
            for(unsigned char c:stops[si]) node=ensure_child(node,c);
            if(nodes[node].terminal<0 || (int)si<nodes[node].terminal)
                nodes[node].terminal=(int)si;
            nodes[node].output=nodes[node].terminal;
        }
        auto better_output=[&](int candidate,int current) {
            if(candidate<0) return false;
            if(current<0) return true;
            const size_t candidate_len=stops[(size_t)candidate].size();
            const size_t current_len=stops[(size_t)current].size();
            return candidate_len>current_len ||
                (candidate_len==current_len && candidate<current);
        };
        std::queue<uint32_t> pending;
        for(int edge=nodes[0].edge;edge>=0;edge=edges[(size_t)edge].next)
            pending.push(edges[(size_t)edge].child);
        while(!pending.empty()) {
            const uint32_t parent=pending.front();
            pending.pop();
            for(int edge=nodes[parent].edge;edge>=0;edge=edges[(size_t)edge].next) {
                const unsigned char c=edges[(size_t)edge].byte;
                const uint32_t child=edges[(size_t)edge].child;
                uint32_t fallback=nodes[parent].fail;
                int next=child_index(fallback,c);
                while(fallback && next<0) {
                    fallback=nodes[fallback].fail;
                    next=child_index(fallback,c);
                }
                if(next>=0 && (uint32_t)next!=child) nodes[child].fail=(uint32_t)next;
                const int inherited=nodes[nodes[child].fail].output;
                if(better_output(inherited,nodes[child].output))
                    nodes[child].output=inherited;
                pending.push(child);
            }
        }
    }

    size_t pending_size() const { return pend.size()-pend_head; }

    void compact_pending() {
        if(pend_head>=4096 && pend_head>=pend.size()-pend_head) {
            pend.erase(0,pend_head);
            pend_head=0;
        }
    }

    bool candidate_ready() const {
        if(candidate_idx<0) return false;
        // A still-active prefix can delay a completed stop only when it began
        // earlier. A longer stop beginning at the same byte must not consume
        // more generated output after a complete stop has already matched.
        const size_t active_start=pending_size()-nodes[state].depth;
        return active_start>=candidate_pos;
    }

    std::string finish_match(bool& stopped) {
        stopped=true;
        matched=candidate_idx;
        std::string out=pend.substr(pend_head,candidate_pos);
        pend.clear();
        pend_head=0;
        state=0;
        candidate_pos=std::string::npos;
        candidate_idx=-1;
        return out;
    }

    bool active() const { return !stops.empty(); }

    // Feed one decoded (already UTF-8-gated) piece. A compact Aho-Corasick
    // automaton makes matching linear in generated bytes regardless of the
    // number or length of accepted Anthropic stop sequences.
    std::string feed(const std::string& piece, bool& stopped) {
        stopped=false;
        if(stops.empty()) return piece;
        for(unsigned char c:piece) {
            pend.push_back((char)c);
            int next=child_index(state,c);
            while(state && next<0) {
                state=nodes[state].fail;
                next=child_index(state,c);
            }
            state=next<0?0:(uint32_t)next;
            const int candidate=nodes[state].output;
            if(candidate>=0) {
                const size_t pos=pending_size()-stops[(size_t)candidate].size();
                if(pos<candidate_pos || (pos==candidate_pos && candidate<candidate_idx)) {
                    candidate_pos=pos;
                    candidate_idx=candidate;
                }
            }
            if(candidate_ready()) return finish_match(stopped);
        }
        size_t emit_len=pending_size()-nodes[state].depth;
        if(candidate_idx>=0) emit_len=std::min(emit_len,candidate_pos);
        std::string out=pend.substr(pend_head,emit_len);
        pend_head+=emit_len;
        if(candidate_idx>=0) candidate_pos-=emit_len;
        compact_pending();
        return out;
    }

    // End of stream resolves any provisional match now that no earlier,
    // longer prefix can complete.
    std::string flush(bool* did_stop=nullptr) {
        if(candidate_idx>=0) {
            bool stopped=false;
            std::string out=finish_match(stopped);
            if(did_stop) *did_stop=true;
            return out;
        }
        std::string out=pend.substr(pend_head);
        pend.clear();
        pend_head=0;
        state=0;
        if(did_stop) *did_stop=false;
        return out;
    }
};

// ---- SSE framing (mirrors src/server.cu) ----

// Invalid-UTF-8-tolerant serialize: json::dump's strict default throws
// type_error.316, and an uncaught throw inside a streaming provider is
// std::terminate. The Utf8Gate keeps split characters intact; this is the
// backstop for everything else. Same as server.cu's file-scope jdump.
inline std::string sse_dump(const nlohmann::json& j) {
    return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

// OpenAI framing: "data: {json}\n\n".
inline std::string sse_data(const nlohmann::json& j) {
    return "data: " + sse_dump(j) + "\n\n";
}

// Anthropic / Responses framing: "event: NAME\ndata: {json}\n\n".
inline std::string sse_event(const std::string& name, const nlohmann::json& j) {
    return "event: " + name + "\ndata: " + sse_dump(j) + "\n\n";
}

// OpenAI stream terminator.
inline std::string sse_done() { return "data: [DONE]\n\n"; }

// One OpenAI streaming delta chunk. `chat` selects chat.completion.chunk
// (delta.content) vs text_completion (text); finish_reason is null in piece
// chunks, exactly as the CUDA server emits them.
inline nlohmann::json openai_stream_chunk(bool chat, const std::string& id, const char* object,
                                          long created, const std::string& model,
                                          const std::string& piece) {
    using nlohmann::json;
    json choice = chat
        ? json{{"index", 0}, {"delta", {{"content", piece}}}, {"finish_reason", nullptr}}
        : json{{"index", 0}, {"text", piece}, {"finish_reason", nullptr}};
    return json{{"id", id}, {"object", object}, {"created", created},
                {"model", model}, {"choices", json::array({choice})}};
}

// Terminal OpenAI streaming chunk: a real finish_reason ("stop"/"length")
// before [DONE], per the OpenAI streaming spec — clients otherwise never
// learn whether generation hit EOS or the token cap. Mirrors the CUDA
// server's shape (upstream security-review fix #7): empty delta object for
// chat, empty text for completions.
inline nlohmann::json openai_stream_final_chunk(bool chat, const std::string& id,
                                                const char* object, long created,
                                                const std::string& model,
                                                const char* finish_reason) {
    using nlohmann::json;
    json choice = chat
        ? json{{"index", 0}, {"delta", json::object()}, {"finish_reason", finish_reason}}
        : json{{"index", 0}, {"text", ""}, {"finish_reason", finish_reason}};
    return json{{"id", id}, {"object", object}, {"created", created},
                {"model", model}, {"choices", json::array({choice})}};
}

inline nlohmann::json openai_stream_usage_chunk(const std::string& id,
                                                const char* object,long created,
                                                const std::string& model,
                                                uint32_t prompt_tokens,
                                                uint32_t completion_tokens) {
    using nlohmann::json;
    return json{{"id",id},{"object",object},{"created",created},{"model",model},
                {"choices",json::array()},
                {"usage",{{"prompt_tokens",prompt_tokens},
                          {"completion_tokens",completion_tokens},
                          {"total_tokens",prompt_tokens+completion_tokens}}}};
}

} // namespace q27
