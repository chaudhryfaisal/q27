// Streaming splitter for Qwopus output markers: <think>...</think> reasoning
// and <tool_call>...</tool_call> function calls, both emitted as plain-text
// tokens. Feeds token-by-token decoded text and routes segments into THINK /
// TEXT / TOOL channels, holding back any tail that could be a partial marker.
// Markers do not nest; tool_calls can appear only outside think blocks in
// well-formed output, but we tolerate them inside by scanning TEXT only.
#pragma once
#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace q27 {

struct StreamSplitter {
    enum Chan { TEXT, THINK, TOOL };
    Chan chan = TEXT;
    std::string hold;

    static constexpr const char* T_OPEN = "<think>";
    static constexpr const char* T_CLOSE = "</think>";
    static constexpr const char* C_OPEN = "<tool_call>";
    static constexpr const char* C_CLOSE = "</tool_call>";

    std::vector<std::pair<Chan, std::string>> feed(const std::string& piece) {
        hold += piece;
        std::vector<std::pair<Chan, std::string>> out;
        for (;;) {
            if (chan == TEXT) {
                // whichever of <think> open / <tool_call> open / a STRAY
                // </tool_call> close comes first. A real </tool_call> follows an
                // opener that already switched us to TOOL, so any </tool_call>
                // seen in TEXT is stray (bare-call wrapper leftover, issue #4) --
                // strip it, never emit it as visible content.
                size_t pt = hold.find(T_OPEN), pc = hold.find(C_OPEN),
                       sc = hold.find(C_CLOSE);
                size_t e = std::min(pt, std::min(pc, sc));
                if (e != std::string::npos) {
                    if (e == pt) {
                        if (pt > 0) out.push_back({TEXT, hold.substr(0, pt)});
                        hold.erase(0, pt + strlen(T_OPEN));
                        chan = THINK;
                        continue;
                    }
                    if (e == pc) {
                        if (pc > 0) out.push_back({TEXT, hold.substr(0, pc)});
                        hold.erase(0, pc + strlen(C_OPEN));
                        chan = TOOL;
                        continue;
                    }
                    if (sc > 0) out.push_back({TEXT, hold.substr(0, sc)});
                    hold.erase(0, sc + strlen(C_CLOSE)); // strip stray close
                    continue;
                }
                // hold back the longest suffix that prefixes any marker
                size_t keep = tail_keep(T_OPEN);
                keep = std::max(keep, tail_keep(C_OPEN));
                keep = std::max(keep, tail_keep(C_CLOSE));
                emit_head(out, keep);
                break;
            }
            const char* closer = chan == THINK ? T_CLOSE : C_CLOSE;
            size_t p = hold.find(closer);
            if (p != std::string::npos) {
                if (p > 0) out.push_back({chan, hold.substr(0, p)});
                hold.erase(0, p + strlen(closer));
                chan = TEXT;
                continue;
            }
            emit_head(out, tail_keep(closer));
            break;
        }
        return out;
    }

    std::vector<std::pair<Chan, std::string>> flush() {
        std::vector<std::pair<Chan, std::string>> out;
        if (!hold.empty()) { out.push_back({chan, hold}); hold.clear(); }
        return out;
    }

  private:
    size_t tail_keep(const char* marker) const {
        size_t mlen = strlen(marker);
        size_t maxk = std::min(hold.size(), mlen - 1);
        for (size_t k = maxk; k > 0; k--)
            if (hold.compare(hold.size() - k, k, marker, k) == 0) return k;
        return 0;
    }
    void emit_head(std::vector<std::pair<Chan, std::string>>& out, size_t keep) {
        if (hold.size() > keep) {
            out.push_back({chan, hold.substr(0, hold.size() - keep)});
            hold.erase(0, hold.size() - keep);
        }
    }
};

// Reasoning-token budget (2026-07-28). Counts tokens generated INSIDE the THINK
// channel and, on trip, hands back T_CLOSE for the caller to feed through the
// splitter -- which flips the channel to TEXT exactly as a model-emitted
// </think> would, so every downstream consumer (routing, tool scan, the
// Anthropic thinking block, reasoning_content) sees an ordinary close.
//
// Call tick() once per generated token, AFTER feeding that token to the
// splitter. It counts only while the splitter is actually in THINK, so a
// prompt-injected opener, a natural close, and tool segments are all handled
// without special cases. Trips at most once per request.
//
// Deliberately NOT a hard stop: closing the block lets the model answer with
// what it has. Truncating the request instead would reproduce the failure this
// exists to prevent.
struct ThinkBudget {
    int limit = -1;  // -1 = unbounded
    int used = 0;
    bool tripped = false;

    const char* tick(const StreamSplitter& sp) {
        if (sp.chan != StreamSplitter::THINK) return nullptr;
        // Count ALWAYS, enforce only when bounded. `used` is reported as
        // usage.reasoning_tokens, and a counter that reads 0 while the model
        // reasoned for 2879 characters is the same accepted-and-inert failure
        // this class exists to prevent -- just one level down.
        ++used;
        if (limit < 0 || tripped) return nullptr;
        if (used < limit) return nullptr;
        tripped = true;
        return StreamSplitter::T_CLOSE;
    }
};

} // namespace q27
