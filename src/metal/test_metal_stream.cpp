// Model-free unit tests for the Metal server's streaming helpers:
// UTF-8 boundary gating, stop-sequence holdback, SSE event framing, and the
// StreamSplitter channel router (incl. the adjacent-call boundary segment).
// These run without the model artifact (part of `make test-metal`) and are
// the regression net for the wire shapes the CUDA reference server defines.
#include "stream_format.h"
#include "serving_policy.h"

#include "../stream_split.h"

#include <cstdio>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", what); failures++; }
}

// ---- Utf8Gate ----
int test_utf8_gate() {
    // A three-byte em dash (E2 80 94) split across two token pieces: the first
    // feed must hold the incomplete lead+continuation back, the second complete
    // it. Emitting the partial would make json::dump throw.
    q27::Utf8Gate g;
    check(g.feed("ab") == "ab", "utf8: ascii passes through");
    check(g.feed("\xE2\x80") == "", "utf8: incomplete multibyte held back");
    check(g.feed("\x94") == "\xE2\x80\x94", "utf8: multibyte completes on continuation");
    check(g.flush() == "", "utf8: clean flush is empty");

    // A dangling partial at end of stream becomes U+FFFD.
    q27::Utf8Gate g2;
    check(g2.feed("x\xE2\x80") == "x", "utf8: emits valid prefix, holds tail");
    check(g2.flush() == "\xEF\xBF\xBD", "utf8: dangling tail flushes to U+FFFD");
    return 0;
}

// ---- StopBuffer ----
int test_stop_buffer() {
    // No stop sequences: identity passthrough, nothing held.
    {
        q27::StopBuffer sb;
        bool stopped = false;
        check(!sb.active(), "stop: empty is inactive");
        check(sb.feed("hello world", stopped) == "hello world", "stop: passthrough");
        check(!stopped, "stop: passthrough not stopped");
        check(sb.flush() == "", "stop: passthrough flush empty");
    }
    // Empty stop strings are dropped (would otherwise match everywhere).
    {
        q27::StopBuffer sb({"", ""});
        check(!sb.active(), "stop: empty strings dropped");
    }
    // Full match within one feed truncates at the stop and reports the index.
    {
        q27::StopBuffer sb({"STOP"});
        bool stopped = false;
        check(sb.feed("abcSTOPdef", stopped) == "abc", "stop: truncates before match");
        check(stopped && sb.matched == 0, "stop: reports stopped + index");
    }
    // Stop sequence split across two feeds: the first half is held back, not
    // leaked, and the match fires when the second half arrives.
    {
        q27::StopBuffer sb({"</s>"});
        bool stopped = false;
        check(sb.feed("hello</", stopped) == "hello", "stop: holds back partial prefix");
        check(!stopped, "stop: partial not yet stopped");
        check(sb.feed("s>world", stopped) == "", "stop: completes across feeds");
        check(stopped && sb.matched == 0, "stop: split match stops");
    }
    // A partial prefix that diverges is released, not swallowed.
    {
        q27::StopBuffer sb({"</s>"});
        bool stopped = false;
        check(sb.feed("a</b", stopped) == "a</b", "stop: diverging partial released");
        check(!stopped, "stop: diverging not stopped");
    }
    // Earliest match across multiple sequences wins.
    {
        q27::StopBuffer sb({"XX", "Y"});
        bool stopped = false;
        check(sb.feed("aYbXX", stopped) == "a", "stop: earliest position wins");
        check(stopped && sb.matched == 1, "stop: matched index is the Y sequence");
    }
    // Held-back partial with no completion is real output at flush.
    {
        q27::StopBuffer sb({"</s>"});
        bool stopped = false;
        check(sb.feed("text</s", stopped) == "text", "stop: holds trailing partial");
        check(!stopped, "stop: trailing partial not stopped");
        check(sb.flush() == "</s", "stop: flush releases held partial");
    }
    // OpenAI documents at most four stops; Anthropic accepts a general list.
    {
        const std::vector<std::string> five={"S0","S1","S2","S3","S4"};
        bool openai_rejected=false,anthropic_accepted=true,total_rejected=false;
        try { q27::validate_stop_sequences(five,true); }
        catch(const std::invalid_argument&) { openai_rejected=true; }
        try { q27::validate_stop_sequences(five,false); }
        catch(const std::invalid_argument&) { anthropic_accepted=false; }
        try {
            q27::StopBuffer sb({std::string(q27::kMaxStopSequenceTotalBytes+1,'x')});
        } catch(const std::invalid_argument&) { total_rejected=true; }
        check(openai_rejected, "stop: OpenAI count bounded");
        check(anthropic_accepted, "stop: Anthropic list count unrestricted");
        check(total_rejected, "stop: aggregate matcher storage bounded");
    }
    // The automaton preserves earliest-position semantics and handles a long
    // repeated prefix across pieces without quadratic suffix rescans.
    {
        q27::StopBuffer overlap({"abcd","b"});
        bool stopped=false;
        check(overlap.feed("abcd",stopped)=="", "stop: earliest start beats earlier completion");
        check(stopped && overlap.matched==0, "stop: earliest-start sequence index");

        q27::StopBuffer split_overlap({"abcd","b"});
        stopped=false;
        check(split_overlap.feed("ab",stopped)=="", "stop: overlapping candidate deferred");
        check(!stopped, "stop: earlier prefix remains eligible");
        check(split_overlap.feed("cd",stopped)=="", "stop: split earlier match suppressed");
        check(stopped && split_overlap.matched==0,
              "stop: split pieces preserve earliest-start sequence");

        q27::StopBuffer same_start({"ab","a"});
        stopped=false;
        check(same_start.feed("a",stopped)=="", "stop: complete same-start prefix suppressed");
        check(stopped && same_start.matched==1,
              "stop: complete same-start prefix fires without waiting for a longer stop");

        q27::StopBuffer failed_prefix({"abcd","b"});
        stopped=false;
        check(failed_prefix.feed("ab",stopped)=="" && !stopped,
              "stop: later match waits for earlier prefix");
        check(failed_prefix.feed("x",stopped)=="a",
              "stop: deferred later match emits only preceding text");
        check(stopped && failed_prefix.matched==1,
              "stop: deferred later match resolves after prefix failure");

        q27::StopBuffer eos_overlap({"abcd","b"});
        stopped=false;
        check(eos_overlap.feed("ab",stopped)=="" && !stopped,
              "stop: provisional match survives to stream end");
        bool flush_stopped=false;
        check(eos_overlap.flush(&flush_stopped)=="a",
              "stop: stream end resolves provisional match");
        check(flush_stopped && eos_overlap.matched==1,
              "stop: flush reports provisional match");

        std::string long_stop(100,'x');
        long_stop+='!';
        q27::StopBuffer long_prefix({long_stop});
        stopped=false;
        check(long_prefix.feed(std::string(80,'x'),stopped)=="", "stop: long prefix held");
        check(!stopped, "stop: long prefix remains pending");
        check(long_prefix.feed(std::string(20,'x')+"!tail",stopped)=="",
              "stop: long split sequence suppressed");
        check(stopped, "stop: long split sequence matched");

        std::string repeated_stop(32768,'a');
        repeated_stop+='b';
        q27::StopBuffer repeated({repeated_stop});
        stopped=false;
        check(repeated.feed(std::string(32768,'a'),stopped)=="" && !stopped,
              "stop: maximal repeated prefix held");
        const std::string one_a="a";
        size_t released=0;
        for(size_t i=0;i<131072;i++) released+=repeated.feed(one_a,stopped).size();
        check(released==131072 && !stopped,
              "stop: repeated-prefix sliding releases one byte per feed");
        check(repeated.pend.size()<=repeated_stop.size()*2+4096,
              "stop: pending storage compacts amortized");
        check(repeated.feed("b",stopped)=="" && stopped,
              "stop: repeated-prefix match still fires after compaction");
    }
    return 0;
}

// ---- SSE framing ----
int test_sse_framing() {
    check(q27::sse_data(json{{"a", 1}}) == "data: {\"a\":1}\n\n", "sse: data frame");
    check(q27::sse_done() == "data: [DONE]\n\n", "sse: done terminator");

    std::string ev = q27::sse_event("message_stop", json{{"type", "message_stop"}});
    check(ev == "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n", "sse: event frame");

    // OpenAI chat delta chunk shape: choices[0].delta.content, null finish.
    json chat = q27::openai_stream_chunk(true, "chatcmpl-metal", "chat.completion.chunk",
                                         1700000000, "q27-metal", "hi");
    check(chat["object"] == "chat.completion.chunk", "sse: chat chunk object");
    check(chat["choices"][0]["delta"]["content"] == "hi", "sse: chat delta content");
    check(chat["choices"][0]["finish_reason"].is_null(), "sse: chat chunk null finish");

    // Completions chunk shape: choices[0].text.
    json txt = q27::openai_stream_chunk(false, "cmpl-metal", "text_completion",
                                        1700000000, "q27-metal", "yo");
    check(txt["object"] == "text_completion", "sse: text chunk object");
    check(txt["choices"][0]["text"] == "yo", "sse: text chunk text");
    check(txt["choices"][0]["finish_reason"].is_null(), "sse: text chunk null finish");

    // Terminal chunk: real finish_reason, empty delta object (chat) / empty
    // text (completions) — server.cu's shape after security-review fix #7.
    json fchat = q27::openai_stream_final_chunk(true, "chatcmpl-metal", "chat.completion.chunk",
                                                1700000000, "q27-metal", "stop");
    check(fchat["choices"][0]["finish_reason"] == "stop", "sse: final chat finish_reason");
    check(fchat["choices"][0]["delta"].is_object() && fchat["choices"][0]["delta"].empty(),
          "sse: final chat empty delta object");
    json ftxt = q27::openai_stream_final_chunk(false, "cmpl-metal", "text_completion",
                                               1700000000, "q27-metal", "length");
    check(ftxt["choices"][0]["finish_reason"] == "length", "sse: final text finish_reason");
    check(ftxt["choices"][0]["text"] == "", "sse: final text empty");

    json usage = q27::openai_stream_usage_chunk("chatcmpl-metal","chat.completion.chunk",
                                                1700000000,"q27-metal",7,3);
    check(usage["choices"].is_array() && usage["choices"].empty(),
          "sse: usage chunk has empty choices");
    check(usage["usage"]["prompt_tokens"] == 7 &&
          usage["usage"]["completion_tokens"] == 3 &&
          usage["usage"]["total_tokens"] == 10,
          "sse: usage chunk totals");

    // Invalid UTF-8 must not throw through the serializer (replace backstop).
    std::string bad = q27::sse_data(q27::openai_stream_chunk(true, "id", "chat.completion.chunk",
                                                             0, "m", std::string("\xE2\x80")));
    check(!bad.empty(), "sse: invalid utf-8 serializes via replace handler");
    return 0;
}

} // namespace

// ---- StreamSplitter ----
namespace {
using Seg = std::pair<q27::StreamSplitter::Chan, std::string>;
bool segs_eq(const std::vector<Seg>& got, std::initializer_list<Seg> want) {
    return got == std::vector<Seg>(want);
}
int test_splitter() {
    using C = q27::StreamSplitter::Chan;
    const C TEXT = q27::StreamSplitter::TEXT, THINK = q27::StreamSplitter::THINK,
            TOOL = q27::StreamSplitter::TOOL;
    {   // plain text/think routing
        q27::StreamSplitter sp;
        check(segs_eq(sp.feed("hello<think>t</think>world"),
                      {{TEXT,"hello"},{THINK,"t"},{TEXT,"world"}}),
              "splitter: text/think/text routing");
        check(sp.flush().empty(), "splitter: clean flush");
    }
    {   // adjacent wrapped calls in one feed: an empty TEXT boundary segment
        // must separate them because consumers buffer one TOOL segment and
        // flush on any non-TOOL segment; without it, calls fold together.
        q27::StreamSplitter sp;
        check(segs_eq(sp.feed("<tool_call>A</tool_call><tool_call>B</tool_call>"),
                      {{TOOL,"A"},{TEXT,""},{TOOL,"B"}}),
              "splitter: adjacent calls emit a boundary segment");
    }
    {   // same adjacency across feed boundaries
        q27::StreamSplitter sp;
        check(segs_eq(sp.feed("<tool_call>A</tool_call>"), {{TOOL,"A"}}),
              "splitter: first call routed");
        check(sp.chan==TEXT && sp.tool_boundary,
              "splitter: exact final closer remains observable");
        check(segs_eq(sp.feed("<tool_call>B</tool_call>"), {{TEXT,""},{TOOL,"B"}}),
              "splitter: cross-feed adjacency emits the boundary");
    }
    {   // closer split across feeds, then an adjacent call
        q27::StreamSplitter sp;
        check(segs_eq(sp.feed("<tool_call>A</tool_"), {{TOOL,"A"}}),
              "splitter: partial closer held back, head emitted");
        check(segs_eq(sp.feed("call><tool_call>B</tool_call>"), {{TEXT,""},{TOOL,"B"}}),
              "splitter: split-closer adjacency emits the boundary");
    }
    {   // A held prefix of another wrapper means the response did not end at
        // the previous closed call, even though tool_boundary is still set.
        q27::StreamSplitter sp;
        check(segs_eq(sp.feed("<tool_call>A</tool_call><tool_"), {{TOOL,"A"}}),
              "splitter: partial next wrapper stays held");
        check(sp.chan==TEXT && sp.tool_boundary && !sp.hold.empty(),
              "splitter: partial next wrapper is not a clean closed-tool tail");
    }
    {   // controls: real text between calls, think after a call, and text
        // before a call must NOT grow boundary segments
        q27::StreamSplitter sp;
        check(segs_eq(sp.feed("<tool_call>A</tool_call>x<tool_call>B</tool_call>"),
                      {{TOOL,"A"},{TEXT,"x"},{TOOL,"B"}}),
              "splitter: text between calls, no boundary segment");
        q27::StreamSplitter sp2;
        check(segs_eq(sp2.feed("<tool_call>A</tool_call><think>t</think>"),
                      {{TOOL,"A"},{THINK,"t"}}),
              "splitter: think after call, no boundary segment");
        q27::StreamSplitter sp3;
        check(segs_eq(sp3.feed("x<tool_call>A</tool_call>"), {{TEXT,"x"},{TOOL,"A"}}),
              "splitter: text before call, no boundary segment");
    }
    return 0;
}
int test_think_budget() {
    using C = q27::StreamSplitter::Chan;
    struct Task {
        bool sampling=false;
        bool accept_one=false;
        const std::vector<int>* forced=nullptr;
        void force(const std::vector<int>& ids) { forced=&ids; accept_one=true; }
        bool force_from_public_budget(const std::vector<int>& ids) { force(ids); return true; }
        void release_accept_one() { if(!forced) accept_one=false; }
    } task;
    const std::vector<int> close_ids{7,8};
    auto apply=[&](q27::ThinkBudgetAction action) {
        if(action==q27::ThinkBudgetAction::FORCE_RESERVED) task.force(close_ids);
        else if(action==q27::ThinkBudgetAction::FORCE_PUBLIC)
            task.force_from_public_budget(close_ids);
    };

    q27::StreamSplitter sp;
    sp.chan = C::THINK;
    q27::ThinkBudgetState budget{2};
    apply(budget.start(sp.chan));
    C before=sp.chan;
    (void)sp.feed("first");
    budget.observe(before,sp.chan);
    apply(budget.finish_round(sp.chan));
    check(!task.forced && budget.used == 1 && !budget.tripped,
          "think budget: counts inside THINK without early trip");
    before=sp.chan;
    (void)sp.feed("second");
    budget.observe(before,sp.chan);
    apply(budget.finish_round(sp.chan));
    check(task.forced == &close_ids && budget.used == 2 && budget.tripped,
          "think budget: queues decoder-visible close exactly at limit");
    task.forced=nullptr;
    before=sp.chan;
    (void)sp.feed(q27::StreamSplitter::T_CLOSE);
    budget.observe(before,sp.chan,true);
    check(sp.chan == C::TEXT, "think budget: forced close resumes TEXT");
    before=sp.chan;
    (void)sp.feed("answer");
    budget.observe(before,sp.chan);
    apply(budget.finish_round(sp.chan));
    check(!task.forced && budget.used == 2,
          "think budget: stops counting after close");

    q27::StreamSplitter unbounded;
    unbounded.chan = C::THINK;
    q27::ThinkBudgetState no_limit{-1};
    before=unbounded.chan;
    (void)unbounded.feed("token");
    no_limit.observe(before,unbounded.chan);
    apply(no_limit.finish_round(unbounded.chan));
    check(!task.forced && no_limit.used == 1 && !no_limit.tripped,
          "think budget: unbounded mode still reports usage");

    q27::ThinkBudgetState zero{0};
    apply(zero.start(C::THINK));
    check(task.forced == &close_ids && zero.tripped && zero.used == 0,
          "think budget: zero cap closes before generation");
    return 0;
}

int test_serving_policy() {
    check(q27::metal_default_max_tokens(q27::MetalEndpoint::Completions)==256 &&
          q27::metal_default_max_tokens(q27::MetalEndpoint::Chat)==256 &&
          q27::metal_default_max_tokens(q27::MetalEndpoint::Messages)==1024 &&
          q27::metal_default_max_tokens(q27::MetalEndpoint::Responses)==4096,
          "policy: endpoint-specific default output limits");
    check(q27::metal_serving_speculation_width(4,true,true,false,false,false)==4,
          "policy: greedy MTP reserves configured width");
    check(q27::metal_serving_speculation_width(4,true,true,true,true,false)==0,
          "policy: forced plain sampling reserves no MTP lanes");
    check(q27::metal_serving_speculation_width(4,true,true,false,false,true)==0,
          "policy: bounded reasoning reserves no speculation lanes");
    check(q27::metal_serving_speculation_width(4,false,true,false,false,false)==0,
          "policy: artifact without MTP reserves no MTP lanes");
    check(q27::metal_serving_speculation_width(4,true,false,false,false,false)==0,
          "policy: serial-only backend reserves no lanes");
    check(q27::metal_tool_constraint_enabled(true,true,true,0,false) &&
          !q27::metal_tool_constraint_enabled(true,true,true,4,false) &&
          !q27::metal_tool_constraint_enabled(true,true,true,0,true),
          "policy: constraints require active serial decode without forced preseed");
    check(q27::metal_tool_constraint_enabled(
              true,true,true,
              q27::metal_serving_speculation_width(4,false,true,false,false,false),false) &&
          q27::metal_tool_constraint_enabled(
              true,true,true,
              q27::metal_serving_speculation_width(4,true,true,false,false,true),false),
          "policy: configured speculation keeps constraints on serial fallback");
    check(!q27::responses_token_limit_remains(true,true,true,false),
          "policy: exact-limit closed eligible tool call remains actionable");
    check(q27::responses_token_limit_remains(true,false,true,false) &&
          q27::responses_token_limit_remains(true,true,false,false) &&
          q27::responses_token_limit_remains(true,true,true,true),
          "policy: token limit remains for missing, rejected, or incomplete calls");
    bool closed_tail=false;
    closed_tail=q27::responses_closed_tool_tail_after_segment(
        closed_tail,true,false," \n");
    check(closed_tail,
          "policy: trailing whitespace preserves a structurally closed tool");
    closed_tail=q27::responses_closed_tool_tail_after_segment(
        closed_tail,false,false,"answer");
    check(!closed_tail,
          "policy: substantive trailing text clears closed tool state");
    check(!q27::responses_closed_tool_tail_after_segment(
              true,false,true,""),
          "policy: a reasoning transition clears closed tool state");
    check(q27::responses_output_index_after_stream_item(0,0,false)==0 &&
          q27::responses_output_index_after_stream_item(0,0,true)==1 &&
          q27::responses_output_index_after_stream_item(2,-1,true)==2,
          "policy: suppressed streamed calls do not consume output indices");
    const uint8_t low_masks[2]={0x01,0x0f};
    const uint8_t high_masks[2]={0x11,0x8f};
    check(q27::metal_snapshot_head_mask_tag(low_masks,2)=="010f" &&
          q27::metal_snapshot_head_mask_tag(high_masks,2)=="118f",
          "policy: snapshot tag includes all eight head-mask bits");
    check(q27::metal_max_prompt_tokens(16,4)==11 &&
          q27::metal_max_prompt_tokens(16,0)==15,
          "policy: prompt ceiling follows effective speculation");
    check(q27::metal_max_prompt_tokens_for_request(16,12,0)==16 &&
          q27::metal_max_prompt_tokens_for_request(16,12,1)==16 &&
          q27::metal_max_prompt_tokens_for_request(16,12,2)==15 &&
          q27::metal_max_prompt_tokens_for_request(16,12,5)==12,
          "policy: request output bounds live speculation reserve");
    check(q27::metal_max_generation_tokens(15,16)==2 &&
          q27::metal_max_generation_tokens(16,16)==1 &&
          q27::metal_max_generation_tokens(0,16)==16,
          "policy: resident prompt logits extend output capacity by one");
    check(q27::metal_generation_fits(15,2,16) &&
          !q27::metal_generation_fits(16,2,16) &&
          q27::metal_generation_fits(16,0,16),
          "policy: full request fit is known before prefill");
    const auto speculative_boundary=q27::resolve_think_decode_limits(
        4,16,12,5,2,true,q27::ThinkCfg{},-1);
    const auto serial_boundary=q27::resolve_think_decode_limits(
        4,16,12,1,2,true,q27::ThinkCfg{},-1);
    check(!speculative_boundary.context_ok && serial_boundary.context_ok &&
              serial_boundary.budget==1,
          "policy: bounded near-boundary request selects serial reserve");
    return 0;
}

} // namespace

int main() {
    test_utf8_gate();
    test_stop_buffer();
    test_sse_framing();
    test_splitter();
    test_think_budget();
    test_serving_policy();
    if (failures) { fprintf(stderr, "%d stream-format check(s) failed\n", failures); return 1; }
    puts("Metal stream format: OK");
    return 0;
}
