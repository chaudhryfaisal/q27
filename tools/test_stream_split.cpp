#include "stream_split.h"

#include <cstdio>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

using Chan = q27::StreamSplitter::Chan;
using Segment = std::pair<Chan, std::string>;

static std::vector<Segment> split(const std::string& input, bool bytewise) {
    q27::StreamSplitter splitter;
    std::vector<Segment> out;
    auto append = [&](std::vector<Segment> part) {
        out.insert(out.end(), part.begin(), part.end());
    };
    if (bytewise) {
        for (char c : input) append(splitter.feed(std::string(1, c)));
    } else {
        append(splitter.feed(input));
    }
    append(splitter.flush());
    return out;
}

static std::vector<Segment> split_pieces(
    std::initializer_list<std::string> pieces, Chan initial = Chan::TEXT) {
    q27::StreamSplitter splitter;
    splitter.chan = initial;
    std::vector<Segment> out;
    for (const auto& piece : pieces) {
        auto part = splitter.feed(piece);
        out.insert(out.end(), part.begin(), part.end());
    }
    auto tail = splitter.flush();
    out.insert(out.end(), tail.begin(), tail.end());
    return out;
}

static bool expect(const char* name, const std::vector<Segment>& got,
                   const std::vector<Segment>& want) {
    if (got == want) {
        std::printf("%s: PASS\n", name);
        return true;
    }
    std::printf("%s: FAIL\n  got:", name);
    for (const auto& [chan, text] : got)
        std::printf(" (%d,%zu,'%s')", (int)chan, text.size(), text.c_str());
    std::printf("\n  want:");
    for (const auto& [chan, text] : want)
        std::printf(" (%d,%zu,'%s')", (int)chan, text.size(), text.c_str());
    std::printf("\n");
    return false;
}

int main() {
    const std::string adjacent =
        "<tool_call>a</tool_call><tool_call>b</tool_call>";
    const std::vector<Segment> adjacent_want = {
        {Chan::TOOL, "a"}, {Chan::TEXT, ""}, {Chan::TOOL, "b"}};

    const std::string empty_think =
        "<tool_call>a</tool_call><think></think><tool_call>b</tool_call>";
    const std::vector<Segment> empty_think_want = {
        {Chan::TOOL, "a"}, {Chan::THINK, ""}, {Chan::TOOL, "b"}};

    const std::string nonempty_think =
        "<tool_call>a</tool_call><think>x</think><tool_call>b</tool_call>";
    const std::vector<Segment> nonempty_think_want = {
        {Chan::TOOL, "a"}, {Chan::THINK, "x"}, {Chan::TOOL, "b"}};

    const std::string text_between =
        "<tool_call>a</tool_call>x<tool_call>b</tool_call>";
    const std::vector<Segment> text_between_want = {
        {Chan::TOOL, "a"}, {Chan::TEXT, "x"}, {Chan::TOOL, "b"}};

    bool ok = true;
    ok = expect("adjacent/full", split(adjacent, false), adjacent_want) && ok;
    ok = expect("adjacent/bytewise", split(adjacent, true), adjacent_want) && ok;
    ok = expect("empty-think/full", split(empty_think, false), empty_think_want) && ok;
    ok = expect("empty-think/bytewise", split(empty_think, true), empty_think_want) && ok;
    ok = expect("nonempty-think/bytewise", split(nonempty_think, true), nonempty_think_want) && ok;
    ok = expect("text-between/bytewise", split(text_between, true), text_between_want) && ok;

    q27::StreamSplitter unfinished;
    std::vector<Segment> unfinished_got = unfinished.feed("<tool_call>a</tool_call><think>");
    auto tail = unfinished.flush();
    unfinished_got.insert(unfinished_got.end(), tail.begin(), tail.end());
    ok = expect("unfinished-empty-think/flush", unfinished_got,
                {{Chan::TOOL, "a"}, {Chan::THINK, ""}}) && ok;

    ok = expect("stray close stripped",
                split_pieces({"hello", "</tool_call>", "world"}),
                {{Chan::TEXT, "hello"}, {Chan::TEXT, "world"}}) && ok;

    const std::vector<Segment> faisal_want = {
        {Chan::TEXT, "{\"name\": \"read\"}\n"},
        {Chan::TEXT, "\n{\"name\": \"read2\"}\n"},
        {Chan::TEXT, "\n"}};
    ok = expect("faisal multi: no </tool_call> in text",
                split_pieces({"{\"name\": \"read\"}\n</tool_call>\n"
                              "{\"name\": \"read2\"}\n</tool_call>\n"}),
                faisal_want) && ok;

    const std::vector<Segment> normal_pair_want = {
        {Chan::TOOL, "{\"a\":1}"}};
    ok = expect("normal pair -> TOOL",
                split_pieces({"<tool_call>", "{\"a\":1}", "</tool_call>"}),
                normal_pair_want) && ok;
    ok = expect("normal pair TEXT=tail only",
                split_pieces({"<tool_call>", "{\"a\":1}", "</tool_call>", "tail"}),
                {{Chan::TOOL, "{\"a\":1}"}, {Chan::TEXT, "tail"}}) && ok;

    ok = expect("split stray tag held+stripped",
                split_pieces({"abc</tool", "_call>def"}),
                {{Chan::TEXT, "abc"}, {Chan::TEXT, "def"}}) && ok;

    const std::vector<Segment> think_want = {
        {Chan::THINK, "reason"}, {Chan::TEXT, "ans"}};
    ok = expect("think still routes",
                split_pieces({"<think>", "reason", "</think>", "ans"}),
                think_want) && ok;
    ok = expect("think text=ans",
                split_pieces({"<think>", "reason", "</think>", "ans"}),
                think_want) && ok;

    const std::vector<Segment> preseeded_want = {
        {Chan::THINK, "reason"}, {Chan::THINK, "ing"},
        {Chan::TEXT, "\n\nans"}};
    ok = expect("preseeded THINK: reasoning routes",
                split_pieces({"reason", "ing", "</think>", "\n\nans"},
                             Chan::THINK),
                preseeded_want) && ok;
    ok = expect("preseeded THINK: answer after </think>",
                split_pieces({"reason", "ing", "</think>", "\n\nans"},
                             Chan::THINK),
                preseeded_want) && ok;
    return ok ? 0 : 1;
}
