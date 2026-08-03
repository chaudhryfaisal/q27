#include "stream_split.h"

#include <cstdio>
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
    return ok ? 0 : 1;
}
