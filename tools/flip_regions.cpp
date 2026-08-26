// Which positions in a rendered token stream are the model's own output, and
// which of those sit inside a tool call. The flip gate (--flip-dump) needs
// this for two reasons:
//
//   1. A flip rate over the WHOLE stream is mostly a flip rate over the
//      prompt, where the model is not choosing anything. thr3e's numbers are
//      over "natural assistant-output tokens" and ours should be too.
//   2. The finding that motivated the gate is that divergence CLUSTERS at
//      tool calls -- a flipped byte inside a path or a CLI verb executes the
//      wrong command, while the same flip inside prose is invisible. Reported
//      as one number those cancel; split, they do not.
//
// Classes (one per target position, i.e. the token being predicted):
//   prompt  - anything outside an assistant turn (system, user, tool results)
//   out     - assistant turn, ordinary output
//   think   - assistant turn, inside <think>...</think>
//   tool    - assistant turn, inside a tool call (wrapped or bare dialect)
//
// Offsets come from decoding each id and accumulating, so the mapping is the
// tokenizer's own; no re-tokenization and no assumption that the markup lands
// on a token boundary. A span that opens mid-token claims that token.
#include "tokenizer.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s model.tok stream.i32 [--positions OUT.i32 CLASS[,CLASS...]] "
                "[--tsv OUT.tsv] [--text OUT.txt]\n"
                "  classes: out think tool prompt\n", argv[0]);
        return 2;
    }
    q27::Tokenizer tok(argv[1]);
    FILE* f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    std::vector<int> ids;
    int v;
    while (fread(&v, 4, 1, f) == 1) ids.push_back(v);
    fclose(f);
    const char* pos_out = nullptr; const char* pos_classes = nullptr;
    const char* tsv_out = nullptr; const char* text_out = nullptr;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--positions") && i + 2 < argc) { pos_out = argv[i+1]; pos_classes = argv[i+2]; i += 2; }
        else if (!strcmp(argv[i], "--tsv") && i + 1 < argc) tsv_out = argv[++i];
        else if (!strcmp(argv[i], "--text") && i + 1 < argc) text_out = argv[++i];
    }
    // Turn boundaries live in the ID stream, not the text: decode_one gives
    // "" for <|im_start|>/<|im_end|>, so a text search for them finds nothing
    // (the first version of this tool reported 0 assistant tokens for that
    // reason). Get the ids from the tokenizer and segment on them.
    auto special = [&](const char* lit) {
        auto e = tok.encode(lit);
        if (e.size() != 1)
            fprintf(stderr, "warning: %s encoded to %zu tokens\n", lit, e.size());
        return e.empty() ? -1 : e[0];
    };
    const int IM_START = special("<|im_start|>"), IM_END_ID = special("<|im_end|>");
    if (IM_START < 0 || IM_END_ID < 0) { fprintf(stderr, "no chat special tokens\n"); return 1; }

    // decode once, recording each token's byte span
    std::string text;
    std::vector<size_t> begin(ids.size()), end(ids.size());
    for (size_t i = 0; i < ids.size(); i++) {
        begin[i] = text.size();
        text += tok.decode_one(ids[i]);
        end[i] = text.size();
    }
    // byte-level class map, then per-token by its FIRST byte
    std::vector<char> cls(text.size(), 'p');
    auto mark = [&](size_t a, size_t b, char c) {
        if (a == std::string::npos) return;
        b = std::min(b, text.size());
        for (size_t i = a; i < b; i++) cls[i] = c;
    };
    // assistant turns, from the id stream: <|im_start|> then a role, output
    // until <|im_end|>. The role tokens themselves are not model output.
    std::vector<std::pair<size_t, size_t>> turns; // [first_out_tok, end_tok)
    for (size_t i = 0; i < ids.size(); i++) {
        if (ids[i] != IM_START) continue;
        std::string role;
        size_t j = i + 1;
        for (; j < ids.size() && role.size() < 24; j++) {
            if (ids[j] == IM_END_ID || ids[j] == IM_START) break;
            const std::string piece = tok.decode_one(ids[j]);
            role += piece;
            if (role.find('\n') != std::string::npos) { j++; break; }
        }
        while (!role.empty() && (role.back() == '\n' || role.back() == '\r')) role.pop_back();
        if (role != "assistant") continue;
        size_t stop = j;
        while (stop < ids.size() && ids[stop] != IM_END_ID && ids[stop] != IM_START) stop++;
        if (j < stop) turns.push_back({j, stop});
    }
    for (auto& tr : turns) {
        const size_t body = begin[tr.first];
        const size_t stop = end[tr.second - 1];
        mark(body, stop, 'o');
        // think blocks
        for (size_t t = text.find("<think>", body); t != std::string::npos && t < stop;
             t = text.find("<think>", t + 1)) {
            size_t te = text.find("</think>", t);
            te = (te == std::string::npos || te > stop) ? stop : te + 8;
            mark(t, te, 'k');
        }
        // wrapped calls
        for (size_t t = text.find("<tool_call>", body); t != std::string::npos && t < stop;
             t = text.find("<tool_call>", t + 1)) {
            size_t te = text.find("</tool_call>", t);
            te = (te == std::string::npos || te > stop) ? stop : te + 12;
            mark(t, te, 't');
        }
        // bare dialect: a <function= with no wrapper is still a call, and it
        // is the shape that cost 11 calls in the 2026-08-22 arms
        for (size_t t = text.find("<function=", body); t != std::string::npos && t < stop;
             t = text.find("<function=", t + 1)) {
            if (cls[t] == 't') continue;
            size_t te = text.find("</function>", t);
            te = (te == std::string::npos || te > stop) ? stop : te + 11;
            mark(t, te, 't');
        }
    }
    long n[4] = {0, 0, 0, 0};
    auto klass = [&](size_t i) { return begin[i] < text.size() ? cls[begin[i]] : 'p'; };
    auto name = [](char c) {
        return c == 'o' ? "out" : c == 'k' ? "think" : c == 't' ? "tool" : "prompt";
    };
    FILE* ft = tsv_out ? fopen(tsv_out, "w") : nullptr;
    if (tsv_out && !ft) { fprintf(stderr, "cannot open %s\n", tsv_out); return 1; }
    if (ft) fprintf(ft, "pos\tclass\tid\tbyte\n");
    std::vector<int> selected;
    std::string want = pos_classes ? pos_classes : "";
    auto wanted = [&](const char* nm) {
        if (want.empty()) return false;
        size_t p = want.find(nm);
        return p != std::string::npos;
    };
    for (size_t i = 0; i < ids.size(); i++) {
        const char c = klass(i);
        n[c == 'o' ? 0 : c == 'k' ? 1 : c == 't' ? 2 : 3]++;
        if (ft) fprintf(ft, "%zu\t%s\t%d\t%zu\n", i, name(c), ids[i], begin[i]);
        // a POSITION is the token being predicted, so i>=1 and the class is
        // the class of the token AT i (what the model had to produce)
        if (i >= 1 && wanted(name(c))) selected.push_back((int)i);
    }
    if (text_out) {
        FILE* fx = fopen(text_out, "w");
        if (fx) { fwrite(text.data(), 1, text.size(), fx); fclose(fx); }
    }
    if (ft) fclose(ft);
    if (pos_out) {
        FILE* fp = fopen(pos_out, "wb");
        if (!fp) { fprintf(stderr, "cannot open %s\n", pos_out); return 1; }
        fwrite(selected.data(), 4, selected.size(), fp);
        fclose(fp);
        fprintf(stderr, "positions -> %s: %zu of %zu (classes: %s)\n",
                pos_out, selected.size(), ids.size(), pos_classes);
    }
    printf("%zu tokens, %zu bytes: out %ld, think %ld, tool %ld, prompt %ld\n",
           ids.size(), text.size(), n[0], n[1], n[2], n[3]);
    return 0;
}
