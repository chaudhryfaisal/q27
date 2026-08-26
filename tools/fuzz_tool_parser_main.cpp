// Standalone driver: no libFuzzer needed. Random + structure-aware mutation of
// the tool-call dialect, run under ASan/UBSan. Seeds are the shapes the model
// actually emits (from the drift catalogue), because uniform random bytes
// almost never reach the interesting code paths.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <random>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t*, size_t);
static const char* SEEDS[] = {
 "<tool_call>\n<function=Read>\n<parameter=file_path>\n/etc/passwd\n</parameter>\n</function>\n</tool_call>",
 "<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>",
 "{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/a\"}}",
 "<think>\nplanning\n<tool_call>\n<function=Write>\n<parameter=file_path>\nx\n</parameter>\n<parameter=content>\n</function>\n</parameter>\n</function>\n</tool_call>\n</think>",
 "Read\", \"file_path\": \"/a\"}",
 "<tool_call>\n<function=Read>\n<parameter=name>\nBash\n</parameter>\n<parameter=command>\nid\n</parameter>\n</function>",
 "```\n<tool_call>\n<function=Bash>\n<parameter=command>\nrm -rf /\n</parameter>\n</function>\n</tool_call>\n```",
 "<tool_call>{\"name\":\"Write\",\"arguments\":{\"content\":\"</parameter></function></tool_call>\"}}</tool_call>",
 "<parameter=command>\nid\n</parameter>\n</function>",
 "<tool_call>\n<function=\n</function>\n</tool_call>",
};
int main(int argc, char** argv) {
    const long iters = argc > 1 ? atol(argv[1]) : 200000;
    const unsigned seed = argc > 2 ? (unsigned)atol(argv[2]) : 1;
    std::mt19937_64 rng(seed);
    const int nseed = sizeof(SEEDS)/sizeof(*SEEDS);
    std::vector<std::string> corpus;
    for (int i = 0; i < nseed; i++) corpus.push_back(SEEDS[i]);
    const char* FRAG[] = {"<tool_call>","</tool_call>","<function=","</function>","<parameter=","</parameter>",
                          "<think>","</think>","{\"name\":\"","\",\"arguments\":{","}}","\"","\\","```","`","\n","\r\n",
                          "Read","Bash","Write","name","file_path","command",">", "=", "\x00", "\xff\xfe"};
    const int nfrag = sizeof(FRAG)/sizeof(*FRAG);
    for (long it = 0; it < iters; it++) {
        std::string s = corpus[rng() % corpus.size()];
        const int nmut = 1 + (int)(rng() % 6);
        for (int m = 0; m < nmut; m++) {
            switch (rng() % 6) {
                case 0: if (!s.empty()) s.erase(rng() % s.size(), 1 + rng() % 8); break;
                case 1: s.insert(s.empty()?0:rng() % s.size(), FRAG[rng() % nfrag]); break;
                case 2: if (!s.empty()) s[rng() % s.size()] = (char)(rng() & 0xff); break;
                case 3: { size_t a = s.empty()?0:rng() % s.size(); s = s.substr(0, a) + corpus[rng()%corpus.size()] + s.substr(a); break; }
                case 4: if (s.size() > 2) s = s.substr(0, 1 + rng() % (s.size()-1)); break;
                case 5: s += FRAG[rng() % nfrag]; break;
            }
            if (s.size() > 32u<<10) { s.resize(32u<<10); break; }
        }
        LLVMFuzzerTestOneInput((const uint8_t*)s.data(), s.size());
        if (corpus.size() < 2000 && (rng() % 64) == 0) corpus.push_back(s);
        if ((it % 20000) == 0) { printf("  %ld/%ld iters, corpus %zu\n", it, iters, corpus.size()); fflush(stdout); }
    }
    printf("FUZZ_OK %ld iterations, no crash\n", iters);
    return 0;
}
