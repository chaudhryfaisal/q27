// test_manifest -- gate for the tensor manifest added after the 8194a2a review.
//
// Two halves, and both matter:
//
//   POSITIVE. Every shipped artifact handed to it must pass unchanged. The
//   manifest hard-fails the process, so a shape that is right in the code but
//   wrong for a real tier would brick that tier at load. Point this at every
//   .q27 on the box before believing the gate.
//
//   NEGATIVE. manifest_one() is the comparison itself, and it reports rather
//   than exiting, so the failure paths are testable in-process: missing tensor,
//   wrong rank, wrong extent, and a quantized tensor where the engine reads raw
//   floats. A gate nobody has watched fail is not a gate.
//
// Usage: test_manifest model.q27 [more.q27 ...]
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../src/engine.cuh"

static int failures = 0;
static void ok(bool cond, const char* what) {
    printf("  %-58s %s\n", what, cond ? "PASS" : "FAIL");
    if (!cond) failures++;
}

// The manifest needs attn_layer[]; derive it the way the engine does, from the
// meta, so the test exercises the same split the ctor will use.
static void parse_attn(const q27::Model& m, bool* attn_layer) {
    const std::string& mj = m.meta_json;
    size_t p = mj.find("\"attn_layers\": [");
    if (p == std::string::npos) { fprintf(stderr, "no attn_layers in meta\n"); exit(1); }
    p += strlen("\"attn_layers\": [");
    while (p < mj.size() && mj[p] != ']') {
        int v = atoi(mj.c_str() + p);
        if (v >= 0 && v <= N_LAYER) attn_layer[v] = true;
        p = mj.find_first_of(",]", p);
        if (p == std::string::npos) break;
        if (mj[p] == ',') p++;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.q27 [more.q27 ...]\n", argv[0]); return 1; }

    // ---- negative half, driven off the first artifact's real tensors
    {
        q27::Model m = q27::Model::open(argv[1]);
        printf("negative cases (against %s):\n", argv[1]);
        std::vector<std::string> bad;

        bad.clear();
        manifest_one(m, "blk.0.no_such_tensor", {"", TKind::WEIGHT, N_EMBD, 0}, bad);
        ok(bad.size() == 1 && bad[0].find("missing") != std::string::npos,
           "missing tensor is reported");

        bad.clear();
        manifest_one(m, "output_norm.weight", {"", TKind::F32V, N_EMBD + 1, 0}, bad);
        ok(bad.size() == 1 && bad[0].find("shape") != std::string::npos,
           "wrong 1-D extent is reported");

        bad.clear();
        manifest_one(m, "token_embd.weight", {"", TKind::WEIGHT, VOCAB, 0}, bad);
        ok(bad.size() == 1 && bad[0].find("shape") != std::string::npos,
           "wrong RANK (2-D declared 1-D) is reported");

        bad.clear();
        manifest_one(m, "token_embd.weight", {"", TKind::WEIGHT, VOCAB, N_EMBD + 8}, bad);
        ok(bad.size() == 1, "wrong 2-D extent is reported");

        // token_embd is quantized in every shipped tier; asking for it as a raw
        // float vector is exactly the confusion that produces an OOB read.
        bad.clear();
        manifest_one(m, "token_embd.weight", {"", TKind::F32V, VOCAB, N_EMBD}, bad);
        ok(bad.size() == 1 && bad[0].find("must be F32") != std::string::npos,
           "quantized tensor rejected where raw floats are read");

        bad.clear();
        manifest_one(m, "output_norm.weight", {"", TKind::F32V, N_EMBD, 0}, bad);
        ok(bad.empty(), "a correct spec reports nothing");
    }

    // ---- positive half: every artifact named must pass the full manifest.
    // validate_tensor_manifest() exits(1) on mismatch, so reaching the print
    // below IS the assertion.
    for (int i = 1; i < argc; i++) {
        q27::Model m = q27::Model::open(argv[i]);
        bool attn_layer[N_LAYER + 1] = {false};
        parse_attn(m, attn_layer);
        int n_attn = 0;
        for (int il = 0; il < N_LAYER; il++) n_attn += attn_layer[il];
        validate_tensor_manifest(m, attn_layer);
        printf("  %-58s PASS (%d attn / %d gdn)\n", argv[i], n_attn, N_LAYER - n_attn);
    }

    printf(failures ? "\ntest_manifest: %d FAILURE(S)\n" : "\ntest_manifest: ALL PASS\n", failures);
    return failures ? 1 : 0;
}
