#include "metal_engine.h"

#include <cstring>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

namespace {

bool rejects_logits(q27::MetalEngine& engine) {
    try {
        (void)engine.read_logits();
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

bool rejects_pending(q27::MetalEngine& engine, uint32_t pending) {
    try {
        (void)engine.generate_from_pending(pending, 1, 2);
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

bool same_logits(const std::vector<float>& left, const std::vector<float>& right) {
    return left.size() == right.size() &&
           std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s model.q27\n", argv[0]);
        return 2;
    }
    try {
        q27::MetalEngine engine(argv[1], 8);
        if (!rejects_logits(engine)) {
            std::fprintf(stderr, "fresh engine exposed non-resident logits\n");
            return 1;
        }

        const uint32_t pending = engine.ingest_prompt({760}, false, true);
        if (engine.position() != 1 ||
            engine.read_logits().size() != q27::MetalEngine::vocabulary_size()) {
            std::fprintf(stderr, "prompt ingestion did not publish current logits\n");
            return 1;
        }
        auto snapshot = engine.capture_state();

        engine.reset();
        if (!rejects_logits(engine) || !rejects_pending(engine, pending)) {
            std::fprintf(stderr, "reset did not invalidate logits and pending state\n");
            return 1;
        }

        engine.restore_state(*snapshot);
        if (engine.position() != 1 ||
            engine.read_logits().size() != q27::MetalEngine::vocabulary_size()) {
            std::fprintf(stderr, "snapshot restore did not restore logits residency\n");
            return 1;
        }

        const std::vector<float> baseline_logits = engine.read_logits();
        const std::vector<uint32_t> unlimited_sample = engine.generate_sampled_from_logits(
            1, q27::SamplingParams{1.0f, 0.95f, 0, 17});
        if (unlimited_sample.size() != 1 || engine.position() != 1 ||
            !same_logits(engine.read_logits(), baseline_logits)) {
            std::fprintf(stderr, "top_k=0 did not use the non-mutating full-logits path\n");
            return 1;
        }
        const std::vector<uint32_t> baseline_retry =
            engine.generate_from_pending(pending, 2, 2);
        engine.restore_state(*snapshot);

        q27::MetalEngine::StopCause cause = q27::MetalEngine::StopCause::MaxTokens;
        bool sink_called = false;
        uint32_t emitted = engine.stream_from_pending(
            pending, 2, pending, 2,
            [&](uint32_t) { sink_called = true; return true; }, cause);
        if (emitted != 0 || cause != q27::MetalEngine::StopCause::Eos || sink_called ||
            engine.position() != 1 || !same_logits(engine.read_logits(), baseline_logits)) {
            std::fprintf(stderr, "MTP EOS changed unconsumed resident state\n");
            return 1;
        }
        if (engine.generate_from_pending(pending, 2, 2) != baseline_retry) {
            std::fprintf(stderr, "MTP EOS changed retry behavior\n");
            return 1;
        }
        engine.restore_state(*snapshot);

        cause = q27::MetalEngine::StopCause::MaxTokens;
        emitted = engine.stream_from_pending(
            pending, 2, UINT32_MAX, 2,
            [](uint32_t) { return false; }, cause);
        if (emitted != 0 || cause != q27::MetalEngine::StopCause::Cancelled ||
            engine.position() != 1 || !same_logits(engine.read_logits(), baseline_logits)) {
            std::fprintf(stderr, "cancelled MTP stream changed unconsumed resident state\n");
            return 1;
        }
        if (engine.generate_from_pending(pending, 2, 2) != baseline_retry) {
            std::fprintf(stderr, "cancelled MTP stream changed retry behavior\n");
            return 1;
        }

        puts("Metal q4s engine contracts: OK");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
