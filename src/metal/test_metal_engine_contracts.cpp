#include "metal_engine.h"

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

        q27::MetalEngine::StopCause cause = q27::MetalEngine::StopCause::MaxTokens;
        const uint32_t emitted = engine.stream_from_pending(
            pending, 2, UINT32_MAX, 2,
            [](uint32_t) { return false; }, cause);
        if (emitted != 0 || cause != q27::MetalEngine::StopCause::Cancelled) {
            std::fprintf(stderr, "cancelled MTP stream did not report cancellation\n");
            return 1;
        }

        puts("Metal q4s engine contracts: OK");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
