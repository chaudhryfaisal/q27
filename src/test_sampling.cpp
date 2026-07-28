#include "sampling.h"

#include <cstdio>
#include <random>
#include <vector>

int main() {
    std::vector<float> logits = {-2.0f, 1.0f, 4.0f, 3.0f};
    q27::SamplingParams greedy;
    std::mt19937_64 rng(1);
    if (q27::sample_logits_cpu(logits, greedy, rng) != 2) return 1;

    q27::SamplingParams k1{0.8f, 1.0f, 1, 7};
    for (int i = 0; i < 20; i++)
        if (q27::sample_logits_cpu(logits, k1, rng) != 2) return 1;

    q27::SamplingParams sampled{1.0f, 0.9f, 3, 12345};
    std::mt19937_64 a(sampled.seed), b(sampled.seed);
    for (int i = 0; i < 100; i++)
        if (q27::sample_logits_cpu(logits, sampled, a) !=
            q27::sample_logits_cpu(logits, sampled, b))
            return 1;

    bool rejected = false;
    try {
        q27::sample_logits_cpu(logits, {1.0f, 0.0f, 0, 0}, rng);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected) return 1;

    // NaNs, positive infinity, and fully masked rows cannot define a served
    // distribution. Negative infinity remains a valid mask when at least one
    // finite logit survives.
    {
        std::vector<uint32_t> ids = {0, 1, 2};
        auto rejected_by_all = [&](const std::vector<float>& invalid) {
            int throws = 0;
            try { (void)q27::sample_logits_cpu(invalid, greedy, rng); }
            catch (const std::runtime_error&) { throws++; }
            try { (void)q27::sample_candidates_cpu(invalid, ids, 3, greedy, rng); }
            catch (const std::runtime_error&) { throws++; }
            try { (void)q27::build_served_distribution(invalid, greedy); }
            catch (const std::runtime_error&) { throws++; }
            try { (void)q27::build_served_from_candidates(
                invalid.data(), ids.data(), 3, greedy); }
            catch (const std::runtime_error&) { throws++; }
            return throws == 4;
        };
        const float inf = std::numeric_limits<float>::infinity();
        if (!rejected_by_all({std::numeric_limits<float>::quiet_NaN(), 1.0f, -inf})) return 1;
        if (!rejected_by_all({inf, 1.0f, -inf})) return 1;
        if (!rejected_by_all({-inf, -inf, -inf})) return 1;
        std::vector<float> masked = {-inf, 1.0f};
        if (q27::sample_logits_cpu(masked, greedy, rng) != 1) return 1;
    }

    // A shuffled exact top-k over-set must consume RNG and select tokens
    // identically to the full-logits path.
    {
        const uint32_t n = 4096, k = 40, count = 57;
        std::vector<float> full(n);
        uint32_t state = 2468;
        for (float& value : full) {
            state = state * 1664525u + 1013904223u;
            value = (float)(state >> 8) / 8388608.0f * 12.0f - 6.0f;
        }
        std::vector<uint32_t> rank(n);
        for (uint32_t i = 0; i < n; i++) rank[i] = i;
        std::sort(rank.begin(), rank.end(), [&](uint32_t x, uint32_t y) {
            return full[x] != full[y] ? full[x] > full[y] : x < y;
        });
        std::vector<float> values(count);
        std::vector<uint32_t> indices(count);
        for (uint32_t i = 0; i < count; i++) {
            indices[i] = rank[(i * 17) % count];
            values[i] = full[indices[i]];
        }
        q27::SamplingParams params{0.9f, 0.95f, k, 777};
        std::mt19937_64 full_rng(params.seed), candidate_rng(params.seed);
        for (int i = 0; i < 100; i++) {
            uint32_t want = q27::sample_logits_cpu(full, params, full_rng);
            uint32_t got = q27::sample_candidates_cpu(
                values, indices, count, params, candidate_rng);
            if (got != want || !(candidate_rng == full_rng)) return 1;
        }
    }

    // Exact ties at the top-p cutoff all remain reachable and both sampling
    // paths map the same random draw to the same token.
    {
        std::vector<float> tied = {0.0f, 0.0f, 0.0f, -100.0f};
        std::vector<uint32_t> ids = {0, 1, 2, 3};
        q27::SamplingParams params{1.0f, 0.5f, 0, 9001};
        std::mt19937_64 full_rng(params.seed), candidate_rng(params.seed);
        bool seen[3] = {};
        for (int i = 0; i < 300; i++) {
            uint32_t want = q27::sample_logits_cpu(tied, params, full_rng);
            uint32_t got = q27::sample_candidates_cpu(
                tied, ids, (uint32_t)tied.size(), params, candidate_rng);
            if (got != want || got > 2) return 1;
            seen[got] = true;
        }
        if (!(seen[0] && seen[1] && seen[2])) return 1;
    }

    // Residual sampling never re-emits an excluded draft token.
    {
        auto distribution = q27::build_served_distribution(logits, {1.0f, 1.0f, 0, 0});
        std::mt19937_64 residual_rng(99);
        for (int i = 0; i < 500; i++)
            if (q27::sample_served(distribution, residual_rng, 2) == 2) return 1;
    }

    std::puts("CPU sampling: PASS");
    return 0;
}
