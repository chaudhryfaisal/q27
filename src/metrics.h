// metrics.h -- q27 Prometheus text-exposition metrics for GET /metrics.
//
// One observation per generation request, accumulated where the [req] line
// fires (gs is final there). Counters are monotonic (Prometheus consumers
// derive rates with rate()/delta()); latency percentiles come from
// histograms (cumulative buckets, +Inf == count by construction).
//
// Naming: q27_* series only, no ds4_/vllm_ compatibility aliases -- the
// stock sparkDash parser keys on those exact prefixes and will not pick
// these up; a Prometheus/Grafana setup (or an extended sparkDash parser)
// is the intended consumer.
//
// Thread contract: observe() runs on request threads, render() on the
// httplib serving thread. Everything shared is std::atomic; the gauge
// snapshot (inflight slots, KV pool, spec counters) is assembled by the
// caller under the server's route_m.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace q27 {
namespace metrics {

using ull = unsigned long long;

// API-shape label. Internal rt.api spellings (oai/cmpl/anth/resp) map to
// the public names below via api_from_str.
enum Api : int { ApiChat = 0, ApiCompletions, ApiMessages, ApiResponses, ApiCount };
constexpr const char* kApiNames[ApiCount] = {"chat", "completions", "messages",
                                             "responses"};

inline Api api_from_str(const char* s) {
    if (!s) return ApiChat;
    if (!std::strcmp(s, "oai")) return ApiChat;
    if (!std::strcmp(s, "cmpl")) return ApiCompletions;
    if (!std::strcmp(s, "anth")) return ApiMessages;
    if (!std::strcmp(s, "resp")) return ApiResponses;
    return ApiChat;
}

// Histogram with fixed finite bucket edges (seconds, ascending) and an
// implicit +Inf bucket. observe() bumps the first bucket whose le covers
// the value; render accumulates buckets into the cumulative form Prometheus
// expects. count is incremented per observation, sum as integer microseconds
// (no floating-point atomics).
struct Histogram {
    static constexpr int kMaxBuckets = 12;
    const double* edges = nullptr; // finite upper bounds, seconds, sorted
    int n = 0;                     // count of finite edges; bucket[n] = +Inf
    std::atomic<ull> bucket[kMaxBuckets] = {};
    std::atomic<ull> count = 0;
    std::atomic<long long> sum_us = 0;

    void init(const double* e, int n_) { edges = e; n = n_; }

    void observe(double seconds) {
        if (!(seconds >= 0.0)) seconds = 0.0;
        int i = 0;
        while (i < n && seconds > edges[i]) i++;
        if (i >= kMaxBuckets) i = kMaxBuckets - 1; // belt: never OOB
        bucket[i]++;
        count++;
        sum_us += (long long)std::llround(seconds * 1e6);
    }
};

// Bucket edges per histogram family (seconds).
static constexpr double kTtftEdgesS[] = {0.010, 0.050, 0.100, 0.250, 0.500,
                                        1.0,   2.5,   5.0}; // prefill wall
static constexpr double kE2eEdgesS[] = {0.100, 0.250, 0.500, 1.0, 2.5, 5.0,
                                        10.0,  30.0,  60.0}; // arrival -> gen done
static constexpr double kItlEdgesS[] = {0.001, 0.002, 0.003, 0.004, 0.005, 0.008,
                                        0.010, 0.015}; // dec_ms / dec per request

// Cumulative per-api counters (ull sums; queue wait stored as microseconds).
struct Counters {
    std::atomic<ull> requests[ApiCount] = {};
    std::atomic<ull> errors[ApiCount] = {};
    std::atomic<ull> prompt_tokens[ApiCount] = {};
    std::atomic<ull> prefill_computed[ApiCount] = {};
    std::atomic<ull> prefill_cached[ApiCount] = {};
    std::atomic<ull> decode_tokens[ApiCount] = {};
    std::atomic<ull> queue_wait_us[ApiCount] = {};
};

// Server-state gauges, assembled by the /metrics handler under route_m.
struct GaugeSnapshot {
    int requests_inflight = 0;     // slots currently claimed (busy)
    int slots_total = 0;
    double kv_usage_perc = -1.0;   // -1 = KV pool inactive -> omit series
    int prefix_cache_entries = -1; // -1 = prefix cache disabled -> omit
    ull spec_draft = 0;            // cumulative MTP lanes fired (engines)
    ull spec_accepted = 0;         // cumulative MTP lanes accepted
    double uptime_seconds = 0.0;
};

struct Metrics {
    Counters counters;
    Histogram ttft[ApiCount]; // prefill wall -> first token
    Histogram e2e[ApiCount];  // arrival -> generation complete
    Histogram itl[ApiCount];  // per-request mean inter-token latency

    Metrics() {
        for (int a = 0; a < ApiCount; a++) {
            ttft[a].init(kTtftEdgesS, (int)(sizeof kTtftEdgesS / sizeof kTtftEdgesS[0]));
            e2e[a].init(kE2eEdgesS, (int)(sizeof kE2eEdgesS / sizeof kE2eEdgesS[0]));
            itl[a].init(kItlEdgesS, (int)(sizeof kItlEdgesS / sizeof kItlEdgesS[0]));
        }
    }

    // One call per generation request, at the [req] point. e2e_ms is wall
    // from request arrival to the observation point.
    void observe(Api api, bool is_error, int prompt, int hit, int pfx, int dec,
                 double qw_ms, double pf_ms, double e2e_ms, double dec_ms) {
        if (api < 0 || api >= ApiCount) api = ApiChat;
        if (prompt < 0) prompt = 0;
        if (hit < 0) hit = 0;
        if (pfx < 0) pfx = 0;
        if (dec < 0) dec = 0;
        if (qw_ms < 0.0) qw_ms = 0.0;
        if (pf_ms < 0.0) pf_ms = 0.0;
        if (e2e_ms < 0.0) e2e_ms = 0.0;
        if (dec_ms < 0.0) dec_ms = 0.0;
        counters.requests[api]++;
        if (is_error) counters.errors[api]++;
        counters.prompt_tokens[api] += (ull)prompt;
        counters.prefill_computed[api] += (ull)std::max(0, prompt - hit - pfx);
        counters.prefill_cached[api] += (ull)(hit + pfx);
        counters.decode_tokens[api] += (ull)dec;
        counters.queue_wait_us[api] += (ull)std::llround(qw_ms * 1000.0);
        ttft[api].observe(pf_ms / 1000.0);
        e2e[api].observe(e2e_ms / 1000.0);
        itl[api].observe(dec > 0 ? dec_ms / (1000.0 * dec) : 0.0);
    }

    std::string render(const GaugeSnapshot& g) const {
        std::string s;
        char buf[512];

        auto emit_counter = [&](const char* name, const char* help, const std::atomic<ull>* v,
                                double scale = 1.0) {
            snprintf(buf, sizeof buf, "# HELP %s %s\n# TYPE %s counter\n", name, help, name);
            s += buf;
            for (int a = 0; a < ApiCount; a++) {
                const double val = (double)v[a].load() * scale;
                if (scale == 1.0)
                    snprintf(buf, sizeof buf, "%s{api=\"%s\"} %llu\n", name, kApiNames[a],
                             (ull)val);
                else
                    snprintf(buf, sizeof buf, "%s{api=\"%s\"} %g\n", name, kApiNames[a],
                             val);
                s += buf;
            }
        };
        auto emit_gauge = [&](const char* name, const char* help, double v) {
            snprintf(buf, sizeof buf, "# HELP %s %s\n# TYPE %s gauge\n%s %g\n", name, help,
                     name, name, v);
            s += buf;
        };
        auto emit_counter_scalar = [&](const char* name, const char* help, ull v) {
            snprintf(buf, sizeof buf, "# HELP %s %s\n# TYPE %s counter\n%s %llu\n", name,
                     help, name, name, v);
            s += buf;
        };

        emit_counter("q27_requests_total", "Generation requests ([req] universe).",
                     counters.requests);
        emit_counter("q27_requests_errors_total", "Generation requests ending end=error.",
                     counters.errors);
        emit_counter("q27_prompt_tokens_total", "Prompt tokens sent to prefill (incl. cached).",
                     counters.prompt_tokens);
        emit_counter("q27_prefill_computed_tokens_total",
                     "Prompt tokens actually computed (not served from prefix cache).",
                     counters.prefill_computed);
        emit_counter("q27_prefill_cached_tokens_total",
                     "Prompt tokens served from the prefix cache (RAM or disk).",
                     counters.prefill_cached);
        emit_counter("q27_decode_tokens_total", "Generated tokens delivered.",
                     counters.decode_tokens);
        emit_counter("q27_queue_wait_seconds_total",
                     "Sum of queue wait (slot claim) across requests.", counters.queue_wait_us,
                     1e-6);

        auto emit_histogram = [&](const char* name, const char* help, const Histogram* h) {
            snprintf(buf, sizeof buf, "# HELP %s %s\n# TYPE %s histogram\n", name, help, name);
            s += buf;
            for (int a = 0; a < ApiCount; a++) {
                const Histogram& hh = h[a];
                ull cum = 0;
                for (int i = 0; i < hh.n; i++) {
                    cum += hh.bucket[i].load();
                    snprintf(buf, sizeof buf, "%s_bucket{api=\"%s\",le=\"%.3f\"} %llu\n", name,
                             kApiNames[a], hh.edges[i], cum);
                    s += buf;
                }
                cum += hh.bucket[hh.n].load();
                snprintf(buf, sizeof buf, "%s_bucket{api=\"%s\",le=\"+Inf\"} %llu\n", name,
                         kApiNames[a], cum);
                s += buf;
                snprintf(buf, sizeof buf, "%s_sum{api=\"%s\"} %g\n%s_count{api=\"%s\"} %llu\n",
                         name, kApiNames[a], (double)hh.sum_us.load() / 1e6, name,
                         kApiNames[a], hh.count.load());
                s += buf;
            }
        };

        emit_histogram("q27_ttft_seconds",
                       "Pre-fill wall per request (first-token latency proxy).", ttft);
        emit_histogram("q27_e2e_seconds",
                       "Wall from request arrival to generation complete.", e2e);
        emit_histogram("q27_itl_seconds",
                       "Per-request mean inter-token latency (dec_ms / dec).", itl);

        emit_gauge("q27_requests_inflight", "Slots currently claimed by a generation.",
                   g.requests_inflight);
        emit_gauge("q27_slots_total", "Configured serving slots.", g.slots_total);
        if (g.kv_usage_perc >= 0.0)
            emit_gauge("q27_kv_usage_perc", "Paged KV pool occupancy (0-1).", g.kv_usage_perc);
        if (g.prefix_cache_entries >= 0)
            emit_gauge("q27_prefix_cache_entries", "Persistent prefix cache entries.",
                       g.prefix_cache_entries);
        emit_counter_scalar("q27_spec_draft_tokens_total",
                            "Cumulative MTP draft lanes fired (accept gate).", g.spec_draft);
        emit_counter_scalar("q27_spec_accepted_tokens_total",
                            "Cumulative MTP draft lanes accepted (accept gate).",
                            g.spec_accepted);
        // q27 never preempts: admission is a FIFO queue, so this counter is
        // honest at 0 (sparkDash's Preempts tile reads it, tooltip: "zero is
        // normal when the server is comfortable").
        emit_counter_scalar("q27_preemptions_total",
                            "Cumulative preemptions (q27 never preempts; FIFO queue).", 0);
        emit_gauge("q27_spec_accept_ratio",
                   "Cumulative MTP accept ratio (accepted / drafted).",
                   g.spec_draft > 0 ? (double)g.spec_accepted / (double)g.spec_draft : 0.0);
        emit_gauge("q27_uptime_seconds", "Server uptime.", g.uptime_seconds);

        return s;
    }
};

} // namespace metrics
} // namespace q27
