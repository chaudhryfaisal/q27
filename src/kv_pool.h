// M2b: the process-wide paged KV pool (plan docs/plans/2026-08-16-m2-paged-kv.md,
// M2b section). Two device allocations (K side, V side) carved into 64-row
// pages; per-side free lists hand pages to slot lineages at admission and take
// them back at lineage takeover. THREAD CONTRACT: construction and the boot
// warm grants happen on the single-threaded server boot path; after listen(),
// every reserve/release runs under the server's route_m (the same mutex that
// serializes claim_slot), so the pool carries no lock of its own. Allocation
// happens ONCE, before any Engine is constructed -- never during serving
// (graph capture under cudaStreamCaptureModeGlobal makes allocation from any
// thread illegal, and P3 re-captures during serving).
//
// A page is identified by its index within its side; the device address is
// base + idx * page_bytes. Engines map pages into their d_kv_tab entries
// themselves (kv_map_* in engine.cuh) -- the pool only tracks ownership.
#pragma once
#include <cstdint>
#include <cstdio>
#include <vector>

#include "cuda_common.h"

namespace q27 {

struct KvPool {
    char* base[2] = {nullptr, nullptr}; // [0]=K side, [1]=V side
    size_t page_bytes[2] = {0, 0};      // 64 * per-row bytes for the boot kv_kind
    int npages[2] = {0, 0};
    std::vector<int> free_list[2];      // LIFO of free page indices
    bool enabled() const { return base[0] != nullptr; }

    // side_bytes are rounded DOWN to whole pages. Returns false (pool
    // disabled) when either side gets zero pages -- callers fall back to
    // per-engine self-provisioned KV (the M2a identity path).
    bool init(size_t k_bytes, size_t v_bytes, size_t k_row, size_t v_row) {
        page_bytes[0] = (size_t)KV_PAGE * k_row;
        page_bytes[1] = (size_t)KV_PAGE * v_row;
        for (int s = 0; s < 2; s++) {
            const size_t want = s ? v_bytes : k_bytes;
            npages[s] = (int)(want / page_bytes[s]);
            if (npages[s] <= 0) return false;
        }
        for (int s = 0; s < 2; s++) {
            CUDA_CHECK(cudaMalloc((void**)&base[s], (size_t)npages[s] * page_bytes[s]));
            free_list[s].reserve(npages[s]);
            // LIFO with high indices first so early grants read low addresses
            // (debugging niceness only; ownership is index-based).
            for (int j = npages[s] - 1; j >= 0; j--) free_list[s].push_back(j);
        }
        return true;
    }

    int free_pages(int side) const { return (int)free_list[side].size(); }
    char* page_ptr(int side, int idx) const {
        return base[side] + (size_t)idx * page_bytes[side];
    }

    // Pop n pages into out (append). Caller must have checked free_pages —
    // under route_m the check-then-pop pair is atomic. Returns false (and
    // pops nothing) if n exceeds the free count, as a belt.
    bool reserve(int side, int n, std::vector<int>& out) {
        if ((int)free_list[side].size() < n) return false;
        for (int i = 0; i < n; i++) {
            out.push_back(free_list[side].back());
            free_list[side].pop_back();
        }
        return true;
    }
    void release(int side, std::vector<int>& pages) {
        for (int idx : pages) free_list[side].push_back(idx);
        pages.clear();
    }
};

} // namespace q27
