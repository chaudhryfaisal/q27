// Upload q27 tensors to a CUDA device. Supports selective upload so tests can
// run while another process holds most of the GPU.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "loader.h"

namespace q27 {

struct DevTensor {
    DType dtype = DType::F32;
    uint64_t rows = 0, cols = 0;
    void* data = nullptr;   // device
    void* scales = nullptr; // device, nullptr if none
    uint64_t data_bytes = 0, scales_bytes = 0;
};

class DeviceModel {
  public:
    explicit DeviceModel(const Model& m) : model_(m) {}
    ~DeviceModel();
    // owns raw CUDA pointers in dev_; a copy would double-free on destruction.
    DeviceModel(const DeviceModel&) = delete;
    DeviceModel& operator=(const DeviceModel&) = delete;

    // Upload one tensor (no-op if already resident). Returns the device tensor.
    const DevTensor& upload(const std::string& name);
    // Upload everything (engine path). ".pf4" sidecars (fp4 prefill copies,
    // ~10.5 GB on a --pf4 pack) are skipped unless with_pf4 -- callers pass
    // q27k::pf4_on() so default boots pay nothing for them.
    // drop_shadowed_q4 (Q27_PF4_INSTRUMENT, CLI-only): ALSO skip the base Q4
    // tensors that have a .pf4 sibling -- prefill-only instrument runs
    // (--nll-long, --pf) never decode, and both weight copies do not fit a
    // 32 GB card at instrument depths (measured 31.8/32.6 GiB at ctx 2048).
    void upload_all(bool with_pf4 = false, bool drop_shadowed_q4 = false);

    const DevTensor& get(const std::string& name) const;
    // Nullable get: for optional sidecars (nullptr when never uploaded).
    const DevTensor* try_get(const std::string& name) const {
        auto it = dev_.find(name);
        return it == dev_.end() ? nullptr : &it->second;
    }
    bool model_has(const std::string& name) const { return model_.find(name) != nullptr; }
    // The Model behind this DeviceModel. The engine sizes the vgemm workspace by
    // walking the weight list BEFORE upload_all(), so it needs shapes/dtypes when
    // no DevTensor exists yet.
    const Model& model() const { return model_; }
    size_t bytes_resident() const { return bytes_; }

    // Weight-integrity checksums: order-independent u64 word-sums of every
    // resident tensor, taken once after upload. checksum_verify() recomputes
    // and reports any drift -- detects OC/heat bit flips in resident weights,
    // which token-identity gates cannot see (they compare against the same
    // corrupted state). ~10 ms for the full 16.75 GB model.
    void checksum_baseline();
    int checksum_verify(bool print) const; // returns number of mismatched tensors
    // One digest over every tensor's baseline sum, comparable ACROSS PROCESSES.
    // checksum_verify() only catches drift AFTER the baseline was taken, so an
    // upload that landed wrong is blessed as correct and verifies clean. This
    // is the only way to see a bad upload: the same artifact must produce the
    // same aggregate on every load. Printed at load under Q27_PRINT_WSUM=1.
    unsigned long long checksum_aggregate() const;
    // HOST-side twin of the above, summed on the CPU from the SOURCE bytes
    // immediately before each cudaMemcpy (enabled by Q27_PRINT_WSUM=1, since it
    // touches the whole model on the CPU). Comparing the two localizes a
    // corrupt load: host==device but both varying across runs => the bytes were
    // already wrong in host memory (page cache / DRAM / read path); host stable
    // while device varies => the copy or VRAM is at fault.
    unsigned long long host_aggregate() const { return host_sum_; }
    void enable_host_sum(bool on) { want_host_sum_ = on; }
    // Q27_WSUM_LOCATE=1 (2026-08-24): per-tensor host-vs-device comparison
    // after upload. The aggregate digest says A load went wrong; this says
    // WHICH tensor, at which byte offset, which bit, which direction, and
    // which tensors sit adjacent in device memory. That is the evidence that
    // separates an out-of-bounds device write (offset at a buffer edge, a
    // neighbour's name) from a transfer fault (random offset). Returns the
    // number of tensors that mismatched.
    int locate_upload_errors() const;

  private:
    const Model& model_;
    std::unordered_map<std::string, DevTensor> dev_;
    std::unordered_map<std::string, unsigned long long> sums_;
    std::unordered_map<std::string, unsigned long long> host_sums_; // per-tensor, data only
    unsigned long long host_sum_ = 0;
    bool want_host_sum_ = false;
    size_t bytes_ = 0;
};

} // namespace q27
