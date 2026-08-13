#pragma once

#include "../backend.h"

#include <memory>
#include <string>

namespace q27 {

class MetalBackend final : public ComputeBackend {
  public:
    MetalBackend();
    ~MetalBackend() override;
    MetalBackend(const MetalBackend&) = delete;
    MetalBackend& operator=(const MetalBackend&) = delete;

    std::string name() const override;
    static const char* shader_abi_tag();
    std::string shader_source_sha1() const;
    bool gemm_half_enabled() const;
    uint32_t gqa_tile() const;
    uint32_t gqa_block() const;
    uint32_t gqa_threshold() const;
    bool gpu_topk_supported() const;
    bool healthy() const noexcept;
    std::shared_ptr<BackendBuffer> allocate(uint64_t bytes) override;
    // GPU-private allocation (never host-read/written): used for the
    // engines' blocked-GQA partials scratch.
    std::shared_ptr<BackendBuffer> allocate_private(uint64_t bytes);
    void write(BackendBuffer& dst, uint64_t offset, const void* src,
               uint64_t bytes) override;
    void read(const BackendBuffer& src, uint64_t offset, void* dst,
              uint64_t bytes) override;
    void zero(BackendBuffer& dst) override;
    void copy(const BackendBuffer& src, uint64_t src_offset,
              BackendBuffer& dst, uint64_t dst_offset, uint64_t bytes) override;
    BackendTensor upload(const Tensor& tensor) override;
    BackendTensor upload(const Model& model, const Tensor& tensor) override;
    void begin_commands() override;
    void end_commands() override;
    void abort_commands() noexcept override;
    // Mark the shared command queue unusable after host-side work fails
    // following an already committed state mutation.
    void poison() noexcept;
    void matvec(const BackendTensor& weight, const BackendBuffer& x,
                BackendBuffer& y) override;
    void matvec_pair(const BackendTensor& a, BackendBuffer& a_out,
                     const BackendTensor& b, BackendBuffer& b_out,
                     const BackendBuffer& x) override;
    BackendQuantized allocate_quantized(uint32_t count) override;
    void quantize(const BackendBuffer& x, BackendQuantized& out) override;
    void matvec_quantized(const BackendTensor& weight,
                          const BackendQuantized& x, BackendBuffer& y) override;
    void matvec_quantized_pair(const BackendTensor& a, BackendBuffer& a_out,
                               const BackendTensor& b, BackendBuffer& b_out,
                               const BackendQuantized& x) override;
void matmul_quantized(const BackendTensor& weight,const BackendQuantized& x,
                          uint32_t x_rows,BackendBuffer& y) override;
    void embedding_q8(const BackendTensor& weight, uint32_t token,
                      BackendBuffer& out) override;
    void rmsnorm(const BackendBuffer& x, const BackendTensor& weight,
                 BackendBuffer& out, uint32_t n, float eps) override;
    void rmsnorm_quantized(const BackendBuffer& x, const BackendTensor& weight,
                           BackendBuffer& out, uint32_t n, float eps,
                           BackendQuantized& quantized) override;
    void rmsnorm_heads(BackendBuffer& x, const BackendTensor& weight,
                       uint32_t heads, uint32_t head_dim, uint32_t stride,
                       float eps) override;
    void l2norm_heads(BackendBuffer& x, uint32_t heads, uint32_t head_dim,
                      float eps) override;
    void silu_mul(const BackendBuffer& gate, const BackendBuffer& up,
                  BackendBuffer& out, uint32_t n) override;
    void add_inplace(BackendBuffer& x, const BackendBuffer& y, uint32_t n) override;
    void concat(const BackendBuffer& a, uint32_t a_count,
                const BackendBuffer& b, uint32_t b_count,
                BackendBuffer& out) override;
    void sigmoid_gate_mul(BackendBuffer& out, const BackendBuffer& qg,
                          uint32_t heads, uint32_t head_dim) override;
    void rope_neox(BackendBuffer& x, uint32_t heads, uint32_t head_dim,
                   uint32_t n_rot, uint32_t stride, uint32_t position,
                   float freq_base) override;
    void argmax(const BackendBuffer& x, uint32_t n, BackendBuffer& out_index) override;
    void topk(const BackendBuffer& x, uint32_t n, uint32_t k,
              BackendBuffer& values, BackendBuffer& indices, BackendBuffer& count,
              uint64_t x_offset_bytes = 0) override;
    void kv_store(const BackendBuffer& k, const BackendBuffer& v,
                  BackendBuffer& k_cache, BackendBuffer& v_cache,
                  uint32_t position, uint32_t kv_heads, uint32_t head_dim,
                  KvFormat format) override;
    void attention(const BackendBuffer& q, uint32_t q_stride,
                   const BackendBuffer& k_cache, const BackendBuffer& v_cache,
                   BackendBuffer& out, uint32_t seq_len, uint32_t q_heads,
                   uint32_t kv_heads, uint32_t head_dim, float scale,
                   KvFormat format, BackendBuffer* partials) override;
    // Constrained decoding: -inf every logit whose bit is clear in the
    // uint32 bitset at mask_offset (word-aligned) inside masks.
    void mask_logits(BackendBuffer& logits, const BackendBuffer& masks,
                     uint64_t mask_offset, uint32_t n);
    // GPU-resident greedy decode: embedding row selected by a device-side
    // token id (the previous step's argmax output) — no CPU sync between
    // chained decode steps.
    void embedding_from_device(const BackendTensor& weight, const BackendBuffer& token,
                               BackendBuffer& out);
    void kv_store_f16(const BackendBuffer& k, const BackendBuffer& v,
                      BackendBuffer& k_cache, BackendBuffer& v_cache,
                      uint32_t position, uint32_t row_length);
    void turbo_wht(BackendBuffer& x, uint32_t heads, uint32_t stride,
                   bool inverse) override;
    void kv_store_turbo3(const BackendBuffer& k, const BackendBuffer& v,
                         BackendBuffer& k_cache, BackendBuffer& v_cache,
                         uint32_t position, uint32_t kv_heads);
    void attention_turbo3(const BackendBuffer& q, uint32_t q_stride,
                          const BackendBuffer& k_cache, const BackendBuffer& v_cache,
                          BackendBuffer& out,
                          uint32_t seq_len, uint32_t q_heads, uint32_t kv_heads,
                          uint32_t head_dim, float scale,
                          BackendBuffer* partials);
    void attention_f16(const BackendBuffer& q, uint32_t q_stride,
                       const BackendBuffer& k_cache, const BackendBuffer& v_cache,
                       BackendBuffer& out, uint32_t seq_len,
                       uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
                       float scale, BackendBuffer* partials);
    void gdn_gates(const BackendBuffer& alpha, const BackendBuffer& beta_raw,
                   const BackendTensor& ssm_a, const BackendTensor& ssm_dt,
                   BackendBuffer& g, BackendBuffer& beta, uint32_t heads) override;
    void conv_step(const BackendBuffer& ring_src, BackendBuffer& ring_dst,
                   const BackendBuffer& qkv, const BackendTensor& conv_weight,
                   BackendBuffer& out, uint32_t channels) override;
    void delta_step(const BackendBuffer& state_src, BackendBuffer& state_dst,
                    const BackendBuffer& conv, const BackendBuffer& g,
                    const BackendBuffer& beta, BackendBuffer& out,
                    uint32_t value_heads, uint32_t qk_heads,
                    uint32_t head_dim) override;
    void gated_norm_gdn(const BackendBuffer& x, const BackendTensor& weight,
                        const BackendBuffer& gate, BackendBuffer& out,
                        uint32_t heads, uint32_t head_dim, float eps) override;
    void embedding_q8_rows(const BackendTensor& weight, const uint32_t* tokens,
                           uint32_t count, BackendBuffer& out) override;
    void rmsnorm_rows_quantized(const BackendBuffer& x, const BackendTensor& weight,
                                BackendBuffer& out, uint32_t n, uint32_t rows,
                                float eps, BackendQuantized& quantized) override;
    void matvec_f16_pair_rows(const BackendTensor& a, BackendBuffer& a_out,
                              const BackendTensor& b, BackendBuffer& b_out,
                              const BackendBuffer& x, uint32_t rows) override;
    void gdn_gates_rows(const BackendBuffer& alpha, const BackendBuffer& beta_raw,
                        const BackendTensor& ssm_a, const BackendTensor& ssm_dt,
                        BackendBuffer& g, BackendBuffer& beta,
                        uint32_t heads, uint32_t tokens) override;
    void conv_chunk(const BackendBuffer& ring_src, BackendBuffer& ring_dst,
                    const BackendBuffer& qkv,
                    const BackendTensor& conv_weight, BackendBuffer& out,
                    uint32_t channels, uint32_t tokens) override;
    void delta_chunk(const BackendBuffer& state_src, BackendBuffer& state_dst,
                     const BackendBuffer& conv,
                     const BackendBuffer& g, const BackendBuffer& beta,
                     BackendBuffer& out, uint32_t value_heads, uint32_t qk_heads,
                     uint32_t head_dim, uint32_t tokens) override;
    void l2norm_rows(BackendBuffer& x, uint32_t heads, uint32_t head_dim,
                     uint32_t row_stride, uint32_t tokens, float eps) override;
    void rope_neox_rows(BackendBuffer& x, uint32_t heads, uint32_t head_dim,
                        uint32_t n_rot, uint32_t stride, uint32_t row_stride,
                        uint32_t position, uint32_t tokens, float freq_base) override;
    void kv_store_rows(const BackendBuffer& k, const BackendBuffer& v,
                       BackendBuffer& k_cache, BackendBuffer& v_cache,
                       uint32_t position, uint32_t kv_heads, uint32_t head_dim,
                       uint32_t tokens, KvFormat format) override;
    void attention_causal(const BackendBuffer& q, uint32_t q_stride,
                          uint32_t q_row_stride, const BackendBuffer& k_cache,
                          const BackendBuffer& v_cache,
                          BackendBuffer& out, uint32_t base_len, uint32_t q_heads,
                          uint32_t kv_heads, uint32_t head_dim, uint32_t tokens,
                          float scale, KvFormat format, BackendBuffer* partials) override;
    void kv_store_f16_rows(const BackendBuffer& k, const BackendBuffer& v,
                           BackendBuffer& k_cache, BackendBuffer& v_cache,
                           uint32_t position, uint32_t row_length, uint32_t tokens);
    // KV fp16 exception cells (docs/metal/plans/2026-07-17-kv-except-production.md),
    // Metal-only concrete entries (not on the ComputeBackend interface): copy
    // one head's K/V rows into a kv_heads=1 fp16 side cache, and re-run f16
    // attention over that head's query-head window against the side cache,
    // overwriting the production dispatch's output rows. Head offsets ride
    // the buffer bindings; production kernels and shader ABI untouched.
    // codec 0 stores fp16; codec 1 rounds each value onto the e4m3 grid
    // first (hot-cells arm, 2026-07-17-kv-e4m3-hot-cells.md) — still
    // stored as half, values-exact for a real 1-byte e4m3 side cache.
    void kv_store_f16_head_rows_side(const BackendBuffer& k, const BackendBuffer& v,
                                     uint32_t head_offset_elems, uint32_t src_stride,
                                     BackendBuffer& k_side, BackendBuffer& v_side,
                                     uint32_t position, uint32_t row_length, uint32_t tokens,
                                     uint32_t codec = 0);
    void attention_f16_window(const BackendBuffer& q, uint32_t q_stride, uint32_t qh_start,
                              const BackendBuffer& k_side, const BackendBuffer& v_side,
                              BackendBuffer& out, uint32_t seq_len,
                              uint32_t win_heads, uint32_t head_dim, float scale);
    void attention_f16_causal_window(const BackendBuffer& q, uint32_t q_stride, uint32_t q_row_stride,
                                     uint32_t qh_start, const BackendBuffer& k_side,
                                     const BackendBuffer& v_side, BackendBuffer& out,
                                     uint32_t out_row_stride, uint32_t base_len_plus_1,
                                     uint32_t win_heads, uint32_t head_dim,
                                     uint32_t tokens, float scale);
    void kv_store_turbo3_rows(const BackendBuffer& k, const BackendBuffer& v,
                              BackendBuffer& k_cache, BackendBuffer& v_cache,
                              uint32_t position, uint32_t kv_heads, uint32_t tokens);
    void attention_f16_causal(const BackendBuffer& q, uint32_t q_stride,
                              uint32_t q_row_stride, const BackendBuffer& k_cache,
                              const BackendBuffer& v_cache,
                              BackendBuffer& out, uint32_t base_len, uint32_t q_heads,
                              uint32_t kv_heads, uint32_t head_dim, uint32_t tokens,
                              float scale, BackendBuffer* partials);
    void attention_turbo3_causal(const BackendBuffer& q, uint32_t q_stride,
                                 uint32_t q_row_stride, const BackendBuffer& k_cache,
                                 const BackendBuffer& v_cache,
                                 BackendBuffer& out, uint32_t base_len, uint32_t q_heads,
                                 uint32_t kv_heads, uint32_t head_dim, uint32_t tokens,
                                 float scale, BackendBuffer* partials);
    void sigmoid_gate_mul_rows(BackendBuffer& out, const BackendBuffer& qg,
                               uint32_t heads, uint32_t head_dim, uint32_t tokens) override;
    void argmax_rows(const BackendBuffer& x, uint32_t n, uint32_t rows,
                     BackendBuffer& out_indices) override;
    void nll_rows(const BackendBuffer& logits, const BackendBuffer& targets,
                  BackendBuffer& nll, uint32_t n, uint32_t rows) override;
    void synchronize() override;

    // Clears Q27_METAL_PROFILE accumulation (stats, command-buffer and
    // busy/wait counters) so benches can exclude their warmup dispatches
    // from the attribution table. No-op when profiling is disabled.
    void profile_reset();

    uint64_t recommended_working_set_size() const;
    // Live device allocation total (weights are already inside this once the
    // artifact is mapped) — the measured-free half of --ctx auto sizing.
    uint64_t current_allocated_size() const;
    // Effective causal-GQA block size (Q27_METAL_GQA_BLOCK or 1024): engines
    // size their per-engine partials buffer from it at construction.
    uint32_t gqa_block_size() const;
    // Envelope-instrument hooks (docs/metal/plans/2026-07-16-envelope-instrument.md):
    // flip the backend-global reduction-order knobs between two engines'
    // lockstep passes. Instrument use only — production reads the env once.
    // CONTRACT: these are BACKEND-scoped, not per-engine. Engines sharing
    // one backend (Shared mapping) see every flip; only one engine may
    // mutate them, and never while another engine's pass is in flight —
    // the envelope instrument flips them sequentially by design. Concurrent
    // flips would corrupt any A/B attribution riding on them.
    void set_gemm_half(bool enabled);
    void set_gqa_threshold(uint32_t threshold);
    uint64_t max_buffer_length() const;
    bool uses_per_tensor_upload(const Model& model) const;
    uint64_t max_threadgroup_memory_length() const;
    bool supports_quantized_matmul() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace q27
