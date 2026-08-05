// Model-free gate for the backend-independent argmax tie contract.
// IEEE-equal values select the lowest index, including -0.0f versus +0.0f.
#include "../src/blocks.cuh"

#include <cstdio>
#include <vector>

int main() {
    constexpr int N = 32769; // spans the multi-block reduction
    float* d_logits = nullptr;
    int *d_plain = nullptr, *d_fused = nullptr;
    float* d_margin = nullptr;
    unsigned long long *d_scratch = nullptr, *d_blk1 = nullptr;
    float* d_blk2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_plain, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fused, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_margin, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_blk1, 128 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_blk2, 128 * sizeof(float)));

    auto run = [&](float first, float second, const char* label) {
        std::vector<float> logits(N, -1.0f);
        logits[0] = first;
        logits[256] = second; // different block: forces the packed reduction tie-break
        CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), (size_t)N * sizeof(float),
                              cudaMemcpyHostToDevice));
        q27k::argmax(d_logits, N, d_plain, d_scratch);
        q27k::argmax_margin(d_logits, N, d_fused, d_margin, d_blk1, d_blk2);
        int plain = -1, fused = -1;
        CUDA_CHECK(cudaMemcpy(&plain, d_plain, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&fused, d_fused, sizeof(int), cudaMemcpyDeviceToHost));
        const bool ok = plain == 0 && fused == 0;
        std::printf("%s: plain=%d fused=%d expected=0 %s\n", label, plain, fused,
                    ok ? "PASS" : "FAIL");
        return ok;
    };

    bool ok = run(-0.0f, +0.0f, "-0 then +0");
    ok = run(+0.0f, -0.0f, "+0 then -0") && ok;

    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_plain));
    CUDA_CHECK(cudaFree(d_fused));
    CUDA_CHECK(cudaFree(d_margin));
    CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaFree(d_blk1));
    CUDA_CHECK(cudaFree(d_blk2));
    return ok ? 0 : 1;
}
