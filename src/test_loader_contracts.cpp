#include "loader.h"

#include <cstdio>

int main() {
    using q27::DType;

    for (DType dtype : {DType::F32, DType::F16, DType::Q8_G128, DType::Q4_G64}) {
        if (!q27::cuda_weight_dtype_supported(dtype)) {
            std::fprintf(stderr, "CUDA-compatible dtype rejected: %s\n",
                         q27::dtype_name(dtype));
            return 1;
        }
    }
    for (DType dtype : {DType::T2_G128, DType::T3_G128, DType::B1_G128}) {
        if (q27::cuda_weight_dtype_supported(dtype)) {
            std::fprintf(stderr, "Metal-only packed dtype accepted by CUDA: %s\n",
                         q27::dtype_name(dtype));
            return 1;
        }
    }

    uint8_t data[128] = {};
    uint8_t scales[2] = {};
    q27::Tensor tensor;
    tensor.name = "test.q8";
    tensor.dtype = DType::Q8_G128;
    tensor.shape = {1, 128};
    tensor.data = data;
    tensor.data_size = sizeof(data);
    tensor.scales_size = sizeof(scales);
    if (q27::validate_tensor_payload(tensor) != "scale payload is missing") {
        std::fputs("missing quantization scales were accepted\n", stderr);
        return 1;
    }
    tensor.data = nullptr;
    tensor.scales = scales;
    if (q27::validate_tensor_payload(tensor) != "data payload is missing") {
        std::fputs("missing tensor data were accepted\n", stderr);
        return 1;
    }

    std::puts("loader CUDA dtype contracts: PASS");
    return 0;
}
