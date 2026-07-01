#include <stdexcept>

#include "cuda_common.h"
#include "device_model.h"

namespace q27 {

DeviceModel::~DeviceModel() {
    for (auto& [k, t] : dev_) {
        if (t.data) cudaFree(t.data);
        if (t.scales) cudaFree(t.scales);
    }
}

const DevTensor& DeviceModel::upload(const std::string& name) {
    auto it = dev_.find(name);
    if (it != dev_.end()) return it->second;

    const Tensor& src = model_.get(name);
    DevTensor d;
    d.dtype = src.dtype;
    d.rows = src.rows();
    d.cols = src.cols();
    CUDA_CHECK(cudaMalloc(&d.data, src.data_size));
    CUDA_CHECK(cudaMemcpy(d.data, src.data, src.data_size, cudaMemcpyHostToDevice));
    bytes_ += src.data_size;
    if (src.scales) {
        CUDA_CHECK(cudaMalloc(&d.scales, src.scales_size));
        CUDA_CHECK(cudaMemcpy(d.scales, src.scales, src.scales_size, cudaMemcpyHostToDevice));
        bytes_ += src.scales_size;
    }
    return dev_.emplace(name, d).first->second;
}

void DeviceModel::upload_all() {
    for (const auto& t : model_.tensors) upload(t.name);
}

const DevTensor& DeviceModel::get(const std::string& name) const {
    auto it = dev_.find(name);
    if (it == dev_.end()) throw std::runtime_error("not resident on device: " + name);
    return it->second;
}

} // namespace q27
