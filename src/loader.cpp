#include "loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace q27 {

static constexpr uint32_t MAGIC = 0x46373251; // "Q27F" LE
static constexpr uint32_t VERSION = 1;
static constexpr uint64_t ALIGN = 256;

const char* dtype_name(DType t) {
    switch (t) {
        case DType::F32:     return "F32";
        case DType::F16:     return "F16";
        case DType::Q8_G128: return "Q8_G128";
        case DType::Q4_G64:  return "Q4_G64";
        case DType::T2_G128: return "T2_G128";
        case DType::T3_G128: return "T3_G128";
        case DType::B1_G128: return "B1_G128";
    }
    return "?";
}

std::string validate_tensor_payload(const Tensor& tensor) {
    const uint64_t rows = tensor.rows(), cols = tensor.cols();
    uint64_t group = 0;
    if (tensor.dtype == DType::Q4_G64) group = 64;
    else if (tensor.dtype == DType::Q8_G128 || tensor.dtype == DType::T2_G128 ||
             tensor.dtype == DType::T3_G128 || tensor.dtype == DType::B1_G128)
        group = 128;
    if (group && cols % group)
        return "cols " + std::to_string(cols) +
               " not divisible by group " + std::to_string(group);

    auto checked_product = [](std::initializer_list<uint64_t> factors, uint64_t& out) {
        out = 1;
        for (uint64_t factor : factors) {
            if (factor && out > std::numeric_limits<uint64_t>::max() / factor)
                return false;
            out *= factor;
        }
        return true;
    };
    uint64_t want_data = 0, want_scales = 0;
    bool sizes_representable = true;
    switch (tensor.dtype) {
        case DType::F32:     sizes_representable = checked_product({rows, cols, 4}, want_data); break;
        case DType::F16:     sizes_representable = checked_product({rows, cols, 2}, want_data); break;
        case DType::Q8_G128: sizes_representable = checked_product({rows, cols}, want_data) &&
            checked_product({rows, cols / 128, 2}, want_scales); break;
        case DType::Q4_G64:  sizes_representable = checked_product({rows, cols / 2}, want_data) &&
            checked_product({rows, cols / 64, 2}, want_scales); break;
        case DType::T2_G128: sizes_representable = checked_product({rows, cols / 4}, want_data) &&
            checked_product({rows, cols / 128, 2}, want_scales); break;
        case DType::T3_G128: sizes_representable = checked_product({rows, cols / 128, 26}, want_data) &&
            checked_product({rows, cols / 128, 2}, want_scales); break;
        case DType::B1_G128: sizes_representable = checked_product({rows, cols / 8}, want_data) &&
            checked_product({rows, cols / 128, 2}, want_scales); break;
        default:
            return "unsupported dtype " +
                   std::to_string(static_cast<unsigned>(tensor.dtype));
    }
    if (!sizes_representable) return "packed payload size overflows uint64";
    if (tensor.data_size != want_data || tensor.scales_size != want_scales)
        return "data " + std::to_string(tensor.data_size) +
               " (want " + std::to_string(want_data) + "), scales " +
               std::to_string(tensor.scales_size) + " (want " +
               std::to_string(want_scales) + ")";
    if (want_data && !tensor.data) return "data payload is missing";
    if (want_scales && !tensor.scales) return "scale payload is missing";


    if (tensor.dtype == DType::T2_G128) {
        for (uint64_t i = 0; i < tensor.data_size; i++) {
            const uint8_t byte = tensor.data[i];
            for (int shift = 0; shift < 8; shift += 2)
                if (((byte >> shift) & 3u) == 3u)
                    return "T2 payload contains reserved code 3";
        }
    } else if (tensor.dtype == DType::T3_G128) {
        const uint64_t groups = rows * (cols / 128);
        for (uint64_t group_index = 0; group_index < groups; group_index++) {
            const uint8_t* packed = tensor.data + group_index * 26;
            for (int i = 0; i < 26; i++)
                if (packed[i] > 242) return "T3 payload byte exceeds 242";
            const uint8_t tail = packed[25];
            if ((tail / 27) % 3 != 1 || (tail / 81) % 3 != 1)
                return "T3 final-byte padding is noncanonical";
        }
    }
    return {};
}
bool cuda_weight_dtype_supported(DType dtype) {
    switch (dtype) {
        case DType::F32:
        case DType::F16:
        case DType::Q8_G128:
        case DType::Q4_G64:
            return true;
        case DType::T2_G128:
        case DType::T3_G128:
        case DType::B1_G128:
            return false;
    }
    return false;
}
void validate_cuda_model(const Model& model) {
    const Tensor& embedding=model.get("token_embd.weight");
    if(embedding.dtype!=DType::Q8_G128)
        throw std::runtime_error("q27 CUDA: token_embd.weight must be Q8_G128");
    for(const Tensor& tensor:model.tensors) {
        if(!cuda_weight_dtype_supported(tensor.dtype))
            throw std::runtime_error("q27 CUDA: unsupported weight dtype " +
                                     std::string(dtype_name(tensor.dtype)) +
                                     " in " + tensor.name);
    }
}

uint64_t Tensor::rows() const {
    if (shape.size() <= 1) return 1;
    uint64_t rows = 1;
    for (size_t i = 0; i + 1 < shape.size(); i++) {
        if (shape[i] && rows > std::numeric_limits<uint64_t>::max() / shape[i])
            throw std::runtime_error("q27: tensor row count overflows uint64: " + name);
        rows *= shape[i];
    }
    return rows;
}
uint64_t Tensor::cols() const { return shape.empty() ? 0 : shape.back(); }
uint64_t Tensor::n_elements() const {
    const uint64_t row_count = rows(), column_count = cols();
    if (column_count && row_count > std::numeric_limits<uint64_t>::max() / column_count)
        throw std::runtime_error("q27: tensor element count overflows uint64: " + name);
    return row_count * column_count;
}

const Tensor* Model::find(const std::string& name) const {
    auto it = index.find(name);
    return it == index.end() ? nullptr : &tensors[it->second];
}
const Tensor& Model::get(const std::string& name) const {
    const Tensor* t = find(name);
    if (!t) throw std::runtime_error("q27: missing tensor: " + name);
    return *t;
}

Model::Model(Model&& o) noexcept { *this = std::move(o); }
Model& Model::operator=(Model&& o) noexcept {
    if (this != &o) {
        // release our mmap; do NOT call ~Model() -- that ends the lifetimes of
        // meta_json/tensors/index, which the moves below then write into (UB).
        if (map_base_) munmap(map_base_, map_size_);
        meta_json = std::move(o.meta_json);
        tensors = std::move(o.tensors);
        index = std::move(o.index);
        map_base_ = o.map_base_; map_size_ = o.map_size_;
        o.map_base_ = nullptr; o.map_size_ = 0;
    }
    return *this;
}
Model::~Model() {
    if (map_base_) munmap(map_base_, map_size_);
    map_base_ = nullptr;
}

namespace {
struct Cursor {
    const uint8_t* p;
    const uint8_t* end;
    template <typename T> T read() {
        if ((size_t)(end - p) < sizeof(T)) throw std::runtime_error("q27: truncated file");
        T v; std::memcpy(&v, p, sizeof(T)); p += sizeof(T);
        return v;
    }
    void bytes(void* dst, size_t n) {
        if (n > (size_t)(end - p)) throw std::runtime_error("q27: truncated file");
        std::memcpy(dst, p, n); p += n;
    }
};
} // namespace

Model Model::open(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("q27: cannot open " + path);
    struct stat st{};
    if (fstat(fd, &st) != 0) { close(fd); throw std::runtime_error("q27: fstat failed"); }
    size_t sz = (size_t)st.st_size;
    void* base = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) throw std::runtime_error("q27: mmap failed");

    Model m;
    m.map_base_ = base;
    m.map_size_ = sz;

    const uint8_t* b = (const uint8_t*)base;
    Cursor c{b, b + sz};
    if (c.read<uint32_t>() != MAGIC)   throw std::runtime_error("q27: bad magic");
    if (c.read<uint32_t>() != VERSION) throw std::runtime_error("q27: unsupported version");
    uint32_t n_tensors = c.read<uint32_t>();
    uint32_t meta_len  = c.read<uint32_t>();
    m.meta_json.resize(meta_len);
    c.bytes(m.meta_json.data(), meta_len);

    m.tensors.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        Tensor t;
        uint16_t nl = c.read<uint16_t>();
        t.name.resize(nl);
        c.bytes(t.name.data(), nl);
        t.dtype = (DType)c.read<uint8_t>();
        uint8_t nd = c.read<uint8_t>();
        t.shape.resize(nd);
        for (uint8_t d = 0; d < nd; d++) t.shape[d] = c.read<uint64_t>();
        (void)t.n_elements(); // reject overflowing shapes before any backend sees them
        uint64_t doff = c.read<uint64_t>();
        t.data_size   = c.read<uint64_t>();
        uint64_t soff = c.read<uint64_t>();
        t.scales_size = c.read<uint64_t>();
        // stash offsets in pointers temporarily; fixed up after base is known
        t.data   = (const uint8_t*)(uintptr_t)doff;
        t.scales = t.scales_size ? (const uint8_t*)(uintptr_t)soff : nullptr;
        m.tensors.push_back(std::move(t));
    }
    uint64_t table_end = (uint64_t)(c.p - b);
    if (table_end > std::numeric_limits<uint64_t>::max() - (ALIGN - 1))
        throw std::runtime_error("q27: tensor table size overflow");
    uint64_t data_base = (table_end + ALIGN - 1) / ALIGN * ALIGN;
    const uint64_t file_size = (uint64_t)sz;
    auto in_file = [&](uint64_t offset, uint64_t length) {
        return data_base <= file_size && offset <= file_size - data_base &&
               length <= file_size - data_base - offset;
    };

    for (size_t i = 0; i < m.tensors.size(); i++) {
        Tensor& t = m.tensors[i];
        uint64_t doff = (uint64_t)(uintptr_t)t.data;
        if (!in_file(doff, t.data_size))
            throw std::runtime_error("q27: tensor data out of range: " + t.name);
        t.data = b + data_base + doff;
        if (t.scales_size) {
            uint64_t soff = (uint64_t)(uintptr_t)t.scales;
            if (!soff)
                throw std::runtime_error("q27: tensor scale offset is zero: " + t.name);
            if (!in_file(soff, t.scales_size))
                throw std::runtime_error("q27: tensor scales out of range: " + t.name);
            t.scales = b + data_base + soff;
        }
        const std::string payload_error = validate_tensor_payload(t);
        if (!payload_error.empty())
            throw std::runtime_error("q27: invalid tensor payload " + t.name + ": " +
                                     payload_error);
        m.index.emplace(t.name, i);
    }
    return m;
}

} // namespace q27
