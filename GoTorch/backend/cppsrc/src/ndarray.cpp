#include "gotorch/ndarray.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

template <typename T>
void ndarray<T>::CreateDeepCopy(const ndarray& other) {
    shape = other.shape;
    strides = other.strides;
    offset = other.offset;
    storage = std::make_shared<Storage<T>>(*other.storage);
}

template <typename T>
void ndarray<T>::CreateNDArray(
    const T* array,
    const std::vector<size_t>& shape_dims,
    const std::vector<size_t>& custom_strides,
    size_t off
) {
    shape = shape_dims;
    offset = off;

    size_t total_size = num_elements(shape);

    if (custom_strides.empty()) {
        strides = default_strides(shape);
    } else {
        strides = custom_strides;
    }

    if (array != nullptr && total_size > 0) {
        storage = std::make_shared<Storage<T>>(array, total_size);
    } else {
        storage = std::make_shared<Storage<T>>(total_size);
    }
}

template <typename T>
ndarray<T>::ndarray(
    const std::vector<T>& vec,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& strides,
    size_t offset
) : offset(0) {
    this->shape = shape;
    this->offset = offset;
    this->strides = strides.empty() ? default_strides(shape) : strides;
    this->storage = std::make_shared<Storage<T>>(vec);
}

template <typename T>
ndarray<T> ndarray<T>::transpose(size_t dim0, size_t dim1) const {
    if (shape.size() < 2) {
        return *this;
    }
    if (dim0 >= shape.size() || dim1 >= shape.size()) {
        throw std::out_of_range("Transpose dimension out of range");
    }
    std::vector<size_t> new_shape = shape;
    std::vector<size_t> new_strides = strides;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);
    return ndarray(storage, std::move(new_shape), std::move(new_strides), offset);
}

template <typename T>
ndarray<T> ndarray<T>::reshape(const std::vector<size_t>& new_shape) const {
    if (!is_contiguous()) {
        throw std::runtime_error("Reshape currently requires a contiguous ndarray");
    }
    if (num_elements(new_shape) != size()) {
        throw std::invalid_argument("Cannot reshape array: total elements must match");
    }
    return ndarray(storage, new_shape, default_strides(new_shape), offset);
}

template <typename T>
ndarray<T> ndarray<T>::to(Device target_device) const {
    if (device() == target_device) {
        return *this;
    }
    Allocator* target_alloc = (target_device == Device::GPU)
        ? gotorch::get_gpu_allocator()
        : gotorch::get_cpu_allocator();

    size_t total_elems = size();
    auto new_storage = std::make_shared<Storage<T>>(total_elems, target_alloc);

    if (is_contiguous()) {
        gotorch::copy_memory(
            new_storage->data(),
            target_device,
            data(),
            device(),
            total_elems * sizeof(T)
        );
        return ndarray<T>(new_storage, shape, strides, 0);
    }

    if (device() == Device::CPU) {
        std::vector<T> host_buf(total_elems);
        std::vector<size_t> coord(shape.size(), 0);
        for (size_t i = 0; i < total_elems; ++i) {
            host_buf[i] = (*this)(coord);
            gotorch::shape_utils::advance_coordinate(coord, shape);
        }
        gotorch::copy_memory(
            new_storage->data(),
            target_device,
            host_buf.data(),
            Device::CPU,
            total_elems * sizeof(T)
        );
    } else {
        ndarray<T> cpu_copy = this->to(Device::CPU);
        return cpu_copy.to(target_device);
    }
    return ndarray<T>(new_storage, shape, default_strides(shape), 0);
}

template <typename T>
ndarray<T> ndarray<T>::unbroadcast(const std::vector<size_t>& target_shape) const {
    return gotorch::shape_utils::unbroadcast(*this, target_shape);
}

// Explicit template instantiations
namespace gotorch {
template class Storage<float>;
template class Storage<double>;
template class Storage<int>;
}  // namespace gotorch

template class ndarray<float>;
template class ndarray<double>;
template class ndarray<int>;
