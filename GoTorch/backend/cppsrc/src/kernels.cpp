#include "gotorch/kernels.hpp"

#include <stdexcept>
#include <type_traits>

#include "gotorch/device.hpp"
#include "gotorch/gpu_kernels.hpp"

template <typename T>
ndarray<T> ndarray<T>::add(const ndarray<T>& addend) const {
    return gotorch::kernels::add(*this, addend);
}

template <typename T>
ndarray<T> ndarray<T>::sub(const ndarray<T>& subtrahend) const {
    return gotorch::kernels::sub(*this, subtrahend);
}

template <typename T>
ndarray<T> ndarray<T>::matmul(const ndarray<T>& other) const {
    if (this->device() != other.device()) {
        throw std::invalid_argument(
            "Device mismatch in matmul: left operand is on " +
            gotorch::device_to_string(this->device()) + " but right operand is on " +
            gotorch::device_to_string(other.device())
        );
    }
    if (this->device() == gotorch::Device::GPU) {
        if constexpr (std::is_same_v<T, float>) {
            return gotorch::gpu::matmul(*this, other);
        } else {
            throw std::runtime_error("GPU matmul is currently only implemented for float");
        }
    }
    return gotorch::kernels::matmul(*this, other);
}

template <typename T>
ndarray<T> ndarray<T>::matmul_naive(const ndarray<T>& other) const {
    if (this->device() != other.device()) {
        throw std::invalid_argument(
            "Device mismatch in matmul_naive: left operand is on " +
            gotorch::device_to_string(this->device()) + " but right operand is on " +
            gotorch::device_to_string(other.device())
        );
    }
    if (this->device() == gotorch::Device::GPU) {
        if constexpr (std::is_same_v<T, float>) {
            return gotorch::gpu::matmul_naive(*this, other);
        } else {
            throw std::runtime_error("GPU matmul_naive is currently only implemented for float");
        }
    }
    return gotorch::kernels::matmul(*this, other);
}

template <typename T>
ndarray<T> ndarray<T>::matmul_tiled(const ndarray<T>& other) const {
    if (this->device() != other.device()) {
        throw std::invalid_argument(
            "Device mismatch in matmul_tiled: left operand is on " +
            gotorch::device_to_string(this->device()) + " but right operand is on " +
            gotorch::device_to_string(other.device())
        );
    }
    if (this->device() == gotorch::Device::GPU) {
        if constexpr (std::is_same_v<T, float>) {
            return gotorch::gpu::matmul_tiled(*this, other);
        } else {
            throw std::runtime_error("GPU matmul_tiled is currently only implemented for float");
        }
    }
    return gotorch::kernels::matmul(*this, other);
}

template <typename T>
ndarray<T> ndarray<T>::matmul_coalesced(const ndarray<T>& other) const {
    if (this->device() != other.device()) {
        throw std::invalid_argument(
            "Device mismatch in matmul_coalesced: left operand is on " +
            gotorch::device_to_string(this->device()) + " but right operand is on " +
            gotorch::device_to_string(other.device())
        );
    }
    if (this->device() == gotorch::Device::GPU) {
        if constexpr (std::is_same_v<T, float>) {
            return gotorch::gpu::matmul_coalesced(*this, other);
        } else {
            throw std::runtime_error("GPU matmul_coalesced is currently only implemented for float");
        }
    }
    return gotorch::kernels::matmul(*this, other);
}

template <typename T>
ndarray<T> ndarray<T>::matmul_coalesced_coarsened(const ndarray<T>& other) const {
    if (this->device() != other.device()) {
        throw std::invalid_argument(
            "Device mismatch in matmul_coalesced_coarsened: left operand is on " +
            gotorch::device_to_string(this->device()) + " but right operand is on " +
            gotorch::device_to_string(other.device())
        );
    }
    if (this->device() == gotorch::Device::GPU) {
        if constexpr (std::is_same_v<T, float>) {
            return gotorch::gpu::matmul_coalesced_coarsened(*this, other);
        } else {
            throw std::runtime_error("GPU matmul_coalesced_coarsened is currently only implemented for float");
        }
    }
    return gotorch::kernels::matmul(*this, other);
}

template <typename T>
ndarray<T> ndarray<T>::relu() const {
    return gotorch::kernels::relu(*this);
}

template <typename T>
ndarray<T> ndarray<T>::relu_backward(const ndarray<T>& grad_output) const {
    return gotorch::kernels::relu_backward(*this, grad_output);
}

template <typename T>
ndarray<T> ndarray<T>::tanh() const {
    return gotorch::kernels::tanh(*this);
}

template <typename T>
ndarray<T> ndarray<T>::tanh_backward(const ndarray<T>& grad_output) const {
    return gotorch::kernels::tanh_backward(*this, grad_output);
}

template <typename T>
ndarray<T> ndarray<T>::sigmoid() const {
    return gotorch::kernels::sigmoid(*this);
}

template <typename T>
ndarray<T> ndarray<T>::sigmoid_backward(const ndarray<T>& grad_output) const {
    return gotorch::kernels::sigmoid_backward(*this, grad_output);
}

template <typename T>
ndarray<T> ndarray<T>::leaky_relu(T alpha) const {
    return gotorch::kernels::leaky_relu(*this, alpha);
}

template <typename T>
ndarray<T> ndarray<T>::leaky_relu_backward(const ndarray<T>& grad_output, T alpha) const {
    return gotorch::kernels::leaky_relu_backward(*this, grad_output, alpha);
}

template <typename T>
ndarray<T> ndarray<T>::sum(int dim, bool keepdim) const {
    return gotorch::kernels::sum(*this, dim, keepdim);
}

template <typename T>
ndarray<T> ndarray<T>::sum() const {
    return gotorch::kernels::sum(*this);
}

// Explicit template instantiations
template class ndarray<float>;
template class ndarray<double>;
template class ndarray<int>;
