#ifndef GOTORCH_GPU_KERNELS_HPP
#define GOTORCH_GPU_KERNELS_HPP

#include <cstddef>

template <typename T>
class ndarray;
using NDArray = ndarray<float>;

namespace gotorch::gpu {

/**
 * @brief Matrix multiplication executed on AMD GPU.
 *
 * @param a Left-hand operand tensor.
 * @param b Right-hand operand tensor.
 * @return NDArray Product tensor on GPU.
 * @throw std::invalid_argument If dimensions or shapes are incompatible.
 */
NDArray matmul(const NDArray& a, const NDArray& b);

}  // namespace gotorch::gpu

#endif  // GOTORCH_GPU_KERNELS_HPP
