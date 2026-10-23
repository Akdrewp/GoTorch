#ifndef GOTORCH_GPU_KERNELS_HPP
#define GOTORCH_GPU_KERNELS_HPP

#include <cstddef>

namespace gotorch {

template <typename T>
class ndarray;
using NDArray = ndarray<float>;

namespace gpu {

/**
 * @brief Thin host wrapper around the HIP tiled matrix multiplication kernel.
 *
 * @param d_A Device pointer to matrix A (M x K).
 * @param d_B Device pointer to matrix B (K x N).
 * @param d_C Device pointer to matrix C (M x N).
 * @param M Number of rows of A and C.
 * @param N Number of columns of B and C.
 * @param K Number of columns of A and rows of B.
 */
void launch_gemm_tiled(
    const float* d_A,
    const float* d_B,
    float* d_C,
    size_t M,
    size_t N,
    size_t K
);

/**
 * @brief Matrix multiplication executed on AMD GPU.
 *
 * @param a Left-hand operand tensor.
 * @param b Right-hand operand tensor.
 * @return NDArray Product tensor on GPU.
 * @throw std::invalid_argument If dimensions or shapes are incompatible.
 */
NDArray matmul(const NDArray& a, const NDArray& b);

}  // namespace gpu
}  // namespace gotorch

#endif  // GOTORCH_GPU_KERNELS_HPP
