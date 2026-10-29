#ifndef GOTORCH_GPU_KERNELS_HPP
#define GOTORCH_GPU_KERNELS_HPP

#include <cstddef>

template <typename T>
class ndarray;
using NDArray = ndarray<float>;

namespace gotorch::gpu {

/**
 * @brief Default optimized matrix multiplication executed on AMD GPU (tiled LDS).
 *
 * @param a Left-hand operand tensor.
 * @param b Right-hand operand tensor.
 * @return NDArray Product tensor on GPU.
 * @throw std::invalid_argument If dimensions or shapes are incompatible.
 */
NDArray matmul(const NDArray& a, const NDArray& b);

/**
 * @brief Naive matrix multiplication executed on AMD GPU (un-tiled baseline).
 *
 * @param a Left-hand operand tensor.
 * @param b Right-hand operand tensor.
 * @return NDArray Product tensor on GPU.
 * @throw std::invalid_argument If dimensions or shapes are incompatible.
 */
NDArray matmul_naive(const NDArray& a, const NDArray& b);

/**
 * @brief Tiled matrix multiplication executed on AMD GPU using LDS shared memory.
 *
 * @param a Left-hand operand tensor.
 * @param b Right-hand operand tensor.
 * @return NDArray Product tensor on GPU.
 * @throw std::invalid_argument If dimensions or shapes are incompatible.
 */
NDArray matmul_tiled(const NDArray& a, const NDArray& b);

/**
 * @brief Memory-coalesced tiled matrix multiplication executed on AMD GPU.
 *
 * Employs Wave32-aligned LDS tile blocks and bank-conflict padding to guarantee
 * 100% coalesced 128-byte global memory transactions across wavefront lanes.
 *
 * @param a Left-hand operand tensor.
 * @param b Right-hand operand tensor.
 * @return NDArray Product tensor on GPU.
 * @throw std::invalid_argument If dimensions or shapes are incompatible.
 */
NDArray matmul_coalesced(const NDArray& a, const NDArray& b);

/**
 * @brief 2D block-tiled matrix multiplication with thread coarsening and coalesced vectorization.
 *
 * Employs 2D register tiling (4x4 per-thread accumulation), vectorized float4 global
 * loads, and cooperative LDS cache blocking.
 *
 * @param a Left-hand operand tensor.
 * @param b Right-hand operand tensor.
 * @return NDArray Product tensor on GPU.
 * @throw std::invalid_argument If dimensions or shapes are incompatible.
 */
NDArray matmul_coalesced_coarsened(const NDArray& a, const NDArray& b);

}  // namespace gotorch::gpu

#endif  // GOTORCH_GPU_KERNELS_HPP
