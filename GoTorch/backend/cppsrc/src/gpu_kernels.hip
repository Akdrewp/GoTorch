#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

#include "gotorch/gpu_kernels.hpp"
#include "gotorch/ndarray.hpp"

namespace gotorch::gpu {

/**
 * @brief Naive matrix multiplication kernel where each thread computes one output element.
 *
 * @param A Left-hand matrix buffer (M x K).
 * @param B Right-hand matrix buffer (K x N).
 * @param C Output matrix buffer (M x N).
 * @param M Row count of A and C.
 * @param N Column count of B and C.
 * @param K Inner shared dimension.
 */
__global__ void matrix_multiplication_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t M,
    size_t N,
    size_t K
) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

NDArray matmul(const NDArray& a, const NDArray& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        throw std::invalid_argument(
            "matmul requires 2D arrays, got " +
            std::to_string(a.shape.size()) + "D and " +
            std::to_string(b.shape.size()) + "D"
        );
    }

    size_t M = a.shape[0];
    size_t K1 = a.shape[1];
    size_t K2 = b.shape[0];
    size_t N = b.shape[1];

    if (K1 != K2) {
        throw std::invalid_argument(
            "shape mismatch: inner dimensions (" +
            std::to_string(K1) + " and " + std::to_string(K2) + ") must match"
        );
    }

    NDArray a_contig = a.is_contiguous() ? a : a.to(Device::CPU).to(Device::GPU);
    NDArray b_contig = b.is_contiguous() ? b : b.to(Device::CPU).to(Device::GPU);

    auto out_storage = std::make_shared<Storage<float>>(M * N, get_gpu_allocator());
    NDArray out(out_storage, {M, N}, NDArray::default_strides({M, N}), 0);

    constexpr unsigned int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (static_cast<unsigned int>(N) + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (static_cast<unsigned int>(M) + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    matrix_multiplication_naive<<<grid, block, 0, 0>>>(
        a_contig.data(),
        b_contig.data(),
        out.data(),
        M,
        N,
        K1
    );

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(
            std::string("HIP naive kernel launch failed: ") + hipGetErrorString(err)
        );
    }
    return out;
}

}  // namespace gotorch::gpu
