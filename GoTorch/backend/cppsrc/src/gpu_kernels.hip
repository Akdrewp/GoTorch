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

/**
 * @brief Tiled matrix multiplication kernel using LDS (Local Data Share) shared memory.
 *
 * 1. Allocate shared memory tiles for blocks of matrix A and matrix B.
 * 2. Cooperatively load tiles from global memory into shared LDS memory.
 * 3. Synchronize barrier (__syncthreads) and compute partial dot products.
 * 4. Advance across the shared dimension K by tile steps and write final accumulated sum to C.
 *
 * @param A Left-hand matrix buffer (M x K).
 * @param B Right-hand matrix buffer (K x N).
 * @param C Output matrix buffer (M x N).
 * @param M Row count of A and C.
 * @param N Column count of B and C.
 * @param K Inner shared dimension.
 */
__global__ void matrix_multiplication_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t M,
    size_t N,
    size_t K
) {
    constexpr unsigned int TILE_DIM = 16;
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    size_t row = blockIdx.y * TILE_DIM + ty;
    size_t col = blockIdx.x * TILE_DIM + tx;

    float acc = 0.0f;
    size_t num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (size_t t = 0; t < num_tiles; ++t) {
        size_t a_col = t * TILE_DIM + tx;
        if (row < M && a_col < K) {
            sA[ty][tx] = A[row * K + a_col];
        } else {
            sA[ty][tx] = 0.0f;
        }

        size_t b_row = t * TILE_DIM + ty;
        if (b_row < K && col < N) {
            sB[ty][tx] = B[b_row * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (unsigned int k = 0; k < TILE_DIM; ++k) {
            acc += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

/**
 * @brief Memory-coalesced tiled matrix multiplication kernel using Wave32 alignment.
 *
 * Employs a 32x32 thread block matching AMD Wave32 wavefront width. Each wavefront
 * maps to a single row, ensuring 100% contiguous coalesced 128-byte global memory loads.
 *
 * @param A Left-hand matrix buffer (M x K).
 * @param B Right-hand matrix buffer (K x N).
 * @param C Output matrix buffer (M x N).
 * @param M Row count of A and C.
 * @param N Column count of B and C.
 * @param K Inner shared dimension.
 */
__global__ void matrix_multiplication_coalesced(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t M,
    size_t N,
    size_t K
) {
    constexpr unsigned int TILE_DIM = 32;
    __shared__ float sA[TILE_DIM][TILE_DIM + 1];
    __shared__ float sB[TILE_DIM][TILE_DIM + 1];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    size_t row = blockIdx.y * TILE_DIM + ty;
    size_t col = blockIdx.x * TILE_DIM + tx;

    float acc = 0.0f;
    size_t num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (size_t t = 0; t < num_tiles; ++t) {
        size_t a_col = t * TILE_DIM + tx;
        if (row < M && a_col < K) {
            sA[ty][tx] = A[row * K + a_col];
        } else {
            sA[ty][tx] = 0.0f;
        }

        size_t b_row = t * TILE_DIM + ty;
        if (b_row < K && col < N) {
            sB[ty][tx] = B[b_row * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (unsigned int k = 0; k < TILE_DIM; ++k) {
            acc += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

/**
 * @brief 2D block-tiled GEMM kernel with thread coarsening and vectorized coalesced loads.
 *
 * Each thread block processes a 64x64 output tile using a 16-deep K-step. Each thread
 * computes a 4x4 sub-tile in VGPR registers, loading float4 vector words for coalescing.
 *
 * @param A Left-hand matrix buffer (M x K).
 * @param B Right-hand matrix buffer (K x N).
 * @param C Output matrix buffer (M x N).
 * @param M Row count of A and C.
 * @param N Column count of B and C.
 * @param K Inner shared dimension.
 */
__global__ void matrix_multiplication_coalesced_coarsened(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t M,
    size_t N,
    size_t K
) {
    constexpr unsigned int BM = 64;
    constexpr unsigned int BN = 64;
    constexpr unsigned int BK = 16;
    constexpr unsigned int TM = 4;
    constexpr unsigned int TN = 4;

    __shared__ float sA[BM][BK + 1];
    __shared__ float sB[BK][BN + 4];

    unsigned int tid = threadIdx.y * (BN / TN) + threadIdx.x;

    unsigned int load_row_A = tid / (BK / 4);
    unsigned int load_col_A = (tid % (BK / 4)) * 4;

    unsigned int load_row_B = tid / (BN / 4);
    unsigned int load_col_B = (tid % (BN / 4)) * 4;

    float acc[TM][TN] = {0.0f};

    size_t num_tiles = (K + BK - 1) / BK;

    for (size_t t = 0; t < num_tiles; ++t) {
        size_t g_row_A = blockIdx.y * BM + load_row_A;
        size_t g_col_A = t * BK + load_col_A;

        bool a_aligned = (reinterpret_cast<uintptr_t>(&A[g_row_A * K + g_col_A]) % sizeof(float4)) == 0;
        if (g_row_A < M && g_col_A + 3 < K && a_aligned) {
            *reinterpret_cast<float4*>(&sA[load_row_A][load_col_A]) =
                *reinterpret_cast<const float4*>(&A[g_row_A * K + g_col_A]);
        } else {
            for (unsigned int i = 0; i < 4; ++i) {
                sA[load_row_A][load_col_A + i] = (g_row_A < M && (g_col_A + i) < K)
                    ? A[g_row_A * K + g_col_A + i] : 0.0f;
            }
        }

        size_t g_row_B = t * BK + load_row_B;
        size_t g_col_B = blockIdx.x * BN + load_col_B;

        bool b_aligned = (reinterpret_cast<uintptr_t>(&B[g_row_B * N + g_col_B]) % sizeof(float4)) == 0;
        if (g_row_B < K && g_col_B + 3 < N && b_aligned) {
            *reinterpret_cast<float4*>(&sB[load_row_B][load_col_B]) =
                *reinterpret_cast<const float4*>(&B[g_row_B * N + g_col_B]);
        } else {
            for (unsigned int i = 0; i < 4; ++i) {
                sB[load_row_B][load_col_B + i] = (g_row_B < K && (g_col_B + i) < N)
                    ? B[g_row_B * N + g_col_B + i] : 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (unsigned int k = 0; k < BK; ++k) {
            float a_frag[TM];
            float b_frag[TN];

            #pragma unroll
            for (unsigned int i = 0; i < TM; ++i) {
                a_frag[i] = sA[threadIdx.y * TM + i][k];
            }

            #pragma unroll
            for (unsigned int j = 0; j < TN; ++j) {
                b_frag[j] = sB[k][threadIdx.x * TN + j];
            }

            #pragma unroll
            for (unsigned int i = 0; i < TM; ++i) {
                #pragma unroll
                for (unsigned int j = 0; j < TN; ++j) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    for (unsigned int i = 0; i < TM; ++i) {
        size_t out_row = blockIdx.y * BM + threadIdx.y * TM + i;
        size_t out_col = blockIdx.x * BN + threadIdx.x * TN;
        if (out_row < M) {
            bool c_aligned = (reinterpret_cast<uintptr_t>(&C[out_row * N + out_col]) % sizeof(float4)) == 0;
            if (out_col + 3 < N && c_aligned) {
                float4 out_val = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
                *reinterpret_cast<float4*>(&C[out_row * N + out_col]) = out_val;
            } else {
                for (unsigned int j = 0; j < TN; ++j) {
                    if (out_col + j < N) {
                        C[out_row * N + out_col + j] = acc[i][j];
                    }
                }
            }
        }
    }
}

namespace {

using KernelPtr = void (*)(const float*, const float*, float*, size_t, size_t, size_t);

/**
 * @brief Validates operand shapes and contiguity, allocating the output NDArray.
 *
 * @param a Left-hand operand tensor.
 * @param b Right-hand operand tensor.
 * @param M Output rows count.
 * @param N Output columns count.
 * @param K Inner shared dimension.
 * @param a_contig Contiguous copy or view of tensor a.
 * @param b_contig Contiguous copy or view of tensor b.
 * @return NDArray Allocated output tensor on GPU.
 * @throw std::invalid_argument If shapes or inner dimensions mismatch.
 */
NDArray prepare_gemm(
    const NDArray& a,
    const NDArray& b,
    size_t& M,
    size_t& N,
    size_t& K,
    NDArray& a_contig,
    NDArray& b_contig
) {
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        throw std::invalid_argument(
            "matmul requires 2D arrays, got " +
            std::to_string(a.shape.size()) + "D and " +
            std::to_string(b.shape.size()) + "D"
        );
    }

    M = a.shape[0];
    K = a.shape[1];
    size_t K2 = b.shape[0];
    N = b.shape[1];

    if (K != K2) {
        throw std::invalid_argument(
            "shape mismatch: inner dimensions (" +
            std::to_string(K) + " and " + std::to_string(K2) + ") must match"
        );
    }

    a_contig = a.is_contiguous() ? a : a.to(Device::CPU).to(Device::GPU);
    b_contig = b.is_contiguous() ? b : b.to(Device::CPU).to(Device::GPU);

    auto out_storage = std::make_shared<Storage<float>>(M * N, get_gpu_allocator());
    return NDArray(out_storage, {M, N}, NDArray::default_strides({M, N}), 0);
}

}  // namespace

NDArray matmul_naive(const NDArray& a, const NDArray& b) {
    size_t M = 0, N = 0, K = 0;
    NDArray a_contig, b_contig;
    NDArray out = prepare_gemm(a, b, M, N, K, a_contig, b_contig);

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
        K
    );

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(
            std::string("HIP kernel launch failed for matrix_multiplication_naive: ") + hipGetErrorString(err)
        );
    }
    return out;
}

NDArray matmul_tiled(const NDArray& a, const NDArray& b) {
    size_t M = 0, N = 0, K = 0;
    NDArray a_contig, b_contig;
    NDArray out = prepare_gemm(a, b, M, N, K, a_contig, b_contig);

    constexpr unsigned int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (static_cast<unsigned int>(N) + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (static_cast<unsigned int>(M) + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    matrix_multiplication_tiled<<<grid, block, 0, 0>>>(
        a_contig.data(),
        b_contig.data(),
        out.data(),
        M,
        N,
        K
    );

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(
            std::string("HIP kernel launch failed for matrix_multiplication_tiled: ") + hipGetErrorString(err)
        );
    }
    return out;
}

NDArray matmul_coalesced(const NDArray& a, const NDArray& b) {
    size_t M = 0, N = 0, K = 0;
    NDArray a_contig, b_contig;
    NDArray out = prepare_gemm(a, b, M, N, K, a_contig, b_contig);

    constexpr unsigned int BLOCK_SIZE = 32;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (static_cast<unsigned int>(N) + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (static_cast<unsigned int>(M) + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    matrix_multiplication_coalesced<<<grid, block, 0, 0>>>(
        a_contig.data(),
        b_contig.data(),
        out.data(),
        M,
        N,
        K
    );

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(
            std::string("HIP kernel launch failed for matrix_multiplication_coalesced: ") + hipGetErrorString(err)
        );
    }
    return out;
}

NDArray matmul_coalesced_coarsened(const NDArray& a, const NDArray& b) {
    size_t M = 0, N = 0, K = 0;
    NDArray a_contig, b_contig;
    NDArray out = prepare_gemm(a, b, M, N, K, a_contig, b_contig);

    constexpr unsigned int BM = 64;
    constexpr unsigned int BN = 64;
    constexpr unsigned int TM = 4;
    constexpr unsigned int TN = 4;

    dim3 block(BN / TN, BM / TM);
    dim3 grid(
        (static_cast<unsigned int>(N) + BN - 1) / BN,
        (static_cast<unsigned int>(M) + BM - 1) / BM
    );

    matrix_multiplication_coalesced_coarsened<<<grid, block, 0, 0>>>(
        a_contig.data(),
        b_contig.data(),
        out.data(),
        M,
        N,
        K
    );

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(
            std::string("HIP kernel launch failed for matrix_multiplication_coalesced_coarsened: ") + hipGetErrorString(err)
        );
    }
    return out;
}

NDArray matmul(const NDArray& a, const NDArray& b) {
    return matmul_coalesced_coarsened(a, b);
}

}  // namespace gotorch::gpu
