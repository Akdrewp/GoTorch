"""Profiles GoTorch GPU matrix multiplication (GEMM) using rocprof / rocprofv3."""

from __future__ import annotations

import argparse
import random

from GoTorch.backend.native_backend import NDArray

from GoTorch import Device


def profile_gpu_gemm(size: int = 1024, warmup: int = 5, iterations: int = 20) -> None:
    """Allocates tensors on GPU and executes matrix multiplication dispatches.

    Args:
        size: Dimension N of the square matrix (N x N).
        warmup: Number of warmup kernel dispatches before measurement.
        iterations: Number of profiled kernel dispatches.
    """
    print(f"Preparing {size}x{size} matrices for GoTorch GPU GEMM...", flush=True)
    raw_a = [random.uniform(-1.0, 1.0) for _ in range(size * size)]
    raw_b = [random.uniform(-1.0, 1.0) for _ in range(size * size)]

    # Transfer input tensors to AMD GPU
    a_gpu = NDArray(raw_a, shape=(size, size)).to(Device.GPU)
    b_gpu = NDArray(raw_b, shape=(size, size)).to(Device.GPU)

    print(f"Executing {warmup} warmup iterations...", flush=True)
    for _ in range(warmup):
        _ = a_gpu.matmul(b_gpu)

    print(f"Executing {iterations} profiled GEMM iterations...", flush=True)
    out_gpu = None
    for _ in range(iterations):
        out_gpu = a_gpu.matmul(b_gpu)

    # Sync result back to host to ensure all queued kernels finalize
    if out_gpu is not None:
        _ = out_gpu.to(Device.CPU)

    print("GPU GEMM completed successfully.", flush=True)


def main() -> None:
    """CLI entry point for profiling GPU GEMM."""
    parser = argparse.ArgumentParser(
        description="Profile GoTorch GPU GEMM under rocprof / rocprofv3"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Square matrix dimension N x N (default: 1024)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations before profiling (default: 5)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of iterations to profile (default: 20)",
    )

    args = parser.parse_args()
    profile_gpu_gemm(size=args.size, warmup=args.warmup, iterations=args.iterations)


if __name__ == "__main__":
    main()
