"""Profiles PyTorch GPU matrix multiplication on AMD ROCm using torch.profiler."""

from __future__ import annotations

import argparse
import os

import torch
import torch.profiler


def profile_pytorch_gpu_gemm(
    size: int = 1024,
    warmup: int = 5,
    iterations: int = 20,
    output_path: str = "profile_output/pytorch_gemm_trace.json",
) -> None:
    """Profiles PyTorch GPU GEMM on AMD ROCm and exports a Perfetto trace.

    Args:
        size: Dimension N of the square matrix (N x N).
        warmup: Number of untimed warmup passes.
        iterations: Number of profiled GEMM passes.
        output_path: Destination path for the exported Perfetto/Chrome JSON trace.

    Raises:
        RuntimeError: If CUDA/ROCm GPU device is not available in PyTorch.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm / CUDA GPU device is not available in PyTorch.")

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print(f"Allocating {size}x{size} matrices on GPU...")

    a = torch.randn(size, size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, size, device="cuda", dtype=torch.float32)

    print(f"Executing {warmup} warmup iterations...")
    for _ in range(warmup):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    print(f"Profiling {iterations} iterations with torch.profiler...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(iterations):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    prof.export_chrome_trace(output_path)
    print(f"\nTrace successfully saved to: {output_path}")
    print("\nProfiler Summary:\n")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def main() -> None:
    """CLI entry point for profiling PyTorch GPU GEMM."""
    parser = argparse.ArgumentParser(description="Profile PyTorch GPU GEMM")
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Matrix dimension N (default: 1024)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Profiled iterations (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="profile_output/pytorch_gemm_trace.json",
        help="Path for exported trace (default: profile_output/pytorch_gemm_trace.json)",
    )
    args = parser.parse_args()

    profile_pytorch_gpu_gemm(
        size=args.size,
        warmup=args.warmup,
        iterations=args.iterations,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
