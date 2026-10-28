"""Matrix multiplication benchmark: GoTorch (CPU & AMD GPU) vs PyTorch (CPU & GPU)."""

from __future__ import annotations

import random

import torch
from GoTorch.backend.native_backend import NDArray

from benchmarks.utils import benchmark_op, print_comparison_table
from GoTorch import Device


def run_matmul_benchmarks(
    sizes: list[int] | None = None,
) -> None:
    """Benchmarks matrix multiplication across matrix dimensions.

    Compares GoTorch (CPU & GPU) directly against PyTorch (CPU & GPU).

    Args:
        sizes: List of square matrix sizes (N x N) to benchmark.
    """
    if sizes is None:
        sizes = [64, 128, 256, 512, 1024]

    pt_gpu_available = torch.cuda.is_available()

    cpu_headers = [
        "Matrix Size",
        "GoTorch CPU (ms)",
        "PyTorch CPU (ms)",
        "GoTorch CPU Slowdown",
        "GoTorch CPU GFLOPS",
        "PyTorch CPU GFLOPS",
    ]
    cpu_rows: list[list[str]] = []

    gpu_headers = [
        "Matrix Size",
        "GoTorch GPU (ms)",
        "PyTorch GPU (ms)",
        "PyTorch GPU Speedup",
        "GoTorch GPU GFLOPS",
        "PyTorch GPU GFLOPS",
    ]
    gpu_rows: list[list[str]] = []

    full_headers = [
        "Matrix Size",
        "GoTorch CPU (ms)",
        "PyTorch CPU (ms)",
        "GoTorch GPU (ms)",
        "PyTorch GPU (ms)",
    ]
    full_rows: list[list[str]] = []

    print("\nRunning Matrix Multiplication (GEMM) benchmarks...")
    if pt_gpu_available:
        print(f"  PyTorch GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("  PyTorch GPU: Not available (running CPU only for PyTorch)")

    for n in sizes:
        print(f"  Benchmarking size {n}x{n}...")
        raw_a = [random.uniform(-1.0, 1.0) for _ in range(n * n)]
        raw_b = [random.uniform(-1.0, 1.0) for _ in range(n * n)]

        # Adaptive iterations to keep large matrix benchmarks fast
        if n >= 1024:
            cpu_iters = 3
            gpu_iters = 20
        elif n >= 512:
            cpu_iters = 5
            gpu_iters = 20
        else:
            cpu_iters = 15
            gpu_iters = 20

        # Total floating point operations: 2 * N^3
        total_flops = 2.0 * (n**3)

        # 1. GoTorch CPU
        gt_a_cpu = NDArray(raw_a, shape=(n, n))
        gt_b_cpu = NDArray(raw_b, shape=(n, n))
        res_gt_cpu = benchmark_op(
            lambda: gt_a_cpu.matmul(gt_b_cpu),
            warmup=2,
            iterations=cpu_iters,
            name=f"GoTorch CPU {n}x{n}",
        )

        # 2. PyTorch CPU
        pt_a_cpu = torch.tensor(raw_a, dtype=torch.float32).reshape(n, n)
        pt_b_cpu = torch.tensor(raw_b, dtype=torch.float32).reshape(n, n)
        res_pt_cpu = benchmark_op(
            lambda: torch.matmul(pt_a_cpu, pt_b_cpu),
            warmup=3,
            iterations=cpu_iters,
            name=f"PyTorch CPU {n}x{n}",
        )

        # 3. GoTorch GPU (AMD gfx1201)
        gt_a_gpu = gt_a_cpu.to(Device.GPU)
        gt_b_gpu = gt_b_cpu.to(Device.GPU)
        res_gt_gpu = benchmark_op(
            lambda: gt_a_gpu.matmul(gt_b_gpu),
            warmup=5,
            iterations=gpu_iters,
            name=f"GoTorch GPU {n}x{n}",
        )

        # 4. PyTorch GPU
        res_pt_gpu = None
        if pt_gpu_available:
            pt_a_gpu = pt_a_cpu.to("cuda")
            pt_b_gpu = pt_b_cpu.to("cuda")

            def pt_gpu_matmul() -> None:
                torch.matmul(pt_a_gpu, pt_b_gpu)
                torch.cuda.synchronize()

            res_pt_gpu = benchmark_op(
                pt_gpu_matmul,
                warmup=5,
                iterations=gpu_iters,
                name=f"PyTorch GPU {n}x{n}",
            )

        # GFLOPS = FLOPs / (ms * 1e6)
        gt_cpu_gflops = total_flops / (res_gt_cpu.mean_ms * 1e6)
        pt_cpu_gflops = total_flops / (res_pt_cpu.mean_ms * 1e6)
        gt_gpu_gflops = total_flops / (res_gt_gpu.mean_ms * 1e6)

        cpu_slowdown = res_gt_cpu.mean_ms / res_pt_cpu.mean_ms

        # CPU-only comparison row
        cpu_rows.append([
            f"{n}x{n}",
            f"{res_gt_cpu.mean_ms:.3f}",
            f"{res_pt_cpu.mean_ms:.3f}",
            f"{cpu_slowdown:.1f}x slower",
            f"{gt_cpu_gflops:.3f}",
            f"{pt_cpu_gflops:.1f}",
        ])

        # GPU comparison row
        if res_pt_gpu is not None:
            pt_gpu_gflops = total_flops / (res_pt_gpu.mean_ms * 1e6)
            pt_vs_gt = res_gt_gpu.mean_ms / res_pt_gpu.mean_ms
            gpu_rows.append([
                f"{n}x{n}",
                f"{res_gt_gpu.mean_ms:.3f}",
                f"{res_pt_gpu.mean_ms:.3f}",
                f"{pt_vs_gt:.2f}x faster",
                f"{gt_gpu_gflops:.1f}",
                f"{pt_gpu_gflops:.1f}",
            ])
            full_rows.append([
                f"{n}x{n}",
                f"{res_gt_cpu.mean_ms:.3f}",
                f"{res_pt_cpu.mean_ms:.3f}",
                f"{res_gt_gpu.mean_ms:.3f}",
                f"{res_pt_gpu.mean_ms:.3f}",
            ])
        else:
            gpu_rows.append([
                f"{n}x{n}",
                f"{res_gt_gpu.mean_ms:.3f}",
                "N/A",
                "N/A",
                f"{gt_gpu_gflops:.1f}",
                "N/A",
            ])
            full_rows.append([
                f"{n}x{n}",
                f"{res_gt_cpu.mean_ms:.3f}",
                f"{res_pt_cpu.mean_ms:.3f}",
                f"{res_gt_gpu.mean_ms:.3f}",
                "N/A",
            ])

    print_comparison_table("1. CPU GEMM Comparison: GoTorch CPU vs PyTorch CPU", cpu_headers, cpu_rows)
    print_comparison_table("2. GPU GEMM Comparison: GoTorch GPU vs PyTorch GPU", gpu_headers, gpu_rows)
    print_comparison_table("3. Overall GEMM Comparison", full_headers, full_rows)


if __name__ == "__main__":
    run_matmul_benchmarks()
