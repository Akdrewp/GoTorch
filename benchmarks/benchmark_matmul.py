"""Matrix multiplication benchmark: GoTorch (CPU & AMD GPU) vs PyTorch (CPU & GPU)."""

from __future__ import annotations

import argparse
import random

import torch
from GoTorch.backend.native_backend import NDArray

from benchmarks.utils import benchmark_op, print_comparison_table
from GoTorch import Device


def run_matmul_benchmarks(
    sizes: list[int] | None = None,
    skip_cpu: bool = True,
) -> None:
    """Benchmarks matrix multiplication across matrix dimensions.

    Compares GoTorch (CPU & GPU) directly against PyTorch (CPU & GPU).

    Args:
        sizes: List of square matrix sizes (N x N) to benchmark.
        skip_cpu: If True, skips GoTorch CPU to keep execution fast.
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
        "Naive (ms)",
        "Tiled (ms)",
        "Coalesced (ms)",
        "Coarsened (ms)",
        "PyTorch GPU (ms)",
        "Coarsened GFLOPS",
        "PyTorch GFLOPS",
    ]
    gpu_rows: list[list[str]] = []

    full_headers = [
        "Matrix Size",
        "GoTorch CPU (ms)",
        "PyTorch CPU (ms)",
        "GoTorch Coarsened (ms)",
        "PyTorch GPU (ms)",
    ]
    full_rows: list[list[str]] = []

    print("\nRunning Matrix Multiplication (GEMM) benchmarks...")
    if skip_cpu:
        print("  GoTorch CPU: Skipped (--skip-cpu is active)")
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

        res_gt_cpu = None
        res_pt_cpu = None
        if not skip_cpu:
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

            # GFLOPS = FLOPs / (ms * 1e6)
            gt_cpu_gflops = total_flops / (res_gt_cpu.mean_ms * 1e6)
            pt_cpu_gflops = total_flops / (res_pt_cpu.mean_ms * 1e6)
            cpu_slowdown = res_gt_cpu.mean_ms / res_pt_cpu.mean_ms

            cpu_rows.append([
                f"{n}x{n}",
                f"{res_gt_cpu.mean_ms:.3f}",
                f"{res_pt_cpu.mean_ms:.3f}",
                f"{cpu_slowdown:.1f}x slower",
                f"{gt_cpu_gflops:.3f}",
                f"{pt_cpu_gflops:.1f}",
            ])

        # 3. GoTorch GPU (AMD gfx1201)
        gt_a_gpu = NDArray(raw_a, shape=(n, n)).to(Device.GPU)
        gt_b_gpu = NDArray(raw_b, shape=(n, n)).to(Device.GPU)

        def sync_device() -> None:
            if pt_gpu_available:
                torch.cuda.synchronize()

        def gt_gpu_matmul_naive() -> None:
            _ = gt_a_gpu.matmul_naive(gt_b_gpu)
            sync_device()

        def gt_gpu_matmul_tiled() -> None:
            _ = gt_a_gpu.matmul_tiled(gt_b_gpu)
            sync_device()

        def gt_gpu_matmul_coalesced() -> None:
            _ = gt_a_gpu.matmul_coalesced(gt_b_gpu)
            sync_device()

        def gt_gpu_matmul_coarsened() -> None:
            _ = gt_a_gpu.matmul_coalesced_coarsened(gt_b_gpu)
            sync_device()

        res_gt_gpu_naive = benchmark_op(
            gt_gpu_matmul_naive,
            warmup=5,
            iterations=gpu_iters,
            name=f"GoTorch GPU Naive {n}x{n}",
        )

        res_gt_gpu_tiled = benchmark_op(
            gt_gpu_matmul_tiled,
            warmup=5,
            iterations=gpu_iters,
            name=f"GoTorch GPU Tiled {n}x{n}",
        )

        res_gt_gpu_coalesced = benchmark_op(
            gt_gpu_matmul_coalesced,
            warmup=5,
            iterations=gpu_iters,
            name=f"GoTorch GPU Coalesced {n}x{n}",
        )

        res_gt_gpu_coarsened = benchmark_op(
            gt_gpu_matmul_coarsened,
            warmup=5,
            iterations=gpu_iters,
            name=f"GoTorch GPU Coarsened {n}x{n}",
        )

        # 4. PyTorch GPU
        res_pt_gpu = None
        if pt_gpu_available:
            pt_a_gpu = torch.tensor(raw_a, dtype=torch.float32, device="cuda").reshape(n, n)
            pt_b_gpu = torch.tensor(raw_b, dtype=torch.float32, device="cuda").reshape(n, n)

            def pt_gpu_matmul() -> None:
                torch.matmul(pt_a_gpu, pt_b_gpu)
                torch.cuda.synchronize()

            res_pt_gpu = benchmark_op(
                pt_gpu_matmul,
                warmup=5,
                iterations=gpu_iters,
                name=f"PyTorch GPU {n}x{n}",
            )

        gt_gpu_coarsened_gflops = total_flops / (res_gt_gpu_coarsened.mean_ms * 1e6)

        if res_pt_gpu is not None:
            pt_gpu_gflops = total_flops / (res_pt_gpu.mean_ms * 1e6)
            gpu_rows.append([
                f"{n}x{n}",
                f"{res_gt_gpu_naive.mean_ms:.3f}",
                f"{res_gt_gpu_tiled.mean_ms:.3f}",
                f"{res_gt_gpu_coalesced.mean_ms:.3f}",
                f"{res_gt_gpu_coarsened.mean_ms:.3f}",
                f"{res_pt_gpu.mean_ms:.3f}",
                f"{gt_gpu_coarsened_gflops:.1f}",
                f"{pt_gpu_gflops:.1f}",
            ])
            full_rows.append([
                f"{n}x{n}",
                f"{res_gt_cpu.mean_ms:.3f}" if res_gt_cpu else "Skipped",
                f"{res_pt_cpu.mean_ms:.3f}" if res_pt_cpu else "Skipped",
                f"{res_gt_gpu_coarsened.mean_ms:.3f}",
                f"{res_pt_gpu.mean_ms:.3f}",
            ])
        else:
            gpu_rows.append([
                f"{n}x{n}",
                f"{res_gt_gpu_naive.mean_ms:.3f}",
                f"{res_gt_gpu_tiled.mean_ms:.3f}",
                f"{res_gt_gpu_coalesced.mean_ms:.3f}",
                f"{res_gt_gpu_coarsened.mean_ms:.3f}",
                "N/A",
                f"{gt_gpu_coarsened_gflops:.1f}",
                "N/A",
            ])
            full_rows.append([
                f"{n}x{n}",
                f"{res_gt_cpu.mean_ms:.3f}" if res_gt_cpu else "Skipped",
                f"{res_pt_cpu.mean_ms:.3f}" if res_pt_cpu else "Skipped",
                f"{res_gt_gpu_coarsened.mean_ms:.3f}",
                "N/A",
            ])

    if not skip_cpu:
        print_comparison_table("1. CPU GEMM Comparison: GoTorch CPU vs PyTorch CPU", cpu_headers, cpu_rows)
    print_comparison_table("2. GPU GEMM Comparison: GoTorch GPU (Naive vs Tiled) vs PyTorch GPU", gpu_headers, gpu_rows)
    if not skip_cpu:
        print_comparison_table("3. Overall GEMM Comparison", full_headers, full_rows)


def main() -> None:
    """CLI orchestrator for GEMM benchmark suites."""
    parser = argparse.ArgumentParser(
        description="Benchmark GoTorch vs PyTorch Matrix Multiplication."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        help="Matrix sizes N x N to benchmark (default: 64 128 256 512 1024)",
    )
    parser.add_argument(
        "--skip-cpu",
        action="store_true",
        default=True,
        help="Skip slow GoTorch CPU benchmarks (default: True)",
    )
    parser.add_argument(
        "--with-cpu",
        dest="skip_cpu",
        action="store_false",
        help="Include GoTorch CPU benchmarks in comparison",
    )
    args = parser.parse_args()

    run_matmul_benchmarks(sizes=args.sizes, skip_cpu=args.skip_cpu)


if __name__ == "__main__":
    main()
