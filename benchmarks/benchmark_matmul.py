"""Matrix multiplication benchmark: GoTorch (CPU & AMD GPU) vs PyTorch (CPU & GPU)."""

from __future__ import annotations

import argparse

import torch
from GoTorch.backend.native_backend import NDArray

from benchmarks.utils import benchmark_op, print_comparison_table
from GoTorch import Device


def run_matmul_benchmarks(
    sizes: list[int] | None = None,
    skip_cpu: bool = True,
) -> None:
    if sizes is None:
        sizes = [64, 128, 256, 512, 1024]

    pt_gpu_available = torch.cuda.is_available()

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

    print("\nRunning Matrix Multiplication (GEMM) benchmarks...")
    if pt_gpu_available:
        print(f"  PyTorch GPU detected: {torch.cuda.get_device_name(0)}")

    for n in sizes:
        print(f"  Benchmarking size {n}x{n}...")

        # Keep iterations stable per problem size
        if n >= 4096:
            gpu_iters = 5
            warmup_iters = 3
        elif n >= 1024:
            gpu_iters = 10
            warmup_iters = 5
        else:
            gpu_iters = 20
            warmup_iters = 5

        total_flops = 2.0 * (n**3)

        # 1. Prepare PyTorch GPU inputs with dedicated pre-allocated output buffer
        res_pt_gpu_ms = None
        if pt_gpu_available:
            # Generate directly on device (fast, avoids multi-GB Python lists)
            pt_a_gpu = torch.randn(n, n, dtype=torch.float32, device="cuda")
            pt_b_gpu = torch.randn(n, n, dtype=torch.float32, device="cuda")
            pt_out_gpu = torch.empty((n, n), dtype=torch.float32, device="cuda")

            # Dedicated HIP event timing for PyTorch to eliminate CPU timing artifacts
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Warmup: force rocBLAS/Tensile to compile heuristics and allocate workspace for THIS N
            for _ in range(warmup_iters):
                torch.matmul(pt_a_gpu, pt_b_gpu, out=pt_out_gpu)
            torch.cuda.synchronize()

            start_event.record()
            for _ in range(gpu_iters):
                torch.matmul(pt_a_gpu, pt_b_gpu, out=pt_out_gpu)
            end_event.record()
            torch.cuda.synchronize()

            res_pt_gpu_ms = start_event.elapsed_time(end_event) / gpu_iters

        # 2. Prepare GoTorch GPU inputs (convert from torch tensor storage to avoid Python list overhead)
        pt_a_cpu = pt_a_gpu.cpu()
        pt_b_cpu = pt_b_gpu.cpu()
        gt_a_gpu = NDArray(pt_a_cpu.numpy().flatten().tolist(), shape=(n, n)).to(Device.GPU)
        gt_b_gpu = NDArray(pt_b_cpu.numpy().flatten().tolist(), shape=(n, n)).to(Device.GPU)

        def sync_all() -> None:
            if pt_gpu_available:
                torch.cuda.synchronize()

        def gt_gpu_matmul_naive() -> None:
            _ = gt_a_gpu.matmul_naive(gt_b_gpu)
            sync_all()

        def gt_gpu_matmul_tiled() -> None:
            _ = gt_a_gpu.matmul_tiled(gt_b_gpu)
            sync_all()

        def gt_gpu_matmul_coalesced() -> None:
            _ = gt_a_gpu.matmul_coalesced(gt_b_gpu)
            sync_all()

        def gt_gpu_matmul_coarsened() -> None:
            _ = gt_a_gpu.matmul_coalesced_coarsened(gt_b_gpu)
            sync_all()

        # Run GoTorch benchmarks
        res_gt_gpu_naive = benchmark_op(gt_gpu_matmul_naive, warmup=warmup_iters, iterations=gpu_iters)
        res_gt_gpu_tiled = benchmark_op(gt_gpu_matmul_tiled, warmup=warmup_iters, iterations=gpu_iters)
        res_gt_gpu_coalesced = benchmark_op(gt_gpu_matmul_coalesced, warmup=warmup_iters, iterations=gpu_iters)
        res_gt_gpu_coarsened = benchmark_op(gt_gpu_matmul_coarsened, warmup=warmup_iters, iterations=gpu_iters)

        # GFLOPS Calculations
        gt_gpu_coarsened_gflops = total_flops / (res_gt_gpu_coarsened.mean_ms * 1e6)
        pt_gpu_gflops = (total_flops / (res_pt_gpu_ms * 1e6)) if res_pt_gpu_ms else 0.0

        gpu_rows.append([
            f"{n}x{n}",
            f"{res_gt_gpu_naive.mean_ms:.3f}",
            f"{res_gt_gpu_tiled.mean_ms:.3f}",
            f"{res_gt_gpu_coalesced.mean_ms:.3f}",
            f"{res_gt_gpu_coarsened.mean_ms:.3f}",
            f"{res_pt_gpu_ms:.3f}" if res_pt_gpu_ms else "N/A",
            f"{gt_gpu_coarsened_gflops:.1f}",
            f"{pt_gpu_gflops:.1f}" if res_pt_gpu_ms else "N/A",
        ])

    print_comparison_table("2. GPU GEMM Comparison: GoTorch GPU vs PyTorch GPU", gpu_headers, gpu_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GoTorch vs PyTorch Matrix Multiplication.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096, 8192],
        help="Matrix sizes N x N to benchmark",
    )
    args = parser.parse_args()
    run_matmul_benchmarks(sizes=args.sizes)


if __name__ == "__main__":
    main()
