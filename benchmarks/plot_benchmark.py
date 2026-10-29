"""Generates comparative visualization plots for GoTorch vs PyTorch GPU GEMM benchmarks."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


def parse_markdown_table(text: str) -> dict[str, list] | None:
    """Parses benchmark markdown table text into numerical arrays.

    Args:
        text: Raw text string containing the markdown table.

    Returns:
        Dictionary mapping column names to lists of values, or None if no table found.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|") and not line.startswith("| -") and not line.startswith("| Matrix"):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 6:
                try:
                    size_match = re.search(r"(\d+)\s*x", parts[0])
                    if not size_match:
                        continue
                    size = int(size_match.group(1))
                    naive = float(parts[1])
                    tiled = float(parts[2])
                    coalesced = float(parts[3])
                    coarsened = float(parts[4])
                    pytorch = float(parts[5]) if parts[5] != "N/A" else np.nan
                    rows.append((size, naive, tiled, coalesced, coarsened, pytorch))
                except (ValueError, IndexError):
                    continue

    if not rows:
        return None

    rows.sort(key=lambda r: r[0])
    return {
        "sizes": [r[0] for r in rows],
        "naive_ms": [r[1] for r in rows],
        "tiled_ms": [r[2] for r in rows],
        "coalesced_ms": [r[3] for r in rows],
        "coarsened_ms": [r[4] for r in rows],
        "pytorch_ms": [r[5] for r in rows],
    }


def run_benchmark(sizes: list[int]) -> dict[str, list]:
    """Runs benchmark_matmul.py in a subprocess and parses the table output.

    Args:
        sizes: List of square matrix sizes to benchmark.

    Returns:
        Dictionary mapping column names to lists of metrics.

    Raises:
        RuntimeError: If subprocess fails or table output cannot be parsed.
    """
    cmd = [
        sys.executable,
        "benchmarks/benchmark_matmul.py",
        "--sizes",
        *[str(s) for s in sizes],
    ]
    print(f"Running benchmark command: {' '.join(cmd)} ...")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    print(res.stdout)

    data = parse_markdown_table(res.stdout)
    if not data:
        raise RuntimeError("Failed to parse markdown table from benchmark output.")
    return data


def plot_benchmark_results(
    data: dict[str, list],
    output_path: str = "benchmarks/profile_output/matmul_benchmark_comparison.png",
    show: bool = False,
) -> str:
    """Generates two-panel latency and throughput plots from benchmark metrics.

    Args:
        data: Dictionary containing matrix sizes and kernel execution times in ms.
        output_path: File system path where the rendered PNG plot is saved.
        show: Whether to display interactive plot window.

    Returns:
        Path to the saved PNG image.
    """
    sizes = np.array(data["sizes"])
    naive_ms = np.array(data["naive_ms"])
    tiled_ms = np.array(data["tiled_ms"])
    coalesced_ms = np.array(data["coalesced_ms"])
    coarsened_ms = np.array(data["coarsened_ms"])
    pytorch_ms = np.array(data["pytorch_ms"])

    # Calculate throughput: (2 * N^3) / (latency_sec) in TFLOPS
    total_flops = 2.0 * (sizes**3)
    naive_tflops = (total_flops / (naive_ms * 1e-3)) / 1e12
    tiled_tflops = (total_flops / (tiled_ms * 1e-3)) / 1e12
    coalesced_tflops = (total_flops / (coalesced_ms * 1e-3)) / 1e12
    coarsened_tflops = (total_flops / (coarsened_ms * 1e-3)) / 1e12
    pytorch_tflops = (total_flops / (pytorch_ms * 1e-3)) / 1e12

    # Matplotlib styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax_lat, ax_tflops) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    colors = {
        "naive": "#d9534f",
        "tiled": "#f0ad4e",
        "coalesced": "#5bc0de",
        "coarsened": "#28a745",
        "pytorch": "#0275d8",
    }

    # -------------------------------------------------------------
    # Subplot 1: Latency (Log-Log Scale)
    # -------------------------------------------------------------
    ax_lat.plot(sizes, naive_ms, marker="o", color=colors["naive"], linewidth=1.8, label="GoTorch: Naive")
    ax_lat.plot(sizes, tiled_ms, marker="s", color=colors["tiled"], linewidth=1.8, label="GoTorch: Tiled")
    ax_lat.plot(sizes, coalesced_ms, marker="^", color=colors["coalesced"], linewidth=1.8, label="GoTorch: Coalesced")
    ax_lat.plot(sizes, coarsened_ms, marker="D", color=colors["coarsened"], linewidth=2.2, label="GoTorch: Coarsened")
    ax_lat.plot(sizes, pytorch_ms, marker="*", markersize=9, color=colors["pytorch"], linewidth=2.0, linestyle="--", label="PyTorch GPU")

    ax_lat.set_xscale("log", base=2)
    ax_lat.set_yscale("log")
    ax_lat.set_xticks(sizes)
    ax_lat.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_lat.set_xlabel("Matrix Size ($N \\times N$)", fontsize=11)
    ax_lat.set_ylabel("Execution Latency (ms)", fontsize=11)
    ax_lat.set_title("Execution Latency (ms)", fontsize=12, fontweight="bold")
    ax_lat.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left", fontsize=10)
    ax_lat.grid(True, which="both", linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # Subplot 2: Throughput (TFLOPS)
    # -------------------------------------------------------------
    ax_tflops.plot(sizes, naive_tflops, marker="o", color=colors["naive"], linewidth=1.8, label="GoTorch: Naive")
    ax_tflops.plot(sizes, tiled_tflops, marker="s", color=colors["tiled"], linewidth=1.8, label="GoTorch: Tiled")
    ax_tflops.plot(sizes, coalesced_tflops, marker="^", color=colors["coalesced"], linewidth=1.8, label="GoTorch: Coalesced")
    ax_tflops.plot(sizes, coarsened_tflops, marker="D", color=colors["coarsened"], linewidth=2.2, label="GoTorch: Coarsened")
    ax_tflops.plot(sizes, pytorch_tflops, marker="*", markersize=9, color=colors["pytorch"], linewidth=2.0, linestyle="--", label="PyTorch GPU")

    ax_tflops.set_xscale("log", base=2)
    ax_tflops.set_xticks(sizes)
    ax_tflops.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_tflops.set_xlabel("Matrix Size ($N \\times N$)", fontsize=11)
    ax_tflops.set_ylabel("Throughput (TFLOPS)", fontsize=11)
    ax_tflops.set_title("Compute Throughput (TFLOPS)", fontsize=12, fontweight="bold")
    ax_tflops.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left", fontsize=10)
    ax_tflops.grid(True, which="both", linestyle=":", alpha=0.6)

    plt.suptitle("GPU Matrix Multiplication: GoTorch vs PyTorch", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n[+] Benchmark graph successfully saved to: {output_path}")

    if show:
        plt.show()

    plt.close(fig)
    return output_path


def main() -> None:
    """CLI orchestrator for plotting GEMM benchmark graphs."""
    parser = argparse.ArgumentParser(description="Generate benchmark comparison plots for GoTorch vs PyTorch GPU.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096, 8192],
        help="Matrix sizes N x N to benchmark if running live",
    )
    parser.add_argument(
        "--from-log",
        type=str,
        default=None,
        help="Path to saved benchmark stdout text/log file to plot from",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/profile_output/matmul_benchmark_comparison.png",
        help="Destination path for generated PNG plot",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive plot display",
    )
    args = parser.parse_args()

    # Determine input data source
    data = None
    if not sys.stdin.isatty():
        # Piped input from benchmark_matmul.py
        stdin_content = sys.stdin.read()
        data = parse_markdown_table(stdin_content)
        if data:
            print("Loaded benchmark data from piped stdin.")

    if not data and args.from_log:
        with open(args.from_log, "r", encoding="utf-8") as f:
            data = parse_markdown_table(f.read())
            if data:
                print(f"Loaded benchmark data from log file: {args.from_log}")

    if not data:
        data = run_benchmark(args.sizes)

    plot_benchmark_results(data, output_path=args.output, show=args.show)


if __name__ == "__main__":
    main()
