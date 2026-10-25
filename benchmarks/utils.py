"""Benchmarking utilities, timing harnesses, and statistical reporters."""

from __future__ import annotations

import statistics
import time
from typing import Callable, NamedTuple


class BenchmarkResult(NamedTuple):
    """Holds performance metrics for a single benchmark run."""

    name: str
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    iterations: int


def benchmark_op(
    func: Callable[[], None],
    warmup: int = 5,
    iterations: int = 20,
    name: str = "Op",
) -> BenchmarkResult:
    """Measures execution latency of a callable across warmup and measured rounds.

    Args:
        func: Parameterless callable to benchmark.
        warmup: Number of untimed warmup iterations.
        iterations: Number of timed measurement iterations.
        name: Label identifying the benchmarked workload.

    Returns:
        BenchmarkResult containing timing statistics in milliseconds.
    """
    for _ in range(warmup):
        func()

    durations: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        durations.append((end - start) * 1000.0)

    mean_ms = statistics.mean(durations)
    median_ms = statistics.median(durations)
    min_ms = min(durations)
    max_ms = max(durations)
    std_ms = statistics.stdev(durations) if len(durations) > 1 else 0.0

    return BenchmarkResult(
        name=name,
        mean_ms=mean_ms,
        median_ms=median_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        std_ms=std_ms,
        iterations=iterations,
    )


def print_comparison_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
) -> None:
    """Prints a formatted ASCII markdown comparison table.

    Args:
        title: Title of the benchmark section.
        headers: List of column header strings.
        rows: List of string rows.
    """
    print(f"\n### {title}\n")
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    print(header_line)
    print(separator_line)
    for row in rows:
        print("| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)) + " |")
    print()
