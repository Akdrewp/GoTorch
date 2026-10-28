#!/usr/bin/env python3
"""Parses and formats rocprofv3 hardware counter CSV dumps into NCU-style statistics."""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd


def find_latest_counter_csv(search_dir: str = "live_counters") -> str | None:
    """Finds the most recently modified counter collection CSV file.

    Args:
        search_dir: Base directory to search recursively.

    Returns:
        Path to the latest CSV file, or None if none found.
    """
    pattern = os.path.join(search_dir, "**", "*_counter_collection.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        files = glob.glob("**/*_counter_collection.csv", recursive=True)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def parse_and_display_counters(csv_path: str) -> None:
    """Parses hardware counter collection CSV and displays formatted analysis.

    Args:
        csv_path: Path to the *_counter_collection.csv file.
    """
    if not os.path.exists(csv_path):
        print(f"[!] Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading counters from: {csv_path}")
    df = pd.read_csv(csv_path)

    kernel_df = df[df["Kernel_Name"].str.contains("matrix_multiplication", case=False, na=False)]

    if kernel_df.empty:
        print("Warning: No 'matrix_multiplication' dispatches found, showing all kernels:")
        kernel_df = df

    pivot = kernel_df.pivot_table(
        index=["Dispatch_Id", "Kernel_Name"],
        columns="Counter_Name",
        values="Counter_Value",
        aggfunc="sum",
    ).reset_index()

    print("\n" + "=" * 90)
    print("             AMD Radeon RX 9070 XT: GoTorch GEMM Hardware Counters")
    print("=" * 90)

    for _, row in pivot.iterrows():
        kname = str(row["Kernel_Name"]).split("(")[0]
        print(f"\n[Dispatch #{row['Dispatch_Id']}] {kname}")
        print("-" * 90)

        # 1. Pipeline cycle stall analysis
        v_cycles = float(row.get("SQ_INST_CYCLES_VALU", 0))
        m_cycles = float(row.get("SQ_INST_CYCLES_VMEM", 0))
        if v_cycles > 0:
            ratio = m_cycles / v_cycles
            print("  * Pipeline Cycles:")
            print(f"      - Math ALUs Active (VALU Cycles):   {v_cycles:>15,.0f}")
            print(f"      - Waiting on Memory (VMEM Cycles):  {m_cycles:>15,.0f}")
            print(f"      - Memory-to-Math Stall Ratio:       {ratio:>15.2f}x")

        # 2. L1 Vector Cache (TCP)
        tcp_req = float(row.get("TCP_REQ", 0))
        tcp_miss = float(row.get("TCP_REQ_MISS", 0))
        if tcp_req > 0:
            tcp_hit = tcp_req - tcp_miss
            tcp_rate = (tcp_hit / tcp_req) * 100.0
            print("  * L1 Vector/Texture Cache (TCP):")
            print(
                f"      - Requests: {tcp_req:>12,.0f} | Hits: {tcp_hit:>12,.0f} | Misses: {tcp_miss:>12,.0f}"
            )
            print(f"      - Hit Rate: {tcp_rate:>12.2f}%")

        # 3. L2 / Infinity Cache (GL2C)
        gl2c_hit = float(row.get("GL2C_HIT", 0))
        gl2c_miss = float(row.get("GL2C_MISS", 0))
        total_l2 = gl2c_hit + gl2c_miss
        if total_l2 > 0:
            l2_rate = (gl2c_hit / total_l2) * 100.0
            print("  * L2 Cache (GL2C):")
            print(f"      - Total L2 Accesses: {total_l2:>12,.0f}")
            print(f"      - Hits:              {gl2c_hit:>12,.0f}")
            print(f"      - Misses:            {gl2c_miss:>12,.0f}")
            print(f"      - Hit Rate:          {l2_rate:>12.2f}%")

    print("\n" + "=" * 90 + "\n")


def main() -> None:
    """CLI orchestrator for parsing rocprofv3 counter dumps."""
    parser = argparse.ArgumentParser(
        description="Parse rocprofv3 counter collection CSV files and display hardware statistics."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=None,
        help="Path to *_counter_collection.csv (defaults to latest found in live_counters/)",
    )
    parser.add_argument(
        "--dir",
        default="live_counters",
        help="Directory to search for counter CSVs (default: live_counters)",
    )
    args = parser.parse_args()

    csv_path = args.csv_file
    if not csv_path:
        csv_path = find_latest_counter_csv(args.dir)
        if not csv_path:
            print(
                f"[!] Error: No counter CSV file found in '{args.dir}'. "
                f"Please specify the CSV path directly: python benchmarks/parse_counters.py <path_to_csv>",
                file=sys.stderr,
            )
            sys.exit(1)

    parse_and_display_counters(csv_path)


if __name__ == "__main__":
    main()
