#!/usr/bin/env python3
"""Safe GPU profiler runner that manages AMD GPU DPM power states during profiling."""

from __future__ import annotations

import glob
import os
import signal
import subprocess
import sys


def get_dpm_files() -> list[str]:
    """Finds all AMD GPU performance level sysfs control files.

    Returns:
        List of file paths for power_dpm_force_performance_level.
    """
    paths = glob.glob("/sys/class/drm/card*/device/power_dpm_force_performance_level")
    if not paths:
        print(
            "[!] Error: No AMD GPU power_dpm_force_performance_level files found under /sys/class/drm/",
            file=sys.stderr,
        )
        sys.exit(1)
    return paths


def set_dpm_level(level: str) -> None:
    """Writes a target performance level ('profile_standard', 'profile_peak', or 'auto').

    Args:
        level: Target DPM performance level string.
    """
    dpm_files = get_dpm_files()
    for path in dpm_files:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(level)
            print(f"[GPU Power] Successfully set {path} -> {level}")
        except PermissionError:
            print(f"\n[!] Error: Permission denied writing to {path}.", file=sys.stderr)
            print("[!] Please run this script with 'sudo'.\n", file=sys.stderr)
            sys.exit(1)
        except OSError as e:
            print(f"[!] Failed to write '{level}' to {path}: {e}", file=sys.stderr)


class SafeGPUProfile:
    """Context manager to ensure the GPU always reverts to 'auto'."""

    def __init__(self, target_level: str = "profile_standard") -> None:
        """Initializes context manager with target DPM level.

        Args:
            target_level: Desired GPU performance level during profiling.
        """
        self.target_level = target_level

    def __enter__(self) -> SafeGPUProfile:
        """Enters context and sets target DPM level."""
        set_dpm_level(self.target_level)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exits context and resets DPM level to 'auto'."""
        print("\n[GPU Power] Cleaning up: Reverting power profile to 'auto'...")
        set_dpm_level("auto")


def main() -> None:
    """CLI entry point for safe GPU profiling."""
    if len(sys.argv) < 2:
        print("Usage: sudo python3 benchmarks/profile_safe.py <command_to_profile> [args...]")
        print("\nExample:")
        print(
            "sudo PYTHONPATH=. python3 benchmarks/profile_safe.py /opt/rocm/bin/rocprofv3 "
            "--pmc SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM SQ_INST_CYCLES_VALU SQ_INST_CYCLES_VMEM "
            "TCP_REQ TCP_REQ_MISS GL2C_HIT GL2C_MISS -f csv -d ./live_counters -- "
            "./.venv/bin/python benchmarks/profile_matmul.py --size 1024 --iterations 3"
        )
        sys.exit(1)

    cmd = sys.argv[1:]

    def handle_signal(sig: int, frame: object) -> None:
        print(f"\n[!] Caught signal {sig}. Ensuring GPU is reset to 'auto' before exiting...")
        set_dpm_level("auto")
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    with SafeGPUProfile(target_level="profile_standard"):
        print(f"\n>>> Running profiled workload: {' '.join(cmd)}\n" + "=" * 80)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[!] Command returned exit status {e.returncode}", file=sys.stderr)
        except OSError as e:
            print(f"\n[!] Execution failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
