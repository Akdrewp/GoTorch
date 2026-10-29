# GoTorch Benchmarking & GPU Profiling Suite

This directory contains the performance benchmarking, hardware profiling, and kernel analysis suite for **GoTorch**. It provides end-to-end tools to measure latency, arithmetic throughput (GFLOPS), memory bandwidth coalescing, and hardware performance counters on AMD GPUs (RDNA / CDNA architectures, specifically optimized for **RDNA 4 Navi 48 / `gfx1201`**).

---

## GEMM Kernel Progression

GoTorch implements multiple matrix multiplication (GEMM) algorithms to demonstrate classical GPU performance optimization progression:

| Kernel Method | C++ Kernel Function | Architectural Highlights |
| :--- | :--- | :--- |
| `matmul_naive` | `matrix_multiplication_naive` | Baseline 1-thread-per-element un-tiled GEMM reading directly from global memory ($O(N^3)$ memory traffic). |
| `matmul_tiled` | `matrix_multiplication_tiled` | $16 \times 16$ LDS (Local Data Share) cache-blocking with cooperative global loads and barrier synchronization. |
| `matmul_coalesced` | `matrix_multiplication_coalesced` | $32 \times 32$ LDS tile aligned to AMD Wave32 wavefront width with padded banks (`[32][33]`) for conflict-free, 100% coalesced 128-byte global transactions. |
| `matmul_coalesced_coarsened` *(default `matmul`)* | `matrix_multiplication_coalesced_coarsened` | 2D Block Tiling ($64 \times 64$ with $BK=16$), thread coarsening ($4 \times 4$ per-thread register accumulation), and vectorized `float4` global memory load/store instructions (`GLOBAL_LOAD_DWORDX4`). |

---

## Directory Layout

```
benchmarks/
├── benchmark_matmul.py       # Multi-kernel speed & throughput (GFLOPS) benchmark
├── profile_matmul.py         # Dedicated GoTorch GPU kernel dispatch driver for profilers
├── profile_pytorch_matmul.py # PyTorch ROCm GEMM profiler exporting Perfetto/Chrome JSON traces
├── profile_safe.py           # Safe GPU harness managing AMD DPM power states during profiling
├── parse_counters.py         # NCU-style hardware counter parser for rocprofv3 CSV collections
├── utils.py                  # Statistical timing harness, warmup loops, and Markdown table formatters
├── live_counters/            # Destination directory for rocprofv3 PMC hardware counter dumps
└── profile_output/           # Destination directory for Perfetto/Chrome traces and profiler logs
```

---

## 1. Running the Speed & Throughput Benchmark

The primary benchmark script [benchmark_matmul.py](benchmark_matmul.py) compares all four GoTorch GPU kernels directly against PyTorch's rocBLAS/Tensile GPU implementation. GoTorch CPU is skipped by default to ensure benchmarks finish in seconds.

### Basic Run (GPU Kernels + PyTorch GPU)
```bash
PYTHONPATH=. ./.venv/bin/python benchmarks/benchmark_matmul.py
```

### Custom Matrix Sizes
```bash
PYTHONPATH=. ./.venv/bin/python benchmarks/benchmark_matmul.py --sizes 256 512 1024 2048
```

### Including CPU Benchmarks
```bash
PYTHONPATH=. ./.venv/bin/python benchmarks/benchmark_matmul.py --with-cpu
```

---

## 2. Hardware Performance Profiling (`rocprofv3`)

AMD RDNA GPUs enable aggressive dynamic power clock gating (`power_dpm_force_performance_level == auto`), which gates off hardware event counters during execution. 

[profile_safe.py](profile_safe.py) acts as a safety wrapper: it sets the GPU power profile to `profile_standard` for consistent, deterministic counter sampling, runs the profiler, and **guarantees** the GPU is restored to `auto` even if interrupted (SIGINT/SIGTERM).

### Profiling Hardware Counters with `rocprofv3`
Run the safe profiler with `sudo` (required to adjust sysfs DPM clock states):

```bash
sudo PYTHONPATH=. python3 benchmarks/profile_safe.py /opt/rocm/bin/rocprofv3 \
  --pmc SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM SQ_INST_CYCLES_VALU SQ_INST_CYCLES_VMEM TCP_REQ TCP_REQ_MISS GL2C_HIT GL2C_MISS \
  -f csv -d ./benchmarks/live_counters -- \
  ./.venv/bin/python benchmarks/profile_matmul.py --size 1024 --iterations 3 --kernel coarsened
```

#### Kernel Variants for Profiling:
You can pass `--kernel {naive,tiled,coalesced,coarsened}` to profile any specific kernel implementation.

---

## 3. Parsing Hardware Performance Counters

Once a `rocprofv3` collection has been captured in `benchmarks/live_counters/`, run [parse_counters.py](parse_counters.py) to format the raw CSV into an NCU-style hardware diagnostics report:

```bash
./.venv/bin/python benchmarks/parse_counters.py
```

By default, it automatically locates the most recent CSV inside `benchmarks/live_counters/` and calculates:
* **Pipeline Cycles**: Active VALU (compute) cycles vs. VMEM (memory stall) cycles and the stall ratio.
* **L1 Vector Cache (TCP)**: Request count, hit count, miss count, and L1 cache hit rate.
* **L2 Cache (GL2C)**: Request count, hit count, miss count, and L2 cache hit rate.

*(To inspect a specific previous run: `./.venv/bin/python benchmarks/parse_counters.py benchmarks/live_counters/<run>/<file>_counter_collection.csv`)*

---

## 4. PyTorch GPU Profiling & Timeline Traces

To generate Chrome/Perfetto timeline traces for PyTorch GPU GEMM on ROCm:

```bash
./.venv/bin/python benchmarks/profile_pytorch_matmul.py --size 1024 --iterations 20
```

The output JSON trace is saved to `benchmarks/profile_output/pytorch_gemm_trace.json`. You can open and inspect the trace directly in any browser at [ui.perfetto.dev](https://ui.perfetto.dev) or Chrome's `chrome://tracing`.
