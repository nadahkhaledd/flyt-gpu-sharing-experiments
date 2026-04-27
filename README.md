# Flyt GPU Sharing Experiments — Matrix Multiplication

This folder contains the matrix multiplication benchmark used to validate
Flyt's GPU virtualization and SM partitioning on an NVIDIA L40S. All
experiments produced verified-correct computational results.

## Setup

The experiments were run on `hamta`, a Politecnico di Milano server with:

- NVIDIA L40S GPU (Ada Lovelace, 142 streaming multiprocessors)
- NVIDIA driver 580.126.09
- CUDA runtime 12.0
- MPS with per-context SM partitioning enabled

The benchmark performs a 2048x2048 dense matrix multiplication on the GPU
across 10 timed iterations after one warmup run, reporting GFLOPS and
verifying the first output element against a CPU computation.

## Folder structure

- `src/` — CUDA source code for the benchmark and a simple smoke test
- `bin/` — Compiled binaries (built using the Flyt Docker image so the host
  does not need a CUDA toolkit installed)
- `logs/` — Raw stdout from each experimental run, including Flyt debug
  messages showing every CUDA call as it is intercepted
- `results/` — Summary CSV with all measurements

## Compilation

The benchmark is compiled inside the Flyt build Docker image, which contains
a complete CUDA 12.0 development toolkit:
docker run --rm 
-v ~/flyt-experiments/matmul/src:/src 
-v ~/flyt-experiments/matmul/bin:/bin-out 
-w /src 
khab/flyt-build:cuda12.0-fix 
nvcc -arch=sm_89 -cudart shared -o /bin-out/matmul_bench matmul_bench.cu
The `-cudart shared` flag is required for Flyt compatibility — it links the
CUDA runtime dynamically so Flyt's interception layer can replace it via
LD_PRELOAD.

## Results summary

All numbers are GFLOPS averaged over 10 iterations of a 2048x2048 matmul.

## Limitations and known issues

The Cricket server emits a segmentation fault during shutdown after Ctrl+C.
This happens after all client work has completed and reported, and does not
affect experimental results. It is a cosmetic cleanup bug in the Cricket
library.

The cricket.testapp included with Flyt's source tree produces incorrect
results on this hardware because it was compiled for sm_61 (Pascal). Custom
benchmarks compiled for sm_89 (Ada Lovelace), as in this folder, work
correctly.

PyTorch training cannot currently be intercepted by Flyt because PyTorch's
caching memory allocator is opaque to the CUDA API. The Flyt paper itself
acknowledges this in Section 5.6 as a known limitation and future work.
The Flyt authors evaluated their framework on TorchServe inference, not
PyTorch training, for this reason.
