# Clean Idle-System Matmul Measurements — June 20, 2026

## Server state at measurement time

- Host: hamta (NVIDIA L40S, 142 SMs, 46 GB)
- Load average: 0.00
- GPU memory used: 0 MiB / 46068 MiB at start
- GPU utilization: 0% at start
- No co-tenant processes
- NVIDIA driver: 580.159.03
- Cricket server: 142 SMs allocated, 24 GB memory pool

## Configuration

- Workload: bin/matmul_bench (custom CUDA microbenchmark)
- Matrix size: 2048 x 2048
- Iterations: 10
- Correctness oracle: C[0,0] == 509.290955

## Results

| Configuration | Avg GFLOPS | Per-iter (ms) | C[0,0] verified | Overhead vs native |
|---|---:|---:|:---:|---:|
| Native | 5,624 | 3.10 | yes | reference |
| Flyt host | 4,968 (May) | approx 3.50 | yes | 11.7% |
| Flyt container | 4,978 | 3.45 | yes | 11.5% |

The Flyt host value is the May 2026 idle-window measurement; the June 20 host
re-run was not collected cleanly due to Cricket-server session state issues
when re-using one server across multiple client runs. The May measurement
was taken under equivalent conditions (idle GPU, same matrix size, same
iteration count) and remains the reference for the host configuration.

## What changed from the May container measurement

The original May container measurement produced 2,731 GFLOPS for the
full-GPU configuration. That run was collected under high system load
(load average around 500, GPU at ~95% utilization due to a co-tenant
ollama process). The June 20 idle re-run yielded 4,978 GFLOPS for the
same configuration, a 1.8x increase. This confirms that the earlier
container degradation was caused by host system contention, not by
an inherent overhead of containerization through Flyt.

## Interpretation

Container and host configurations under Flyt produce essentially
identical throughput on an idle system (4,978 vs 4,968, a 0.2%
difference). The containerization pattern described in Section 3.4
of the report (Docker container with --network=host, no GPU
passthrough, no NVIDIA container toolkit) therefore adds no
detectable overhead beyond Flyt itself. The 11.5% TCP virtualization
overhead observed matches the 12-15% range reported in the Flyt
paper for co-located deployments.
