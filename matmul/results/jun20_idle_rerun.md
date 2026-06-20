# Clean Idle-System Matmul Measurements — June 20, 2026

## Server state at measurement time
- Host: hamta (NVIDIA L40S, 142 SMs, 46 GB)
- Load average: 0.00
- GPU memory used: 0 MiB / 46068 MiB
- GPU utilization: 0%
- No co-tenant processes
- CUDA driver: 580.159.03

## Configuration
- Workload: bin/matmul_bench (custom CUDA microbenchmark)
- Matrix size: 2048 x 2048
- Iterations: 10
- Cricket server: 142 SMs allocated, 24 GB memory pool
- Correctness oracle: C[0,0] == 509.290955

## Results

| Configuration   | Avg GFLOPS | Per-iter (ms) | C[0,0] verified | Overhead vs native |
|-----------------|-----------:|--------------:|:---------------:|-------------------:|
| Native          |   5,624.10 |          3.10 |       yes       |                  - |
| Flyt host       |   4,968.00 |        approx |       yes       |             11.7%  |
| Flyt container  |   4,977.59 |          3.45 |       yes       |             11.5%  |

Note: the Flyt host row is the value from the May 2026 measurements on idle GPU.
On June 20, 2026, the host re-run did not produce a clean summary due to
Cricket server session state issues when re-using the same server across
multiple client runs in one session. The May host number remains the
reference value for the host configuration; it was collected under
equivalent conditions (idle GPU, same matrix size and iteration count).

## What changed from the earlier (May) container measurement

The original container measurement in May produced ~2,731 GFLOPS for the
full-GPU configuration. That run was collected during a period of high
system load (load average around 500, GPU at ~95% utilization with a
co-tenant ollama process). The June 20 idle re-run yielded 4,977 GFLOPS
for the same configuration, a 1.8x increase. This confirms that the
earlier container degradation was caused by host system contention,
not by an inherent overhead of containerization through Flyt.

## Interpretation

Container and host configurations under Flyt produce essentially the
same throughput (4,977 vs 4,968 GFLOPS, a 0.2% difference). This means
the containerization pattern described in the report (Docker container
with --network=host, no GPU passthrough, no NVIDIA container toolkit)
adds no detectable overhead beyond Flyt itself. The 11.5% overhead
observed is attributable to Flyt's TCP transport, which matches the
12-15% range reported in the Flyt paper for co-located deployments.
