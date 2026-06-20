# TorchServe-through-Flyt retry on idle GPU, June 17 2026

## Setup
- GPU: NVIDIA L40S, 0 MiB used at start (no ollama, no other tenants)
- System load: 0.00
- Cricket server: 142 SMs, 24 GB memory pool
- TorchServe native baseline run immediately before: SUCCESS
  (airplane 0.6991, ship 0.2429, truck 0.0264; 198 ms latency)

## Result: FAILURE (different failure mode from May 3rd)

The Flyt-instrumented TorchServe worker dies during model load with
RuntimeError: CUDA error: invalid argument. Auto-recovery retries five
times, all identical failures.

The exception originates inside torch.load() at:
  torch/_utils.py:102, in _to
    untyped_storage.copy_(self, non_blocking)

This is PyTorch deserializing the saved weights and copying them
host-to-device. Unlike the May 3rd failure (which was a memory contention
issue, ollama holding 25.9/46 GB), this failure occurs on a completely
free GPU. The cause is at the Cricket-PyTorch interaction layer, in the
storage copy code path, not a memory issue.

The cascade error "Failed to exec spawn helper" appearing later in the
log is a downstream consequence of the first crash, not a separate issue:
the JVM tries to restart workers via jspawnhelper, which inherits the
LD_PRELOAD and cannot complete initialization cleanly.

## Implication for the report

The diagnosis in Section 5.3 should be updated. The earlier conclusion
("would work on a dedicated GPU") is not supported by this data. The
correct conclusion is: TorchServe through Flyt over TCP does not work
end-to-end on the L40S, regardless of VRAM availability, due to a
compatibility issue between Cricket's transport layer and PyTorch's
storage-copy code path used by torch.load().

This is consistent with the Flyt paper's Section 5.6 note about the
PyTorch caching allocator being opaque to Flyt's memory virtualization.
