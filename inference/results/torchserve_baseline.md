# TorchServe Baseline (without Flyt)

Date: Sunday May 3, 2026
Server: hamta (NVIDIA L40S, 142 SMs, 46 GB VRAM)
GPU contention: ollama using 25.9 GB (56% VRAM), 93% GPU utilization

## Setup
- Model: resnet18 with CIFAR-10 trained checkpoint
- Archive: model_store/resnet18_cifar10.mar (119 MB)
- TorchServe: v0.12.0
- PyTorch venv: ~/pytorch-env (torch 2.7.1+cu118)

## Result for airplane/00003.png

Inference latency: 279.1 ms (PredictionTime)
End-to-end request latency: 283.7 ms (ts_inference_latency_microseconds)

Top-3 predictions:
- airplane: 0.6991
- ship: 0.2429
- truck: 0.0264

Prediction is correct (image is airplane).

## Significance

This establishes the "ground truth" for the model+image combination. Any
TorchServe-through-Flyt run should produce the same predictions (within
floating-point rounding). Latency is expected to increase substantially with
Flyt-TCP per Table 8 of the Flyt revision paper, which reports ~5.2 img/s
for VGG16 and ~2.0 img/s for ResNet50 over TCP (vs ~300 img/s native).
