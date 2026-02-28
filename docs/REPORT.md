# Project 8: Training-Time vs. Inference-Time Architecture

**CECS 530 – Advanced Computer Architecture I**  
**CSULB – Fall 2025**

---

## 1. Introduction

### Motivation

Modern AI systems are often built on hardware originally optimized for training, then repurposed for inference. The assumption that "fast training hardware implies fast inference" is increasingly false, especially for large language models (LLMs).

### Project Objective

This project systematically compares training-time and inference-time architectures along multiple dimensions and implements an **analytical comparison framework** with:

- Workload characterization (operation mix, arithmetic intensity, memory patterns)
- Architecture evaluation (training-optimized vs. inference-optimized)
- Roofline-style performance modeling
- Cost and deployment implications

---

## 2. Workload Characterization (Goal 1)

### 2.1 Model Family

We use a **Transformer-XXL** family (~7B parameters, sequence length 2048) as the reference model for both workloads.

### 2.2 Training Workload

- **Operation mix**: Matrix multiplies (65%), attention (20%), layer-norm (5%), activations (5%), other (5%)
- **Parallelism**: Data-parallel (16-way), tensor-parallel (8-way), pipeline (8 stages)
- **Memory**: Activations, gradients, optimizer state; no KV-cache
- **Precision**: BF16 compute, FP32 accumulation

### 2.3 Inference Workload

- **Operation mix**: Matrix multiplies (60%), attention (30%), layer-norm (5%), activations (3%), other (2%)
- **Parallelism**: Limited (data-parallel 1, tensor 2, pipeline 2)
- **Memory**: Weights, KV-cache reads/writes
- **Precision**: INT8 compute, INT32 accumulation

### 2.4 Roofline Comparison

Run the framework to generate the comparison:

```bash
python scripts/run_experiments.py --config config/experiments.yaml --stdout
```

See `results/*.json` for full outputs.

---

## 3. Architecture Evaluation (Goal 2)

### 3.1 Architectures

| Architecture | Peak FLOPs (TFLOP/s) | Mem BW (GB/s) | Optimized For |
|---------------|---------------------|---------------|---------------|
| Training-GPU-like | 300 | 1550 | Large-batch training |
| Inference-Accel-like | 150 | 2200 | Low-latency inference |
| Unified-Compromise | 220 | 1700 | Balanced |

### 3.2 Mismatch Analysis

- **Training GPU on inference**: Underutilizes FLOPs; memory-bound behavior in KV-cache streaming.
- **Inference accelerator on training**: Insufficient peak FLOPs; weaker data/tensor parallelism support.

---

## 4. Specialized Architecture Designs (Goal 3)

### 4.1 Training-Optimized

- **Datapath**: Wide matrix units, fused GEMM+activation
- **Memory**: HBM2e, high bandwidth
- **Interconnect**: NVLink-style
- **Scheduling**: Static graphs, large batches

### 4.2 Inference-Optimized

- **Datapath**: Mixed-precision (INT8/INT4), KV-cache optimized
- **Memory**: Large on-chip cache for weights/KV
- **Interconnect**: Scaled for concurrent streams
- **Scheduling**: Dynamic batching, token-level

---

## 5. Cost and Deployment (Goal 5)

- **Training**: Throughput (tokens/sec), scaling efficiency
- **Inference**: Token latency, energy per token, tail latency
- **Deployment**: Separate training cluster + inference fleet recommended

---

## 6. Implementation

- **Analytical comparison framework** in `src/`
- **Parameterized configs** in `config/`
- **Reproducible**: `python scripts/run_experiments.py`

---

## 7. References

- Megatron-LM, DeepSpeed, vLLM, llama.cpp, TVM, PyTorch
