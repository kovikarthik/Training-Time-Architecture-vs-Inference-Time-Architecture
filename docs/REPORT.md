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

Run the framework to produce the comparison:

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

### 4.1 Training-Optimized Architecture

A training accelerator must behave like a supercomputer node:
- **Datapath:** Deep, dense matrix-multiply units (e.g., Tensor Cores) heavily optimized for BF16 instruction fusion (GEMM + Activation).
- **Memory Hierarchy:** High-capacity HBM3/4 is mandatory. On-chip SRAM is merely a staging ground to feed the massive matrix units.
- **Interconnect:** High-radix NVLink-style fabrics (900+ GB/s) mapping directly to the 3D topology of Data/Tensor/Pipeline parallelism.
- **Scheduling:** Static, graph-level execution. Kernels can be heavily fused at compile time because the exact sequence of massive, uniform batch operations is known.

### 4.2 Inference-Optimized Architecture

An inference accelerator must behave like a low-latency network switch:
- **Datapath:** Mixed-precision processing (INT8/INT4), prioritizing vector-matrix math over giant dense matrix-matrix math. KV-cache lookup optimizations are critical.
- **Memory Hierarchy:** Replace HBM with massive distributed on-chip SRAM. Moving weights from SRAM to execution units takes picojoules compared to nanojoules from HBM.
- **Interconnect:** Concurrent stream routing across multiple chips via deterministic interconnects to beat the single-chip memory capacity limit without incurring PCIe/NVLink switch latency.
- **Scheduling:** Dynamic, hardware-managed batching (e.g., continuous/iteration-level batching) to swap in incoming user requests at the microsecond level.

---

## 5. Bonus: Real Hardware Measurement (Apple Silicon)

To contextualize architectural performance, we executed a hardware benchmark on a local **Apple M4 Pro (Metal 4, 14-Core CPU, 20-Core GPU)**. Testing matrix multiplications scaled to the hidden dimensions of our 8B model (N=4096).

**Apple Silicon Benchmark Results:**
- N=1024: 3.38 TFLOP/s
- N=2048: 6.48 TFLOP/s
- N=4096: 6.73 TFLOP/s
- N=8192: 6.16 TFLOP/s

*Analysis:* The M4 Pro achieves an impressive ~6.7 TFLOP/s on medium matrices, largely due to Apple's Unified Memory Architecture (UMA), which provides exceptionally high bandwidth for an SoC (~273 GB/s). This proves the core thesis: high local memory bandwidth enables high practical utilization, validating why Apple Silicon is uniquely proficient at local LLM inference compared to standard x86/PCIe-GPU consumer setups.

---

## 6. Cost and Deployment Implications (Goal 5)

Using our parameterized analytical framework, we modeled the 3-year Total Cost of Ownership (TCO) and Energy-Delay Product (EDP) for a fleet running the Llama 3 8B model. 

### 6.1 Energy-Delay Product (EDP)
EDP perfectly captures the balance between latency and power consumption. Lower is better.

| Architecture | Workload | Energy/Token (mJ) | EDP (µJ·s) | Efficiency vs Baseline |
|--------------|----------|-------------------|------------|------------------------|
| Training-GPU (H100) | Inference | 160.98 | 37.02 | Baseline |
| Unified-GPU (L40S) | Inference | 312.08 | 278.27 | 7.5x Worse |
| Inference-ASIC | Inference | 3.85 | 0.20 | **185x Better** |

### 6.2 3-Year Total Cost of Ownership
Assuming a $0.10/kWh electricity cost and max roofline utilization over 3 years:

| Architecture | Target | Cost per 1M Tokens | Impact |
|--------------|--------|---------------------|--------|
| Training-GPU (H100) | Inference | $0.0774 | Baseline |
| Unified-GPU (L40S) | Inference | $0.1029 | 33% more expensive |
| Inference-ASIC | Inference | $0.0017 | **45x cheaper** |

Attempting to run a unified architecture fleet (e.g., L40S or generic GPUs) results in unacceptable TCO. A unified chip compromises memory bandwidth to gain moderate FLOPs, resulting in a system that is slightly too slow for training and too power-hungry for inference.

**Deployment Recommendation:**
Enterprise fleets must bifurcate. 
1. **Training Centers:** Built with dense, HBM-equipped monolithic accelerators (H100/TPUv5p).
2. **Inference Edge/Cloud:** Deployed using low-precision, high-SRAM ASICs (Groq-style). Because inference consumes over 80% of life-cycle energy costs per model, specialized inference chips that minimize the Energy per Token metric directly control the profitability of the service.

---

## 7. Conclusion

Training optimizes for global throughput via massive batching and deep parallelism; inference optimizes for latency and token predictability via minimal batching and autoregressive generation. As demonstrated analytically and via hardware benchmarking, the resulting workloads have profoundly different Arithmetic Intensities. A specialized training architecture wastes its compute logic on memory stalls during inference, while an inference architecture lacks the capacity to even attempt training. The future of AI hardware definitively lies in strict specialization.
