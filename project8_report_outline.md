### Project 8 – Training-Time vs. Inference-Time Architectures

> CECS 530 – Advanced Computer Architecture I  
> Student: _[Your Name]_  
> Date: _[Date]_

---

### 1. Introduction

- **Motivation**
  - Why training hardware ≠ good inference hardware
  - Modern LLMs and inference TCO dominance
- **Project objective**
  - Characterize one training workload and one inference workload (same transformer model family)
  - Evaluate training-optimized vs. inference-optimized architectures
  - Propose specialized architectures; optionally a unified design

---

### 2. Workload Characterization (Goal 1)

- **2.1 Model family**
  - Transformer-XXL (or specific open model family)
  - Attributes: parameter scale, sequence length, vocabulary size

- **2.2 Training workload**
  - Operation mix: matrix multiplies, attention, layer-norm, elementwise ops
  - Parallelism: deep data/tensor/pipeline; large global batch
  - Memory: activations (long lifetimes), gradient storage, optimizer state
  - Arithmetic intensity: FLOPs/byte, compute-bound vs. memory-bound

- **2.3 Inference workload**
  - Operation mix: forward pass, autoregressive decoding
  - Parallelism: shallow pipeline; small batch; autoregressive dependency
  - Memory: KV-cache reads/writes; weights read-only
  - Arithmetic intensity: closer to memory-bound on inference accelerators

- **2.4 Roofline-style comparison**
  - Table: FLOPs/token, bytes/token, FLOPs/byte, tokens/s (compute vs. memory bound)
  - Dominant bottlenecks per workload

---

### 3. Architecture Evaluation (Goal 2)

- **3.1 Architectures considered**
  - Training-GPU-like: high FLOPs, HBM, large-batch optimized
  - Inference-Accel-like: lower FLOPs, KV-cache support, INT8/INT4

- **3.2 Methodology**
  - Arithmetic intensity and roofline-limited tokens/s per workload × architecture
  - Bottleneck classification (compute vs. memory)

- **3.3 Results**
  - Compute-bound vs. memory-bound per pair
  - Utilization and inefficiencies
  - Training GPU excels at training; underutilized on inference. Inference accelerator efficient for inference; less ideal for training.

---

### 4. Specialized Architecture Designs (Goal 3)

- **4.1 Training-optimized**
  - Datapath: wide matrix units, fused GEMM+activation
  - Memory: HBM, on-chip SRAM for activations/gradients
  - Interconnect: NVLink-style
  - Scheduling: static graphs, large batches
  - Rationale tied to Section 2

- **4.2 Inference-optimized**
  - Datapath: INT8/INT4 units, KV-cache access
  - Memory: on-chip cache for weights/KV
  - Interconnect: concurrent streams
  - Scheduling: dynamic batching, token-level
  - Rationale tied to inference bottlenecks

---

### 5. Optional Unified Architecture (Goal 4 – Bonus)

- Unified design sketch and compromises
- Efficiency loss vs. specialized designs
- Irreducible conflicts (e.g., batch size vs. tail latency)

---

### 6. Cost and Deployment (Goal 5)

- Metrics: throughput, scaling, latency, energy per token
- Hardware utilization
- Energy and TCO comparison
- Deployment: separate training cluster + inference fleet

---

### 7. Implementation / Modeling

- Analytical comparison framework (`project8_analysis.py` or `scripts/run_experiments.py`)
- Workload and architecture definitions
- Reproducibility: Python version, run command, output reference

---

### 8. Conclusion

- Critical differences between training-time and inference-time architectures
- Why a single accelerator is difficult in practice
- Future directions: compilers, schedulers, hardware–software co-design
