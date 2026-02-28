### Project 8 – Training-Time vs. Inference-Time Architectures

> CECS 530 – Advanced Computer Architecture I  
> Student: _[Your Name]_  
> Date: _[Date]_

---

### 1. Introduction

- **Motivation**
  - Briefly explain why training hardware ≠ good inference hardware.
  - Mention modern LLMs and why inference TCO dominates cost.
- **Project objective**
  - State that you will:
    - Characterize one training workload and one inference workload for the same transformer model family.
    - Evaluate a training-optimized vs. inference-optimized architecture.
    - Propose specialized architectures and optionally a unified design.

---

### 2. Workload Characterization (Goal 1)

- **2.1 Model family**
  - Choose a generic "Transformer-XXL" (or a specific open model family if you like).
  - List basic attributes: parameter scale (e.g., tens of billions), sequence length, vocabulary size.

- **2.2 Training workload**
  - **Operation mix**: matrix multiplies, attention, layer-norm, elementwise ops.
  - **Parallelism**: deep data/tensor/pipeline parallelism; large global batch.
  - **Memory behavior**:
    - Activations with long lifetimes (needed for backprop).
    - Gradient storage.
    - High optimizer state footprint.
  - **Arithmetic intensity**:
    - Reference numbers from `project8_analysis.py` training workload.
    - Compute FLOPs/byte and note whether it tends to be compute-bound on a training GPU.

- **2.3 Inference workload**
  - **Operation mix**: mostly forward pass, autoregressive decoding.
  - **Parallelism**: shallow pipeline; small batch; limited data parallelism; strong autoregressive dependency.
  - **Memory behavior**:
    - KV-cache reads/writes for attention.
    - Model weights mostly read-only.
  - **Arithmetic intensity**:
    - Reference numbers from `project8_analysis.py` inference workload.
    - Show that it is closer to memory-bound on inference accelerators.

- **2.4 Roofline-style comparison**
  - Insert a small table comparing:
    - FLOPs/token, bytes/token, FLOPs/byte (AI), tokens/s (compute-bound vs. memory-bound) for training vs. inference.
  - Briefly identify dominant bottlenecks for each workload.

---

### 3. Architecture Evaluation (Goal 2)

- **3.1 Existing architectures considered**
  - **Training-GPU-like**:
    - High peak FLOPs, high-bandwidth HBM, optimized for large batches.
    - Good support for data/tensor/pipeline parallelism.
  - **Inference-Accel-like**:
    - Lower raw FLOPs, but:
      - Better memory bandwidth per watt.
      - Support for low-precision (INT8 / INT4).
      - Hardware features for KV-cache, batching, and autoregressive decoding.

- **3.2 Methodology**
  - Describe how you use `project8_analysis.py`:
    - For each workload and architecture, compute arithmetic intensity and roofline-limited tokens/s.
    - Classify bottleneck as compute or memory.

- **3.3 Results and analysis**
  - For each (workload, architecture) pair:
    - Summarize whether it is compute-bound or memory-bound.
    - Comment on utilization and where the design wastes resources.
  - **Key insight**:
    - Training GPU excels at high-throughput training but underutilizes FLOPs on small-batch, memory-heavy inference.
    - Inference accelerator is more efficient for latency-sensitive inference but less ideal for large-scale training.

---

### 4. Specialized Architecture Designs (Goal 3)

- **4.1 Training-optimized architecture**
  - **Datapath design**:
    - Wide SIMD / matrix units, optimized for BF16/BF8 training.
    - Strong support for fused kernels (GEMM + activation + layer-norm).
  - **Memory hierarchy**:
    - HBM with very high bandwidth.
    - On-chip SRAM for activations and gradients reuse.
  - **Interconnect assumptions**:
    - High-radix GPU-style network for data/tensor parallelism (e.g., NVLink-like).
  - **Scheduling model**:
    - Static graphs, large batches, deep pipeline parallelism.
  - Explain why this design aligns with the training workload characteristics from Section 2.

- **4.2 Inference-optimized architecture**
  - **Datapath design**:
    - Mixed-precision units (INT8/INT4) with accurate accumulators.
    - Specialized units for attention / KV-cache access.
  - **Memory hierarchy**:
    - Larger on-chip cache for hot weights and KV-cache slices.
    - Emphasis on memory bandwidth and locality over raw compute.
  - **Interconnect assumptions**:
    - Scaled to serve many concurrent streams rather than deep data parallelism.
  - **Scheduling model**:
    - Dynamic batching, token-level scheduling, support for variable-length sequences.
  - Connect this to the inference workload bottlenecks (latency, KV-cache, memory-bound behavior).

---

### 5. Optional Unified Architecture (Goal 4 – Bonus)

- **5.1 Unified design sketch**
  - Briefly describe a single architecture that tries to handle both training and inference.
  - Mention compromises: moderate FLOPs, moderate bandwidth, moderate on-chip storage, generic control logic.

- **5.2 Efficiency loss analysis**
  - Where it loses vs. the training-optimized design:
    - Less extreme FLOP density, less ideal for huge batches.
  - Where it loses vs. the inference-optimized design:
    - Less specialized KV-cache handling, worse latency/energy per token.
  - Argue that some conflicts (e.g., batch size vs. tail latency) are irreducible.

---

### 6. Cost and Deployment Implications (Goal 5)

- **6.1 Metrics**
  - Training:
    - Throughput (tokens/sec), scaling efficiency, convergence stability.
  - Inference:
    - Token latency, tail latency (p99), energy per token.

- **6.2 Hardware utilization**
  - How well each proposed architecture keeps its compute units and memory system busy for the two workloads.
  - Link back to your analytical results.

- **6.3 Energy and TCO**
  - Qualitatively compare:
    - Energy per training step vs. energy per token.
    - Fleet-level cost when most of the time is spent on inference.
  - Argue why specialization for inference can significantly reduce TCO.

- **6.4 Deployment recommendation**
  - Recommend:
    - A training cluster with training-optimized accelerators.
    - An inference fleet with inference-optimized accelerators.
  - Optionally discuss where a unified design might make sense (e.g., small-scale labs or edge deployment).

---

### 7. Implementation / Modeling Description

- **7.1 Implementation choice**
  - State that you implemented an **analytical comparison framework** (`project8_analysis.py`).

- **7.2 Parameters and experiments**
  - Describe how you:
    - Defined one training workload and one inference workload for the same transformer family.
    - Defined three architectures (training-optimized, inference-optimized, unified).
    - Computed arithmetic intensity and roofline-limited throughput.

- **7.3 Reproducibility**
  - Mention:
    - Python version and how to run: `python project8_analysis.py`.
    - Which outputs (tables/numbers) you copied into the report.

---

### 8. Conclusion

- **Key lessons**
  - Summarize the critical differences between training-time and inference-time architectures.
  - Emphasize why “one accelerator to rule them all” is hard in practice.

- **Future directions**
  - Brief notes on:
    - Better compilers/schedulers that can close some of the gap.
    - Potential hardware–software co-design for LLMs.

