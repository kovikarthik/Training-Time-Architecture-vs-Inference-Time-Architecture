# Training-Time vs. Inference-Time Architecture Analysis

**CECS 530 – Advanced Computer Architecture **  

This repository implements an **analytical comparison framework** and **architecture-level performance model** to systematically compare training-time and inference-time architectures for large language models.

---

## Project Overview

Modern AI systems often assume that fast training hardware implies fast inference. This project demonstrates why that assumption is false. We analyze how differences in:

- **Parallelism structure** (data/tensor/pipeline, autoregressive dependency)
- **Memory behavior** (activations, KV-cache, gradients)
- **Numerical precision** (FP32/BF16 training vs INT8/INT4 inference)
- **Control flow** (static graphs vs dynamic batching)
- **Latency vs. throughput objectives**

…lead to fundamentally different optimal architectures.

---

## Requirements

- Python
- (Optional) PyTorch + MPS for Apple Silicon benchmarks

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/training-vs-inference-arch.git
cd training-vs-inference-arch

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run full experiment suite
python scripts/run_experiments.py --config config/experiments.yaml

# Run workload characterization (Goal 1)
python -m src.analysis.workload_characterization

# Run architecture evaluation (Goal 2)
python -m src.analysis.architecture_evaluation
```

---

## Repository Structure

```
├── config/                    # Parameterized configurations
│   ├── workloads.yaml         # Workload definitions (training & inference)
│   ├── architectures.yaml     # Architecture definitions
│   └── experiments.yaml       # Experiment specifications
├── src/
│   ├── workload/              # Workload modeling
│   │   ├── models.py          # Workload dataclass
│   │   ├── parallelism.py    # Parallelism taxonomy & analysis
│   │   └── memory.py         # Memory behavior (activations, KV-cache, gradients)
│   ├── architecture/          # Architecture modeling
│   │   ├── models.py         # Architecture dataclass
│   │   └── roofline.py       # Roofline performance model
│   ├── analysis/              # Analysis modules
│   │   ├── workload_characterization.py   
│   │   └── architecture_evaluation.py     
│   └── metrics/               # Evaluation metrics
│       ├── training_metrics.py
│       └── inference_metrics.py
├── scripts/
│   ├── run_experiments.py     # Main experiment runner
│   └── benchmark_apple_silicon.py  
├── docs/
│   └── REPORT.md              # Technical report
├── requirements.txt
└── README.md
```

---

## Implementation Highlights

### 1. Workload Characterization 

- **Operation mix**: Matrix multiplies, attention, layer-norm for training vs. inference
- **Arithmetic intensity**: FLOPs/byte for roofline classification
- **Memory access patterns**: Activation lifetimes, weight reuse, KV-cache (inference), gradient storage (training)
- **Deliverable**: Roofline-style comparison, bottleneck identification

### 2. Architecture Evaluation 

- Evaluates training-optimized and inference-optimized architectures against both workloads
- Explains where each excels and where it fails
- **Deliverable**: Architecture critique, mismatch analysis

### 3. Specialized Architecture Designs

- **Training-optimized**: Datapath, memory hierarchy, interconnect, scheduling model
- **Inference-optimized**: KV-cache aware, low-precision support, dynamic batching
- **Deliverable**: Design rationale, parameterization

### 4. Unified Architecture 

- Explains compromises and quantifies efficiency loss
- **Deliverable**: Efficiency loss analysis

### 5. Cost & Deployment 

- Hardware utilization analysis
- Energy per token vs. per step
- Fleet-level cost impact

---

## Evaluation Metrics

| Training | Inference |
|----------|-----------|
| Throughput (tokens/sec) | Token latency |
| Scaling efficiency | Energy per token |
| Convergence stability | Tail latency (p99) |

---

## Reproducibility

All experiments are parameterized via YAML configs. To reproduce:

```bash
python scripts/run_experiments.py --config config/experiments.yaml --output results/
```

---


## References

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [vLLM](https://github.com/vllm-project/vllm)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)


