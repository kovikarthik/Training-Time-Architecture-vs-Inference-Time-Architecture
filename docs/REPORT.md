**Training-Time Architecture vs. Inference-Time Architecture**
CECS 530 – Advanced Computer Architecture | California State University, Long Beach
by Karthik Kovi, Mohit Krishna Emani, Abishek Mannam

**Abstract** – This project establishes an analytical comparison framework to investigate the reasons behind the poor performance of hardware optimized for LLM training during inference, and vice versa. We use Llama 3 8B model (~8B parameters) to characterize both training and inference workloads through first-principles FLOP counting and memory traffic estimation. Subsequently, we evaluate these workloads on three architectures using a roofline bottleneck model: a Training-GPU (H100-class, 989 TFLOP/s, 3.35 TB/s, 700 W), an Inference-ASIC (400 TFLOP/s, 15 TB/s SRAM, 75 W), and a Unified-GPU (L40S-class, 733 TFLOP/s, 864 GB/s, 350 W). Our findings show that training is compute-bound, with an arithmetic intensity of 1,103 operations per byte, while inference is memory-bound, with an AI of 20.85 operations per byte. The Inference-ASIC demonstrates a remarkable 4.5x higher inference throughput, 42x lower energy consumption per token, 187x improved energy-delay product (EDP), and 45x reduced cost per million tokens compared to the H100. On the other hand, the Unified-GPU performs the worst during inference, exhibiting a staggering 7.5x worse EDP than the H100, highlighting the failure of compromise architectures at both extremes.

**I. Introduction**

Modern AI systems make use of the same GPU hardware for both training and inference tasks. During training, the system optimizes for throughput by performing large-batch matrix multiplications (GEMMs) with high data reuse. In contrast, during inference, the system generates tokens autoregressively, which involves collapsing GEMMs into matrix-vector multiplies (GEMVs) with minimal weight reuse. These two workloads are fundamentally different: training is compute-bound, while inference is memory-bandwidth-bound. This project aims to quantify the mismatch between these workloads and evaluate specialized architectures that can address this disparity.

**II. Methodology**

*   **Implementation:** The framework is available at github.com/kovikarthik/Training-Time-Architecture-vs-Inference-Time-Architecture.
    It consists of: 1. Standalone roofline evaluation with dataclasses for model, workload, and architecture specifications.2. Modular package with workload characterization (parallelism taxonomy, memory footprint, precision analysis), architecture evaluation, and metrics computation.3. YAML-driven configuration for reproducible parameterization; and a module for automated experiment execution producing timestamped JSON results.

*   **FLOP Model:** Following standard approximations, training is done. FLOPs = 6P Per token (forward pass, backward pass, and update), and inference FLOPs = 2P For Llama 3 8B, the forward pass requires a token per parameter, where P represents the total number of parameters. training = 48.18 x 10^9 Floating point operations per token. inference = 16.06 x 10^9 Floating point operations per token.

*   **Memory Traffic Model:** Training amortizes weight reads across a global batch of 4,096 tokens, incorporating optimizer state traffic (8 bytes per parameter for Adam). Inference reads all weights per token (amortized over batch 16) plus KV-cache entries proportional to the sequence length (4,096 tokens). Training consumes 43.67 MB of bytes per token, while inference consumes 770.40 MB of bytes per token.

*   **Roofline Model:** Compute-Bound throughput= peakFLOP/s/ FLOPs For each token. Memory-bound throughput= Bandwidth, measured in bytes per token, is effective. throughput = min(compute-bound, Memory-bound). Energy per token = TDP x (0.4 + 0.6 x utilization) / throughput. EDP = energy x latencyThe total cost of ownership (TCO) was calculated over a period of three years at a rate of $0.10 per kilowatt-hour.

*   **Architectures and Workloads:**

| Spec | Training-GPU (H100) | Inference-ASIC | Unified (L40S) |
| --- | --- | --- | --- |
| Peak Compute | 989 TFLOP/s (BF16) | 400 TFLOP/s (INT8) | 733 TFLOP/s (INT8) |
| Memory BW | 3,350 GB/s (HBM3) | 15,000 GB/s (SRAM) | 864 GB/s (GDDR6) |
| TDP / Cost | 700 W / $30K | 75 W / $3K | 350 W / $10K |

Training workload: Llama 3 8B, BF16, batch 4096, seq 4096, DP=32, TP=4, PP=4.
Inference workload: Llama, INT8, batch 16, seq 4096, KV-cache, DP=1, TP=2.

**III. Results**

**A. Workload Characterization:**

| Metric | Training | Inference | Insight |
| --- | --- | --- | --- |
| FLOPs/token | 48.18 x 10^9 | 16.06 x 10^9 | Training does 3x more compute |
| Bytes/token | 43.67 MB | 770.40 MB | Inference moves 17.6x more data |
| Arithmetic Intensity | 1,103 ops/byte | 20.85 ops/byte | 53x gap – opposing roofline regimes |
| Parallelism | DP=32, TP=4, PP=4 | DP=1, TP=2, PP=1 | Training parallelizable; inference constrained |

**B. Cross-Architecture Evaluation:**

| Architecture | Workload | Throughput | Bottleneck | Energy/tok | EDP (uJ.s) | $/1M tok |
| --- | --- | --- | --- | --- | --- | --- |
| Training-GPU (H100) | Training | 20,527 tok/s | Compute | 34.10 mJ | 1.66 | $0.0164 |
| Training-GPU (H100) | Inference | 4,348 tok/s | Memory | 160.98 mJ | 37.02 | $0.0774 |
| Inference-ASIC | Training | 8,302 tok/s | Compute | 9.03 mJ | 1.09 | $0.0041 |
| Inference-ASIC | Inference | 19,471 tok/s | Memory | 3.85 mJ | 0.20 | $0.0017 |
| Unified (L40S) | Training | 15,213 tok/s | Compute | 23.01 mJ | 1.51 | $0.0076 |
| Unified (L40S) | Inference | 1,122 tok/s | Memory | 312.08 mJ | 278.27 | $0.1029 |

**C. Key Comparisons (Inference workload):**

| Metric | H100 | ASIC | ASIC Improvement |
| --- | --- | --- | --- |
| Throughput | 4,348 tok/s | 19,471 tok/s | 4.5x higher |
| Energy/token | 160.98 mJ | 3.85 mJ | 42x lower |
| EDP | 37.02 uJ.s | 0.20 uJ.s | 187x better |
| Cost/1M tokens | $0.0774 | $0.0017 | 45x cheaper |
| Compute utilization | ~7% | ~78% | Right-sized |

The Unified-GPU (L40S) achieves the worst inference results of all three: 1,122 tok/s (3.9x slower than H100), 312.08 mJ/token (1.9x worse than H100), and EDP 278.27 uJ.s (7.5x worse than H100). Its 864 GB/s GDDR6 bandwidth is insufficient for memory-bound inference, while its 733 TFLOP/s compute is only 74% of the H100 for training. The unified design fails at both workloads.

**IV. Conclusions**
1. Training and inference are fundamentally different workloads: training is compute-bound (AI = 1,103 ops/byte), inference is memory-bound (AI = 20.85 ops/byte). The 53x arithmetic intensity gap means one chip cannot efficiently serve both.
2. Training-optimized GPUs waste 93% of their compute during inference. The H100 achieves 20,527 tok/s on training but only 4,348 tok/s on inference.
3. Specialized inference hardware delivers transformative gains: 42x lower energy, 187x better EDP, 45x lower cost per million tokens.
4. Unified architectures fail at both extremes. The L40S is worse than the H100 on inference (7.5x worse EDP) and slower on training (74% throughput).
5. Architectural specialization is an economic necessity, not a luxury. As inference dominates 80-90% of production LLM compute, the 45x cost penalty of using training hardware is economically untenable.

**References**
[1] Megatron-LM. https://github.com/NVIDIA/Megatron-LM [2] DeepSpeed. https://github.com/microsoft/DeepSpeed [3] vLLM. https://github.com/vllm-project/vllm [4] llama.cpp. https://github.com/ggerganov/llama.cpp [5] TVM. https://github.com/apache/tvm [6] PyTorch. https://github.com/pytorch/pytorch [7] S. Williams, A. Waterman, D. Patterson, “Roofline: An Insightful Visual Perfor-mance Model for Multicore Architectures,” CACM, vol. 52, no. 4, pp. 65-76, 2009. [8] NVIDIA Corporation, “NVIDIA H100 Tensor Core GPU Architecture,” Whitepaper, 2022.
