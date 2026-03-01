from dataclasses import dataclass
from typing import Literal


WorkloadType = Literal["training", "inference"]


@dataclass
class ModelArchitecture:
    """Hyperparameters of the target LLM."""
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int

    @property
    def total_parameters(self) -> int:
        emb = self.vocab_size * self.hidden_size
        qkv_size = self.hidden_size + 2 * (self.hidden_size * self.num_key_value_heads // self.num_attention_heads)
        attn = self.hidden_size * qkv_size + self.hidden_size * self.hidden_size
        mlp = 3 * self.hidden_size * self.intermediate_size
        norm = 2 * self.hidden_size
        layer_params = attn + mlp + norm
        return emb + self.num_layers * layer_params + emb

@dataclass
class Workload:
    name: str
    kind: WorkloadType
    model: ModelArchitecture

    global_batch_size: int
    data_parallel_degree: int
    tensor_parallel_degree: int
    pipeline_stages: int

    sequence_length: int
    has_gradients: bool
    has_kv_cache: bool
    precision: str = "FP16"

    @property
    def precision_bytes(self) -> int:
        return 1 if self.precision == "INT8" else 2

    @property
    def flops_per_token(self) -> float:
        p = self.model.total_parameters
        return 6.0 * p if self.has_gradients else 2.0 * p

    @property
    def bytes_read_per_token(self) -> float:
        p = self.model.total_parameters
        if self.has_gradients:
            act_bytes = 34 * self.model.hidden_size * self.model.num_layers * self.sequence_length / self.global_batch_size
            return p * (self.precision_bytes + 8) / self.global_batch_size + act_bytes
        else:
            kv_size_per_token = 2 * self.model.num_layers * (self.model.hidden_size * self.model.num_key_value_heads // self.model.num_attention_heads) * self.precision_bytes
            return p * self.precision_bytes / self.global_batch_size + kv_size_per_token * self.sequence_length

    @property
    def bytes_written_per_token(self) -> float:
        p = self.model.total_parameters
        if self.has_gradients:
            return p * (self.precision_bytes + 8) / self.global_batch_size
        else:
            kv_size_per_token = 2 * self.model.num_layers * (self.model.hidden_size * self.model.num_key_value_heads // self.model.num_attention_heads) * self.precision_bytes
            return kv_size_per_token + self.model.hidden_size * self.precision_bytes


@dataclass
class Architecture:
    """
    Simplified architecture model, roughly "GPU-like".

    Units:
    - peak_flops: TFLOP/s (10^12 FLOPs / second) at chosen precision
    - mem_bandwidth: GB/s (10^9 bytes / second) of effective DRAM bandwidth
    """

    name: str
    kind: Literal["training_opt", "inference_opt", "unified"]

    peak_flops_tflops: float
    mem_bandwidth_gbs: float
    tdp_watts: float
    cost_usd: float

    # Parallelism support (normalized 0–1, rough capability scores)
    data_parallel_friendly: float
    tensor_parallel_friendly: float
    pipeline_parallel_friendly: float
    autoregressive_friendly: float


@dataclass
class EvaluationResult:
    workload: Workload
    arch: Architecture

    arithmetic_intensity: float  # FLOPs / byte
    roofline_flops_bound_toks_per_s: float
    roofline_mem_bound_toks_per_s: float
    bottleneck: Literal["compute", "memory"]
    
    # New metrics for Phase 4
    energy_per_token_mj: float
    edp_uj_s: float
    cost_per_million_tokens_usd: float


def arithmetic_intensity(workload: Workload) -> float:
    bytes_total = workload.bytes_read_per_token + workload.bytes_written_per_token
    if bytes_total == 0:
        return float("inf")
    return workload.flops_per_token / bytes_total


def roofline_throughput(workload: Workload, arch: Architecture) -> EvaluationResult:
    """
    Simple roofline-style model:
      - compute-bound rate: peak_flops / flops_per_token
      - memory-bound rate: mem_bandwidth / bytes_per_token
    """

    ai = arithmetic_intensity(workload)
    bytes_total = workload.bytes_read_per_token + workload.bytes_written_per_token

    flops_per_token = workload.flops_per_token
    if flops_per_token <= 0 or bytes_total <= 0:
        raise ValueError("Invalid workload parameters.")

    # Convert TFLOP/s to FLOP/s and GB/s to B/s
    peak_flops = arch.peak_flops_tflops * 1e12
    mem_bandwidth = arch.mem_bandwidth_gbs * 1e9

    compute_bound_toks_per_s = peak_flops / flops_per_token
    mem_bound_toks_per_s = mem_bandwidth / bytes_total

    if compute_bound_toks_per_s < mem_bound_toks_per_s:
        bottleneck: Literal["compute", "memory"] = "compute"
        actual_toks_per_s = compute_bound_toks_per_s
    else:
        bottleneck = "memory"
        actual_toks_per_s = mem_bound_toks_per_s

    # Phase 4 Metrics: Energy and Cost
    # Assuming baseline 40% power + up to 60% based on utilization
    # Since we are assuming roofline limits, utilization of wildcard resource is 1.0
    utilization = 1.0 # simplified model
    power_watts = arch.tdp_watts * (0.4 + 0.6 * utilization)
    
    # Energy per token in milliJoules
    time_per_token_s = 1.0 / actual_toks_per_s
    energy_per_token_j = power_watts * time_per_token_s
    energy_per_token_mj = energy_per_token_j * 1000.0
    
    # Energy-Delay Product (EDP): Energy (J) * Delay (s)
    edp_uj_s = (energy_per_token_j * 1e6) * time_per_token_s
    
    # TCO over 3 years: assuming $0.10/kWh
    years = 3
    hours = years * 8760
    # $0.10 per kWh
    electricity_cost = (power_watts / 1000) * hours * 0.10
    total_tco = arch.cost_usd + electricity_cost
    
    # Total tokens this chip can process in 3 years at peak roofline 
    total_tokens_3yrs = actual_toks_per_s * hours * 3600
    cost_per_million_tokens_usd = (total_tco / total_tokens_3yrs) * 1e6

    return EvaluationResult(
        workload=workload,
        arch=arch,
        arithmetic_intensity=ai,
        roofline_flops_bound_toks_per_s=compute_bound_toks_per_s,
        roofline_mem_bound_toks_per_s=mem_bound_toks_per_s,
        bottleneck=bottleneck,
        energy_per_token_mj=energy_per_token_mj,
        edp_uj_s=edp_uj_s,
        cost_per_million_tokens_usd=cost_per_million_tokens_usd,
    )


def pretty_print_result(result: EvaluationResult) -> None:
    w = result.workload
    a = result.arch
    print(f"=== {w.name} on {a.name} ===")
    print(f"  Workload type     : {w.kind}")
    print(f"  Arithmetic intensity (FLOPs/byte): {result.arithmetic_intensity:8.3f}")
    print(
        f"  Compute-bound throughput (tokens/s): "
        f"{result.roofline_flops_bound_toks_per_s:10.2f}"
    )
    print(
        f"  Memory-bound throughput  (tokens/s): "
        f"{result.roofline_mem_bound_toks_per_s:10.2f}"
    )
    print(f"  Bottleneck         : {result.bottleneck}")
    print(f"  Energy/token (mJ)  : {result.energy_per_token_mj:10.2f}")
    print(f"  EDP (µJ·s)         : {result.edp_uj_s:10.2f}")
    print(f"  TCO Cost / 1M toks : ${result.cost_per_million_tokens_usd:9.4f}")
    print()


def example_scenarios() -> None:
    """
    Example characterization for a modern Llama 3 8B-class model:
      - Training: data-parallel, large batch, gradients + activations
      - Inference: single-stream autoregressive with KV cache
    """

    llama3_8b_arch = ModelArchitecture(
        hidden_size=4096,
        intermediate_size=14336,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=128256
    )

    training_workload = Workload(
        name="LLM-8B Training (BF16)",
        kind="training",
        model=llama3_8b_arch,
        global_batch_size=4096,
        data_parallel_degree=32,
        tensor_parallel_degree=4,
        pipeline_stages=4,
        sequence_length=4096,
        has_gradients=True,
        has_kv_cache=False,
        precision="BF16"
    )

    inference_workload = Workload(
        name="LLM-8B Inference (INT8/KV-cache)",
        kind="inference",
        model=llama3_8b_arch,
        global_batch_size=16, 
        data_parallel_degree=1,
        tensor_parallel_degree=2,
        pipeline_stages=1,
        sequence_length=4096,
        has_gradients=False,
        has_kv_cache=True,
        precision="INT8"
    )

    # Architecture 1: Training-optimized GPU (e.g., H100-class)
    training_gpu = Architecture(
        name="Training-GPU (H100-class)",
        kind="training_opt",
        peak_flops_tflops=989.0,  # BF16
        mem_bandwidth_gbs=3350.0,
        tdp_watts=700.0,
        cost_usd=30000.0,
        data_parallel_friendly=0.9,
        tensor_parallel_friendly=0.9,
        pipeline_parallel_friendly=0.8,
        autoregressive_friendly=0.4,
    )

    # Architecture 2: Inference-optimized accelerator (e.g., massive SRAM ASIC)
    inference_accel = Architecture(
        name="Inference-ASIC (SRAM-heavy)",
        kind="inference_opt",
        peak_flops_tflops=400.0,
        mem_bandwidth_gbs=15000.0, # 15 TB/s SRAM bandwidth
        tdp_watts=75.0,
        cost_usd=3000.0,
        data_parallel_friendly=0.2,
        tensor_parallel_friendly=0.6,
        pipeline_parallel_friendly=0.8,
        autoregressive_friendly=1.0,
    )

    # Optional: unified architecture to discuss compromises (e.g. L40S)
    unified_arch = Architecture(
        name="Unified-GPU (L40S-class)",
        kind="unified",
        peak_flops_tflops=733.0,
        mem_bandwidth_gbs=864.0,
        tdp_watts=350.0,
        cost_usd=10000.0,
        data_parallel_friendly=0.7,
        tensor_parallel_friendly=0.6,
        pipeline_parallel_friendly=0.7,
        autoregressive_friendly=0.6,
    )

    for arch in (training_gpu, inference_accel, unified_arch):
        for workload in (training_workload, inference_workload):
            result = roofline_throughput(workload, arch)
            pretty_print_result(result)


if __name__ == "__main__":
    example_scenarios()

