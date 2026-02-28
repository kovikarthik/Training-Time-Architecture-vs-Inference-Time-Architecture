from dataclasses import dataclass
from typing import Literal


WorkloadType = Literal["training", "inference"]


@dataclass
class Workload:
    """
    Simple analytical description of a transformer-style workload.

    All rates are in 'per token' unless otherwise noted.
    """

    name: str
    kind: WorkloadType

    # Model / algorithmic properties
    flops_per_token: float          # total floating-point ops per token
    bytes_read_per_token: float     # bytes read from memory per token
    bytes_written_per_token: float  # bytes written to memory per token

    # Parallelism / execution configuration
    global_batch_size: int
    data_parallel_degree: int
    tensor_parallel_degree: int
    pipeline_stages: int

    # Training-only knobs
    sequence_length: int
    has_gradients: bool
    has_kv_cache: bool


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
    else:
        bottleneck = "memory"

    return EvaluationResult(
        workload=workload,
        arch=arch,
        arithmetic_intensity=ai,
        roofline_flops_bound_toks_per_s=compute_bound_toks_per_s,
        roofline_mem_bound_toks_per_s=mem_bound_toks_per_s,
        bottleneck=bottleneck,
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
    print()


def example_scenarios() -> None:
    """
    Example characterization for a single LLM family:
      - Training: data-parallel, large batch, gradients + activations
      - Inference: single-stream autoregressive with KV cache

    Numbers are illustrative, not precise.
    """

    # Rough per-token numbers for a medium transformer at BF16 / FP16
    # These are intentionally simple for the project writeup.
    training_workload = Workload(
        name="Transformer-XXL Training",
        kind="training",
        flops_per_token=2.0e12,  # 2 TFLOPs / token (forward + backward)
        bytes_read_per_token=200.0e6,  # params + activations + gradients
        bytes_written_per_token=120.0e6,
        global_batch_size=2048,
        data_parallel_degree=16,
        tensor_parallel_degree=8,
        pipeline_stages=8,
        sequence_length=2048,
        has_gradients=True,
        has_kv_cache=False,
    )

    inference_workload = Workload(
        name="Transformer-XXL Inference",
        kind="inference",
        flops_per_token=4.0e11,  # ~0.4 TFLOPs / token inference-only
        bytes_read_per_token=80.0e6,  # weights + KV cache
        bytes_written_per_token=32.0e6,  # KV cache + output token
        global_batch_size=16,  # small batch (interactive)
        data_parallel_degree=1,
        tensor_parallel_degree=2,
        pipeline_stages=2,
        sequence_length=2048,
        has_gradients=False,
        has_kv_cache=True,
    )

    # Architecture 1: Training-optimized GPU (e.g., A100-class)
    training_gpu = Architecture(
        name="Training-GPU-like",
        kind="training_opt",
        peak_flops_tflops=300.0,  # at BF16/FP16
        mem_bandwidth_gbs=1550.0,
        data_parallel_friendly=0.9,
        tensor_parallel_friendly=0.9,
        pipeline_parallel_friendly=0.8,
        autoregressive_friendly=0.4,
    )

    # Architecture 2: Inference-optimized accelerator (e.g., KV-cache aware, INT4)
    inference_accel = Architecture(
        name="Inference-Accel-like",
        kind="inference_opt",
        peak_flops_tflops=150.0,  # lower raw FLOPs but good for low-precision
        mem_bandwidth_gbs=2200.0,  # very high effective bandwidth
        data_parallel_friendly=0.4,
        tensor_parallel_friendly=0.5,
        pipeline_parallel_friendly=0.6,
        autoregressive_friendly=0.9,
    )

    # Optional: unified architecture to discuss compromises
    unified_arch = Architecture(
        name="Unified-Compromise",
        kind="unified",
        peak_flops_tflops=220.0,
        mem_bandwidth_gbs=1700.0,
        data_parallel_friendly=0.7,
        tensor_parallel_friendly=0.7,
        pipeline_parallel_friendly=0.7,
        autoregressive_friendly=0.7,
    )

    for arch in (training_gpu, inference_accel, unified_arch):
        for workload in (training_workload, inference_workload):
            result = roofline_throughput(workload, arch)
            pretty_print_result(result)


if __name__ == "__main__":
    example_scenarios()

