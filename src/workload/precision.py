"""
Precision and numerical behavior - Section 3.3 of Project 8.
- FP32/BF16 training requirements
- INT8/INT4 inference opportunities
- Accumulation accuracy vs. throughput
"""

from dataclasses import dataclass

from .models import Workload, WorkloadType


@dataclass
class PrecisionTradeoff:
    """Precision tradeoff for training vs. inference."""

    workload_kind: WorkloadType
    compute_precision: str  # FP32, BF16, FP16, INT8, INT4
    accumulation_precision: str
    effective_flops_per_token: float
    effective_bytes_per_token: float
    accuracy_impact: str  # "high", "moderate", "low"
    throughput_benefit: str  # "low", "moderate", "high"


def analyze_precision(workload: Workload) -> PrecisionTradeoff:
    """
    Per project spec Section 3.3:
    - Training: FP32/BF16 for stability, accumulation in FP32
    - Inference: INT8/INT4 for efficiency, lower accumulation requirements
    """
    if workload.kind == "training":
        return PrecisionTradeoff(
            workload_kind="training",
            compute_precision="BF16",
            accumulation_precision="FP32",
            effective_flops_per_token=workload.flops_per_token,
            effective_bytes_per_token=workload.bytes_total_per_token,
            accuracy_impact="high",
            throughput_benefit="low",
        )
    else:
        # Inference can use INT8/INT4
        return PrecisionTradeoff(
            workload_kind="inference",
            compute_precision="INT8",
            accumulation_precision="INT32",
            effective_flops_per_token=workload.flops_per_token * 0.5,  # lower precision = fewer effective FLOPs
            effective_bytes_per_token=workload.bytes_total_per_token * 0.5,
            accuracy_impact="low",
            throughput_benefit="high",
        )
