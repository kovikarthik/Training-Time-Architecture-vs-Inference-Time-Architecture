"""
Roofline performance model - Section 3, Goal 1 & 2.
Compute-bound vs. memory-bound throughput.
"""

from dataclasses import dataclass
from typing import Literal

from ..workload.models import Workload
from .models import Architecture


@dataclass
class EvaluationResult:
    """Result of roofline analysis for a workload-architecture pair."""

    workload: Workload
    arch: Architecture

    arithmetic_intensity: float
    roofline_flops_bound_toks_per_s: float
    roofline_mem_bound_toks_per_s: float
    bottleneck: Literal["compute", "memory"]

    @property
    def effective_throughput_toks_per_s(self) -> float:
        """Throughput limited by the active bottleneck."""
        return min(
            self.roofline_flops_bound_toks_per_s,
            self.roofline_mem_bound_toks_per_s,
        )


def roofline_throughput(workload: Workload, arch: Architecture) -> EvaluationResult:
    """
    Roofline model:
    - compute-bound: peak_flops / flops_per_token
    - memory-bound: mem_bandwidth / bytes_per_token
    """
    bytes_total = workload.bytes_total_per_token
    if bytes_total <= 0 or workload.flops_per_token <= 0:
        raise ValueError("Invalid workload parameters")

    peak_flops = arch.peak_flops_tflops * 1e12
    mem_bandwidth = arch.mem_bandwidth_gbs * 1e9

    compute_bound = peak_flops / workload.flops_per_token
    mem_bound = mem_bandwidth / bytes_total

    bottleneck: Literal["compute", "memory"] = "compute" if compute_bound < mem_bound else "memory"

    return EvaluationResult(
        workload=workload,
        arch=arch,
        arithmetic_intensity=workload.arithmetic_intensity,
        roofline_flops_bound_toks_per_s=compute_bound,
        roofline_mem_bound_toks_per_s=mem_bound,
        bottleneck=bottleneck,
    )
