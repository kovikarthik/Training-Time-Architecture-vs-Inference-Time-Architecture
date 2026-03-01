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

    energy_per_token_mj: float
    edp_uj_s: float
    cost_per_million_tokens_usd: float

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
    actual_toks_per_s = min(compute_bound, mem_bound)

    utilization = 1.0 # simplified model
    power_watts = arch.tdp_watts * (0.4 + 0.6 * utilization)
    
    time_per_token_s = 1.0 / actual_toks_per_s
    energy_per_token_j = power_watts * time_per_token_s
    energy_per_token_mj = energy_per_token_j * 1000.0
    
    edp_uj_s = (energy_per_token_j * 1e6) * time_per_token_s
    
    years = 3
    hours = years * 8760
    electricity_cost = (power_watts / 1000) * hours * 0.10
    total_tco = arch.cost_usd + electricity_cost
    
    total_tokens_3yrs = actual_toks_per_s * hours * 3600
    cost_per_million_tokens_usd = (total_tco / total_tokens_3yrs) * 1e6

    return EvaluationResult(
        workload=workload,
        arch=arch,
        arithmetic_intensity=workload.arithmetic_intensity,
        roofline_flops_bound_toks_per_s=compute_bound,
        roofline_mem_bound_toks_per_s=mem_bound,
        bottleneck=bottleneck,
        energy_per_token_mj=energy_per_token_mj,
        edp_uj_s=edp_uj_s,
        cost_per_million_tokens_usd=cost_per_million_tokens_usd,
    )
