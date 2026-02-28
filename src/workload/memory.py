"""
Memory behavior analysis - Section 3.2 of Project 8.
- Training: activation lifetimes, weight reuse, gradient storage
- Inference: KV-cache, weight reuse, no gradients
"""

from dataclasses import dataclass

from .models import Workload


@dataclass
class MemoryFootprint:
    """Memory footprint comparison for a workload."""

    workload: Workload
    activation_bytes: float
    weight_bytes: float
    gradient_bytes: float
    kv_cache_bytes: float
    total_bytes_per_token: float

    @property
    def bandwidth_pressure(self) -> str:
        """Classify as bandwidth- or capacity-bound."""
        # Inference: often bandwidth-bound due to KV-cache streaming
        if self.workload.has_kv_cache and self.kv_cache_bytes > self.weight_bytes:
            return "bandwidth"
        # Training: can be capacity-bound (large activation checkpoints)
        if self.workload.has_gradients and self.gradient_bytes > 0.5 * self.total_bytes_per_token:
            return "capacity"
        return "bandwidth"


def analyze_memory(workload: Workload) -> MemoryFootprint:
    """
    Estimate memory components from workload parameters.
    Uses simplified heuristic: read/write bytes map to components.
    """
    total = workload.bytes_read_per_token + workload.bytes_written_per_token

    if workload.kind == "training":
        # Training: activations + weights + gradients
        activation_bytes = workload.bytes_read_per_token * 0.4  # activations
        weight_bytes = workload.bytes_read_per_token * 0.35
        gradient_bytes = workload.bytes_written_per_token * 0.7
        kv_cache_bytes = 0.0
    else:
        # Inference: weights + KV-cache dominant
        weight_bytes = workload.bytes_read_per_token * 0.5
        kv_cache_bytes = workload.bytes_read_per_token * 0.3 + workload.bytes_written_per_token * 0.5
        activation_bytes = workload.bytes_read_per_token * 0.2
        gradient_bytes = 0.0

    return MemoryFootprint(
        workload=workload,
        activation_bytes=activation_bytes,
        weight_bytes=weight_bytes,
        gradient_bytes=gradient_bytes,
        kv_cache_bytes=kv_cache_bytes,
        total_bytes_per_token=total,
    )
