"""
Inference metrics - Section 6 of Project 8.
- Token latency (ms/token)
- Energy per token (notional)
- Tail latency (p99)
"""

from dataclasses import dataclass


@dataclass
class InferenceMetrics:
    token_latency_ms: float
    energy_per_token_j: float   # notional
    tail_latency_p99_ms: float  # approximate as 1.5x mean


def compute_inference_metrics(
    effective_throughput_tokens_per_s: float,
    power_watts: float = 150.0,  # example accelerator power
) -> InferenceMetrics:
    """Derive inference metrics from throughput."""
    tokens_per_s = effective_throughput_tokens_per_s
    if tokens_per_s <= 0:
        tokens_per_s = 1e-6  # avoid div by zero
    latency_ms = 1000.0 / tokens_per_s
    energy_per_token = power_watts / tokens_per_s  # J/token
    p99 = latency_ms * 1.5  # simplified tail latency model
    return InferenceMetrics(
        token_latency_ms=latency_ms,
        energy_per_token_j=energy_per_token,
        tail_latency_p99_ms=p99,
    )
