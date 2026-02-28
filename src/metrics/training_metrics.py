"""
Training metrics - Section 6 of Project 8.
- Throughput (tokens/sec)
- Scaling efficiency
- Convergence stability (notional)
"""

from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    throughput_tokens_per_s: float
    scaling_efficiency: float  # 0–1, fraction of ideal scaling
    convergence_stable: bool   # qualitative


def compute_training_metrics(
    effective_throughput: float,
    batch_size: int,
    num_devices: int,
) -> TrainingMetrics:
    """Derive training metrics from roofline-effective throughput."""
    # Scaling efficiency: heuristic based on batch and devices
    ideal_throughput = effective_throughput * num_devices
    actual_throughput = effective_throughput  # per-device view
    scaling_efficiency = min(1.0, actual_throughput / (ideal_throughput / num_devices))
    # Simplified: assume large batch -> more stable
    convergence_stable = batch_size >= 256
    return TrainingMetrics(
        throughput_tokens_per_s=effective_throughput * num_devices,
        scaling_efficiency=scaling_efficiency,
        convergence_stable=convergence_stable,
    )
