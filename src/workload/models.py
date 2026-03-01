"""
Workload models for transformer-style training and inference.
CECS 530 Project 8 - Workload Characterization (Goal 1)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

WorkloadType = Literal["training", "inference"]


@dataclass
class Workload:
    """
    Analytical description of a transformer-style workload.
    All rates are per-token unless otherwise noted.
    """

    name: str
    kind: WorkloadType

    # Model / algorithmic properties
    flops_per_token: float
    bytes_read_per_token: float
    bytes_written_per_token: float

    # Parallelism / execution configuration
    global_batch_size: int
    data_parallel_degree: int
    tensor_parallel_degree: int
    pipeline_stages: int
    sequence_length: int

    # Workload-specific
    has_gradients: bool
    has_kv_cache: bool

    # Optional metadata
    description: str = ""
    precision: str = "FP16"
    operation_mix: dict[str, float] | None = None

    @property
    def bytes_total_per_token(self) -> float:
        return self.bytes_read_per_token + self.bytes_written_per_token

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte (arithmetic intensity)."""
        bt = self.bytes_total_per_token
        return self.flops_per_token / bt if bt > 0 else float("inf")


def load_workload_from_config(config_path: str | Path, key: str = "training") -> Workload:
    """Load a Workload from a YAML config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if key not in data:
        raise KeyError(f"Workload key '{key}' not found in {config_path}")

    w = data[key]
    return Workload(
        name=w["name"],
        kind=w["kind"],
        flops_per_token=float(w["flops_per_token"]),
        bytes_read_per_token=float(w["bytes_read_per_token"]),
        bytes_written_per_token=float(w["bytes_written_per_token"]),
        global_batch_size=int(w["global_batch_size"]),
        data_parallel_degree=int(w["data_parallel_degree"]),
        tensor_parallel_degree=int(w["tensor_parallel_degree"]),
        pipeline_stages=int(w["pipeline_stages"]),
        sequence_length=int(w["sequence_length"]),
        has_gradients=bool(w["has_gradients"]),
        has_kv_cache=bool(w["has_kv_cache"]),
        description=w.get("description", ""),
        precision=w.get("precision", "FP16"),
        operation_mix=w.get("operation_mix"),
    )
