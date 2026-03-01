"""
Architecture models for training vs. inference accelerators.
CECS 530 Project 8 - Goal 2 & 3
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

ArchitectureKind = Literal["training_opt", "inference_opt", "unified"]


@dataclass
class Architecture:
    """
    Simplified architecture model (GPU/accelerator-like).
    - peak_flops_tflops: TFLOP/s at chosen precision
    - mem_bandwidth_gbs: GB/s effective DRAM bandwidth
    - parallelism_*: 0–1 capability scores
    """

    name: str
    kind: ArchitectureKind

    peak_flops_tflops: float
    mem_bandwidth_gbs: float
    
    tdp_watts: float
    cost_usd: float

    data_parallel_friendly: float
    tensor_parallel_friendly: float
    pipeline_parallel_friendly: float
    autoregressive_friendly: float

    description: str = ""
    datapath: str = ""
    memory_hierarchy: str = ""
    interconnect: str = ""


def load_architecture_from_config(
    config_path: str | Path, key: str = "training_optimized"
) -> Architecture:
    """Load an Architecture from YAML config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if key not in data:
        raise KeyError(f"Architecture key '{key}' not found in {config_path}")

    a = data[key]
    return Architecture(
        name=a["name"],
        kind=a["kind"],
        peak_flops_tflops=float(a["peak_flops_tflops"]),
        mem_bandwidth_gbs=float(a["mem_bandwidth_gbs"]),
        tdp_watts=float(a.get("tdp_watts", 400.0)),
        cost_usd=float(a.get("cost_usd", 20000.0)),
        data_parallel_friendly=float(a["data_parallel_friendly"]),
        tensor_parallel_friendly=float(a["tensor_parallel_friendly"]),
        pipeline_parallel_friendly=float(a["pipeline_parallel_friendly"]),
        autoregressive_friendly=float(a["autoregressive_friendly"]),
        description=a.get("description", ""),
        datapath=a.get("datapath", ""),
        memory_hierarchy=a.get("memory_hierarchy", ""),
        interconnect=a.get("interconnect", ""),
    )
