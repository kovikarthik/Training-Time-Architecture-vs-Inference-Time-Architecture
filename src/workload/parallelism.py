"""
Parallelism structure analysis - Section 3.1 of Project 8.
Training: dominant data/tensor/pipeline parallelism.
Inference: minimal data parallelism, fundamental autoregressive dependency.
"""

from dataclasses import dataclass

from .models import Workload, WorkloadType


@dataclass
class ParallelismTaxonomy:
    """Parallelism taxonomy for a workload."""

    workload: Workload
    data_parallelism: str  # "Dominant" | "Minimal" | "Moderate"
    tensor_parallelism: str  # "Common" | "Limited" | "Moderate"
    pipeline_parallelism: str  # "Deep" | "Shallow" | "Moderate"
    autoregressive_dependency: str  # "None" | "Fundamental" | "Present"

    def to_dict(self) -> dict:
        return {
            "workload": self.workload.name,
            "data_parallelism": self.data_parallelism,
            "tensor_parallelism": self.tensor_parallelism,
            "pipeline_parallelism": self.pipeline_parallelism,
            "autoregressive_dependency": self.autoregressive_dependency,
        }


def analyze_parallelism(workload: Workload) -> ParallelismTaxonomy:
    """
    Classify parallelism structure per project spec Section 3.1:
    - Training: data dominant, tensor common, pipeline deep, autoregressive none
    - Inference: data minimal, tensor limited, pipeline shallow, autoregressive fundamental
    """
    if workload.kind == "training":
        return ParallelismTaxonomy(
            workload=workload,
            data_parallelism="Dominant" if workload.data_parallel_degree >= 8 else "Moderate",
            tensor_parallelism="Common" if workload.tensor_parallel_degree >= 4 else "Limited",
            pipeline_parallelism="Deep" if workload.pipeline_stages >= 4 else "Shallow",
            autoregressive_dependency="None",
        )
    else:
        return ParallelismTaxonomy(
            workload=workload,
            data_parallelism="Minimal" if workload.data_parallel_degree <= 2 else "Moderate",
            tensor_parallelism="Limited" if workload.tensor_parallel_degree <= 4 else "Moderate",
            pipeline_parallelism="Shallow" if workload.pipeline_stages <= 4 else "Moderate",
            autoregressive_dependency="Fundamental",
        )


def scalability_summary(workload: Workload) -> str:
    """Brief scalability analysis."""
    if workload.kind == "training":
        return (
            f"Scales with data_parallel={workload.data_parallel_degree}, "
            f"tensor_parallel={workload.tensor_parallel_degree}, "
            f"pipeline={workload.pipeline_stages}. "
            "Throughput-optimized; batch size dominates."
        )
    else:
        return (
            f"Limited scalability: data_parallel={workload.data_parallel_degree}, "
            f"tensor_parallel={workload.tensor_parallel_degree}. "
            "Latency-optimized; autoregressive dependency restricts parallelism."
        )
