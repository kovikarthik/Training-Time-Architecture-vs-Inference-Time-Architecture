"""
Workload models for transformer-style training and inference.
CECS 530 Project 8 - Workload Characterization (Goal 1)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

WorkloadType = Literal["training", "inference"]


WorkloadType = Literal["training", "inference"]

@dataclass
class ModelArchitecture:
    """Hyperparameters of the target LLM."""
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int

    @property
    def total_parameters(self) -> int:
        emb = self.vocab_size * self.hidden_size
        qkv_size = self.hidden_size + 2 * (self.hidden_size * self.num_key_value_heads // self.num_attention_heads)
        attn = self.hidden_size * qkv_size + self.hidden_size * self.hidden_size
        mlp = 3 * self.hidden_size * self.intermediate_size
        norm = 2 * self.hidden_size
        layer_params = attn + mlp + norm
        return emb + self.num_layers * layer_params + emb # Includes tied or untied output head


@dataclass
class Workload:
    """
    Analytical description of a transformer-style workload.
    All rates are per-token unless otherwise noted.
    """

    name: str
    kind: WorkloadType

    name: str
    kind: WorkloadType
    model: ModelArchitecture

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
    def precision_bytes(self) -> int:
        return 1 if self.precision == "INT8" else 2

    @property
    def flops_per_token(self) -> float:
        # Standard analytical approximation:
        # Forward pass is ~2*P FLOPs per token.
        # Backward pass is ~4*P FLOPs per token.
        p = self.model.total_parameters
        return 6.0 * p if self.has_gradients else 2.0 * p

    @property
    def bytes_read_per_token(self) -> float:
        p = self.model.total_parameters
        if self.has_gradients:
            # Training: Read weights (2 bytes) + optimizer states (8 bytes) + act checkpoints
            # Very rough analytical estimation per token per replica
            act_bytes = 34 * self.model.hidden_size * self.model.num_layers * self.sequence_length / self.global_batch_size
            return p * (self.precision_bytes + 8) / self.global_batch_size + act_bytes
        else:
            # Inference: Read weights (1-2 bytes) + KV cache read per token (for generation phase)
            kv_size_per_token = 2 * self.model.num_layers * (self.model.hidden_size * self.model.num_key_value_heads // self.model.num_attention_heads) * self.precision_bytes
            return p * self.precision_bytes / self.global_batch_size + kv_size_per_token * self.sequence_length

    @property
    def bytes_written_per_token(self) -> float:
        p = self.model.total_parameters
        if self.has_gradients:
            # Training: Write gradients and optimizer states
            return p * (self.precision_bytes + 8) / self.global_batch_size
        else:
            # Inference: Write new KV cache token + output token
            kv_size_per_token = 2 * self.model.num_layers * (self.model.hidden_size * self.model.num_key_value_heads // self.model.num_attention_heads) * self.precision_bytes
            return kv_size_per_token + self.model.hidden_size * self.precision_bytes

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
    
    # Check if we have dynamic model definition or fallback to older hardcoded values
    m = w.get("model", {})
    hidden_size = int(m.get("hidden_size", 4096))
    intermediate_size = int(m.get("intermediate_size", 14336))
    num_layers = int(m.get("num_layers", 32))
    num_attention_heads = int(m.get("num_attention_heads", 32))
    num_key_value_heads = int(m.get("num_key_value_heads", 8))
    vocab_size = int(m.get("vocab_size", 128256))
    
    model_arch = ModelArchitecture(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        vocab_size=vocab_size
    )

    return Workload(
        name=w["name"],
        kind=w["kind"],
        model=model_arch,
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
