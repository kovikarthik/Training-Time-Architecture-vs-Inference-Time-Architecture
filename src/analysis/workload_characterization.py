"""
Goal 1: Workload Characterization
- Operation mix, arithmetic intensity, memory access patterns
- Roofline-style comparison, bottleneck identification
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.workload import Workload, load_workload_from_config
from src.workload.parallelism import analyze_parallelism, scalability_summary
from src.workload.memory import analyze_memory


def run(config_dir: Path) -> dict:
    """Run workload characterization and return structured results."""
    config_dir = Path(config_dir)
    workload_config = config_dir / "workloads.yaml"

    results = {}

    for key in ["training", "inference"]:
        w = load_workload_from_config(workload_config, key)

        # Parallelism taxonomy (Section 3.1)
        par = analyze_parallelism(w)
        results[key] = {
            "workload": w.name,
            "kind": w.kind,
            "flops_per_token": w.flops_per_token,
            "bytes_read_per_token": w.bytes_read_per_token,
            "bytes_written_per_token": w.bytes_written_per_token,
            "arithmetic_intensity": w.arithmetic_intensity,
            "parallelism": par.to_dict(),
            "scalability": scalability_summary(w),
        }

        # Memory behavior (Section 3.2)
        mem = analyze_memory(w)
        results[key]["memory"] = {
            "activation_bytes": mem.activation_bytes,
            "weight_bytes": mem.weight_bytes,
            "gradient_bytes": mem.gradient_bytes,
            "kv_cache_bytes": mem.kv_cache_bytes,
            "bandwidth_pressure": mem.bandwidth_pressure,
        }

    return results


def print_results(results: dict) -> None:
    """Pretty-print workload characterization."""
    for key, r in results.items():
        print(f"\n=== {r['workload']} ===")
        print(f"  Kind: {r['kind']}")
        print(f"  FLOPs/token: {r['flops_per_token']:.2e}")
        print(f"  Bytes/token (R/W): {r['bytes_read_per_token']:.2e} / {r['bytes_written_per_token']:.2e}")
        print(f"  Arithmetic intensity: {r['arithmetic_intensity']:.2f} FLOPs/byte")
        print(f"  Parallelism: {r['parallelism']}")
        print(f"  Scalability: {r['scalability']}")
        print(f"  Memory: {r['memory']}")


if __name__ == "__main__":
    config_dir = Path(__file__).resolve().parents[2] / "config"
    res = run(config_dir)
    print_results(res)
