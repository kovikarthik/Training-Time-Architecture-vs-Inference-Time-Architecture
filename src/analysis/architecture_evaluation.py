"""
Goal 2: Architecture Evaluation
- Evaluate training- vs. inference-optimized architectures against both workloads
- Explain where each excels and fails
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.workload import load_workload_from_config
from src.architecture import load_architecture_from_config, roofline_throughput


def run(config_dir: Path, workloads: list[str], architectures: list[str]) -> list[dict]:
    """Run architecture evaluation for all workload x architecture pairs."""
    config_dir = Path(config_dir)
    w_config = config_dir / "workloads.yaml"
    a_config = config_dir / "architectures.yaml"

    results = []

    for w_key in workloads:
        w = load_workload_from_config(w_config, w_key)
        for a_key in architectures:
            a = load_architecture_from_config(a_config, a_key)
            ev = roofline_throughput(w, a)
            results.append({
                "workload": w.name,
                "architecture": a.name,
                "kind": w.kind,
                "arithmetic_intensity": ev.arithmetic_intensity,
                "compute_bound_toks_s": ev.roofline_flops_bound_toks_per_s,
                "mem_bound_toks_s": ev.roofline_mem_bound_toks_per_s,
                "bottleneck": ev.bottleneck,
                "effective_throughput": ev.effective_throughput_toks_per_s,
            })

    return results


def print_results(results: list[dict]) -> None:
    """Pretty-print architecture evaluation."""
    for r in results:
        print(f"\n=== {r['workload']} on {r['architecture']} ===")
        print(f"  Workload type: {r['kind']}")
        print(f"  Arithmetic intensity: {r['arithmetic_intensity']:.2f} FLOPs/byte")
        print(f"  Compute-bound throughput: {r['compute_bound_toks_s']:.2f} tokens/s")
        print(f"  Memory-bound throughput:  {r['mem_bound_toks_s']:.2f} tokens/s")
        print(f"  Bottleneck: {r['bottleneck']}")
        print(f"  Effective throughput: {r['effective_throughput']:.2f} tokens/s")


if __name__ == "__main__":
    config_dir = Path(__file__).resolve().parents[2] / "config"
    res = run(
        config_dir,
        workloads=["training", "inference"],
        architectures=["training_optimized", "inference_optimized", "unified"],
    )
    print_results(res)
