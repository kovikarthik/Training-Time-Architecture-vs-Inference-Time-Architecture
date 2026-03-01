#!/usr/bin/env python3
"""
Main experiment runner - CECS 530 Project 8.
Runs all configured experiments with clear parameterization and reproducible results.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.workload import load_workload_from_config
from src.architecture import load_architecture_from_config, roofline_throughput
from src.analysis.workload_characterization import run as run_workload_char
from src.analysis.architecture_evaluation import run as run_arch_eval


def load_experiment_config(config_path: Path) -> dict:
    """Load experiments.yaml."""
    with open(config_path) as f:
        import yaml
        return yaml.safe_load(f)


def run_roofline_comparison(config_dir: Path, workloads: list, architectures: list) -> list[dict]:
    """Side-by-side roofline comparison."""
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
                "workload_kind": w.kind,
                "arch_kind": a.kind,
                "arithmetic_intensity": ev.arithmetic_intensity,
                "compute_bound_tokens_per_s": ev.roofline_flops_bound_toks_per_s,
                "memory_bound_tokens_per_s": ev.roofline_mem_bound_toks_per_s,
                "bottleneck": ev.bottleneck,
                "effective_throughput_tokens_per_s": ev.effective_throughput_toks_per_s,
                "energy_per_token_mj": ev.energy_per_token_mj,
                "edp_uj_s": ev.edp_uj_s,
                "cost_per_million_tokens_usd": ev.cost_per_million_tokens_usd,
            })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="CECS 530 Project 8 - Experiment Runner")
    parser.add_argument("--config", type=Path, default=Path("config/experiments.yaml"))
    parser.add_argument("--output", "-o", type=Path, default=Path("results"))
    parser.add_argument("--stdout", action="store_true", help="Print results to stdout")
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[1] / config_path
    config_dir = config_path.parent

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    exp_config = load_experiment_config(config_path)
    workloads = exp_config.get("experiments", [{}])[0].get("workloads", ["training", "inference"])
    architectures = exp_config.get("experiments", [{}])[0].get("architectures", [
        "training_optimized", "inference_optimized", "unified"
    ])

    # Ensure workload/arch lists from first experiment
    for ex in exp_config.get("experiments", []):
        if ex.get("name") == "roofline_comparison":
            workloads = ex.get("workloads", workloads)
            architectures = ex.get("architectures", architectures)
            break

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "workload_characterization": run_workload_char(config_dir),
        "architecture_evaluation": run_arch_eval(config_dir, workloads, architectures),
        "roofline_comparison": run_roofline_comparison(config_dir, workloads, architectures),
    }

    # Output (resolve relative to project root)
    output_dir = args.output
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[1] / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / f"results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    if args.stdout:
        print(json.dumps(all_results, indent=2))
    else:
        print(f"Results written to {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
