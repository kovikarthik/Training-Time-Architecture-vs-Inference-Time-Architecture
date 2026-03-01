#!/usr/bin/env python3
"""
Visualization module: generates roofline plots and comparison bar charts.
Produces publication-quality figures saved as PNG files in the results directory.
Reads from the most recently generated results JSON file.
"""

import json
import glob
from pathlib import Path

def get_latest_results_file() -> Path:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    json_files = sorted(results_dir.glob("results_*.json"))
    if not json_files:
        raise FileNotFoundError("No results JSON files found.")
    return json_files[-1]


def plot_roofline(results_data: dict):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n  [matplotlib not installed — skipping roofline plot]")
        print("  Install with: pip install matplotlib numpy")
        return

    # Extract architectures from architecture_evaluation section
    # And we'll plot the data points from roofline_comparison
    # However, to draw actual rooflines we need peak TFLOPS and mem bandwidth.
    # We can infer them from the JSON or load config directly. 
    # For simplicity, we load config:
    import yaml
    config_path = Path(__file__).resolve().parents[1] / "config" / "architectures.yaml"
    with open(config_path) as f:
        arch_config = yaml.safe_load(f)
        
    architectures = []
    for key, arch in arch_config.items():
        architectures.append({
            "name": arch["name"],
            "peak": arch["peak_flops_tflops"],
            "bw": arch["mem_bandwidth_gbs"] / 1000.0 # Convert to TB/s
        })

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ai_range = np.logspace(-1, 4, 500)
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    markers = {"training": "^", "inference": "o"}

    for i, arch in enumerate(architectures):
        peak = arch["peak"]
        bw = arch["bw"]
        roof = np.minimum(peak, ai_range * bw)
        ax.plot(ai_range, roof, color=colors[i % len(colors)], linewidth=2, label=f"{arch['name']} Roofline")
        ax.axhline(y=peak, color=colors[i % len(colors)], linestyle=":", alpha=0.3)

    marker_colors = {"training": "#1B5E20", "inference": "#B71C1C"}
    
    # Plot evaluations
    roofline_evals = results_data.get("roofline_comparison", [])
    for r in roofline_evals:
        mode = r["workload_kind"]
        ai = r["arithmetic_intensity"]
        achievable = r["effective_throughput_tokens_per_s"] 
        # Actually achievable TFLOPS = achievable tokens * flops_per_token / 1e12
        # But our `project8_analysis.py` model computes `effective_throughput_tokens_per_s`
        # We need to turn this back into TFLOPS for the roofline chart.
        # Compute bound tokens = peak_flops / flops_per_token -> flops_per_token = peak_flops / compute_bound_tokens
        
        # We can find the architecture peak flops to reverse engineer
        arch_name = r["architecture"]
        peak = next((a["peak"] for a in architectures if a["name"] == arch_name), 1000)
        flops_per_token_tflops = peak / r["compute_bound_tokens_per_s"]
        achievable_tflops = r["effective_throughput_tokens_per_s"] * flops_per_token_tflops
        
        ax.scatter(
            ai, achievable_tflops,
            marker=markers.get(mode, "s"), s=150,
            color=marker_colors.get(mode, "gray"),
            edgecolors="black", linewidth=1.5, zorder=5,
        )
        
        ax.annotate(
            f"{arch_name}\n({mode.title()})",
            (ai, achievable_tflops),
            textcoords="offset points", xytext=(10, 10), fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (ops/byte)", fontsize=13)
    ax.set_ylabel("Achievable Performance (TFLOPS)", fontsize=13)
    ax.set_title("Roofline Analysis: Training vs. Inference Architectures", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.1, 10000)
    ax.set_ylim(0.1, 2000)
    plt.tight_layout()
    
    out_path = Path(__file__).resolve().parents[1] / "results" / "roofline_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Roofline plot saved to: {out_path}")
    plt.close()


def plot_comparison_bars(results_data: dict):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    inference_results = [r for r in results_data.get("roofline_comparison", []) if r["workload_kind"] == "inference"]
    if len(inference_results) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    names = [r["architecture"] for r in inference_results]
    x = np.arange(len(names))
    
    latencies = [1000.0 / r["effective_throughput_tokens_per_s"] for r in inference_results]
    energies = [r["energy_per_token_mj"] for r in inference_results]
    throughputs = [r["effective_throughput_tokens_per_s"] for r in inference_results]
    bar_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for ax, data, ylabel, title, fmt in [
        (axes[0], latencies, "Latency per Token (ms)", "Token Latency", ".2f"),
        (axes[1], energies, "Energy per Token (mJ)", "Energy Efficiency", ".1f"),
        (axes[2], throughputs, "Tokens / Second", "Throughput", ".0f"),
    ]:
        ax.bar(x, data, color=bar_colors[:len(names)], edgecolor="black", linewidth=0.8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        for i, v in enumerate(data):
            # dynamic offset for text
            ax.text(i, v + max(data) * 0.02, f"{v:{fmt}}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Inference Performance Comparison (Llama 3 8B)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    out_path = Path(__file__).resolve().parents[1] / "results" / "inference_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Bar chart saved to: {out_path}")
    plt.close()

def main():
    latest_json = get_latest_results_file()
    print(f"Reading data from {latest_json.name}")
    with open(latest_json) as f:
        results_data = json.load(f)
        
    plot_roofline(results_data)
    plot_comparison_bars(results_data)

if __name__ == "__main__":
    main()
