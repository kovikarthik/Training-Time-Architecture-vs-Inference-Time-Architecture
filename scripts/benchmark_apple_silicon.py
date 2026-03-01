#!/usr/bin/env python3
"""
Bonus: Real hardware measurement on Apple Silicon (M4 Pro, Metal 4).
Measures small GEMM / matrix operations to demonstrate actual throughput.
Requires: pip install torch
"""

import sys
from pathlib import Path

def check_pytorch_mps() -> bool:
    """Check if PyTorch with MPS is available."""
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        return False


def run_benchmark() -> dict:
    """Run simple matrix multiply benchmark on MPS (Apple Silicon GPU)."""
    import torch
    import time

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    results = {"device": str(device), "runs": []}

    # Matrix sizes: representative of transformer blocks (e.g. Llama 3 8B has hidden size 4096)
    for n in [1024, 2048, 4096, 8192]:  # Added 8192 for FFN size proxy
        a = torch.randn(n, n, dtype=torch.float32, device=device)
        b = torch.randn(n, n, dtype=torch.float32, device=device)

        # Warmup
        for _ in range(3):
            c = torch.mm(a, b)
        if device.type == "mps":
            torch.mps.synchronize()

        # Timed runs
        n_iters = 20
        t0 = time.perf_counter()
        for _ in range(n_iters):
            c = torch.mm(a, b)
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0

        flops_per_matmul = 2 * n * n * n
        total_flops = flops_per_matmul * n_iters
        tflops = (total_flops / elapsed) / 1e12
        results["runs"].append({
            "n": n,
            "flops_per_matmul": flops_per_matmul,
            "elapsed_s": elapsed,
            "tflops": tflops,
        })

    return results


def main() -> int:
    if not check_pytorch_mps():
        print("PyTorch with MPS not available. Install: pip install torch")
        print("Falling back to CPU benchmark...")
    else:
        print("Apple Silicon MPS detected. Running GPU benchmark.")

    try:
        res = run_benchmark()
        print("\n=== Apple Silicon Benchmark Results ===")
        print(f"Device: {res['device']}")
        for r in res["runs"]:
            print(f"  N={r['n']}: {r['tflops']:.2f} TFLOP/s")
        return 0
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
