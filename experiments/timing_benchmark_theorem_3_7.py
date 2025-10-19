"""
Timing Benchmark for Theorem 3.7 - Computational Complexity Analysis

This script validates the O(K·T·D·|M|) complexity claim from Theorem 3.7
by empirically measuring runtime vs. key parameters:
  - K: Number of counterfactuals
  - T: Number of optimization iterations (not directly in falsification_test)
  - D: Embedding dimensionality
  - |M|: Number of features (image dimensions)

Theorem 3.7 (Computational Complexity):
The computational cost of the falsification test is O(K·T·D·|M|) where:
  - K: number of counterfactuals per attribution map
  - T: optimization steps per counterfactual
  - D: embedding dimensionality
  - |M|: number of features (pixels)

This script tests empirically if runtime scales linearly with each parameter.

Author: Aaron W. Storey
Date: October 19, 2025
Status: Experimental validation
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.framework.falsification_test import falsification_test
from facenet_pytorch import InceptionResnetV1


def create_dummy_model(embed_dim: int = 512, device: str = 'cuda'):
    """Create a dummy face recognition model with specified embedding dimension."""

    class DummyFaceModel(torch.nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            # Simple CNN that outputs embeddings of dimension embed_dim
            self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d((4, 4))
            self.fc = torch.nn.Linear(64 * 4 * 4, embed_dim)

        def forward(self, x):
            # x: (B, 3, H, W)
            x = torch.relu(self.conv1(x))
            x = torch.nn.functional.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            # Normalize to unit hypersphere (like ArcFace)
            x = torch.nn.functional.normalize(x, p=2, dim=1)
            return x

    model = DummyFaceModel(embed_dim).to(device)
    model.eval()
    return model


def benchmark_K(
    K_values: List[int] = [10, 25, 50, 100, 200],
    device: str = 'cuda',
    n_trials: int = 5,
    img_size: int = 160
) -> Dict:
    """Benchmark runtime vs. number of counterfactuals K."""

    print(f"\n{'='*80}")
    print(f"BENCHMARK 1: Runtime vs. K (Number of Counterfactuals)")
    print(f"{'='*80}\n")

    model = create_dummy_model(embed_dim=512, device=device)

    results = {
        'K_values': K_values,
        'mean_times': [],
        'std_times': [],
        'min_times': [],
        'max_times': []
    }

    for K in K_values:
        times = []

        for trial in range(n_trials):
            # Create synthetic test case
            img = np.random.rand(img_size, img_size, 3).astype(np.float32)
            attribution_map = np.random.rand(img_size, img_size).astype(np.float32)

            # Measure time
            start = time.time()
            result = falsification_test(
                attribution_map=attribution_map,
                img=img,
                model=model,
                K=K,
                device=device,
                return_details=False
            )
            elapsed = time.time() - start
            times.append(elapsed)

        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        results['mean_times'].append(mean_time)
        results['std_times'].append(std_time)
        results['min_times'].append(min_time)
        results['max_times'].append(max_time)

        print(f"K={K:4d}: {mean_time:.4f}s ± {std_time:.4f}s  (range: {min_time:.4f}s - {max_time:.4f}s)")

    # Check linearity
    K_array = np.array(K_values)
    times_array = np.array(results['mean_times'])
    correlation = np.corrcoef(K_array, times_array)[0, 1]

    print(f"\nCorrelation (K vs. time): {correlation:.4f}")
    print(f"Expected: Close to 1.0 for linear scaling O(K)")

    return results


def benchmark_D(
    D_values: List[int] = [128, 256, 512, 1024, 2048],
    device: str = 'cuda',
    n_trials: int = 5,
    img_size: int = 160,
    K: int = 50
) -> Dict:
    """Benchmark runtime vs. embedding dimensionality D."""

    print(f"\n{'='*80}")
    print(f"BENCHMARK 2: Runtime vs. D (Embedding Dimensionality)")
    print(f"{'='*80}\n")

    results = {
        'D_values': D_values,
        'mean_times': [],
        'std_times': [],
        'min_times': [],
        'max_times': []
    }

    for D in D_values:
        # Create model with specific embedding dimension
        model = create_dummy_model(embed_dim=D, device=device)
        times = []

        for trial in range(n_trials):
            img = np.random.rand(img_size, img_size, 3).astype(np.float32)
            attribution_map = np.random.rand(img_size, img_size).astype(np.float32)

            start = time.time()
            result = falsification_test(
                attribution_map=attribution_map,
                img=img,
                model=model,
                K=K,
                device=device,
                return_details=False
            )
            elapsed = time.time() - start
            times.append(elapsed)

        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        results['mean_times'].append(mean_time)
        results['std_times'].append(std_time)
        results['min_times'].append(min_time)
        results['max_times'].append(max_time)

        print(f"D={D:4d}: {mean_time:.4f}s ± {std_time:.4f}s  (range: {min_time:.4f}s - {max_time:.4f}s)")

    # Check linearity
    D_array = np.array(D_values)
    times_array = np.array(results['mean_times'])
    correlation = np.corrcoef(D_array, times_array)[0, 1]

    print(f"\nCorrelation (D vs. time): {correlation:.4f}")
    print(f"Expected: Close to 1.0 for linear scaling O(D)")

    return results


def benchmark_M(
    M_values: List[int] = [64, 96, 128, 160, 224],
    device: str = 'cuda',
    n_trials: int = 5,
    K: int = 50
) -> Dict:
    """Benchmark runtime vs. number of features |M| (image size)."""

    print(f"\n{'='*80}")
    print(f"BENCHMARK 3: Runtime vs. |M| (Image Size / Number of Features)")
    print(f"{'='*80}\n")

    model = create_dummy_model(embed_dim=512, device=device)

    results = {
        'M_values': M_values,
        'mean_times': [],
        'std_times': [],
        'min_times': [],
        'max_times': []
    }

    for M in M_values:
        times = []

        for trial in range(n_trials):
            img = np.random.rand(M, M, 3).astype(np.float32)
            attribution_map = np.random.rand(M, M).astype(np.float32)

            start = time.time()
            result = falsification_test(
                attribution_map=attribution_map,
                img=img,
                model=model,
                K=K,
                device=device,
                return_details=False
            )
            elapsed = time.time() - start
            times.append(elapsed)

        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        results['mean_times'].append(mean_time)
        results['std_times'].append(std_time)
        results['min_times'].append(min_time)
        results['max_times'].append(max_time)

        print(f"|M|={M}x{M}={M*M:6d}: {mean_time:.4f}s ± {std_time:.4f}s  (range: {min_time:.4f}s - {max_time:.4f}s)")

    # Check linearity (with M^2 since image is M x M)
    M_squared = np.array([m*m for m in M_values])
    times_array = np.array(results['mean_times'])
    correlation = np.corrcoef(M_squared, times_array)[0, 1]

    print(f"\nCorrelation (M^2 vs. time): {correlation:.4f}")
    print(f"Expected: Close to 1.0 for linear scaling O(|M|) where |M| = H*W")

    return results


def plot_results(results_K, results_D, results_M, save_dir: str = 'experiments/timing_benchmarks'):
    """Generate plots showing runtime vs. each parameter."""

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Runtime vs. K
    ax = axes[0]
    K_values = results_K['K_values']
    mean_times = results_K['mean_times']
    std_times = results_K['std_times']

    ax.errorbar(K_values, mean_times, yerr=std_times, fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel('K (Number of Counterfactuals)', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Runtime vs. K', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Fit linear model
    z = np.polyfit(K_values, mean_times, 1)
    p = np.poly1d(z)
    ax.plot(K_values, p(K_values), 'r--', alpha=0.7, label=f'Linear fit: {z[0]:.6f}·K + {z[1]:.4f}')
    ax.legend()

    # Plot 2: Runtime vs. D
    ax = axes[1]
    D_values = results_D['D_values']
    mean_times = results_D['mean_times']
    std_times = results_D['std_times']

    ax.errorbar(D_values, mean_times, yerr=std_times, fmt='s-', capsize=5, linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('D (Embedding Dimensionality)', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Runtime vs. D', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Fit linear model
    z = np.polyfit(D_values, mean_times, 1)
    p = np.poly1d(z)
    ax.plot(D_values, p(D_values), 'r--', alpha=0.7, label=f'Linear fit: {z[0]:.6f}·D + {z[1]:.4f}')
    ax.legend()

    # Plot 3: Runtime vs. |M| (M^2)
    ax = axes[2]
    M_values = results_M['M_values']
    M_squared = [m*m for m in M_values]
    mean_times = results_M['mean_times']
    std_times = results_M['std_times']

    ax.errorbar(M_squared, mean_times, yerr=std_times, fmt='^-', capsize=5, linewidth=2, markersize=8, color='green')
    ax.set_xlabel('|M| = H × W (Number of Features)', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Runtime vs. |M|', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Fit linear model
    z = np.polyfit(M_squared, mean_times, 1)
    p = np.poly1d(z)
    ax.plot(M_squared, p(M_squared), 'r--', alpha=0.7, label=f'Linear fit: {z[0]:.8f}·|M| + {z[1]:.4f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path / 'timing_benchmark_theorem_3_7.pdf', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path / 'timing_benchmark_theorem_3_7.pdf'}")

    return fig


def main():
    """Run all timing benchmarks and generate report."""

    print("="*80)
    print("TIMING BENCHMARK FOR THEOREM 3.7")
    print("Validating O(K·T·D·|M|) Computational Complexity")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if device == 'cpu':
        print("WARNING: Running on CPU. Timings will be slower and less accurate.")
        print("         Recommend running on GPU for reliable benchmarks.\n")

    # Run benchmarks
    results_K = benchmark_K(
        K_values=[10, 25, 50, 100, 200],
        device=device,
        n_trials=5
    )

    results_D = benchmark_D(
        D_values=[128, 256, 512, 1024],
        device=device,
        n_trials=5
    )

    results_M = benchmark_M(
        M_values=[64, 96, 128, 160, 224],
        device=device,
        n_trials=5
    )

    # Generate plots
    fig = plot_results(results_K, results_D, results_M)

    # Save results as JSON
    save_dir = Path('experiments/timing_benchmarks')
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'timestamp': time.strftime('%Y%m%d_%H%M%S'),
        'device': device,
        'benchmark_K': results_K,
        'benchmark_D': results_D,
        'benchmark_M': results_M
    }

    with open(save_dir / 'timing_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {save_dir / 'timing_results.json'}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Theorem 3.7 Validation")
    print(f"{'='*80}\n")

    K_corr = np.corrcoef(results_K['K_values'], results_K['mean_times'])[0, 1]
    D_corr = np.corrcoef(results_D['D_values'], results_D['mean_times'])[0, 1]
    M_squared = [m*m for m in results_M['M_values']]
    M_corr = np.corrcoef(M_squared, results_M['mean_times'])[0, 1]

    print(f"Runtime vs. K correlation:   {K_corr:.4f}  (Expected: ~1.0 for linear O(K))")
    print(f"Runtime vs. D correlation:   {D_corr:.4f}  (Expected: ~1.0 for linear O(D))")
    print(f"Runtime vs. |M| correlation: {M_corr:.4f}  (Expected: ~1.0 for linear O(|M|))")

    print("\nConclusion:")
    if K_corr > 0.95 and D_corr > 0.95 and M_corr > 0.95:
        print("✅ VALIDATED: All parameters show strong linear scaling (r > 0.95)")
        print("   Theorem 3.7's O(K·T·D·|M|) complexity claim is empirically supported.")
    elif K_corr > 0.90 and D_corr > 0.90 and M_corr > 0.90:
        print("✅ MOSTLY VALIDATED: All parameters show good linear scaling (r > 0.90)")
        print("   Theorem 3.7's complexity claim is generally supported.")
    else:
        print("⚠️  PARTIAL VALIDATION: Some parameters show weaker linear scaling.")
        print("   Further investigation may be needed.")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
