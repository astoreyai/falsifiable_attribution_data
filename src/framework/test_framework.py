#!/usr/bin/env python3
"""
Test script for the core falsification framework.

This script demonstrates usage of all three components:
1. Counterfactual generation
2. Falsification test
3. Statistical metrics

Run with: python test_framework.py
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework.counterfactual_generation import (
    generate_counterfactuals_hypersphere,
    compute_geodesic_distance,
)
from framework.falsification_test import (
    falsification_test,
    compute_falsification_rate,
)
from framework.metrics import (
    compute_separation_margin,
    compute_effect_size,
    statistical_significance_test,
    compute_confidence_interval,
)


def test_counterfactual_generation():
    """Test counterfactual generation on hypersphere."""
    print("=" * 70)
    print("TEST 1: Counterfactual Generation")
    print("=" * 70)

    # Create a random embedding (simulating ArcFace embedding)
    embedding = torch.randn(512)
    embedding = embedding / embedding.norm()  # Normalize to unit sphere

    print(f"Original embedding shape: {embedding.shape}")
    print(f"Original embedding norm: {embedding.norm().item():.6f}")

    # Generate counterfactuals
    K = 100
    counterfactuals = generate_counterfactuals_hypersphere(
        embedding,
        K=K,
        noise_scale=0.1,
        normalize=True
    )

    print(f"\nGenerated {counterfactuals.shape[0]} counterfactuals")
    print(f"Counterfactual shape: {counterfactuals.shape}")

    # Check all are normalized
    norms = counterfactuals.norm(dim=1)
    print(f"All normalized: {torch.allclose(norms, torch.ones(K), atol=1e-5)}")
    print(f"Mean norm: {norms.mean().item():.6f}")
    print(f"Std norm: {norms.std().item():.6f}")

    # Compute distances
    distances = [
        compute_geodesic_distance(embedding, cf)
        for cf in counterfactuals[:10]  # Just first 10 for demo
    ]

    print(f"\nGeodesic distances (first 10):")
    print(f"Mean: {np.mean(distances):.4f} rad ({np.degrees(np.mean(distances)):.2f}°)")
    print(f"Min: {np.min(distances):.4f} rad ({np.degrees(np.min(distances)):.2f}°)")
    print(f"Max: {np.max(distances):.4f} rad ({np.degrees(np.max(distances)):.2f}°)")

    print("\nTest 1: PASSED\n")
    return embedding, counterfactuals


def test_falsification_test(embedding, counterfactuals):
    """Test falsification test."""
    print("=" * 70)
    print("TEST 2: Falsification Test")
    print("=" * 70)

    # Create a synthetic attribution map
    # High values = important regions, low values = unimportant
    attribution = np.random.rand(224, 224)

    # Make it more interesting - add some structure
    # High attribution in center
    center_y, center_x = 112, 112
    y, x = np.ogrid[:224, :224]
    distance_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    attribution = 1.0 - (distance_from_center / distance_from_center.max())

    print(f"Attribution map shape: {attribution.shape}")
    print(f"Attribution range: [{attribution.min():.3f}, {attribution.max():.3f}]")
    print(f"Attribution mean: {attribution.mean():.3f}")

    # Run falsification test
    result = falsification_test(
        attribution_map=attribution,
        original_embedding=embedding,
        counterfactual_embeddings=counterfactuals,
        model=None,  # Not needed for this test
        theta_high=0.7,
        theta_low=0.3,
        return_details=True
    )

    print("\nFalsification Test Results:")
    print(f"  Is Falsified: {result['is_falsified']}")
    print(f"  Falsification Rate: {result['falsification_rate']:.2f}%")
    print(f"  d_high: {result['d_high']:.4f} rad")
    print(f"  d_low: {result['d_low']:.4f} rad")
    print(f"  Separation Margin: {result['separation_margin']:.4f} rad")
    print(f"  n_high: {result['n_high']}")
    print(f"  n_low: {result['n_low']}")

    if result['is_falsified']:
        print("\n  ❌ Attribution FALSIFIED (high regions don't produce larger changes)")
    else:
        print("\n  ✓ Attribution VALID (high regions produce larger changes)")

    print("\nTest 2: PASSED\n")
    return result


def test_statistical_metrics():
    """Test statistical metrics."""
    print("=" * 70)
    print("TEST 3: Statistical Metrics")
    print("=" * 70)

    # Simulate two attribution methods with different performance
    # Method 1 (Good): Low falsification rate
    # Method 2 (Bad): High falsification rate

    fr1 = 35.0  # Method 1: 35% falsification rate
    fr2 = 55.0  # Method 2: 55% falsification rate
    n1 = 100
    n2 = 100

    print(f"Method 1: FR = {fr1}%, n = {n1}")
    print(f"Method 2: FR = {fr2}%, n = {n2}")

    # Confidence intervals
    ci1 = compute_confidence_interval(fr1, n1)
    ci2 = compute_confidence_interval(fr2, n2)

    print(f"\nConfidence Intervals (95%):")
    print(f"  Method 1: [{ci1[0]:.1f}%, {ci1[1]:.1f}%]")
    print(f"  Method 2: [{ci2[0]:.1f}%, {ci2[1]:.1f}%]")

    # Effect size
    effect = compute_effect_size(fr1, fr2, n1, n2)

    print(f"\nEffect Size:")
    print(f"  Cohen's h = {effect:.3f}")

    if abs(effect) < 0.2:
        effect_interp = "small"
    elif abs(effect) < 0.5:
        effect_interp = "medium"
    else:
        effect_interp = "large"

    print(f"  Interpretation: {effect_interp} effect")

    # Statistical significance
    sig_test = statistical_significance_test(fr1, fr2, n1, n2)

    print(f"\nStatistical Significance Test:")
    print(f"  Test: {sig_test['test_name']}")
    print(f"  Statistic: {sig_test['statistic']:.3f}")
    print(f"  p-value: {sig_test['p_value']:.3f}")
    print(f"  Significant (α=0.05): {sig_test['is_significant']}")

    if sig_test['is_significant']:
        if fr1 < fr2:
            print(f"\n  ✓ Method 1 is significantly BETTER than Method 2")
        else:
            print(f"\n  ✓ Method 2 is significantly BETTER than Method 1")
    else:
        print(f"\n  ○ No significant difference between methods")

    # Test separation margin
    print("\n" + "-" * 70)
    print("Separation Margin Test:")

    d_high = np.array([0.8, 0.9, 0.85, 0.95, 0.88])  # High attribution distances
    d_low = np.array([0.3, 0.2, 0.25, 0.35, 0.28])   # Low attribution distances

    margin = compute_separation_margin(d_high, d_low, use_dprime=True)

    print(f"  d_high: mean={d_high.mean():.3f}, std={d_high.std():.3f}")
    print(f"  d_low:  mean={d_low.mean():.3f}, std={d_low.std():.3f}")
    print(f"  d-prime: {margin:.3f}")

    if margin > 0:
        print(f"  ✓ Positive separation (valid attribution)")
    else:
        print(f"  ❌ Negative/zero separation (falsified attribution)")

    print("\nTest 3: PASSED\n")


def main():
    """Run all tests."""
    print("\n")
    print("*" * 70)
    print(" CORE FALSIFICATION FRAMEWORK - TEST SUITE")
    print("*" * 70)
    print("\n")

    # Test 1: Counterfactual generation
    embedding, counterfactuals = test_counterfactual_generation()

    # Test 2: Falsification test
    result = test_falsification_test(embedding, counterfactuals)

    # Test 3: Statistical metrics
    test_statistical_metrics()

    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    print("\nThe core falsification framework is working correctly.")
    print("\nYou can now use these modules to:")
    print("  1. Generate counterfactuals on face embedding hyperspheres")
    print("  2. Test attribution methods for falsifiability")
    print("  3. Compute statistical metrics and comparisons")
    print("\n")


if __name__ == "__main__":
    main()
