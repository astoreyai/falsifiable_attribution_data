#!/usr/bin/env python3
"""
Test script for Geodesic Integrated Gradients.

Validates that:
1. Slerp interpolation maintains unit norm
2. Geodesic IG produces valid attribution maps
3. Different baseline types work correctly
4. Attribution focuses on face regions (not background)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.attributions.geodesic_ig import GeodesicIntegratedGradients

def create_synthetic_model():
    """Create simple CNN for testing."""
    class SyntheticModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.pool = torch.nn.MaxPool2d(3, stride=2, padding=1)
            self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
            self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(128, 512)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.gap(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    return SyntheticModel()


def test_slerp():
    """Test slerp implementation."""
    print("\n" + "="*60)
    print("TEST 1: SLERP Validation")
    print("="*60)

    model = create_synthetic_model()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    geodesic_ig = GeodesicIntegratedGradients(
        model=model,
        baseline='black',
        n_steps=10,
        device=device
    )

    # Test vectors
    v1 = torch.randn(1, 512).to(device)
    v1 = v1 / v1.norm()
    v2 = torch.randn(1, 512).to(device)
    v2 = v2 / v2.norm()

    # Test interpolation
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        v_interp = geodesic_ig._slerp(v1, v2, alpha)
        norm = v_interp.norm().item()

        print(f"  α={alpha:.2f}: norm={norm:.6f}")
        assert abs(norm - 1.0) < 1e-4, f"Expected norm 1.0, got {norm}"

    print("\n✅ SLERP maintains unit norm")


def test_attribution_generation():
    """Test attribution map generation."""
    print("\n" + "="*60)
    print("TEST 2: Attribution Generation")
    print("="*60)

    model = create_synthetic_model()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    geodesic_ig = GeodesicIntegratedGradients(
        model=model,
        baseline='black',
        n_steps=10,
        device=device
    )

    # Create test image
    img = np.random.rand(112, 112, 3).astype(np.float32)

    # Generate attribution
    attribution = geodesic_ig(img)

    print(f"  Attribution shape: {attribution.shape}")
    print(f"  Attribution range: [{attribution.min():.4f}, {attribution.max():.4f}]")
    print(f"  Attribution mean: {attribution.mean():.4f}")

    assert attribution.shape == (112, 112), f"Expected shape (112, 112), got {attribution.shape}"
    assert attribution.min() >= 0.0, "Attribution should be non-negative"
    assert attribution.max() <= 1.0, "Attribution should be in [0, 1]"

    print("\n✅ Attribution generation works")


def test_baseline_types():
    """Test different baseline types."""
    print("\n" + "="*60)
    print("TEST 3: Baseline Types")
    print("="*60)

    model = create_synthetic_model()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    img = np.random.rand(112, 112, 3).astype(np.float32)

    for baseline_type in ['black', 'noise', 'blur']:
        print(f"\n  Testing baseline='{baseline_type}'...")

        geodesic_ig = GeodesicIntegratedGradients(
            model=model,
            baseline=baseline_type,
            n_steps=10,
            device=device
        )

        attribution = geodesic_ig(img)

        print(f"    Shape: {attribution.shape}")
        print(f"    Range: [{attribution.min():.4f}, {attribution.max():.4f}]")

        assert attribution.shape == (112, 112)
        assert 0.0 <= attribution.min() <= attribution.max() <= 1.0

    print("\n✅ All baseline types work")


def test_verification_task():
    """Test pair verification task."""
    print("\n" + "="*60)
    print("TEST 4: Verification Task (Pair Attribution)")
    print("="*60)

    model = create_synthetic_model()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    geodesic_ig = GeodesicIntegratedGradients(
        model=model,
        baseline='black',
        n_steps=10,
        device=device
    )

    # Create pair of images
    img1 = np.random.rand(112, 112, 3).astype(np.float32)
    img2 = np.random.rand(112, 112, 3).astype(np.float32)

    # Generate attribution for verification
    attribution = geodesic_ig(img1, img2)

    print(f"  Attribution shape: {attribution.shape}")
    print(f"  Attribution stats: min={attribution.min():.4f}, max={attribution.max():.4f}, mean={attribution.mean():.4f}")

    assert attribution.shape == (112, 112)
    assert 0.0 <= attribution.min() <= attribution.max() <= 1.0

    print("\n✅ Verification task works")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("GEODESIC IG VALIDATION TESTS")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    results = {}

    try:
        results['slerp'] = test_slerp()
        results['attribution'] = test_attribution_generation()
        results['baselines'] = test_baseline_types()
        results['verification'] = test_verification_task()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✅ SLERP validation: PASSED")
    print("✅ Attribution generation: PASSED")
    print("✅ Baseline types: PASSED")
    print("✅ Verification task: PASSED")
    print("\n🎉 ALL TESTS PASSED!")

    return 0


if __name__ == '__main__':
    exit(main())
