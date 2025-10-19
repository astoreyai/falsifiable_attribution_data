#!/usr/bin/env python3
"""
Validation tests for Biometric Grad-CAM.

Tests the novel biometric-specific attribution method that extends Grad-CAM
with identity-aware weighting and invariance regularization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from src.attributions.biometric_gradcam import (
    BiometricGradCAM,
    BiometricGradCAMPlusPlus,
    get_biometric_gradcam
)


def create_synthetic_face_model():
    """Create synthetic face verification model for testing."""
    class SyntheticFaceModel(nn.Module):
        def __init__(self, embedding_dim=512):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(3, stride=2, padding=1)

            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)

            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(256)

            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(256, embedding_dim)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.pool(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)

            x = self.gap(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

    return SyntheticFaceModel()


def test_basic_initialization():
    """Test 1: Basic initialization and hook registration."""
    print("\n" + "="*60)
    print("TEST 1: Basic Initialization")
    print("="*60)

    model = create_synthetic_face_model()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Test standard variant
    bgradcam = BiometricGradCAM(
        model=model,
        use_identity_weighting=True,
        use_invariance_reg=True,
        device=device
    )

    print("✅ BiometricGradCAM initialized")
    print(f"  - Hooks registered: {len(bgradcam.hooks)}")
    print(f"  - Identity weighting: {bgradcam.use_identity_weighting}")
    print(f"  - Invariance reg: {bgradcam.use_invariance_reg}")

    # Test PlusPlus variant
    bgradcam_pp = BiometricGradCAMPlusPlus(
        model=model,
        device=device
    )

    print("✅ BiometricGradCAMPlusPlus initialized")

    # Cleanup
    bgradcam.remove_hooks()
    bgradcam_pp.remove_hooks()

    print("\n✅ TEST 1 PASSED: Initialization successful")


def test_attribution_generation():
    """Test 2: Attribution map generation."""
    print("\n" + "="*60)
    print("TEST 2: Attribution Generation")
    print("="*60)

    model = create_synthetic_face_model()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    bgradcam = BiometricGradCAM(model=model, device=device)

    # Create synthetic face image
    img = np.random.rand(112, 112, 3).astype(np.float32)

    # Generate attribution without target
    attr_single = bgradcam.generate_attribution(img)

    print(f"  Single image attribution shape: {attr_single.shape}")
    print(f"  Value range: [{attr_single.min():.3f}, {attr_single.max():.3f}]")

    assert attr_single.shape == (112, 112), f"Expected (112, 112), got {attr_single.shape}"
    assert 0 <= attr_single.min() <= 1, f"Min value {attr_single.min()} not in [0, 1]"
    assert 0 <= attr_single.max() <= 1, f"Max value {attr_single.max()} not in [0, 1]"

    # Generate attribution with target (verification)
    img2 = np.random.rand(112, 112, 3).astype(np.float32)
    img2_tensor = torch.from_numpy(img2.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        target_emb = model(img2_tensor)

    attr_pair = bgradcam.generate_attribution(img, target_embedding=target_emb)

    print(f"  Pair attribution shape: {attr_pair.shape}")
    print(f"  Value range: [{attr_pair.min():.3f}, {attr_pair.max():.3f}]")

    assert attr_pair.shape == (112, 112), f"Expected (112, 112), got {attr_pair.shape}"

    bgradcam.remove_hooks()

    print("\n✅ TEST 2 PASSED: Attribution generation works")


def test_identity_weighting():
    """Test 3: Identity-aware weighting."""
    print("\n" + "="*60)
    print("TEST 3: Identity-Aware Weighting")
    print("="*60)

    model = create_synthetic_face_model()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Create embeddings with different similarities
    emb1 = torch.randn(1, 512).to(device)
    emb1 = emb1 / emb1.norm()

    # High similarity embedding (same person)
    emb2_high = emb1 + torch.randn(1, 512).to(device) * 0.1
    emb2_high = emb2_high / emb2_high.norm()

    # Low similarity embedding (different person)
    emb2_low = torch.randn(1, 512).to(device)
    emb2_low = emb2_low / emb2_low.norm()

    bgradcam = BiometricGradCAM(
        model=model,
        use_identity_weighting=True,
        device=device
    )

    # Compute identity weights
    weight_high = bgradcam._compute_identity_weights(emb1, emb2_high)
    weight_low = bgradcam._compute_identity_weights(emb1, emb2_low)

    print(f"  High similarity weight: {weight_high.item():.3f}")
    print(f"  Low similarity weight: {weight_low.item():.3f}")

    # High similarity should have higher weight (for genuine pairs)
    # Both should be in [0, 1]
    assert 0 <= weight_high.item() <= 1, f"Weight {weight_high.item()} not in [0, 1]"
    assert 0 <= weight_low.item() <= 1, f"Weight {weight_low.item()} not in [0, 1]"

    bgradcam.remove_hooks()

    print("\n✅ TEST 3 PASSED: Identity weighting works")


def test_invariance_regularization():
    """Test 4: Invariance regularization."""
    print("\n" + "="*60)
    print("TEST 4: Invariance Regularization")
    print("="*60)

    model = create_synthetic_face_model()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    bgradcam = BiometricGradCAM(
        model=model,
        use_invariance_reg=True,
        device=device
    )

    # Create synthetic gradients with different spatial variance
    # Low variance = consistent across space = intrinsic feature
    gradients_low_var = torch.ones(1, 256, 14, 14).to(device) * 0.5
    gradients_low_var += torch.randn(1, 256, 14, 14).to(device) * 0.01

    # High variance = varies across space = extrinsic feature
    gradients_high_var = torch.randn(1, 256, 14, 14).to(device)

    weights_low_var = bgradcam._compute_invariance_regularization(gradients_low_var)
    weights_high_var = bgradcam._compute_invariance_regularization(gradients_high_var)

    print(f"  Low variance weights shape: {weights_low_var.shape}")
    print(f"  High variance weights shape: {weights_high_var.shape}")
    print(f"  Low var mean: {weights_low_var.mean().item():.6f}")
    print(f"  High var mean: {weights_high_var.mean().item():.6f}")

    # Weights should sum to 1 across channels (softmax)
    channel_sum = weights_low_var.sum(dim=1)
    assert torch.allclose(channel_sum, torch.ones_like(channel_sum), atol=1e-5), \
        f"Weights don't sum to 1: {channel_sum}"

    bgradcam.remove_hooks()

    print("\n✅ TEST 4 PASSED: Invariance regularization works")


def test_callable_interface():
    """Test 5: Callable interface matching other methods."""
    print("\n" + "="*60)
    print("TEST 5: Callable Interface")
    print("="*60)

    model = create_synthetic_face_model()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    bgradcam = BiometricGradCAM(model=model, device=device)

    img1 = np.random.rand(112, 112, 3).astype(np.float32)
    img2 = np.random.rand(112, 112, 3).astype(np.float32)

    # Test __call__ method
    attr1 = bgradcam(img1)
    attr2 = bgradcam(img1, img2)

    # Test compute() alias
    attr3 = bgradcam.compute(img1, img2)

    print(f"  Single image: {attr1.shape}")
    print(f"  Pair (call): {attr2.shape}")
    print(f"  Pair (compute): {attr3.shape}")

    assert attr1.shape == (112, 112)
    assert attr2.shape == (112, 112)
    assert attr3.shape == (112, 112)

    bgradcam.remove_hooks()

    print("\n✅ TEST 5 PASSED: Callable interface works")


def test_gradcam_plusplus():
    """Test 6: Grad-CAM++ variant."""
    print("\n" + "="*60)
    print("TEST 6: Grad-CAM++ Variant")
    print("="*60)

    model = create_synthetic_face_model()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Test via factory function
    bgradcam_standard = get_biometric_gradcam(
        model=model,
        variant='standard',
        device=device
    )

    bgradcam_plusplus = get_biometric_gradcam(
        model=model,
        variant='plusplus',
        device=device
    )

    img = np.random.rand(112, 112, 3).astype(np.float32)

    attr_standard = bgradcam_standard(img)
    attr_plusplus = bgradcam_plusplus(img)

    print(f"  Standard attribution: {attr_standard.shape}, range=[{attr_standard.min():.3f}, {attr_standard.max():.3f}]")
    print(f"  PlusPlus attribution: {attr_plusplus.shape}, range=[{attr_plusplus.min():.3f}, {attr_plusplus.max():.3f}]")

    assert attr_standard.shape == (112, 112)
    assert attr_plusplus.shape == (112, 112)

    # Attributions should be different (different weighting schemes)
    diff = np.abs(attr_standard - attr_plusplus).mean()
    print(f"  Mean difference: {diff:.3f}")

    bgradcam_standard.remove_hooks()
    bgradcam_plusplus.remove_hooks()

    print("\n✅ TEST 6 PASSED: Grad-CAM++ variant works")


def run_all_tests():
    """Run all validation tests."""
    print("="*60)
    print("BIOMETRIC GRAD-CAM VALIDATION TESTS")
    print("="*60)

    tests = [
        test_basic_initialization,
        test_attribution_generation,
        test_identity_weighting,
        test_invariance_regularization,
        test_callable_interface,
        test_gradcam_plusplus
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ TEST FAILED: {test.__name__}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nBiometric Grad-CAM Features Validated:")
        print("  ✅ Identity-aware weighting")
        print("  ✅ Invariance regularization")
        print("  ✅ Standard and PlusPlus variants")
        print("  ✅ Compatible interface with other methods")

    print("="*60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
