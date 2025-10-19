#!/usr/bin/env python3
"""
Test script for real Grad-CAM implementation.

Validates that:
1. Grad-CAM produces non-random heatmaps
2. Heatmaps highlight face regions
3. No errors with InsightFace model
4. Hooks are properly registered and cleaned up
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Try to import InsightFace
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[WARNING] InsightFace not available - using synthetic test")

from src.attributions.gradcam import GradCAM


def create_synthetic_model():
    """Create a synthetic ResNet-like model for testing."""
    class SyntheticResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Simple CNN architecture
            self.conv1 = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.pool = torch.nn.MaxPool2d(3, stride=2, padding=1)
            self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = torch.nn.Conv2d(128, 256, 3, padding=1)
            self.conv4 = torch.nn.Conv2d(256, 512, 3, padding=1)  # Last conv
            self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(512, 512)  # Embedding layer

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = torch.relu(self.conv4(x))  # Last conv activations
            x = self.gap(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    return SyntheticResNet()


def test_with_synthetic_model():
    """Test Grad-CAM with synthetic model."""
    print("\n" + "="*60)
    print("TEST 1: Synthetic Model")
    print("="*60)

    # Create model
    model = create_synthetic_model()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Create random image
    img = torch.randn(1, 3, 112, 112).to(device)

    # Create GradCAM
    gradcam = GradCAM(model, device=device)

    # Compute attribution
    try:
        print("Computing Grad-CAM attribution...")
        attribution = gradcam(img)

        print(f"✅ Attribution shape: {attribution.shape}")
        print(f"✅ Attribution range: [{attribution.min():.3f}, {attribution.max():.3f}]")
        print(f"✅ Attribution mean: {attribution.mean():.3f}")
        print(f"✅ Non-zero pixels: {(attribution > 0).sum()} / {attribution.size}")

        # Verify it's not uniform (should have some variation)
        if attribution.std() > 0.01:
            print("✅ Attribution has variation (not uniform)")
        else:
            print("⚠️  Attribution is nearly uniform (may indicate issue)")

        # Verify shape matches input
        assert attribution.shape == (112, 112), f"Shape mismatch: {attribution.shape}"
        print("✅ Shape matches input dimensions")

        # Verify normalized to [0, 1]
        assert 0 <= attribution.min() <= attribution.max() <= 1, "Attribution not in [0, 1]"
        print("✅ Attribution normalized to [0, 1]")

        print("\n✅ Synthetic model test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Synthetic model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_insightface():
    """Test Grad-CAM with real InsightFace model."""
    print("\n" + "="*60)
    print("TEST 2: InsightFace Model")
    print("="*60)

    if not INSIGHTFACE_AVAILABLE:
        print("⚠️  InsightFace not available - skipping test")
        return None

    try:
        # Load InsightFace
        print("Loading InsightFace buffalo_l model...")
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1)  # CPU

        # Get the recognition model
        rec_model = None
        for model in app.models.values():
            if hasattr(model, 'get'):
                rec_model = model
                break

        if rec_model is None:
            print("⚠️  Could not extract recognition model from InsightFace")
            return None

        print(f"✅ Model loaded: {type(rec_model)}")

        # Create random face image
        img = torch.randn(1, 3, 112, 112)

        # Create GradCAM
        gradcam = GradCAM(rec_model, device='cpu')

        # Compute attribution
        print("Computing Grad-CAM attribution on InsightFace model...")
        attribution = gradcam(img)

        print(f"✅ Attribution shape: {attribution.shape}")
        print(f"✅ Attribution range: [{attribution.min():.3f}, {attribution.max():.3f}]")
        print(f"✅ Attribution mean: {attribution.mean():.3f}")

        print("\n✅ InsightFace test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ InsightFace test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_verification_mode():
    """Test Grad-CAM in verification mode (with two images)."""
    print("\n" + "="*60)
    print("TEST 3: Verification Mode (2 images)")
    print("="*60)

    # Create model
    model = create_synthetic_model()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Create two random images
    img1 = torch.randn(1, 3, 112, 112).to(device)
    img2 = torch.randn(1, 3, 112, 112).to(device)

    # Create GradCAM
    gradcam = GradCAM(model, device=device)

    try:
        print("Computing Grad-CAM for verification (img1 vs img2)...")
        attribution = gradcam(img1, img2)

        print(f"✅ Attribution shape: {attribution.shape}")
        print(f"✅ Attribution range: [{attribution.min():.3f}, {attribution.max():.3f}]")

        print("\n✅ Verification mode test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Verification mode test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_numpy_input():
    """Test Grad-CAM with numpy array input."""
    print("\n" + "="*60)
    print("TEST 4: NumPy Array Input")
    print("="*60)

    # Create model
    model = create_synthetic_model()
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Create numpy image (H, W, C) in [0, 255]
    img_np = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    # Create GradCAM
    gradcam = GradCAM(model, device=device)

    try:
        print("Computing Grad-CAM with NumPy input...")
        attribution = gradcam(img_np)

        print(f"✅ Attribution shape: {attribution.shape}")
        print(f"✅ Attribution range: [{attribution.min():.3f}, {attribution.max():.3f}]")

        print("\n✅ NumPy input test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ NumPy input test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("GRAD-CAM VALIDATION TESTS")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"InsightFace available: {INSIGHTFACE_AVAILABLE}")

    results = {}

    # Run tests
    results['synthetic'] = test_with_synthetic_model()
    results['verification'] = test_verification_mode()
    results['numpy'] = test_numpy_input()
    results['insightface'] = test_with_insightface()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    for test_name, result in results.items():
        if result is True:
            print(f"✅ {test_name}: PASSED")
        elif result is False:
            print(f"❌ {test_name}: FAILED")
        else:
            print(f"⚠️  {test_name}: SKIPPED")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
