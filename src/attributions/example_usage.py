"""
Example usage of attribution methods for face verification.

This script demonstrates how to use Grad-CAM, SHAP, and LIME to explain
face verification model decisions.
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image

from gradcam import GradCAM, get_gradcam_for_model
from shap_wrapper import SHAPAttribution, create_background_samples
from lime_wrapper import LIMEAttribution


def load_example_model():
    """
    Load a pre-trained ResNet-50 as a face verification model.

    In practice, this would be a model fine-tuned for face verification.
    """
    model = models.resnet50(pretrained=True)

    # Remove classification head, keep embedding
    model = nn.Sequential(*list(model.children())[:-1])

    # Add flatten layer
    model.add_module('flatten', nn.Flatten())

    model.eval()
    return model


def load_and_preprocess_image(image_path: str) -> tuple:
    """
    Load and preprocess an image for the model.

    Args:
        image_path: Path to image file

    Returns:
        image_tensor: Preprocessed tensor (1, 3, 224, 224)
        image_np: Original numpy array (224, 224, 3) in [0, 1]
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))

    # Convert to numpy
    image_np = np.array(image).astype(np.float32) / 255.0

    # Preprocessing for model
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = preprocess(image).unsqueeze(0)

    return image_tensor, image_np


def example_gradcam():
    """Example: Generate Grad-CAM attribution."""
    print("=" * 60)
    print("Example 1: Grad-CAM Attribution")
    print("=" * 60)

    # Load model
    model = load_example_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Create Grad-CAM explainer
    gradcam = get_gradcam_for_model(model, model_arch='resnet50')

    # Example: explain a single face image
    print("\n1. Explaining face embedding (no target)")

    # Dummy image
    image = torch.randn(1, 3, 224, 224).to(device)

    # Generate attribution
    attribution = gradcam.generate_attribution(image)

    print(f"   Attribution shape: {attribution.shape}")
    print(f"   Attribution range: [{attribution.min():.3f}, {attribution.max():.3f}]")

    # Example: explain verification decision
    print("\n2. Explaining face verification (with target)")

    # Generate target embedding
    with torch.no_grad():
        target_embedding = model(torch.randn(1, 3, 224, 224).to(device))

    attribution = gradcam.generate_attribution(image, target_embedding=target_embedding)

    print(f"   Attribution shape: {attribution.shape}")
    print(f"   Attribution range: [{attribution.min():.3f}, {attribution.max():.3f}]")

    # Cleanup
    gradcam.remove_hooks()

    print("\n✓ Grad-CAM example completed\n")


def example_shap():
    """Example: Generate SHAP attribution."""
    print("=" * 60)
    print("Example 2: SHAP Attribution")
    print("=" * 60)

    # Load model
    model = load_example_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Create background samples (random for demo)
    print("\n1. Creating background samples...")
    background = np.random.rand(10, 224, 224, 3).astype(np.float32)

    # Create SHAP explainer
    print("2. Initializing SHAP explainer (KernelSHAP)...")
    shap_explainer = SHAPAttribution(
        model,
        background_samples=background,
        n_samples=50,  # Small for demo
        method='kernel',
        device=device
    )

    # Generate attribution
    print("3. Generating SHAP attribution...")
    image = np.random.rand(224, 224, 3).astype(np.float32)

    attribution = shap_explainer.generate_attribution(image)

    print(f"   Attribution shape: {attribution.shape}")
    print(f"   Attribution range: [{attribution.min():.3f}, {attribution.max():.3f}]")

    # With target
    print("\n4. Generating SHAP attribution with target embedding...")
    with torch.no_grad():
        target_embedding = model(
            torch.randn(1, 3, 224, 224).to(device)
        ).cpu().numpy()

    attribution = shap_explainer.generate_attribution(
        image,
        target_embedding=target_embedding[0]
    )

    print(f"   Attribution shape: {attribution.shape}")
    print(f"   Attribution range: [{attribution.min():.3f}, {attribution.max():.3f}]")

    print("\n✓ SHAP example completed\n")


def example_lime():
    """Example: Generate LIME attribution."""
    print("=" * 60)
    print("Example 3: LIME Attribution")
    print("=" * 60)

    # Load model
    model = load_example_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Create LIME explainer
    print("\n1. Initializing LIME explainer...")
    lime_explainer = LIMEAttribution(
        model,
        num_samples=100,  # Small for demo
        num_features=5,
        device=device
    )

    # Generate attribution
    print("2. Generating LIME attribution...")
    image = np.random.rand(224, 224, 3).astype(np.float32)

    attribution = lime_explainer.generate_attribution(image)

    print(f"   Attribution shape: {attribution.shape}")
    print(f"   Attribution range: [{attribution.min():.3f}, {attribution.max():.3f}]")

    # With target
    print("\n3. Generating LIME attribution with target embedding...")
    with torch.no_grad():
        target_embedding = model(
            torch.randn(1, 3, 224, 224).to(device)
        ).cpu().numpy()

    attribution = lime_explainer.generate_attribution(
        image,
        target_embedding=target_embedding[0]
    )

    print(f"   Attribution shape: {attribution.shape}")
    print(f"   Attribution range: [{attribution.min():.3f}, {attribution.max():.3f}]")

    # Get top features
    print("\n4. Extracting top contributing features...")
    mask, boundary_img, weights = lime_explainer.generate_top_features(
        image,
        target_embedding=target_embedding[0],
        num_features=5
    )

    print(f"   Feature mask shape: {mask.shape}")
    print(f"   Boundary image shape: {boundary_img.shape}")
    print(f"   Top features: {len(weights)} segments")

    print("\n✓ LIME example completed\n")


def example_comparison():
    """Example: Compare all three methods."""
    print("=" * 60)
    print("Example 4: Comparing Attribution Methods")
    print("=" * 60)

    # Load model
    model = load_example_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Dummy image
    image_np = np.random.rand(224, 224, 3).astype(np.float32)
    image_tensor = torch.from_numpy(
        image_np.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    # Get target embedding
    with torch.no_grad():
        target_embedding = model(torch.randn(1, 3, 224, 224).to(device))

    print("\n1. Grad-CAM...")
    gradcam = get_gradcam_for_model(model, model_arch='resnet50')
    attr_gradcam = gradcam.generate_attribution(image_tensor, target_embedding)
    print(f"   Range: [{attr_gradcam.min():.3f}, {attr_gradcam.max():.3f}]")
    gradcam.remove_hooks()

    print("\n2. SHAP...")
    background = np.random.rand(10, 224, 224, 3).astype(np.float32)
    shap_exp = SHAPAttribution(
        model, background_samples=background, n_samples=50, device=device
    )
    attr_shap = shap_exp.generate_attribution(
        image_np, target_embedding=target_embedding.cpu().numpy()[0]
    )
    print(f"   Range: [{attr_shap.min():.3f}, {attr_shap.max():.3f}]")

    print("\n3. LIME...")
    lime_exp = LIMEAttribution(model, num_samples=100, device=device)
    attr_lime = lime_exp.generate_attribution(
        image_np, target_embedding=target_embedding.cpu().numpy()[0]
    )
    print(f"   Range: [{attr_lime.min():.3f}, {attr_lime.max():.3f}]")

    print("\n✓ Comparison completed\n")

    # Summary
    print("Summary:")
    print(f"  Grad-CAM: Fast, gradient-based, layer-specific")
    print(f"  SHAP:     Slower, model-agnostic, theoretically grounded")
    print(f"  LIME:     Moderate, interpretable, superpixel-based")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Attribution Methods for Face Verification")
    print("=" * 60 + "\n")

    try:
        example_gradcam()
    except Exception as e:
        print(f"Grad-CAM example failed: {e}\n")

    try:
        example_shap()
    except Exception as e:
        print(f"SHAP example failed: {e}\n")

    try:
        example_lime()
    except Exception as e:
        print(f"LIME example failed: {e}\n")

    try:
        example_comparison()
    except Exception as e:
        print(f"Comparison example failed: {e}\n")

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
