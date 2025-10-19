"""
Region-Specific Counterfactual Generation for Falsification Testing

This module implements spatially-grounded counterfactual generation by masking
specific image regions identified by attribution maps.

This is the CORRECT implementation of Definition 3.1 (Falsifiability Criterion)
from the theoretical framework, which requires testing whether perturbations to
high-attribution regions cause larger changes than low-attribution regions.

Key Difference from Naive Approach:
- Naive: Generate embeddings on hypersphere, sort by distance (circular logic)
- Proper: Mask spatial regions, recompute embeddings (causal test)

Theoretical Justification:
If an attribution method is valid, then:
- Masking high-attribution pixels → large embedding change (high d_geodesic)
- Masking low-attribution pixels → small embedding change (low d_geodesic)

If d_high <= d_low, the attribution is FALSIFIED (it incorrectly identifies
important regions).
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Callable, Dict
from .counterfactual_generation import compute_geodesic_distance


def prepare_image_tensor(
    img: np.ndarray,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Convert image to PyTorch tensor in correct format.

    Args:
        img: Image as numpy array
             - (H, W, C) in [0, 255] or [0, 1]
             - (C, H, W) in [0, 1]
        device: Target device

    Returns:
        Image tensor (1, C, H, W) in [0, 1], ready for model
    """
    # Convert to float
    img_float = img.astype(np.float32)

    # Normalize to [0, 1] if needed
    if img_float.max() > 1.0:
        img_float = img_float / 255.0

    # Convert to tensor
    img_tensor = torch.from_numpy(img_float).float().to(device)

    # Handle different input formats
    if img_tensor.ndim == 2:
        # Grayscale (H, W) → (1, 1, H, W)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    elif img_tensor.ndim == 3:
        # Check if (H, W, C) or (C, H, W)
        if img_tensor.shape[2] == 3 or img_tensor.shape[2] == 1:
            # (H, W, C) → (C, H, W) → (1, C, H, W)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            # Already (C, H, W) → (1, C, H, W)
            img_tensor = img_tensor.unsqueeze(0)
    elif img_tensor.ndim == 4:
        # Already (B, C, H, W)
        pass
    else:
        raise ValueError(f"Unexpected image dimensions: {img_tensor.shape}")

    return img_tensor


def generate_regional_counterfactuals(
    img: np.ndarray,
    attribution_map: np.ndarray,
    model: Callable,
    theta_high: float = 0.7,
    theta_low: float = 0.3,
    K: int = 100,
    masking_strategy: str = 'zero',
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate counterfactuals by masking spatial regions.

    This implements the CORRECT falsification test from the theoretical framework:
    1. Identify high-attribution regions (pixels > theta_high)
    2. Identify low-attribution regions (pixels < theta_low)
    3. For each region type, generate K masked versions of the image
    4. Recompute embeddings for each masked image
    5. Return embeddings for comparison

    Args:
        img: Original image (H, W, C) in [0, 255] or [0, 1]
        attribution_map: Attribution map (H, W) in [0, 1]
        model: Face verification model (takes image tensor, returns embedding)
        theta_high: Threshold for high-attribution regions (default 0.7)
        theta_low: Threshold for low-attribution regions (default 0.3)
        K: Number of counterfactuals per region type (default 100)
        masking_strategy: How to mask pixels
            - 'zero': Set masked pixels to 0
            - 'mean': Set to image mean
            - 'noise': Replace with Gaussian noise
        device: Device for computation

    Returns:
        high_counterfactuals: (K, d) - embeddings with high-attr regions masked
        low_counterfactuals: (K, d) - embeddings with low-attr regions masked

    Example:
        >>> img = load_image('face.jpg')  # (112, 112, 3)
        >>> attr_map = gradcam(img)       # (112, 112)
        >>> high_cf, low_cf = generate_regional_counterfactuals(
        ...     img, attr_map, model, K=100
        ... )
        >>> # Test: Do high regions cause larger changes?
        >>> d_high = mean([geodesic_distance(orig_emb, cf) for cf in high_cf])
        >>> d_low = mean([geodesic_distance(orig_emb, cf) for cf in low_cf])
        >>> assert d_high > d_low  # Valid attribution
    """
    # Normalize attribution map to [0, 1]
    attr_min = attribution_map.min()
    attr_max = attribution_map.max()
    if attr_max > attr_min:
        attr_norm = (attribution_map - attr_min) / (attr_max - attr_min)
    else:
        # Uniform attribution - all pixels equal
        attr_norm = np.ones_like(attribution_map) * 0.5

    # Identify regions
    high_mask = attr_norm > theta_high  # (H, W) boolean
    low_mask = attr_norm < theta_low    # (H, W) boolean

    # Check if we have any pixels in each region
    if not high_mask.any():
        raise ValueError(
            f"No high-attribution pixels found (threshold={theta_high}). "
            f"Attribution range: [{attr_norm.min():.3f}, {attr_norm.max():.3f}]"
        )

    if not low_mask.any():
        raise ValueError(
            f"No low-attribution pixels found (threshold={theta_low}). "
            f"Attribution range: [{attr_norm.min():.3f}, {attr_norm.max():.3f}]"
        )

    # Prepare masking value based on strategy
    if masking_strategy == 'zero':
        fill_value = 0.0
    elif masking_strategy == 'mean':
        fill_value = img.mean()
    elif masking_strategy == 'noise':
        fill_value = None  # Will use random noise per sample
    else:
        raise ValueError(f"Unknown masking_strategy: {masking_strategy}")

    # Generate counterfactuals for high-attribution regions
    high_counterfactuals = []
    for i in range(K):
        # Create masked image (mask out high-attribution pixels)
        img_masked = img.copy()

        if masking_strategy == 'noise':
            # Replace high-attr pixels with Gaussian noise
            noise = np.random.randn(*img.shape).astype(np.float32) * 0.1
            img_masked[high_mask] = noise[high_mask]
        else:
            # Replace with constant value
            img_masked[high_mask] = fill_value

        # Add random perturbation for diversity across K samples
        # This prevents identical counterfactuals
        # Scale: 0.02 provides diversity while maintaining semantic similarity
        perturbation = np.random.randn(*img.shape).astype(np.float32) * 0.02
        img_perturbed = np.clip(img_masked + perturbation, 0, 1)

        # Convert to tensor and compute embedding
        img_tensor = prepare_image_tensor(img_perturbed, device)

        with torch.no_grad():
            # Get embedding
            if hasattr(model, 'get_embedding'):
                embedding = model.get_embedding(img_tensor)
            elif hasattr(model, 'model'):
                embedding = model.model(img_tensor)
            else:
                embedding = model(img_tensor)

            # Normalize to unit sphere (standard for face verification)
            embedding = F.normalize(embedding, p=2, dim=-1)

        high_counterfactuals.append(embedding.squeeze().cpu())

    # Generate counterfactuals for low-attribution regions
    low_counterfactuals = []
    for i in range(K):
        # Create masked image (mask out low-attribution pixels)
        img_masked = img.copy()

        if masking_strategy == 'noise':
            noise = np.random.randn(*img.shape).astype(np.float32) * 0.1
            img_masked[low_mask] = noise[low_mask]
        else:
            img_masked[low_mask] = fill_value

        # Add random perturbation
        perturbation = np.random.randn(*img.shape).astype(np.float32) * 0.02
        img_perturbed = np.clip(img_masked + perturbation, 0, 1)

        # Convert to tensor and compute embedding
        img_tensor = prepare_image_tensor(img_perturbed, device)

        with torch.no_grad():
            if hasattr(model, 'get_embedding'):
                embedding = model.get_embedding(img_tensor)
            elif hasattr(model, 'model'):
                embedding = model.model(img_tensor)
            else:
                embedding = model(img_tensor)

            embedding = F.normalize(embedding, p=2, dim=-1)

        low_counterfactuals.append(embedding.squeeze().cpu())

    # Stack into tensors
    high_counterfactuals = torch.stack(high_counterfactuals)  # (K, d)
    low_counterfactuals = torch.stack(low_counterfactuals)    # (K, d)

    return high_counterfactuals, low_counterfactuals


def compute_regional_statistics(
    original_embedding: torch.Tensor,
    high_counterfactuals: torch.Tensor,
    low_counterfactuals: torch.Tensor
) -> Dict:
    """
    Compute statistics for regional counterfactuals.

    Args:
        original_embedding: Original embedding (d,) or (1, d)
        high_counterfactuals: High-region counterfactuals (K, d)
        low_counterfactuals: Low-region counterfactuals (K, d)

    Returns:
        Dictionary with:
        - d_high: Mean distance for high-attribution perturbations
        - d_low: Mean distance for low-attribution perturbations
        - d_high_std: Std of high distances
        - d_low_std: Std of low distances
        - separation_margin: d_high - d_low
        - distances_high: Individual high distances (K,)
        - distances_low: Individual low distances (K,)
    """
    # Flatten original embedding if needed
    if original_embedding.dim() > 1:
        original_embedding = original_embedding.squeeze()

    # Ensure embeddings on same device (move to CPU for distance computation)
    original_embedding = original_embedding.cpu()
    high_counterfactuals = high_counterfactuals.cpu()
    low_counterfactuals = low_counterfactuals.cpu()

    # Compute geodesic distances for high-attribution counterfactuals
    distances_high = np.array([
        compute_geodesic_distance(original_embedding, cf)
        for cf in high_counterfactuals
    ])

    # Compute geodesic distances for low-attribution counterfactuals
    distances_low = np.array([
        compute_geodesic_distance(original_embedding, cf)
        for cf in low_counterfactuals
    ])

    # Statistics
    d_high = np.mean(distances_high)
    d_low = np.mean(distances_low)
    d_high_std = np.std(distances_high)
    d_low_std = np.std(distances_low)
    separation_margin = d_high - d_low

    return {
        'd_high': float(d_high),
        'd_low': float(d_low),
        'd_high_std': float(d_high_std),
        'd_low_std': float(d_low_std),
        'separation_margin': float(separation_margin),
        'distances_high': distances_high,
        'distances_low': distances_low,
    }
