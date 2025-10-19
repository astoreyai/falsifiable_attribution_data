"""Falsification test for attribution methods (PROPER IMPLEMENTATION).

This module implements the core falsification framework from Chapter 3:
- Definition 3.1: Falsifiability Criterion
- Theorem 3.9: Expected Falsification Rate
- Algorithm 3.1: Falsification Test Procedure

IMPORTANT: This is the CORRECT implementation that uses region-specific
counterfactual generation, not the naive distance-sorting approach.

The key idea is to test whether attribution maps correctly identify
important vs. unimportant regions by measuring how MASKING each
region affects the model output (via embedding changes).

Theoretical Framework:
- High-attribution regions (> theta_high) should cause LARGE changes when masked
- Low-attribution regions (< theta_low) should cause SMALL changes when masked
- If d_high <= d_low, the attribution is FALSIFIED

This is a **causal test**: Does masking high-attribution pixels actually
cause larger embedding changes than masking low-attribution pixels?
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
from .regional_counterfactuals import (
    generate_regional_counterfactuals,
    compute_regional_statistics,
    prepare_image_tensor,
)
from .counterfactual_generation import compute_geodesic_distance


def falsification_test(
    attribution_map: np.ndarray,
    img: np.ndarray,
    model: Callable,
    theta_high: float = 0.7,
    theta_low: float = 0.3,
    K: int = 100,
    masking_strategy: str = 'zero',
    device: str = 'cuda',
    return_details: bool = False
) -> Dict:
    """
    Test if attribution is falsifiable using region-specific counterfactuals.

    Based on Definition 3.1 (Falsifiability Criterion).

    An attribution is falsifiable if:
    - High-attribution regions (> theta_high) produce LARGE changes when masked
    - Low-attribution regions (< theta_low) produce SMALL changes when masked
    - Separation margin d_high - d_low is statistically significant

    If d_high <= d_low, the attribution is FALSIFIED (it claims important
    regions that don't actually matter, or vice versa).

    IMPORTANT: This is the CORRECT implementation that generates counterfactuals
    by masking spatial regions and recomputing embeddings. This tests the causal
    relationship between attribution importance and embedding change.

    Parameters
    ----------
    attribution_map : np.ndarray
        Attribution/saliency map for the input image.
        Shape: (H, W) or (H, W, C)
        Values should be in [0, 1] or will be normalized
    img : np.ndarray
        Original input image
        Shape: (H, W, C) in [0, 255] or [0, 1]
        This is used to generate region-masked counterfactuals
    model : Callable
        Face recognition model (takes image tensor, returns embedding)
        Should support: embedding = model(img_tensor)
    theta_high : float, optional
        Threshold for high-attribution regions (default 0.7)
    theta_low : float, optional
        Threshold for low-attribution regions (default 0.3)
    K : int, optional
        Number of counterfactuals per region type (default 100)
    masking_strategy : str, optional
        How to mask pixels: 'zero', 'mean', or 'noise' (default 'zero')
    device : str, optional
        Device for computation (default 'cuda')
    return_details : bool, optional
        If True, return per-counterfactual distances (default False)

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'is_falsified': bool
            True if attribution is falsified (d_high <= d_low)
        - 'falsification_rate': float
            Percentage of pairwise comparisons failing test (0-100)
        - 'd_high': float
            Mean geodesic distance for high-attribution perturbations
        - 'd_low': float
            Mean geodesic distance for low-attribution perturbations
        - 'separation_margin': float
            d_high - d_low (positive = valid, negative = falsified)
        - 'd_high_std': float
            Standard deviation of high-attribution distances
        - 'd_low_std': float
            Standard deviation of low-attribution distances
        - 'n_high': int
            Number of counterfactuals in high region (= K)
        - 'n_low': int
            Number of counterfactuals in low region (= K)
        - 'distances_high': np.ndarray (if return_details=True)
            Individual distances for high-attribution perturbations
        - 'distances_low': np.ndarray (if return_details=True)
            Individual distances for low-attribution perturbations

    Raises
    ------
    ValueError
        If theta_high <= theta_low
        If no pixels found in high/low regions

    Examples
    --------
    >>> # Load image and compute attribution
    >>> img = load_image('face.jpg')
    >>> attribution = gradcam(img)
    >>>
    >>> # Run falsification test
    >>> result = falsification_test(
    ...     attribution_map=attribution,
    ...     img=img,
    ...     model=face_model,
    ...     K=100,
    ...     device='cuda'
    ... )
    >>> result['is_falsified']
    False  # or True depending on attribution quality
    >>> result['separation_margin']
    0.234  # positive = valid attribution
    """
    if theta_high <= theta_low:
        raise ValueError(
            f"theta_high ({theta_high}) must be > theta_low ({theta_low})"
        )

    # Normalize attribution map if needed
    if attribution_map.ndim == 3 and attribution_map.shape[2] == 3:
        # Convert RGB attribution to grayscale
        attribution_map = np.mean(attribution_map, axis=2)

    # Get original embedding
    img_tensor = prepare_image_tensor(img, device)

    with torch.no_grad():
        if hasattr(model, 'get_embedding'):
            original_embedding = model.get_embedding(img_tensor)
        elif hasattr(model, 'model'):
            original_embedding = model.model(img_tensor)
        else:
            original_embedding = model(img_tensor)

        original_embedding = F.normalize(original_embedding, p=2, dim=-1)

    # Generate region-specific counterfactuals
    # This is the CORE of the proper implementation
    try:
        high_counterfactuals, low_counterfactuals = generate_regional_counterfactuals(
            img=img,
            attribution_map=attribution_map,
            model=model,
            theta_high=theta_high,
            theta_low=theta_low,
            K=K,
            masking_strategy=masking_strategy,
            device=device
        )
    except ValueError as e:
        # No pixels in high/low regions - attribution is likely uniform
        raise ValueError(
            f"Cannot perform falsification test: {e}\n"
            f"This typically indicates the attribution map is uniform or "
            f"has insufficient contrast. Consider adjusting theta_high/theta_low."
        )

    # Compute regional statistics
    stats = compute_regional_statistics(
        original_embedding,
        high_counterfactuals,
        low_counterfactuals
    )

    d_high = stats['d_high']
    d_low = stats['d_low']
    d_high_std = stats['d_high_std']
    d_low_std = stats['d_low_std']
    separation_margin = stats['separation_margin']
    distances_high = stats['distances_high']
    distances_low = stats['distances_low']

    # Falsification criterion: d_high should be > d_low
    # If d_high <= d_low, the attribution is falsified
    is_falsified = separation_margin <= 0

    # Compute falsification rate
    # FR = percentage of pairwise comparisons where d_high <= d_low
    # For each pair, count violations
    violations = 0
    total_comparisons = 0

    for d_h in distances_high:
        for d_l in distances_low:
            if d_h <= d_l:
                violations += 1
            total_comparisons += 1

    falsification_rate = (violations / total_comparisons * 100) if total_comparisons > 0 else 0.0

    # Build result
    result = {
        'is_falsified': bool(is_falsified),
        'falsification_rate': float(falsification_rate),
        'd_high': float(d_high),
        'd_low': float(d_low),
        'separation_margin': float(separation_margin),
        'd_high_std': float(d_high_std),
        'd_low_std': float(d_low_std),
        'n_high': K,
        'n_low': K,
    }

    if return_details:
        result['distances_high'] = distances_high
        result['distances_low'] = distances_low

    return result


def compute_falsification_rate(
    attribution_maps: List[np.ndarray],
    images: List[np.ndarray],
    model: Callable,
    K: int = 100,
    theta_high: float = 0.7,
    theta_low: float = 0.3,
    masking_strategy: str = 'zero',
    device: str = 'cuda',
    verbose: bool = False
) -> float:
    """
    Compute falsification rate across multiple samples.

    Based on Theorem 3.9 (Expected Falsification Rate).

    FR = (# falsified attributions) / (# total attributions)

    This aggregates results across multiple test samples to provide
    a robust measure of attribution method quality.

    Parameters
    ----------
    attribution_maps : list of np.ndarray
        List of N attribution maps
    images : list of np.ndarray
        List of N original images (required for regional masking)
    model : Callable
        Face recognition model
    K : int, optional
        Number of counterfactuals per region (default 100)
    theta_high : float, optional
        High attribution threshold (default 0.7)
    theta_low : float, optional
        Low attribution threshold (default 0.3)
    masking_strategy : str, optional
        Masking strategy (default 'zero')
    device : str, optional
        Device for computation (default 'cuda')
    verbose : bool, optional
        Print progress information (default False)

    Returns
    -------
    falsification_rate : float
        Overall falsification rate as percentage (0-100)
        Higher = more attributions falsified (bad)
        Lower = fewer falsified (good)

    Raises
    ------
    ValueError
        If input lists have different lengths

    Examples
    --------
    >>> # Test 100 samples
    >>> attr_maps = [gradcam(img) for img in images]
    >>>
    >>> fr = compute_falsification_rate(
    ...     attr_maps, images, model=face_model, K=100
    ... )
    >>> print(f"Falsification Rate: {fr:.2f}%")
    """
    N = len(attribution_maps)

    if len(images) != N:
        raise ValueError(
            f"Input lists must have same length: "
            f"attribution_maps={len(attribution_maps)}, "
            f"images={len(images)}"
        )

    if N == 0:
        raise ValueError("Cannot compute FR on empty list")

    falsified_count = 0
    total_count = N
    errors = 0

    for i in range(N):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processing sample {i+1}/{N}...")

        try:
            result = falsification_test(
                attribution_map=attribution_maps[i],
                img=images[i],
                model=model,
                theta_high=theta_high,
                theta_low=theta_low,
                K=K,
                masking_strategy=masking_strategy,
                device=device,
                return_details=False
            )

            if result['is_falsified']:
                falsified_count += 1

        except ValueError as e:
            # Skip samples where no high/low regions found
            if verbose:
                print(f"  Warning: Sample {i+1} skipped ({e})")
            errors += 1
            total_count -= 1

    if total_count == 0:
        raise ValueError(
            f"All {N} samples failed (no valid high/low regions). "
            f"Check attribution maps and thresholds."
        )

    falsification_rate = (falsified_count / total_count) * 100

    if verbose:
        print(f"\nFalsified: {falsified_count}/{total_count}")
        print(f"Errors: {errors}/{N}")
        print(f"Falsification Rate: {falsification_rate:.2f}%")

    return float(falsification_rate)


def compute_separation_ratio(
    attribution_map: np.ndarray,
    img: np.ndarray,
    model: Callable,
    theta_high: float = 0.7,
    theta_low: float = 0.3,
    K: int = 100,
    device: str = 'cuda'
) -> float:
    """
    Compute separation ratio d_high / d_low.

    A ratio > 1 indicates valid attribution (high-importance regions
    produce larger changes). A ratio <= 1 indicates falsification.

    Parameters
    ----------
    attribution_map : np.ndarray
        Attribution map
    img : np.ndarray
        Original image
    model : Callable
        Face recognition model
    theta_high : float, optional
        High attribution threshold
    theta_low : float, optional
        Low attribution threshold
    K : int, optional
        Number of counterfactuals
    device : str, optional
        Device for computation

    Returns
    -------
    ratio : float
        Separation ratio d_high / d_low
        > 1.0: valid attribution
        = 1.0: no separation
        < 1.0: falsified attribution

    Examples
    --------
    >>> ratio = compute_separation_ratio(attribution, img, model)
    >>> if ratio > 1.0:
    ...     print("Valid attribution")
    ... else:
    ...     print("Falsified attribution")
    """
    result = falsification_test(
        attribution_map=attribution_map,
        img=img,
        model=model,
        theta_high=theta_high,
        theta_low=theta_low,
        K=K,
        device=device,
        return_details=False
    )

    d_high = result['d_high']
    d_low = result['d_low']

    if d_low == 0:
        return float('inf') if d_high > 0 else 1.0

    ratio = d_high / d_low
    return float(ratio)


def batch_falsification_test(
    attribution_maps: np.ndarray,
    images: np.ndarray,
    model: Callable,
    theta_high: float = 0.7,
    theta_low: float = 0.3,
    K: int = 100,
    device: str = 'cuda'
) -> List[Dict]:
    """
    Run falsification test on batch of samples efficiently.

    Parameters
    ----------
    attribution_maps : np.ndarray
        Batch of attribution maps, shape (N, H, W) or (N, H, W, C)
    images : np.ndarray
        Batch of original images, shape (N, H, W, C)
    model : Callable
        Face recognition model
    theta_high : float, optional
        High attribution threshold
    theta_low : float, optional
        Low attribution threshold
    K : int, optional
        Number of counterfactuals per region
    device : str, optional
        Device for computation

    Returns
    -------
    results : list of dict
        List of N result dictionaries from falsification_test

    Examples
    --------
    >>> N = 100
    >>> attr_maps = np.random.rand(N, 224, 224)
    >>> images = np.random.rand(N, 224, 224, 3)
    >>>
    >>> results = batch_falsification_test(attr_maps, images, model)
    >>> falsified = sum(r['is_falsified'] for r in results)
    >>> print(f"Falsified: {falsified}/{N}")
    """
    N = attribution_maps.shape[0]

    if images.shape[0] != N:
        raise ValueError(
            f"Batch size mismatch: attribution_maps={N}, "
            f"images={images.shape[0]}"
        )

    results = []

    for i in range(N):
        try:
            result = falsification_test(
                attribution_map=attribution_maps[i],
                img=images[i],
                model=model,
                theta_high=theta_high,
                theta_low=theta_low,
                K=K,
                device=device,
                return_details=False
            )
            results.append(result)
        except ValueError:
            # Skip samples with errors
            results.append({
                'is_falsified': None,
                'falsification_rate': None,
                'd_high': None,
                'd_low': None,
                'separation_margin': None,
                'd_high_std': None,
                'd_low_std': None,
                'n_high': K,
                'n_low': K,
            })

    return results
