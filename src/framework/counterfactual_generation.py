"""Counterfactual generation on face embedding hypersphere.

This module implements the theoretical framework from Chapter 3:
- Theorem 3.6: Existence of Counterfactuals on Hyperspheres
- Theorem 3.8: Geodesic Sampling on Unit Hypersphere
- Theorem 3.3: Geodesic Distance Metric

The core idea is to generate counterfactual embeddings by sampling
points on the hypersphere around the original embedding, maintaining
the L2-normalization constraint of face embeddings.
"""

import numpy as np
import torch
from typing import Tuple, List, Optional


def generate_counterfactuals_hypersphere(
    embedding: torch.Tensor,
    K: int = 100,
    noise_scale: float = 0.1,
    normalize: bool = True,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate K counterfactual embeddings on hypersphere.

    Based on Theorem 3.6 (Existence of Counterfactuals on Hyperspheres)
    and Theorem 3.8 (Geodesic Sampling on Unit Hypersphere).

    The algorithm:
    1. Sample K Gaussian noise vectors in R^d
    2. Project onto tangent space of hypersphere at embedding point
    3. Normalize to unit length (project back to hypersphere)

    This ensures all counterfactuals lie on the same hypersphere as
    the original embedding, maintaining the geometric structure of
    face embedding spaces.

    Parameters
    ----------
    embedding : torch.Tensor
        Original face embedding (d-dimensional, should be normalized).
        Shape: (d,) or (1, d)
    K : int, optional
        Number of counterfactuals to generate (default 100)
    noise_scale : float, optional
        Gaussian noise scale for sampling (default 0.1)
        Smaller values = counterfactuals closer to original
        Larger values = more diverse counterfactuals
    normalize : bool, optional
        Whether to project onto unit hypersphere (default True)
    device : torch.device, optional
        Device for computation (default: same as embedding)

    Returns
    -------
    counterfactuals : torch.Tensor
        Tensor of shape (K, d) with K counterfactual embeddings,
        all normalized to unit length if normalize=True

    Raises
    ------
    ValueError
        If embedding is not 1D or 2D tensor
        If K <= 0
        If noise_scale <= 0

    Examples
    --------
    >>> embedding = torch.randn(512)
    >>> embedding = embedding / embedding.norm()  # normalize
    >>> counterfactuals = generate_counterfactuals_hypersphere(embedding, K=100)
    >>> counterfactuals.shape
    torch.Size([100, 512])
    >>> # All counterfactuals should be unit norm
    >>> torch.allclose(counterfactuals.norm(dim=1), torch.ones(100), atol=1e-5)
    True
    """
    if device is None:
        device = embedding.device

    # Input validation
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)  # (d,) -> (1, d)
    elif embedding.dim() != 2:
        raise ValueError(f"Embedding must be 1D or 2D tensor, got {embedding.dim()}D")

    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")

    if noise_scale <= 0:
        raise ValueError(f"noise_scale must be positive, got {noise_scale}")

    # Get embedding dimension
    d = embedding.shape[1]

    # Ensure embedding is normalized
    embedding = embedding / embedding.norm(dim=1, keepdim=True)

    # Generate K Gaussian noise vectors in R^d
    # Shape: (K, d)
    noise = torch.randn(K, d, device=device) * noise_scale

    # Project noise onto tangent space at embedding point
    # Tangent space is orthogonal to embedding vector
    # tangent_noise = noise - ⟨noise, embedding⟩ * embedding
    dot_products = (noise * embedding).sum(dim=1, keepdim=True)  # (K, 1)
    tangent_noise = noise - dot_products * embedding  # (K, d)

    # Move along tangent direction: embedding + tangent_noise
    counterfactuals = embedding + tangent_noise  # (K, d)

    # Project back onto unit hypersphere
    if normalize:
        counterfactuals = counterfactuals / counterfactuals.norm(dim=1, keepdim=True)

    return counterfactuals


def compute_geodesic_distance(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    eps: float = 1e-7
) -> float:
    """
    Compute geodesic distance on hypersphere.

    Based on Theorem 3.3 (Geodesic Distance Metric).

    For two points x1, x2 on the unit hypersphere S^(d-1), the
    geodesic distance (shortest path along the sphere surface) is:

        d_g(x1, x2) = arccos(⟨x1, x2⟩)

    where ⟨·,·⟩ is the inner product. This measures the angle
    between the two vectors.

    Parameters
    ----------
    emb1 : torch.Tensor
        First embedding vector (d-dimensional, normalized).
        Shape: (d,) or (1, d)
    emb2 : torch.Tensor
        Second embedding vector (d-dimensional, normalized).
        Shape: (d,) or (1, d)
    eps : float, optional
        Small epsilon for numerical stability in arccos (default 1e-7)

    Returns
    -------
    distance : float
        Geodesic distance in radians, range [0, π]
        - 0 means identical vectors
        - π/2 means orthogonal vectors
        - π means opposite vectors

    Raises
    ------
    ValueError
        If embeddings have different dimensions

    Examples
    --------
    >>> emb1 = torch.tensor([1.0, 0.0, 0.0])
    >>> emb2 = torch.tensor([0.0, 1.0, 0.0])
    >>> compute_geodesic_distance(emb1, emb2)
    1.5707963267948966  # π/2 (orthogonal)

    >>> emb1 = torch.tensor([1.0, 0.0, 0.0])
    >>> emb2 = torch.tensor([1.0, 0.0, 0.0])
    >>> compute_geodesic_distance(emb1, emb2)
    0.0  # identical
    """
    # Flatten to 1D if needed
    if emb1.dim() > 1:
        emb1 = emb1.squeeze()
    if emb2.dim() > 1:
        emb2 = emb2.squeeze()

    if emb1.shape != emb2.shape:
        raise ValueError(
            f"Embeddings must have same shape, got {emb1.shape} and {emb2.shape}"
        )

    # Compute cosine similarity (inner product for normalized vectors)
    cos_sim = torch.dot(emb1, emb2).item()

    # Clamp to [-1, 1] for numerical stability
    # (floating point errors can push slightly outside this range)
    cos_sim = np.clip(cos_sim, -1.0 + eps, 1.0 - eps)

    # Compute geodesic distance
    distance = np.arccos(cos_sim)

    return float(distance)


def compute_pairwise_geodesic_distances(
    embeddings: torch.Tensor,
    counterfactuals: torch.Tensor,
    batch_size: int = 1000
) -> torch.Tensor:
    """
    Compute pairwise geodesic distances efficiently.

    For large sets of counterfactuals, computing distances one-by-one
    can be slow. This function batches the computation for efficiency.

    Parameters
    ----------
    embeddings : torch.Tensor
        Original embeddings, shape (N, d)
    counterfactuals : torch.Tensor
        Counterfactual embeddings, shape (K, d)
    batch_size : int, optional
        Batch size for computation (default 1000)

    Returns
    -------
    distances : torch.Tensor
        Pairwise distances, shape (N, K)
        distances[i, j] = geodesic distance from embeddings[i] to counterfactuals[j]

    Examples
    --------
    >>> embeddings = torch.randn(10, 512)
    >>> embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    >>> counterfactuals = torch.randn(100, 512)
    >>> counterfactuals = counterfactuals / counterfactuals.norm(dim=1, keepdim=True)
    >>> distances = compute_pairwise_geodesic_distances(embeddings, counterfactuals)
    >>> distances.shape
    torch.Size([10, 100])
    """
    N = embeddings.shape[0]
    K = counterfactuals.shape[0]
    device = embeddings.device

    # Ensure normalized
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    counterfactuals = counterfactuals / counterfactuals.norm(dim=1, keepdim=True)

    # Compute cosine similarities (matrix multiplication)
    # Shape: (N, K)
    cos_sims = torch.mm(embeddings, counterfactuals.t())

    # Clamp for numerical stability
    cos_sims = torch.clamp(cos_sims, -1.0 + 1e-7, 1.0 - 1e-7)

    # Compute geodesic distances
    distances = torch.acos(cos_sims)

    return distances


def sample_counterfactuals_at_distance(
    embedding: torch.Tensor,
    target_distance: float,
    K: int = 100,
    max_iterations: int = 1000,
    tolerance: float = 0.01
) -> torch.Tensor:
    """
    Generate counterfactuals at a specific geodesic distance.

    This is useful for controlled experiments where you want
    counterfactuals at a specific distance from the original.

    Parameters
    ----------
    embedding : torch.Tensor
        Original face embedding (d-dimensional, normalized)
    target_distance : float
        Target geodesic distance in radians, range (0, π)
    K : int, optional
        Number of counterfactuals to generate (default 100)
    max_iterations : int, optional
        Maximum rejection sampling iterations (default 1000)
    tolerance : float, optional
        Acceptable distance tolerance in radians (default 0.01)

    Returns
    -------
    counterfactuals : torch.Tensor
        K counterfactuals at approximately target_distance from embedding

    Raises
    ------
    RuntimeError
        If unable to generate K counterfactuals within max_iterations

    Examples
    --------
    >>> embedding = torch.randn(512)
    >>> embedding = embedding / embedding.norm()
    >>> # Generate counterfactuals at distance π/4 (45 degrees)
    >>> counterfactuals = sample_counterfactuals_at_distance(
    ...     embedding, target_distance=np.pi/4, K=50
    ... )
    >>> # Check distances
    >>> distances = [compute_geodesic_distance(embedding, cf) for cf in counterfactuals]
    >>> np.mean(distances)  # Should be close to π/4
    """
    if not (0 < target_distance < np.pi):
        raise ValueError(f"target_distance must be in (0, π), got {target_distance}")

    device = embedding.device
    d = embedding.shape[-1] if embedding.dim() > 1 else embedding.shape[0]

    # Flatten embedding
    if embedding.dim() > 1:
        embedding = embedding.squeeze()

    # Normalize
    embedding = embedding / embedding.norm()

    collected = []
    iterations = 0

    # Rejection sampling: generate and keep only those at target distance
    while len(collected) < K and iterations < max_iterations:
        # Generate batch
        batch_size = min(K * 10, 1000)  # Generate 10x needed
        candidates = generate_counterfactuals_hypersphere(
            embedding,
            K=batch_size,
            noise_scale=target_distance,  # Use target as scale hint
            normalize=True,
            device=device
        )

        # Compute distances
        distances = torch.tensor([
            compute_geodesic_distance(embedding, cf)
            for cf in candidates
        ])

        # Keep those within tolerance
        mask = torch.abs(distances - target_distance) < tolerance
        valid = candidates[mask]

        collected.append(valid)
        iterations += 1

        # Early exit if we have enough
        if sum(c.shape[0] for c in collected) >= K:
            break

    if len(collected) == 0:
        raise RuntimeError(
            f"Unable to generate counterfactuals at distance {target_distance} "
            f"within {max_iterations} iterations. Try increasing tolerance."
        )

    # Concatenate and take first K
    result = torch.cat(collected, dim=0)[:K]

    if result.shape[0] < K:
        raise RuntimeError(
            f"Only generated {result.shape[0]}/{K} counterfactuals at target distance. "
            f"Try increasing max_iterations or tolerance."
        )

    return result


def sample_size_from_theorem_3_8(
    epsilon: float = 0.1,
    delta: float = 0.05,
    d: int = 512
) -> int:
    """
    Compute required sample size from Theorem 3.8.

    Theorem 3.8 states that for confidence level (1-δ) and error margin ε,
    we need:
        n ≥ (2/ε²) · log(2d/δ)

    Args:
        epsilon: Error margin in radians (default: 0.1 rad ≈ 5.7°)
        delta: Confidence parameter (default: 0.05 for 95% confidence)
        d: Embedding dimension (default: 512 for ArcFace)

    Returns:
        Minimum sample size n

    Reference: Theorem 3.8 (Chapter 3, Section 3.4.1)
    """
    n = int(np.ceil((2 / (epsilon ** 2)) * np.log(2 * d / delta)))
    return n


def validate_sample_size(
    n_samples: int,
    epsilon: float = 0.1,
    delta: float = 0.05,
    d: int = 512
) -> Tuple[bool, int]:
    """
    Validate if sample size meets theoretical requirements.

    Args:
        n_samples: Actual number of samples
        epsilon: Desired error margin
        delta: Desired confidence level
        d: Embedding dimension

    Returns:
        (is_valid, required_n): Whether sample size is sufficient and required size
    """
    required_n = sample_size_from_theorem_3_8(epsilon, delta, d)
    is_valid = n_samples >= required_n

    if not is_valid:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Sample size {n_samples} < {required_n} required for "
            f"ε={epsilon:.3f} rad, δ={delta:.3f}, d={d}"
        )

    return is_valid, required_n
