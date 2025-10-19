"""Data utilities for face verification experiments."""

import torch
import torchvision.transforms as transforms
from typing import Tuple, List, Dict, Any
import numpy as np


def get_default_transforms(
    image_size: int = 112,
    normalize: bool = True,
    augment: bool = False
) -> transforms.Compose:
    """
    Get default image transformations for face verification.

    Args:
        image_size: Target image size (default 112 for ArcFace/InsightFace)
        normalize: Whether to normalize to [-1, 1] (default True)
        augment: Whether to apply data augmentation (default False)

    Returns:
        transform: Composed transformations
    """
    transform_list = []

    # Resize
    transform_list.append(transforms.Resize((image_size, image_size)))

    # Optional augmentation
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomRotation(degrees=5),
        ])

    # Convert to tensor
    transform_list.append(transforms.ToTensor())

    # Normalize
    if normalize:
        # Standard normalization for face recognition models
        # Maps [0, 1] to [-1, 1]
        transform_list.append(
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )

    return transforms.Compose(transform_list)


def get_insightface_transforms(image_size: int = 112) -> transforms.Compose:
    """
    Get transformations compatible with InsightFace/ArcFace models.

    Args:
        image_size: Target image size (default 112)

    Returns:
        transform: Composed transformations
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_facenet_transforms(image_size: int = 160) -> transforms.Compose:
    """
    Get transformations compatible with FaceNet models.

    Args:
        image_size: Target image size (default 160)

    Returns:
        transform: Composed transformations
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_vggface_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get transformations compatible with VGGFace models.

    Args:
        image_size: Target image size (default 224)

    Returns:
        transform: Composed transformations
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])


def collate_verification_pairs(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for face verification pairs.

    Args:
        batch: List of (img1, img2, label) tuples

    Returns:
        img1_batch: Batch of first images [B, C, H, W]
        img2_batch: Batch of second images [B, C, H, W]
        labels: Batch of labels [B]
    """
    img1_list = []
    img2_list = []
    label_list = []

    for img1, img2, label in batch:
        img1_list.append(img1)
        img2_list.append(img2)
        label_list.append(label)

    img1_batch = torch.stack(img1_list, dim=0)
    img2_batch = torch.stack(img2_list, dim=0)
    labels = torch.tensor(label_list, dtype=torch.long)

    return img1_batch, img2_batch, labels


def collate_attribute_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for images with attributes.

    Args:
        batch: List of (image, attributes) tuples

    Returns:
        images: Batch of images [B, C, H, W]
        attributes: Batch of attribute vectors [B, N_attrs]
    """
    img_list = []
    attr_list = []

    for img, attrs in batch:
        img_list.append(img)
        attr_list.append(attrs)

    images = torch.stack(img_list, dim=0)
    attributes = torch.stack(attr_list, dim=0)

    return images, attributes


def compute_embedding_statistics(embeddings: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics for face embeddings.

    Args:
        embeddings: Embedding tensor [N, D]

    Returns:
        Dictionary of statistics
    """
    embeddings_np = embeddings.detach().cpu().numpy()

    # Compute L2 norms
    norms = np.linalg.norm(embeddings_np, axis=1)

    # Compute pairwise cosine similarities (sample 1000 pairs)
    n_samples = min(1000, len(embeddings_np))
    indices = np.random.choice(len(embeddings_np), size=n_samples, replace=False)
    sampled_embs = embeddings_np[indices]

    # Normalize for cosine similarity
    sampled_embs_norm = sampled_embs / (np.linalg.norm(sampled_embs, axis=1, keepdims=True) + 1e-8)

    # Compute similarity matrix
    sim_matrix = np.dot(sampled_embs_norm, sampled_embs_norm.T)

    # Get upper triangle (excluding diagonal)
    upper_triangle = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]

    return {
        'embedding_dim': embeddings.shape[1],
        'n_samples': embeddings.shape[0],
        'norm_mean': float(norms.mean()),
        'norm_std': float(norms.std()),
        'norm_min': float(norms.min()),
        'norm_max': float(norms.max()),
        'cosine_similarity_mean': float(upper_triangle.mean()),
        'cosine_similarity_std': float(upper_triangle.std()),
        'cosine_similarity_min': float(upper_triangle.min()),
        'cosine_similarity_max': float(upper_triangle.max()),
    }


def compute_verification_metrics(
    similarities: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray = None
) -> Dict[str, Any]:
    """
    Compute face verification metrics at multiple thresholds.

    Args:
        similarities: Similarity scores [N]
        labels: Ground truth labels [N] (1 for genuine, 0 for impostor)
        thresholds: Threshold values to evaluate (default: 100 values from min to max)

    Returns:
        Dictionary containing:
            - tpr: True positive rates at each threshold
            - fpr: False positive rates at each threshold
            - thresholds: Threshold values
            - best_threshold: Threshold maximizing (TPR - FPR)
            - best_accuracy: Accuracy at best threshold
    """
    if thresholds is None:
        thresholds = np.linspace(similarities.min(), similarities.max(), 100)

    tpr_list = []
    fpr_list = []
    acc_list = []

    for threshold in thresholds:
        # Predictions
        predictions = (similarities >= threshold).astype(int)

        # True positives, false positives, true negatives, false negatives
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        # Metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        acc = (tp + tn) / len(labels)

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        acc_list.append(acc)

    tpr_array = np.array(tpr_list)
    fpr_array = np.array(fpr_list)
    acc_array = np.array(acc_list)

    # Find best threshold (maximize TPR - FPR)
    gap = tpr_array - fpr_array
    best_idx = np.argmax(gap)

    return {
        'tpr': tpr_array,
        'fpr': fpr_array,
        'accuracy': acc_array,
        'thresholds': thresholds,
        'best_threshold': float(thresholds[best_idx]),
        'best_tpr': float(tpr_array[best_idx]),
        'best_fpr': float(fpr_array[best_idx]),
        'best_accuracy': float(acc_array[best_idx]),
    }


def split_dataset_indices(
    n_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset indices into train/val/test sets.

    Args:
        n_samples: Total number of samples
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        train_indices, val_indices, test_indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    np.random.seed(seed)
    indices = np.random.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    return train_indices, val_indices, test_indices


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize embeddings.

    Args:
        embeddings: Embedding tensor [N, D]

    Returns:
        Normalized embeddings [N, D]
    """
    norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    return embeddings / (norms + 1e-8)


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between embedding pairs.

    Args:
        emb1: First embeddings [N, D]
        emb2: Second embeddings [N, D]

    Returns:
        Similarities [N]
    """
    # Normalize
    emb1_norm = normalize_embeddings(emb1)
    emb2_norm = normalize_embeddings(emb2)

    # Dot product
    similarities = (emb1_norm * emb2_norm).sum(dim=1)

    return similarities


def euclidean_distance(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """
    Compute Euclidean distance between embedding pairs.

    Args:
        emb1: First embeddings [N, D]
        emb2: Second embeddings [N, D]

    Returns:
        Distances [N]
    """
    return torch.norm(emb1 - emb2, p=2, dim=1)
