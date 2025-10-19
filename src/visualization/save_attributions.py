"""
Attribution Visualization and Saving Module

This module provides utilities to save attribution maps as images and data files,
ensuring all experimental results are properly documented and reproducible.

Features:
- Save attribution maps as PNG/JPG heatmaps
- Save raw attribution data as NumPy arrays
- Create comparison visualizations (multiple methods side-by-side)
- Generate publication-quality figures
- Save metadata (method name, parameters, timestamps)

Citation: Supporting all experiments (6.1-6.6)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def save_attribution_heatmap(
    attribution: np.ndarray,
    output_path: str,
    original_image: Optional[np.ndarray] = None,
    overlay_alpha: float = 0.5,
    cmap: str = 'jet',
    title: Optional[str] = None,
    dpi: int = 150
) -> None:
    """
    Save attribution map as heatmap image.

    Args:
        attribution: Attribution map (H, W) in [0, 1]
        output_path: Path to save image (e.g., 'output/gradcam_sample001.png')
        original_image: Optional original image (H, W, C) to overlay
        overlay_alpha: Transparency for overlay (0=only heatmap, 1=only image)
        cmap: Colormap ('jet', 'hot', 'viridis', etc.)
        title: Optional title for plot
        dpi: Image resolution

    Example:
        >>> attr = gradcam.explain(img1, img2)
        >>> save_attribution_heatmap(
        ...     attr,
        ...     'results/gradcam_pair_001.png',
        ...     original_image=img1,
        ...     overlay_alpha=0.6,
        ...     title='Grad-CAM Attribution'
        ... )
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    if original_image is not None:
        # Show original image
        ax.imshow(original_image)

        # Overlay heatmap
        heatmap = cm.get_cmap(cmap)(attribution)
        ax.imshow(heatmap, alpha=overlay_alpha)
    else:
        # Show only heatmap
        im = ax.imshow(attribution, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        ax.set_title(title, fontsize=14)

    ax.axis('off')
    plt.tight_layout()

    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved attribution heatmap: {output_path}")


def quick_save(
    attribution: np.ndarray,
    output_path: str,
    image: Optional[np.ndarray] = None,
    method: str = 'attribution'
) -> None:
    """
    Quick save with sensible defaults.

    Args:
        attribution: Attribution map
        output_path: Output path (will save PNG)
        image: Optional original image
        method: Method name

    Example:
        >>> quick_save(attr, 'results/sample001', img1, 'Grad-CAM')
    """
    save_attribution_heatmap(
        attribution,
        f"{output_path}.png",
        original_image=image,
        title=method
    )


if __name__ == '__main__':
    print("Attribution visualization module loaded.")
