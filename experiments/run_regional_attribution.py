"""
Regional Attribution Analysis using CelebAMask-HQ.

Tests if attribution methods correctly identify relevant facial regions:
- Eye-focused attributions should highlight eye regions in mask
- Nose-focused attributions should highlight nose region
- Mouth-focused attributions should highlight mouth region

Hypothesis: Methods with high falsification rate (SHAP, LIME) will show
poor regional alignment (attributions scattered across face rather than
localized to decision-relevant regions).

This provides an interpretable explanation for WHY some methods are more
reliable: they produce spatially coherent attributions that match facial anatomy.

Example Usage:
    python experiments/run_regional_attribution.py --model resnet50 --method gradcam --n-samples 100
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import argparse
from tqdm import tqdm
import json

# Import dataset
from data.celeba_mask_dataset import CelebAMaskHQ


def compute_regional_consistency(attribution_map, segmentation_mask, dataset, regions=None):
    """
    Compute consistency between attribution map and semantic regions.

    Args:
        attribution_map (np.ndarray): (H, W) saliency map (normalized 0-1)
        segmentation_mask (np.ndarray): (H, W) class IDs (0-18)
        dataset (CelebAMaskHQ): Dataset instance for region extraction
        regions (list, optional): Regions to test. Default: ['eyes', 'nose', 'mouth']

    Returns:
        dict: Region name -> overlap percentage
    """
    if regions is None:
        regions = ['eyes', 'nose', 'mouth', 'face']

    results = {}
    for region in regions:
        overlap = dataset.compute_region_overlap(attribution_map, segmentation_mask, region)
        results[region] = overlap

    return results


def analyze_method_regional_alignment(model, dataset, attribution_method, n_samples=100):
    """
    Analyze regional alignment for a specific attribution method.

    This is the core experiment: Does the attribution method produce spatially
    coherent explanations that align with facial anatomy?

    Args:
        model: Trained face model
        dataset: CelebAMaskHQ dataset
        attribution_method: Attribution generation function
        n_samples: Number of samples to analyze

    Returns:
        dict: {
            'mean_overlap': {region: percentage},
            'std_overlap': {region: percentage},
            'spatial_coherence': float,
            'samples': list of per-sample results
        }
    """
    regions = ['eyes', 'nose', 'mouth', 'face']
    overlaps = {region: [] for region in regions}
    sample_results = []

    # Subset if needed
    if n_samples < len(dataset):
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        subset = Subset(dataset, indices)
    else:
        subset = dataset

    print(f"Analyzing {len(subset)} samples...")

    for idx in tqdm(range(len(subset))):
        sample = subset[idx]
        image = sample['image']
        mask = sample['mask'].numpy()
        img_id = sample['image_id']

        # Generate attribution (placeholder - replace with actual method)
        # attribution_map = attribution_method(model, image)
        # For now, simulate with random attribution
        attribution_map = np.random.rand(512, 512)

        # Resize attribution to match mask if needed
        if attribution_map.shape != mask.shape:
            from PIL import Image
            attribution_map = np.array(
                Image.fromarray(attribution_map).resize(
                    mask.shape[::-1],  # (W, H) for PIL
                    resample=Image.BILINEAR
                )
            )

        # Normalize attribution
        if attribution_map.max() > 0:
            attribution_map = attribution_map / attribution_map.max()

        # Compute regional overlaps
        region_overlaps = compute_regional_consistency(
            attribution_map, mask, dataset, regions
        )

        # Store results
        for region, overlap in region_overlaps.items():
            overlaps[region].append(overlap)

        sample_results.append({
            'image_id': int(img_id),
            'overlaps': region_overlaps
        })

    # Compute statistics
    results = {
        'n_samples': len(subset),
        'mean_overlap': {
            region: float(np.mean(overlaps[region]))
            for region in regions
        },
        'std_overlap': {
            region: float(np.std(overlaps[region]))
            for region in regions
        },
        'median_overlap': {
            region: float(np.median(overlaps[region]))
            for region in regions
        },
        'samples': sample_results[:10]  # Store first 10 for inspection
    }

    return results


def compare_methods(model, dataset, methods, n_samples=100):
    """
    Compare regional alignment across multiple attribution methods.

    Expected results:
    - Grad-CAM (10% FR): High regional precision (70-80%)
    - SHAP/LIME (93% FR): Low regional precision (30-50%)
    - Random baseline: 1/19 = 5.3% (uniform distribution)

    This demonstrates that low FR correlates with semantic coherence!

    Args:
        model: Trained face model
        dataset: CelebAMaskHQ dataset
        methods: dict of {name: attribution_function}
        n_samples: Number of samples per method

    Returns:
        dict: {method_name: results}
    """
    comparison = {}

    for method_name, method_fn in methods.items():
        print(f"\nAnalyzing method: {method_name}")
        results = analyze_method_regional_alignment(
            model, dataset, method_fn, n_samples
        )
        comparison[method_name] = results

        # Print summary
        print(f"  Mean overlap - Eyes: {results['mean_overlap']['eyes']:.2f}%")
        print(f"  Mean overlap - Nose: {results['mean_overlap']['nose']:.2f}%")
        print(f"  Mean overlap - Mouth: {results['mean_overlap']['mouth']:.2f}%")
        print(f"  Mean overlap - Face: {results['mean_overlap']['face']:.2f}%")

    return comparison


def main():
    parser = argparse.ArgumentParser(description='Regional attribution analysis')
    parser.add_argument('--data-root', type=str,
                        default='/home/aaron/projects/xai/data/celeba_mask',
                        help='Path to CelebAMask-HQ dataset')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of samples to analyze')
    parser.add_argument('--output', type=str,
                        default='regional_attribution_results.json',
                        help='Output JSON file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    print(f"Loading CelebAMask-HQ from {args.data_root}...")
    dataset = CelebAMaskHQ(
        root=args.data_root,
        return_mask=True,
        mask_size=512
    )

    print(f"Dataset loaded: {len(dataset)} images")

    # Placeholder model (replace with actual trained model)
    model = None

    # Define attribution methods (placeholders - replace with actual implementations)
    methods = {
        'random': lambda m, x: np.random.rand(512, 512),  # Baseline
        # 'gradcam': gradcam_attribution,
        # 'shap': shap_attribution,
        # 'lime': lime_attribution,
        # 'integrated_gradients': ig_attribution,
    }

    # Run comparison
    print(f"\nRunning regional attribution analysis on {args.n_samples} samples...")
    results = compare_methods(model, dataset, methods, args.n_samples)

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "="*60)
    print("REGIONAL ATTRIBUTION ANALYSIS SUMMARY")
    print("="*60)
    print(f"{'Method':<20} {'Eyes %':<10} {'Nose %':<10} {'Mouth %':<10} {'Face %':<10}")
    print("-"*60)
    for method_name, method_results in results.items():
        mean_overlap = method_results['mean_overlap']
        print(f"{method_name:<20} "
              f"{mean_overlap['eyes']:>8.2f}% "
              f"{mean_overlap['nose']:>8.2f}% "
              f"{mean_overlap['mouth']:>8.2f}% "
              f"{mean_overlap['face']:>8.2f}%")
    print("="*60)

    print("\nINTERPRETATION:")
    print("- High overlap (>50%) indicates spatially coherent attributions")
    print("- Low overlap (<20%) suggests scattered, incoherent attributions")
    print("- Random baseline should be ~5.3% (uniform across 19 classes)")
    print("\nHypothesis: Low FR methods (Grad-CAM) should have high overlap")
    print("            High FR methods (SHAP, LIME) should have low overlap")


if __name__ == '__main__':
    # If run without arguments, show help
    if len(sys.argv) == 1:
        print(__doc__)
        print("\nDataset ready for regional attribution analysis!")
        print("="*60)
        print("\nQuick test:")
        print("  python experiments/run_regional_attribution.py --n-samples 10")
        print("\nFull analysis:")
        print("  python experiments/run_regional_attribution.py --n-samples 500")
        print("\nNext steps:")
        print("  1. Integrate with actual attribution methods (Grad-CAM, SHAP, LIME)")
        print("  2. Load trained face recognition model")
        print("  3. Run analysis and compare FR with regional consistency")
        print("  4. Add to Chapter 6: 'Regional Attribution Validation'")
        print("="*60)
    else:
        main()
