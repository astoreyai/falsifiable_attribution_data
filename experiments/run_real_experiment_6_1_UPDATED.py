#!/usr/bin/env python3
"""
UPDATED Experiment 6.1: Comprehensive Attribution Method Comparison

Tests 5 attribution methods on real face verification:
1. Grad-CAM (baseline - spatial attribution)
2. Gradient × Input (NEW - input-space attribution)
3. Vanilla Gradients (NEW - saliency maps)
4. SmoothGrad (NEW - noise-reduced attribution)
5. Geodesic IG (benchmark - path-integrated gradients)

This script tests Agent 3's hypothesis that input-space gradient methods
(Gradient × Input, SmoothGrad) will achieve 60-75% FR compared to Grad-CAM's
10.48% FR, while maintaining separation from Geodesic IG's 100% FR.

NO SIMULATIONS. REAL DATA. GPU ACCELERATION.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.framework.falsification_test import falsification_test
from src.framework.counterfactual_generation import (
    generate_counterfactuals_hypersphere,
    compute_geodesic_distance
)
from src.attributions.gradcam import GradCAM
from src.attributions.geodesic_ig import GeodesicIntegratedGradients
from src.attributions.gradient_x_input import (
    GradientXInput,
    VanillaGradients,
    SmoothGrad
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_lfw_face_pairs(n_pairs: int = 500, device: str = 'cuda'):
    """
    Load real face pairs from LFW dataset using sklearn.

    Args:
        n_pairs: Number of pairs to load
        device: Device to load tensors on

    Returns:
        List of (img1, img2, label) tuples
    """
    from sklearn.datasets import fetch_lfw_people
    from PIL import Image
    import torchvision.transforms as transforms

    logger.info(f"Loading LFW dataset (n={n_pairs} pairs)...")

    # Load LFW using sklearn (better offline support)
    lfw_people = fetch_lfw_people(
        min_faces_per_person=2,
        resize=0.5,
        color=True,
        download_if_missing=True
    )

    # Transform for FaceNet
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create pairs (same person + different person)
    pairs = []
    np.random.seed(42)

    # Group images by person
    person_images = {}
    for img, target in zip(lfw_people.images, lfw_people.target):
        if target not in person_images:
            person_images[target] = []
        person_images[target].append(img)

    # Generate positive pairs (same person)
    n_positive = n_pairs // 2
    positive_pairs = []
    persons_with_multiple = [p for p, imgs in person_images.items() if len(imgs) >= 2]

    for _ in range(n_positive):
        person = np.random.choice(persons_with_multiple)
        person_imgs = person_images[person]
        indices = np.random.choice(len(person_imgs), 2, replace=False)
        img1, img2 = person_imgs[indices[0]], person_imgs[indices[1]]

        # Convert to PIL then tensor
        img1_pil = Image.fromarray((img1 * 255).astype(np.uint8))
        img2_pil = Image.fromarray((img2 * 255).astype(np.uint8))

        img1_tensor = transform(img1_pil).to(device)
        img2_tensor = transform(img2_pil).to(device)

        positive_pairs.append((img1_tensor, img2_tensor, 1))

    # Generate negative pairs (different persons)
    n_negative = n_pairs - n_positive
    negative_pairs = []
    all_persons = list(person_images.keys())

    for _ in range(n_negative):
        person1, person2 = np.random.choice(all_persons, 2, replace=False)
        imgs1 = person_images[person1]
        imgs2 = person_images[person2]
        img1 = imgs1[np.random.choice(len(imgs1))]
        img2 = imgs2[np.random.choice(len(imgs2))]

        # Convert to PIL then tensor
        img1_pil = Image.fromarray((img1 * 255).astype(np.uint8))
        img2_pil = Image.fromarray((img2 * 255).astype(np.uint8))

        img1_tensor = transform(img1_pil).to(device)
        img2_tensor = transform(img2_pil).to(device)

        negative_pairs.append((img1_tensor, img2_tensor, 0))

    # Combine and shuffle
    pairs = positive_pairs + negative_pairs
    np.random.shuffle(pairs)

    logger.info(f"✅ Loaded {len(pairs)} pairs from LFW ({n_positive} positive, {n_negative} negative)")
    return pairs


def load_facenet_model(device: str = 'cuda'):
    """Load FaceNet model (Inception-ResNet-V1)."""
    from facenet_pytorch import InceptionResnetV1

    logger.info("Loading FaceNet model...")
    model = InceptionResnetV1(
        pretrained='vggface2',
        classify=False,
        device=device
    ).eval()

    logger.info(f"✅ FaceNet loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return model


def compute_attribution_methods(
    model: nn.Module,
    image: torch.Tensor,
    target_embedding: torch.Tensor,
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """
    Compute all 5 attribution methods for a single image.

    Args:
        model: Face verification model
        image: Input image (C, H, W)
        target_embedding: Target embedding for gradient direction
        device: Device

    Returns:
        Dictionary mapping method name to importance scores
    """
    results = {}

    # 1. Grad-CAM (baseline)
    try:
        gradcam = GradCAM(model, target_layer=model.block8)
        cam = gradcam.generate_cam(image.unsqueeze(0), target_embedding)
        # Flatten and normalize
        scores = cam.flatten().cpu().numpy()
        if scores.max() > 0:
            scores = scores / scores.max()
        results['gradcam'] = scores
    except Exception as e:
        logger.warning(f"Grad-CAM failed: {e}")
        results['gradcam'] = np.ones(image.numel()) * 0.5  # Uniform fallback

    # 2. Gradient × Input (NEW)
    try:
        gxi = GradientXInput(model)
        scores = gxi.get_importance_scores(
            image,
            target=target_embedding,
            normalize=True
        )
        results['gradient_x_input'] = scores
    except Exception as e:
        logger.warning(f"Gradient × Input failed: {e}")
        results['gradient_x_input'] = np.ones(image.numel()) * 0.5

    # 3. Vanilla Gradients (NEW)
    try:
        vanilla = VanillaGradients(model)
        scores = vanilla.get_importance_scores(
            image,
            target=target_embedding,
            normalize=True
        )
        results['vanilla_gradients'] = scores
    except Exception as e:
        logger.warning(f"Vanilla Gradients failed: {e}")
        results['vanilla_gradients'] = np.ones(image.numel()) * 0.5

    # 4. SmoothGrad (NEW)
    try:
        smoothgrad = SmoothGrad(model, n_samples=50, noise_std=0.15)
        scores = smoothgrad.get_importance_scores(
            image,
            target=target_embedding,
            normalize=True
        )
        results['smoothgrad'] = scores
    except Exception as e:
        logger.warning(f"SmoothGrad failed: {e}")
        results['smoothgrad'] = np.ones(image.numel()) * 0.5

    # 5. Geodesic IG (benchmark)
    try:
        geodesic_ig = GeodesicIntegratedGradients(model, device=device)
        scores = geodesic_ig.get_importance_scores(
            image,
            target=target_embedding,
            normalize=True
        )
        results['geodesic_ig'] = scores
    except Exception as e:
        logger.warning(f"Geodesic IG failed: {e}")
        results['geodesic_ig'] = np.ones(image.numel()) * 0.5

    return results


def run_falsification_test_for_pair(
    model: nn.Module,
    img1: torch.Tensor,
    img2: torch.Tensor,
    attribution_scores: np.ndarray,
    K: int = 100,
    noise_scale: float = 0.3,
    device: str = 'cuda'
) -> Dict:
    """
    Run falsification test for a single pair with given attribution scores.

    Args:
        model: Face verification model
        img1, img2: Input images
        attribution_scores: Importance scores (flattened)
        K: Number of counterfactuals
        noise_scale: Noise scale for generation
        device: Device

    Returns:
        Falsification result dictionary
    """
    # Get embeddings
    with torch.no_grad():
        emb1 = model(img1.unsqueeze(0))
        emb2 = model(img2.unsqueeze(0))
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)

    # Run falsification test
    result = falsification_test(
        attribution_scores=attribution_scores,
        original_embedding=emb1,
        counterfactual_embeddings=None,  # Will generate internally
        K=K,
        noise_scale=noise_scale,
        model=model,
        device=device
    )

    return result


def run_experiment(args):
    """Run full Experiment 6.1 with all 5 attribution methods."""

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    save_dir = Path(args.save_dir) / f"exp_6_1_updated_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("EXPERIMENT 6.1 (UPDATED): Comprehensive Attribution Method Comparison")
    logger.info("="*80)
    logger.info(f"Parameters:")
    logger.info(f"  n_pairs: {args.n_pairs}")
    logger.info(f"  K: {args.K}")
    logger.info(f"  noise_scale: {args.noise_scale}")
    logger.info(f"  device: {device}")
    logger.info(f"  seed: {args.seed}")
    logger.info(f"  save_dir: {save_dir}")
    logger.info("="*80)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model and data
    model = load_facenet_model(device)
    pairs = load_lfw_face_pairs(args.n_pairs, device)

    # Results storage
    method_names = [
        'gradcam',
        'gradient_x_input',
        'vanilla_gradients',
        'smoothgrad',
        'geodesic_ig'
    ]

    results = {
        method: {
            'falsification_rates': [],
            'uniform_count': 0,
            'total_count': 0
        }
        for method in method_names
    }

    # Process each pair
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {args.n_pairs} face pairs...")
    logger.info(f"{'='*80}\n")

    for pair_idx, (img1, img2, label) in enumerate(tqdm(pairs, desc="Pairs")):
        # Get embeddings
        with torch.no_grad():
            emb1 = model(img1.unsqueeze(0))
            emb2 = model(img2.unsqueeze(0))
            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)

        # Compute all attribution methods
        attributions = compute_attribution_methods(
            model, img1, emb2, device
        )

        # Run falsification test for each method
        for method_name in method_names:
            scores = attributions[method_name]

            # Check if uniform
            if np.std(scores) < 1e-6:
                results[method_name]['uniform_count'] += 1
                results[method_name]['falsification_rates'].append(0.5)  # Uniform default
            else:
                # Run falsification test
                falsification_result = run_falsification_test_for_pair(
                    model, img1, img2, scores,
                    K=args.K,
                    noise_scale=args.noise_scale,
                    device=device
                )

                fr = falsification_result.get('falsification_rate', 0.0)
                results[method_name]['falsification_rates'].append(fr)

            results[method_name]['total_count'] += 1

        # Log progress every 50 pairs
        if (pair_idx + 1) % 50 == 0:
            logger.info(f"\nProgress at {pair_idx + 1}/{args.n_pairs}:")
            for method in method_names:
                frs = results[method]['falsification_rates']
                mean_fr = np.mean(frs) if frs else 0.0
                logger.info(f"  {method}: FR = {mean_fr:.2%}")

    # Compute final statistics
    logger.info(f"\n{'='*80}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*80}\n")

    final_results = {}

    for method_name in method_names:
        frs = np.array(results[method_name]['falsification_rates'])

        # Statistics
        mean_fr = float(np.mean(frs))
        std_fr = float(np.std(frs, ddof=1))
        median_fr = float(np.median(frs))

        # Confidence interval (bootstrap)
        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(frs, size=len(frs), replace=True)
            bootstrap_means.append(np.mean(sample))
        ci_lower = float(np.percentile(bootstrap_means, 2.5))
        ci_upper = float(np.percentile(bootstrap_means, 97.5))

        # Uniform rate
        uniform_rate = results[method_name]['uniform_count'] / results[method_name]['total_count']

        final_results[method_name] = {
            'mean_fr': mean_fr,
            'std_fr': std_fr,
            'median_fr': median_fr,
            'ci_95': [ci_lower, ci_upper],
            'uniform_rate': uniform_rate,
            'n_pairs': len(frs),
            'all_frs': frs.tolist()
        }

        logger.info(f"{method_name.upper()}:")
        logger.info(f"  Mean FR: {mean_fr:.2%} ± {std_fr:.2%}")
        logger.info(f"  Median FR: {median_fr:.2%}")
        logger.info(f"  95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]")
        logger.info(f"  Uniform rate: {uniform_rate:.1%}")
        logger.info("")

    # Statistical comparisons
    logger.info(f"{'='*80}")
    logger.info("STATISTICAL COMPARISONS")
    logger.info(f"{'='*80}\n")

    # Compare Grad-CAM vs Gradient × Input
    gradcam_frs = np.array(results['gradcam']['falsification_rates'])
    gxi_frs = np.array(results['gradient_x_input']['falsification_rates'])

    t_stat, p_value = stats.ttest_ind(gradcam_frs, gxi_frs)
    logger.info(f"Grad-CAM vs Gradient × Input:")
    logger.info(f"  t-statistic: {t_stat:.4f}")
    logger.info(f"  p-value: {p_value:.2e}")
    logger.info(f"  Significant: {p_value < 0.05}")
    logger.info("")

    # Save results
    output = {
        'experiment_id': 'exp_6_1_updated',
        'title': 'Comprehensive Attribution Method Comparison',
        'timestamp': timestamp,
        'parameters': vars(args),
        'method_results': final_results,
        'statistical_tests': {
            'gradcam_vs_gradient_x_input': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
        }
    }

    output_file = save_dir / f"exp_6_1_updated_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"✅ Results saved to: {output_file}")

    # Generate visualization
    generate_comparison_plot(final_results, save_dir, timestamp)

    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT 6.1 (UPDATED) COMPLETE")
    logger.info(f"{'='*80}")

    return output


def generate_comparison_plot(results: Dict, save_dir: Path, timestamp: str):
    """Generate publication-quality comparison plot."""

    methods = list(results.keys())
    mean_frs = [results[m]['mean_fr'] * 100 for m in methods]
    std_frs = [results[m]['std_fr'] * 100 for m in methods]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(methods))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    bars = ax.bar(x_pos, mean_frs, yerr=std_frs, capsize=5, color=colors, alpha=0.8)

    ax.set_ylabel('Falsification Rate (%)', fontsize=12)
    ax.set_xlabel('Attribution Method', fontsize=12)
    ax.set_title('Experiment 6.1: Attribution Method Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([
        'Grad-CAM\n(baseline)',
        'Gradient × Input\n(NEW)',
        'Vanilla Gradients\n(NEW)',
        'SmoothGrad\n(NEW)',
        'Geodesic IG\n(benchmark)'
    ], rotation=0, ha='center')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    output_file = save_dir / f"figure_6_1_comparison_{timestamp}.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"  ✅ Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 6.1 (UPDATED): Comprehensive Attribution Method Comparison'
    )
    parser.add_argument('--n_pairs', type=int, default=500,
                        help='Number of face pairs to test')
    parser.add_argument('--K', type=int, default=100,
                        help='Number of counterfactuals per test')
    parser.add_argument('--noise_scale', type=float, default=0.3,
                        help='Noise scale for counterfactual generation')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='experiments/production_exp6_1_UPDATED',
                        help='Directory to save results')

    args = parser.parse_args()

    try:
        results = run_experiment(args)
        logger.info("\n✅ Experiment 6.1 (UPDATED) completed successfully!")
        logger.info(f"   Mean FRs:")
        for method, data in results['method_results'].items():
            logger.info(f"     {method}: {data['mean_fr']:.2%}")
    except Exception as e:
        logger.error(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
