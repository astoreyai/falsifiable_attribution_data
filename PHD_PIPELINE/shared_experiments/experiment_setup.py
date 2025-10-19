#!/usr/bin/env python3
"""
Experimental Setup for Articles A & B Validation

Purpose: Minimal but decisive experiments to validate falsifiable attribution framework
Timeline: Weeks 6-8
Hardware: NVIDIA RTX 3090 (24 GB VRAM) or equivalent

This is a SKELETON implementation with extensive comments.
Fill in the stubs marked with "TODO: IMPLEMENT" during execution phase.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

# Third-party libraries (install via requirements.txt)
try:
    from captum.attr import LayerGradCam, IntegratedGradients, KernelShap
    import lpips
    from scipy.stats import pearsonr, ttest_rel
    from scipy.stats import bootstrap as scipy_bootstrap
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install dependencies: pip install -r requirements.txt")
    raise


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    Centralized configuration for experiments.
    All hyperparameters and thresholds defined here.
    """
    # Dataset
    dataset_name: str = "LFW"  # Labeled Faces in the Wild
    dataset_path: str = "./data/lfw"
    num_pairs: int = 200  # 100 matched + 100 mismatched

    # Model
    model_name: str = "ArcFace-ResNet50"
    model_weights_path: str = "./models/ms1mv3_arcface_r50_fp16.pth"
    embedding_dim: int = 512
    input_size: Tuple[int, int, int] = (112, 112, 3)

    # Attribution methods
    attribution_methods: List[str] = None  # ["GradCAM", "IntegratedGradients", "SHAP"]

    def __post_init__(self):
        if self.attribution_methods is None:
            # Default: Use Grad-CAM and Integrated Gradients (minimal but decisive)
            self.attribution_methods = ["GradCAM", "IntegratedGradients"]

    # Feature classification thresholds (from Chapter 4, Section 4.3.3)
    theta_high: float = 0.7  # 70th percentile (high-attribution threshold)
    theta_low: float = 0.4   # 40th percentile (low-attribution threshold)

    # Counterfactual generation (from Chapter 4, Section 4.4)
    delta_target: float = 0.8  # Target geodesic distance (radians)
    K: int = 10  # Counterfactuals per feature set (S_high, S_low)
    max_iterations: int = 100
    learning_rate: float = 0.01
    lambda_reg: float = 0.1  # Proximity loss weight
    tolerance: float = 0.01  # Convergence tolerance (radians)

    # Plausibility gate thresholds (pre-registered, from Article B)
    lpips_threshold: float = 0.3   # Perceptual similarity
    fid_threshold: float = 50.0    # Fréchet Inception Distance
    l2_max: float = 0.5            # Maximum L2 perturbation
    intensity_min: float = 0.1     # Minimum mean intensity
    intensity_max: float = 0.9     # Maximum mean intensity

    # Statistical thresholds (pre-registered, from Article B)
    correlation_floor: float = 0.7    # Minimum correlation for NOT FALSIFIED
    ci_coverage_min: float = 0.90     # Minimum CI calibration
    ci_coverage_max: float = 1.00     # Maximum CI calibration
    separation_margin: float = 0.15   # Minimum separation (radians)

    # Computation
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 16
    num_workers: int = 4
    seed: int = 42

    # Output paths
    output_dir: str = "./results"
    figures_dir: str = "./figures"
    log_file: str = "./experiment.log"


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(config: ExperimentConfig):
    """Configure logging to console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(config: ExperimentConfig):
    """
    Load LFW dataset with pre-defined verification pairs.

    Returns:
        pairs: List of tuples (img1, img2, label) where label ∈ {0, 1}
               0 = mismatched pair, 1 = matched pair
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading {config.dataset_name} from {config.dataset_path}")

    # TODO: IMPLEMENT
    # 1. Download LFW if not present: http://vis-www.cs.umass.edu/lfw/
    # 2. Load pairs.txt (6,000 verification pairs)
    # 3. Sample config.num_pairs pairs (stratified: 50% matched, 50% mismatched)
    # 4. Load images and preprocess:
    #    - Resize to 112x112
    #    - Normalize to [0, 1]
    #    - Convert to RGB
    # 5. Return list of (img1_tensor, img2_tensor, label)

    # STUB: Return dummy data for now
    pairs = []
    logger.warning("STUB: load_dataset() not implemented. Returning empty list.")
    return pairs


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(config: ExperimentConfig):
    """
    Load pretrained ArcFace model with ResNet-50 backbone.

    Returns:
        model: PyTorch model that maps 112x112x3 images to 512-D unit-norm embeddings
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading {config.model_name} from {config.model_weights_path}")

    # TODO: IMPLEMENT
    # 1. Download pretrained weights from InsightFace:
    #    https://github.com/deepinsight/insightface
    #    Model: ms1mv3_arcface_r50_fp16
    # 2. Load model architecture (ResNet-50 backbone + ArcFace head)
    # 3. Load pretrained weights
    # 4. Set model to eval mode: model.eval()
    # 5. Move to device: model.to(config.device)
    # 6. Verify embedding is L2-normalized (unit hypersphere)

    # STUB: Return dummy model
    class DummyModel(nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            # Return random unit-norm embeddings
            emb = torch.randn(batch_size, config.embedding_dim, device=x.device)
            return emb / torch.norm(emb, dim=1, keepdim=True)

    model = DummyModel()
    logger.warning("STUB: load_model() not implemented. Returning dummy model.")
    return model


# =============================================================================
# ATTRIBUTION GENERATION
# =============================================================================

class AttributionGenerator:
    """
    Wrapper for attribution methods (Grad-CAM, Integrated Gradients, SHAP).
    """

    def __init__(self, model: nn.Module, method_name: str, config: ExperimentConfig):
        self.model = model
        self.method_name = method_name
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize attribution method
        if method_name == "GradCAM":
            # TODO: IMPLEMENT
            # 1. Identify target layer (e.g., model.layer4 for ResNet-50)
            # 2. Initialize Captum LayerGradCam
            # self.method = LayerGradCam(model, target_layer)
            self.method = None
            self.logger.warning("STUB: GradCAM not implemented")

        elif method_name == "IntegratedGradients":
            # TODO: IMPLEMENT
            # 1. Initialize Captum IntegratedGradients
            # 2. Set baseline (black image)
            # 3. Set number of interpolation steps (50)
            # self.method = IntegratedGradients(model)
            self.method = None
            self.logger.warning("STUB: IntegratedGradients not implemented")

        elif method_name == "SHAP":
            # TODO: IMPLEMENT (OPTIONAL - computationally expensive)
            # 1. Initialize Captum KernelShap
            # 2. Set number of samples (1000)
            # 3. Define feature map (superpixels)
            # self.method = KernelShap(model)
            self.method = None
            self.logger.warning("STUB: SHAP not implemented")

        else:
            raise ValueError(f"Unknown attribution method: {method_name}")

    def generate_attribution(self, img1: torch.Tensor, img2: torch.Tensor) -> np.ndarray:
        """
        Generate attribution map for img1 with respect to similarity to img2.

        Args:
            img1: Input image (1, 3, 112, 112)
            img2: Reference image (1, 3, 112, 112)

        Returns:
            attribution: Attribution map (7, 7) for spatial methods or (50,) for superpixel methods
        """
        # TODO: IMPLEMENT
        # 1. Compute embedding for img2: target_emb = model(img2)
        # 2. Run attribution method with img1 as input, target_emb as target
        # 3. Post-process attribution to spatial grid (7x7) or superpixels
        # 4. Return as numpy array

        # STUB: Return random attribution
        if self.method_name in ["GradCAM", "IntegratedGradients"]:
            attribution = np.random.rand(7, 7)  # Spatial 7x7 grid
        else:  # SHAP
            attribution = np.random.rand(50)  # 50 superpixels

        self.logger.debug(f"Generated attribution with shape {attribution.shape}")
        return attribution


def classify_features(attribution: np.ndarray, config: ExperimentConfig) -> Tuple[set, set]:
    """
    Classify features into high-attribution and low-attribution sets.

    Args:
        attribution: Attribution map (7x7 or 50,)
        config: Experiment configuration

    Returns:
        S_high: Set of high-attribution feature indices
        S_low: Set of low-attribution feature indices
    """
    # Flatten attribution to 1D
    attr_flat = attribution.flatten()

    # Absolute values (importance magnitude regardless of sign)
    attr_abs = np.abs(attr_flat)

    # Classify features
    S_high = set(np.where(attr_abs > config.theta_high)[0])
    S_low = set(np.where(attr_abs < config.theta_low)[0])

    logger = logging.getLogger(__name__)
    logger.debug(f"Feature classification: |S_high|={len(S_high)}, |S_low|={len(S_low)}")

    # Non-triviality check
    if len(S_high) == 0 or len(S_low) == 0:
        logger.warning("Non-triviality check failed: empty feature set")

    return S_high, S_low


# =============================================================================
# COUNTERFACTUAL GENERATION
# =============================================================================

def generate_counterfactual(
    x: torch.Tensor,
    model: nn.Module,
    mask_features: set,
    config: ExperimentConfig,
    spatial_resolution: Tuple[int, int] = (7, 7)
) -> Tuple[torch.Tensor, bool, float, int]:
    """
    Generate counterfactual image via gradient descent on unit hypersphere.

    Implements Algorithm 3.1 from Chapter 4, Section 4.4.

    Args:
        x: Original image (1, 3, 112, 112)
        model: Face recognition model
        mask_features: Set of feature indices to preserve (S_high or S_low)
        config: Experiment configuration
        spatial_resolution: Grid resolution for feature mapping (7, 7)

    Returns:
        x_prime: Counterfactual image (1, 3, 112, 112)
        converged: Whether optimization converged
        final_distance: Final geodesic distance achieved
        iterations: Number of iterations used
    """
    logger = logging.getLogger(__name__)

    # TODO: IMPLEMENT
    # 1. Create binary mask M_S:
    #    - Map feature indices to spatial regions (16x16 pixel blocks for 7x7 grid)
    #    - Set M_S[i,j,k] = 1 for pixels in mask_features, 0 otherwise
    # 2. Initialize x_prime = x.clone().detach().requires_grad_(True)
    # 3. Compute original embedding: phi_x = model(x).detach()
    # 4. Optimization loop (t = 1 to max_iterations):
    #    a. Forward pass: phi_x_prime = model(x_prime)
    #    b. Compute geodesic distance: d_g = arccos(dot(phi_x, phi_x_prime))
    #    c. Compute loss: L = (d_g - delta_target)^2 + lambda * ||x_prime - x||^2
    #    d. Backward pass: L.backward()
    #    e. Gradient descent update: x_prime = x_prime - alpha * grad
    #    f. Apply mask: x_prime = M_S * x + (1 - M_S) * x_prime
    #    g. Clamp to [0, 1]: x_prime = clamp(x_prime, 0, 1)
    #    h. Early stopping: if |d_g - delta_target| < tolerance: break
    # 5. Return x_prime, converged, final_distance, iterations

    # STUB: Return original image with small noise
    x_prime = x + 0.01 * torch.randn_like(x)
    x_prime = torch.clamp(x_prime, 0, 1)

    converged = True
    final_distance = config.delta_target + 0.005  # Close to target
    iterations = 50  # Dummy iteration count

    logger.debug(f"Generated counterfactual: converged={converged}, d_g={final_distance:.4f}, iters={iterations}")

    return x_prime, converged, final_distance, iterations


# =============================================================================
# PLAUSIBILITY GATE
# =============================================================================

class PlausibilityGate:
    """
    Filter unrealistic counterfactuals using perceptual and distributional metrics.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize LPIPS model
        # TODO: IMPLEMENT
        # self.lpips_model = lpips.LPIPS(net='alex').to(config.device)
        self.lpips_model = None
        self.logger.warning("STUB: LPIPS model not initialized")

    def check(self, x: torch.Tensor, x_prime: torch.Tensor) -> Tuple[bool, str]:
        """
        Check if counterfactual passes plausibility gate.

        Args:
            x: Original image (1, 3, 112, 112)
            x_prime: Counterfactual image (1, 3, 112, 112)

        Returns:
            passed: Whether counterfactual passes all checks
            reason: Reason for rejection (if failed)
        """
        # TODO: IMPLEMENT
        # 1. LPIPS check:
        #    lpips_score = self.lpips_model(x, x_prime)
        #    if lpips_score > config.lpips_threshold: return False, "LPIPS too high"
        # 2. FID check (compute over batch for efficiency):
        #    fid_score = compute_fid([x], [x_prime])
        #    if fid_score > config.fid_threshold: return False, "FID too high"
        # 3. L2 norm check:
        #    l2_dist = torch.norm(x_prime - x, p=2)
        #    if l2_dist > config.l2_max: return False, "Perturbation too large"
        # 4. Intensity check:
        #    mean_intensity = torch.mean(x_prime)
        #    if mean_intensity < config.intensity_min or mean_intensity > config.intensity_max:
        #        return False, "Extreme intensity"
        # 5. Return True, "ACCEPTED" if all checks pass

        # STUB: Always accept for now
        passed = True
        reason = "ACCEPTED"

        self.logger.debug(f"Plausibility check: {reason}")
        return passed, reason


# =============================================================================
# DELTA-SCORE MEASUREMENT
# =============================================================================

def compute_geodesic_distance(phi1: torch.Tensor, phi2: torch.Tensor) -> float:
    """
    Compute geodesic distance on unit hypersphere.

    Args:
        phi1: Embedding 1 (1, 512), L2-normalized
        phi2: Embedding 2 (1, 512), L2-normalized

    Returns:
        d_g: Geodesic distance in radians [0, π]
    """
    # Dot product (cosine similarity)
    cos_sim = torch.sum(phi1 * phi2, dim=1)

    # Clamp to [-1, 1] for numerical stability
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)

    # Arccosine
    d_g = torch.acos(cos_sim)

    return d_g.item()


def measure_delta_scores(
    x: torch.Tensor,
    counterfactuals_high: List[torch.Tensor],
    counterfactuals_low: List[torch.Tensor],
    model: nn.Module
) -> Tuple[float, float, List[float], List[float]]:
    """
    Measure realized Δ-scores for high and low attribution counterfactuals.

    Args:
        x: Original image
        counterfactuals_high: List of counterfactuals with S_high masked
        counterfactuals_low: List of counterfactuals with S_low masked
        model: Face recognition model

    Returns:
        d_high_mean: Mean geodesic distance for high-attribution counterfactuals
        d_low_mean: Mean geodesic distance for low-attribution counterfactuals
        d_high_values: Individual distances for high-attribution counterfactuals
        d_low_values: Individual distances for low-attribution counterfactuals
    """
    logger = logging.getLogger(__name__)

    # Compute original embedding
    with torch.no_grad():
        phi_x = model(x)

    # Measure distances for high-attribution counterfactuals
    d_high_values = []
    for x_prime in counterfactuals_high:
        with torch.no_grad():
            phi_x_prime = model(x_prime)
        d_g = compute_geodesic_distance(phi_x, phi_x_prime)
        d_high_values.append(d_g)

    # Measure distances for low-attribution counterfactuals
    d_low_values = []
    for x_prime in counterfactuals_low:
        with torch.no_grad():
            phi_x_prime = model(x_prime)
        d_g = compute_geodesic_distance(phi_x, phi_x_prime)
        d_low_values.append(d_g)

    # Compute means
    d_high_mean = np.mean(d_high_values) if d_high_values else 0.0
    d_low_mean = np.mean(d_low_values) if d_low_values else 0.0

    logger.debug(f"Δ-scores: d_high={d_high_mean:.4f}, d_low={d_low_mean:.4f}")

    return d_high_mean, d_low_mean, d_high_values, d_low_values


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_correlation(
    predictions: List[Tuple[float, float]],
    realizations: List[Tuple[float, float]]
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Compute correlation between predicted and realized Δ-scores.

    Args:
        predictions: List of (Δ_pred_high, Δ_pred_low) tuples
        realizations: List of (d_high_mean, d_low_mean) tuples

    Returns:
        rho: Pearson correlation coefficient
        p_value: Statistical significance
        ci: 95% confidence interval (lower, upper)
    """
    logger = logging.getLogger(__name__)

    # Flatten predictions and realizations
    predicted = np.array([p[0] for p in predictions] + [p[1] for p in predictions])
    realized = np.array([r[0] for r in realizations] + [r[1] for r in realizations])

    # Compute correlation
    rho, p_value = pearsonr(predicted, realized)

    # Bootstrap 95% CI
    def compute_corr(x, y):
        return pearsonr(x, y)[0]

    # TODO: IMPLEMENT bootstrap (requires scipy >= 1.11)
    # res = scipy_bootstrap((predicted, realized), compute_corr, n_resamples=10000, confidence_level=0.95)
    # ci = (res.confidence_interval.low, res.confidence_interval.high)

    # STUB: Use simple approximation
    ci = (rho - 0.05, rho + 0.05)

    logger.info(f"Correlation: ρ={rho:.3f}, p={p_value:.4f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")

    return rho, p_value, ci


def compute_separation(
    realizations: List[Tuple[float, float]]
) -> Tuple[float, float, float]:
    """
    Compute separation margin between high and low attribution distances.

    Args:
        realizations: List of (d_high_mean, d_low_mean) tuples

    Returns:
        separation: Mean separation (d_high - d_low)
        t_stat: t-statistic from paired t-test
        p_value: Statistical significance
    """
    logger = logging.getLogger(__name__)

    d_high_values = [r[0] for r in realizations]
    d_low_values = [r[1] for r in realizations]

    # Mean separation
    separation = np.mean(d_high_values) - np.mean(d_low_values)

    # Paired t-test
    t_stat, p_value = ttest_rel(d_high_values, d_low_values)

    logger.info(f"Separation: Δ={separation:.4f}, t={t_stat:.3f}, p={p_value:.4f}")

    return separation, t_stat, p_value


def make_falsification_decision(
    rho: float,
    p_value: float,
    ci_calibration: float,
    separation: float,
    config: ExperimentConfig
) -> str:
    """
    Make final falsification decision based on pre-registered thresholds.

    Args:
        rho: Correlation coefficient
        p_value: Statistical significance
        ci_calibration: CI calibration (actual coverage)
        separation: Separation margin (d_high - d_low)
        config: Experiment configuration

    Returns:
        verdict: "NOT FALSIFIED" or "FALSIFIED"
    """
    logger = logging.getLogger(__name__)

    # Check thresholds
    correlation_pass = (rho > config.correlation_floor) and (p_value < 0.05)
    calibration_pass = (config.ci_coverage_min <= ci_calibration <= config.ci_coverage_max)
    separation_pass = (separation > config.separation_margin)

    # Final verdict
    if correlation_pass and calibration_pass and separation_pass:
        verdict = "NOT FALSIFIED"
    else:
        verdict = "FALSIFIED"
        reasons = []
        if not correlation_pass:
            reasons.append(f"correlation ρ={rho:.3f} ≤ {config.correlation_floor}")
        if not calibration_pass:
            reasons.append(f"calibration {ci_calibration:.2%} outside [{config.ci_coverage_min:.2%}, {config.ci_coverage_max:.2%}]")
        if not separation_pass:
            reasons.append(f"separation {separation:.4f} ≤ {config.separation_margin}")
        logger.info(f"FALSIFIED: {', '.join(reasons)}")

    logger.info(f"Final verdict: {verdict}")

    return verdict


# =============================================================================
# MAIN EXPERIMENTAL LOOP
# =============================================================================

def run_experiment(config: ExperimentConfig):
    """
    Run full experimental validation for Articles A & B.
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Starting experimental validation")
    logger.info("=" * 80)

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load dataset and model
    pairs = load_dataset(config)
    model = load_model(config)

    # Initialize plausibility gate
    plausibility_gate = PlausibilityGate(config)

    # Store results
    results = {method_name: {
        "predictions": [],
        "realizations": [],
        "rejection_counts": 0,
        "convergence_failures": 0
    } for method_name in config.attribution_methods}

    # Loop over attribution methods
    for method_name in config.attribution_methods:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Testing attribution method: {method_name}")
        logger.info(f"{'=' * 80}\n")

        # Initialize attribution generator
        attr_generator = AttributionGenerator(model, method_name, config)

        # Loop over image pairs
        for i, (img1, img2, label) in enumerate(pairs):
            logger.info(f"Processing pair {i+1}/{len(pairs)} (label={label})")

            # Move to device
            img1 = img1.to(config.device)
            img2 = img2.to(config.device)

            # Generate attribution
            attribution = attr_generator.generate_attribution(img1, img2)

            # Classify features
            S_high, S_low = classify_features(attribution, config)

            # Non-triviality check
            if len(S_high) == 0 or len(S_low) == 0:
                logger.warning(f"Pair {i+1}: Non-triviality check failed. Skipping.")
                continue

            # Generate counterfactuals for S_high
            counterfactuals_high = []
            for k in range(config.K):
                x_prime, converged, final_dist, iters = generate_counterfactual(
                    img1, model, S_high, config
                )
                if not converged:
                    results[method_name]["convergence_failures"] += 1
                    logger.debug(f"Counterfactual {k+1} (S_high) failed to converge")
                    continue

                # Apply plausibility gate
                passed, reason = plausibility_gate.check(img1, x_prime)
                if passed:
                    counterfactuals_high.append(x_prime)
                else:
                    results[method_name]["rejection_counts"] += 1
                    logger.debug(f"Counterfactual {k+1} (S_high) rejected: {reason}")

            # Generate counterfactuals for S_low
            counterfactuals_low = []
            for k in range(config.K):
                x_prime, converged, final_dist, iters = generate_counterfactual(
                    img1, model, S_low, config
                )
                if not converged:
                    results[method_name]["convergence_failures"] += 1
                    logger.debug(f"Counterfactual {k+1} (S_low) failed to converge")
                    continue

                # Apply plausibility gate
                passed, reason = plausibility_gate.check(img1, x_prime)
                if passed:
                    counterfactuals_low.append(x_prime)
                else:
                    results[method_name]["rejection_counts"] += 1
                    logger.debug(f"Counterfactual {k+1} (S_low) rejected: {reason}")

            # Check if we have enough counterfactuals
            if len(counterfactuals_high) < 5 or len(counterfactuals_low) < 5:
                logger.warning(f"Pair {i+1}: Insufficient counterfactuals after filtering. Skipping.")
                continue

            # Measure Δ-scores
            d_high_mean, d_low_mean, d_high_vals, d_low_vals = measure_delta_scores(
                img1, counterfactuals_high, counterfactuals_low, model
            )

            # Store predictions and realizations
            # TODO: IMPLEMENT predicted Δ-scores based on attribution magnitudes
            # For now, use placeholder: predicted = sum of attribution magnitudes
            Δ_pred_high = np.sum(attribution[list(S_high)])
            Δ_pred_low = np.sum(attribution[list(S_low)])

            results[method_name]["predictions"].append((Δ_pred_high, Δ_pred_low))
            results[method_name]["realizations"].append((d_high_mean, d_low_mean))

            logger.info(f"Pair {i+1}: Δ_pred=(high:{Δ_pred_high:.3f}, low:{Δ_pred_low:.3f}), "
                       f"Δ_real=(high:{d_high_mean:.3f}, low:{d_low_mean:.3f})")

        # Compute statistics for this method
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Results for {method_name}")
        logger.info(f"{'=' * 80}\n")

        predictions = results[method_name]["predictions"]
        realizations = results[method_name]["realizations"]

        if len(predictions) < 10:
            logger.warning(f"Insufficient data for {method_name} ({len(predictions)} pairs). Skipping statistics.")
            continue

        # Correlation
        rho, p_value, ci = compute_correlation(predictions, realizations)

        # Separation
        separation, t_stat, sep_p_value = compute_separation(realizations)

        # CI calibration (TODO: IMPLEMENT)
        ci_calibration = 0.95  # STUB

        # Falsification decision
        verdict = make_falsification_decision(rho, p_value, ci_calibration, separation, config)

        # Store results
        results[method_name]["statistics"] = {
            "correlation": rho,
            "p_value": p_value,
            "ci": ci,
            "separation": separation,
            "ci_calibration": ci_calibration,
            "verdict": verdict
        }

        logger.info(f"Rejection rate: {results[method_name]['rejection_counts'] / (2 * config.K * len(pairs)):.2%}")
        logger.info(f"Convergence failures: {results[method_name]['convergence_failures']}")

    # Save results
    save_results(results, config)

    # Generate visualizations
    generate_visualizations(results, config)

    logger.info("\n" + "=" * 80)
    logger.info("Experimental validation complete")
    logger.info("=" * 80)


# =============================================================================
# RESULTS SAVING AND VISUALIZATION
# =============================================================================

def save_results(results: Dict, config: ExperimentConfig):
    """
    Save experimental results to disk.
    """
    logger = logging.getLogger(__name__)

    # TODO: IMPLEMENT
    # 1. Save predictions and realizations to CSV
    # 2. Save statistics to JSON
    # 3. Save verdicts to summary file

    logger.info(f"Results saved to {config.output_dir}")


def generate_visualizations(results: Dict, config: ExperimentConfig):
    """
    Generate all figures for Articles A & B.
    """
    logger = logging.getLogger(__name__)

    # TODO: IMPLEMENT
    # 1. Scatter plot: predicted vs realized Δ-score (Article A, Figure 4)
    # 2. Plausibility gate visualization (Article A, Figure 5)
    # 3. Calibration plot (Article B, Figure 3)
    # 4. Example reports (Article B, Figure 4)
    # 5. Results table (Article B, Table 2)

    logger.info(f"Figures saved to {config.figures_dir}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Create configuration
    config = ExperimentConfig()

    # Setup logging
    logger = setup_logging(config)

    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.figures_dir).mkdir(parents=True, exist_ok=True)

    # Run experiment
    try:
        run_experiment(config)
    except Exception as e:
        logger.exception("Experiment failed with exception:")
        raise

    logger.info("Experiment complete. Check results in {}".format(config.output_dir))
