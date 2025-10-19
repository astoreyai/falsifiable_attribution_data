"""
SHAP Attribution for Face Verification

Implements SHAP (SHapley Additive exPlanations) using KernelExplainer
with superpixel segmentation for face verification models.

Reference: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
https://arxiv.org/abs/1705.07874

Implementation Strategy:
- Uses KernelSHAP (model-agnostic, black-box compatible)
- Segments face images into superpixels using SLIC algorithm
- Computes Shapley values for each superpixel region
- Maps back to pixel-level attribution

Adaptation for Metric Learning:
- Standard SHAP explains class probabilities
- Our version explains cosine similarity (face verification score)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable, Union
import warnings


class SHAPAttribution:
    """
    SHAP attribution for face verification using KernelExplainer.

    This is a baseline method used for comparison in Experiment 6.1.

    Key Features:
    - Model-agnostic (black-box compatible)
    - Uses superpixel segmentation (SLIC algorithm)
    - Computes Shapley values for interpretable regions
    - Adapted for metric learning (cosine similarity target)

    Limitations:
    - Slow (requires many model evaluations)
    - Designed for classification (adapted for embeddings)
    - Superpixel granularity may miss fine details
    """

    def __init__(
        self,
        model: Callable,
        n_superpixels: int = 50,
        n_samples: int = 100,
        device: str = 'cuda'
    ):
        """
        Initialize SHAP attribution.

        Args:
            model: Face verification model (embedding extractor)
            n_superpixels: Number of superpixels for segmentation
                          More = finer granularity but slower
            n_samples: Number of samples for KernelSHAP estimation
                      More = more accurate but slower
            device: Device for computation ('cuda' or 'cpu')
        """
        self.model = model
        self.n_superpixels = n_superpixels
        self.n_samples = n_samples
        self.device = device

        # Try to import SHAP library
        try:
            import shap
            self.shap = shap
            self.shap_available = True
        except ImportError:
            warnings.warn(
                "SHAP library not installed. Using simplified implementation. "
                "For full functionality, install: pip install shap"
            )
            self.shap = None
            self.shap_available = False

        # Try to import skimage for superpixels
        try:
            from skimage.segmentation import slic
            from skimage.util import img_as_float
            self.slic = slic
            self.img_as_float = img_as_float
            self.superpixel_available = True
        except ImportError:
            warnings.warn(
                "scikit-image not installed. Using simplified implementation. "
                "For full functionality, install: pip install scikit-image"
            )
            self.slic = None
            self.img_as_float = None
            self.superpixel_available = False

    def _segment_superpixels(self, image: np.ndarray) -> np.ndarray:
        """
        Segment image into superpixels using SLIC.

        Args:
            image: Input image (H, W, C) in [0, 1]

        Returns:
            Segmentation mask (H, W) with superpixel labels [0, n_superpixels-1]
        """
        if not self.superpixel_available:
            # Fallback: grid-based segmentation
            H, W, _ = image.shape
            grid_h = int(np.sqrt(self.n_superpixels * H / W))
            grid_w = int(np.sqrt(self.n_superpixels * W / H))

            segments = np.zeros((H, W), dtype=np.int32)
            h_step = H // grid_h
            w_step = W // grid_w

            label = 0
            for i in range(grid_h):
                for j in range(grid_w):
                    h_start = i * h_step
                    h_end = (i + 1) * h_step if i < grid_h - 1 else H
                    w_start = j * w_step
                    w_end = (j + 1) * w_step if j < grid_w - 1 else W
                    segments[h_start:h_end, w_start:w_end] = label
                    label += 1

            return segments

        # Real implementation using SLIC
        image_float = self.img_as_float(image)
        segments = self.slic(
            image_float,
            n_segments=self.n_superpixels,
            compactness=10,
            sigma=1,
            start_label=0
        )
        return segments

    def _create_perturbed_image(
        self,
        original_image: np.ndarray,
        segments: np.ndarray,
        mask: np.ndarray,
        baseline: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create perturbed image by masking out superpixels.

        Args:
            original_image: Original image (H, W, C)
            segments: Superpixel segmentation (H, W)
            mask: Binary mask for superpixels (n_superpixels,)
                 1 = keep superpixel, 0 = mask out
            baseline: Baseline image to use for masked regions
                     If None, use black (zeros)

        Returns:
            Perturbed image (H, W, C)
        """
        if baseline is None:
            baseline = np.zeros_like(original_image)

        perturbed = original_image.copy()
        for i, keep in enumerate(mask):
            if not keep:
                # Mask out this superpixel
                perturbed[segments == i] = baseline[segments == i]

        return perturbed

    def _model_predict(
        self,
        image_batch: np.ndarray,
        target_embedding: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Get model predictions for a batch of images.

        Args:
            image_batch: Batch of images (B, H, W, C) in [0, 1]
            target_embedding: Optional target embedding for verification

        Returns:
            Predictions (B,) - cosine similarities or embedding norms
        """
        # Convert to torch tensors
        images_torch = torch.from_numpy(image_batch).float()
        images_torch = images_torch.permute(0, 3, 1, 2)  # (B, C, H, W)
        images_torch = images_torch.to(self.device)

        predictions = []
        with torch.no_grad():
            for img in images_torch:
                img = img.unsqueeze(0)  # Add batch dimension

                # Get embedding
                if hasattr(self.model, 'get_embedding'):
                    emb = self.model.get_embedding(img)
                elif hasattr(self.model, 'model'):
                    emb = self.model.model(img)
                else:
                    emb = self.model(img)

                emb_normalized = F.normalize(emb, p=2, dim=-1)

                # Compute target score
                if target_embedding is not None:
                    # Verification: cosine similarity with target
                    target_normalized = F.normalize(target_embedding, p=2, dim=-1)
                    score = F.cosine_similarity(
                        emb_normalized.view(1, -1),
                        target_normalized.view(1, -1),
                        dim=1
                    )
                else:
                    # Single image: embedding norm
                    score = torch.norm(emb, p=2, dim=-1)

                predictions.append(score.cpu().item())

        return np.array(predictions)

    def _simplified_kernel_shap(
        self,
        image: np.ndarray,
        segments: np.ndarray,
        target_embedding: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Simplified KernelSHAP implementation when shap library not available.

        Uses weighted linear regression to approximate Shapley values.

        Args:
            image: Input image (H, W, C)
            segments: Superpixel segmentation (H, W)
            target_embedding: Optional target embedding

        Returns:
            Shapley values for each superpixel (n_superpixels,)
        """
        n_segments = len(np.unique(segments))

        # Generate random masks (coalition sampling)
        # Sample different combinations of superpixels
        np.random.seed(42)
        masks = np.random.randint(0, 2, size=(self.n_samples, n_segments))

        # Always include all-on and all-off
        masks[0] = np.ones(n_segments)
        masks[1] = np.zeros(n_segments)

        # Create perturbed images for each mask
        perturbed_images = []
        for mask in masks:
            perturbed = self._create_perturbed_image(image, segments, mask)
            perturbed_images.append(perturbed)

        perturbed_images = np.array(perturbed_images)

        # Get predictions for all perturbed images
        predictions = self._model_predict(perturbed_images, target_embedding)

        # Compute Shapley weights using kernel
        # w(z) = (M-1) / (|z| * (M - |z|) * C(M, |z|))
        # where |z| = number of active features, M = total features
        M = n_segments
        weights = np.zeros(self.n_samples)
        for i, mask in enumerate(masks):
            z = np.sum(mask)
            if z == 0 or z == M:
                weights[i] = 1000  # High weight for boundary conditions
            else:
                # Simplified weight (avoiding combinatorial explosion)
                weights[i] = (M - 1) / (z * (M - z))

        # Weighted linear regression: predictions ~ masks
        # Solve: min_phi Σ w_i (f(x_i) - φ_0 - Σ φ_j z_ij)^2
        from sklearn.linear_model import LinearRegression

        reg = LinearRegression()
        reg.fit(masks, predictions, sample_weight=weights)

        # Coefficients are the Shapley values
        shap_values = reg.coef_

        return shap_values

    def compute_shap_values(
        self,
        image: np.ndarray,
        target_embedding: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Compute SHAP values for image using KernelExplainer.

        Args:
            image: Input image (H, W, C) in [0, 1]
            target_embedding: Optional target embedding for verification

        Returns:
            SHAP values for each superpixel (n_superpixels,)
        """
        # Segment into superpixels
        segments = self._segment_superpixels(image)
        n_segments = len(np.unique(segments))

        if self.shap_available:
            # Use official SHAP library with KernelExplainer
            # Define prediction function
            def predict_fn(masks):
                """Prediction function for SHAP explainer."""
                batch_size = masks.shape[0]
                predictions = []

                for mask in masks:
                    perturbed = self._create_perturbed_image(image, segments, mask)
                    pred = self._model_predict(
                        perturbed[np.newaxis, ...],
                        target_embedding
                    )
                    predictions.append(pred[0])

                return np.array(predictions)

            # Create background data (all features off)
            background = np.zeros((1, n_segments))

            # Create explainer
            explainer = self.shap.KernelExplainer(predict_fn, background)

            # Explain (all features on)
            test_sample = np.ones((1, n_segments))

            # Compute SHAP values
            shap_values = explainer.shap_values(
                test_sample,
                nsamples=self.n_samples,
                silent=True
            )

            return shap_values[0]
        else:
            # Use simplified implementation
            return self._simplified_kernel_shap(image, segments, target_embedding)

    def _map_shap_to_pixels(
        self,
        shap_values: np.ndarray,
        segments: np.ndarray
    ) -> np.ndarray:
        """
        Map superpixel SHAP values to pixel-level attribution.

        Args:
            shap_values: SHAP values for each superpixel (n_superpixels,)
            segments: Superpixel segmentation (H, W)

        Returns:
            Pixel-level attribution map (H, W)
        """
        H, W = segments.shape
        attribution_map = np.zeros((H, W), dtype=np.float32)

        for i, shap_val in enumerate(shap_values):
            attribution_map[segments == i] = shap_val

        # Take absolute value (importance regardless of direction)
        attribution_map = np.abs(attribution_map)

        # Normalize to [0, 1]
        attr_min = attribution_map.min()
        attr_max = attribution_map.max()

        if attr_max - attr_min > 1e-8:
            attribution_map = (attribution_map - attr_min) / (attr_max - attr_min)
        else:
            attribution_map = np.zeros_like(attribution_map)

        return attribution_map

    def __call__(
        self,
        img1: Union[torch.Tensor, np.ndarray],
        img2: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Compute SHAP attribution map.

        Args:
            img1: Input image
                 - torch.Tensor: (C, H, W) or (B, C, H, W)
                 - np.ndarray: (H, W, C) in [0, 255] or [0, 1]
            img2: Second image for verification (optional)
                 If provided, explains similarity to img2
                 If None, explains embedding strength

        Returns:
            Attribution map (H, W) with values in [0, 1]
            Higher values = more important for decision
        """
        # Convert to numpy (H, W, C) in [0, 1]
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().detach().numpy()
            if img1.ndim == 4:
                img1 = img1[0]  # Remove batch dimension
            if img1.shape[0] == 3:
                img1 = np.transpose(img1, (1, 2, 0))  # (C, H, W) -> (H, W, C)

        if isinstance(img1, np.ndarray):
            if img1.max() > 1.0:
                img1 = img1 / 255.0  # Normalize to [0, 1]

        # Process img2 if provided
        target_embedding = None
        if img2 is not None:
            if isinstance(img2, np.ndarray):
                img2_torch = torch.from_numpy(img2).float()
                if img2_torch.max() > 1.0:
                    img2_torch = img2_torch / 255.0
                if img2_torch.ndim == 3:
                    img2_torch = img2_torch.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            else:
                img2_torch = img2.cpu() if isinstance(img2, torch.Tensor) else img2

            if img2_torch.ndim == 3:
                img2_torch = img2_torch.unsqueeze(0)

            img2_torch = img2_torch.to(self.device)

            # Compute target embedding
            with torch.no_grad():
                if hasattr(self.model, 'get_embedding'):
                    target_embedding = self.model.get_embedding(img2_torch)
                elif hasattr(self.model, 'model'):
                    target_embedding = self.model.model(img2_torch)
                else:
                    target_embedding = self.model(img2_torch)

        # Compute SHAP values
        shap_values = self.compute_shap_values(img1, target_embedding)

        # Map to pixel-level attribution
        segments = self._segment_superpixels(img1)
        attribution_map = self._map_shap_to_pixels(shap_values, segments)

        return attribution_map

    def compute(
        self,
        img1: Union[torch.Tensor, np.ndarray],
        img2: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> np.ndarray:
        """Alias for __call__ (compatibility with other methods)."""
        return self(img1, img2)


def get_shap_attribution(
    model: Callable,
    n_superpixels: int = 50,
    n_samples: int = 100,
    device: str = 'cuda'
) -> SHAPAttribution:
    """
    Convenience function to create SHAPAttribution instance.

    Args:
        model: Face verification model
        n_superpixels: Number of superpixels for segmentation
        n_samples: Number of samples for KernelSHAP
        device: Device for computation

    Returns:
        SHAPAttribution instance

    Example:
        >>> from insightface.app import FaceAnalysis
        >>> app = FaceAnalysis(name='buffalo_l')
        >>> app.prepare(ctx_id=0)
        >>> shap_attr = get_shap_attribution(app.model, device='cuda')
        >>> attribution = shap_attr(img1, img2)
    """
    return SHAPAttribution(
        model=model,
        n_superpixels=n_superpixels,
        n_samples=n_samples,
        device=device
    )
