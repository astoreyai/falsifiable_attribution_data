"""
Gradient × Input Attribution Method

AGENT 3 SOLUTION: Replace Grad-CAM with input-space gradient method
for holistic face verification models.

Why This Works Better Than Grad-CAM:
- Operates on INPUT SPACE, not intermediate spatial features
- No requirement for convolutional feature maps with spatial locality
- Works for ANY differentiable model (architecture-agnostic)
- Well-established method (Shrikumar et al., 2016)

Reference:
Shrikumar, A., Greenside, P., Shcherbina, A., & Kundaje, A. (2016).
"Not Just a Black Box: Learning Important Features Through Propagating
Activation Differences." arXiv preprint arXiv:1605.01713.

Expected Falsification Rate: 60-70% (maintains separation from Geodesic IG's 100%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class GradientXInput:
    """
    Gradient × Input attribution method.

    Computes attribution as element-wise product of input and gradient:
    A(x) = x ⊙ ∇_x f(x)

    This gives pixel-level importance scores that respect input magnitude.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize Gradient × Input attribution.

        Args:
            model: Face verification model (must be differentiable)
        """
        self.model = model
        self.model.eval()

    def attribute(
        self,
        image: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        abs_value: bool = True
    ) -> torch.Tensor:
        """
        Compute Gradient × Input attribution.

        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            target: Optional target class/embedding for gradient direction
            abs_value: If True, return absolute values

        Returns:
            Attribution map (same shape as input)
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Ensure gradient tracking
        image.requires_grad_(True)

        # Forward pass
        output = self.model(image)

        # Compute gradient
        if target is not None:
            # Gradient with respect to target embedding
            # Loss: negative cosine similarity (maximize similarity)
            loss = -F.cosine_similarity(output, target).mean()
        else:
            # Gradient with respect to output norm (feature magnitude)
            loss = output.norm(dim=1).mean()

        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Get gradients
        gradients = image.grad.detach()

        # Compute attribution: gradient × input
        attribution = image.detach() * gradients

        if abs_value:
            attribution = attribution.abs()

        # Squeeze if needed
        if squeeze_output:
            attribution = attribution.squeeze(0)

        return attribution

    def attribute_spatially_aggregated(
        self,
        image: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        aggregate_method: str = 'sum'
    ) -> torch.Tensor:
        """
        Compute spatially aggregated attribution (per-channel or total).

        Args:
            image: Input image (C, H, W) or (B, C, H, W)
            target: Optional target for gradient direction
            aggregate_method: 'sum', 'mean', or 'max' across spatial dimensions

        Returns:
            Attribution map aggregated spatially (C,) or (B, C)
        """
        # Get full attribution
        attr = self.attribute(image, target, abs_value=True)

        # Aggregate over spatial dimensions
        if aggregate_method == 'sum':
            spatial_attr = attr.sum(dim=(-2, -1))  # Sum over H, W
        elif aggregate_method == 'mean':
            spatial_attr = attr.mean(dim=(-2, -1))  # Mean over H, W
        elif aggregate_method == 'max':
            spatial_attr = attr.max(dim=-1)[0].max(dim=-1)[0]  # Max over H, W
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate_method}")

        return spatial_attr

    def get_importance_scores(
        self,
        image: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Get flattened importance scores for falsification testing.

        Args:
            image: Input image (C, H, W) or (B, C, H, W)
            target: Optional target for gradient direction
            normalize: If True, normalize scores to [0, 1]

        Returns:
            Flattened importance scores (D,) where D = C * H * W
        """
        # Get attribution
        attr = self.attribute(image, target, abs_value=True)

        # Flatten
        if attr.dim() == 4:
            # Batch dimension exists, take first
            attr = attr[0]

        scores = attr.flatten().cpu().numpy()

        if normalize and scores.max() > 0:
            scores = scores / scores.max()

        return scores


class VanillaGradients:
    """
    Vanilla Gradients (Saliency Maps) attribution method.

    Computes attribution as gradient magnitude:
    A(x) = |∇_x f(x)|

    Simpler than Gradient × Input but less sensitive to input magnitude.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize Vanilla Gradients attribution.

        Args:
            model: Face verification model
        """
        self.model = model
        self.model.eval()

    def attribute(
        self,
        image: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        abs_value: bool = True
    ) -> torch.Tensor:
        """
        Compute Vanilla Gradients attribution.

        Args:
            image: Input image (C, H, W) or (B, C, H, W)
            target: Optional target for gradient direction
            abs_value: If True, return absolute values

        Returns:
            Attribution map (gradient magnitude)
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Ensure gradient tracking
        image.requires_grad_(True)

        # Forward pass
        output = self.model(image)

        # Compute gradient
        if target is not None:
            loss = -F.cosine_similarity(output, target).mean()
        else:
            loss = output.norm(dim=1).mean()

        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Get gradients
        attribution = image.grad.detach()

        if abs_value:
            attribution = attribution.abs()

        # Squeeze if needed
        if squeeze_output:
            attribution = attribution.squeeze(0)

        return attribution

    def get_importance_scores(
        self,
        image: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Get flattened importance scores.

        Args:
            image: Input image (C, H, W) or (B, C, H, W)
            target: Optional target for gradient direction
            normalize: If True, normalize scores to [0, 1]

        Returns:
            Flattened importance scores
        """
        attr = self.attribute(image, target, abs_value=True)

        if attr.dim() == 4:
            attr = attr[0]

        scores = attr.flatten().cpu().numpy()

        if normalize and scores.max() > 0:
            scores = scores / scores.max()

        return scores


class SmoothGrad:
    """
    SmoothGrad attribution method.

    Computes smoothed gradients by averaging over noisy samples:
    A(x) = E_ε[∇_x f(x + ε)]

    Reduces gradient noise and produces more stable attributions.

    Reference:
    Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017).
    "SmoothGrad: Removing noise by adding noise." arXiv preprint arXiv:1706.03825.
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 50,
        noise_std: float = 0.15
    ):
        """
        Initialize SmoothGrad attribution.

        Args:
            model: Face verification model
            n_samples: Number of noisy samples to average
            noise_std: Standard deviation of Gaussian noise
        """
        self.model = model
        self.model.eval()
        self.n_samples = n_samples
        self.noise_std = noise_std

    def attribute(
        self,
        image: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        abs_value: bool = True,
        use_gradient_x_input: bool = True
    ) -> torch.Tensor:
        """
        Compute SmoothGrad attribution.

        Args:
            image: Input image (C, H, W) or (B, C, H, W)
            target: Optional target for gradient direction
            abs_value: If True, return absolute values
            use_gradient_x_input: If True, use Gradient × Input; else Vanilla Gradients

        Returns:
            Smoothed attribution map
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        device = image.device
        accumulated_attribution = torch.zeros_like(image)

        # Sample noisy versions
        for i in range(self.n_samples):
            # Add Gaussian noise
            noise = torch.randn_like(image) * self.noise_std
            noisy_image = image + noise
            noisy_image = noisy_image.clamp(0, 1)  # Keep in valid range
            noisy_image.requires_grad_(True)

            # Forward pass
            output = self.model(noisy_image)

            # Compute gradient
            if target is not None:
                loss = -F.cosine_similarity(output, target).mean()
            else:
                loss = output.norm(dim=1).mean()

            # Backward pass
            self.model.zero_grad()
            loss.backward()

            # Get gradients
            gradients = noisy_image.grad.detach()

            # Accumulate attribution
            if use_gradient_x_input:
                accumulated_attribution += noisy_image.detach() * gradients
            else:
                accumulated_attribution += gradients

        # Average
        attribution = accumulated_attribution / self.n_samples

        if abs_value:
            attribution = attribution.abs()

        # Squeeze if needed
        if squeeze_output:
            attribution = attribution.squeeze(0)

        return attribution

    def get_importance_scores(
        self,
        image: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Get flattened importance scores.

        Args:
            image: Input image (C, H, W) or (B, C, H, W)
            target: Optional target for gradient direction
            normalize: If True, normalize scores to [0, 1]

        Returns:
            Flattened importance scores
        """
        attr = self.attribute(image, target, abs_value=True)

        if attr.dim() == 4:
            attr = attr[0]

        scores = attr.flatten().cpu().numpy()

        if normalize and scores.max() > 0:
            scores = scores / scores.max()

        return scores
