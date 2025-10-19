"""
Biometric Grad-CAM for Face Verification.

Novel attribution method that adapts Grad-CAM for biometric face verification
by explicitly modeling identity-preserving transformations and demographic
fairness considerations.

Key Innovation: Identity-aware activation weighting that prioritizes features
contributing to identity preservation rather than generic classification.

Based on:
- Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization" https://arxiv.org/abs/1610.02391
- Original contribution: Identity preservation and demographic fairness weighting

This is a PROPOSED METHOD for Experiment 6.1 - it should outperform standard
Grad-CAM by accounting for the unique characteristics of face verification:
1. Identity preservation (not just classification)
2. Invariance to pose, illumination, expression
3. Fairness across demographic groups
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable, Dict, Union, List


class BiometricGradCAM:
    """
    Biometric Grad-CAM.

    Extends Grad-CAM with biometric-specific enhancements for face verification.

    Standard Grad-CAM computes:
        L_GradCAM = ReLU(Σ_k α_k * A_k)
        where α_k = GAP(∂y_c/∂A_k)

    Biometric Grad-CAM enhances this with:
    1. Identity-aware weighting: α_k weighted by contribution to identity preservation
    2. Invariance regularization: Downweight features sensitive to pose/illumination
    3. Demographic fairness: Optional debiasing across protected attributes

    Theoretical Motivation:
    - Face verification is not classification (no fixed classes)
    - Goal is identity preservation under transformations
    - Should focus on intrinsic identity features (eyes, nose, jawline)
    - Should ignore extrinsic factors (lighting, background, pose)

    Expected Performance:
    - Higher precision than standard Grad-CAM (fewer false positive regions)
    - Better localization on identity-critical features (facial landmarks)
    - More robust to pose and illumination variations
    - Fairer attributions across demographic groups
    """

    def __init__(
        self,
        model: Callable,
        target_layer: Optional[str] = None,
        use_identity_weighting: bool = True,
        use_invariance_reg: bool = True,
        use_demographic_fairness: bool = False,
        device: str = 'cuda'
    ):
        """
        Initialize Biometric Grad-CAM.

        Args:
            model: Face verification model (embedding extractor)
            target_layer: Name of target convolutional layer
                        If None, automatically selects last conv layer
            use_identity_weighting: Apply identity-preserving weights
            use_invariance_reg: Apply invariance regularization
            use_demographic_fairness: Apply demographic fairness correction
            device: Device for computation
        """
        self.model = model
        self.target_layer_name = target_layer
        self.use_identity_weighting = use_identity_weighting
        self.use_invariance_reg = use_invariance_reg
        self.use_demographic_fairness = use_demographic_fairness
        self.device = device

        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        self.hooks = []

        # Register hooks on target layer
        self._register_hooks()

    def _find_target_layer(self) -> nn.Module:
        """
        Find target layer for Grad-CAM.

        If target_layer_name is specified, find that layer.
        Otherwise, use the last convolutional layer.

        Returns:
            Target layer module
        """
        if self.target_layer_name is not None:
            # Find by name
            for name, module in self.model.named_modules():
                if name == self.target_layer_name:
                    return module
            raise ValueError(f"Layer '{self.target_layer_name}' not found in model")
        else:
            # Find last Conv2d layer
            last_conv = None
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module

            if last_conv is None:
                raise ValueError("No Conv2d layer found in model")

            return last_conv

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        target_layer = self._find_target_layer()

        def forward_hook(module, input, output):
            """Capture activations during forward pass."""
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            """Capture gradients during backward pass."""
            self.gradients = grad_output[0].detach()

        # Register hooks
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Remove registered hooks (cleanup)."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _compute_identity_weights(
        self,
        embedding: torch.Tensor,
        target_embedding: Optional[torch.Tensor],
        threshold: float = 0.6
    ) -> torch.Tensor:
        """
        Compute identity-preserving weights.

        Weights activations by how much they contribute to maintaining
        identity similarity (high cosine similarity with target).

        For genuine pairs (same identity):
            - High similarity = high weight (preserve features)
        For impostor pairs (different identity):
            - Low similarity = high weight (distinguish features)

        Args:
            embedding: Query embedding
            target_embedding: Target embedding (if verification task)
            threshold: Similarity threshold for genuine vs impostor

        Returns:
            Weight scalar in [0, 1]
        """
        if target_embedding is None:
            # No target: weight = 1 (neutral)
            return torch.ones(1, device=embedding.device)

        # Compute cosine similarity
        sim = F.cosine_similarity(
            embedding,
            target_embedding.detach(),
            dim=-1
        )

        # For genuine pairs (sim > threshold): weight = sim
        # For impostor pairs (sim < threshold): weight = 1 - sim
        # This creates U-shaped weighting: high weight when confident
        weight = torch.where(
            sim > threshold,
            sim,  # Genuine: higher sim = higher weight
            1.0 - sim  # Impostor: lower sim = higher weight
        )

        # Apply sigmoid to smooth and bound to [0, 1]
        weight = torch.sigmoid(5 * (weight - 0.5))

        return weight

    def _compute_invariance_regularization(
        self,
        gradients: torch.Tensor,
        temperature: float = 0.5
    ) -> torch.Tensor:
        """
        Compute invariance regularization weights.

        Downweights features with high gradient variance across spatial locations,
        as these are likely sensitive to pose/illumination rather than identity.

        Identity features (eyes, nose, mouth) should have consistent importance
        across the face region. Extrinsic features (background, lighting artifacts)
        will have high spatial variance.

        Args:
            gradients: Gradient tensor (B, C, H, W)
            temperature: Temperature for softmax normalization

        Returns:
            Invariance weights per channel (B, C, 1, 1)
        """
        # Compute spatial variance for each channel
        # High variance = sensitive to location = likely extrinsic feature
        spatial_mean = torch.mean(gradients, dim=(2, 3), keepdim=True)
        spatial_var = torch.mean(
            (gradients - spatial_mean) ** 2,
            dim=(2, 3),
            keepdim=True
        )

        # Convert variance to weight: low variance = high weight
        # Use inverse and normalize
        inv_var = 1.0 / (spatial_var + 1e-6)
        weights = F.softmax(inv_var / temperature, dim=1)

        return weights

    def _compute_demographic_fairness_correction(
        self,
        gradients: torch.Tensor,
        demographic_attributes: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compute demographic fairness correction.

        This is a placeholder for demographic debiasing. In a full implementation,
        this would use learned bias directions (similar to fairness-aware ML)
        to downweight features correlated with protected attributes.

        For example, if certain facial regions are known to correlate with
        gender/race but not identity, we can downweight them.

        Args:
            gradients: Gradient tensor (B, C, H, W)
            demographic_attributes: Optional demographic info
                                  e.g., {'gender': 0/1, 'age': float, 'race': int}

        Returns:
            Fairness correction weights (B, C, 1, 1)
        """
        # Placeholder: In real implementation, would use:
        # 1. Pre-computed bias directions from diverse dataset
        # 2. Project gradients onto bias-free subspace
        # 3. Reweight to minimize demographic dependence

        # For now, return uniform weights (no correction)
        return torch.ones(
            gradients.size(0),
            gradients.size(1),
            1,
            1,
            device=gradients.device
        )

    def generate_attribution(
        self,
        image: Union[torch.Tensor, np.ndarray],
        target_embedding: Optional[torch.Tensor] = None,
        demographic_attributes: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Generate Biometric Grad-CAM attribution map.

        Args:
            image: Input face image
                  - torch.Tensor: (C, H, W) or (B, C, H, W)
                  - np.ndarray: (H, W, C)
            target_embedding: Optional target embedding for verification
            demographic_attributes: Optional demographic info for fairness
                                  e.g., {'gender': 0/1, 'age': 25.0, 'race': 0}

        Returns:
            attribution_map: (H, W) attribution heatmap in [0, 1]
        """
        # Convert numpy to torch if needed
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Move to device and enable gradients
        image = image.to(self.device).requires_grad_(True)

        # Forward pass through model
        embedding = self.model(image)

        # Normalize embedding (face verification models use cosine similarity)
        embedding_normalized = F.normalize(embedding, p=2, dim=-1)

        # Compute target for backward pass
        if target_embedding is not None:
            # Verification task: maximize cosine similarity
            target_embedding = F.normalize(target_embedding, p=2, dim=-1)
            score = F.cosine_similarity(
                embedding_normalized,
                target_embedding.detach(),
                dim=-1
            ).sum()
        else:
            # Embedding task: maximize L2 norm
            score = torch.norm(embedding, dim=-1).sum()

        # Backward pass to compute gradients
        self.model.zero_grad()
        if image.grad is not None:
            image.grad.zero_()
        score.backward(retain_graph=True)

        # Get activations and gradients from hooks
        activations = self.activations  # (B, C, H', W')
        gradients = self.gradients      # (B, C, H', W')

        if activations is None or gradients is None:
            raise RuntimeError("Activations or gradients not captured. Check hooks.")

        # Standard Grad-CAM: Global Average Pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Enhancement 1: Identity-aware weighting
        if self.use_identity_weighting and target_embedding is not None:
            identity_weight = self._compute_identity_weights(
                embedding_normalized,
                target_embedding
            )
            # Broadcast identity weight across channels
            weights = weights * identity_weight.view(-1, 1, 1, 1)

        # Enhancement 2: Invariance regularization
        if self.use_invariance_reg:
            invariance_weights = self._compute_invariance_regularization(gradients)
            weights = weights * invariance_weights

        # Enhancement 3: Demographic fairness correction
        if self.use_demographic_fairness:
            fairness_weights = self._compute_demographic_fairness_correction(
                gradients,
                demographic_attributes
            )
            weights = weights * fairness_weights

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (B, 1, H', W')

        # Apply ReLU (only positive contributions)
        # Negative gradients indicate features that decrease similarity
        cam = F.relu(cam)

        # Upsample to input image size
        cam = F.interpolate(
            cam,
            size=(image.size(2), image.size(3)),
            mode='bilinear',
            align_corners=False
        )

        # Convert to numpy
        cam = cam.squeeze().cpu().detach().numpy()

        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def __call__(
        self,
        img1: Union[torch.Tensor, np.ndarray],
        img2: Optional[torch.Tensor] = None,
        demo_attr: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Callable interface matching other attribution methods.

        Args:
            img1: Input image
            img2: Optional second image for verification (computes target embedding)
            demo_attr: Optional demographic attributes

        Returns:
            Attribution map (H, W) in [0, 1]
        """
        # If img2 provided, compute its embedding as target
        if img2 is not None:
            if isinstance(img2, np.ndarray):
                img2 = torch.from_numpy(img2.transpose(2, 0, 1)).float()
            if img2.dim() == 3:
                img2 = img2.unsqueeze(0)
            img2 = img2.to(self.device)

            with torch.no_grad():
                target_embedding = self.model(img2)
                target_embedding = F.normalize(target_embedding, p=2, dim=-1)
        else:
            target_embedding = None

        return self.generate_attribution(img1, target_embedding, demo_attr)

    def compute(
        self,
        img1: Union[torch.Tensor, np.ndarray],
        img2: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """Alias for __call__ (compatibility with other methods)."""
        return self(img1, img2)

    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        self.remove_hooks()


class BiometricGradCAMPlusPlus(BiometricGradCAM):
    """
    Biometric Grad-CAM++ variant.

    Extends Biometric Grad-CAM with Grad-CAM++ weighting scheme:
        α_k = Σ_i Σ_j (∂²y/∂A_k_ij²) * ReLU(∂y/∂A_k_ij)

    This gives better localization for multiple instances of the same class.
    For face verification, this helps when multiple facial features contribute
    to identity (e.g., both eyes + nose + mouth).

    Reference: Chattopadhyay et al. (2018) "Grad-CAM++: Improved Visual
    Explanations for Deep Convolutional Networks"
    """

    def generate_attribution(
        self,
        image: Union[torch.Tensor, np.ndarray],
        target_embedding: Optional[torch.Tensor] = None,
        demographic_attributes: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Generate Biometric Grad-CAM++ attribution.

        Uses second-order gradients for improved localization.
        """
        # Convert numpy to torch if needed
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device).requires_grad_(True)

        # Forward pass
        embedding = self.model(image)
        embedding_normalized = F.normalize(embedding, p=2, dim=-1)

        # Compute target
        if target_embedding is not None:
            target_embedding = F.normalize(target_embedding, p=2, dim=-1)
            score = F.cosine_similarity(
                embedding_normalized,
                target_embedding.detach(),
                dim=-1
            ).sum()
        else:
            score = torch.norm(embedding, dim=-1).sum()

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        activations = self.activations
        gradients = self.gradients

        # Grad-CAM++ weighting:
        # α_k = Σ_ij α_k_ij * ReLU(∂y/∂A_k_ij)
        # where α_k_ij = (∂²y/∂A_k_ij²) / (2*(∂²y/∂A_k_ij²) + Σ_ab A_k_ab * (∂³y/∂A_k_ab³))

        # Compute second derivatives (approximate with gradient of gradient)
        # For efficiency, use spatial variance as proxy
        grad_squared = gradients ** 2
        spatial_sum = torch.sum(activations * (gradients ** 3), dim=(2, 3), keepdim=True)

        alpha = grad_squared / (2 * grad_squared + spatial_sum + 1e-8)

        # Weight by ReLU of gradients
        weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)

        # Apply identity weighting and other enhancements
        if self.use_identity_weighting and target_embedding is not None:
            identity_weight = self._compute_identity_weights(
                embedding_normalized,
                target_embedding
            )
            weights = weights * identity_weight.view(-1, 1, 1, 1)

        if self.use_invariance_reg:
            invariance_weights = self._compute_invariance_regularization(gradients)
            weights = weights * invariance_weights

        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample
        cam = F.interpolate(
            cam,
            size=(image.size(2), image.size(3)),
            mode='bilinear',
            align_corners=False
        )

        # Convert to numpy and normalize
        cam = cam.squeeze().cpu().detach().numpy()
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def get_biometric_gradcam(
    model: Callable,
    target_layer: Optional[str] = None,
    use_identity_weighting: bool = True,
    use_invariance_reg: bool = True,
    use_demographic_fairness: bool = False,
    variant: str = 'standard',
    device: str = 'cuda'
) -> Union[BiometricGradCAM, BiometricGradCAMPlusPlus]:
    """
    Convenience function to create Biometric Grad-CAM instance.

    Args:
        model: Face verification model
        target_layer: Name of target layer (None = auto-detect)
        use_identity_weighting: Apply identity-aware weighting
        use_invariance_reg: Apply invariance regularization
        use_demographic_fairness: Apply fairness correction
        variant: 'standard' or 'plusplus' (Grad-CAM++)
        device: Device for computation

    Returns:
        BiometricGradCAM or BiometricGradCAMPlusPlus instance
    """
    if variant == 'plusplus':
        return BiometricGradCAMPlusPlus(
            model=model,
            target_layer=target_layer,
            use_identity_weighting=use_identity_weighting,
            use_invariance_reg=use_invariance_reg,
            use_demographic_fairness=use_demographic_fairness,
            device=device
        )
    else:
        return BiometricGradCAM(
            model=model,
            target_layer=target_layer,
            use_identity_weighting=use_identity_weighting,
            use_invariance_reg=use_invariance_reg,
            use_demographic_fairness=use_demographic_fairness,
            device=device
        )
