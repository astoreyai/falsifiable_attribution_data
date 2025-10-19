"""
Grad-CAM Attribution for Face Verification

Implements Grad-CAM (Selvaraju et al., 2017) for face verification models.

Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization"
https://arxiv.org/abs/1610.02391

Implementation adapted for metric learning (face verification) by using
cosine similarity as the target instead of class probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable, Union, List


class GradCAM:
    """
    Grad-CAM attribution method for face verification.

    Computes Class Activation Mapping (CAM) using gradients flowing into
    the final convolutional layer. Adapted for metric learning by using
    embedding similarity as target instead of class scores.

    Algorithm:
    1. Forward pass: capture activations A^k from target conv layer
    2. Backward pass: capture gradients ∂y/∂A^k
    3. Compute weights: α_k = GAP(∂y/∂A^k)  [Global Average Pooling]
    4. Weighted sum: L_GradCAM = ReLU(Σ_k α_k * A^k)
    5. Upsample to input size and normalize

    This is a baseline method used for comparison in Experiment 6.1.
    """

    def __init__(
        self,
        model: Callable,
        target_layer: Optional[str] = None,
        device: str = 'cuda'
    ):
        """
        Initialize Grad-CAM.

        Args:
            model: Face verification model (embedding extractor)
                  Should be a PyTorch nn.Module or compatible callable
            target_layer: Name of target convolutional layer
                         (if None, automatically find last conv layer)
            device: Device for computation ('cuda' or 'cpu')
        """
        self.model = model
        self.target_layer_name = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        self.hooks = []

        # Set model to eval mode
        if hasattr(self.model, 'eval'):
            self.model.eval()

    def _find_target_layer(self) -> nn.Module:
        """
        Find the last convolutional layer in the model.

        For ResNet-based models (like ArcFace-ResNet50), this is typically
        the last conv layer before global average pooling.

        Returns:
            Target layer module
        """
        if self.target_layer_name is not None:
            # User specified a layer name
            for name, module in self.model.named_modules():
                if name == self.target_layer_name:
                    return module
            raise ValueError(f"Layer '{self.target_layer_name}' not found in model")

        # Auto-detect: find last Conv2d layer
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module

        if last_conv is None:
            # Try to handle models with nested structure (InsightFace)
            if hasattr(self.model, 'model'):
                for module in self.model.model.modules():
                    if isinstance(module, nn.Conv2d):
                        last_conv = module

        if last_conv is None:
            raise ValueError("No Conv2d layer found in model. Please specify target_layer explicitly.")

        return last_conv

    def _register_hooks(self):
        """
        Register forward and backward hooks on target layer.

        Forward hook: captures activations during forward pass
        Backward hook: captures gradients during backward pass
        """
        def forward_hook(module, input, output):
            """Capture activations."""
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            """Capture gradients."""
            self.gradients = grad_output[0].detach()

        # Find target layer
        target_layer = self._find_target_layer()

        # Register hooks
        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_full_backward_hook(backward_hook)

        # Store handles for cleanup
        self.hooks = [handle_forward, handle_backward]

    def _remove_hooks(self):
        """Remove registered hooks to prevent memory leaks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def _compute_cam(
        self,
        image: torch.Tensor,
        target_embedding: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Args:
            image: Input image tensor (B, C, H, W)
            target_embedding: Optional target embedding for verification task
                             If None, explains the embedding norm

        Returns:
            CAM heatmap (H, W) normalized to [0, 1]
        """
        # Ensure image requires grad
        image = image.clone().detach().requires_grad_(True)

        # Register hooks
        self._register_hooks()

        try:
            # Forward pass
            if hasattr(self.model, 'get_embedding'):
                # Custom model with get_embedding method
                embedding = self.model.get_embedding(image)
            elif hasattr(self.model, 'model'):
                # Nested model structure (InsightFace wrapper)
                embedding = self.model.model(image)
            else:
                # Standard PyTorch model
                embedding = self.model(image)

            # Normalize embedding to unit sphere (standard for face verification)
            embedding_normalized = F.normalize(embedding, p=2, dim=-1)

            # Compute target score
            if target_embedding is not None:
                # Verification task: maximize cosine similarity with target
                target_embedding_normalized = F.normalize(target_embedding, p=2, dim=-1)
                target_score = F.cosine_similarity(
                    embedding_normalized.view(1, -1),
                    target_embedding_normalized.view(1, -1),
                    dim=1
                ).sum()
            else:
                # Embedding task: maximize embedding norm
                # (explains what makes the embedding strong)
                target_score = torch.norm(embedding, p=2, dim=-1).sum()

            # Backward pass
            self.model.zero_grad()
            target_score.backward(retain_graph=False)

            # Get activations and gradients
            activations = self.activations  # (B, C, H, W)
            gradients = self.gradients      # (B, C, H, W)

            # Compute weights using Global Average Pooling
            # α_k = (1/Z) Σ_i Σ_j (∂y/∂A^k_ij)
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

            # Weighted combination of activation maps
            # L_GradCAM = Σ_k α_k * A^k
            cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (B, 1, H, W)

            # Apply ReLU (only positive influence)
            cam = F.relu(cam)

            # Upsample to input image size
            input_size = (image.shape[2], image.shape[3])
            cam = F.interpolate(
                cam,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )

            # Convert to numpy and normalize
            cam = cam.squeeze().cpu().detach().numpy()

            # Normalize to [0, 1]
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                # Uniform activation (no gradient signal)
                cam = np.zeros_like(cam)

            return cam

        finally:
            # Always remove hooks to prevent memory leaks
            self._remove_hooks()

    def __call__(
        self,
        img1: Union[torch.Tensor, np.ndarray],
        img2: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Compute Grad-CAM attribution map.

        Args:
            img1: First image
                 - torch.Tensor: (C, H, W) or (B, C, H, W)
                 - np.ndarray: (H, W, C) in [0, 255] or [0, 1]
            img2: Second image for verification (optional)
                 If provided, explains similarity to img2
                 If None, explains embedding strength

        Returns:
            Attribution map (H, W) with values in [0, 1]
            Higher values = more important for the target
        """
        # Convert numpy to torch if needed
        if isinstance(img1, np.ndarray):
            # Assume numpy is (H, W, C) in [0, 255] or [0, 1]
            img1 = torch.from_numpy(img1).float()
            if img1.max() > 1.0:
                img1 = img1 / 255.0
            # Transpose to (C, H, W)
            if img1.ndim == 3:
                img1 = img1.permute(2, 0, 1)

        # Ensure batch dimension
        if img1.ndim == 3:
            img1 = img1.unsqueeze(0)

        # Move to device
        img1 = img1.to(self.device)

        # Process img2 if provided
        target_embedding = None
        if img2 is not None:
            if isinstance(img2, np.ndarray):
                img2 = torch.from_numpy(img2).float()
                if img2.max() > 1.0:
                    img2 = img2 / 255.0
                if img2.ndim == 3:
                    img2 = img2.permute(2, 0, 1)

            if img2.ndim == 3:
                img2 = img2.unsqueeze(0)

            img2 = img2.to(self.device)

            # Compute target embedding
            with torch.no_grad():
                if hasattr(self.model, 'get_embedding'):
                    target_embedding = self.model.get_embedding(img2)
                elif hasattr(self.model, 'model'):
                    target_embedding = self.model.model(img2)
                else:
                    target_embedding = self.model(img2)

        # Compute CAM
        cam = self._compute_cam(img1, target_embedding)

        return cam

    def compute(
        self,
        img1: Union[torch.Tensor, np.ndarray],
        img2: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> np.ndarray:
        """Alias for __call__ (compatibility with other methods)."""
        return self(img1, img2)


def get_gradcam(
    model: Callable,
    target_layer: Optional[str] = None,
    device: str = 'cuda'
) -> GradCAM:
    """
    Convenience function to create GradCAM instance.

    Args:
        model: Face verification model
        target_layer: Name of target layer (None = auto-detect)
        device: Device for computation

    Returns:
        GradCAM instance

    Example:
        >>> from insightface.app import FaceAnalysis
        >>> app = FaceAnalysis(name='buffalo_l')
        >>> app.prepare(ctx_id=0)
        >>> gradcam = get_gradcam(app.model, device='cuda')
        >>> attribution = gradcam(img1, img2)
    """
    return GradCAM(model=model, target_layer=target_layer, device=device)
