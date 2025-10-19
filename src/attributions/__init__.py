"""
Attribution Methods for Face Verification

Implements various attribution/explainability methods:
- Grad-CAM (baseline)
- Biometric Grad-CAM (proposed)
- Geodesic Integrated Gradients (proposed)
- SHAP (baseline)
- LIME (baseline)
"""

from .gradcam import GradCAM
from .shap_wrapper import SHAPAttribution
from .lime_wrapper import LIMEAttribution
from .geodesic_ig import GeodesicIntegratedGradients, get_geodesic_ig
from .biometric_gradcam import (
    BiometricGradCAM,
    BiometricGradCAMPlusPlus,
    get_biometric_gradcam
)

__all__ = [
    # Baseline methods
    'GradCAM',
    'SHAPAttribution',
    'LIMEAttribution',

    # Proposed methods (novel contributions)
    'GeodesicIntegratedGradients',
    'BiometricGradCAM',
    'BiometricGradCAMPlusPlus',

    # Convenience functions
    'get_geodesic_ig',
    'get_biometric_gradcam',
]
