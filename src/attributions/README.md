# Attribution Methods for Face Verification

This module provides baseline attribution/explainability methods for face verification models. These methods help understand which parts of a face image contribute to verification decisions.

## Overview

The module implements three baseline attribution methods:

1. **Grad-CAM** (Gradient-weighted Class Activation Mapping)
2. **SHAP** (SHapley Additive exPlanations)
3. **LIME** (Local Interpretable Model-agnostic Explanations)

All methods follow a consistent interface and return attribution maps normalized to [0, 1].

## Installation

Install required dependencies:

```bash
pip install -r ../../requirements.txt
```

Key packages:
- `torch>=2.0.0` - PyTorch for deep learning
- `shap>=0.42.0` - SHAP attribution library
- `lime>=0.2.0.1` - LIME attribution library
- `scikit-image>=0.21.0` - Image segmentation (superpixels)

## File Structure

```
src/attributions/
├── __init__.py          # Module initialization
├── gradcam.py           # Grad-CAM implementation
├── shap_wrapper.py      # SHAP wrapper for face verification
├── lime_wrapper.py      # LIME wrapper for face verification
├── example_usage.py     # Usage examples
└── README.md            # This file
```

## Usage

### Basic Interface

All attribution methods implement the same interface:

```python
attribution = method(img1, img2=None)
```

**Arguments:**
- `img1`: Input image tensor `(C, H, W)` or `(B, C, H, W)`
- `img2`: Optional second image for verification task

**Returns:**
- `attribution_map`: NumPy array `(H, W)` with values in [0, 1]

### 1. Grad-CAM

Grad-CAM uses gradient information from a target convolutional layer to generate attribution maps.

**Key Features:**
- Fast computation (gradient-based)
- Layer-specific explanations
- Good for CNN-based models

**Example:**

```python
from attributions import GradCAM
import torch

# Initialize
model = load_face_verification_model()
gradcam = GradCAM(model, target_layer='layer4')

# Single image attribution (explain embedding)
img = torch.randn(1, 3, 224, 224)
attribution = gradcam(img)

# Verification task (explain similarity)
img1 = torch.randn(1, 3, 224, 224)
img2 = torch.randn(1, 3, 224, 224)
attribution = gradcam(img1, img2)
```

**How it works:**
1. Forward pass through model to target layer
2. Backward pass to compute gradients
3. Global average pooling on gradients → weights
4. Weighted combination of activation maps
5. ReLU + normalization

**Reference:** Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"

### 2. SHAP

SHAP uses Shapley values from game theory to assign importance to each pixel.

**Key Features:**
- Theoretically grounded (Shapley values)
- Model-agnostic
- Can use KernelSHAP or DeepSHAP

**Example:**

```python
from attributions import SHAPAttribution
import torch

# Initialize
model = load_face_verification_model()
shap_explainer = SHAPAttribution(model)

# Generate attribution
img = torch.randn(1, 3, 224, 224)
attribution = shap_explainer(img)

# With verification target
img1 = torch.randn(1, 3, 224, 224)
img2 = torch.randn(1, 3, 224, 224)
attribution = shap_explainer(img1, img2)
```

**How it works:**
1. Creates background distribution of images
2. Perturbs input image by masking features
3. Computes Shapley values for each feature
4. Aggregates to pixel-level attributions

**Note:** Current implementation is a placeholder. Full implementation requires:
- Background sample selection
- KernelSHAP or DeepSHAP explainer
- Feature coalition generation
- Shapley value computation

**Reference:** Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"

### 3. LIME

LIME creates local interpretable explanations by fitting linear models to perturbed samples.

**Key Features:**
- Model-agnostic
- Interpretable superpixel-based features
- Good for local explanations

**Example:**

```python
from attributions import LIMEAttribution
import torch

# Initialize
model = load_face_verification_model()
lime_explainer = LIMEAttribution(model)

# Generate attribution
img = torch.randn(1, 3, 224, 224)
attribution = lime_explainer(img)

# With verification target
img1 = torch.randn(1, 3, 224, 224)
img2 = torch.randn(1, 3, 224, 224)
attribution = lime_explainer(img1, img2)
```

**How it works:**
1. Segments image into superpixels (SLIC algorithm)
2. Creates perturbed samples by masking superpixels
3. Gets model predictions for perturbed samples
4. Fits linear model to explain predictions
5. Returns superpixel weights as attribution

**Note:** Current implementation is a placeholder. Full implementation requires:
- Superpixel segmentation (SLIC/quickshift)
- Perturbation strategy
- Linear model fitting
- Attribution map generation

**Reference:** Ribeiro et al. (2016) "Why Should I Trust You? Explaining the Predictions of Any Classifier"

## Face Verification Adaptations

All methods are adapted for face verification tasks:

### Task 1: Embedding Explanation
**Question:** Which parts of the face contribute to the embedding representation?

```python
attribution = method(face_image)
```

### Task 2: Verification Explanation
**Question:** Which parts of the face contribute to matching another face?

```python
attribution = method(face_image_1, face_image_2)
```

## Implementation Notes

### Current Status

The current implementations are **placeholders** that return random attribution maps. They demonstrate the interface but need full implementation:

**Grad-CAM:**
- ✅ Interface defined
- ⚠️ Hook registration needed
- ⚠️ Gradient computation needed
- ⚠️ CAM generation needed

**SHAP:**
- ✅ Interface defined
- ⚠️ Background samples needed
- ⚠️ KernelSHAP/DeepSHAP integration needed
- ⚠️ Shapley value computation needed

**LIME:**
- ✅ Interface defined
- ⚠️ Superpixel segmentation needed
- ⚠️ Perturbation strategy needed
- ⚠️ Linear model fitting needed

### Full Implementation Requirements

#### Grad-CAM
1. Register forward hook on target layer to capture activations
2. Register backward hook to capture gradients
3. Compute weighted combination: α_k = GAP(∂y/∂A^k)
4. Generate CAM: ReLU(Σ α_k * A^k)
5. Upsample to input size

#### SHAP
1. Create background sample distribution (100+ images)
2. Initialize KernelSHAP or DeepSHAP explainer
3. For verification: compute similarity function
4. Generate Shapley values for each pixel
5. Aggregate and normalize

#### LIME
1. Segment image into superpixels (SLIC)
2. Create perturbed samples (1000+ samples)
3. Get model predictions for samples
4. Fit interpretable linear model
5. Extract superpixel weights
6. Map to pixel-level attribution

## Comparison

| Method | Speed | Accuracy | Interpretability | Model-Agnostic |
|--------|-------|----------|------------------|----------------|
| Grad-CAM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ❌ (CNN only) |
| SHAP | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ |
| LIME | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ |

**Grad-CAM:**
- Pros: Very fast, gradient-based, layer-specific insights
- Cons: Only works for CNNs, depends on layer choice

**SHAP:**
- Pros: Theoretically grounded (Shapley values), model-agnostic, faithful
- Cons: Computationally expensive, requires background samples

**LIME:**
- Pros: Highly interpretable (superpixels), model-agnostic, local explanations
- Cons: Slower than Grad-CAM, perturbation-dependent

## Example Application: Face Verification

```python
import torch
from attributions import GradCAM, SHAPAttribution, LIMEAttribution

# Load model and images
model = load_face_verification_model()
face1 = load_image("person_a.jpg")  # (1, 3, 224, 224)
face2 = load_image("person_b.jpg")  # (1, 3, 224, 224)

# Compare all methods
gradcam = GradCAM(model, target_layer='layer4')
shap = SHAPAttribution(model)
lime = LIMEAttribution(model)

# Generate attributions
attr_gradcam = gradcam(face1, face2)
attr_shap = shap(face1, face2)
attr_lime = lime(face1, face2)

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(face1.squeeze().permute(1, 2, 0).cpu().numpy())
axes[0].set_title("Input Face")

axes[1].imshow(attr_gradcam, cmap='jet')
axes[1].set_title("Grad-CAM")

axes[2].imshow(attr_shap, cmap='jet')
axes[2].set_title("SHAP")

axes[3].imshow(attr_lime, cmap='jet')
axes[3].set_title("LIME")

plt.show()
```

## Testing

Run the example script to test all methods:

```bash
cd /home/aaron/projects/xai
python -m src.attributions.example_usage
```

## Integration with Experiments

These baseline methods are used in:

**Experiment 6.1:** Comparative Evaluation
- Compare Grad-CAM, SHAP, LIME with proposed methods
- Metrics: Localization accuracy, computation time, faithfulness

**Experiment 6.2:** Fairness Analysis
- Use attributions to detect demographic biases
- Analyze which facial features are weighted differently

**Experiment 6.3:** Model Interpretability
- Explain verification decisions
- Identify failure modes

## References

1. **Grad-CAM:** Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" ICCV 2017

2. **SHAP:** Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions" NeurIPS 2017

3. **LIME:** Ribeiro et al. (2016) "Why Should I Trust You? Explaining the Predictions of Any Classifier" KDD 2016

## Next Steps

To complete the implementations:

1. **Grad-CAM:**
   - Implement hook registration system
   - Add support for multiple architectures (ResNet, VGG, ViT)
   - Add Guided Grad-CAM variant

2. **SHAP:**
   - Integrate actual SHAP library
   - Create background sample loader
   - Implement both KernelSHAP and DeepSHAP

3. **LIME:**
   - Integrate actual LIME library
   - Implement SLIC superpixel segmentation
   - Add perturbation sampling strategy

4. **Validation:**
   - Add unit tests for each method
   - Benchmark on standard datasets
   - Compare with reference implementations

## License

Part of the XAI dissertation project.
