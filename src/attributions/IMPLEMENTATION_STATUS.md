# Attribution Methods Implementation Status

**Created:** October 18, 2025
**Location:** `/home/aaron/projects/xai/src/attributions/`
**Purpose:** Baseline attribution methods for face verification experiments

---

## Summary

Successfully created the attribution methods module structure with three baseline methods adapted for face verification:

1. **Grad-CAM** - Gradient-weighted Class Activation Mapping
2. **SHAP** - Shapley Additive Explanations
3. **LIME** - Local Interpretable Model-agnostic Explanations

---

## Files Created

### Core Implementation Files

| File | Lines | Description |
|------|-------|-------------|
| `gradcam.py` | 77 | Grad-CAM attribution implementation |
| `shap_wrapper.py` | 50 | SHAP wrapper for face verification |
| `lime_wrapper.py` | 50 | LIME wrapper for face verification |
| `__init__.py` | 20 | Module initialization and exports |

### Documentation & Examples

| File | Lines | Description |
|------|-------|-------------|
| `README.md` | 368 | Comprehensive usage documentation |
| `example_usage.py` | 304 | Usage examples and demonstrations |
| `IMPLEMENTATION_STATUS.md` | - | This file |

**Total:** 869 lines of code and documentation

---

## Implementation Details

### 1. Grad-CAM (`gradcam.py`)

**Status:** ⚠️ Placeholder Implementation

**Interface:**
```python
class GradCAM:
    def __init__(self, model: Callable, target_layer: Optional[str] = None)
    def __call__(self, img1: torch.Tensor, img2: Optional[torch.Tensor] = None) -> np.ndarray
    def compute(self, img1: torch.Tensor, img2: Optional[torch.Tensor] = None) -> np.ndarray
```

**Current State:**
- ✅ Class structure defined
- ✅ Interface implemented
- ✅ Documentation complete
- ⚠️ Returns random attribution maps (placeholder)

**To Complete:**
- [ ] Implement hook registration for target layer
- [ ] Capture activations in forward pass
- [ ] Capture gradients in backward pass
- [ ] Compute weighted combination: α_k = GAP(∂y/∂A^k)
- [ ] Generate CAM: ReLU(Σ α_k * A^k)
- [ ] Upsample to input resolution

**Reference:** Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"

---

### 2. SHAP (`shap_wrapper.py`)

**Status:** ⚠️ Placeholder Implementation

**Interface:**
```python
class SHAPAttribution:
    def __init__(self, model: Callable)
    def __call__(self, img1: torch.Tensor, img2: Optional[torch.Tensor] = None) -> np.ndarray
    def compute(self, img1: torch.Tensor, img2: Optional[torch.Tensor] = None) -> np.ndarray
```

**Current State:**
- ✅ Class structure defined
- ✅ Interface implemented
- ✅ Documentation complete
- ⚠️ Returns random attribution maps (placeholder)

**To Complete:**
- [ ] Integrate actual `shap` library
- [ ] Create background sample distribution (100+ images)
- [ ] Initialize KernelSHAP or DeepSHAP explainer
- [ ] Implement prediction wrapper function
- [ ] Compute Shapley values for pixels/features
- [ ] Aggregate to pixel-level attribution map

**Reference:** Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"

---

### 3. LIME (`lime_wrapper.py`)

**Status:** ⚠️ Placeholder Implementation

**Interface:**
```python
class LIMEAttribution:
    def __init__(self, model: Callable)
    def __call__(self, img1: torch.Tensor, img2: Optional[torch.Tensor] = None) -> np.ndarray
    def compute(self, img1: torch.Tensor, img2: Optional[torch.Tensor] = None) -> np.ndarray
```

**Current State:**
- ✅ Class structure defined
- ✅ Interface implemented
- ✅ Documentation complete
- ⚠️ Returns random attribution maps (placeholder)

**To Complete:**
- [ ] Integrate actual `lime` library
- [ ] Implement superpixel segmentation (SLIC)
- [ ] Create perturbation sampling strategy
- [ ] Implement prediction wrapper function
- [ ] Fit interpretable linear model
- [ ] Extract and map superpixel weights to pixels

**Reference:** Ribeiro et al. (2016) "Why Should I Trust You? Explaining the Predictions of Any Classifier"

---

## Unified Interface

All three methods share a consistent interface:

**Input:**
- `img1`: Tensor of shape `(C, H, W)` or `(B, C, H, W)`
- `img2`: Optional tensor for verification task

**Output:**
- `attribution_map`: NumPy array `(H, W)` with values in [0, 1]

**Usage Pattern:**
```python
from attributions import GradCAM, SHAPAttribution, LIMEAttribution

# Initialize
gradcam = GradCAM(model)
shap = SHAPAttribution(model)
lime = LIMEAttribution(model)

# Single image (explain embedding)
attr = method(face_image)

# Verification (explain similarity)
attr = method(face_image_1, face_image_2)
```

---

## Dependencies Added

Updated `/home/aaron/projects/xai/requirements.txt`:

```python
# Attribution Methods
shap>=0.42.0  # SHAP (Shapley Additive Explanations)
lime>=0.2.0.1  # LIME (Local Interpretable Model-agnostic Explanations)
scikit-image>=0.21.0  # For image segmentation (superpixels)
```

**Installation:**
```bash
pip install shap lime scikit-image
```

---

## Face Verification Adaptations

### Task 1: Embedding Explanation
**Question:** Which facial regions contribute to the embedding representation?

```python
attribution = method(face_image)
```

The attribution map highlights regions that strongly influence the learned embedding.

### Task 2: Verification Explanation
**Question:** Which facial regions contribute to matching two faces?

```python
attribution = method(face_image_1, face_image_2)
```

The attribution map highlights regions that contribute to the similarity/dissimilarity decision.

---

## Integration with Dissertation Experiments

### Experiment 6.1: Comparative Evaluation
- **Purpose:** Compare baseline methods (Grad-CAM, SHAP, LIME) with proposed methods
- **Metrics:**
  - Localization accuracy (pointing game, IoU with ground truth)
  - Computation time (seconds per image)
  - Faithfulness (insertion/deletion curves)
  - Stability (perturbation robustness)

### Experiment 6.2: Fairness Analysis
- **Purpose:** Use attributions to detect demographic biases
- **Approach:**
  - Generate attributions for different demographic groups
  - Compare which facial features are weighted differently
  - Identify potential sources of unfair bias

### Experiment 6.3: Model Interpretability
- **Purpose:** Explain verification decisions and failure modes
- **Approach:**
  - Visualize attributions for correct/incorrect verifications
  - Identify common failure patterns
  - Understand model decision boundaries

---

## Next Steps

### Phase 1: Complete Implementations (Priority)

1. **Grad-CAM (Estimated: 2-3 hours)**
   - Implement hook registration system
   - Add gradient capture and CAM computation
   - Test on ResNet-50 face verification model
   - Validate against reference implementation

2. **SHAP (Estimated: 3-4 hours)**
   - Integrate SHAP library (KernelSHAP or DeepSHAP)
   - Create background sample loader
   - Implement prediction wrapper for verification
   - Test on sample face images
   - Validate Shapley values sum correctly

3. **LIME (Estimated: 3-4 hours)**
   - Integrate LIME library
   - Implement SLIC superpixel segmentation
   - Create perturbation sampling
   - Implement prediction wrapper
   - Test on sample face images

### Phase 2: Testing & Validation

4. **Unit Tests**
   - Test each method independently
   - Verify output shape and range [0, 1]
   - Test both single-image and verification modes
   - Check edge cases (empty images, extreme values)

5. **Integration Tests**
   - Test with actual face verification models
   - Verify computational performance
   - Compare outputs across methods
   - Validate on benchmark datasets (LFW, CelebA)

### Phase 3: Benchmarking

6. **Performance Benchmarking**
   - Measure computation time per image
   - Compare memory usage
   - Profile bottlenecks
   - Optimize if needed

7. **Quality Benchmarking**
   - Compute localization accuracy metrics
   - Run faithfulness evaluations (insertion/deletion)
   - Measure stability under perturbations
   - Compare with reference implementations

### Phase 4: Dissertation Integration

8. **Experiment Scripts**
   - Create experiment runners for Chapter 6
   - Generate comparison figures
   - Compute quantitative metrics
   - Save results for analysis

9. **Visualization**
   - Create publication-quality visualizations
   - Generate heatmap overlays
   - Create comparison grids
   - Export figures for dissertation

---

## Testing

Run examples to verify structure:

```bash
cd /home/aaron/projects/xai
python -m src.attributions.example_usage
```

**Expected Output:**
- All three methods initialize successfully
- Return attribution maps of correct shape
- Values normalized to [0, 1]
- No errors or exceptions

**Note:** Current outputs will be random (placeholder) until full implementations are completed.

---

## Technical Specifications

### Grad-CAM Algorithm

1. **Forward Pass:**
   ```
   A^k = activations from target layer k
   y^c = model output for class/target c
   ```

2. **Backward Pass:**
   ```
   α_k^c = (1/Z) * Σ(∂y^c/∂A^k)  # Global average pooling of gradients
   ```

3. **CAM Generation:**
   ```
   L_GradCAM = ReLU(Σ_k α_k^c * A^k)
   ```

4. **Normalization:**
   ```
   L_GradCAM = upsample(L_GradCAM) / max(L_GradCAM)
   ```

### SHAP Algorithm

1. **Background Distribution:**
   ```
   X_bg = {x_1, x_2, ..., x_n}  # Representative samples
   ```

2. **Shapley Value:**
   ```
   φ_i = Σ_{S⊆F\{i}} [|S|!(|F|-|S|-1)!/|F|!] * [f_x(S∪{i}) - f_x(S)]
   ```

3. **Attribution:**
   ```
   SHAP(x) = {φ_1, φ_2, ..., φ_d}
   ```

### LIME Algorithm

1. **Superpixel Segmentation:**
   ```
   S = SLIC(image, n_segments=100)
   ```

2. **Perturbation:**
   ```
   X' = {x'_1, x'_2, ..., x'_m}  # Masked versions
   Z = {0,1}^K  # Binary masks
   ```

3. **Local Model:**
   ```
   ξ(x) = argmin_{g∈G} L(f, g, π_x) + Ω(g)
   ```

4. **Attribution:**
   ```
   LIME(x) = weights from g
   ```

---

## File Paths (Absolute)

All files located under: `/home/aaron/projects/xai/src/attributions/`

- `/home/aaron/projects/xai/src/attributions/gradcam.py`
- `/home/aaron/projects/xai/src/attributions/shap_wrapper.py`
- `/home/aaron/projects/xai/src/attributions/lime_wrapper.py`
- `/home/aaron/projects/xai/src/attributions/__init__.py`
- `/home/aaron/projects/xai/src/attributions/example_usage.py`
- `/home/aaron/projects/xai/src/attributions/README.md`
- `/home/aaron/projects/xai/src/attributions/IMPLEMENTATION_STATUS.md`

---

## Conclusion

The attribution methods module structure is **complete and ready for implementation**. All three baseline methods have:

✅ Consistent interfaces
✅ Proper documentation
✅ Example usage code
✅ Integration with requirements.txt
✅ Face verification adaptations

**Next priority:** Complete the actual implementations to replace placeholder code with functional attribution computation.

---

**Estimated Total Implementation Time:** 8-11 hours
- Grad-CAM: 2-3 hours
- SHAP: 3-4 hours
- LIME: 3-4 hours

**Estimated Testing & Validation Time:** 4-6 hours

**Total Project Completion:** 12-17 hours
