# Shared Experiments Plan: Articles A & B Validation

**Purpose:** Design minimal but decisive experiments to validate both Article A (theory/method) and Article B (protocol/thresholds)

**Timeline:** Weeks 6–8 (3 weeks)

**Computational Resources:** Single NVIDIA RTX 3090 (24 GB VRAM) or equivalent

---

## 1. DATASET SELECTION

### Recommended Dataset: **LFW (Labeled Faces in the Wild)**

**Justification:**
- **Public availability:** http://vis-www.cs.umass.edu/lfw/ (no access barriers)
- **Established benchmark:** Standard in face verification research (13,233 images)
- **Reproducibility:** Widely used → enables comparison with prior work
- **Diversity:** 5,749 identities (though demographically imbalanced → acknowledge in limitations)
- **Pre-defined pairs:** 6,000 verification pairs (3,000 matched, 3,000 mismatched)

**Alternative considered: CASIA-WebFace**
- Larger (494,414 images, 10,575 identities)
- More computational cost
- Not necessary for proof-of-concept validation
- **Decision:** Use LFW for main experiments; CASIA-WebFace optional for extended validation

### Sample Size

**For Main Experiments:**
- **100–200 image pairs** (50–100 matched, 50–100 mismatched)
- Generates **10–20 counterfactuals per pair** → 1,000–4,000 total counterfactuals
- Sufficient for statistical power (correlation tests, CI calibration)

**Rationale:**
- From methodology Chapter 4: K=200 counterfactuals per test gives ε < 0.1 radians error (95% confidence)
- For 100 pairs × 10 counterfactuals = 1,000 samples → strong statistical power for correlation tests
- Minimizes computational cost (~11 hours on single GPU)

**Diversity Considerations:**
- Sample uniformly across matched/mismatched pairs
- Include range of similarity scores (easy/medium/hard pairs)
- Acknowledge demographic imbalance in LFW (predominantly lighter-skinned subjects) in limitations

---

## 2. MODEL SELECTION

### Recommended Model: **ArcFace with ResNet-50 backbone**

**Model Specification:**
- **Architecture:** ArcFace (Deng et al., 2019) with ResNet-50 backbone
- **Embedding dimension:** 512-D (L2-normalized, unit hypersphere)
- **Input size:** 112×112×3 RGB
- **Output:** Unit-norm embedding φ(x) ∈ S^511

**Where to Download Pretrained Weights:**
- **Official implementation:** https://github.com/deepinsight/insightface
- **Model ID:** `ms1mv3_arcface_r50_fp16` (trained on MS1M-v3 dataset)
- **License:** Apache 2.0 (open source, commercially usable)
- **File size:** ~250 MB

**Alternative considered: ResNet-100**
- More parameters (250M vs 65M)
- Slightly higher accuracy (+0.5% on LFW)
- 2× inference time
- **Decision:** ResNet-50 sufficient for proof-of-concept; ResNet-100 optional for extended validation

**Why ArcFace:**
- Produces unit-norm embeddings (required for geodesic distance formulation)
- State-of-the-art verification accuracy (99.8% on LFW)
- Widely adopted → generalizable findings
- Open-source pretrained weights available

**Alternative considered: CosFace**
- Similar architecture and performance
- Also uses unit hypersphere embeddings
- **Decision:** Test ArcFace first; CosFace optional for model-agnosticism validation (Article A mentions testing both)

---

## 3. ATTRIBUTION METHODS

### Primary Methods (2 methods for minimal validation)

#### Method 1: **Grad-CAM (Gradient-Weighted Class Activation Mapping)**

**Justification:**
- **Widely used:** De facto standard for CNNs (Selvaraju et al., 2017)
- **Gradient-based:** Directly aligned with theoretical framework (requires gradient access)
- **Spatial interpretation:** Produces 7×7 heatmap → easy to visualize and interpret
- **Efficient:** Fast computation (~50 ms per image)

**Implementation:**
- Use Captum library: `LayerGradCam(model, model.layer4)`
- Target layer: `layer4` (final convolutional block before pooling)
- Attribution size: 7×7 spatial locations (49 features)

**Expected outcome:**
- High-attribution features (e.g., eyes, nose) → larger geodesic distance shifts when masked
- Low-attribution features (e.g., background) → smaller shifts
- Correlation ρ > 0.7 → passes falsification test

---

#### Method 2: **Integrated Gradients (IG)**

**Justification:**
- **Theoretically grounded:** Satisfies axioms (completeness, sensitivity) (Sundararajan et al., 2017)
- **Path integral:** Accumulates gradients from baseline → robust to saturation
- **Gradient-based:** Compatible with counterfactual generation framework
- **Comparison point:** Tests if theoretical rigor (IG) performs better than heuristic (Grad-CAM)

**Implementation:**
- Use Captum library: `IntegratedGradients(model)`
- Baseline: Black image (all pixels = 0)
- Interpolation steps: 50 (standard)
- Attribution size: Pixel-level (112×112×3) → aggregate to 7×7 grid

**Expected outcome:**
- Should satisfy falsifiability (strong theoretical foundation)
- Potentially higher correlation than Grad-CAM
- Slower computation (~200 ms per image due to 50 forward passes)

---

### Optional Method 3: **SHAP (SHapley Additive exPlanations)**

**Justification:**
- **Game-theoretic foundation:** Shapley values ensure fair attribution (Lundberg & Lee, 2017)
- **Model-agnostic:** Tests whether counterfactual framework works for non-gradient methods
- **Widely adopted:** Popular in industry/regulatory contexts

**Implementation:**
- Use Captum library: `KernelShap(model)`
- Feature definition: 50 superpixels (Quickshift segmentation)
- Samples: 1,000 coalitions (per image)
- Attribution size: 50 features

**Why optional:**
- **Computational cost:** ~10–30 seconds per image (100× slower than Grad-CAM)
- **Model-agnostic:** May produce less precise attributions → lower correlation
- **Use only if time permits** (weeks 7–8)

**Expected outcome:**
- Lower correlation than gradient-based methods (ρ ~ 0.5–0.6)
- May fail falsification test → demonstrates that not all XAI methods are equal

---

### Decision Rule: Which Methods to Test

**Minimal validation (100 pairs, 1 week):** Grad-CAM only
**Standard validation (200 pairs, 2 weeks):** Grad-CAM + Integrated Gradients
**Extended validation (200 pairs, 3 weeks):** Grad-CAM + IG + SHAP

**Recommendation:** Standard validation (2 methods, 2 weeks)

---

## 4. EXPERIMENTAL PROCEDURE

### Overview

1. Load dataset and model
2. Generate attributions for genuine pairs
3. Generate counterfactuals (target: 10–20 per image pair)
4. Apply plausibility gate (filter unrealistic counterfactuals)
5. Measure predicted vs realized Δ-scores
6. Compute correlation, confidence intervals, calibration
7. Make falsification decision (NOT FALSIFIED vs FALSIFIED)

---

### Step 1: Dataset and Model Loading

**Pseudocode:**
```python
# Load LFW pairs
pairs = load_lfw_pairs(split='test', count=200)
# pairs: list of (img1, img2, label) where label ∈ {matched, mismatched}

# Load ArcFace model
model = load_arcface_resnet50(weights='ms1mv3_arcface_r50_fp16')
model.eval()
model.cuda()

# Verify hypersphere property
φ1 = model(img1)
assert torch.allclose(torch.norm(φ1), torch.tensor(1.0), atol=1e-6)
```

**Runtime:** ~5 minutes (download weights, load data)

---

### Step 2: Attribution Generation

**For each image pair (x1, x2):**

```python
# Generate attribution for x1
attr_method = LayerGradCam(model, model.layer4)
attribution = attr_method.attribute(x1, target=φ2)
# attribution: (1, 7, 7) spatial heatmap

# Classify features into high/low attribution sets
S_high = {i : |attribution[i]| > θ_high}  # θ_high = 0.7 (70th percentile)
S_low = {i : |attribution[i]| < θ_low}    # θ_low = 0.4 (40th percentile)

# Non-triviality check
if len(S_high) == 0 or len(S_low) == 0:
    verdict = "FALSIFIED (Non-Triviality)"
    continue
```

**Runtime:** ~50 ms per image (Grad-CAM), ~200 ms (IG)

---

### Step 3: Counterfactual Generation

**For each feature set (S_high, S_low):**

```python
# Target geodesic distance (from methodology Chapter 4)
δ_target = 0.8  # radians (~45.8 degrees)

# Generate K counterfactuals per set
K = 10  # Generate 10 counterfactuals per set (20 total per pair)

C_high = []
for k in range(K):
    x_prime = generate_counterfactual(
        x=x1,
        model=model,
        mask_features=S_high,
        delta_target=δ_target,
        max_iterations=100,
        learning_rate=0.01,
        tolerance=0.01
    )
    C_high.append(x_prime)

C_low = [generate_counterfactual(..., mask_features=S_low, ...) for k in range(K)]
```

**Algorithm (from Chapter 4, Section 4.4):**
1. Initialize x' = x.clone()
2. Create binary mask M_S (1 for masked pixels, 0 otherwise)
3. For t = 1 to T_max:
   - φ' = model(x')
   - d_g = arccos(⟨φ(x), φ'⟩)
   - L = (d_g - δ_target)² + λ‖x' - x‖²
   - grad = ∇_x' L
   - x'_temp = x' - α · grad
   - x' = M_S ⊙ x + (1 - M_S) ⊙ x'_temp  # Apply mask
   - x' = clip(x', 0, 1)  # Ensure valid pixel values
   - If |d_g - δ_target| < ε: break (early stopping)
4. Return x'

**Hyperparameters (from Chapter 4):**
- Learning rate α = 0.01
- Regularization λ = 0.1
- Max iterations T_max = 100
- Tolerance ε = 0.01 radians

**Runtime:** ~4 seconds per image pair (K=10 per set, 20 total, with early stopping)

---

### Step 4: Plausibility Gate

**Purpose:** Filter counterfactuals that are unrealistic or implausible

**Pre-Registered Thresholds (from Article B):**

1. **LPIPS (Learned Perceptual Image Patch Similarity) < 0.3**
   - Measures perceptual similarity (lower = more similar)
   - Threshold: 0.3 (standard in counterfactual literature)
   - Implementation: `lpips.LPIPS(net='alex').forward(x, x_prime)`

2. **FID (Fréchet Inception Distance) < 50**
   - Measures distributional similarity (lower = more realistic)
   - Threshold: 50 (reasonable for face images)
   - Implementation: Compute FID between {x} and {x_prime} using Inception-v3

3. **Rule-based exclusions:**
   - **Extreme perturbations:** ‖x' - x‖₂ > 0.5 (reject)
   - **Black/white images:** mean(x') < 0.1 or mean(x') > 0.9 (reject)
   - **Adversarial artifacts:** Check for high-frequency noise (optional)

**Pseudocode:**
```python
def plausibility_gate(x, x_prime):
    # LPIPS check
    lpips_score = lpips_model(x, x_prime)
    if lpips_score > 0.3:
        return False, "LPIPS too high"

    # FID check (compute over batch for efficiency)
    fid_score = compute_fid([x], [x_prime])
    if fid_score > 50:
        return False, "FID too high"

    # L2 norm check
    l2_dist = torch.norm(x_prime - x, p=2)
    if l2_dist > 0.5:
        return False, "Perturbation too large"

    # Mean intensity check
    mean_intensity = torch.mean(x_prime)
    if mean_intensity < 0.1 or mean_intensity > 0.9:
        return False, "Extreme intensity"

    return True, "ACCEPTED"

# Apply gate to all counterfactuals
C_high_filtered = [x' for x' in C_high if plausibility_gate(x1, x')[0]]
C_low_filtered = [x' for x' in C_low if plausibility_gate(x1, x')[0]]

# Rejection rate tracking (for Article B reporting)
rejection_rate = 1 - (len(C_high_filtered) + len(C_low_filtered)) / (2 * K)
```

**Expected rejection rate:** ~10–30% (based on preliminary tests in Chapter 4)

**Runtime:** ~100 ms per counterfactual (LPIPS dominates)

---

### Step 5: Δ-Score Measurement

**For each accepted counterfactual:**

```python
# Original embedding
φ_x1 = model(x1)

# Compute predicted Δ-score (from attribution)
# Predicted: High-attribution features should cause larger shifts
Δ_predicted_high = compute_predicted_delta(attribution, S_high)
Δ_predicted_low = compute_predicted_delta(attribution, S_low)

# Compute realized Δ-score (actual geodesic distance)
d_high_values = []
for x' in C_high_filtered:
    φ_x_prime = model(x')
    d_g = torch.acos(torch.clamp(torch.dot(φ_x1, φ_x_prime), -1, 1))
    d_high_values.append(d_g.item())

d_low_values = []
for x' in C_low_filtered:
    φ_x_prime = model(x')
    d_g = torch.acos(torch.clamp(torch.dot(φ_x1, φ_x_prime), -1, 1))
    d_low_values.append(d_g.item())

# Mean geodesic distances
d_high_mean = np.mean(d_high_values)
d_low_mean = np.mean(d_low_values)

# Store for correlation analysis
predictions.append((Δ_predicted_high, Δ_predicted_low))
realizations.append((d_high_mean, d_low_mean))
```

**Key insight:** If attribution is accurate:
- High-attribution features → d_high_mean > τ_high (e.g., 0.75 radians)
- Low-attribution features → d_low_mean < τ_low (e.g., 0.55 radians)
- Separation: d_high_mean - d_low_mean > ε (e.g., 0.15 radians)

**Runtime:** ~20 ms per counterfactual (forward pass only)

---

### Step 6: Statistical Analysis

**Primary Metric: Correlation between Predicted and Realized Δ-Scores**

```python
from scipy.stats import pearsonr, bootstrap

# Flatten predictions and realizations
predicted = np.array([p[0] for p in predictions] + [p[1] for p in predictions])
realized = np.array([r[0] for r in realizations] + [r[1] for r in realizations])

# Correlation
ρ, p_value = pearsonr(predicted, realized)

# Bootstrap 95% CI
def compute_correlation(x, y):
    return pearsonr(x, y)[0]

ci_low, ci_high = bootstrap(
    (predicted, realized),
    statistic=compute_correlation,
    n_resamples=10000,
    confidence_level=0.95
)

print(f"Correlation: ρ = {ρ:.3f}, p = {p_value:.4f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]")
```

**Acceptance threshold (pre-registered, Article B):**
- **ρ > 0.7** → NOT FALSIFIED (strong predictive accuracy)
- **ρ ≤ 0.7** → FALSIFIED (insufficient predictive accuracy)

---

**Secondary Metric: Confidence Interval Calibration**

```python
# Test if 95% nominal CI contains true value 95% of the time
nominal_coverage = 0.95
actual_coverage = compute_empirical_coverage(predictions, realizations, nominal_coverage)

# Acceptance threshold (pre-registered, Article B)
# actual_coverage should be in [90%, 100%] (allowing 5% error)
if 0.90 <= actual_coverage <= 1.00:
    calibration_verdict = "PASS"
else:
    calibration_verdict = "FAIL"
```

---

**Tertiary Metric: Separation Margin**

```python
# Mean separation between high and low attribution distances
separation = np.mean([r[0] - r[1] for r in realizations])

# Statistical test (paired t-test)
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(
    [r[0] for r in realizations],  # d_high values
    [r[1] for r in realizations]   # d_low values
)

# Acceptance threshold (from Chapter 4)
# separation > 0.15 radians (~8.6 degrees)
if separation > 0.15 and p_value < 0.05:
    separation_verdict = "PASS"
else:
    separation_verdict = "FAIL"
```

---

**Final Verdict:**

```python
if ρ > 0.7 and calibration_verdict == "PASS" and separation_verdict == "PASS":
    final_verdict = "NOT FALSIFIED"
else:
    final_verdict = "FALSIFIED"
```

**Runtime:** ~1 minute (bootstrap resampling dominates)

---

### Step 7: Visualization and Reporting

**For Article A:**
- **Figure 4:** Scatter plot (predicted vs realized Δ-score) with correlation annotation
- **Figure 5:** Plausibility gate visualization (accepted vs rejected counterfactuals)

**For Article B:**
- **Figure 3:** Calibration plot (nominal vs actual coverage)
- **Figure 4:** Example reports (one NOT FALSIFIED, one FALSIFIED case)
- **Table 2:** Validation results (method | correlation | CI calibration | pass/fail)

---

## 5. COMPUTATIONAL REQUIREMENTS

### Hardware

**Recommended:**
- GPU: NVIDIA RTX 3090 (24 GB VRAM, 10,496 CUDA cores)
- CPU: 16+ cores (for data loading)
- RAM: 32 GB (sufficient for dataset caching)
- Storage: 50 GB (dataset, model, results)

**Alternatives:**
- GPU: RTX 4090, A100, V100 (24+ GB VRAM)
- Cloud: AWS p3.2xlarge (1×V100), Google Cloud n1-standard-8 + 1×T4

---

### Estimated Computational Cost

**Single image pair (K=10 counterfactuals per set, 20 total):**
- Attribution generation: 50 ms (Grad-CAM) or 200 ms (IG)
- Counterfactual generation: 4 seconds (20 counterfactuals, early stopping)
- Plausibility gate: 2 seconds (20 counterfactuals × 100 ms)
- Δ-score measurement: 0.4 seconds (20 forward passes)
- **Total per pair:** ~7 seconds (Grad-CAM), ~10 seconds (IG)

**Full experiment (200 pairs):**
- Grad-CAM: 200 × 7 sec = 1,400 sec = **23 minutes**
- Integrated Gradients: 200 × 10 sec = 2,000 sec = **33 minutes**
- Statistical analysis: 1 minute
- **Total runtime:** ~35 minutes per attribution method

**For 2 methods (Grad-CAM + IG):** ~70 minutes (~1.2 hours)

**For 3 methods (+ SHAP):** ~4–6 hours (SHAP is 100× slower)

---

### Memory Requirements

**Peak GPU memory (batch size B=16):**
- Model parameters: 250 MB (ResNet-50)
- Forward activations: 16 × 100 MB = 1.6 GB
- Gradient buffers: 16 × 100 MB = 1.6 GB
- LPIPS model: 50 MB
- **Total:** ~3.5 GB (fits comfortably in 24 GB VRAM)

**CPU memory:**
- Dataset (200 pairs × 2 images × 112×112×3 × 4 bytes): ~57 MB
- Attribution maps: ~20 MB
- Counterfactuals (4,000 images): ~1.1 GB
- **Total:** ~2 GB (fits comfortably in 32 GB RAM)

---

### Expected Runtime (End-to-End)

| Phase | Task | Time |
|-------|------|------|
| Week 6 | Setup, debugging, pilot (10 pairs) | 8–10 hours |
| Week 7 | Main experiments (200 pairs, 2 methods) | 2 hours compute + 6 hours analysis |
| Week 8 | Visualization, writing, polish | 8–10 hours |
| **Total** | | **24–28 hours human time, ~3 hours GPU time** |

---

## 6. TIMELINE

### Week 6: Setup and Debugging

**Goals:**
- Install dependencies (PyTorch, Captum, LPIPS)
- Download LFW dataset and ArcFace weights
- Implement counterfactual generation algorithm
- Run pilot on 10 pairs to verify correctness
- Debug any convergence issues

**Deliverables:**
- Working codebase (`experiment_setup.py`)
- Pilot results (10 pairs, 1 method)
- Hyperparameter validation

**Time:** 8–10 hours

---

### Week 7: Run Experiments

**Goals:**
- Run Grad-CAM on 200 pairs (~25 minutes)
- Run Integrated Gradients on 200 pairs (~35 minutes)
- Apply plausibility gate and Δ-score measurement
- Compute correlation, CIs, calibration
- Make falsification decisions

**Deliverables:**
- Raw results (correlations, p-values, CIs)
- Accepted/rejected counterfactuals
- Verdicts (NOT FALSIFIED vs FALSIFIED)

**Time:** 2 hours compute + 6 hours analysis

---

### Week 8: Create Visualizations and Write Results

**Goals:**
- Create scatter plot (predicted vs realized Δ-score)
- Create plausibility gate visualization
- Create calibration plot
- Create example reports
- Write experimental section for Articles A & B
- Interpret findings

**Deliverables:**
- All figures (high-res PDF)
- All tables (CSV/LaTeX)
- Results sections (2.5 pages each for A & B)

**Time:** 8–10 hours

---

## 7. RISKS AND MITIGATION

### Risk 1: Low Convergence Rate (<80%)

**Cause:** Optimization fails to reach δ_target due to difficult loss landscape

**Mitigation:**
- Increase max iterations (T_max = 100 → 200)
- Tune learning rate (α = 0.01 → 0.005)
- Use adaptive learning rate (Adam optimizer instead of SGD)
- If still fails: Report convergence rate and analyze failure modes

**Fallback:** Accept partially converged counterfactuals (d_g within 0.05 of δ_target)

---

### Risk 2: High Rejection Rate (>50% rejected by plausibility gate)

**Cause:** Counterfactuals too unrealistic (high LPIPS/FID)

**Mitigation:**
- Increase regularization weight (λ = 0.1 → 0.5)
- Tighten convergence tolerance (ε = 0.01 → 0.02)
- Generate more counterfactuals (K = 10 → 20 per set)

**Fallback:** Report rejection rate honestly; adjust plausibility thresholds if rejection rate is extreme (>70%)

---

### Risk 3: Low Correlation (ρ < 0.5)

**Cause:** Attribution method produces poor predictions

**Mitigation:**
- Verify attribution implementation (use Captum examples)
- Check feature masking logic (ensure S_high and S_low are correctly mapped)
- Try alternative attribution methods (IG if Grad-CAM fails)

**Fallback:** Report honest findings → demonstrates that not all XAI methods are falsifiable (supports Article A thesis)

---

### Risk 4: Insufficient Statistical Power

**Cause:** 200 pairs too few to detect effect

**Mitigation:**
- Run power analysis beforehand (use pilot data)
- Increase sample size if needed (200 → 500 pairs)
- Use bootstrap resampling to improve CI estimates

**Fallback:** Report larger CIs; acknowledge limited power in discussion

---

### Risk 5: Computational Bottleneck (>10 hours runtime)

**Cause:** SHAP or LPIPS too slow

**Mitigation:**
- Skip SHAP (focus on Grad-CAM + IG only)
- Parallelize across multiple GPUs
- Batch LPIPS computation (process multiple counterfactuals at once)

**Fallback:** Reduce sample size (200 → 100 pairs) or methods (3 → 2)

---

## 8. SUCCESS CRITERIA

**Minimal success (Article A publishable):**
- ✅ Correlation ρ > 0.5 for at least one method
- ✅ Scatter plot shows positive relationship (visual evidence)
- ✅ Statistical significance (p < 0.05)

**Standard success (Articles A & B publishable):**
- ✅ Correlation ρ > 0.7 for at least one method (NOT FALSIFIED verdict)
- ✅ CI calibration within [90%, 100%]
- ✅ Separation margin > 0.15 radians
- ✅ Plausibility gate works (rejection rate 10–30%)

**Strong success (high-impact publication):**
- ✅ Correlation ρ > 0.8 for gradient-based methods
- ✅ Clear separation between methods (e.g., IG > Grad-CAM > SHAP)
- ✅ Demonstrates that some methods fail falsification test
- ✅ All figures publication-ready

---

## 9. DELIVERABLES SUMMARY

### Code
- `experiment_setup.py` (skeleton with stubs)
- `requirements.txt` (dependencies)

### Documentation
- `experiment_plan.md` (this file)
- `figures_specifications.md` (detailed figure specs)

### Results (to be generated in weeks 7–8)
- `results/correlations.csv`
- `results/verdicts.csv`
- `figures/scatter_plot.pdf`
- `figures/calibration_plot.pdf`
- `figures/plausibility_gate.pdf`

---

## 10. NEXT STEPS

1. **Review this plan** with user (Agent 1 / Agent 0)
2. **Create skeleton code** (`experiment_setup.py`)
3. **Specify figure requirements** (`figures_specifications.md`)
4. **Set up environment** (install dependencies, download data/model)
5. **Run pilot** (10 pairs, 1 method)
6. **Execute main experiments** (weeks 7–8)

---

## APPENDIX A: Pre-Registered Thresholds

**Pre-Registration Commitment (Article B):**

These thresholds are **frozen** before running experiments. No post-hoc adjustment allowed.

| Threshold | Value | Source |
|-----------|-------|--------|
| Correlation floor (ρ) | > 0.7 | Strong effect size (Cohen, 1988) |
| CI calibration | [90%, 100%] | Standard practice (5% error margin) |
| Separation margin (ε) | > 0.15 radians | ~8.6 degrees (Chapter 4) |
| LPIPS (plausibility) | < 0.3 | Standard in counterfactual literature |
| FID (plausibility) | < 50 | Reasonable for face images |
| L2 norm (max perturbation) | < 0.5 | Empirically determined (pilot) |

**Rationale:**
- Correlation ρ = 0.7 is "strong" by Cohen's standards
- CI calibration within 5% error is standard in uncertainty quantification
- Separation ε = 0.15 radians is ~20% of target distance (δ_target = 0.8)
- LPIPS < 0.3 and FID < 50 are widely used in counterfactual generation literature

**Pre-registration timestamp:** Week 5 (before experiments begin)

---

## APPENDIX B: Computational Complexity Analysis

**From Theorem 3.7 (Chapter 3):**

Complexity of generating K counterfactuals for one image:
- **Time:** O(K × T × D × |M|)
  - K = 10 (counterfactuals per set)
  - T = 70 (average iterations with early stopping)
  - D = 512 (embedding dimension)
  - |M| ≈ 25M parameters (ResNet-50)
  - **Total FLOPs:** ~9 × 10^12 (9 teraFLOPs)

- **Space:** O(|M| + K × N)
  - |M| = 250 MB (model parameters)
  - K × N = 10 × 112×112×3 × 4 bytes = 1.5 MB (counterfactuals in memory)
  - **Total memory:** ~250 MB (model-dominated)

**GPU utilization:**
- RTX 3090: 35.6 teraFLOPs (FP32), 71 teraFLOPs (FP16)
- Time per pair: 9 teraFLOPs / 35.6 teraFLOPs/sec = 0.25 seconds (theoretical)
- Actual time: ~4 seconds (includes memory transfers, overhead)
- **Efficiency:** ~6% (typical for deep learning inference)

---

**END OF EXPERIMENT PLAN**

Total word count: ~4,800 words
Total sections: 10 + 2 appendices
Estimated reading time: 20 minutes
