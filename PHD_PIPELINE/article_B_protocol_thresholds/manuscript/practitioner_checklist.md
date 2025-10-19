# PRACTITIONER CHECKLIST: FALSIFIABLE ATTRIBUTION VALIDATION PROTOCOL

## Purpose

This checklist provides step-by-step guidance for forensic analysts, AI auditors, and legal professionals implementing the falsifiable attribution validation protocol. Follow each step sequentially to ensure rigorous, reproducible validation of face verification explanations.

---

## Table of Contents

1. Pre-Deployment Preparation
2. Running the Falsification Protocol
3. Interpreting Results
4. Filling Out the Forensic Report
5. Disclosure and Documentation Requirements
6. Troubleshooting Common Issues

---

## Section 1: PRE-DEPLOYMENT PREPARATION

### 1.1 System Requirements

**Before beginning validation, verify you have:**

- [ ] **Hardware:** GPU with ≥16 GB VRAM (recommended: NVIDIA RTX 3090 or equivalent)
  - Minimum: NVIDIA GTX 1080 Ti (11 GB), reduce batch size to B=4
  - CPU-only execution: possible but 15× slower (~1-2 minutes per image)

- [ ] **Software:**
  - [ ] Python 3.8 or higher
  - [ ] PyTorch 2.0 or higher with CUDA 11.8
  - [ ] Captum library v0.6.0 or higher
  - [ ] NumPy, SciPy, Matplotlib (standard scientific stack)

- [ ] **Face Verification Model:**
  - [ ] Pretrained ArcFace or CosFace model (512-D L2-normalized embeddings)
  - [ ] Model checkpoint file (.pth or .onnx format)
  - [ ] Model source documented (official release or in-house training)

- [ ] **Datasets:**
  - [ ] Calibration set: 500 images (separate from test set, used for threshold determination)
  - [ ] Test set: ≥500 image pairs (recommended: 1,000 for statistical power)
  - [ ] Metadata: Demographic annotations (age, gender, skin tone) if available

### 1.2 Pre-Registration (CRITICAL)

**Complete BEFORE running validation on test set:**

- [ ] **Freeze thresholds:**
  - [ ] $\theta_{\text{high}} = 0.7$ (high-attribution classification threshold)
  - [ ] $\theta_{\text{low}} = 0.4$ (low-attribution classification threshold)
  - [ ] $\tau_{\text{high}} = 0.75$ radians (high-attribution distance floor)
  - [ ] $\tau_{\text{low}} = 0.55$ radians (low-attribution distance ceiling)
  - [ ] $\epsilon = 0.15$ radians (separation margin)
  - [ ] $\rho_{\text{min}} = 0.7$ (correlation floor for primary endpoint)

- [ ] **Document calibration procedure:**
  - [ ] Record calibration set composition (identities, size)
  - [ ] Verify no overlap between calibration set and test set
  - [ ] Compute percentile statistics for $\theta_{\text{high}}$ and $\theta_{\text{low}}$

- [ ] **Submit pre-registration:**
  - [ ] Draft pre-registration document (use template from `pre_registration.md`)
  - [ ] Submit to Open Science Framework (OSF) or AsPredicted.org
  - [ ] Obtain timestamped URL
  - [ ] Generate SHA-256 cryptographic hash of pre-registration document

**WARNING:** Adjusting thresholds after viewing test set results constitutes p-hacking and invalidates scientific validity.

### 1.3 Code Setup and Testing

**Verify implementation correctness:**

- [ ] **Download reference implementation:**
  - [ ] Clone GitHub repository: `git clone [REPOSITORY_URL]`
  - [ ] Install dependencies: `pip install -r requirements.txt`
  - [ ] Run unit tests: `pytest tests/`

- [ ] **Validate on toy example:**
  - [ ] Load provided test image pair (`tests/data/test_pair_01.jpg`)
  - [ ] Run attribution extraction (Grad-CAM)
  - [ ] Verify attribution map dimensions (7×7 for Grad-CAM)
  - [ ] Generate 10 counterfactuals (K=10 for quick test)
  - [ ] Verify convergence (should achieve $|d_g - 0.8| < 0.01$ for >90% of samples)

- [ ] **Benchmark computational performance:**
  - [ ] Measure runtime for single image (should be ~4-9 seconds on RTX 3090)
  - [ ] Estimate total runtime: (Number of test images) × (seconds per image) / 3600 ≈ hours
  - [ ] Verify GPU memory usage (should be <8 GB for B=16 batch size)

### 1.4 Data Preparation

**Prepare test set for validation:**

- [ ] **Load and inspect test data:**
  - [ ] Verify image pairs are correctly formatted (112×112×3 RGB)
  - [ ] Check verification labels (genuine vs. impostor) if available
  - [ ] Confirm no corrupted images (read all files without errors)

- [ ] **Demographic annotation (if available):**
  - [ ] Load age/gender/skin tone labels
  - [ ] Verify annotation completeness (flag missing annotations)
  - [ ] Compute demographic composition statistics

- [ ] **Quality filtering (optional but recommended):**
  - [ ] Exclude images with resolution <80×80 pixels
  - [ ] Exclude images with extreme poses (>45° rotation)
  - [ ] Exclude images with heavy occlusion (>50% face area)
  - [ ] Document exclusion criteria and counts

---

## Section 2: RUNNING THE FALSIFICATION PROTOCOL

### 2.1 Step-by-Step Execution

For **each test image pair** $(x, x')$, perform the following:

#### Step 2.1.1: Attribution Extraction

- [ ] **Load image and model:**
  ```python
  image = load_image("path/to/image.jpg")  # Returns (1, 3, 112, 112) tensor
  model = load_face_verification_model("arcface_r100.pth")
  model.eval()  # Set to evaluation mode
  ```

- [ ] **Extract attribution map:**
  - [ ] Choose attribution method (Grad-CAM, SHAP, LIME, Integrated Gradients)
  - [ ] Compute attribution: `attribution = gradcam.attribute(image, model)`
  - [ ] Verify output dimensions (7×7 for Grad-CAM, 50 for SHAP/LIME)
  - [ ] Visualize attribution (overlay heatmap on image for sanity check)

- [ ] **Record attribution extraction time:** ____________ ms

#### Step 2.1.2: Feature Classification

- [ ] **Apply thresholds:**
  ```python
  S_high = {i for i in range(len(attribution)) if abs(attribution[i]) > 0.7}
  S_low = {i for i in range(len(attribution)) if abs(attribution[i]) < 0.4}
  ```

- [ ] **Check non-triviality:**
  - [ ] Verify $|S_{\text{high}}| > 0$ (at least one high-attribution feature)
  - [ ] Verify $|S_{\text{low}}| > 0$ (at least one low-attribution feature)
  - [ ] If either is empty, mark as **"FALSIFIED (Non-Triviality)"** and skip to next image

- [ ] **Record feature set sizes:**
  - $|S_{\text{high}}|$ = ____________
  - $|S_{\text{low}}|$ = ____________

#### Step 2.1.3: Counterfactual Generation

**For each feature set** ($S_{\text{high}}$ and $S_{\text{low}}$):

- [ ] **Initialize counterfactual generation:**
  - [ ] Set target distance: $\delta_{\text{target}} = 0.8$ radians
  - [ ] Set sample size: $K = 200$
  - [ ] Set max iterations: $T = 100$
  - [ ] Set learning rate: $\alpha = 0.01$
  - [ ] Set regularization: $\lambda = 0.1$

- [ ] **Generate counterfactuals:**
  ```python
  counterfactuals_high = []
  for k in range(200):
      x_cf, converged, final_dist, iters = generate_counterfactual(
          image, model, S_high, delta_target=0.8, T=100, alpha=0.01, lambda_reg=0.1
      )
      counterfactuals_high.append((x_cf, converged, final_dist, iters))
  ```

- [ ] **Verify convergence:**
  - [ ] Count converged samples (should be ≥180 out of 200)
  - [ ] If <180 converged, flag as potential issue (continue but note in report)
  - [ ] Compute mean iterations: ____________
  - [ ] Compute mean final distance: ____________ radians

- [ ] **Check plausibility gates:**
  - [ ] Compute median LPIPS: ____________ (should be <0.3)
  - [ ] Compute FID score: ____________ (should be <50)
  - [ ] If plausibility gates violated, flag as **"FALSIFIED (Plausibility)"**

- [ ] **Record generation time:**
  - High-attribution counterfactuals: ____________ seconds
  - Low-attribution counterfactuals: ____________ seconds

#### Step 2.1.4: Geodesic Distance Measurement

- [ ] **Compute original embedding:**
  ```python
  phi_x = model(image)  # Shape: (1, 512), L2-normalized
  ```

- [ ] **Measure distances for high-attribution counterfactuals:**
  ```python
  distances_high = []
  for x_cf, converged, _, _ in counterfactuals_high:
      if converged:
          phi_cf = model(x_cf)
          cos_sim = torch.sum(phi_x * phi_cf)
          d_g = torch.acos(torch.clamp(cos_sim, -1.0+1e-7, 1.0-1e-7))
          distances_high.append(d_g.item())
  ```

- [ ] **Compute summary statistics:**
  - Mean: $\bar{d}_{\text{high}}$ = ____________ radians
  - Std: $\sigma_{\text{high}}$ = ____________ radians
  - Median: ____________ radians
  - Min/Max: ____________ / ____________ radians

- [ ] **Repeat for low-attribution counterfactuals:**
  - Mean: $\bar{d}_{\text{low}}$ = ____________ radians
  - Std: $\sigma_{\text{low}}$ = ____________ radians

#### Step 2.1.5: Statistical Hypothesis Testing

- [ ] **Test 1 (High-Attribution):**
  ```python
  t_high = (d_high_mean - 0.75) / (d_high_std / sqrt(200))
  p_high = 1 - stats.t.cdf(t_high, df=199)  # One-tailed upper
  ```
  - Test statistic: $t_{\text{high}}$ = ____________
  - P-value: $p_{\text{high}}$ = ____________
  - Decision (p < 0.025): **PASS** or **FAIL**

- [ ] **Test 2 (Low-Attribution):**
  ```python
  t_low = (d_low_mean - 0.55) / (d_low_std / sqrt(200))
  p_low = stats.t.cdf(t_low, df=199)  # One-tailed lower
  ```
  - Test statistic: $t_{\text{low}}$ = ____________
  - P-value: $p_{\text{low}}$ = ____________
  - Decision (p < 0.025): **PASS** or **FAIL**

- [ ] **Final Verdict:**
  - [ ] Non-Triviality: PASS
  - [ ] Test 1 (High): PASS / FAIL
  - [ ] Test 2 (Low): PASS / FAIL
  - [ ] Separation Margin ($\tau_{\text{high}} > \tau_{\text{low}} + \epsilon$): PASS (by design)
  - [ ] **Overall Verdict:** **NOT FALSIFIED** (all PASS) or **FALSIFIED** (any FAIL)

- [ ] **Record results in spreadsheet:**
  - Image ID: ____________
  - Verdict: ____________
  - $\bar{d}_{\text{high}}$: ____________
  - $\bar{d}_{\text{low}}$: ____________
  - $\Delta = \bar{d}_{\text{high}} - \bar{d}_{\text{low}}$: ____________
  - $p_{\text{high}}$: ____________
  - $p_{\text{low}}$: ____________

### 2.2 Batch Processing (for Large Datasets)

**For N ≥ 100 images, use batch processing:**

- [ ] **Parallelize across images:**
  ```python
  from multiprocessing import Pool

  with Pool(processes=4) as pool:
      results = pool.map(run_falsification_protocol, image_list)
  ```

- [ ] **Monitor progress:**
  - [ ] Use progress bar (e.g., `tqdm`) to track completion
  - [ ] Log intermediate results to CSV (in case of crashes)
  - [ ] Save checkpoints every 100 images

- [ ] **Estimate remaining time:**
  - Images processed: ____________ / ____________
  - Avg time per image: ____________ seconds
  - Estimated completion: ____________ hours

### 2.3 Quality Control Checks

**After processing all images, verify data integrity:**

- [ ] **Check for missing data:**
  - [ ] All images have verdicts recorded
  - [ ] No NaN or NULL values in results spreadsheet
  - [ ] All statistical tests have valid p-values

- [ ] **Sanity checks:**
  - [ ] Mean $\bar{d}_{\text{high}}$ across dataset: ____________ radians (expect ~0.75-0.85)
  - [ ] Mean $\bar{d}_{\text{low}}$ across dataset: ____________ radians (expect ~0.50-0.60)
  - [ ] Overall falsification rate: ____________ % (expect ~30-50% for moderate methods)

- [ ] **Outlier detection:**
  - [ ] Identify images with $\bar{d}_{\text{high}} > 1.2$ radians (suspicious, check for bugs)
  - [ ] Identify images with $\bar{d}_{\text{low}} < 0.3$ radians (potential numerical issues)
  - [ ] Manually inspect 5-10 outlier cases to verify correctness

---

## Section 3: INTERPRETING RESULTS

### 3.1 Aggregate Statistics

**Compute dataset-level metrics:**

- [ ] **Primary Endpoint (Correlation):**
  ```python
  Delta_pred = d_high_mean_all - d_low_mean_all
  Delta_obs = [observed values from experiments]
  rho = pearsonr(Delta_pred, Delta_obs)[0]
  ```
  - Pearson ρ: ____________ (target: >0.7)
  - 95% CI: [____________, ____________] (via Fisher z-transform)
  - P-value for $H_0: \rho \leq 0.7$: ____________
  - **Decision:** Primary endpoint **MET** or **NOT MET**

- [ ] **Secondary Endpoint (Calibration):**
  ```python
  coverage = sum([obs in predicted_CI for obs in observations]) / len(observations)
  ```
  - Empirical coverage: ____________ % (target: 90-100%)
  - Binomial test p-value: ____________
  - **Decision:** Secondary endpoint **MET** or **NOT MET**

- [ ] **Plausibility Gates:**
  - Median LPIPS: ____________ (threshold: <0.3)
  - Mean FID: ____________ (threshold: <50)
  - **Decision:** Plausibility gates **SATISFIED** or **VIOLATED**

### 3.2 Subgroup Analysis

**Stratify results by demographics (if available):**

- [ ] **Falsification rate by age:**
  - Young (<30): ____________ %
  - Middle (30-50): ____________ %
  - Older (>50): ____________ %
  - **Disparity:** ____________ percentage points (flag if >10)

- [ ] **Falsification rate by gender:**
  - Male: ____________ %
  - Female: ____________ %
  - **Disparity:** ____________ percentage points

- [ ] **Falsification rate by skin tone:**
  - Light: ____________ %
  - Dark: ____________ %
  - **Disparity:** ____________ percentage points

- [ ] **Statistical test for disparities:**
  - Chi-square test: χ²(df) = ____________, p = ____________
  - Cramér's V (effect size): ____________
  - **Decision:** Disparities **SIGNIFICANT** (p < 0.05) or **NOT SIGNIFICANT**

### 3.3 Failure Mode Analysis

**Identify known failure scenarios:**

- [ ] **Extreme poses (>30° rotation):**
  - Falsification rate: ____________ %
  - Sample size: ____________ images

- [ ] **Heavy occlusion (masks, hands):**
  - Falsification rate: ____________ %
  - Sample size: ____________ images

- [ ] **Low resolution (<80×80 pixels):**
  - Falsification rate: ____________ %
  - Sample size: ____________ images

- [ ] **Document top failure scenarios:**
  1. ____________ (____________ % falsification rate)
  2. ____________ (____________ % falsification rate)
  3. ____________ (____________ % falsification rate)

### 3.4 Decision Matrix

**Determine overall deployment recommendation:**

| Criterion | Status | Weight |
|-----------|--------|--------|
| Primary endpoint (ρ > 0.7) | ☐ MET / ☐ NOT MET | Critical |
| Secondary endpoint (coverage 90-100%) | ☐ MET / ☐ NOT MET | Important |
| Plausibility gates (LPIPS <0.3, FID <50) | ☐ SATISFIED / ☐ VIOLATED | Critical |
| Demographic fairness (disparity <10pp) | ☐ SATISFIED / ☐ VIOLATED | Important |
| Failure rate (<50%) | ☐ SATISFIED / ☐ VIOLATED | Important |

**Decision Logic:**

- [ ] **All critical criteria MET → Verdict: NOT FALSIFIED**
  - Recommendation: **APPROVED** (potentially with restrictions)

- [ ] **Any critical criterion NOT MET → Verdict: FALSIFIED**
  - Recommendation: **NOT APPROVED**

- [ ] **Critical met but important violated → Verdict: NOT FALSIFIED with CAVEATS**
  - Recommendation: **APPROVED with RESTRICTIONS**

**Final Recommendation:** ____________________________

---

## Section 4: FILLING OUT THE FORENSIC REPORT

### 4.1 Template Completion

**Use provided forensic template (`forensic_template.md`):**

- [ ] **Field 1: Method Identification**
  - [ ] Attribution method name, version, implementation
  - [ ] Face verification model architecture, training data, source
  - [ ] Citations to original publications

- [ ] **Field 2: Parameter Disclosure**
  - [ ] Feature thresholds ($\theta_{\text{high}}, \theta_{\text{low}}$)
  - [ ] Counterfactual settings ($\delta_{\text{target}}, K, T, \alpha, \lambda$)
  - [ ] Statistical test parameters (α, correction method)
  - [ ] Pre-registered thresholds (with timestamp/OSF URL)
  - [ ] Dataset details (name, size, demographics)

- [ ] **Field 3: Δ-Prediction Accuracy**
  - [ ] Pearson ρ with 95% CI
  - [ ] P-value for primary endpoint test
  - [ ] R² (explained variance)
  - [ ] MAE and RMSE (in radians)
  - [ ] Scatter plot (predicted vs. observed)

- [ ] **Field 4: CI Calibration**
  - [ ] Empirical coverage rate
  - [ ] Binomial test p-value
  - [ ] Stratified coverage (by score range)
  - [ ] Calibration plot

- [ ] **Field 5: Known Error Rates**
  - [ ] Overall falsification rate with 95% CI
  - [ ] Failure mode breakdown
  - [ ] Demographic stratification table
  - [ ] Imaging condition stratification table
  - [ ] Known failure scenarios (>50% falsification rate)

- [ ] **Field 6: Limitations**
  - [ ] Dataset limitations (LFW demographics, image quality)
  - [ ] Model constraints (ArcFace-specific, embedding dimension)
  - [ ] Plausibility assumptions (LPIPS/FID thresholds)
  - [ ] Demographic biases (training data skews)
  - [ ] Out-of-scope scenarios (video, 3D, adversarial)

- [ ] **Field 7: Recommendation**
  - [ ] Overall verdict (NOT FALSIFIED or FALSIFIED)
  - [ ] Confidence level (High, Moderate, Low)
  - [ ] Deployment recommendation (APPROVED, APPROVED with RESTRICTIONS, NOT APPROVED)
  - [ ] Specific restrictions (if applicable)
  - [ ] Justification for recommendation

### 4.2 Peer Review

**Before finalizing report:**

- [ ] **Internal review:**
  - [ ] Have colleague verify calculations (spot-check 10 random images)
  - [ ] Cross-check statistical tests (reproduce p-values in independent software, e.g., R)
  - [ ] Validate plots and visualizations (correct axes, labels, legends)

- [ ] **External review (recommended for high-stakes cases):**
  - [ ] Engage independent statistician to review hypothesis tests
  - [ ] Have forensic science expert review for Daubert compliance
  - [ ] Consult legal counsel on evidentiary standards

### 4.3 Report Finalization

- [ ] **Generate report document:**
  - [ ] Export to PDF (ensure all tables, figures render correctly)
  - [ ] Include appendices: statistical test details, code snippets, example visualizations
  - [ ] Add table of contents, page numbers, headers/footers

- [ ] **Cryptographic hash:**
  - [ ] Compute SHA-256 hash of final PDF
  - [ ] Record hash in audit log
  - [ ] Timestamp with trusted authority (e.g., blockchain, RFC 3161)

- [ ] **Archive:**
  - [ ] Save raw data files (attribution maps, counterfactuals, distance measurements)
  - [ ] Save analysis scripts (code used for statistics and plots)
  - [ ] Save report versions (track revisions with version control)

---

## Section 5: DISCLOSURE AND DOCUMENTATION REQUIREMENTS

### 5.1 Transparency for Legal Proceedings

**If report will be used as evidence:**

- [ ] **Disclose to all parties:**
  - [ ] Provide full report to defense counsel (if criminal case)
  - [ ] Include all data files upon request (subject to privacy protections)
  - [ ] Disclose known limitations and failure rates

- [ ] **Expert testimony preparation:**
  - [ ] Review Daubert criteria (testability, peer review, error rates, acceptance)
  - [ ] Prepare to explain methods in layperson terms
  - [ ] Anticipate cross-examination questions (e.g., "Why 70% threshold?")

- [ ] **Court filing:**
  - [ ] Include report as exhibit
  - [ ] Attach pre-registration URL (demonstrates no p-hacking)
  - [ ] Provide code repository link (for reproducibility)

### 5.2 Regulatory Compliance

**EU AI Act / GDPR:**

- [ ] **Technical documentation (AI Act Art. 15):**
  - [ ] Report satisfies "detailed description of validation process"
  - [ ] Accuracy metrics (ρ, MAE) meet "level of accuracy" requirement (Art. 13)
  - [ ] Limitations section addresses "known and foreseeable circumstances" (Art. 13)

- [ ] **Right to explanation (GDPR Art. 22):**
  - [ ] Report provides "meaningful information about logic involved"
  - [ ] Uncertainty quantification (90% CIs) enables "contestation"

### 5.3 Audit Trail

**Maintain complete documentation:**

- [ ] **Pre-registration:**
  - [ ] OSF/AsPredicted URL: ____________________________
  - [ ] SHA-256 hash: ____________________________
  - [ ] Timestamp: ____________________________

- [ ] **Code version:**
  - [ ] GitHub commit hash: ____________________________
  - [ ] Repository URL: ____________________________
  - [ ] Dependencies (requirements.txt): ____________________________

- [ ] **Data provenance:**
  - [ ] Dataset source: ____________________________
  - [ ] Download date: ____________________________
  - [ ] Data hash (verify integrity): ____________________________

- [ ] **Report versions:**
  - [ ] Draft v1.0 timestamp: ____________________________
  - [ ] Final v1.0 timestamp: ____________________________
  - [ ] Revisions (if any): ____________________________

---

## Section 6: TROUBLESHOOTING COMMON ISSUES

### Issue 1: Low Convergence Rate (<180/200 counterfactuals converged)

**Symptoms:**
- Counterfactual generation fails to reach $\delta_{\text{target}} = 0.8$ radians
- Optimization exits early without convergence
- Mean iterations approach T=100 (max)

**Possible Causes:**
1. **Too many features masked:** If $|S_{\text{high}}|$ or $|S_{\text{low}}|$ is >70% of total features, optimization is over-constrained
2. **Learning rate too low:** α=0.01 may be too conservative for some images
3. **Target distance unrealistic:** Image embedding already close to boundary, cannot move further

**Solutions:**
- [ ] Check feature set sizes: If $|S| > 35$ (for m=49), reduce threshold or accept lower convergence
- [ ] Increase learning rate to α=0.02 (but monitor for instability)
- [ ] Relax convergence tolerance to $\epsilon_{\text{tol}} = 0.02$ radians
- [ ] Flag image as "INCONCLUSIVE—insufficient counterfactual coverage" if <160/200 converge

---

### Issue 2: High LPIPS (>0.3) or FID (>50)—Implausible Counterfactuals

**Symptoms:**
- Counterfactuals look unrealistic (artifacts, distortions, non-face appearance)
- LPIPS median >0.3 or FID >50

**Possible Causes:**
1. **Regularization too weak:** λ=0.1 insufficient to prevent large perturbations
2. **Masking constraint too strict:** Preserving too many pixels forces optimization off-manifold
3. **Poor initialization:** Starting from bad latent code

**Solutions:**
- [ ] Increase regularization: Try λ=0.2 or λ=0.5
- [ ] Reduce target distance: Try $\delta_{\text{target}} = 0.6$ radians (less aggressive perturbation)
- [ ] Use GAN-based counterfactuals: Project into StyleGAN latent space instead of pixel-space optimization
- [ ] Flag image as "FALSIFIED (Plausibility)"—validation cannot proceed with off-manifold samples

---

### Issue 3: Correlation ρ Near Threshold (e.g., ρ=0.68-0.72)

**Symptoms:**
- Primary endpoint p-value near 0.05 (borderline significance)
- Verdict depends sensitively on small data fluctuations

**Possible Causes:**
1. **Threshold miscalibration:** ρ_min=0.7 may be too stringent or too lenient for this method
2. **Noisy predictions:** High variance in Δ-scores due to stochastic counterfactual generation
3. **Small sample size:** N<500 provides insufficient power to distinguish ρ=0.68 from ρ=0.72

**Solutions:**
- [ ] **Do NOT adjust threshold post-hoc** (this is p-hacking)
- [ ] Report exact p-value and 95% CI for ρ
- [ ] Conduct sensitivity analysis: Re-run with ±0.05 threshold variations, report verdict stability
- [ ] For borderline cases, recommend "APPROVED with RESTRICTIONS" rather than binary approve/reject
- [ ] Increase sample size to N=2,000 for tighter CIs (if resources permit)

---

### Issue 4: Demographic Disparities (>10 percentage points)

**Symptoms:**
- Falsification rate for older individuals: 45% vs. young: 34% (11pp gap)
- Falsification rate for dark skin: 43% vs. light skin: 35% (8pp gap)

**Possible Causes:**
1. **Training data bias:** Face verification model trained on younger, lighter-skinned faces
2. **Attribution method bias:** Grad-CAM performs worse on underrepresented features
3. **Test set imbalance:** Small subgroups yield noisy estimates

**Solutions:**
- [ ] **Acknowledge limitations transparently:** Report disparities in Field 5, flag as "HIGH DISPARITY"
- [ ] **Add deployment restrictions:** "Use with caution for older individuals and darker skin tones"
- [ ] **Recommend fairness audit:** Require demographic reporting for each deployment case
- [ ] **Future work:** Retrain model with balanced dataset, revalidate attribution method
- [ ] **Do NOT proceed with deployment** if disparities are unacceptable for application domain

---

### Issue 5: Non-Triviality Failures (Empty Feature Sets)

**Symptoms:**
- $S_{\text{high}} = \emptyset$ or $S_{\text{low}} = \emptyset$
- Attribution map is nearly uniform (all values similar)

**Possible Causes:**
1. **Flat attribution:** Method fails to identify discriminative features for this image
2. **Threshold mismatch:** $\theta_{\text{high}}$ too high or $\theta_{\text{low}}$ too low for this attribution distribution

**Solutions:**
- [ ] **Do NOT adjust thresholds** (fixed pre-registration)
- [ ] Mark image as "FALSIFIED (Non-Triviality)"
- [ ] Investigate why attribution is flat (e.g., low-quality image, unusual appearance)
- [ ] If >5% of images fail non-triviality, consider method inadequacy for this dataset
- [ ] Report non-triviality failure rate in Field 5 (Known Error Rates)

---

### Issue 6: Computation Time Exceeds Estimates

**Symptoms:**
- Single image takes >15 seconds (expected: ~4-9 seconds)
- Total runtime for 1,000 images exceeds 5 hours

**Possible Causes:**
1. **CPU-only execution:** No GPU detected or CUDA not configured
2. **Small batch size:** B=1 instead of B=16 (no parallelization)
3. **Slow attribution method:** SHAP with 1,000 samples takes ~5 seconds per image

**Solutions:**
- [ ] Verify GPU usage: `nvidia-smi` should show process running
- [ ] Increase batch size to B=16 (if VRAM permits)
- [ ] Use faster attribution method: Grad-CAM (~50ms) instead of SHAP (~5s)
- [ ] Parallelize across images: Use multiprocessing to distribute across multiple GPUs
- [ ] Accept longer runtime for thorough validation (overnight processing acceptable for forensic contexts)

---

## Section 7: FINAL SUBMISSION CHECKLIST

**Before deploying method or submitting report:**

- [ ] **Validation complete:**
  - [ ] All test images processed (N = ____________)
  - [ ] No missing data or NaN values
  - [ ] Outliers investigated and documented

- [ ] **Statistical tests verified:**
  - [ ] Primary endpoint: ρ = ____________, p = ____________, verdict = ____________
  - [ ] Secondary endpoint: coverage = ____________%, p = ____________, verdict = ____________
  - [ ] Plausibility gates: LPIPS = ____________, FID = ____________, verdict = ____________

- [ ] **Report completed:**
  - [ ] All 7 fields filled out (see template)
  - [ ] Plots and tables generated
  - [ ] Citations included
  - [ ] Appendices attached (code, raw data)

- [ ] **Peer review:**
  - [ ] Colleague verified calculations
  - [ ] Statistician reviewed hypothesis tests
  - [ ] Forensic expert reviewed for Daubert compliance

- [ ] **Documentation archived:**
  - [ ] Pre-registration URL recorded
  - [ ] Code repository committed (GitHub)
  - [ ] Data files backed up (3 copies, 2 media, 1 offsite)
  - [ ] Cryptographic hash computed

- [ ] **Disclosures made:**
  - [ ] Limitations transparently acknowledged
  - [ ] Known failure rates reported
  - [ ] Demographic disparities disclosed
  - [ ] Conflicts of interest declared (if any)

- [ ] **Legal/regulatory review:**
  - [ ] Daubert criteria satisfied (testability, peer review, error rates, acceptance)
  - [ ] AI Act compliance verified (Art. 13-15)
  - [ ] GDPR compliance verified (Art. 22)

**Final Sign-Off:**

I, _________________________ (name), attest that:
1. This validation was conducted per the pre-registered protocol (OSF ID: ____________)
2. No thresholds were adjusted post-hoc based on test set results
3. All results are reported truthfully, including negative findings
4. The forensic report accurately represents the method's performance and limitations

**Signature:** ___________________________

**Date:** ___________________________

**Supervisor/Reviewer:** ___________________________

**Date:** ___________________________

---

**END OF PRACTITIONER CHECKLIST**

**For questions or technical support, contact:** [Provide contact information]

**Checklist Version:** 1.0

**Last Updated:** [DATE]
