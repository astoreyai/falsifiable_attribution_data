# FORENSIC REPORTING TEMPLATE FOR ATTRIBUTION VALIDATION

## Purpose and Scope

This template provides a standardized format for documenting attribution validation results in forensic and legal contexts. The seven-field structure is designed to meet evidentiary requirements from:

- **U.S. Federal Rules of Evidence, Rule 702 (Daubert Standard):** Testability, peer review, known error rates, general acceptance
- **EU AI Act Articles 13-15:** Transparency, technical documentation, accuracy metrics
- **GDPR Article 22:** Meaningful information about automated decision logic
- **NRC 2009 (Forensic Science Standards):** Objective criteria, known error rates, proficiency testing

---

## Template Structure

### Field 1: Method Identification
### Field 2: Parameter Disclosure
### Field 3: Δ-Prediction Accuracy
### Field 4: Confidence Interval Calibration
### Field 5: Known Error Rates and Failure Modes
### Field 6: Limitations and Scope
### Field 7: Recommendation and Confidence Assessment

---

## FIELD 1: METHOD IDENTIFICATION

### Purpose
Specify the exact attribution method and face verification model tested to enable reproducibility and peer review.

### Required Information

**Attribution Method:**
- Method name (e.g., "Grad-CAM," "SHAP," "Integrated Gradients")
- Method version/implementation (e.g., "Captum v0.6.0, PyTorch 2.0.1")
- Any modifications to standard implementation
- Citation to original publication

**Face Verification Model:**
- Model architecture (e.g., "ArcFace ResNet-100")
- Embedding dimension and normalization (e.g., "512-D L2-normalized embeddings")
- Loss function (e.g., "Angular margin loss with m=0.5")
- Training dataset (e.g., "VGGFace2-HQ, 3.31M images, 9,131 identities")
- Model source (e.g., "Official author release, DOI: [LINK]" or "Trained in-house")
- Model version/checkpoint (e.g., "epoch_150.pth")

### Example

**METHOD IDENTIFICATION**

**Attribution Method:**
Gradient-Weighted Class Activation Mapping (Grad-CAM) [Selvaraju et al., 2017, ICCV]
- Implementation: Captum v0.6.0 (PyTorch 2.0.1)
- Target layer: conv5_3 (final convolutional layer before global average pooling)
- Output: 7×7 spatial attribution map (49 features)
- No modifications to standard implementation

**Face Verification Model:**
ArcFace ResNet-100 [Deng et al., 2019, CVPR]
- Architecture: ResNet-100 backbone with 512-D fully connected layer
- Embeddings: L2-normalized (unit hypersphere)
- Loss function: Additive angular margin loss (m=0.5, s=64)
- Training: VGGFace2-HQ dataset (3.31M images, 9,131 identities)
- Source: Official author release, available at https://github.com/deepinsight/insightface
- Checkpoint: glint360k_r100.pth (pretrained on MS1MV2, fine-tuned on VGGFace2)

---

## FIELD 2: PARAMETER DISCLOSURE

### Purpose
Document all configuration parameters and hyperparameters affecting validation results to ensure transparency and reproducibility.

### Required Information

**Feature Classification Thresholds:**
- $\theta_{\text{high}}$ (high-attribution threshold)
- $\theta_{\text{low}}$ (low-attribution threshold)
- Source (calibration set details)

**Counterfactual Generation Settings:**
- Target geodesic distance $\delta_{\text{target}}$ (radians)
- Sample size $K$ (number of counterfactuals per feature set)
- Maximum iterations $T$
- Learning rate $\alpha$
- Regularization weight $\lambda$
- Convergence tolerance $\epsilon_{\text{tol}}$

**Statistical Test Parameters:**
- Significance level α
- Multiple testing correction method (e.g., Bonferroni)
- Corrected significance level (if applicable)
- Test type (e.g., one-sample t-test, two-tailed)

**Pre-Registered Thresholds:**
- $\tau_{\text{high}}$ (high-attribution distance floor)
- $\tau_{\text{low}}$ (low-attribution distance ceiling)
- $\epsilon$ (separation margin)
- $\rho_{\text{min}}$ (correlation floor for primary endpoint)
- Calibration coverage range (for secondary endpoint)

**Dataset Information:**
- Dataset name and version
- Dataset size (number of image pairs)
- Class balance (genuine vs. impostor pairs)
- Any filtering or selection criteria
- Demographic composition (if known)

### Example

**PARAMETER DISCLOSURE**

**Feature Thresholds:**
- $\theta_{\text{high}} = 0.7$ (70th percentile of $|\phi|$ distribution)
- $\theta_{\text{low}} = 0.4$ (40th percentile of $|\phi|$ distribution)
- Source: Calibration set (N=500 LFW images, identities 0001-0500, alphabetically sorted)
- Calibration set is separate from test set (no identity overlap)

**Counterfactual Settings:**
- Target distance: $\delta_{\text{target}} = 0.8$ radians (~45.8°, cosine similarity ≈ 0.697)
- Sample size: $K = 200$ counterfactuals per feature set (400 total per image)
- Max iterations: $T = 100$
- Learning rate: $\alpha = 0.01$
- Regularization: $\lambda = 0.1$ (L2 proximity loss weight)
- Convergence tolerance: $\epsilon_{\text{tol}} = 0.01$ radians
- Early stopping: Enabled (halt when $|d_g - \delta_{\text{target}}| < 0.01$)

**Statistical Tests:**
- Significance level: α = 0.05
- Correction: Bonferroni correction for 2 tests (high-attribution and low-attribution)
- Corrected significance: α_corrected = 0.025
- Test type: One-sample t-tests (one-tailed for high, one-tailed for low)

**Pre-Registered Thresholds:**
- $\tau_{\text{high}} = 0.75$ radians (high-attribution distance floor)
- $\tau_{\text{low}} = 0.55$ radians (low-attribution distance ceiling)
- Separation margin: $\epsilon = 0.15$ radians (~8.6°)
- Correlation floor: $\rho_{\text{min}} = 0.7$ (primary endpoint)
- Calibration coverage: 90-100% (secondary endpoint)
- Pre-registration timestamp: 2024-10-15, OSF ID: [TO BE INSERTED]

**Dataset:**
- Name: Labeled Faces in the Wild (LFW) test set
- Version: Original LFW release (2007)
- Size: 1,000 image pairs (balanced: 500 genuine, 500 impostor)
- Selection: Random sample stratified by identity frequency (no demographic filtering)
- Demographics: ~77% male, ~83% light skin tone (based on available annotations)
- Image quality: High resolution (≥112×112 pixels), frontal/near-frontal poses (<30° rotation)

---

## FIELD 3: Δ-PREDICTION ACCURACY

### Purpose
Report primary validation metric (correlation between predicted and observed score changes) with uncertainty quantification.

### Required Information

**Pearson Correlation:**
- Point estimate $\rho$
- 95% confidence interval (via Fisher z-transformation)
- p-value for hypothesis test $H_0: \rho \leq \rho_{\text{min}}$ vs. $H_1: \rho > \rho_{\text{min}}$
- Decision: "Primary endpoint MET" or "Primary endpoint NOT MET"

**Effect Size:**
- R² (explained variance)
- Interpretation (e.g., "weak," "moderate," "strong" per Cohen's guidelines)

**Prediction Error:**
- Mean Absolute Error (MAE) in radians
- Root Mean Squared Error (RMSE) in radians
- Interpretation (e.g., "typical prediction error ~6.3°")

**Visualizations:**
- Scatter plot: Predicted vs. observed Δ-scores with regression line
- Residual plot: Prediction errors vs. predicted values
- Histogram of residuals (check normality assumption)

### Example

**Δ-PREDICTION ACCURACY**

**Correlation:**
- Pearson ρ = 0.73
- 95% CI: [0.68, 0.78] (via Fisher z-transformation with N=1,000)
- Hypothesis test: $H_0: \rho \leq 0.7$ vs. $H_1: \rho > 0.7$
  - Test statistic: z = 2.51
  - p-value: p = 0.012 (one-tailed)
  - Decision: **Reject $H_0$ at α=0.05; Primary endpoint MET**

**Effect Size:**
- R² = 0.53 (53% of variance in observed Δ-scores explained by predictions)
- Interpretation: Moderate predictive accuracy per Cohen (1988): R² ∈ [0.26, 0.64] is "medium"

**Prediction Error:**
- Mean Absolute Error (MAE): 0.11 radians (6.3°)
- Root Mean Squared Error (RMSE): 0.15 radians (8.6°)
- Interpretation: Typical prediction error ~6.3°; predictions directionally correct but imperfect magnitude estimation

**Visual Calibration:**
- Scatter plot shows strong linear trend with slope ≈ 0.73
- Residuals appear approximately normally distributed (Shapiro-Wilk p=0.18)
- No systematic bias (residuals centered near zero across prediction range)

**Interpretation:**
Predicted geodesic distance changes demonstrate moderate-to-strong correlation with observed changes. Attributions show directional correctness (high-attribution features cause larger shifts than low-attribution features) but magnitude estimation is imperfect. For forensic purposes, this indicates attributions can distinguish between important and unimportant features but should be interpreted with caution for precise quantitative claims.

---

## FIELD 4: CONFIDENCE INTERVAL CALIBRATION

### Purpose
Assess whether uncertainty estimates (confidence intervals) are well-calibrated, providing reliable measures of prediction reliability.

### Required Information

**Empirical Coverage Rate:**
- Observed percentage of cases where $\bar{d}_{\text{obs}} \in \text{CI}_{\text{pred}}$ (90% CI)
- Expected coverage: 90% (by construction)
- 95% confidence interval for coverage rate (binomial proportion CI)

**Calibration Test:**
- Binomial test for $H_0: p_{\text{coverage}} = 0.90$ vs. $H_1: p_{\text{coverage}} \neq 0.90$
- p-value
- Decision: "Well-calibrated" (p > 0.05) or "Miscalibrated" (p < 0.05)

**Stratified Coverage (Optional but Recommended):**
- Coverage by verification score range (high similarity, medium, low)
- Coverage by demographic group (if applicable)
- Identify subgroups with poor calibration

**Visualizations:**
- Calibration plot: Predicted probability vs. observed frequency
- Coverage by score bin (bar chart)

### Example

**CONFIDENCE INTERVAL CALIBRATION**

**Coverage Rate:**
- Empirical coverage: 91.3% (913 of 1,000 test cases within predicted 90% CI)
- Expected coverage: 90%
- 95% CI for coverage: [89.5%, 93.1%] (Wilson score interval)

**Calibration Test:**
- Binomial test: $H_0: p = 0.90$ vs. $H_1: p \neq 0.90$
- Observed: 913 successes in 1,000 trials
- p-value: p = 0.42 (two-tailed)
- Decision: **Fail to reject $H_0$; coverage consistent with nominal 90% (well-calibrated)**

**Stratified Coverage:**

| Verification Score Range | N | Coverage | 95% CI |
|-------------------------|---|----------|---------|
| High similarity (cos > 0.8) | 298 | 89.7% | [85.9%, 92.7%] |
| Medium similarity (0.5 < cos < 0.8) | 412 | 92.1% | [89.2%, 94.4%] |
| Low similarity (cos < 0.5) | 290 | 90.8% | [87.1%, 93.7%] |

**Interpretation:**
Confidence intervals are well-calibrated across all verification score ranges. Observed coverage (91.3%) closely matches nominal 90%, indicating that uncertainty quantification is reliable. Practitioners can trust that reported 90% CIs will contain true values approximately 90% of the time. Slight over-coverage (91.3% vs. 90%) suggests conservative (wider) intervals, which is acceptable in forensic contexts where underconfidence is preferable to overconfidence.

---

## FIELD 5: KNOWN ERROR RATES AND FAILURE MODES

### Purpose
Meet Daubert requirement for known error rates; identify specific conditions where method fails to ensure appropriate deployment restrictions.

### Required Information

**Overall Falsification Rate:**
- Percentage of test cases returning "FALSIFIED" verdict
- 95% confidence interval (binomial proportion CI)
- Breakdown by failure mode (Non-Triviality, Statistical Evidence, Separation Margin)

**Demographic Stratification:**
- Falsification rate by age group (if annotations available)
- Falsification rate by gender
- Falsification rate by skin tone / ethnicity
- Chi-square test for independence (demographic category vs. verdict)
- Cramér's V (effect size for association)
- Fairness flag: "HIGH DISPARITY" if difference >10 percentage points

**Imaging Condition Stratification:**
- Falsification rate by pose (frontal vs. profile)
- Falsification rate by occlusion (none vs. partial vs. heavy)
- Falsification rate by resolution (high vs. medium vs. low)

**Known Failure Scenarios:**
- Specific conditions with >50% falsification rate
- Qualitative description of failure modes (e.g., "unreliable for extreme poses")

### Example

**KNOWN ERROR RATES AND FAILURE MODES**

**Overall Falsification Rate:**
- 38% (380 of 1,000 test cases FALSIFIED)
- 95% CI: [35.1%, 40.9%]

**Failure Mode Breakdown:**
- Non-Triviality Failure: 2.1% (21 cases; $S_{\text{high}} = \emptyset$ or $S_{\text{low}} = \emptyset$)
- Insufficient Statistical Evidence: 35.9% (359 cases; $p_{\text{high}} > 0.025$ OR $p_{\text{low}} > 0.025$)
- Separation Margin Violation: 0% (by design; pre-registered thresholds satisfy $\tau_{\text{high}} > \tau_{\text{low}} + \epsilon$)

**Demographic Stratification (LFW Annotations, N=1,000):**

| Demographic Group | N | Falsification Rate | 95% CI |
|------------------|---|-------------------|---------|
| **Age** |
| Young (<30y) | 287 | 34% | [28.7%, 39.8%] |
| Middle (30-50y) | 485 | 37% | [32.8%, 41.5%] |
| Older (>50y) | 228 | 45% | [38.7%, 51.5%] |
| **Gender** |
| Male | 768 | 36% | [32.7%, 39.5%] |
| Female | 232 | 42% | [35.9%, 48.4%] |
| **Skin Tone** |
| Light | 831 | 35% | [31.9%, 38.3%] |
| Dark | 169 | 43% | [35.7%, 50.6%] |

**Statistical Tests:**
- Age vs. Verdict: χ²(2) = 8.73, p = 0.013, Cramér's V = 0.093 (small effect)
- Gender vs. Verdict: χ²(1) = 3.21, p = 0.073, Cramér's V = 0.057 (small effect)
- Skin Tone vs. Verdict: χ²(1) = 5.12, p = 0.024, Cramér's V = 0.071 (small effect)

**Fairness Assessment:**
- Age disparity: 45% (older) - 34% (young) = 11 percentage points → **HIGH DISPARITY**
- Gender disparity: 42% (female) - 36% (male) = 6 percentage points → Moderate
- Skin tone disparity: 43% (dark) - 35% (light) = 8 percentage points → Moderate

**Imaging Condition Stratification (N=1,000):**

| Condition | N | Falsification Rate | 95% CI |
|-----------|---|-------------------|---------|
| **Pose** |
| Frontal (<15° rotation) | 612 | 32% | [28.5%, 35.8%] |
| Near-frontal (15-30°) | 298 | 41% | [35.6%, 46.7%] |
| Profile (>30°) | 90 | 52% | [41.7%, 62.2%] |
| **Occlusion** |
| None | 734 | 34% | [30.7%, 37.5%] |
| Partial (sunglasses, beard) | 198 | 45% | [38.2%, 51.9%] |
| Heavy (mask, hands) | 68 | 61% | [49.0%, 72.2%] |
| **Resolution** |
| High (>112×112 px) | 856 | 36% | [32.9%, 39.3%] |
| Medium (80-112 px) | 118 | 44% | [35.3%, 53.1%] |
| Low (<80×80 px) | 26 | 48% | [29.4%, 66.5%] |

**Known Failure Scenarios:**
1. **Extreme Poses (>45° rotation):** 52% falsification rate; attributions unreliable for profile views
2. **Heavy Occlusion (surgical masks, hands covering face):** 61% falsification rate; insufficient visible features for attribution
3. **Low Resolution (<80×80 pixels):** 48% falsification rate; fine-grained feature importance difficult to estimate
4. **Older Individuals (>50 years):** 45% falsification rate; potential bias from training data skewing toward younger faces
5. **Non-Trivial Attributions (uniform importance):** 2.1% failure rate; some images yield flat attribution maps without clear high/low distinction

**Interpretation:**
Method achieves "NOT FALSIFIED" status for 62% of cases but exhibits systematic biases:
- **Age Bias:** Higher failure rates for older individuals (45% vs. 34% for young), likely due to training data demographics
- **Pose Sensitivity:** Unreliable for profile views (>30° rotation)
- **Occlusion Failures:** Cannot validate attributions when faces are heavily occluded
- **Demographic Disparities:** Moderately higher failure rates for females and darker skin tones

**Recommendation:** Use with caution in demographically diverse forensic contexts; restrict to high-quality frontal images; mandatory demographic audit for each deployment case.

---

## FIELD 6: LIMITATIONS AND SCOPE

### Purpose
Transparently acknowledge constraints on generalizability and applicability to prevent overclaiming and misuse.

### Required Information

**Dataset Limitations:**
- Dataset composition (e.g., celebrity images, frontal poses, high resolution)
- Demographic biases (e.g., 77% male, 83% light skin)
- Scenarios NOT covered (e.g., surveillance footage, thermal imagery)

**Model Architecture Constraints:**
- Specific to tested model (e.g., ArcFace ResNet-100)
- May not generalize to other architectures (e.g., CosFace, transformers)
- Embedding dimension and normalization assumptions

**Plausibility Assumptions:**
- Counterfactuals remain on natural face manifold (LPIPS < 0.3, FID < 50)
- Cannot cover all realistic variations (e.g., rare facial features, medical conditions)

**Demographic Biases:**
- Training data imbalances (if known)
- Test set representativeness (if known)

**Out-of-Scope Scenarios:**
- Video-based verification
- 3D face models
- Face identification (1:N search)
- Adversarial robustness
- Real-time deployment constraints

### Example

**LIMITATIONS AND SCOPE**

**Dataset Limitations:**
- Validation conducted on Labeled Faces in the Wild (LFW), N=1,000 image pairs
- LFW primarily contains celebrity images with frontal/near-frontal poses, adequate lighting, high resolution (≥112×112 pixels)
- Demographics: ~77% male, ~83% light skin tone (based on available annotations)
- Results may NOT generalize to:
  - Low-quality surveillance footage (resolution <80×80 pixels, poor lighting)
  - Infrared or thermal imagery
  - Non-Western demographics (LFW skews toward North American/European faces)
  - Extreme poses (>45° rotation), heavy occlusion (full face masks)
  - Video sequences (temporal dependencies not modeled)

**Model Architecture Constraints:**
- Tested exclusively on ArcFace ResNet-100 (512-D L2-normalized embeddings, angular margin loss)
- Results may differ for:
  - Other architectures: CosFace, AdaCos, SphereFace, transformer-based models
  - Different embedding dimensions (e.g., 128-D, 256-D)
  - Models trained on different datasets (e.g., WebFace, CASIA-WebFace, proprietary datasets)
  - Models with different normalization schemes (non-L2-normalized embeddings)

**Plausibility Assumptions:**
- Counterfactuals generated via gradient-based optimization maintain:
  - Perceptual similarity: LPIPS < 0.3 (minor variations like lighting, subtle expressions)
  - Distributional similarity: FID < 50 (close to natural face distribution)
- Counterfactuals CANNOT cover all realistic face variations:
  - Extreme expressions (e.g., wide-open mouth, squinting eyes)
  - Rare facial features (e.g., cleft lip, facial scars, birthmarks)
  - Specific medical conditions (e.g., facial paralysis, severe acne)
  - Cultural/ethnic facial features underrepresented in training data

**Demographic Biases:**
- ArcFace training data (MS1MV2, VGGFace2) known to skew toward:
  - Celebrity faces (high-quality studio photos)
  - Younger individuals (median age ~35-40 years)
  - Light skin tones (Western datasets)
- Attribution faithfulness may differ on underrepresented groups (older, darker skin, non-Western)

**Out-of-Scope Applications:**
- **Video-based verification:** This validation applies to 2D still images; temporal dynamics not modeled
- **3D face models:** Validation assumes 2D RGB images; depth/3D geometry not considered
- **Face identification (1:N search):** Tested on pairwise verification (1:1 matching); scalability to large galleries unknown
- **Adversarial robustness:** Validation assumes benign perturbations; adversarial attacks not tested
- **Real-time deployment:** Computational cost (~4-9 seconds per image) prohibits real-time use; designed for offline forensic analysis

**Honest Assessment:**
These limitations reflect genuine constraints on what can be validated within a single study. Findings should be interpreted as:
- **Valid for:** High-quality frontal face images, ArcFace-based verification systems, offline forensic analysis
- **Uncertain for:** Surveillance footage, non-Western demographics, other model architectures
- **Invalid for:** Video, 3D faces, adversarial scenarios, real-time deployment

Practitioners must exercise judgment in applying these findings to novel contexts.

---

## FIELD 7: RECOMMENDATION AND CONFIDENCE ASSESSMENT

### Purpose
Provide clear, actionable guidance for forensic and legal practitioners with explicit deployment restrictions.

### Required Information

**Overall Verdict:**
- "NOT FALSIFIED" or "FALSIFIED"
- Brief justification (e.g., "Primary and secondary endpoints met")

**Confidence Level:**
- "High," "Moderate," or "Low"
- Justification based on correlation strength, calibration quality, error rates

**Deployment Recommendation:**
- "APPROVED for forensic use" (unrestricted)
- "APPROVED for forensic use with RESTRICTIONS" (specify restrictions)
- "NOT APPROVED for forensic use" (method fails validation)

**Restrictions (if applicable):**
- Image quality requirements (resolution, pose, lighting)
- Demographic audit requirements
- Human expert review triggers
- Uncertainty disclosure requirements
- Evidentiary limitations (e.g., "investigative aid only, not sole evidence")

**Justification:**
- Evidence supporting recommendation
- Rationale for restrictions
- Alternative methods (if current method not approved)

### Example (Scenario 1: Moderate Performance with Restrictions)

**RECOMMENDATION AND CONFIDENCE ASSESSMENT**

**Verdict:** NOT FALSIFIED

**Justification:**
- Primary endpoint MET: ρ = 0.73 (95% CI: [0.68, 0.78]), p = 0.012 (reject $H_0: \rho \leq 0.7$)
- Secondary endpoint MET: 91.3% calibration coverage (consistent with nominal 90%, binomial p = 0.42)
- Plausibility gates SATISFIED: Median LPIPS = 0.22, FID = 41

**Confidence Level:** MODERATE

**Rationale:**
- Correlation (ρ=0.73) demonstrates moderate-to-strong predictive accuracy
- Well-calibrated confidence intervals provide reliable uncertainty estimates
- However:
  - 38% falsification rate indicates method fails for substantial minority of cases
  - Demographic disparities observed (45% failure for older individuals vs. 34% for young)
  - Known failure scenarios (extreme poses, heavy occlusion, low resolution)

**Deployment Recommendation:** APPROVED for forensic use with RESTRICTIONS

**Mandatory Restrictions:**

1. **Image Quality Requirements:**
   - Resolution: Minimum 100×100 pixels (preferably ≥112×112)
   - Pose: Frontal or near-frontal only (<30° rotation); reject profile views
   - Lighting: Adequate illumination (no extreme shadows or backlighting)
   - Occlusion: None or minimal (reject if sunglasses, masks, or hands covering >20% of face)

2. **Demographic Audit:**
   - For each case, record subject's demographic category (age, gender, skin tone)
   - Report falsification rate for that demographic group (from Field 5 stratification table)
   - Flag cases in high-disparity groups (e.g., older individuals) for additional scrutiny

3. **Human Expert Review Triggers:**
   - If attribution highlights unusual regions (e.g., >30% importance on background, clothing, or hair), require forensic examiner review
   - If falsification verdict is borderline (p-value between 0.025 and 0.10), flag for expert interpretation
   - If counterfactual generation fails for >10% of samples, mark case as "INCONCLUSIVE—insufficient evidence"

4. **Uncertainty Disclosure:**
   - Always report 90% confidence intervals alongside point estimates for $\bar{d}_{\text{high}}$ and $\bar{d}_{\text{low}}$
   - Include prediction error (MAE ≈ 6.3°) in forensic reports to communicate imprecision
   - State explicitly: "Attributions are directionally reliable but magnitude estimates are approximate"

5. **Evidentiary Limitations:**
   - Attributions should NOT be used as sole evidence for identification
   - Use as investigative aid to prioritize follow-up (e.g., "focus on eye region discrepancies")
   - Require corroboration from traditional forensic methods (e.g., manual feature comparison, anthropometric measurements)
   - In court testimony, disclose known error rates and demographic disparities

**Contraindications (DO NOT USE):**
- Surveillance footage with resolution <80×80 pixels
- Profile views or extreme poses (>30° rotation)
- Heavily occluded faces (surgical masks, hands covering face)
- Video-based verification (temporal dynamics not validated)
- Real-time deployment (computational cost ~4-9 seconds per image)

**Justification for Restrictions:**
Method demonstrates moderate predictive accuracy (ρ=0.73, R²=0.53) and well-calibrated uncertainty (91.3% coverage), indicating attributions provide useful forensic insights. However:
- 38% falsification rate and known failure scenarios (extreme poses, occlusion, low resolution) necessitate image quality restrictions
- Demographic disparities (11 percentage point gap for age) require mandatory demographic audit
- Moderate (not high) correlation suggests attributions are directionally correct but not precise enough for sole reliance

These restrictions balance:
- **Utility:** Enabling forensic use where validation is strongest (high-quality frontal images)
- **Safety:** Preventing misuse in scenarios where validation fails (poor quality, extreme poses)
- **Transparency:** Disclosing known limitations to legal professionals and oversight bodies

**Alternative Methods (if restrictions cannot be met):**
- If image quality is poor (low resolution, extreme pose), rely on traditional anthropometric analysis without attribution-based explanations
- If demographic disparities are unacceptable, defer deployment until methods with better fairness properties are validated
- If real-time deployment is required, explore approximations (e.g., reduce K from 200 to 50, accept lower precision)

---

### Example (Scenario 2: Weak Performance—Not Approved)

**RECOMMENDATION AND CONFIDENCE ASSESSMENT**

**Verdict:** FALSIFIED

**Justification:**
- Primary endpoint NOT MET: ρ = 0.54 (95% CI: [0.48, 0.60]), p = 0.18 (fail to reject $H_0: \rho \leq 0.7$)
- Secondary endpoint NOT MET: 83.2% calibration coverage (below nominal 90%, binomial p = 0.002)
- Correlation below pre-registered threshold; confidence intervals under-calibrated (overconfident predictions)

**Confidence Level:** LOW

**Rationale:**
- Weak correlation (ρ=0.54) indicates attributions explain only 29% of variance in observed score changes
- Under-calibrated confidence intervals (83% vs. 90% nominal) suggest unreliable uncertainty estimates
- High falsification rate (67%) indicates method fails for majority of cases

**Deployment Recommendation:** NOT APPROVED for forensic use

**Justification:**
Attributions fail to meet pre-registered validation criteria for predictive accuracy and calibration. With ρ=0.54, predicted score changes explain only 29% of variance, meaning 71% of variation is unexplained—indicating attributions do not reliably identify causal features. Under-calibrated confidence intervals (83% coverage vs. 90% nominal) compound unreliability by overestimating prediction certainty. These failures create unacceptable risk of misleading forensic analysts and legal decision-makers.

**Do NOT use this method for:**
- Evidentiary purposes in criminal or civil proceedings
- Investigative prioritization (unreliable feature importance)
- Explanations to defendants or affected individuals (may be misleading)
- System audits or accountability assessments

**Alternative Recommendations:**
1. **Use validated alternative methods:** Test other attribution methods (e.g., Grad-CAM, Integrated Gradients) that may achieve higher correlation and better calibration
2. **Develop new methods:** Invest in attribution techniques explicitly designed for face verification (most existing methods were designed for classification)
3. **Defer deployment:** Wait for improved validation protocols or model architectures with better explainability properties
4. **Traditional forensics:** Rely on manual feature comparison and anthropometric measurements without automated attribution-based explanations

**Research Follow-Up:**
- Investigate why this method fails (e.g., incorrect feature localization, sensitivity to hyperparameters)
- Conduct ablation studies to identify failure modes
- Explore whether method performs better on specific subsets (e.g., high-quality images only)
- Consider method refinement before revalidation

---

## Usage Instructions for Practitioners

### When to Complete This Template

Complete this forensic template for each:
1. **New attribution method deployment:** Before using a method in forensic investigations
2. **Model update:** When face verification model is retrained or fine-tuned
3. **Dataset shift:** When deploying on significantly different data distribution (e.g., surveillance footage after validating on LFW)
4. **Annual review:** Periodic revalidation to ensure continued performance

### How to Fill Out the Template

**Step 1:** Run falsification protocol (Section 3 of Article B) on representative test set (N≥500 recommended)

**Step 2:** Compute all required metrics (correlation, coverage, error rates, demographic stratification)

**Step 3:** Complete Fields 1-7 systematically, providing all required information

**Step 4:** Have report reviewed by independent forensic examiner or statistician

**Step 5:** Archive report with cryptographic hash for audit trail

**Step 6:** Update report if any parameters change (model, dataset, thresholds)

### Legal and Ethical Considerations

**Admissibility (Daubert Compliance):**
- Field 1 (Method ID) addresses "peer review and publication" prong
- Field 3 (Δ-Prediction Accuracy) addresses "testability" prong
- Field 5 (Error Rates) addresses "known error rates" prong
- Field 7 (Recommendation) addresses "general acceptance" (if method widely used)

**GDPR/AI Act Compliance:**
- Field 2 (Parameters) provides "meaningful information about logic" (GDPR Art. 22)
- Field 3 (Accuracy) meets AI Act Art. 13 "level of accuracy" requirement
- Field 6 (Limitations) ensures "transparent and comprehensible" explanations (AI Act Art. 13)

**Forensic Standards (NRC 2009):**
- Field 5 (Error Rates) provides "known and potential error rates"
- Field 7 (Restrictions) establishes "objective standards" for deployment

**Transparency:**
- Always disclose completed template to defendants, legal counsel, and oversight bodies
- Make template available upon FOIA/public records requests
- Include template as exhibit in court proceedings where attribution evidence is presented

---

## Template Versioning and Updates

**Current Version:** 1.0

**Last Updated:** [DATE]

**Change Log:**
- v1.0 (Initial Release): Seven-field structure based on Article B protocol

**Recommended Update Frequency:**
- Review template structure annually
- Update field requirements if regulatory frameworks change (e.g., EU AI Act amendments)
- Revise thresholds if new scientific evidence emerges (e.g., updated correlation benchmarks)

---

**END OF FORENSIC REPORTING TEMPLATE**

**For questions or technical support, contact:** [Provide contact information]

**Template Citation:**
[Author(s)], "Forensic Reporting Template for Attribution Validation in Face Verification Systems," IEEE Transactions on Information Forensics and Security, vol. [TBD], no. [TBD], pp. [TBD], [YEAR].
