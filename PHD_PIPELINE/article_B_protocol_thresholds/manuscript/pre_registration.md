# PRE-REGISTRATION: FALSIFIABLE ATTRIBUTION VALIDATION PROTOCOL

## Metadata

**Protocol Name:** Counterfactual Validation of Attribution Methods in Face Verification Systems

**Pre-Registration Date:** [TO BE TIMESTAMPED UPON FINALIZATION]

**Principal Investigator:** [Name]

**Institution:** [Institution]

**OSF/AsPredicted ID:** [To be assigned upon public registration]

**Status:** FROZEN — No post-hoc modifications permitted after experimental execution begins

---

## 1. Research Hypothesis

**Primary Hypothesis (H1):**
Attribution methods that correctly identify causal features driving face verification decisions will demonstrate strong positive correlation (ρ > 0.7) between predicted geodesic distance changes (based on feature importance) and observed geodesic distance changes under counterfactual perturbations.

**Secondary Hypotheses:**

**H2:** Attributions deemed "NOT FALSIFIED" will produce well-calibrated uncertainty estimates, with 90-100% empirical coverage of 90% confidence intervals.

**H3:** Attribution methods will exhibit heterogeneous performance across demographic groups (age, gender, skin tone), with systematic disparities indicating fairness concerns.

**H4:** Counterfactual perturbations maintaining perceptual plausibility (LPIPS < 0.3) and distributional similarity (FID < 50) will provide more reliable validation than unrestricted perturbations.

---

## 2. Pre-Registered Endpoints

### 2.1 Primary Endpoint

**Endpoint:** Δ-Score Correlation (Pearson's ρ)

**Definition:** Pearson correlation coefficient between predicted geodesic distance changes and observed geodesic distance changes across all test cases.

**Measurement Procedure:**
1. For each test image $x$ and attribution method $A$, extract high-attribution features $S_{\text{high}}$ and low-attribution features $S_{\text{low}}$ using thresholds $\theta_{\text{high}} = 0.7$ and $\theta_{\text{low}} = 0.4$
2. Generate $K=200$ counterfactuals for each feature set using target geodesic distance $\delta_{\text{target}} = 0.8$ radians
3. Measure mean geodesic distances: $\bar{d}_{\text{high}}$ and $\bar{d}_{\text{low}}$
4. Compute predicted differential: $\Delta_{\text{pred}} = \bar{d}_{\text{high}} - \bar{d}_{\text{low}}$
5. Compute observed differential from empirical measurements: $\Delta_{\text{obs}}$
6. Calculate Pearson correlation: $\rho = \text{corr}(\Delta_{\text{pred}}, \Delta_{\text{obs}})$

**Success Criterion:** $\rho > 0.7$ with statistical significance $p < 0.05$ (one-tailed test for $H_0: \rho \leq 0.7$ vs. $H_1: \rho > 0.7$)

**Justification:**
- Cohen (1988): For predictive models, $R^2 > 0.5$ (ρ > 0.71) indicates moderate explanatory power
- Psychometric standards (Koo & Li, 2016): Test-retest reliability ρ > 0.7 is "acceptable," ρ > 0.8 is "good"
- Forensic context: High-stakes decisions require strong predictive evidence; 49% explained variance (ρ=0.7) is minimum threshold
- Pilot data: Grad-CAM ρ ≈ 0.68-0.74 on N=100 calibration images, suggesting threshold is achievable yet rigorous

**Statistical Power:** With N=1,000 test cases, power >0.99 to detect ρ=0.7 vs. null ρ=0.5 at α=0.05 (Cohen, 1988)

---

### 2.2 Secondary Endpoint: Confidence Interval Calibration

**Endpoint:** Empirical Coverage Rate of 90% Confidence Intervals

**Definition:** Percentage of test cases where observed mean geodesic distance $\bar{d}_{\text{obs}}$ falls within predicted 90% confidence interval $[\bar{d} - 1.645 \cdot SE, \bar{d} + 1.645 \cdot SE]$ where $SE = \sigma / \sqrt{K}$

**Success Criterion:** Coverage rate ∈ [90%, 100%] with binomial test $p > 0.05$ for $H_0: p_{\text{coverage}} = 0.90$

**Justification:**
- Conformal prediction theory (Vovk et al., 2005): Properly constructed prediction intervals should achieve nominal coverage
- Clinical calibration standards (Steyerberg, 2009): Observed frequencies should match predicted probabilities
- Under-coverage (< 90%) indicates overconfidence, dangerous in forensic contexts
- Over-coverage (up to 100%) is acceptable (conservative intervals)

**Tolerance:** Accept coverage rates 88-100% to account for sampling variability with N=1,000

---

## 3. Pre-Registered Thresholds

### 3.1 Feature Classification Thresholds

**High-Attribution Threshold:** $\theta_{\text{high}} = 0.7$

**Low-Attribution Threshold:** $\theta_{\text{low}} = 0.4$

**Source:** Determined from calibration set analysis (N=500 LFW images, separate from test set)

**Rationale:**
- 70th percentile: ~30% of features classified as high-attribution
- 40th percentile: ~40% of features classified as low-attribution
- Middle 30% excluded as neutral features
- Ensures both $S_{\text{high}}$ and $S_{\text{low}}$ are non-empty for >97% of images (empirical validation on calibration set)

**Frozen:** These values will NOT be adjusted based on test set performance

---

### 3.2 Geodesic Distance Thresholds

**High-Attribution Distance Floor:** $\tau_{\text{high}} = 0.75$ radians

**Low-Attribution Distance Ceiling:** $\tau_{\text{low}} = 0.55$ radians

**Separation Margin:** $\epsilon = 0.15$ radians

**Target Distance:** $\delta_{\text{target}} = 0.8$ radians

**Rationale:**

**Target Distance ($\delta_{\text{target}} = 0.8$ rad):**
- ArcFace verification decision boundary analysis:
  - $d_g < 0.6$ rad → "same identity" (cosine similarity > 0.825)
  - $d_g > 1.0$ rad → "different identity" (cosine similarity < 0.540)
  - $d_g \approx 0.8$ rad → boundary region (cosine similarity ≈ 0.697)
- Sensitive test: boundary region maximizes discriminative power

**High Threshold ($\tau_{\text{high}} = 0.75$ rad):**
- Expectation: Masking truly important features prevents counterfactuals from reaching $\delta_{\text{target}} = 0.8$
- Predicted range: $\bar{d}_{\text{high}} \in [0.75, 0.85]$ rad based on pilot data
- Setting floor at 0.75 rad allows modest shortfall while requiring substantial separation

**Low Threshold ($\tau_{\text{low}} = 0.55$ rad):**
- Expectation: Masking unimportant features allows counterfactuals to easily reach/exceed target
- Predicted range: $\bar{d}_{\text{low}} \in [0.50, 0.60]$ rad based on pilot data
- Setting ceiling at 0.55 rad requires clear demonstration of low impact

**Separation Margin ($\epsilon = 0.15$ rad ≈ 8.6°):**
- Ensures meaningful distinction: $\tau_{\text{high}} - \tau_{\text{low}} = 0.20$ rad > $\epsilon = 0.15$ rad
- Corresponds to cosine similarity difference $\Delta \cos \approx 0.05$
- Minimum detectable effect size for practical significance

**Verification:** $\tau_{\text{high}} > \tau_{\text{low}} + \epsilon$ → $0.75 > 0.55 + 0.15 = 0.70$ ✓

**Frozen:** These thresholds will NOT be adjusted based on test set results

---

### 3.3 Plausibility Gates

#### 3.3.1 Perceptual Similarity Gate

**Metric:** Learned Perceptual Image Patch Similarity (LPIPS) using AlexNet features

**Threshold:** LPIPS$(x, x') < 0.3$

**Rationale:**
- Zhang et al. (2018, CVPR): LPIPS correlates with human perceptual judgments
- Empirical ranges:
  - LPIPS < 0.1: Nearly imperceptible
  - LPIPS 0.1-0.3: Noticeable but minor variations (lighting, subtle expressions)
  - LPIPS 0.3-0.5: Moderate differences (expressions, accessories)
  - LPIPS > 0.5: Major structural changes
- Pilot data: Median LPIPS ≈ 0.22 (IQR: 0.18-0.28) for $\delta_{\text{target}} = 0.8$ rad counterfactuals
- Threshold balances realism (reject major changes) with flexibility (allow noticeable variations)

**Decision Rule:** Reject counterfactuals with LPIPS ≥ 0.3 as "off-manifold"

---

#### 3.3.2 Distributional Similarity Gate

**Metric:** Fréchet Inception Distance (FID) using Inception-v3 features

**Threshold:** FID < 50 (computed between counterfactual distribution and real face distribution)

**Rationale:**
- Heusel et al. (2017, NeurIPS): FID measures distributional similarity for generative models
- GAN benchmarks:
  - FID < 10: Near-perfect (StyleGAN2 on FFHQ)
  - FID 10-50: Good quality
  - FID 50-100: Moderate quality
  - FID > 100: Poor quality
- Counterfactuals are perturbed real images (not generated), so looser threshold than GANs
- Pilot data: FID ≈ 38-44 for counterfactual sets (N=200) vs. LFW test distribution
- Conservative threshold ensures distributional plausibility without requiring perfect realism

**Decision Rule:** If FID ≥ 50, counterfactual set fails plausibility gate

---

## 4. Sample Size and Statistical Power

### 4.1 Primary Endpoint (Correlation)

**Planned Sample Size:** N = 1,000 test image pairs

**Power Analysis:**
- Effect size: ρ = 0.7 (target correlation)
- Null hypothesis: ρ₀ = 0.5 (weak correlation)
- Significance level: α = 0.05 (one-tailed)
- Power: >0.99 to detect ρ = 0.7 vs. ρ₀ = 0.5 (G*Power calculation)

**Justification:** N=1,000 provides very high power to detect meaningful correlations, reducing Type II error risk.

---

### 4.2 Secondary Endpoint (Calibration)

**Expected Coverage:** 90% (by construction of 90% CI)

**Observed Coverage Variability:** With N=1,000, binomial standard error = $\sqrt{0.9 \times 0.1 / 1000} \approx 0.0095$ (0.95%)

**95% CI for Coverage:** [88.1%, 91.9%] under null $p = 0.90$

**Decision Rule:** Reject null if observed coverage < 88% or if binomial test $p < 0.05$

---

### 4.3 Counterfactual Sample Size

**Per-Image Counterfactuals:** K = 200 per feature set

**Total Counterfactuals per Image:** 400 (200 for $S_{\text{high}}$, 200 for $S_{\text{low}}$)

**Estimation Error:** By Hoeffding's inequality, with K=200, estimation error $\epsilon < 0.1$ rad with 95% confidence

**Justification:** K=200 balances statistical robustness (tight confidence intervals) with computational feasibility (~4 seconds per image on NVIDIA RTX 3090)

---

## 5. Planned Statistical Tests

### 5.1 Primary Endpoint Test

**Test:** One-sample t-test for correlation coefficient

**Null Hypothesis:** $H_0: \rho \leq 0.7$

**Alternative Hypothesis:** $H_1: \rho > 0.7$ (one-tailed)

**Significance Level:** α = 0.05

**Procedure:**
1. Compute Pearson correlation ρ from test data
2. Apply Fisher z-transformation: $z = \frac{1}{2} \ln\left(\frac{1+\rho}{1-\rho}\right)$
3. Compute standard error: $SE_z = \frac{1}{\sqrt{N-3}}$ where N = 1,000
4. Test statistic: $t = \frac{z - z_0}{SE_z}$ where $z_0 = \frac{1}{2} \ln\left(\frac{1+0.7}{1-0.7}\right)$
5. P-value: $p = 1 - \Phi(t)$ where Φ is standard normal CDF
6. Decision: Reject $H_0$ if $p < 0.05$

---

### 5.2 Secondary Endpoint Test

**Test:** Binomial test for coverage rate

**Null Hypothesis:** $H_0: p_{\text{coverage}} = 0.90$

**Alternative Hypothesis:** $H_1: p_{\text{coverage}} \neq 0.90$ (two-tailed)

**Significance Level:** α = 0.05

**Procedure:**
1. Count number of test cases where $\bar{d}_{\text{obs}} \in [\bar{d} - 1.645 \cdot SE, \bar{d} + 1.645 \cdot SE]$
2. Compute observed coverage rate: $\hat{p} = k / N$ where k = count, N = 1,000
3. Binomial test: $p\text{-value} = 2 \times \min(P(X \leq k | p=0.9), P(X \geq k | p=0.9))$ where $X \sim \text{Binomial}(1000, 0.9)$
4. Decision: Fail to reject $H_0$ if $p > 0.05$ (calibration is acceptable)

---

### 5.3 Falsification Decision Tests (Per Image)

**Test 1 (High-Attribution):** $H_0: \mathbb{E}[d_{\text{high}}] \leq \tau_{\text{high}} = 0.75$ vs. $H_1: \mathbb{E}[d_{\text{high}}] > 0.75$

**Test 2 (Low-Attribution):** $H_0: \mathbb{E}[d_{\text{low}}] \geq \tau_{\text{low}} = 0.55$ vs. $H_1: \mathbb{E}[d_{\text{low}}] < 0.55$

**Test Statistics:**

$$t_{\text{high}} = \frac{\bar{d}_{\text{high}} - 0.75}{\sigma_{\text{high}} / \sqrt{200}}$$

$$t_{\text{low}} = \frac{\bar{d}_{\text{low}} - 0.55}{\sigma_{\text{low}} / \sqrt{200}}$$

**Bonferroni Correction:** Corrected significance level α = 0.05 / 2 = 0.025 (two tests per image)

**Decision Rule:** Attribution is NOT FALSIFIED if $p_{\text{high}} < 0.025$ AND $p_{\text{low}} < 0.025$

---

## 6. Multiple Comparisons Correction

**Scenario:** Testing 4-5 attribution methods (Grad-CAM, SHAP, LIME, Integrated Gradients, potentially Grad-CAM++)

**Issue:** Testing multiple methods increases family-wise Type I error rate (false discovery of "NOT FALSIFIED" status)

**Correction Method:** Benjamini-Hochberg False Discovery Rate (FDR) control

**Procedure:**
1. Compute raw p-values for primary endpoint test for all M methods
2. Sort p-values: $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(M)}$
3. Find largest k such that $p_{(k)} \leq \frac{k}{M} \cdot \alpha$ where α = 0.05
4. Reject null hypotheses for methods 1 through k (declare "NOT FALSIFIED")
5. Report both raw p-values and FDR-adjusted q-values

**Justification:** FDR control balances Type I error protection with statistical power, appropriate for exploratory comparison of multiple methods (Benjamini & Hochberg, 1995)

---

## 7. Data Exclusion Criteria

### 7.1 Image-Level Exclusions

**Criterion 1: Convergence Failure**

**Definition:** If >10% of counterfactuals (>20 out of 200) fail to converge within T=100 iterations for either $S_{\text{high}}$ or $S_{\text{low}}$, exclude image from analysis

**Rationale:** Insufficient counterfactual coverage compromises statistical validity

**Expected Exclusion Rate:** <2% based on pilot data (1.6% per-counterfactual failure rate, rare for >10% failures per image)

---

**Criterion 2: Non-Triviality Failure**

**Definition:** If $S_{\text{high}} = \emptyset$ or $S_{\text{low}} = \emptyset$, exclude from quantitative analysis (report separately as "trivial attribution")

**Rationale:** Cannot compute differential prediction without both feature sets

**Expected Exclusion Rate:** <3% based on pilot data

---

**Criterion 3: Extreme Outliers**

**Definition:** If geodesic distance measurement yields $d_g > \pi$ (mathematically impossible on hypersphere), exclude as numerical error

**Rationale:** Indicates floating-point overflow or implementation bug

**Expected Exclusion Rate:** 0% (should not occur with proper numerical stability handling)

---

### 7.2 Counterfactual-Level Exclusions

**Criterion 4: Plausibility Gate Violations**

**Definition:** Exclude individual counterfactuals with LPIPS ≥ 0.3 or if counterfactual set has FID ≥ 50

**Rationale:** Off-manifold perturbations yield unreliable validation

**Handling:** If >20% of counterfactuals fail plausibility gates for an image, exclude entire image from analysis

**Expected Exclusion Rate:** <5% based on pilot data (median LPIPS = 0.22, median FID = 41)

---

## 8. Subgroup Analyses (Pre-Registered)

### 8.1 Demographic Stratification

**Stratification Variables (LFW Annotations):**
- **Age:** Young (<30), Middle (30-50), Older (>50)
- **Gender:** Male, Female
- **Skin Tone:** Light, Dark (based on available annotations)

**Analyses:**
1. Compute falsification rate (% FALSIFIED) for each demographic subgroup
2. Test for disparity: Chi-square test for independence between demographic category and falsification verdict
3. Report effect sizes: Cramér's V for association strength
4. Fairness threshold: Flag as "HIGH DISPARITY" if falsification rate difference >10 percentage points across groups

**Justification:** Demographic fairness is critical for forensic deployment; systematic disparities indicate bias

---

### 8.2 Imaging Condition Stratification

**Stratification Variables:**
- **Pose:** Frontal (<15° rotation), Near-frontal (15-30°), Profile (>30°)
- **Occlusion:** None, Partial (sunglasses, beard), Heavy (surgical mask, hands)
- **Resolution:** High (>112×112), Medium (80-112), Low (<80×80)

**Analyses:**
1. Compute falsification rate and correlation ρ for each condition
2. Test for heterogeneity: ANOVA for ρ across conditions
3. Report known failure scenarios: conditions with >50% falsification rate

**Justification:** Real-world forensic images vary in quality; protocol must identify conditions where validation fails

---

## 9. Sensitivity Analyses (Pre-Registered)

### 9.1 Threshold Robustness

**Procedure:**
1. Re-run falsification protocol with threshold variations:
   - $\tau_{\text{high}} \in \{0.70, 0.75, 0.80\}$
   - $\tau_{\text{low}} \in \{0.50, 0.55, 0.60\}$
   - $\theta_{\text{high}} \in \{0.65, 0.70, 0.75\}$
   - $\theta_{\text{low}} \in \{0.35, 0.40, 0.45\}$
2. Count verdict flips: images transitioning from "NOT FALSIFIED" to "FALSIFIED" or vice versa
3. Report stability: % of images with consistent verdict across threshold variations

**Interpretation:** High stability (>90% consistent verdicts) indicates robust threshold selection; low stability (<80%) suggests borderline cases sensitive to arbitrary choices

---

### 9.2 Sample Size Reduction

**Procedure:**
1. Re-run with reduced counterfactual sample sizes: K ∈ {50, 100, 150, 200}
2. Measure impact on correlation estimates (ρ) and verdict consistency
3. Determine minimum K for acceptable precision (target: <5% change in ρ)

**Purpose:** Establish computational efficiency trade-offs; identify minimum sample size for resource-constrained settings

---

## 10. Reporting Standards

### 10.1 CONSORT-Style Flowchart

**Required Reporting:**
- Total images in test set: N_total
- Excluded (convergence failure): N_conv_fail
- Excluded (non-triviality): N_trivial
- Excluded (plausibility gates): N_implausible
- Analyzed: N_analyzed
- NOT FALSIFIED: N_pass
- FALSIFIED: N_fail

---

### 10.2 Effect Size Reporting

**Required Metrics:**
- Pearson correlation ρ with 95% CI (Fisher z-transformation)
- R² (explained variance)
- Mean Absolute Error (MAE) in radians
- Cohen's d for mean difference $\bar{d}_{\text{high}} - \bar{d}_{\text{low}}$

**Justification:** Statistical significance (p-values) alone is insufficient; effect sizes indicate practical importance

---

### 10.3 Uncertainty Quantification

**Required Reporting:**
- Confidence intervals for all point estimates (ρ, coverage rate, MAE)
- Bootstrapped 95% CIs for non-parametric estimates
- Prediction intervals for future observations (not just confidence intervals for means)

**Justification:** Forensic deployment requires transparent uncertainty communication

---

## 11. Deviations from Pre-Registration

**Policy:** Any deviations from this pre-registered protocol must be:
1. **Documented:** Record in "Protocol Deviations" appendix with timestamp and justification
2. **Justified:** Provide scientific rationale (e.g., unexpected data distribution, implementation bug discovered)
3. **Transparent:** Clearly distinguish pre-registered (confirmatory) from post-hoc (exploratory) analyses in manuscript
4. **Reported:** Include deviations section in final publication

**Prohibited Deviations:**
- Adjusting thresholds ($\tau_{\text{high}}, \tau_{\text{low}}, \rho_{\text{min}}$) based on test set results
- Selectively reporting subgroup analyses that yield favorable results
- Changing success criteria after observing data
- Excluding outliers without pre-specified criteria

**Permitted Deviations (with justification):**
- Fixing implementation bugs that affect all methods equally
- Adding supplementary exploratory analyses (clearly labeled)
- Adjusting dataset size if technical issues arise (e.g., dataset corruption)

---

## 12. Open Science Commitment

### 12.1 Pre-Registration

**Platform:** Open Science Framework (OSF) or AsPredicted.org

**Timing:** Pre-registration submitted BEFORE executing full-scale experiments on test set

**Public URL:** [To be inserted upon registration]

---

### 12.2 Code and Data Availability

**Code Release:** All implementation code released on GitHub with MIT license upon publication

**Repository Contents:**
- Falsification testing protocol (Sections 3.2-3.6 of Article B)
- Counterfactual generation pipeline (Algorithm 3.1)
- Statistical analysis scripts (hypothesis tests, sensitivity analyses)
- Visualization scripts (calibration plots, demographic stratifications)

**Data Release:** Benchmark datasets (LFW, CelebA) are publicly available; our analysis scripts and results will be released

---

### 12.3 Reproducibility

**Computational Environment:**
- Docker container specification with exact package versions
- Hardware specifications documented (GPU model, CUDA version)
- Random seed fixed: `random.seed(42)`, `torch.manual_seed(42)`
- Deterministic mode enabled: `torch.backends.cudnn.deterministic = True`

---

## 13. Timeline

**Phase 1: Pre-Registration (CURRENT)**
- Finalize threshold specifications
- Submit pre-registration to OSF
- Obtain timestamped cryptographic hash

**Phase 2: Experimental Execution (PLANNED)**
- Run falsification protocol on test set (N=1,000 LFW pairs)
- No threshold adjustments permitted
- Record all results (including negative findings)

**Phase 3: Analysis and Reporting (PLANNED)**
- Conduct pre-registered statistical tests
- Perform sensitivity analyses
- Generate visualizations and tables
- Write Sections 7-8 of Article B

**Phase 4: Publication (PLANNED)**
- Submit to IEEE T-IFS or Pattern Recognition
- Include pre-registration URL in manuscript
- Release code and data upon acceptance

---

## 14. Attestation

**I, [Principal Investigator Name], attest that:**

1. The thresholds and success criteria specified in this document were determined **before** analyzing the test set data.
2. These values are **frozen** and will not be adjusted based on test set performance.
3. Any deviations from this protocol will be documented transparently.
4. All results will be reported, including null findings.
5. This pre-registration will be publicly timestamped before experimental execution.

**Signature:** ___________________________

**Date:** ___________________________

**Witness (Advisor/Collaborator):** ___________________________

**Date:** ___________________________

---

## 15. Cryptographic Hash (To Be Generated)

**SHA-256 Hash of this Document:** [To be computed and recorded upon finalization]

**Purpose:** Tamper-proof evidence that thresholds were specified before data analysis

**Verification:** Hash will be publicly posted on OSF alongside timestamp

---

## 16. Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | [DATE] | Initial pre-registration | [Name] |

---

**END OF PRE-REGISTRATION DOCUMENT**

**Status:** DRAFT — Awaiting finalization and public timestamp

**Next Steps:**
1. Review thresholds with advisor/collaborators
2. Finalize exact wording of hypotheses
3. Submit to OSF or AsPredicted
4. Obtain cryptographic hash and public timestamp
5. Freeze document (no further edits permitted)
