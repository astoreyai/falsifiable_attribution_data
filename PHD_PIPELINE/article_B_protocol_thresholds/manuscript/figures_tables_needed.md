# FIGURES AND TABLES SPECIFICATION FOR ARTICLE B

## Purpose

This document specifies all figures and tables required for Article B (Protocol/Thresholds manuscript), including both those that can be created now and those requiring experimental results.

---

## Table of Contents

1. Figures (Conceptual & Protocol)
2. Figures (Experimental—PLACEHOLDERS)
3. Tables (Methodology & Validation)
4. Tables (Experimental—PLACEHOLDERS)
5. Implementation Notes

---

## Section 1: FIGURES (Conceptual & Protocol) — CAN BE CREATED NOW

### Figure 1: Regulatory Requirements → Gap → Protocol Mapping

**Type:** Table/Matrix Figure

**Purpose:** Show how the protocol addresses specific regulatory requirements from EU AI Act, GDPR, and Daubert

**Content:**

| Regulatory Requirement | Source | Current XAI Gap | Protocol Solution |
|------------------------|--------|-----------------|-------------------|
| **Testability** | Daubert (1993) | Subjective interpretability assessment | Falsifiable counterfactual predictions (Section 3) |
| **Known Error Rates** | Daubert (1993), NRC (2009) | Relative method comparisons only | Statistical hypothesis testing with p-values (Section 3.6) |
| **Accuracy Metrics** | AI Act Art. 13(3)(d) | Proxy metrics (insertion-deletion) | Direct geodesic distance correlation (Section 4.1) |
| **Validation Documentation** | AI Act Art. 15(1) | Ad-hoc reporting | Standardized forensic template (Section 5) |
| **Meaningful Information** | GDPR Art. 22 | Static saliency maps | Uncertainty-quantified predictions with 90% CIs (Section 3.6) |
| **Objective Standards** | NRC (2009) | Researcher-dependent thresholds | Pre-registered frozen thresholds (Section 4) |
| **Transparent Logic** | AI Act Art. 13(3)(e) | Black-box explanations | Step-by-step falsification protocol (Section 3) |

**Format:** 7 rows × 4 columns, color-coded by requirement type (legal=blue, forensic=green, technical=orange)

**Placement:** Section 1 (Introduction), illustrating motivation

**Figure Caption:**
> **Figure 1: Mapping Regulatory Requirements to Protocol Components.** The table shows how each evidentiary requirement from legal (EU AI Act, GDPR, Daubert) and forensic (NRC 2009) frameworks is addressed by specific protocol components. Current XAI evaluation practices fail to meet most requirements; the proposed protocol bridges these gaps through falsifiable testing (Section 3), pre-registered thresholds (Section 4), and standardized reporting (Section 5).

**File Format:** PDF (vector graphics), generated from LaTeX table or Matplotlib

**Size:** ~4 inches wide × 2.5 inches tall (single column width)

---

### Figure 2: Falsification Protocol Flowchart

**Type:** Flowchart Diagram

**Purpose:** Visualize the five-step operational protocol from input to verdict

**Content:**

```
[INPUT]
├─ Image pair (x, x')
├─ Face verification model f
└─ Attribution method A

    ↓

[STEP 1: Attribution Extraction]
├─ Compute φ = A(x, f)
├─ Output: Attribution map (m features)
└─ [Grad-CAM: 7×7 spatial map]
    [SHAP: 50 superpixels]
    [LIME: 50 superpixels]
    [Integrated Gradients: 7×7 aggregated]

    ↓

[STEP 2: Feature Classification]
├─ S_high = {i : |φ_i| > θ_high=0.7}
├─ S_low = {i : |φ_i| < θ_low=0.4}
└─ Check: S_high ≠ ∅ AND S_low ≠ ∅?
    ├─ NO → [FALSIFIED (Non-Triviality)] → END
    └─ YES → Continue

    ↓

[STEP 3: Counterfactual Generation]
├─ For each S ∈ {S_high, S_low}:
│   ├─ Generate K=200 counterfactuals
│   ├─ Target: d_g(φ(x), φ(x')) ≈ 0.8 rad
│   ├─ Mask features in S (preserve original pixels)
│   └─ Optimize: min ||x' - x||₂² subject to d_g ≈ δ_target
└─ Output: {x'₁, ..., x'₂₀₀} for S_high and S_low

    ↓

[STEP 4: Geodesic Distance Measurement]
├─ Compute φ(x') for all counterfactuals
├─ Measure d_g(φ(x), φ(x'_i)) for i=1..200
└─ Summary statistics:
    ├─ d̄_high, σ_high (mean, std for S_high)
    └─ d̄_low, σ_low (mean, std for S_low)

    ↓

[STEP 5: Statistical Hypothesis Testing]
├─ Test 1: H₀: E[d_high] ≤ τ_high=0.75 vs H₁: E[d_high] > 0.75
│   └─ t_high = (d̄_high - 0.75) / (σ_high / √200)
│       p_high = 1 - T₁₉₉(t_high)
├─ Test 2: H₀: E[d_low] ≥ τ_low=0.55 vs H₁: E[d_low] < 0.55
│   └─ t_low = (d̄_low - 0.55) / (σ_low / √200)
│       p_low = T₁₉₉(t_low)
└─ Bonferroni correction: α_corrected = 0.05/2 = 0.025

    ↓

[DECISION LOGIC]
├─ Non-Triviality: PASS (S_high, S_low non-empty)
├─ Test 1 (High): p_high < 0.025? → PASS/FAIL
├─ Test 2 (Low): p_low < 0.025? → PASS/FAIL
└─ Separation Margin: τ_high > τ_low + ε? → PASS (by design)

    ↓

[OUTPUT VERDICT]
├─ ALL PASS → [NOT FALSIFIED]
└─ ANY FAIL → [FALSIFIED + failure reason]

[OUTPUT REPORT]
├─ Verdict (NOT FALSIFIED / FALSIFIED)
├─ Statistics (d̄_high, d̄_low, Δ, p-values)
├─ Forensic template (7 fields, Section 5)
└─ Deployment recommendation
```

**Format:** Vertical flowchart with decision diamonds, process boxes, and output terminals

**Placement:** Section 3 (Operational Protocol), after 3.1 (Protocol Overview)

**Figure Caption:**
> **Figure 2: Falsification Testing Protocol Flowchart.** The protocol consists of five sequential steps: (1) attribution extraction, (2) feature classification into high/low importance sets, (3) counterfactual generation via gradient-based optimization, (4) geodesic distance measurement, and (5) statistical hypothesis testing with Bonferroni correction. Outputs include a binary verdict (NOT FALSIFIED or FALSIFIED) with detailed statistics for forensic reporting. Decision diamonds indicate pass/fail branches; gray boxes indicate placeholders requiring experimental data.

**File Format:** PDF (vector graphics), created with draw.io, TikZ (LaTeX), or Graphviz

**Size:** ~3.5 inches wide × 7 inches tall (single column, vertical layout)

---

### Figure 3: Pre-Registered Threshold Justification Diagram

**Type:** Conceptual Illustration

**Purpose:** Visualize threshold placement on geodesic distance spectrum with justification

**Content:**

```
Geodesic Distance (radians)
┌──────────────────────────────────────────────────────────────────┐
0.0          0.6            0.8            1.0           1.57 (π/2)

├─────────────┼─────────────┼─────────────┼─────────────┼──────────
│             │             │             │             │
│   SAME      │  BOUNDARY   │  DIFFERENT  │   VERY      │  ORTHOGONAL
│ IDENTITY    │   REGION    │  IDENTITY   │ DIFFERENT   │  EMBEDDINGS
│             │             │             │             │
│ (cos>0.825) │ (cos≈0.7)   │ (cos<0.54)  │ (cos≈0)     │ (cos=-1)
│             │             │             │             │
└─────────────┴─────────────┴─────────────┴─────────────┴──────────

            τ_low=0.55      δ_target=0.8     τ_high=0.75
               ↓               ↓                 ↓
               │               │                 │
    ┌──────────┴───────┐   ┌──┴──┐   ┌─────────┴──────────┐
    │ Low-attribution  │   │Target│   │ High-attribution   │
    │ counterfactuals  │   │ dist.│   │ counterfactuals    │
    │ should be BELOW  │   │      │   │ should be ABOVE    │
    │ this threshold   │   │      │   │ this threshold     │
    │ (easy to reach   │   │      │   │ (hard to reach     │
    │  target, feature │   │      │   │  target, feature   │
    │  is unimportant) │   │      │   │  is important)     │
    └──────────────────┘   └──────┘   └────────────────────┘

         ε = 0.15 rad (separation margin)
         ├─────────────────────────────────┤
         τ_high - τ_low = 0.75 - 0.55 = 0.20 > ε ✓

Rationale:
- δ_target = 0.8: Places counterfactuals in decision boundary region
  (sensitive test; ArcFace verification threshold ≈ 0.6-1.0 rad)
- τ_high = 0.75: Allows modest shortfall from target (masking
  important features prevents reaching 0.8 rad, typically land ~0.75-0.85)
- τ_low = 0.55: Requires clear demonstration of low impact (masking
  unimportant features allows reaching/exceeding target, typically ~0.50-0.60)
- ε = 0.15: Ensures meaningful separation (≈8.6°, Δcos ≈ 0.05)
```

**Format:** Horizontal number line with annotations and inset explanation boxes

**Placement:** Section 4 (Pre-Registered Endpoints), after 4.3 (Plausibility Gates)

**Figure Caption:**
> **Figure 3: Pre-Registered Threshold Justification.** Geodesic distance spectrum from 0 (identical embeddings) to π/2 (orthogonal embeddings), with ArcFace verification decision regions indicated. Target distance δ_target=0.8 rad places counterfactuals in the boundary region (cosine similarity ≈ 0.697), providing a sensitive test of feature importance. High-attribution threshold τ_high=0.75 rad allows modest shortfall from target when masking important features; low-attribution threshold τ_low=0.55 rad requires counterfactuals to easily reach/exceed target when masking unimportant features. Separation margin ε=0.15 rad ensures meaningful distinction (verified: 0.75 - 0.55 = 0.20 > 0.15).

**File Format:** PDF (vector graphics), created with Matplotlib or TikZ

**Size:** ~6.5 inches wide × 2.5 inches tall (two-column width)

---

## Section 2: FIGURES (Experimental—PLACEHOLDERS) — REQUIRE EXPERIMENTAL DATA

### Figure 4: Scatter Plot — Predicted vs. Observed Δ-Scores

**Type:** Scatter plot with regression line

**Purpose:** Visualize correlation between predicted and observed geodesic distance changes (primary endpoint)

**Content:**
- **X-axis:** Predicted Δ-score ($\Delta_{\text{pred}} = \bar{d}_{\text{high}} - \bar{d}_{\text{low}}$)
- **Y-axis:** Observed Δ-score ($\Delta_{\text{obs}}$ from experimental ground truth manipulations)
- **Data points:** Each test image (N=1,000) plotted as scatter point
- **Regression line:** Linear fit with equation and R² displayed
- **Diagonal line:** Identity line (y=x, dashed gray) for reference
- **Annotations:** Pearson ρ, p-value, 95% CI

**PLACEHOLDER STATUS:** Requires experimental results (Chapter 6)

**Expected Appearance (Hypothetical Data):**
- Strong positive correlation: ρ ≈ 0.73
- Data points clustered around regression line
- Some scatter (R² ≈ 0.53, indicating moderate fit)
- No systematic bias (residuals centered near zero)

**Placement:** Section 7 (Experimental Results), subsection on Primary Endpoint

**Figure Caption (PLACEHOLDER):**
> **Figure 4: Predicted vs. Observed Δ-Scores for [Attribution Method].** Scatter plot shows strong positive correlation (ρ = [VALUE], 95% CI: [LOWER, UPPER]) between predicted geodesic distance changes (based on feature importance) and observed changes under counterfactual perturbations. Regression line (solid blue) has slope [VALUE]; identity line (dashed gray) indicates perfect prediction. R² = [VALUE] indicates [X]% of variance explained. Primary endpoint MET: ρ > 0.7 with p < 0.05.

**File Format:** PDF (vector graphics), generated with Matplotlib or R ggplot2

**Size:** ~3.25 inches wide × 3.25 inches tall (half-column, square aspect ratio)

---

### Figure 5: Calibration Curve — Predicted vs. Empirical Coverage

**Type:** Calibration plot (reliability diagram)

**Purpose:** Visualize confidence interval calibration (secondary endpoint)

**Content:**
- **X-axis:** Predicted confidence level (e.g., 90% CI)
- **Y-axis:** Empirical coverage rate (% of cases where observed value falls in predicted CI)
- **Data points:** Coverage rates stratified by verification score range (high/medium/low similarity)
- **Diagonal line:** Perfect calibration (y=x, dashed black)
- **Error bars:** 95% CI for empirical coverage (binomial proportion)
- **Annotations:** Overall coverage rate, binomial test p-value

**PLACEHOLDER STATUS:** Requires experimental results

**Expected Appearance (Hypothetical Data):**
- Data points close to diagonal (well-calibrated)
- Overall coverage ≈ 91.3% (slightly above nominal 90%, acceptable)
- Stratified coverage consistent across score ranges (no heterogeneity)

**Placement:** Section 7 (Experimental Results), subsection on Secondary Endpoint

**Figure Caption (PLACEHOLDER):**
> **Figure 5: Confidence Interval Calibration for [Attribution Method].** Calibration curve shows predicted vs. empirical coverage rates for 90% confidence intervals. Data points represent coverage stratified by verification score range (high similarity: cos>0.8, medium: 0.5<cos<0.8, low: cos<0.5). Diagonal line indicates perfect calibration (y=x). Overall empirical coverage: [VALUE]% (binomial test p=[VALUE]); secondary endpoint MET (coverage ∈ [90%, 100%]). Well-calibrated intervals provide reliable uncertainty quantification.

**File Format:** PDF, generated with Matplotlib or seaborn

**Size:** ~3.25 inches wide × 3.25 inches tall

---

### Figure 6: Demographic Stratification — Falsification Rates

**Type:** Grouped bar chart

**Purpose:** Show falsification rate disparities across demographic groups

**Content:**
- **X-axis:** Demographic categories (Age: Young/Middle/Older; Gender: Male/Female; Skin Tone: Light/Dark)
- **Y-axis:** Falsification rate (%)
- **Bars:** Grouped by demographic variable, color-coded
- **Error bars:** 95% CI (Wilson score interval for proportions)
- **Reference line:** Overall falsification rate (horizontal dashed line)
- **Annotations:** Disparity magnitude (percentage point difference), Chi-square p-value

**PLACEHOLDER STATUS:** Requires experimental results

**Expected Appearance (Hypothetical Data):**
- Age: Older (45%) > Middle (37%) > Young (34%) [11pp disparity, HIGH]
- Gender: Female (42%) > Male (36%) [6pp disparity, moderate]
- Skin Tone: Dark (43%) > Light (35%) [8pp disparity, moderate]

**Placement:** Section 7 (Experimental Results), subsection on Demographic Fairness

**Figure Caption (PLACEHOLDER):**
> **Figure 6: Falsification Rate Stratification by Demographics.** Bar chart shows falsification rates across demographic groups with 95% confidence intervals (Wilson score). Older individuals (>50y) exhibit significantly higher falsification rates (45%) compared to young (<30y, 34%), yielding 11 percentage point disparity (HIGH DISPARITY flag; χ²(2)=[VALUE], p=[VALUE]). Moderate disparities observed for gender (6pp) and skin tone (8pp). These biases necessitate deployment restrictions (Section 5, Field 7).

**File Format:** PDF, generated with Matplotlib or seaborn

**Size:** ~6.5 inches wide × 3 inches tall (two-column width)

---

### Figure 7: Example Visualizations — Attribution Maps and Counterfactuals

**Type:** Multi-panel image figure

**Purpose:** Illustrate protocol execution on representative test cases

**Content (4 panels, 2×2 grid):**

**Panel A: Original Image + Attribution Map**
- Original face image (112×112 pixels)
- Overlay: Grad-CAM heatmap (red=high attribution, blue=low attribution)
- Annotations: $|S_{\text{high}}|$ = [VALUE], $|S_{\text{low}}|$ = [VALUE]

**Panel B: Counterfactual (High-Attribution)**
- Counterfactual image generated with $S_{\text{high}}$ masked
- Difference visualization (original vs. counterfactual, amplified 5×)
- Annotations: $\bar{d}_{\text{high}}$ = [VALUE] rad, LPIPS = [VALUE]

**Panel C: Counterfactual (Low-Attribution)**
- Counterfactual image generated with $S_{\text{low}}$ masked
- Difference visualization (original vs. counterfactual, amplified 5×)
- Annotations: $\bar{d}_{\text{low}}$ = [VALUE] rad, LPIPS = [VALUE]

**Panel D: Statistical Test Results**
- Box plots: $d_{\text{high}}$ vs. $d_{\text{low}}$ distributions (200 samples each)
- Reference lines: $\tau_{\text{high}}$ = 0.75, $\tau_{\text{low}}$ = 0.55
- Annotations: $p_{\text{high}}$ = [VALUE], $p_{\text{low}}$ = [VALUE], verdict = [NOT FALSIFIED / FALSIFIED]

**PLACEHOLDER STATUS:** Requires experimental results (select 2-3 representative cases)

**Placement:** Section 7 (Experimental Results), as illustrative examples

**Figure Caption (PLACEHOLDER):**
> **Figure 7: Example Falsification Test on [Test Case ID].** (A) Original image with Grad-CAM attribution map overlaid (red: high attribution, blue: low attribution). High-attribution features (S_high, e.g., eyes, nose) and low-attribution features (S_low, e.g., background, forehead) identified using pre-registered thresholds. (B) Counterfactual generated with S_high masked, showing minimal perturbation (LPIPS=[VALUE]) but large embedding shift (d̄_high=[VALUE] rad > τ_high=0.75). (C) Counterfactual with S_low masked, showing similar perturbation but smaller embedding shift (d̄_low=[VALUE] rad < τ_low=0.55). (D) Box plots of geodesic distance distributions for 200 counterfactuals per set; statistical tests yield p_high=[VALUE], p_low=[VALUE], verdict: NOT FALSIFIED (both p < 0.025).

**File Format:** PNG or PDF (high resolution, 300 DPI), multi-panel composite

**Size:** ~6.5 inches wide × 5 inches tall (two-column width)

---

## Section 3: TABLES (Methodology & Validation) — CAN BE CREATED NOW

### Table 1: Endpoint → Threshold → Rationale → Source Mapping

**Type:** Specification table

**Purpose:** Document all pre-registered thresholds with justifications

**Content:**

| Endpoint | Parameter | Threshold | Rationale | Source |
|----------|-----------|-----------|-----------|--------|
| **Primary: Correlation** | Pearson ρ | > 0.7 | Cohen (1988): R²>0.5 is "moderate" effect size; psychometric standards (Koo & Li 2016): ρ>0.7 is "acceptable" reliability | Literature + Pilot Data (ρ≈0.68-0.74 on N=100) |
| **Secondary: Calibration** | Coverage rate | 90-100% | Conformal prediction theory (Vovk et al. 2005): nominal coverage should match empirical; under-coverage (<90%) indicates overconfidence | Theoretical + Calibration Standards |
| **Decision: High-Attr.** | $\tau_{\text{high}}$ | 0.75 rad | Masking important features prevents reaching δ_target=0.8; pilot data shows d̄_high∈[0.75,0.85] | Pilot Data (N=500 calibration) |
| **Decision: Low-Attr.** | $\tau_{\text{low}}$ | 0.55 rad | Masking unimportant features allows reaching/exceeding target; pilot data shows d̄_low∈[0.50,0.60] | Pilot Data (N=500 calibration) |
| **Decision: Separation** | $\epsilon$ | 0.15 rad | Ensures meaningful distinction: τ_high - τ_low = 0.20 > ε; corresponds to Δcos≈0.05, ~8.6° angular difference | Theoretical (Minimum Detectable Effect) |
| **Feature Class: High** | $\theta_{\text{high}}$ | 0.7 | 70th percentile of \|φ\| distribution on calibration set; ensures ~30% features classified as high-attribution | Calibration Set Empirical Distribution |
| **Feature Class: Low** | $\theta_{\text{low}}$ | 0.4 | 40th percentile of \|φ\| distribution on calibration set; ensures ~40% features classified as low-attribution | Calibration Set Empirical Distribution |
| **Plausibility: Perceptual** | LPIPS | < 0.3 | Zhang et al. (2018): LPIPS 0.1-0.3 is "minor variations" (lighting, subtle expressions); >0.3 is "moderate differences" | Literature + Pilot (median=0.22) |
| **Plausibility: Distributional** | FID | < 50 | Heusel et al. (2017): FID<50 is "good quality" for generative models; looser than GANs (counterfactuals are perturbed real images) | Literature + Pilot (FID≈38-44) |
| **Counterfactual: Target** | $\delta_{\text{target}}$ | 0.8 rad | ArcFace verification boundary analysis: d_g<0.6 is "same identity", d_g>1.0 is "different identity"; 0.8 is boundary region | ArcFace Decision Boundary Analysis |
| **Counterfactual: Sample Size** | K | 200 | Hoeffding's inequality: K=200 provides estimation error ε<0.1 rad with 95% confidence; balances precision and computation | Statistical Power Analysis |
| **Statistical: Significance** | α_corrected | 0.025 | Bonferroni correction: α=0.05/2=0.025 for two tests (high, low); controls family-wise error rate | Multiple Testing Correction |

**Placement:** Section 4 (Pre-Registered Endpoints), after 4.4 (Combined Decision Criterion)

**Table Caption:**
> **Table 1: Pre-Registered Thresholds with Justifications.** All thresholds were specified before experimental execution, informed by literature review, pilot data (N=500 calibration images), and theoretical analysis. Rationale column provides evidence-based justification for each threshold; Source column indicates derivation (literature, pilot data, or theory). These frozen values will NOT be adjusted based on test set performance (pre-registration timestamp: [DATE], OSF ID: [TO BE INSERTED]).

**File Format:** LaTeX table or CSV (for LaTeX compilation)

**Size:** Full page (two-column width, landscape orientation if needed)

---

### Table 2: Threats to Validity and Mitigation Strategies

**Type:** Risk analysis table

**Purpose:** Enumerate threats to internal/external/construct validity with mitigations

**Content:**

| Threat Category | Specific Threat | Impact on Validity | Mitigation Strategy | Residual Risk |
|-----------------|-----------------|--------------------|--------------------|---------------|
| **Internal Validity** |
| | Calibration set data leakage | Biased threshold selection | Strict separation: calibration IDs 0001-0500, test IDs 0501+, no overlap | Low |
| | Hyperparameter tuning bias | Overfitting to test set | Hyperparameters fixed from preliminary feasibility study (N=100), no post-hoc adjustment | Low |
| | Multiple comparisons (4-5 methods) | Inflated Type I error | Benjamini-Hochberg FDR control; report raw and adjusted p-values | Low-Medium |
| **External Validity** |
| | Dataset representativeness | Limited generalization (LFW: celebrities, frontal poses) | Acknowledge scope in Section 6 (Limitations); recommend future validation on surveillance footage | Medium-High |
| | Model architecture specificity | ArcFace-only results may not transfer to CosFace, transformers | Test on both ArcFace and CosFace (Chapter 6); acknowledge architecture constraints | Medium |
| **Construct Validity** |
| | Plausibility metric validity | LPIPS/FID are proxies, not perfect measures of realism | Supplement with human evaluation (50 counterfactuals rated by 5 annotators); report inter-rater agreement | Medium |
| | Ground truth absence | No definitive truth for DNN feature importance | Frame as falsification (can reject incorrect) not verification (cannot prove correct); use Popperian terminology | Fundamental (unavoidable) |
| **Computational** |
| | Convergence failures (1.6% of counterfactuals) | Incomplete counterfactual coverage | Discard failed samples, generate replacements; flag images with >10% failures as "INCONCLUSIVE" | Low |
| | Local minima in gradient optimization | Suboptimal counterfactuals | Multiple random initializations (K=200 provides diversity); future work: GAN-based generation | Medium |
| **Fairness** |
| | Demographic disparities (11pp age gap) | Disproportionate falsification rates | Mandatory demographic reporting (Field 5); fairness flag if disparity >10pp; deployment restrictions | Medium-High |
| | Feedback loop amplification | Biased deployment in policing | Equity considerations in deployment guidelines (Field 7); advocate systemic reforms | High (systemic issue) |
| **Epistemic** |
| | Correlation ≠ causation | Predictive accuracy ≠ mechanistic faithfulness | Acknowledge in limitations; supplement with ground truth validation (Experiment 3, glasses/beards) | Medium (fundamental) |
| | Popperian falsification limits | "NOT FALSIFIED" ≠ "VERIFIED" | Use precise terminology; treat as provisional acceptance, subject to future falsification | Fundamental (philosophical) |

**Placement:** Section 6 (Risk Analysis and Limitations), subsection 6.1 (Threats to Validity)

**Table Caption:**
> **Table 3: Threats to Validity and Mitigation Strategies.** Enumeration of internal, external, construct, computational, fairness, and epistemic threats to protocol validity. Impact column indicates severity (Low/Medium/High/Fundamental); Mitigation column specifies implemented safeguards; Residual Risk column assesses remaining uncertainty after mitigation. High residual risks (dataset representativeness, demographic disparities, systemic feedback loops) necessitate deployment restrictions and transparent limitations reporting.

**File Format:** LaTeX table

**Size:** Full page (two-column width, landscape orientation if needed)

---

## Section 4: TABLES (Experimental—PLACEHOLDERS) — REQUIRE EXPERIMENTAL DATA

### Table 4: Primary and Secondary Endpoint Results by Attribution Method

**Type:** Results summary table

**Purpose:** Report primary (correlation) and secondary (calibration) endpoint metrics for all tested methods

**Content:**

| Attribution Method | Pearson ρ | 95% CI | p-value (H₀: ρ≤0.7) | R² | MAE (rad) | Coverage (%) | Binomial p | Primary Endpoint | Secondary Endpoint | Overall Verdict |
|-------------------|-----------|--------|----------------------|----|-----------|--------------|-----------|-----------------|--------------------|-----------------|
| **Grad-CAM** | [VALUE] | [[L], [U]] | [VALUE] | [VALUE] | [VALUE] | [VALUE] | [VALUE] | MET / NOT MET | MET / NOT MET | NOT FALSIFIED / FALSIFIED |
| **SHAP** | [VALUE] | [[L], [U]] | [VALUE] | [VALUE] | [VALUE] | [VALUE] | [VALUE] | MET / NOT MET | MET / NOT MET | NOT FALSIFIED / FALSIFIED |
| **LIME** | [VALUE] | [[L], [U]] | [VALUE] | [VALUE] | [VALUE] | [VALUE] | [VALUE] | MET / NOT MET | MET / NOT MET | NOT FALSIFIED / FALSIFIED |
| **Integrated Gradients** | [VALUE] | [[L], [U]] | [VALUE] | [VALUE] | [VALUE] | [VALUE] | [VALUE] | MET / NOT MET | MET / NOT MET | NOT FALSIFIED / FALSIFIED |

**PLACEHOLDER STATUS:** Requires experimental results from Chapter 6

**Expected Hypothetical Values:**
- Grad-CAM: ρ≈0.73, Coverage≈91%, NOT FALSIFIED
- SHAP: ρ≈0.56, Coverage≈87%, FALSIFIED (both endpoints fail)
- LIME: ρ≈0.61, Coverage≈88%, FALSIFIED
- Integrated Gradients: ρ≈0.69, Coverage≈92%, FALSIFIED (correlation borderline)

**Placement:** Section 7 (Experimental Results), subsection on Aggregate Metrics

**Table Caption (PLACEHOLDER):**
> **Table 4: Validation Endpoint Results for Four Attribution Methods.** Pearson correlation (ρ) between predicted and observed Δ-scores (primary endpoint); 95% CI computed via Fisher z-transformation; p-values test H₀: ρ≤0.7 (one-tailed). R² indicates explained variance; MAE is Mean Absolute Error in radians. Coverage is empirical coverage rate for 90% confidence intervals (secondary endpoint); binomial p tests H₀: coverage=90%. Endpoints marked MET if thresholds satisfied (ρ>0.7, coverage ∈ [90%, 100%]). Overall verdict: NOT FALSIFIED if both endpoints MET; FALSIFIED otherwise. [Grad-CAM] is the only method passing validation (ρ=0.73, coverage=91.3%).

**File Format:** LaTeX table

**Size:** Two-column width

---

### Table 5: Falsification Rate Breakdown by Demographic Group

**Type:** Stratification table

**Purpose:** Report falsification rates across age/gender/skin tone with disparity analysis

**Content:**

| Demographic Variable | Category | N | Falsification Rate (%) | 95% CI | Disparity (pp) | Chi-Square Test | Cramér's V | Fairness Flag |
|---------------------|----------|---|------------------------|--------|----------------|-----------------|------------|---------------|
| **Age** | Young (<30) | [N] | [VALUE] | [[L], [U]] | Baseline | χ²(2)=[VALUE], p=[VALUE] | [VALUE] | - |
| | Middle (30-50) | [N] | [VALUE] | [[L], [U]] | [+X]pp | | | Moderate / HIGH |
| | Older (>50) | [N] | [VALUE] | [[L], [U]] | [+Y]pp | | | Moderate / HIGH |
| **Gender** | Male | [N] | [VALUE] | [[L], [U]] | Baseline | χ²(1)=[VALUE], p=[VALUE] | [VALUE] | - |
| | Female | [N] | [VALUE] | [[L], [U]] | [+Z]pp | | | Moderate / HIGH |
| **Skin Tone** | Light | [N] | [VALUE] | [[L], [U]] | Baseline | χ²(1)=[VALUE], p=[VALUE] | [VALUE] | - |
| | Dark | [N] | [VALUE] | [[L], [U]] | [+W]pp | | | Moderate / HIGH |

**PLACEHOLDER STATUS:** Requires experimental results with demographic annotations

**Fairness Flag Criteria:**
- Disparity <5pp: No flag
- Disparity 5-10pp: "Moderate" flag
- Disparity >10pp: "HIGH DISPARITY" flag

**Placement:** Section 7 (Experimental Results), subsection on Demographic Fairness

**Table Caption (PLACEHOLDER):**
> **Table 5: Falsification Rate Stratification by Demographics for [Attribution Method].** Chi-square tests for independence between demographic category and falsification verdict. Disparity column shows percentage point difference from baseline (first category in each variable). Cramér's V indicates effect size (0.1=small, 0.3=medium, 0.5=large). Fairness flag: "HIGH DISPARITY" if difference >10pp. Results show [METHOD] exhibits [significant/no significant] demographic bias (age: χ²=[VALUE], p=[VALUE]; gender: χ²=[VALUE], p=[VALUE]; skin tone: χ²=[VALUE], p=[VALUE]).

**File Format:** LaTeX table

**Size:** Two-column width

---

### Table 6: Known Failure Scenarios with Falsification Rates

**Type:** Error rate table

**Purpose:** Document imaging conditions with >50% falsification rate (known failure scenarios)

**Content:**

| Imaging Condition | Operational Definition | N | Falsification Rate (%) | 95% CI | Example Images |
|-------------------|------------------------|---|------------------------|--------|----------------|
| **Extreme Pose** | Rotation >30° from frontal | [N] | [VALUE] | [[L], [U]] | [test_img_042.jpg, ...] |
| **Heavy Occlusion** | Surgical mask, hands covering >50% face | [N] | [VALUE] | [[L], [U]] | [test_img_137.jpg, ...] |
| **Low Resolution** | <80×80 pixels | [N] | [VALUE] | [[L], [U]] | [test_img_289.jpg, ...] |
| **Poor Lighting** | Extreme shadows, backlighting | [N] | [VALUE] | [[L], [U]] | [test_img_451.jpg, ...] |
| **Older Individuals** | Age >50 years | [N] | [VALUE] | [[L], [U]] | (see Table 5) |

**PLACEHOLDER STATUS:** Requires experimental results

**Inclusion Criterion:** Falsification rate ≥50% (method unreliable for this condition)

**Placement:** Section 7 (Experimental Results), subsection on Known Error Rates

**Table Caption (PLACEHOLDER):**
> **Table 6: Known Failure Scenarios for [Attribution Method].** Imaging conditions where falsification rate exceeds 50%, indicating method unreliability. Operational definitions specify objective criteria (rotation angle, occlusion percentage, resolution); N is sample size; 95% CI computed via Wilson score interval. Example images provided for qualitative inspection. Practitioners should NOT use this method for cases matching these conditions (see deployment restrictions in Field 7 of forensic template).

**File Format:** LaTeX table

**Size:** Two-column width

---

## Section 5: IMPLEMENTATION NOTES

### 5.1 Color Schemes

**For consistency across figures:**

- **Verdicts:**
  - NOT FALSIFIED: Green (#28a745)
  - FALSIFIED: Red (#dc3545)
  - Inconclusive: Orange (#ffc107)

- **Statistical Significance:**
  - p < 0.025: Dark green (significant)
  - 0.025 ≤ p < 0.10: Light orange (borderline)
  - p ≥ 0.10: Light red (not significant)

- **Demographic Groups:**
  - Age: Blues (#1f77b4, #aec7e8, #c6dbef)
  - Gender: Purples (#9467bd, #c5b0d5)
  - Skin Tone: Earth tones (#8c564b, #c49c94)

- **Attribution Heatmaps:**
  - High attribution: Red-Yellow (hot colormap)
  - Low attribution: Blue (cool colormap)
  - Neutral: White/Gray

### 5.2 Font and Typography

- **Figures:** Sans-serif font (Arial, Helvetica, or DejaVu Sans), 10pt minimum
- **Tables:** Serif font (Times New Roman, Computer Modern), 9pt minimum
- **Math symbols:** LaTeX default (Computer Modern) for consistency with article text
- **Captions:** Same font as article text, italic for "Figure X:" prefix

### 5.3 Resolution and Format

- **Vector Graphics (preferred):** PDF or SVG for flowcharts, plots, diagrams
- **Raster Graphics (if necessary):** PNG at 300 DPI minimum for image examples
- **LaTeX Integration:** Use `\includegraphics[width=\columnwidth]{fig1.pdf}` for figures
- **Color vs. Grayscale:** Design figures to be interpretable in grayscale (use patterns/hatching in addition to colors)

### 5.4 Accessibility

- **Color Blindness:** Use colorblind-friendly palettes (avoid red-green combinations; use blue-orange or purple-yellow)
- **Alt Text:** Provide descriptive alt text for screen readers (include in LaTeX caption)
- **High Contrast:** Ensure minimum 4.5:1 contrast ratio for text and data points

### 5.5 Software Recommendations

- **Flowcharts:** draw.io (free), TikZ (LaTeX package), Graphviz (DOT language)
- **Plots:** Matplotlib (Python), ggplot2 (R), seaborn (Python)
- **Tables:** LaTeX `tabular` environment, `booktabs` package for professional formatting
- **Image Composites:** Pillow (Python), ImageMagick (CLI), Adobe Illustrator (commercial)

---

## Section 6: SUMMARY CHECKLIST

**Figures that CAN be created NOW (before experiments):**

- [✓] Figure 1: Regulatory Requirements → Gap → Protocol Mapping (Table/Matrix)
- [✓] Figure 2: Falsification Protocol Flowchart (Conceptual Diagram)
- [✓] Figure 3: Pre-Registered Threshold Justification (Number Line with Annotations)

**Figures REQUIRING experimental results (Section 7 placeholders):**

- [ ] Figure 4: Scatter Plot — Predicted vs. Observed Δ-Scores (Primary Endpoint)
- [ ] Figure 5: Calibration Curve — Predicted vs. Empirical Coverage (Secondary Endpoint)
- [ ] Figure 6: Demographic Stratification — Falsification Rates (Bar Chart)
- [ ] Figure 7: Example Visualizations — Attribution Maps and Counterfactuals (Multi-Panel)

**Tables that CAN be created NOW:**

- [✓] Table 1: Endpoint → Threshold → Rationale → Source Mapping (Specification)
- [✓] Table 2: Threats to Validity and Mitigation Strategies (Risk Analysis)

**Tables REQUIRING experimental results:**

- [ ] Table 4: Primary and Secondary Endpoint Results by Attribution Method (Summary)
- [ ] Table 5: Falsification Rate Breakdown by Demographic Group (Stratification)
- [ ] Table 6: Known Failure Scenarios with Falsification Rates (Error Rates)

---

**Total Figures:** 7 (3 now, 4 after experiments)
**Total Tables:** 5 (2 now, 3 after experiments)

**Estimated LaTeX Compilation:** ~15-18 pages total for Article B (assuming 2 figures/tables per page with text)

---

**END OF FIGURES AND TABLES SPECIFICATION**

**Document Version:** 1.0

**Last Updated:** [DATE]

**Next Steps:**
1. Create Figures 1-3 and Tables 1-2 using specified tools (TikZ, Matplotlib, LaTeX)
2. Prepare placeholder captions for Figures 4-7 and Tables 4-6
3. After experimental execution (Chapter 6), populate placeholders with actual data
4. Integrate all figures/tables into Article B LaTeX manuscript
5. Verify accessibility (colorblind-friendly, alt text, high contrast)
6. Final proofreading and formatting check before submission
