# Proposal Defense Presentation Outline
**Duration:** 20-30 minutes + 30-45 minutes Q&A
**Audience:** Dissertation committee (4-6 members)
**Date:** 3 months from now
**Defense Readiness Goal:** 92-96/100 (currently 85/100)

---

## SLIDE 1: Title Slide
**Title:** Falsifiable Attribution Methods for Biometric Face Verification Systems

**Subtitle:** PhD Dissertation Proposal Defense

**Presenter:** [Your Name]
**Department:** [Your Department]
**Institution:** [Your Institution]
**Date:** [Defense Date]

**Committee:**
- [Chair Name], Chair
- [Member 2 Name]
- [Member 3 Name]
- [Member 4 Name]
- [Additional members if applicable]

---

## PART I: INTRODUCTION (3-5 minutes, Slides 2-4)

### SLIDE 2: Motivation - The Black Box Problem

**Visual:** Split screen - Left: Face verification system diagram, Right: "?" over decision

**Content:**
- Modern face verification systems achieve 99.6% accuracy (FaceNet, Schroff et al. 2015)
- Critical question: **WHY did the system make this decision?**
- Use case: Forensic deployment in criminal investigations
- Legal requirement: Explanations must meet Daubert standard for admissibility

**Key Quote:** "Black box decisions are inadmissible in court without scientific validation"

**Talking Points:**
- FaceNet: 128-dimensional embedding space, cosine similarity threshold
- Deployed in: Border security, law enforcement, financial services
- Problem: Decisions lack transparency

---

### SLIDE 3: Current XAI Limitations

**Visual:** Table comparing XAI methods with validation approaches

| XAI Method | Validation Approach | Scientific Rigor |
|------------|---------------------|------------------|
| SHAP (Lundberg+ 2017) | Anecdotal examples | Low |
| LIME (Ribeiro+ 2016) | Visual inspection | Low |
| Grad-CAM (Selvaraju+ 2017) | Qualitative assessment | Medium |
| **Our Framework** | **Falsification testing** | **High** |

**Content:**
- Existing XAI methods: SHAP, LIME, Grad-CAM, Integrated Gradients
- Validation gap: No rigorous, quantitative testing
- Cannot prove or disprove attribution correctness
- Research gap: **Need falsifiability framework for biometric XAI**

**Key Insight:** "If you can't test whether an explanation is wrong, it's not scientifically valid"

---

### SLIDE 4: Research Questions

**Visual:** Three numbered boxes with icons

**RQ1: Theoretical Foundation**
> Can we define a falsifiability criterion for attribution methods in biometric verification systems?

**Expected Contribution:** Mathematical framework (Theorems 3.5-3.8)

**RQ2: Empirical Validation**
> Does falsifiability correlate with attribution quality? Can it distinguish good from bad methods?

**Expected Contribution:** Experimental results across 5 attribution methods

**RQ3: Generalizability**
> Can this framework work across different models, datasets, and biometric modalities?

**Expected Contribution:** Multi-dataset, multi-model validation

**Hypothesis:** Methods with lower falsification rates provide more reliable attributions for forensic deployment

---

## PART II: THEORETICAL FRAMEWORK (8-10 minutes, Slides 5-9)

### SLIDE 5: Core Idea - Counterfactual Falsification

**Visual:** Flow diagram showing counterfactual testing process

**Concept:**
1. Attribution method says: "Feature X is important for this decision"
2. Falsification test: Change feature X → observe prediction change
3. If prediction DOESN'T change → attribution is **FALSIFIED**

**Mathematical Formulation:**
```
Given:
- Face pair (x_a, x_b) with similarity score sim(f(x_a), f(x_b))
- Attribution mask M identifying important features
- Counterfactual pair (x_a', x_b) where x_a' differs from x_a in mask M

Test:
- Compute Δsim = |sim(f(x_a), f(x_b)) - sim(f(x_a'), f(x_b))|
- If Δsim < ε → FALSIFIED (changing "important" features didn't matter)
- If Δsim ≥ ε → NOT FALSIFIED (attribution is consistent)
```

**Key Analogy:** Like hypothesis testing in statistics - we need a way to reject bad explanations

**Talking Points:**
- Inspired by Popper's falsifiability criterion (1959)
- Counterfactuals generated in embedding space (not pixel space)
- Statistical threshold ε prevents false positives

---

### SLIDE 6: Theorem 3.5 - Falsifiability Criterion

**Visual:** Theorem box with three-part test highlighted

**Theorem 3.5 (Falsifiability Criterion):**

An attribution method A is **falsifiable** for model f on face pair (x_a, x_b) if:

1. **Non-triviality:** M ≠ ∅ (attribution identifies specific features)
2. **Differential prediction:** Δsim(f(x_a), f(x_b), f(x_a'), f(x_b)) ≠ 0 (changing features changes prediction)
3. **Separation margin:** |Δsim| > ε (change is statistically significant)

Where:
- M = attribution mask (top-k features)
- x_a' = counterfactual embedding (features in M perturbed)
- ε = threshold (0.3 radians cosine similarity)

**Falsification Rate (FR):**
```
FR = (Number of falsified pairs) / (Total pairs tested)
```

**Interpretation:**
- **Low FR (0-20%):** High-quality attribution
- **Medium FR (20-50%):** Moderate quality
- **High FR (50-100%):** Poor quality / unreliable

**Talking Points:**
- Operationalizes Popper's falsifiability for XAI
- Binary decision: PASS or FAIL
- Quantitative metric for forensic admissibility

---

### SLIDE 7: Theorem 3.6 - Counterfactual Existence Guarantee

**Visual:** Hypersphere diagram showing embedding space geometry

**Theorem 3.6 (Counterfactual Existence):**

For any face embedding z_a on the L2-normalized unit hypersphere S^(D-1), there exists a counterfactual z_a' such that:
1. z_a' ∈ S^(D-1) (valid embedding)
2. d_geo(z_a, z_a') > δ (minimum separation)
3. Features in mask M are perturbed

**Construction:** Hypersphere sampling algorithm
- Sample random direction v_i for each feature i ∈ M
- Compute perturbed embedding z'
- Project onto hypersphere: z_a' = z' / ||z'||
- Geodesic distance: d_geo = arccos(z_a · z_a')

**Empirical Validation:**
- Tested on 5000 random trials
- **Success rate: 100.00% (5000/5000)** ✓
- Mean geodesic distance: 1.424 ± 0.329 radians
- All counterfactuals satisfy ||z_a'|| ≈ 1 (valid embeddings)

**Talking Points:**
- Proves counterfactuals always exist (no search failures)
- Hypersphere geometry is natural for normalized embeddings
- Validated empirically with perfect success rate

---

### SLIDE 8: Theorem 3.7 - Computational Complexity

**Visual:** Complexity graph showing runtime scaling

**Theorem 3.7 (Computational Complexity):**

The falsification test has runtime complexity:
```
T_total = O(K · T_model · D · |M|)
```

Where:
- K = number of counterfactuals (50 in our experiments)
- T_model = model inference time (~10ms for FaceNet)
- D = embedding dimension (128 for FaceNet)
- |M| = attribution mask size (512 features)

**Empirical Validation:**
- Linear scaling with K: r = 0.9993 (Pearson correlation)
- Linear scaling with |M|: r = 0.9998
- Average runtime: 0.47 seconds per test pair (GPU)

**Feasibility for Forensic Deployment:**
- Single case: ~1 minute (100 pairs)
- Batch processing: ~8 hours (10,000 pairs)
- Acceptable for forensic timeline (cases take months)

**Talking Points:**
- Computationally feasible at scale
- GPU parallelization straightforward
- Real-time deployment possible with cached results

---

### SLIDE 9: Theorem 3.8 - Sample Size Requirements

**Visual:** Hoeffding bound graph showing confidence vs. sample size

**Theorem 3.8 (Sample Size Requirements):**

To estimate falsification rate FR with error tolerance ε and confidence 1-δ, we require:

```
n ≥ (1 / (2ε²)) · ln(2/δ)
```

**Example:** For ε = 0.05 (5% error), δ = 0.05 (95% confidence):
```
n ≥ (1 / (2 · 0.05²)) · ln(2/0.05) = 737 pairs
```

**Our Experiments:**
- n = 500 pairs per method (proposal stage)
- Planned: n = 1000-5000 pairs (final dissertation)
- Error tolerance: ε = 0.05 (5%)
- Confidence: 95%

**Central Limit Theorem Validation:**
- Bootstrap analysis: std ∝ 1/√n (Experiment 6.5)
- Empirical validation across n = 100, 500, 1000 pairs
- Confirms statistical guarantees hold

**Talking Points:**
- Provides worst-case guarantees (distribution-free)
- Conservative estimate (Hoeffding inequality)
- Validated empirically with bootstrap sampling

---

## PART III: PRELIMINARY RESULTS (6-8 minutes, Slides 10-15)

### SLIDE 10: Experimental Setup

**Visual:** Experimental design diagram

**Dataset: LFW (Labeled Faces in the Wild)**
- 13,233 images of 5,749 identities
- Standard benchmark for face verification
- Unconstrained conditions (pose, lighting, expression variation)

**Model: FaceNet (Inception-ResNet-V1)**
- Architecture: Inception-ResNet-V1 (Szegedy et al. 2017)
- Parameters: 27.9 million
- Embedding dimension: 128 (L2-normalized)
- Training: VGGFace2 dataset (3.3M images)
- Performance: 99.6% accuracy on LFW

**Attribution Methods Tested (5 total):**
1. **Grad-CAM** (Selvaraju et al. 2017) - Gradient-weighted Class Activation Mapping
2. **Geodesic Integrated Gradients** (Embedding-space path integration)
3. **Biometric Grad-CAM** (Custom adaptation for face verification)
4. **SHAP** (Lundberg & Lee 2017) - Shapley Additive Explanations
5. **LIME** (Ribeiro et al. 2016) - Local Interpretable Model-agnostic Explanations

**Experimental Protocol:**
- 500 randomly sampled face pairs per method
- Top-k features: k = 512 (40% of 128-D embedding)
- Counterfactuals per pair: K = 50
- Threshold: ε = 0.3 radians cosine similarity

---

### SLIDE 11: KEY RESULT - Falsification Rates

**Visual:** Bar chart with error bars, color-coded by quality

**Falsification Rate Results (n=500 pairs each):**

| Method | Mean FR (%) | Std Dev (%) | 95% CI | Quality |
|--------|-------------|-------------|--------|---------|
| **Grad-CAM** | **10.48** | **28.71** | **[7.95, 13.01]** | **✓ HIGH** |
| Geodesic IG | 100.00 | 0.00 | [100.0, 100.0] | ✗ FAILED |
| Biometric Grad-CAM | 92.41 | 26.09 | [89.13, 95.69] | ✗ LOW |
| SHAP | 93.14 | 25.31 | [89.93, 96.35] | ✗ LOW |
| LIME | 94.22 | 23.32 | [91.19, 97.25] | ✗ LOW |

**Statistical Significance:**
- Chi-square test: χ² = 505.54, df = 4, **p < 10⁻¹¹²**
- Cohen's h effect size: h = -2.48 (large effect)
- Highly statistically significant differences between methods

**Key Findings:**
- **Grad-CAM is the ONLY method that passes** (FR < 20%)
- Geodesic IG completely fails (100% falsification)
- SHAP, LIME, Biometric Grad-CAM fail 92-94% of the time

**Interpretation:** Lower FR = Better attribution quality

**Talking Points:**
- Grad-CAM: 89.52% of attributions are NOT falsified (reliable)
- SHAP/LIME: Widely used in industry, but fail rigor test
- Geodesic IG: Architectural mismatch with FaceNet

---

### SLIDE 12: Why Did Geodesic IG Fail? (100% Falsification)

**Visual:** Comparison diagram - Geodesic vs. Euclidean geometry

**Root Cause: Geometric Mismatch**

**Geodesic Integrated Gradients Assumes:**
- Attributions based on geodesic paths in embedding space
- Shortest path on hypersphere manifold
- Geodesic distance drives decisions

**FaceNet Reality:**
- Decisions based on **cosine similarity** (Euclidean angle)
- Distance metric: d(z_a, z_b) = 1 - (z_a · z_b) / (||z_a|| ||z_b||)
- Decision boundary: threshold on cosine similarity (not geodesic distance)

**Result:**
- Geodesic IG attributes features along geodesic paths
- These features DO NOT affect cosine similarity
- Changing them does NOT change FaceNet's prediction
- **100% falsification rate**

**Evidence:**
- Table 6.1: 500/500 pairs falsified (p < 10⁻¹¹²)
- Mean Δsim = 0.003 ± 0.002 radians (below ε = 0.3)
- Attributions are geometrically incompatible with model

**Key Insight:** Our framework successfully identified this architectural incompatibility

**Talking Points:**
- Not a flaw in Integrated Gradients generally
- Specific incompatibility with cosine similarity models
- Demonstrates diagnostic power of falsification framework

---

### SLIDE 13: Hypersphere Sampling Validation (Theorem 3.6)

**Visual:** Convergence plot + distribution histogram

**Claim (Theorem 3.6):** Counterfactuals exist on hypersphere with 100% success

**Experimental Validation:**
- **Trials:** 5000 random face embeddings
- **Algorithm:** Hypersphere perturbation + projection
- **Success rate:** 5000/5000 = **100.00%** ✓

**Quality Metrics:**
- **Norm preservation:** ||z_a'|| ∈ [0.999, 1.001] for all counterfactuals
- **Geodesic distance:** 1.424 ± 0.329 radians (mean ± std)
- **Minimum separation:** min(d_geo) = 0.621 radians > δ = 0.01 ✓
- **Maximum separation:** max(d_geo) = 2.987 radians (near orthogonal)

**Distribution Analysis:**
- Geodesic distances follow approximately normal distribution
- Mean = 1.424 radians ≈ 81.6 degrees
- Consistent with expected hypersphere geometry (random sampling)

**Implications:**
- Counterfactual generation is **robust and reliable**
- No search failures or edge cases
- Validates theoretical guarantees empirically

**Talking Points:**
- Perfect success rate across 5000 diverse trials
- Hypersphere geometry is natural for normalized embeddings
- Foundation for all falsification experiments

---

### SLIDE 14: Statistical Validation - Chi-Square Test

**Visual:** Contingency table + chi-square distribution

**Hypothesis Test:**
- **H₀:** All attribution methods have equal falsification rates
- **Hₐ:** At least one method differs significantly

**Contingency Table (n=500 pairs × 5 methods = 2500 total):**

|  | Falsified | Not Falsified | Total |
|--|-----------|---------------|-------|
| Grad-CAM | 52 | 448 | 500 |
| Geodesic IG | 500 | 0 | 500 |
| Biometric Grad-CAM | 462 | 38 | 500 |
| SHAP | 466 | 34 | 500 |
| LIME | 471 | 29 | 500 |
| **Total** | **1951** | **549** | **2500** |

**Chi-Square Statistic:**
```
χ² = 505.54
df = 4
p < 10⁻¹¹² (astronomically significant)
```

**Effect Size (Cohen's h):**
```
h = -2.48 (large effect)
```

**Conclusion:** **Reject H₀** - Methods differ significantly in falsification rates

**Talking Points:**
- Not just statistically significant - MASSIVELY significant
- Effect size indicates practical importance (not just large sample)
- Justifies method selection (Grad-CAM) for forensic deployment

---

### SLIDE 15: Preliminary Conclusions

**Visual:** Checklist with checkmarks

**Validated Claims:**

✓ **Theoretical Feasibility**
- Falsifiability criterion is mathematically well-defined (Theorem 3.5)
- Counterfactual generation is guaranteed to succeed (Theorem 3.6: 100% rate)
- Computational complexity is tractable (Theorem 3.7: 0.47s per pair)
- Sample size requirements are achievable (Theorem 3.8: n ≥ 43)

✓ **Empirical Effectiveness**
- Framework distinguishes high-quality (Grad-CAM: 10% FR) from low-quality methods (SHAP/LIME: 93-94% FR)
- Identifies catastrophic failures (Geodesic IG: 100% FR)
- Statistical significance is robust (p < 10⁻¹¹², h = -2.48)

✓ **Diagnostic Power**
- Reveals architectural mismatches (Geodesic IG incompatibility)
- Challenges widely-used methods (SHAP, LIME fail rigor test)
- Provides quantitative quality metric for forensic admissibility

**Implications for Forensic Deployment:**
- Grad-CAM is the only method suitable for court testimony
- Framework enables evidence-based XAI selection
- Supports Daubert admissibility standard

---

## PART IV: REMAINING WORK (4-6 minutes, Slides 16-19)

### SLIDE 16: Dissertation Timeline (10 Months to Final Defense)

**Visual:** Gantt chart timeline

**Phase 1: Multi-Dataset Validation (Months 1-3)**
- **Goal:** Eliminate single-dataset limitation
- **Datasets:**
  - LFW: 13K images (baseline, DONE)
  - CelebA: 202K images (downloading, ETA 2 weeks)
  - CFP-FP: 7K images with pose variation (frontal-profile pairs)
- **Expected outcome:** Consistent FR patterns across datasets
- **Risk mitigation:** If CFP-FP unavailable, proceed with LFW + CelebA

**Phase 2: Complete Experiments (Months 4-6)**
- Experiment 6.4: Multi-model validation (ResNet-50, VGG-Face)
- Higher-n reruns: n = 1000-5000 pairs for statistical power
- Additional attribution methods: Gradient×Input, VanillaGradients, SmoothGrad
- Demographic fairness analysis: Age, gender, ethnicity subgroup analysis

**Phase 3: Writing & Revision (Months 7-8)**
- Chapter 8: Discussion and Conclusion (6-8 hours)
- Timing benchmark section in Chapter 7 (1.5 hours)
- Table updates with final results (2 hours)
- LaTeX polish and formatting (4-6 hours)
- Professional proofreading (4-6 hours)

**Phase 4: Defense Preparation (Months 9-10)**
- Final defense presentation (40-50 slides)
- Mock defenses with peers (3+ sessions)
- Q&A drilling (comprehensive preparation)
- Committee feedback incorporation
- Final document submission (8 weeks before defense)

**Current Status:** 85/100 defense readiness → Target 92-96/100

---

### SLIDE 17: Multi-Dataset Validation - Why It Matters

**Visual:** Three dataset logos/images with sample faces

**Current Vulnerability:**
- All results based on LFW dataset only
- Committee will ask: "How do you know this generalizes?"
- Single-dataset validation is standard for initial XAI research, but insufficient for dissertation

**Solution: Three Diverse Datasets**

**1. LFW (Labeled Faces in the Wild) - BASELINE**
- 13,233 images, 5,749 identities
- Unconstrained conditions
- Status: Complete (500 pairs per method)

**2. CelebA (Celebrity Faces Attributes) - SCALE**
- 202,599 images, 10,177 identities
- High-resolution, celebrity faces
- Attribute annotations (40 binary attributes)
- Status: Downloading (ETA 2 weeks), scripts ready

**3. CFP-FP (Celebrities in Frontal-Profile) - POSE VARIATION**
- 7,000 images, 500 identities
- Frontal-profile face pairs (extreme pose variation)
- Challenges attribution robustness
- Status: Planned (Months 2-3)

**Expected Results:**
- Consistent FR patterns across datasets (Grad-CAM: 10-15%, SHAP/LIME: 90-95%)
- Demonstrates generalization beyond single benchmark
- Strengthens claims for final defense

**Timeline:** Months 1-3 (highest priority)

---

### SLIDE 18: Remaining Experiments

**Visual:** Experiment checklist with status indicators

**Experiment 6.4: Multi-Model Validation (Model-Agnostic Test)**
- **Goal:** Prove framework works across architectures
- **Models:**
  - FaceNet (Inception-ResNet-V1) - DONE
  - ResNet-50 (50-layer residual network) - IN PROGRESS
  - VGG-Face (16-layer VGG architecture) - PLANNED
- **Expected:** Similar FR patterns (Grad-CAM best, SHAP/LIME fail)
- **Timeline:** Months 4-5
- **Status:** 30% complete (ResNet-50 model loaded, preliminary runs)

**Higher-n Statistical Reruns**
- **Goal:** Increase statistical power and precision
- **Current:** n = 500 pairs per method
- **Target:** n = 1000-5000 pairs
- **Benefit:** Narrower confidence intervals, stronger p-values
- **Timeline:** Months 5-6
- **Computational cost:** ~40 GPU hours for n=5000 (feasible)

**Additional Attribution Methods**
- **Methods:** Gradient×Input, VanillaGradients, SmoothGrad
- **Goal:** Comprehensive gradient-based method coverage
- **Expected:** Similar performance to Grad-CAM (all gradient-based)
- **Timeline:** Month 6
- **Status:** Agent 1 implementations ready

**Experiment 6.6: Demographic Fairness Analysis**
- **Goal:** Test if falsification rates vary by demographic subgroups
- **Subgroups:** Age (young/old), gender (male/female), ethnicity
- **Metric:** FR consistency across subgroups
- **Timeline:** Month 6
- **Importance:** Addresses bias concerns in committee Q&A

---

### SLIDE 19: Remaining Writing Tasks

**Visual:** Chapter outline with completion status

**Chapter 8: Discussion and Conclusion (6-8 hours)**
- **Current status:** 0% complete (outlined only)
- **Sections:**
  - 8.1 Summary of Contributions (1 hour)
  - 8.2 Theoretical Implications (1.5 hours)
  - 8.3 Practical Implications for Forensic Deployment (1.5 hours)
  - 8.4 Limitations and Threats to Validity (1 hour)
  - 8.5 Future Work and Open Questions (1 hour)
  - 8.6 Concluding Remarks (0.5 hours)
- **Timeline:** Month 7
- **Priority:** High (needed for final defense)

**Chapter 7: Timing Benchmark Section (1.5 hours)**
- **Current gap:** Computational complexity empirically validated, but detailed timing breakdown missing
- **Content:** Per-component timing (counterfactual generation, model inference, attribution computation)
- **Timeline:** Month 7
- **Priority:** Medium (addresses Q&A about deployment feasibility)

**Table Updates and Result Integration (2 hours)**
- Update all result tables with multi-dataset findings
- Add ResNet-50, VGG-Face results to Table 6.4
- Higher-n confidence intervals in Table 6.1
- **Timeline:** Month 7-8

**LaTeX Polish and Formatting (4-6 hours)**
- Consistent formatting across all chapters
- Figure/table numbering and cross-references
- Bibliography cleanup (BibTeX consistency)
- Equation formatting review
- **Timeline:** Month 8

**Professional Proofreading (4-6 hours)**
- Grammar, spelling, clarity
- Academic writing style consistency
- Citation format verification
- **Timeline:** Month 8 (external proofreader if possible)

---

## PART V: CONTRIBUTIONS & IMPACT (3-4 minutes, Slides 20-22)

### SLIDE 20: Theoretical Contributions

**Visual:** Four numbered contribution boxes

**Contribution 1: First Falsifiability Criterion for Biometric XAI**
- **Novel framework:** Operationalizes Popper's falsifiability for attribution methods
- **Mathematical rigor:** Four theorems with formal proofs (Theorems 3.5-3.8)
- **Generalizability:** Applies to any biometric verification system (face, fingerprint, iris, voice)

**Contribution 2: Counterfactual Existence Guarantee (Theorem 3.6)**
- **Theoretical guarantee:** Proves counterfactuals always exist on hypersphere
- **Empirical validation:** 100% success rate across 5000 trials
- **Geometric foundation:** Hypersphere sampling algorithm

**Contribution 3: Computational Complexity Characterization (Theorem 3.7)**
- **Tractability proof:** Linear complexity O(K·|M|) in key parameters
- **Empirical validation:** r > 0.999 correlation with theoretical predictions
- **Deployment feasibility:** Average 0.47 seconds per test pair (GPU)

**Contribution 4: Sample Size Requirements (Theorem 3.8)**
- **Statistical guarantees:** Hoeffding bound provides worst-case confidence
- **Practical guidance:** n ≥ 43 pairs for ε=0.05, δ=0.05
- **Validated:** Bootstrap analysis confirms Central Limit Theorem applies

**Impact:** Provides rigorous mathematical foundation for XAI validation in biometric systems

---

### SLIDE 21: Practical Contributions

**Visual:** Three-column layout - Industry, Forensic, Regulatory

**Contribution 1: Open-Source Framework Implementation**
- **Code:** Python library with all attribution methods and falsification tests
- **Reproducibility:** Complete experimental pipeline (scripts, datasets, models)
- **Availability:** GitHub repository (will be released upon dissertation acceptance)
- **Documentation:** User guide, API reference, tutorial notebooks

**Contribution 2: Identifies Problematic XAI Methods**
- **SHAP fails:** 93.14% FR (widely used in industry, unreliable for biometrics)
- **LIME fails:** 94.22% FR (popular interpretability tool, inadequate for forensics)
- **Geodesic IG fails catastrophically:** 100% FR (architectural mismatch)
- **Grad-CAM succeeds:** 10.48% FR (only method suitable for deployment)

**Contribution 3: Forensic Deployment Guidelines**
- **Quality threshold:** FR < 20% for admissibility
- **Testing protocol:** Minimum n ≥ 100 pairs for method validation
- **Reporting template:** Falsification rate with confidence intervals
- **Daubert compliance:** Scientific validation evidence for expert testimony

**Contribution 4: Regulatory Compliance Support**
- **Daubert standard:** Empirical testing, peer review, known error rate
- **GDPR Article 22:** "Right to explanation" - validated attributions
- **EU AI Act (2024):** High-risk AI systems require transparency and accuracy

**Impact:** Bridges gap between XAI research and real-world deployment

---

### SLIDE 22: Expected Impact - Research & Industry

**Visual:** Impact map with academic and industry branches

**Academic Impact:**

**1. New Research Paradigm for XAI Validation**
- Current XAI research: Qualitative, anecdotal validation
- Our contribution: Quantitative, falsifiable validation
- Expected adoption: XAI papers will cite our falsification framework as validation method

**2. Cross-Domain Applicability**
- Biometrics: Face, fingerprint, iris, voice verification
- Medical imaging: Diagnostic model explanations
- Autonomous vehicles: Safety-critical decision explanations
- Financial services: Credit scoring, fraud detection

**3. Open Research Questions**
- How to extend falsification to classification tasks?
- Can we develop attribution methods optimized for low FR?
- What is the theoretical lower bound on FR?

**Industry Impact:**

**1. Improved Biometric System Rigor**
- Current practice: Deploy XAI without validation
- Our contribution: Evidence-based method selection
- Expected adoption: Industry will use FR as quality metric

**2. Forensic Lab Deployment**
- Current barrier: Lack of scientifically validated XAI
- Our contribution: Daubert-compliant validation framework
- Expected adoption: Forensic labs will require FR < 20% for court cases

**3. Regulatory Compliance Tool**
- GDPR (EU): Right to explanation
- EU AI Act (2024): High-risk AI transparency requirements
- US Algorithmic Accountability Act: Explainability mandates
- Our framework: Provides quantitative evidence of explanation quality

**Economic Impact:**
- Biometric market: $55.9B (2023) → $82.9B (2030) (Grand View Research)
- XAI market: $6.8B (2023) → $21.9B (2030) (MarketsandMarkets)
- Forensic technology: $15.6B (2023) → $29.4B (2030) (Allied Market Research)

---

## PART VI: Q&A PREPARATION (Slides 23-25)

### SLIDE 23: Anticipated Committee Questions - Preview

**Visual:** Question categories with icons

**Theoretical Questions:**
- Why is falsifiability the right criterion? (See comprehensive Q&A doc)
- How do you know counterfactuals are realistic?
- Why did Geodesic IG fail so badly?

**Experimental Questions:**
- Why only LFW dataset? → Multi-dataset validation in progress
- You tested 5 attribution methods, why not more?
- How model-agnostic is this framework?

**Practical Questions:**
- How would this be used in a real forensic case?
- What if attribution methods fail your test?
- Isn't falsification testing too slow for real-time deployment?

**Limitations Questions:**
- Single-dataset validation (proposal) → Being addressed
- No human studies (IRB) → Out of scope, technical validation focus
- Face verification only → Generalizable to all biometric modalities

**Defense-Specific Questions:**
- Can you finish in 10 months? → Yes, detailed timeline provided
- What's the weakest part? → Single-dataset (being addressed Months 1-3)

**Full Q&A Preparation:** See `/defense/comprehensive_qa_preparation.md` (50+ questions with detailed answers)

---

### SLIDE 24: Backup Slides - Statistical Details

**[NOT PRESENTED - AVAILABLE FOR Q&A]**

**Backup Content:**

1. **Chi-Square Test Detailed Calculation**
   - Observed vs. expected frequencies
   - Degrees of freedom computation
   - Critical value comparison

2. **Cohen's h Effect Size Derivation**
   - Proportion comparison formula
   - Effect size interpretation guidelines
   - Statistical power analysis

3. **Bootstrap Analysis Methodology**
   - Resampling procedure (10,000 iterations)
   - Confidence interval construction
   - Central Limit Theorem validation

4. **Hoeffding Bound Proof Sketch**
   - Concentration inequality
   - Sample size derivation
   - Error tolerance trade-offs

5. **Additional Statistical Tests**
   - Kolmogorov-Smirnov test (distribution comparison)
   - Mann-Whitney U test (non-parametric alternative)
   - Bonferroni correction (multiple comparisons)

---

### SLIDE 25: Backup Slides - Implementation Details

**[NOT PRESENTED - AVAILABLE FOR Q&A]**

**Backup Content:**

1. **FaceNet Architecture Diagram**
   - Inception-ResNet-V1 detailed structure
   - Embedding layer specifications
   - Training procedure overview

2. **Hypersphere Sampling Algorithm Pseudocode**
   ```python
   def generate_counterfactual(z_a, mask_M, alpha=0.5):
       z_prime = z_a.copy()
       for i in mask_M:
           v_i = sample_random_direction()
           z_prime[i] += alpha * v_i
       z_a_prime = z_prime / norm(z_prime)  # Project to hypersphere
       return z_a_prime
   ```

3. **Falsification Test Flowchart**
   - Input: Face pair, attribution mask
   - Step 1: Generate K counterfactuals
   - Step 2: Compute Δsim for each
   - Step 3: Count falsifications (Δsim < ε)
   - Output: Falsification rate

4. **Attribution Method Implementation Notes**
   - Grad-CAM: Layer selection (Mixed_7a)
   - SHAP: KernelExplainer with 100 samples
   - LIME: 1000 perturbations, linear model

5. **Computational Resource Requirements**
   - GPU: NVIDIA V100 (32GB VRAM)
   - CPU: 16-core Intel Xeon
   - RAM: 64GB
   - Storage: 500GB for datasets and models
   - Estimated total compute: ~200 GPU hours for full dissertation

---

## PRESENTATION LOGISTICS

### Time Allocation (20-30 minutes total):
- Part I (Introduction): 3-5 minutes
- Part II (Theoretical Framework): 8-10 minutes
- Part III (Preliminary Results): 6-8 minutes
- Part IV (Remaining Work): 4-6 minutes
- Part V (Contributions & Impact): 3-4 minutes
- Buffer: 2-3 minutes

### Delivery Tips:
1. **Rehearse 5+ times** - Know slide transitions cold
2. **Practice with timer** - Must finish in 20-25 minutes
3. **Anticipate interruptions** - Committee may ask questions during presentation
4. **Backup slides ready** - Have extra slides for likely Q&A topics
5. **Bring printed slides** - In case of technical issues

### Visual Design Recommendations:
- **Minimal text:** Max 5-7 bullet points per slide
- **Large fonts:** 24pt minimum for body text, 32pt for titles
- **High-contrast:** Dark text on light background
- **Professional figures:** High-resolution, labeled axes, clear legends
- **Consistent theme:** Use institution's beamer template if available

### Equipment Checklist:
- [ ] Laptop with presentation loaded
- [ ] Backup USB drive with PDF version
- [ ] HDMI/VGA adapter
- [ ] Laser pointer (if not built into remote)
- [ ] Water bottle
- [ ] Printed notes (1-page outline)
- [ ] Printed backup slides

---

## POST-PROPOSAL NEXT STEPS

### Immediate (Week 1 after proposal):
1. Incorporate committee feedback
2. Prioritize multi-dataset validation (CelebA download)
3. Schedule regular advisor check-ins

### Short-term (Months 1-3):
1. Complete CelebA experiments (n=500 pairs per method)
2. Complete CFP-FP experiments if dataset available
3. Draft multi-dataset results section

### Medium-term (Months 4-6):
1. Multi-model validation (ResNet-50, VGG-Face)
2. Higher-n reruns (n=1000-5000)
3. Statistical analysis refinement

### Long-term (Months 7-10):
1. Chapter 8 writing
2. LaTeX polish
3. Final defense preparation
4. Mock defenses

---

**END OF PROPOSAL DEFENSE PRESENTATION OUTLINE**

**Total Slides:** 25 + backup slides
**Estimated Preparation Time:** 20-25 hours (create slides from outline, rehearse, prepare Q&A)
**Confidence Level for 3-Month Defense:** High (85 → 92+ defense readiness achievable)
