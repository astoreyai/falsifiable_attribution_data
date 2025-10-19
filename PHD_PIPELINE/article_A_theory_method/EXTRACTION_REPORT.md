# Article A Extraction Report: Theory/Method Manuscript

**Agent:** Agent 1 - Article A Extraction Specialist

**Date:** October 15, 2025

**Task:** Extract and prepare Article A (Theory/Method) manuscript from dissertation chapters

---

## ✅ DELIVERABLES COMPLETED

### 1. Main Manuscript Draft (Sections 1-4)

**File:** `article_A_theory_method/manuscript/article_A_draft_sections_1-4.md`

**Line Count:** 650 lines

**Content Extracted:**

| Section | Source Chapter(s) | Word Count | Page Estimate | Status |
|---------|-------------------|------------|---------------|--------|
| **Section 1: Introduction** | Chapter 1 (1.1-1.2) | ~450 | 1.5 pages | ✅ Complete |
| **Section 2: Background** | Chapter 2 (2.2-2.3) | ~600 | 2.0 pages | ✅ Complete |
| **Section 3: Theory** | Chapter 3 (3.1-3.3) | ~900 | 3.0 pages | ✅ Complete |
| **Section 4: Method** | Chapter 4 (4.3-4.4) | ~600 | 2.0 pages | ✅ Complete |
| **Section 5: Experiments** | Chapter 6 (pending) | N/A | PLACEHOLDER | ⏳ Awaiting experiments |
| **Section 6: Discussion** | Chapter 7 (pending) | N/A | PLACEHOLDER | ⏳ To be written |
| **TOTAL (Sections 1-4)** | | **~2,550 words** | **~6.5 pages** | ✅ Complete |

**Key Condensations Applied:**

1. **Introduction (1.5 pages):**
   - Extracted motivation from Chapter 1.1 (wrongful arrests, XAI opacity problem)
   - Condensed contribution summary from Chapter 1.4 to 3 main points
   - Removed detailed literature review (deferred to Section 2)
   - Focused on falsifiability gap and forensic urgency

2. **Background (2 pages):**
   - Extracted ArcFace/CosFace technical details from Chapter 2.2
   - Condensed Grad-CAM, IG, SHAP explanations from Chapter 2.3
   - Removed full proofs and historical context
   - Kept mathematical formulations for reproducibility

3. **Theory (3 pages):**
   - Promoted Theorem 3.5 (Falsifiability Criterion) to **BOXED THEOREM**
   - Extracted proof sketch (sufficiency + necessity, condensed from 4 pages to 1 page)
   - Removed Corollaries 3.1-3.4 (can add to appendix if needed)
   - Kept geometric intuition and Popper connection

4. **Method (2 pages):**
   - Condensed Algorithm 3.1 from full PyTorch implementation to pseudocode
   - Extracted counterfactual generation procedure from Chapter 4.4
   - Stated Theorem 3.6 (Existence) without full proof (proof → appendix)
   - Included Theorem 3.7 (Complexity) analysis

**Placeholders Added:**

- Section 5: `[TO BE ADDED AFTER EXPERIMENTS RUN]` - expects empirical validation results
- Section 6: `[TO BE WRITTEN AFTER RESULTS]` - discussion and deployment guidelines

---

### 2. Theorem Box

**File:** `article_A_theory_method/manuscript/theorem_box.md`

**Line Count:** 210 lines

**Content:**

- **Formal statement** of Theorem 1 (Falsifiability Criterion) with all 3 conditions
- **Testable predictions** (high-attribution vs low-attribution)
- **Geometric intuition** (unit hypersphere visualization)
- **Connection to Popper's falsifiability** (scientific demarcation)
- **Typical parameter values** (based on ArcFace verification thresholds)

**Visual Design:** Formatted for prominent boxed display in journal article (LaTeX framed environment)

---

### 3. Assumptions Box

**File:** `article_A_theory_method/manuscript/assumptions_box.md`

**Line Count:** 280 lines

**Content:**

**5 Formal Assumptions Documented:**

1. **Unit-Norm Embeddings (Hypersphere Geometry)**
   - Statement: $\|\phi(x)\|_2 = 1$
   - Satisfied by: ArcFace, CosFace, SphereFace
   - NOT satisfied by: FaceNet (unnormalized), VGGFace (Euclidean)

2. **Geodesic Metric on Hypersphere**
   - Statement: $d_g(u,v) = \arccos(\langle u, v \rangle)$
   - Equivalence to cosine similarity explained
   - Decision rule examples provided

3. **Plausibility Constraints**
   - Statement: $\|x' - x\|_2 < \epsilon_{\text{pixel}}$
   - Justification: Avoid distribution shift
   - Implementation: Feature masking, gradient clipping, regularization

4. **Scope - Verification (1:1) Only**
   - NOT applicable to: Identification (1:N), classification, other modalities
   - Adaptation required for extensions

5. **Differentiability (Gradient Access)**
   - Required for: Algorithm 1, IG, Grad-CAM
   - NOT required for: SHAP/LIME (model-agnostic)
   - Limitation: Commercial APIs lack gradient access

**Summary Table:** Quick reference for assumption checking

**Violation Handling:** Guidance for when assumptions don't hold

---

### 4. Figures Specification

**File:** `article_A_theory_method/manuscript/figures_needed.md`

**Line Count:** 370 lines

**5 Figures Specified:**

#### Figure 1: Comparison Table (Ready to Create)
- **Type:** Table comparing plausibility vs faithfulness vs falsifiability
- **Purpose:** Distinguish our approach from prior evaluation methods
- **Placement:** Section 1 or 3
- **Status:** Specification complete, awaiting LaTeX table creation

#### Figure 2: Geometric Interpretation (Ready to Create)
- **Type:** 3D diagram with panels (hypersphere + counterfactuals)
- **Purpose:** Visualize geodesic distance and falsification test
- **Content:**
  - Panel A: Unit hypersphere geometry (geodesic vs Euclidean)
  - Panel B: High vs low attribution perturbations
  - Panel C: Falsification test verdict
- **Placement:** Section 3 (after Theorem 1)
- **Status:** Specification complete, awaiting TikZ/Inkscape creation
- **Priority:** **CRITICAL** (core theorem visualization)

#### Figure 3: Algorithm Flowchart (Ready to Create)
- **Type:** Flowchart of Algorithm 1
- **Purpose:** Illustrate counterfactual generation pipeline
- **Content:** Full flowchart from input → iteration → convergence
- **Placement:** Section 4 (Method)
- **Status:** Specification complete, awaiting flowchart creation

#### Figure 4: Δ-Prediction Scatter Plot (PLACEHOLDER)
- **Type:** Scatter plot ($\bar{d}_{\text{low}}$ vs $\bar{d}_{\text{high}}$)
- **Purpose:** Validate differential prediction (Theorem 1, Condition 2)
- **Data Needed:** 1,000 LFW images × 4 attribution methods
- **Expected Results:** SHAP 75% in green zone, Grad-CAM 60%, LIME 50%
- **Status:** Awaiting experiments (Chapter 5 implementation)

#### Figure 5: Plausibility-Convergence Trade-off (PLACEHOLDER)
- **Type:** Multi-panel (histogram + scatter + image grid)
- **Purpose:** Show LPIPS vs convergence rate analysis
- **Data Needed:** 5,000 counterfactual samples with varying λ
- **Status:** Awaiting experiments

**Production Notes:**
- Figures 1-3: Immediate creation using LaTeX/TikZ/Matplotlib
- Figures 4-5: Post-experiment (need empirical data)
- Accessibility: Color-blind safe, alt-text, high-contrast

---

## KEY CONTENT EXTRACTED FROM EACH CHAPTER

### From Chapter 1 (Introduction)

**Extracted:**
- Motivation: Wrongful arrests (Hill2020, Hill2023, Parks2019)
- Problem statement: Lack of falsifiability in current XAI
- Daubert standard requirement (FRE702, NRC2009)
- 3 main contributions (condensed from 8 in dissertation)

**Omitted:**
- Detailed background on face verification systems (moved to Section 2)
- Full research questions (condensed to contributions)
- Extensive legal analysis (brief mention only)
- Scope and limitations (condensed to Assumptions box)

**Compression:** 12,000 words → 450 words (~3.75% retention, 26x compression)

---

### From Chapter 2 (Literature Review)

**Extracted:**
- ArcFace/CosFace technical details (loss functions, embeddings)
- Grad-CAM, IG, SHAP mathematical formulations
- Insertion-deletion metrics limitations
- Counterfactual explanations overview

**Omitted:**
- xCos (Lin2021) detailed analysis (verification-specific, not core to falsifiability)
- Full Shapley value derivation (cited Lundberg2017 instead)
- Dataset descriptions (VGGFace2, LFW - deferred to experiments)
- Sanity checks (Adebayo2018) - brief mention only
- Zhou2022 framework (mentioned in related work, not fully explained)

**Compression:** 28,991 words (full chapter) → 600 words (~2% retention, 48x compression)

---

### From Chapter 3 (Theory)

**Extracted:**
- **Theorem 3.5 (Falsifiability Criterion)** - FULL STATEMENT + PROOF SKETCH
- Definitions 3.1-3.6 (condensed to theorem prerequisites)
- Geometric intuition (unit hypersphere, geodesic distance)
- Popper connection (falsifiability as scientific demarcation)

**Omitted:**
- Full proofs of Background Theorems 3.1-3.4 (Shapley, ArcFace, Geodesic Metric, Hoeffding) - cited instead
- Corollaries 3.1-3.4 (Falsifiable ≠ Correct, Non-Falsifiable, Method-Agnostic, Daubert) - can add to appendix
- Detailed mathematical notation table (condensed to inline definitions)
- Grad-CAM falsification example (3.3.4) - too detailed for journal space

**Kept:**
- Complete Theorem 3.5 statement (necessary + sufficient conditions)
- Proof sketch showing both directions (sufficiency + necessity)
- Connection to Popper's philosophy of science

**Compression:** 13,550 words → 900 words (~6.6% retention, 15x compression)

---

### From Chapter 4 (Methodology)

**Extracted:**
- **Algorithm 3.1 (Counterfactual Generation)** - PSEUDOCODE VERSION
- Feature masking (Grad-CAM spatial maps vs SHAP superpixels)
- Loss function: geodesic distance error + proximity penalty
- **Theorem 3.6 (Existence)** - STATEMENT ONLY
- **Theorem 3.7 (Complexity)** - O(K·T·D) analysis

**Omitted:**
- Full PyTorch implementation (condensed to pseudocode)
- Falsification testing protocol (Section 4.3) - too detailed
- Experimental design (Section 4.6) - deferred to Section 5 placeholder
- Computational optimizations (GPU parallelization, early stopping) - brief mention
- Reproducibility protocol (Section 4.9) - not relevant for journal article

**Kept:**
- Algorithm 1 core logic (initialization → iteration → convergence)
- Binary masking procedure
- Hyperparameter values (α=0.01, λ=0.1, T=100, ε_tol=0.01)
- Complexity analysis with practical runtime (~4 sec/image)

**Compression:** 9,300 words → 600 words (~6.5% retention, 15.5x compression)

---

## GAPS IDENTIFIED

### 1. Missing Experimental Validation (Section 5)

**Status:** PLACEHOLDER - Awaiting experiments

**Required Content:**
- Falsification rates for 4 attribution methods (Grad-CAM, SHAP, LIME, IG) on 1,000 LFW images
- Separation margin analysis ($\Delta = \bar{d}_{\text{high}} - \bar{d}_{\text{low}}$)
- Attribute validation (CelebA: glasses, beards)
- Model-agnostic comparison (ArcFace vs CosFace)
- Convergence analysis (Algorithm 1 success rates)

**Hypotheses to Test:**
- H1: Not all methods are falsifiable (some fail Theorem 1 conditions)
- H2: Lower falsification rate → higher faithfulness
- H3: Separation margin Δ > 0.15 radians required for reliable testing

**Data Sources:**
- LFW dataset (1,000 images)
- CelebA dataset (attribute annotations)
- Pretrained ArcFace-ResNet100 and CosFace-ResNet64 models

**Estimated Timeline:** 2-3 weeks (implementation + experiments + analysis)

---

### 2. Missing Discussion & Deployment Guidelines (Section 6)

**Status:** PLACEHOLDER - To be written after results

**Required Content:**
- Interpretation of empirical findings
- Deployment thresholds for forensic contexts (when is faithfulness "good enough"?)
- Comparison to Daubert standard requirements
- Limitations and generalization
- Future work directions

**Dependencies:**
- Section 5 results (need to see which methods are falsifiable)
- Forensic case analysis (which thresholds prevent wrongful arrests?)

**Estimated Timeline:** 1 week after Section 5 completion

---

### 3. Proofs in Appendix

**Missing (to add to appendix):**
- Full proof of Theorem 3.6 (Existence of Counterfactuals) via Intermediate Value Theorem
- Convergence analysis of Algorithm 1 (non-convex optimization, empirical validation)
- Corollaries 3.1-3.4 from Chapter 3
- Detailed Hoeffding bound derivation for sample size K=200

**Rationale:** Main paper space limits; proofs provide rigor but can be deferred to appendix

---

### 4. Figures Awaiting Creation

**Immediate (Ready to Create):**
- Figure 1: Comparison table (plausibility vs faithfulness vs falsifiability) - 1 hour
- Figure 2: Geometric interpretation (hypersphere + counterfactuals) - 3-4 hours (TikZ)
- Figure 3: Algorithm flowchart - 2 hours

**Post-Experiment (Need Data):**
- Figure 4: Δ-prediction scatter plot - awaiting LFW experiments
- Figure 5: Plausibility-convergence trade-off - awaiting counterfactual samples

---

## RECOMMENDATIONS FOR NEXT STEPS

### Immediate Actions (Week 1)

1. **Create Figures 1-3** (estimated 6-7 hours)
   - Figure 1: LaTeX table (comparison paradigms)
   - Figure 2: TikZ/Inkscape diagram (hypersphere geometry) - **CRITICAL**
   - Figure 3: Flowchart (Algorithm 1)

2. **Write Appendix** (estimated 8-10 hours)
   - Full proof of Theorem 3.6
   - Corollaries 3.1-3.4
   - Detailed algorithm analysis
   - Extended notation table

3. **Bibliography Cleanup**
   - Extract 25-30 core references from dissertation bibliography
   - Format for TPAMI/IJCV submission style
   - Add any missing citations from condensation

---

### Experimental Phase (Weeks 2-4)

4. **Implement Experiments (Section 5)** - Agent 2's task
   - Falsification testing protocol (Chapter 4.3 implementation)
   - LFW evaluation (1,000 images × 4 methods)
   - Convergence analysis (5,000 counterfactuals)
   - Generate data for Figures 4-5

5. **Write Section 5: Experiments** (estimated 12-15 hours after data collection)
   - Results tables (falsification rates, separation margins)
   - Statistical analysis (t-tests, confidence intervals)
   - Figure 4-5 generation from experimental data

---

### Finalization (Week 5)

6. **Write Section 6: Discussion** (estimated 8-10 hours)
   - Interpret results in context of Theorem 1
   - Deployment thresholds for forensic use
   - Limitations and scope boundaries
   - Future work

7. **Integration & Review** (estimated 6-8 hours)
   - Merge all sections into single manuscript
   - Check cross-references (figures, theorems, equations)
   - Proofread for clarity and flow
   - Format for TPAMI/IJCV submission

---

## QUALITY METRICS

### Scientific Rigor

✅ **Honest Claims:**
- No aspirational statements ("will enable", "could be used")
- All claims grounded in theory or marked as PLACEHOLDER
- Limitations explicitly stated (Assumptions box)

✅ **Citations:**
- 25 references extracted from dissertation bibliography
- All mathematical results cited (Hoeffding, Shapley, ArcFace, CosFace)
- Prior XAI work properly attributed (Grad-CAM, IG, SHAP, Zhou2022)

✅ **Reproducibility:**
- Algorithm 1 pseudocode provided
- Hyperparameters specified (α, λ, T, ε_tol)
- Datasets identified (LFW, CelebA)
- Code availability statement (to be added: GitHub link)

---

### Compression Efficiency

**Overall Compression:**
- Dissertation Chapters 1-4: ~63,841 words
- Article Sections 1-4: ~2,550 words
- **Compression Ratio:** 25:1 (4% retention)

**Target vs Actual:**
- Target: 6.5 pages (~2,600 words at 300 words/page)
- Actual: ~2,550 words
- **Achievement:** 98% of target ✅

---

### Content Preservation

**Core Theoretical Contributions Retained:**
- ✅ Theorem 1 (Falsifiability Criterion) - FULL statement + proof sketch
- ✅ Algorithm 1 (Counterfactual Generation) - Pseudocode version
- ✅ Theorem 2 (Existence) - Statement (proof → appendix)
- ✅ Theorem 3 (Complexity) - O(K·T·D) analysis

**Mathematical Rigor:**
- ✅ All formal definitions preserved (hypersphere, geodesic, counterfactual)
- ✅ Assumptions explicitly stated (5 assumptions documented)
- ✅ Scope clearly bounded (verification 1:1 only, no identification)

---

## AGENT 1 TASK COMPLETION SUMMARY

### Deliverables Created: 4/4 ✅

1. ✅ `article_A_draft_sections_1-4.md` (650 lines, ~2,550 words)
2. ✅ `theorem_box.md` (210 lines, boxed Theorem 1 for prominent display)
3. ✅ `assumptions_box.md` (280 lines, 5 formal assumptions documented)
4. ✅ `figures_needed.md` (370 lines, 5 figures specified with detailed descriptions)

### Total Output: 1,510 lines of structured manuscript content

---

### Content Extraction Quality

**From Chapter 1 (Introduction):**
- ✅ Motivation extracted (wrongful arrests, Daubert standard)
- ✅ Contributions condensed (3 main points from 8)
- Compression: 12,000 → 450 words (26x)

**From Chapter 2 (Literature Review):**
- ✅ ArcFace/CosFace technical details extracted
- ✅ XAI methods (Grad-CAM, IG, SHAP) condensed
- Compression: ~28,991 → 600 words (48x)

**From Chapter 3 (Theory):**
- ✅ **Theorem 3.5 PROMOTED to boxed theorem**
- ✅ Proof sketch condensed (4 pages → 1 page)
- ✅ Geometric intuition preserved
- Compression: 13,550 → 900 words (15x)

**From Chapter 4 (Methodology):**
- ✅ Algorithm 1 condensed to pseudocode
- ✅ Feature masking procedure explained
- ✅ Complexity analysis retained
- Compression: 9,300 → 600 words (15.5x)

---

### Gaps Identified & Documented: 4 Major Gaps

1. ⏳ **Section 5: Experiments** - PLACEHOLDER (awaiting data from experiments)
2. ⏳ **Section 6: Discussion** - PLACEHOLDER (to be written after results)
3. ⏳ **Figures 4-5** - PLACEHOLDER (need experimental data)
4. 📝 **Appendix** - Proofs and corollaries (to be written)

---

### Recommendations Provided

**Immediate (Week 1):**
- Create Figures 1-3 (LaTeX table, TikZ diagram, flowchart)
- Write appendix (full proofs, corollaries)
- Clean bibliography (25-30 references)

**Experimental Phase (Weeks 2-4):**
- Implement experiments (Agent 2's task)
- Generate Figures 4-5 data
- Write Section 5

**Finalization (Week 5):**
- Write Section 6 (discussion)
- Integration & review
- Format for submission

---

## FILES CREATED

All files in: `/home/aaron/projects/xai/PHD_PIPELINE/article_A_theory_method/manuscript/`

1. `article_A_draft_sections_1-4.md` - Main manuscript draft (Sections 1-4)
2. `theorem_box.md` - Falsifiability Criterion for prominent display
3. `assumptions_box.md` - 5 formal assumptions documented
4. `figures_needed.md` - 5 figures specified with detailed descriptions

**Total Lines:** 1,510 lines of structured content

---

## NEXT AGENT HANDOFF

**To:** Agent 2 (Experimental Validation Specialist)

**Task:** Implement experiments and generate Section 5 content

**Required:**
- Falsification testing protocol (Chapter 4.3 implementation in code)
- LFW evaluation (1,000 images × 4 attribution methods)
- Generate data for Figures 4-5
- Write Section 5: Experiments (~1,500 words)

**Dependencies:**
- Pretrained ArcFace/CosFace models
- LFW and CelebA datasets
- PyTorch implementation of Algorithm 1

**Deliverables Expected:**
- `section_5_experiments.md` (experimental results)
- `figure_4_data.csv` (Δ-prediction scatter plot data)
- `figure_5_data.csv` (plausibility-convergence data)
- Statistical analysis (falsification rates, separation margins, p-values)

---

**END OF EXTRACTION REPORT**

**Agent 1 Status:** ✅ TASK COMPLETE

**Estimated Time:** 6-8 hours of extraction, condensation, and specification work
