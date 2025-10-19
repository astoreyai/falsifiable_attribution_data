# Article A: Falsifiable Attribution for Face Verification via Counterfactual Score Prediction

**Target Venues:** IJCV, IEEE TPAMI
**Article Type:** Theory + Method + Demonstration
**Timeline:** Weeks 3–5 (with experiments in weeks 6–8)
**Status:** NOT STARTED

---

## OBJECTIVE

Extract and package the **theoretical contribution** and **method design** from the dissertation into a focused journal article demonstrating that attribution explanations can be falsified through counterfactual score prediction on the unit hypersphere.

---

## WHAT'S ALREADY IN THE DISSERTATION

From `falsifiable_attribution_dissertation/chapters/`:

- **Theory of falsifiability** with formal criteria, conditions, and corollaries (chapter_03_theory_COMPLETE.md)
- **ArcFace/CosFace geometry** with geodesic interpretation and unit hypersphere definitions
- **Worked decision examples** showing NOT FALSIFIED vs FALSIFIED outcomes
- **Counterfactual generation pipeline** with algorithm pseudocode
- **Method-agnostic scope** applicable to any gradient-based attribution method

---

## ARTICLE STRUCTURE (10–12 pages)

### 1. Introduction (1.5 pages)
**Extract from:** dissertation chapter_01_introduction.md

**Content:**
- Problem: XAI methods lack falsifiability → no scientific grounding
- Gap: Prior work evaluates plausibility/faithfulness but not testable predictions
- Contribution: First falsifiable criterion for attributions via Δ-score prediction
- Scope: Face verification (1:1) with ArcFace/CosFace on unit hypersphere

**New work needed:**
- [ ] Tighten to 1.5 pages
- [ ] Add ONE motivating example (real vs non-falsifiable explanation)
- [ ] Forward-reference to theorem box

### 2. Background & Related Work (2 pages)
**Extract from:** dissertation chapter_02_literature_review.md

**Content:**
- Brief overview of attribution methods (Grad-CAM, IG, SHAP)
- Prior evaluation approaches (plausibility, faithfulness, robustness)
- Why they lack testable predictions
- Face verification geometry (unit hypersphere, cosine similarity, geodesics)

**New work needed:**
- [ ] Condense to 2 pages
- [ ] Add 3-column comparison table: plausibility vs faithfulness vs falsifiability
- [ ] Mark gap: "no prior work tests counterfactual score-prediction accuracy"

### 3. Theory: Falsifiability Criterion (3 pages)
**Extract from:** dissertation chapter_03_theory_COMPLETE.md

**Content:**
- **Boxed "Executive Definition + Theorem"** stating the criterion
- Two testable predictions:
  1. High-attribution features → larger score delta when perturbed
  2. Low-attribution features → smaller score delta when perturbed
- Geometric intuition on unit hypersphere (geodesic movement)
- Popper/Daubert alignment (1 paragraph)
- Formal properties (uniqueness, method-agnostic applicability)

**New work needed:**
- [ ] Promote criterion to boxed theorem at start of section
- [ ] Add ONE geometric figure showing high vs low attribution feature impact
- [ ] Keep proofs compact (move long proofs to appendix if needed)
- [ ] Add "Assumptions & Validity" box: unit-norm embeddings, geodesic metric, plausibility constraints

### 4. Method: Counterfactual Generation on the Hypersphere (2 pages)
**Extract from:** dissertation chapter_04_methodology_COMPLETE.md

**Content:**
- Algorithm sketch: perturb → project → verify Δ-prediction
- Plausibility constraints (LPIPS/FID thresholds, rule-based exclusions)
- Computational properties (batch processing, iteration guidance)
- Failure modes and when test is inconclusive

**New work needed:**
- [ ] Condense to 2 pages
- [ ] Add method flowchart: data → attribution → counterfactual gate → Δ-test → decision
- [ ] Specify computational complexity (O-notation)

### 5. Experimental Validation (2.5 pages)
**Extract from:** dissertation experiments (to be run in weeks 6–8)

**Content:**
- Datasets: LFW or CASIA-WebFace (reproducible, public)
- Models: ArcFace backbone (ResNet-50 or similar)
- Attribution methods tested: Grad-CAM, IG, SHAP
- Primary metric: Correlation between predicted and realized score-deltas
- Results: Table showing which methods pass/fail falsification test
- 2-3 clean figures showing Δ-prediction accuracy

**New work needed:**
- [ ] Run minimal but decisive experiments (2–3 attribution methods)
- [ ] Produce ONE scatter plot: predicted vs realized Δ-score
- [ ] Produce ONE table: method → correlation → pass/fail threshold
- [ ] Add ONE figure showing accepted vs rejected counterfactuals (plausibility gate)

### 6. Discussion & Conclusion (1 page)
**Content:**
- What the criterion enables (scientific testability for XAI)
- Limitations: scope (verification only), datasets, architectures
- Future work: extension to identification (1:N), other domains
- Call to action: adopt falsifiability as standard

**New work needed:**
- [ ] Write fresh (1 page)
- [ ] Tie back to theorem box
- [ ] Acknowledge limits without over-claiming

---

## EXTRACTION WORKFLOW

### Phase 1: Content Assembly (Week 3)

**Step 1.1:** Create manuscript skeleton
```bash
cd /home/aaron/projects/xai/PHD_PIPELINE/article_A_theory_method/manuscript
touch article_A_draft.md
```

**Step 1.2:** Extract introduction
- [ ] Copy relevant sections from `../falsifiable_attribution_dissertation/chapters/chapter_01_introduction.md`
- [ ] Trim to 1.5 pages
- [ ] Add motivating example
- [ ] Forward-reference theorem

**Step 1.3:** Extract background
- [ ] Copy relevant sections from `chapter_02_literature_review.md`
- [ ] Condense to 2 pages
- [ ] Create 3-column comparison table (use template below)

**Step 1.4:** Extract theory
- [ ] Copy falsifiability criterion from `chapter_03_theory_COMPLETE.md`
- [ ] Create boxed theorem
- [ ] Add assumptions box
- [ ] Trim proofs (move long ones to appendix)

**Step 1.5:** Extract method
- [ ] Copy counterfactual generation algorithm from `chapter_04_methodology_COMPLETE.md`
- [ ] Condense to 2 pages
- [ ] Add method flowchart

### Phase 2: Figures & Tables (Week 4)

**Step 2.1:** Create comparison table
```markdown
| Evaluation Approach | Tests What | Testable Prediction | Prior Work |
|---------------------|-----------|-------------------|------------|
| Plausibility | Human-alignment | None (subjective) | [refs] |
| Faithfulness | Model-alignment | None (correlation) | [refs] |
| **Falsifiability** | **Δ-score prediction** | **Counterfactual score change** | **This work** |
```

**Step 2.2:** Create geometric figure
- [ ] Illustrate unit hypersphere with two embeddings
- [ ] Show geodesic path
- [ ] Mark high-attribution vs low-attribution feature perturbations
- [ ] Show predicted vs realized Δ-score

**Step 2.3:** Create method flowchart
- [ ] Data → Attribution → Plausibility Gate → Δ-Test → Decision
- [ ] Use existing diagram from dissertation if available, else create new

### Phase 3: Experiments (Weeks 6–8)

**Step 3.1:** Set up minimal experimental environment
- [ ] Select dataset: LFW (public, reproducible)
- [ ] Load pretrained ArcFace model
- [ ] Implement 2–3 attribution methods (Grad-CAM, IG, SHAP)

**Step 3.2:** Run Δ-prediction test
- [ ] Generate counterfactuals for 100–200 image pairs
- [ ] Apply plausibility gate
- [ ] Measure predicted vs realized score deltas
- [ ] Compute correlation coefficient

**Step 3.3:** Create results visualizations
- [ ] Scatter plot: predicted vs realized Δ-score
- [ ] Table: method → correlation → pass/fail
- [ ] Figure: accepted vs rejected counterfactuals

**Step 3.4:** Write experimental section
- [ ] Describe setup (dataset, model, methods)
- [ ] Report results (table + figures)
- [ ] Interpret findings (which methods pass/fail)

### Phase 4: Polish & Review (Week 5 + Week 9)

**Step 4.1:** Internal consistency check
- [ ] Theorem box matches experimental claims
- [ ] Figures referenced in text
- [ ] Assumptions stated early and respected throughout
- [ ] No scope creep (stay in verification, unit hypersphere)

**Step 4.2:** Writing polish
- [ ] Abstract (150–200 words)
- [ ] Keywords (5–7)
- [ ] Section transitions
- [ ] Citation formatting (check venue requirements)

**Step 4.3:** Artifacts preparation
- [ ] Open implementation of falsification test harness
- [ ] Toy counterfactual set
- [ ] README for code release
- [ ] License file

**Step 4.4:** Pre-submission checklist
- [ ] Length: 10–12 pages (check venue limits)
- [ ] Figures: high resolution (300 DPI minimum)
- [ ] References: complete and formatted
- [ ] Author affiliations and contact
- [ ] Acknowledgments (funding, compute resources)

---

## KEY DESIGN DECISIONS

### What to INCLUDE
✅ Formal falsifiability criterion (theorem box)
✅ Geometric interpretation (unit hypersphere, geodesics)
✅ Counterfactual generation algorithm
✅ Minimal decisive experiments (2–3 methods)
✅ Method-agnostic scope

### What to EXCLUDE (save for other articles)
❌ Detailed regulatory discussion (Article C)
❌ Forensic reporting templates (Article B)
❌ Deployment thresholds (Article B)
❌ Policy implications (Article C)
❌ Over-claiming beyond verification

---

## CONTENT MAPPING: DISSERTATION → ARTICLE A

| Article Section | Dissertation Source | Transformation Needed |
|----------------|---------------------|----------------------|
| Introduction | chapter_01 sections 1.1–1.3 | Trim to 1.5 pages, add example |
| Background | chapter_02 sections 2.1–2.3 | Condense to 2 pages, add table |
| Theory | chapter_03 (all) | Promote theorem, add assumptions box |
| Method | chapter_04 sections 4.1–4.2 | Condense to 2 pages, add flowchart |
| Experiments | NEW (weeks 6–8) | Run minimal tests, create figs/tables |
| Discussion | NEW (week 5) | Write fresh, 1 page |

---

## PROGRESS TRACKING

Use `TodoWrite` tool to track:
- [ ] Content extraction completed
- [ ] Figures created (3 total: comparison table, geometric figure, method flowchart)
- [ ] Experiments run and results visualized
- [ ] Draft complete (10–12 pages)
- [ ] Internal review passed
- [ ] Ready for submission

---

## DELIVERABLES CHECKLIST

### Manuscript Files
- [ ] `manuscript/article_A_draft.md` (or .tex)
- [ ] `manuscript/abstract.txt`
- [ ] `manuscript/keywords.txt`

### Figures (high-res)
- [ ] `figures/fig1_comparison_table.pdf`
- [ ] `figures/fig2_geometric_interpretation.pdf`
- [ ] `figures/fig3_method_flowchart.pdf`
- [ ] `figures/fig4_delta_prediction_scatter.pdf`
- [ ] `figures/fig5_plausibility_gate.pdf`

### Tables
- [ ] `tables/table1_results_summary.csv`

### Code & Data
- [ ] `code/falsification_test.py`
- [ ] `code/README.md`
- [ ] `code/requirements.txt`
- [ ] `code/LICENSE`

### Bibliography
- [ ] `bibliography/article_A_refs.bib`

### Submission Materials
- [ ] `submission/cover_letter.md`
- [ ] `submission/author_contributions.md`
- [ ] `submission/competing_interests.md`

---

## TIMELINE (8–10 WEEKS)

| Week | Activity | Deliverable |
|------|----------|------------|
| 3 | Extract introduction, background, theory | Draft sections 1–3 |
| 4 | Extract method, create figures | Draft section 4, 3 figures |
| 5 | Write discussion, polish | Complete draft v1 |
| 6–8 | Run experiments | Results tables + figures |
| 9 | Integrate results, final polish | Complete draft v2 |
| 10 | Internal review, prepare submission | Submission package |

---

## NEXT STEPS

1. **Start here:** Extract introduction (section 1) from dissertation chapter 1
2. **Create:** Comparison table (plausibility vs faithfulness vs falsifiability)
3. **Promote:** Falsifiability criterion to boxed theorem
4. **Plan:** Minimal experimental setup (dataset, model, methods)

---

## NOTES

- **Article A is the CORE contribution** → prioritize clarity and rigor
- **Keep experiments lean** → reproducible, decisive, minimal
- **Avoid scope creep** → stay in verification, unit hypersphere, ArcFace/CosFace
- **Tie to Popper/Daubert** → but don't over-elaborate (1 paragraph max)
- **Open artifacts** → code, data, counterfactuals for reproducibility

---

**Status:** Ready to begin content extraction.
**Next Action:** Extract introduction from dissertation chapter 1.
