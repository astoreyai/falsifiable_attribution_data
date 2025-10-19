# Article B: Evidence Thresholds for Explainable Face Verification

**Full Title:** Evidence Thresholds for Explainable Face Verification: Counterfactual Faithfulness, Uncertainty, and Reporting

**Target Venues:** IEEE T-IFS, Pattern Recognition, Forensic Sci. Int.: Digital Investigation
**Article Type:** Protocol + Deployment Thresholds + Reporting Template
**Timeline:** Weeks 4–10
**Status:** NOT STARTED

---

## OBJECTIVE

Extract and package the **operational protocol** and **deployment-ready reporting framework** from the dissertation into a practitioner-focused article that enables forensic and regulatory compliance for face verification explanations.

---

## WHAT'S ALREADY IN THE DISSERTATION

From `falsifiable_attribution_dissertation/chapters/`:

- **Operational protocol** with pseudocode (chapter_04_methodology_COMPLETE.md)
- **Pre-registered endpoints** (rank-correlation, CI calibration)
- **Plausibility criteria** (LPIPS/FID thresholds, exclusion rules)
- **Statistical plan** (paired tests, bootstrap CIs, multiplicity control)
- **Counterfactual generation pipeline** (Algorithm 3.1 / Theorem 3.6)
- **Worked decision examples** (NOT FALSIFIED vs FALSIFIED)

---

## ARTICLE STRUCTURE (12–15 pages)

### 1. Introduction (2 pages)
**Extract from:** dissertation chapter_01_introduction.md

**Content:**
- Problem: XAI methods deployed without validation standards
- Gap: No accepted thresholds for "good enough" explanations in forensic/regulated contexts
- Contribution: Pre-registered protocol + acceptance thresholds + reporting template
- Scope: Face verification (1:1) for evidentiary/compliance use cases

**New work needed:**
- [ ] Tighten to 2 pages
- [ ] Add ONE motivating case: forensic investigation needing explanation evidence
- [ ] Emphasize practitioner/auditor audience

### 2. Background: Evidentiary Requirements (2 pages)
**Extract from:** dissertation chapter_02_literature_review.md (regulatory sections)

**Content:**
- Brief overview of AI Act / GDPR / Daubert requirements
- What "meaningful information" and "testability" demand
- Why XAI alone is insufficient → need validation protocol
- Prior work on XAI evaluation (what's missing: pre-registered thresholds)

**New work needed:**
- [ ] Condense regulatory discussion to 1 page
- [ ] Add table: Requirement → Evidence Gap → Protocol Component
- [ ] Cite Article A for falsifiability theory (if published)

### 3. Operational Protocol (4 pages)
**Extract from:** dissertation chapter_04_methodology_COMPLETE.md

**Content:**
- **Inputs**: Image pair, model, attribution method
- **Step 1: Attribution Generation** (brief)
- **Step 2: Plausibility Gate**
  - LPIPS/FID thresholds (specify exact values)
  - Rule-based exclusions (list all)
  - Acceptance/rejection decision
- **Step 3: Δ-Prediction Test**
  - Predicted vs realized score deltas
  - Statistical test (correlation, CI calibration)
  - Pass/fail thresholds (PRE-REGISTERED)
- **Step 4: Decision**
  - NOT FALSIFIED if thresholds met
  - FALSIFIED otherwise
- **Uncertainty quantification**: Bootstrap CIs, multiplicity control
- **Algorithm pseudocode** (1 page box)

**New work needed:**
- [ ] Freeze all thresholds (no "TBD")
  - LPIPS threshold: [specify, e.g., < 0.3]
  - FID threshold: [specify, e.g., < 50]
  - Correlation floor: [specify, e.g., ρ > 0.7]
  - CI calibration: [specify, e.g., 95% nominal coverage]
- [ ] Add flowchart: inputs → gate → test → decision → report
- [ ] Add worked example (step-by-step with real numbers)

### 4. Pre-Registered Endpoints & Thresholds (2 pages)
**NEW section** (based on dissertation methodology)

**Content:**
- **Primary endpoint**: Correlation between predicted and realized score-deltas
  - Acceptance threshold: ρ > 0.7 (example; freeze actual value)
  - Rationale: ensures predictions are informative
- **Secondary endpoint**: CI calibration
  - Acceptance threshold: 95% nominal coverage within [90%, 100%]
  - Rationale: ensures uncertainty is honest
- **Plausibility gate**:
  - LPIPS < 0.3 (example; freeze actual value)
  - FID < 50 (example; freeze actual value)
  - Rationale: ensures counterfactuals are realistic
- **Decision rule**:
  - NOT FALSIFIED if all endpoints meet thresholds
  - FALSIFIED if any endpoint fails
- **Pre-registration commitment**: Thresholds set before experiments, no post-hoc adjustment

**New work needed:**
- [ ] Freeze all threshold values (based on pilot experiments or literature)
- [ ] Justify each threshold (cite pilot data or prior work)
- [ ] Add table: Endpoint → Threshold → Rationale → Source

### 5. Forensic Reporting Template (2 pages)
**NEW section** (based on dissertation regulatory analysis)

**Content:**
- **Purpose**: Enable Daubert-style disclosure and regulatory compliance
- **Template structure**:
  1. **Method**: Attribution algorithm used (Grad-CAM, IG, SHAP, etc.)
  2. **Parameters**: Hyperparameters, counterfactual settings
  3. **Δ-Prediction Accuracy**: Correlation, p-value, CI
  4. **CI Calibration**: Coverage rate, calibration plot
  5. **Known Error Rates**: Failure modes, limits to generality
  6. **Dataset/Model Limitations**: Training data, architecture, demographic stratification
  7. **Recommendation**: NOT FALSIFIED or FALSIFIED + confidence statement
- **Example filled template** (1 page)
- **Integration with existing forensic workflows** (brief paragraph)

**New work needed:**
- [ ] Create actual template (LaTeX or Markdown)
- [ ] Fill in one example report with real/simulated data
- [ ] Add calibration plot example figure

### 6. Risk Analysis & Limitations (1.5 pages)
**Extract from:** dissertation chapter_04_methodology_COMPLETE.md (limitations section)

**Content:**
- **Dataset limitations**: LFW, CASIA-WebFace (not representative of all populations)
- **Model limitations**: ArcFace/CosFace only (not all verification architectures)
- **Demographic stratification**: Acknowledge need for subgroup analysis (but out of scope)
- **Transfer risks**: Findings may not generalize beyond tested conditions
- **Reporting requirements**: Practitioners must disclose limits in forensic reports

**New work needed:**
- [ ] Condense to 1.5 pages
- [ ] Add "Threats to Validity" table
- [ ] Emphasize that limitations MUST be disclosed in reporting template

### 7. Experimental Validation (2.5 pages)
**Extract from:** dissertation experiments (to be run in weeks 6–8)

**Content:**
- **Datasets**: LFW or CASIA-WebFace
- **Models**: ArcFace (ResNet-50 or similar)
- **Attribution methods**: Grad-CAM, IG, SHAP (2–3 methods)
- **Primary results**:
  - Table: Method → Correlation → CI Calibration → Pass/Fail
  - Figure: Calibration plot (predicted vs actual coverage)
  - Figure: Decision examples (NOT FALSIFIED vs FALSIFIED cases)
- **Interpretation**: Which methods meet thresholds, which fail
- **Practitioner guidance**: Recommendations for deployment

**New work needed:**
- [ ] Run validation experiments (same as Article A)
- [ ] Produce 2–3 result tables
- [ ] Produce 2–3 result figures
- [ ] Write interpretation section

### 8. Discussion & Practitioner Guidance (1 page)
**Content:**
- **What the protocol enables**: Auditable, evidence-grade explanations
- **When to use**: Forensic, regulatory, compliance contexts
- **When NOT to use**: Limitations (datasets, models, architectures)
- **Integration**: How to incorporate into existing forensic pipelines
- **Future work**: Demographic stratification, extended architectures

**New work needed:**
- [ ] Write fresh (1 page)
- [ ] Provide clear practitioner recommendations
- [ ] Link to reporting template

---

## EXTRACTION WORKFLOW

### Phase 1: Content Assembly (Week 4)

**Step 1.1:** Create manuscript skeleton
```bash
cd /home/aaron/projects/xai/PHD_PIPELINE/article_B_protocol_thresholds/manuscript
touch article_B_draft.md
```

**Step 1.2:** Extract introduction
- [ ] Copy relevant sections from `../falsifiable_attribution_dissertation/chapters/chapter_01_introduction.md`
- [ ] Trim to 2 pages
- [ ] Add forensic case example
- [ ] Emphasize practitioner audience

**Step 1.3:** Extract background (regulatory)
- [ ] Copy regulatory sections from `chapter_02_literature_review.md`
- [ ] Condense to 1 page
- [ ] Create Requirement → Gap → Protocol table

**Step 1.4:** Extract operational protocol
- [ ] Copy protocol from `chapter_04_methodology_COMPLETE.md`
- [ ] Freeze all thresholds (no "TBD")
- [ ] Add pseudocode box
- [ ] Add worked example

**Step 1.5:** Create pre-registered endpoints section (NEW)
- [ ] Define primary/secondary endpoints
- [ ] Specify all thresholds with rationale
- [ ] Create endpoint table
- [ ] Commit to pre-registration (no post-hoc changes)

**Step 1.6:** Create forensic reporting template (NEW)
- [ ] Design template structure (7 sections)
- [ ] Fill in one example report
- [ ] Add calibration plot figure
- [ ] Integrate with Daubert requirements

**Step 1.7:** Extract limitations
- [ ] Copy limitations from `chapter_04_methodology_COMPLETE.md`
- [ ] Condense to 1.5 pages
- [ ] Create "Threats to Validity" table
- [ ] Emphasize disclosure requirements

### Phase 2: Figures & Tables (Week 5)

**Step 2.1:** Create Requirement → Gap → Protocol table
```markdown
| Requirement (Source) | Evidence Gap | Protocol Component |
|---------------------|--------------|-------------------|
| "Meaningful information" (GDPR Art. 22) | No validation standard | Δ-prediction test |
| Testability (Daubert) | No error rates | CI calibration, known failure modes |
| Accuracy (AI Act Art. 13) | No acceptance threshold | Pre-registered correlation floor |
```

**Step 2.2:** Create Endpoint → Threshold → Rationale table
```markdown
| Endpoint | Threshold | Rationale | Source |
|---------|-----------|-----------|--------|
| Δ-score correlation | ρ > 0.7 | Strong predictive accuracy | Pilot data / [cite] |
| CI calibration | 90–100% coverage | Honest uncertainty | Standard practice |
| LPIPS (plausibility) | < 0.3 | Perceptual similarity | [cite] |
| FID (plausibility) | < 50 | Distributional realism | [cite] |
```

**Step 2.3:** Create flowchart
- [ ] Inputs (image pair, model, method) → Attribution → Plausibility Gate (pass/fail) → Δ-Test (pass/fail) → Decision (NOT FALSIFIED / FALSIFIED) → Reporting Template
- [ ] Use existing diagram from dissertation if available

**Step 2.4:** Create forensic reporting template (actual template)
```markdown
## Forensic Report: Face Verification Explanation

**1. Method**: [e.g., Grad-CAM]
**2. Parameters**: [e.g., target layer: conv5_3, counterfactual budget: 100 iterations]
**3. Δ-Prediction Accuracy**: ρ = 0.82, p < 0.001, 95% CI [0.76, 0.88]
**4. CI Calibration**: 94% coverage (nominal 95%)
**5. Known Error Rates**: [list failure modes]
**6. Limitations**: [dataset, model, demographic scope]
**7. Recommendation**: NOT FALSIFIED (passes all pre-registered thresholds)
```

**Step 2.5:** Create calibration plot figure
- [ ] X-axis: Nominal coverage (e.g., 90%, 95%, 99%)
- [ ] Y-axis: Actual coverage
- [ ] Reference line: y = x (perfect calibration)
- [ ] Plot: Actual coverage points with error bars

**Step 2.6:** Create "Threats to Validity" table
```markdown
| Threat | Description | Mitigation | Disclosure Required |
|--------|-------------|------------|---------------------|
| Dataset bias | LFW/CASIA not representative | Acknowledge in report | Yes |
| Architecture scope | ArcFace/CosFace only | Test on target architecture | Yes |
| Demographic stratification | No subgroup analysis | Future work | Yes |
```

### Phase 3: Freeze Thresholds (Week 4–5)

**Step 3.1:** Review pilot experiments or literature
- [ ] Determine realistic threshold for Δ-score correlation (e.g., ρ > 0.7)
- [ ] Determine realistic threshold for LPIPS (e.g., < 0.3)
- [ ] Determine realistic threshold for FID (e.g., < 50)
- [ ] Determine realistic threshold for CI calibration (e.g., 90–100%)

**Step 3.2:** Document rationale for each threshold
- [ ] Cite pilot data or prior work
- [ ] Explain why threshold is appropriate
- [ ] Commit to NO post-hoc adjustment

**Step 3.3:** Pre-register thresholds
- [ ] Create "Pre-Registration Document" (appendix or separate file)
- [ ] Timestamp before running validation experiments
- [ ] Include all thresholds and decision rules

### Phase 4: Experiments (Weeks 6–8)

**Step 4.1:** Set up validation environment
- [ ] Select dataset: LFW (public, reproducible)
- [ ] Load pretrained ArcFace model
- [ ] Implement 2–3 attribution methods (Grad-CAM, IG, SHAP)

**Step 4.2:** Run protocol validation
- [ ] Generate counterfactuals for 100–200 image pairs
- [ ] Apply plausibility gate (measure LPIPS/FID)
- [ ] Run Δ-prediction test (measure correlation, CIs)
- [ ] Evaluate CI calibration

**Step 4.3:** Create results visualizations
- [ ] Table: Method → Correlation → CI Calibration → Pass/Fail
- [ ] Calibration plot (nominal vs actual coverage)
- [ ] Example reports (one NOT FALSIFIED, one FALSIFIED)

**Step 4.4:** Write experimental section
- [ ] Describe setup (dataset, model, methods)
- [ ] Report results (tables + figures)
- [ ] Interpret findings (which methods pass/fail)
- [ ] Provide practitioner recommendations

### Phase 5: Polish & Review (Weeks 9–10)

**Step 5.1:** Internal consistency check
- [ ] All thresholds match pre-registration
- [ ] Reporting template includes all required fields
- [ ] Limitations are honestly disclosed
- [ ] No scope creep beyond verification
- [ ] No over-claiming beyond tested conditions

**Step 5.2:** Writing polish
- [ ] Abstract (150–200 words)
- [ ] Keywords (5–7)
- [ ] Practitioner-friendly language (minimize jargon)
- [ ] Section transitions
- [ ] Citation formatting (check venue requirements)

**Step 5.3:** Artifacts preparation
- [ ] Practitioner checklist (how to use protocol)
- [ ] Reference reports (example filled templates)
- [ ] Calibration plots
- [ ] Code for protocol implementation
- [ ] README for code release

**Step 5.4:** Pre-submission checklist
- [ ] Length: 12–15 pages (check venue limits)
- [ ] Figures: high resolution (300 DPI minimum)
- [ ] Reporting template: included in appendix or supplementary
- [ ] Pre-registration document: included
- [ ] References: complete and formatted
- [ ] Author affiliations and contact

---

## KEY DESIGN DECISIONS

### What to INCLUDE
✅ Operational protocol (step-by-step)
✅ Pre-registered endpoints and thresholds
✅ Plausibility criteria (LPIPS/FID + exclusion rules)
✅ Forensic reporting template (Daubert-aligned)
✅ Risk analysis and limitations
✅ Practitioner guidance

### What to EXCLUDE (save for other articles)
❌ Detailed theory (Article A)
❌ Extensive policy discussion (Article C)
❌ Over-claiming beyond tested conditions
❌ Demographic stratification (acknowledge as future work)

---

## CONTENT MAPPING: DISSERTATION → ARTICLE B

| Article Section | Dissertation Source | Transformation Needed |
|----------------|---------------------|----------------------|
| Introduction | chapter_01 sections 1.1–1.3 | Add forensic case example |
| Background | chapter_02 (regulatory sections) | Condense to 1 page + table |
| Protocol | chapter_04 sections 4.1–4.3 | Freeze thresholds, add pseudocode |
| Endpoints | chapter_04 (statistical plan) | NEW section, formalize pre-registration |
| Reporting | NEW (based on Daubert analysis) | Create template + example |
| Limitations | chapter_04 section 4.5 | Condense to 1.5 pages + table |
| Experiments | NEW (weeks 6–8) | Run validation, create tables/figs |
| Discussion | NEW (week 9) | Write fresh, practitioner-focused |

---

## PROGRESS TRACKING

Use `TodoWrite` tool to track:
- [ ] Thresholds frozen (pre-registered)
- [ ] Reporting template created
- [ ] Content extraction completed
- [ ] Figures created (flowchart, calibration plot, tables)
- [ ] Experiments run and results visualized
- [ ] Draft complete (12–15 pages)
- [ ] Internal review passed
- [ ] Ready for submission

---

## DELIVERABLES CHECKLIST

### Manuscript Files
- [ ] `manuscript/article_B_draft.md` (or .tex)
- [ ] `manuscript/abstract.txt`
- [ ] `manuscript/keywords.txt`
- [ ] `manuscript/pre_registration.md`

### Figures (high-res)
- [ ] `figures/fig1_requirement_gap_protocol_table.pdf`
- [ ] `figures/fig2_protocol_flowchart.pdf`
- [ ] `figures/fig3_calibration_plot.pdf`
- [ ] `figures/fig4_example_reports.pdf`

### Tables
- [ ] `tables/table1_endpoint_thresholds.csv`
- [ ] `tables/table2_validation_results.csv`
- [ ] `tables/table3_threats_to_validity.csv`

### Templates
- [ ] `manuscript/forensic_reporting_template.md`
- [ ] `manuscript/practitioner_checklist.md`

### Code & Data
- [ ] `code/protocol_implementation.py`
- [ ] `code/README.md`
- [ ] `code/requirements.txt`
- [ ] `code/LICENSE`

### Bibliography
- [ ] `bibliography/article_B_refs.bib`

### Submission Materials
- [ ] `submission/cover_letter.md`
- [ ] `submission/author_contributions.md`
- [ ] `submission/competing_interests.md`

---

## TIMELINE (7 WEEKS)

| Week | Activity | Deliverable |
|------|----------|------------|
| 4 | Extract protocol, freeze thresholds | Draft sections 1–4 |
| 5 | Create reporting template, figures | Draft sections 5–6 |
| 6–8 | Run validation experiments | Results tables + figures |
| 9 | Integrate results, write discussion | Complete draft v1 |
| 10 | Final polish, prepare submission | Submission package |

---

## NEXT STEPS

1. **Start here:** Freeze all thresholds (correlation floor, LPIPS, FID, CI calibration)
2. **Create:** Pre-registration document
3. **Extract:** Protocol from dissertation chapter 4
4. **Design:** Forensic reporting template

---

## NOTES

- **Article B is the PRACTITIONER contribution** → prioritize clarity and usability
- **Pre-register thresholds** → no post-hoc adjustment allowed
- **Honest limitations** → disclose all dataset/model/demographic scope constraints
- **Forensic focus** → align reporting template with Daubert/GDPR/AI Act
- **Open artifacts** → code, templates, checklists for reproducibility

---

**Status:** Ready to begin threshold freezing and content extraction.
**Next Action:** Freeze pre-registered thresholds based on pilot data or literature.
