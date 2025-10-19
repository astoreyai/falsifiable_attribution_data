# Article A: Falsifiable Attribution for Face Verification via Counterfactual Score Prediction

**Status:** NOT STARTED
**Target Venues:** IJCV, IEEE TPAMI
**Timeline:** Weeks 3–10
**Article Type:** Theory + Method + Demonstration

---

## Quick Overview

This article extracts the **core theoretical and methodological contribution** from the dissertation: a falsifiable criterion for attribution explanations based on counterfactual score prediction on the unit hypersphere.

### What This Article Provides

- **Formal falsifiability criterion** with testable predictions
- **Geometric interpretation** on unit hypersphere (ArcFace/CosFace)
- **Counterfactual generation algorithm**
- **Experimental validation** showing which attribution methods pass/fail
- **Method-agnostic framework** applicable to any gradient-based attribution

### Target Length
10–12 pages

### Key Contribution
First XAI evaluation criterion that produces **testable predictions** (Δ-score accuracy) rather than just correlation or subjective assessment.

---

## Directory Structure

```
article_A_theory_method/
├── README.md                     ← This file
├── WORKFLOW.md                   ← Detailed step-by-step workflow
├── manuscript/                   ← Article drafts
│   ├── article_A_draft.md
│   ├── abstract.txt
│   └── keywords.txt
├── figures/                      ← High-resolution figures
│   ├── fig1_comparison_table.pdf
│   ├── fig2_geometric_interpretation.pdf
│   ├── fig3_method_flowchart.pdf
│   ├── fig4_delta_prediction_scatter.pdf
│   └── fig5_plausibility_gate.pdf
├── tables/                       ← Results tables
│   └── table1_results_summary.csv
├── code/                         ← Reproducible implementation
│   ├── falsification_test.py
│   ├── README.md
│   ├── requirements.txt
│   └── LICENSE
├── bibliography/                 ← References
│   └── article_A_refs.bib
├── reviews/                      ← Peer review responses
└── submission/                   ← Submission materials
    ├── cover_letter.md
    ├── author_contributions.md
    └── competing_interests.md
```

---

## Getting Started

### Step 1: Read the Workflow
```bash
cat WORKFLOW.md
```

The workflow provides detailed instructions for:
- Extracting content from dissertation chapters
- Creating required figures and tables
- Running minimal experiments
- Polishing and preparing for submission

### Step 2: Extract Content
Begin with introduction extraction:
- Source: `../falsifiable_attribution_dissertation/chapters/chapter_01_introduction.md`
- Target: `manuscript/article_A_draft.md` (section 1)

### Step 3: Follow the Workflow
See `WORKFLOW.md` for complete phase-by-phase instructions.

---

## Key Files to Create

### 1. Manuscript
- `manuscript/article_A_draft.md` - Main article text
- `manuscript/abstract.txt` - 150–200 word abstract
- `manuscript/keywords.txt` - 5–7 keywords

### 2. Figures (5 total)
1. **Comparison table** - Plausibility vs faithfulness vs falsifiability
2. **Geometric interpretation** - Unit hypersphere showing Δ-score prediction
3. **Method flowchart** - Data → Attribution → Gate → Test → Decision
4. **Δ-prediction scatter** - Predicted vs realized score deltas
5. **Plausibility gate** - Accepted vs rejected counterfactuals

### 3. Code
- `code/falsification_test.py` - Implementation of the test
- `code/README.md` - Usage instructions
- `code/requirements.txt` - Dependencies
- `code/LICENSE` - Open source license (MIT recommended)

---

## Content Sources (from Dissertation)

| Article Section | Dissertation Source |
|----------------|---------------------|
| Introduction (1.5 pages) | `chapters/chapter_01_introduction.md` sections 1.1–1.3 |
| Background (2 pages) | `chapters/chapter_02_literature_review.md` sections 2.1–2.3 |
| Theory (3 pages) | `chapters/chapter_03_theory_COMPLETE.md` (all) |
| Method (2 pages) | `chapters/chapter_04_methodology_COMPLETE.md` sections 4.1–4.2 |
| Experiments (2.5 pages) | NEW - to be run in weeks 6–8 |
| Discussion (1 page) | NEW - to be written |

---

## What to Include vs Exclude

### ✅ INCLUDE in Article A
- Falsifiability criterion (theorem box)
- Geometric interpretation (unit hypersphere)
- Counterfactual generation algorithm
- Minimal decisive experiments
- Method-agnostic scope
- Popper/Daubert alignment (1 paragraph)

### ❌ EXCLUDE from Article A (save for other articles)
- Detailed regulatory discussion → Article C
- Forensic reporting templates → Article B
- Deployment thresholds → Article B
- Policy implications → Article C
- Over-claiming beyond verification

---

## Timeline

| Week | Activity |
|------|----------|
| 3 | Extract intro, background, theory |
| 4 | Extract method, create figures |
| 5 | Write discussion, polish |
| 6–8 | Run experiments, create result figures |
| 9 | Integrate results, final polish |
| 10 | Internal review, prepare submission |

---

## Success Criteria

Article A is complete when:

- [ ] Manuscript is 10–12 pages
- [ ] All 5 figures are high-resolution (300+ DPI)
- [ ] Experiments are reproducible (code + data available)
- [ ] Theorem box clearly states falsifiability criterion
- [ ] Results show which attribution methods pass/fail test
- [ ] No scope creep beyond verification + unit hypersphere
- [ ] All claims are supported by theory or experiments
- [ ] Ready for submission to IJCV or TPAMI

---

## Notes

- This is the **CORE contribution** → prioritize clarity and rigor
- Keep experiments **lean and reproducible**
- Stay focused on **verification** (1:1), don't creep into identification (1:N)
- **Open artifacts** for reproducibility (code, data, counterfactuals)

---

## Next Steps

1. **Read** `WORKFLOW.md` for detailed instructions
2. **Start** with content extraction (introduction from dissertation)
3. **Create** comparison table (plausibility vs faithfulness vs falsifiability)
4. **Promote** falsifiability criterion to boxed theorem

---

**Questions?** See `WORKFLOW.md` for step-by-step guidance.
