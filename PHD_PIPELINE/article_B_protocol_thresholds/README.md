# Article B: Evidence Thresholds for Explainable Face Verification

**Full Title:** Evidence Thresholds for Explainable Face Verification: Counterfactual Faithfulness, Uncertainty, and Reporting

**Status:** NOT STARTED
**Target Venues:** IEEE T-IFS, Pattern Recognition, Forensic Sci. Int.: Digital Investigation
**Timeline:** Weeks 4–10
**Article Type:** Protocol + Deployment Thresholds + Reporting Template

---

## Quick Overview

This article extracts the **operational protocol** and **deployment framework** from the dissertation, providing practitioners with a pre-registered, auditable method for validating face verification explanations in forensic and regulatory contexts.

### What This Article Provides

- **Operational protocol** (step-by-step, with pseudocode)
- **Pre-registered endpoints and acceptance thresholds**
- **Plausibility criteria** (LPIPS/FID + exclusion rules)
- **Forensic reporting template** (Daubert/GDPR/AI Act aligned)
- **Risk analysis** and limitations disclosure
- **Practitioner guidance** for deployment

### Target Length
12–15 pages

### Key Contribution
First **pre-registered validation protocol** for XAI in face verification with frozen thresholds and evidence-grade reporting template.

---

## Directory Structure

```
article_B_protocol_thresholds/
├── README.md                     ← This file
├── WORKFLOW.md                   ← Detailed step-by-step workflow
├── manuscript/                   ← Article drafts
│   ├── article_B_draft.md
│   ├── abstract.txt
│   ├── keywords.txt
│   ├── pre_registration.md
│   ├── forensic_reporting_template.md
│   └── practitioner_checklist.md
├── figures/                      ← High-resolution figures
│   ├── fig1_requirement_gap_protocol_table.pdf
│   ├── fig2_protocol_flowchart.pdf
│   ├── fig3_calibration_plot.pdf
│   └── fig4_example_reports.pdf
├── tables/                       ← Results tables
│   ├── table1_endpoint_thresholds.csv
│   ├── table2_validation_results.csv
│   └── table3_threats_to_validity.csv
├── code/                         ← Reproducible implementation
│   ├── protocol_implementation.py
│   ├── README.md
│   ├── requirements.txt
│   └── LICENSE
├── bibliography/                 ← References
│   └── article_B_refs.bib
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
- Freezing pre-registered thresholds
- Extracting protocol from dissertation
- Creating forensic reporting template
- Running validation experiments
- Preparing submission materials

### Step 2: Freeze Thresholds (CRITICAL FIRST STEP)
Before extracting any content:
- [ ] Determine Δ-score correlation floor (e.g., ρ > 0.7)
- [ ] Determine LPIPS threshold (e.g., < 0.3)
- [ ] Determine FID threshold (e.g., < 50)
- [ ] Determine CI calibration range (e.g., 90–100%)
- [ ] Document rationale for each
- [ ] Create `manuscript/pre_registration.md`
- [ ] **NO post-hoc adjustments allowed**

### Step 3: Follow the Workflow
See `WORKFLOW.md` for complete phase-by-phase instructions.

---

## Key Files to Create

### 1. Manuscript
- `manuscript/article_B_draft.md` - Main article text
- `manuscript/abstract.txt` - 150–200 word abstract
- `manuscript/keywords.txt` - 5–7 keywords
- `manuscript/pre_registration.md` - Frozen thresholds + rationale

### 2. Templates
- `manuscript/forensic_reporting_template.md` - Evidence-grade report structure
- `manuscript/practitioner_checklist.md` - How to use the protocol

### 3. Figures (4 total)
1. **Requirement → Gap → Protocol table** - Links regulatory needs to protocol
2. **Protocol flowchart** - Inputs → Gate → Test → Decision → Report
3. **Calibration plot** - Nominal vs actual CI coverage
4. **Example reports** - One NOT FALSIFIED, one FALSIFIED

### 4. Tables (3 total)
1. **Endpoint → Threshold → Rationale** - All pre-registered thresholds
2. **Validation results** - Method → Correlation → Calibration → Pass/Fail
3. **Threats to validity** - Limitations and disclosure requirements

### 5. Code
- `code/protocol_implementation.py` - Implementation of the protocol
- `code/README.md` - Usage instructions
- `code/requirements.txt` - Dependencies
- `code/LICENSE` - Open source license

---

## Content Sources (from Dissertation)

| Article Section | Dissertation Source |
|----------------|---------------------|
| Introduction (2 pages) | `chapters/chapter_01_introduction.md` sections 1.1–1.3 |
| Background (2 pages) | `chapters/chapter_02_literature_review.md` (regulatory) |
| Protocol (4 pages) | `chapters/chapter_04_methodology_COMPLETE.md` sections 4.1–4.3 |
| Endpoints (2 pages) | `chapters/chapter_04_methodology_COMPLETE.md` (stats plan) + NEW |
| Reporting (2 pages) | NEW - based on Daubert analysis in chapter 2 |
| Limitations (1.5 pages) | `chapters/chapter_04_methodology_COMPLETE.md` section 4.5 |
| Experiments (2.5 pages) | NEW - to be run in weeks 6–8 |
| Discussion (1 page) | NEW - to be written |

---

## What to Include vs Exclude

### ✅ INCLUDE in Article B
- Operational protocol (step-by-step)
- Pre-registered thresholds (frozen before experiments)
- Plausibility gate (LPIPS/FID + rules)
- Forensic reporting template
- CI calibration + uncertainty quantification
- Risk analysis + limitations
- Practitioner guidance

### ❌ EXCLUDE from Article B (save for other articles)
- Detailed theory / proofs → Article A
- Extensive policy discussion → Article C
- Over-claiming beyond tested conditions
- Demographic stratification (acknowledge as future work)

---

## Timeline

| Week | Activity |
|------|----------|
| 4 | Freeze thresholds, extract protocol, create template |
| 5 | Create figures and tables |
| 6–8 | Run validation experiments |
| 9 | Integrate results, write discussion |
| 10 | Final polish, prepare submission |

---

## Critical Success Factors

Article B succeeds when:

- [ ] **Thresholds are pre-registered** (before experiments, NO post-hoc changes)
- [ ] **Reporting template is complete** (all 7 required fields)
- [ ] **Limitations are honestly disclosed** (dataset, model, demographic scope)
- [ ] **Validation experiments confirm thresholds** (some methods pass, some fail)
- [ ] **Practitioner-ready** (clear guidance, usable checklist)
- [ ] **Regulatory-aligned** (Daubert/GDPR/AI Act requirements addressed)

---

## Success Criteria

Article B is complete when:

- [ ] Manuscript is 12–15 pages
- [ ] All thresholds are frozen and justified
- [ ] Pre-registration document is timestamped
- [ ] Forensic reporting template is complete (with example)
- [ ] All 4 figures are high-resolution (300+ DPI)
- [ ] All 3 tables are complete
- [ ] Validation experiments are reproducible (code + data available)
- [ ] Limitations are honestly disclosed
- [ ] Ready for submission to T-IFS/Pattern Recognition/FSI-DI

---

## Pre-Registration Commitment

**CRITICAL:** This article's credibility depends on **pre-registering all thresholds before running validation experiments**.

Steps:
1. Freeze thresholds based on pilot data or literature (Week 4)
2. Document rationale in `manuscript/pre_registration.md`
3. Timestamp the pre-registration file
4. Run validation experiments WITHOUT changing thresholds (Weeks 6–8)
5. Report results honestly (even if some methods fail)

**NO post-hoc adjustment allowed.**

---

## Practitioner Focus

This article is written for:
- Forensic investigators using face verification
- Auditors evaluating AI systems for compliance
- Regulatory bodies reviewing XAI evidence
- Researchers needing validation protocols

Language should be:
- Clear and jargon-free where possible
- Step-by-step and procedural
- Honest about limitations
- Usable by non-experts

---

## Notes

- **Article B is the PRACTITIONER contribution** → prioritize clarity and usability
- **Pre-register thresholds** → freeze before experiments, no post-hoc changes
- **Honest limitations** → disclose all scope constraints
- **Forensic focus** → align with Daubert/GDPR/AI Act
- **Open artifacts** → code, templates, checklists

---

## Next Steps

1. **Read** `WORKFLOW.md` for detailed instructions
2. **Freeze** all thresholds (correlation, LPIPS, FID, CI calibration)
3. **Create** pre-registration document
4. **Extract** protocol from dissertation chapter 4
5. **Design** forensic reporting template

---

**Questions?** See `WORKFLOW.md` for step-by-step guidance.
