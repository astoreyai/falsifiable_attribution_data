# Dissertation-to-Articles Extraction Plan

**Dissertation:** Falsifiable Attribution for Face Verification
**Target:** 3 Journal Articles
**Timeline:** 8–10 weeks (parallelizable)
**Status:** Setup Complete, Ready to Begin

---

## EXECUTIVE SUMMARY

Your dissertation contains **three distinct, publication-ready contributions** that can be extracted with modest packaging work and limited top-up experiments. This document provides the master plan for extracting and publishing all three articles in parallel.

---

## THE THREE ARTICLES

### Article A: Falsifiable Attribution for Face Verification via Counterfactual Score Prediction
**Type:** Theory + Method + Demonstration
**Length:** 10–12 pages
**Venues:** IJCV, IEEE TPAMI
**Audience:** Computer vision / XAI researchers
**Core Contribution:** First falsifiable criterion for attributions via Δ-score prediction on unit hypersphere

**Location:** `/home/aaron/projects/xai/PHD_PIPELINE/article_A_theory_method/`

**What it provides:**
- Formal falsifiability criterion (theorem box)
- Geometric interpretation (unit hypersphere, geodesics)
- Counterfactual generation algorithm
- Minimal decisive experiments (2–3 attribution methods)
- Method-agnostic framework

**See:** `article_A_theory_method/README.md` and `WORKFLOW.md`

---

### Article B: Evidence Thresholds for Explainable Face Verification
**Type:** Protocol + Deployment Thresholds + Reporting Template
**Length:** 12–15 pages
**Venues:** IEEE T-IFS, Pattern Recognition, Forensic Sci. Int.: Digital Investigation
**Audience:** Practitioners, forensic scientists, auditors
**Core Contribution:** First pre-registered validation protocol with frozen thresholds and evidence-grade reporting

**Location:** `/home/aaron/projects/xai/PHD_PIPELINE/article_B_protocol_thresholds/`

**What it provides:**
- Operational protocol (step-by-step)
- Pre-registered endpoints and acceptance thresholds
- Plausibility criteria (LPIPS/FID + exclusion rules)
- Forensic reporting template (Daubert/GDPR/AI Act aligned)
- Risk analysis and limitations disclosure
- Practitioner guidance

**See:** `article_B_protocol_thresholds/README.md` and `WORKFLOW.md`

---

### Article C: From "Meaningful Information" to Testable Explanations
**Type:** Policy Synthesis / Short Communications
**Length:** 6–8 pages
**Venues:** AI & Law, Forensic Science Policy & Management, CACM
**Audience:** Regulators, policy makers, legal professionals
**Core Contribution:** First policy synthesis translating legal requirements into concrete validation standards

**Location:** `/home/aaron/projects/xai/PHD_PIPELINE/article_C_policy_standards/`

**What it provides:**
- Regulatory requirements (AI Act, GDPR, Daubert) condensed
- Evidentiary gap analysis
- Minimal evidence requirements (table)
- Compliance template (simplified from Article B)
- Policy recommendations

**See:** `article_C_policy_standards/README.md` and `WORKFLOW.md`

---

## HOW THE ARTICLES RELATE

```
                    DISSERTATION
                         |
         +---------------+---------------+
         |               |               |
    ARTICLE A        ARTICLE B      ARTICLE C
    (Theory/         (Protocol/      (Policy/
     Method)         Thresholds)     Standards)
         |               |               |
    Falsifiability   Pre-registered   Requirements
    Criterion        Validation       → Validation
         |               |               |
    Unit Hyper-      Reporting        Compliance
    sphere Geom.     Template         Pathway
         |               |               |
    Counter-         CI Calibration   Daubert/GDPR/
    factuals         + Thresholds     AI Act
         |               |               |
         +-------+-------+-------+-------+
                 |               |
            CORE PAPERS      POLICY PAPER
          (cite each other)  (cites A & B)
```

### Citation Strategy
- **Article C** cites Articles A & B (if published) or describes approach generically
- **Article B** cites Article A for theoretical foundation
- **Article A** stands alone (core theory)

### Publication Order
**Recommended sequence:**
1. **Submit A first** (establishes theoretical foundation)
2. **Submit B second** (cites A for theory, adds protocol)
3. **Submit C last** (cites A & B, policy synthesis)

**Alternative (parallel):**
- Submit all three simultaneously if timelines are tight
- Mention "companion papers" in cover letters
- Update citations in revision if others are accepted

---

## WORKPLAN (8–10 WEEKS, PARALLELIZABLE)

### Weeks 1–2: Dissertation Polish (Optional)
**Goal:** Strengthen dissertation as baseline (optional, can be done in parallel)

**Tasks:**
- [ ] Insert "Executive Definition + Theorem" box in chapter 3
- [ ] Add assumptions box (unit hypersphere, geodesics, plausibility)
- [ ] Add method flowchart to chapter 4
- [ ] Compress regulatory discussion in chapter 2 to table + paragraph
- [ ] Add cross-walk table mapping research questions → results

**Outcome:** Cleaner dissertation that makes article extraction easier

---

### Weeks 3–5: Article A (Theory/Method)
**Goal:** Extract theoretical contribution and package with minimal experiments

**Week 3 Tasks:**
- [ ] Extract introduction (1.5 pages) from dissertation chapter 1
- [ ] Extract background (2 pages) from dissertation chapter 2
- [ ] Extract theory (3 pages) from dissertation chapter 3
  - [ ] Promote falsifiability criterion to boxed theorem
  - [ ] Add assumptions box
  - [ ] Add geometric interpretation
- [ ] Create 3-column comparison table (plausibility vs faithfulness vs falsifiability)

**Week 4 Tasks:**
- [ ] Extract method (2 pages) from dissertation chapter 4
  - [ ] Condense counterfactual generation algorithm
  - [ ] Add computational complexity
- [ ] Create method flowchart
- [ ] Create geometric figure (unit hypersphere)
- [ ] Plan minimal experiments (dataset, model, 2–3 attribution methods)

**Week 5 Tasks:**
- [ ] Write discussion (1 page)
- [ ] Polish abstract and introduction
- [ ] Prepare figures for submission (high-res)
- [ ] Internal review

**Parallel with weeks 6–8:** Run experiments, integrate results

**Outcome:** Article A draft ready for submission (after experiments complete)

---

### Weeks 4–6: Article B (Protocol/Thresholds)
**Goal:** Extract protocol and create deployment-ready reporting framework

**Week 4 Tasks:**
- [ ] **CRITICAL:** Freeze all pre-registered thresholds
  - [ ] Δ-score correlation floor (e.g., ρ > 0.7)
  - [ ] LPIPS threshold (e.g., < 0.3)
  - [ ] FID threshold (e.g., < 50)
  - [ ] CI calibration range (e.g., 90–100%)
  - [ ] Document rationale (cite pilot data or literature)
  - [ ] Create `pre_registration.md` (timestamp before experiments)
- [ ] Extract introduction (2 pages) from dissertation chapter 1
- [ ] Extract background (2 pages) from dissertation chapter 2 (regulatory)

**Week 5 Tasks:**
- [ ] Extract operational protocol (4 pages) from dissertation chapter 4
  - [ ] Add pseudocode box
  - [ ] Add worked example
- [ ] Create pre-registered endpoints section (2 pages, NEW)
- [ ] Create forensic reporting template (2 pages, NEW)
  - [ ] Design template structure
  - [ ] Fill in one example
- [ ] Extract limitations (1.5 pages) from dissertation chapter 4

**Week 6 Tasks:**
- [ ] Create protocol flowchart
- [ ] Create Requirement → Gap → Protocol table
- [ ] Create Endpoint → Threshold → Rationale table
- [ ] Create calibration plot template
- [ ] Create practitioner checklist

**Parallel with weeks 6–8:** Run validation experiments, integrate results

**Outcome:** Article B draft ready for submission (after experiments complete)

---

### Weeks 6–10: Article C (Policy/Standards)
**Goal:** Extract regulatory analysis and create policy synthesis

**Week 6 Tasks:**
- [ ] Extract introduction (1 page) from dissertation chapters 1–2
- [ ] Extract regulatory requirements (2 pages) from dissertation chapter 2
  - [ ] AI Act (Arts. 13–15)
  - [ ] GDPR (Art. 22)
  - [ ] Daubert
- [ ] Extract evidentiary gap (1.5 pages) from dissertation chapter 2

**Week 7 Tasks:**
- [ ] Create minimal evidence section (2 pages, NEW)
  - [ ] Synthesize Requirement → Evidence → Method table
  - [ ] Write explanation
- [ ] Create compliance template (1.5 pages)
  - [ ] Adapt Article B's template for policy audience
  - [ ] Simplify technical details
  - [ ] Fill in example

**Week 8 Tasks:**
- [ ] Write discussion (1 page)
  - [ ] Policy implications
  - [ ] Who benefits (regulators, auditors, developers, courts)
  - [ ] Remaining gaps
  - [ ] Call to action
- [ ] Create tables:
  - [ ] Requirement → Current Practice → Gap
  - [ ] Requirement → Minimal Evidence → Method

**Weeks 9–10 Tasks:**
- [ ] Polish for policy audience (minimize jargon)
- [ ] Internal review
- [ ] Prepare submission package

**Outcome:** Article C ready for submission

---

### Weeks 6–8: EXPERIMENTS (SHARED ACROSS ARTICLES A & B)
**Goal:** Run minimal but decisive validation experiments

**Week 6 Tasks:**
- [ ] Set up experimental environment
  - [ ] Select dataset: LFW (public, reproducible)
  - [ ] Load pretrained ArcFace model (ResNet-50 or similar)
  - [ ] Implement 2–3 attribution methods (Grad-CAM, IG, SHAP)

**Week 7 Tasks:**
- [ ] Generate counterfactuals for 100–200 image pairs
- [ ] Apply plausibility gate (measure LPIPS/FID)
- [ ] Run Δ-prediction test (measure correlation, CIs)
- [ ] Evaluate CI calibration

**Week 8 Tasks:**
- [ ] Create results visualizations:
  - [ ] Scatter plot: predicted vs realized Δ-score (Article A)
  - [ ] Table: method → correlation → pass/fail (Articles A & B)
  - [ ] Calibration plot (Article B)
  - [ ] Example reports (Article B)
- [ ] Write experimental sections for Articles A & B
- [ ] Interpret findings

**Outcome:** Experimental results integrated into Articles A & B

---

### Weeks 9–10: FINALIZATION & SUBMISSION
**Goal:** Polish all three articles and prepare submission packages

**Week 9 Tasks:**
- [ ] Article A: Integrate experiments, final polish
- [ ] Article B: Integrate experiments, final polish
- [ ] Article C: Final polish (no experiments)
- [ ] Internal review of all three articles
- [ ] Check cross-references (Article C → A & B)

**Week 10 Tasks:**
- [ ] Prepare submission packages (all three):
  - [ ] Cover letters
  - [ ] Author contributions
  - [ ] Competing interests
  - [ ] High-resolution figures (300+ DPI)
  - [ ] Formatted references
- [ ] Submit Article A (IJCV or TPAMI)
- [ ] Submit Article B (T-IFS or Pattern Recognition)
- [ ] Submit Article C (AI & Law or Forensic Policy)

**Outcome:** All three articles submitted

---

## CONTENT MAPPING: DISSERTATION → ARTICLES

### From Dissertation Chapter 1 (Introduction)
- **Article A:** Introduction (1.5 pages) - motivation, contribution, scope
- **Article B:** Introduction (2 pages) - forensic/regulatory motivation
- **Article C:** Introduction (1 page) - policy/legal motivation

### From Dissertation Chapter 2 (Literature Review)
- **Article A:** Background (2 pages) - XAI methods, evaluation approaches, verification geometry
- **Article B:** Background (2 pages) - regulatory requirements (AI Act/GDPR/Daubert)
- **Article C:** Requirements (2 pages) + Gap (1.5 pages) - full regulatory analysis

### From Dissertation Chapter 3 (Theory)
- **Article A:** Theory (3 pages) - falsifiability criterion, geometric interpretation, formal properties
- **Article B:** Reference only (cite Article A)
- **Article C:** Minimal evidence section (synthesize)

### From Dissertation Chapter 4 (Methodology)
- **Article A:** Method (2 pages) - counterfactual generation algorithm
- **Article B:** Protocol (4 pages) + Endpoints (2 pages) + Limitations (1.5 pages) - full operational protocol
- **Article C:** Compliance template (adapted from Article B)

### NEW Content (to be written)
- **Article A:** Experiments (2.5 pages), Discussion (1 page)
- **Article B:** Pre-registered endpoints (2 pages), Forensic reporting template (2 pages), Experiments (2.5 pages), Discussion (1 page)
- **Article C:** Minimal evidence section (2 pages), Discussion (1 page)

---

## DELIVERABLES CHECKLIST

### Article A
- [ ] Manuscript (10–12 pages)
- [ ] 5 figures (comparison table, geometric interpretation, flowchart, scatter plot, gate)
- [ ] 1 table (results summary)
- [ ] Code release (falsification test harness)
- [ ] Submission package

### Article B
- [ ] Manuscript (12–15 pages)
- [ ] Pre-registration document
- [ ] Forensic reporting template
- [ ] Practitioner checklist
- [ ] 4 figures (tables, flowchart, calibration plot, example reports)
- [ ] 3 tables (thresholds, results, threats to validity)
- [ ] Code release (protocol implementation)
- [ ] Submission package

### Article C
- [ ] Manuscript (6–8 pages)
- [ ] Compliance template
- [ ] 2–3 tables (requirement → gap, requirement → evidence)
- [ ] Optional: synthesis flowchart
- [ ] Submission package

---

## CRITICAL SUCCESS FACTORS

### For All Articles
✅ **No scope creep** - Stay within verification (1:1), unit hypersphere, ArcFace/CosFace
✅ **Honest limitations** - Acknowledge dataset/model/demographic constraints
✅ **Reproducible** - Open code, clear methods, public datasets
✅ **Cross-references** - Cite companion articles appropriately

### Article-Specific
**Article A:**
- ✅ Theorem box clearly states falsifiability criterion
- ✅ Experiments are minimal but decisive (2–3 methods)
- ✅ Geometric interpretation is clear

**Article B:**
- ✅ Thresholds are pre-registered (before experiments)
- ✅ Reporting template is complete and usable
- ✅ NO post-hoc threshold adjustment

**Article C:**
- ✅ Written for policy audience (minimal jargon)
- ✅ Regulatory citations are accurate
- ✅ Actionable recommendations

---

## EFFORT ESTIMATE

### Total Effort: 95–121 Hours (8–10 Weeks)

**Article A:** 35–45 hours
- Content extraction: 10–15 hours
- Experiments: 15–20 hours
- Figures/writing: 10 hours

**Article B:** 40–50 hours
- Content extraction: 15–20 hours
- Template creation: 10–15 hours
- Experiments: 10 hours (shared with A)
- Figures/writing: 15 hours

**Article C:** 20–26 hours
- Content extraction: 10–12 hours
- Policy synthesis: 10–14 hours

**Shared work:**
- Experiments (weeks 6–8): Already counted in A & B
- Finalization (weeks 9–10): 10 hours

---

## RISKS & MITIGATION

### Risk 1: Thresholds too strict (no methods pass)
**Mitigation:** Pre-register thresholds based on pilot data or literature; choose realistic values

### Risk 2: Experiments take longer than expected
**Mitigation:** Start experimental setup early (week 6); use public datasets and pretrained models

### Risk 3: Article rejection due to scope
**Mitigation:** Stay within verification (1:1); acknowledge limitations upfront; don't over-claim

### Risk 4: Cross-references break if articles publish out of order
**Mitigation:** Use "companion paper" language in submissions; update citations in revision phase

---

## DIRECTORY STRUCTURE

```
PHD_PIPELINE/
├── ARTICLES_OVERVIEW.md              ← This file (master plan)
│
├── falsifiable_attribution_dissertation/
│   ├── chapters/                     ← SOURCE MATERIAL (DO NOT MODIFY)
│   ├── bibliography/
│   ├── figures/
│   └── ...
│
├── article_A_theory_method/
│   ├── README.md                     ← Quick overview
│   ├── WORKFLOW.md                   ← Detailed workflow
│   ├── manuscript/
│   ├── figures/
│   ├── tables/
│   ├── code/
│   ├── bibliography/
│   └── submission/
│
├── article_B_protocol_thresholds/
│   ├── README.md
│   ├── WORKFLOW.md
│   ├── manuscript/
│   ├── figures/
│   ├── tables/
│   ├── code/
│   ├── bibliography/
│   └── submission/
│
└── article_C_policy_standards/
    ├── README.md
    ├── WORKFLOW.md
    ├── manuscript/
    ├── tables/
    ├── figures/ (optional)
    ├── bibliography/
    └── submission/
```

---

## GETTING STARTED

### Step 1: Choose Starting Point

**Option A: Sequential (recommended if learning process)**
1. Start with Article A (weeks 3–5)
2. Then Article B (weeks 4–6)
3. Finally Article C (weeks 6–10)
4. Run experiments (weeks 6–8)
5. Finalize all (weeks 9–10)

**Option B: Parallel (recommended if time-constrained)**
1. Start all three simultaneously (weeks 3–7)
2. Run experiments (weeks 6–8)
3. Finalize all (weeks 9–10)

### Step 2: Navigate to Article Folder

```bash
cd /home/aaron/projects/xai/PHD_PIPELINE/article_A_theory_method/
# or
cd /home/aaron/projects/xai/PHD_PIPELINE/article_B_protocol_thresholds/
# or
cd /home/aaron/projects/xai/PHD_PIPELINE/article_C_policy_standards/
```

### Step 3: Read Article-Specific Workflow

```bash
cat README.md
cat WORKFLOW.md
```

### Step 4: Begin Content Extraction

Follow the phase-by-phase instructions in each article's `WORKFLOW.md`.

---

## PROGRESS TRACKING

Use TodoWrite tool to track:

### Phase 1: Content Extraction (Weeks 3–7)
- [ ] Article A: Introduction, Background, Theory, Method extracted
- [ ] Article B: Introduction, Background, Protocol, Endpoints, Reporting, Limitations extracted
- [ ] Article C: Introduction, Requirements, Gap, Minimal Evidence, Compliance extracted

### Phase 2: Figures & Tables (Weeks 4–7)
- [ ] Article A: 5 figures created
- [ ] Article B: 4 figures + 3 tables created
- [ ] Article C: 2–3 tables created

### Phase 3: Experiments (Weeks 6–8)
- [ ] Experimental environment set up
- [ ] Counterfactuals generated
- [ ] Δ-prediction tests run
- [ ] CI calibration evaluated
- [ ] Results visualized

### Phase 4: Writing (Weeks 5–9)
- [ ] Article A: Experiments section written, Discussion written
- [ ] Article B: Experiments section written, Discussion written
- [ ] Article C: Discussion written

### Phase 5: Finalization (Weeks 9–10)
- [ ] All articles internally reviewed
- [ ] All submission packages prepared
- [ ] All articles submitted

---

## VENUE SELECTION GUIDE

### Article A (Theory/Method)
**Primary targets:**
- **IJCV (International Journal of Computer Vision)** - Top theory+method venue
- **IEEE TPAMI (Transactions on Pattern Analysis and Machine Intelligence)** - Flagship CV journal

**Choose based on:**
- Emphasis on geometric interpretation → IJCV
- Emphasis on machine learning → TPAMI

### Article B (Protocol/Thresholds)
**Primary targets:**
- **IEEE T-IFS (Transactions on Information Forensics and Security)** - Forensic focus
- **Pattern Recognition** - Methodology focus
- **Forensic Science International: Digital Investigation** - Pure forensic audience

**Choose based on:**
- Forensic use case → T-IFS or FSI-DI
- Broader methodology → Pattern Recognition

### Article C (Policy/Standards)
**Primary targets:**
- **AI & Law** - Legal/regulatory focus
- **Forensic Science Policy & Management** - Forensic policy focus
- **CACM (Communications of the ACM)** - Technical+policy audience

**Choose based on:**
- Legal audience → AI & Law
- Forensic audience → Forensic Science Policy
- Broad CS audience → CACM

---

## FINAL NOTES

### What Makes This Plan Feasible

✅ **Content already exists** in dissertation (85%+ of material)
✅ **Experiments are shared** across Articles A & B (efficiency)
✅ **Clear scoping** (verification only, unit hypersphere only)
✅ **Honest claims** (no over-reach, acknowledge limitations)
✅ **Reproducible** (public datasets, open code)

### What Could Go Wrong

❌ **Scope creep** - Stay disciplined, don't expand to identification (1:N)
❌ **Threshold freezing failure** - Pre-register before experiments, no post-hoc changes
❌ **Over-claiming** - Acknowledge dataset/model/demographic limitations
❌ **Experimental delays** - Start setup early, use pretrained models

### Your Advantage

🎯 **Dissertation is 100% complete** - All source material ready
🎯 **Theory is proven** - Falsifiability criterion is sound
🎯 **Protocol is designed** - Just needs threshold freezing
🎯 **Regulatory analysis is done** - Policy synthesis is straightforward

---

## NEXT ACTIONS

1. **Decide:** Sequential or parallel approach?
2. **Navigate:** `cd article_X/` (choose starting article)
3. **Read:** `cat README.md && cat WORKFLOW.md`
4. **Begin:** Follow phase 1 instructions (content extraction)
5. **Track:** Use TodoWrite tool to monitor progress

---

**Status:** All three article folders are set up and ready.
**Next Step:** Choose starting article (A recommended) and begin content extraction.

**Good luck! You have a complete, systematic plan to extract three journal articles from your dissertation. 🎓📄**
