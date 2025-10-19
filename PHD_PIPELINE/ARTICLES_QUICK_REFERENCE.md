# Quick Reference: Three Journal Articles from Dissertation

**Last Updated:** October 15, 2025
**Status:** Setup Complete, Ready to Begin

---

## AT A GLANCE

| Article | Type | Length | Venues | Timeline | Status |
|---------|------|--------|--------|----------|--------|
| **A** | Theory/Method | 10–12 pp | IJCV, TPAMI | Weeks 3–10 | NOT STARTED |
| **B** | Protocol/Thresholds | 12–15 pp | T-IFS, Pattern Rec | Weeks 4–10 | NOT STARTED |
| **C** | Policy Synthesis | 6–8 pp | AI & Law, CACM | Weeks 6–10 | NOT STARTED |

---

## ARTICLE A: Falsifiable Attribution for Face Verification

**Contribution:** First falsifiable criterion for attributions via counterfactual score prediction

**Location:** `article_A_theory_method/`

**What it extracts from dissertation:**
- Chapter 3 (Theory) → Falsifiability criterion, geometric interpretation
- Chapter 4 (Methodology) → Counterfactual generation algorithm
- NEW: Minimal experiments (2–3 attribution methods)

**Key deliverables:**
- Boxed theorem (falsifiability criterion)
- Geometric figure (unit hypersphere)
- Method flowchart
- Experimental results (scatter plot, results table)
- Code release (falsification test harness)

**Audience:** Computer vision / XAI researchers

**Start here:** Extract introduction from dissertation chapter 1

---

## ARTICLE B: Evidence Thresholds for Explainable Face Verification

**Contribution:** First pre-registered validation protocol with frozen thresholds and forensic reporting template

**Location:** `article_B_protocol_thresholds/`

**What it extracts from dissertation:**
- Chapter 2 (Literature Review) → Regulatory requirements (AI Act/GDPR/Daubert)
- Chapter 4 (Methodology) → Operational protocol, plausibility criteria, limitations
- NEW: Pre-registered endpoints, forensic reporting template, validation experiments

**Key deliverables:**
- Pre-registration document (frozen thresholds)
- Forensic reporting template
- Practitioner checklist
- Protocol flowchart
- Calibration plots
- Code release (protocol implementation)

**Audience:** Practitioners, forensic scientists, auditors

**Start here:** Freeze pre-registered thresholds (CRITICAL FIRST STEP)

---

## ARTICLE C: From "Meaningful Information" to Testable Explanations

**Contribution:** First policy synthesis translating legal requirements into concrete validation standards

**Location:** `article_C_policy_standards/`

**What it extracts from dissertation:**
- Chapter 2 (Literature Review) → Regulatory analysis (AI Act, GDPR, Daubert), evidentiary gap
- NEW: Minimal evidence requirements (synthesized), compliance template (adapted from Article B)

**Key deliverables:**
- Requirement → Gap → Evidence tables (2 tables)
- Compliance template (simplified for policy audience)
- Policy recommendations

**Audience:** Regulators, policy makers, legal professionals

**Start here:** Extract regulatory analysis from dissertation chapter 2

---

## CONTENT SOURCES (QUICK MAP)

### Dissertation Chapter 1 (Introduction)
- **Article A:** Motivation + contribution (1.5 pages)
- **Article B:** Forensic/regulatory motivation (2 pages)
- **Article C:** Policy/legal motivation (1 page)

### Dissertation Chapter 2 (Literature Review)
- **Article A:** XAI background (2 pages)
- **Article B:** Regulatory requirements (2 pages)
- **Article C:** Full regulatory analysis + gap (3.5 pages)

### Dissertation Chapter 3 (Theory)
- **Article A:** Falsifiability criterion (3 pages) ← MAIN SOURCE
- **Article B:** Reference only
- **Article C:** Synthesize into minimal evidence table

### Dissertation Chapter 4 (Methodology)
- **Article A:** Counterfactual generation (2 pages)
- **Article B:** Full protocol (4 pages) + endpoints (2 pages) + limitations (1.5 pages) ← MAIN SOURCE
- **Article C:** Adapt for compliance template

### NEW Content (Write from Scratch)
- **Article A:** Experiments (2.5 pages), Discussion (1 page)
- **Article B:** Pre-registered endpoints (2 pages), Forensic template (2 pages), Experiments (2.5 pages), Discussion (1 page)
- **Article C:** Minimal evidence (2 pages), Discussion (1 page)

---

## TIMELINE (8–10 WEEKS)

### Sequential Approach (Recommended for Learning)
```
Week 3-5:  Article A (extract theory/method, plan experiments)
Week 4-6:  Article B (extract protocol, freeze thresholds)
Week 6-10: Article C (extract policy, write synthesis)
Week 6-8:  EXPERIMENTS (shared across A & B)
Week 9-10: FINALIZE ALL (polish, submit)
```

### Parallel Approach (Recommended for Speed)
```
Week 3-7:  ALL ARTICLES (extract content simultaneously)
Week 6-8:  EXPERIMENTS (shared across A & B)
Week 9-10: FINALIZE ALL (polish, submit)
```

---

## GETTING STARTED

### 1. Choose Your Article
- **Start with Article A?** (Theory/method, most foundational)
- **Start with Article B?** (Protocol/deployment, most practical)
- **Start with Article C?** (Policy, shortest and simplest)

### 2. Navigate to Folder
```bash
cd /home/aaron/projects/xai/PHD_PIPELINE/article_X/
```

### 3. Read Documentation
```bash
cat README.md       # Quick overview
cat WORKFLOW.md     # Detailed step-by-step
```

### 4. Begin Extraction
Follow Phase 1 instructions in `WORKFLOW.md`

---

## CRITICAL SUCCESS FACTORS

### For All Articles
✅ **No scope creep** - Stay in verification (1:1), unit hypersphere, ArcFace/CosFace
✅ **Honest limitations** - Acknowledge dataset/model/demographic constraints
✅ **Reproducible** - Open code, public datasets
✅ **No over-claiming** - Only claim what's proven

### Article-Specific
**Article A:**
- ✅ Boxed theorem clearly states criterion
- ✅ Experiments are minimal but decisive
- ✅ Geometric interpretation is clear

**Article B:**
- ✅ Thresholds pre-registered BEFORE experiments
- ✅ Reporting template is complete and usable
- ✅ NO post-hoc threshold changes

**Article C:**
- ✅ Written for policy audience (minimal jargon)
- ✅ Regulatory citations are accurate
- ✅ Actionable recommendations

---

## EXPERIMENTS (SHARED ACROSS A & B)

### What You Need
- **Dataset:** LFW (public, reproducible)
- **Model:** Pretrained ArcFace (ResNet-50 or similar)
- **Attribution Methods:** 2–3 methods (Grad-CAM, IG, SHAP)
- **Compute:** Modest GPU (experiments are lean)

### What You'll Produce
- **For Article A:**
  - Scatter plot: predicted vs realized Δ-score
  - Results table: method → correlation → pass/fail
- **For Article B:**
  - Calibration plot: nominal vs actual coverage
  - Example reports: one NOT FALSIFIED, one FALSIFIED
  - Validation results table

### Timeline
- **Week 6:** Setup (dataset, model, methods)
- **Week 7:** Run tests (generate counterfactuals, measure Δ-prediction)
- **Week 8:** Visualize (create figures/tables, write results sections)

---

## DELIVERABLES SUMMARY

### Article A (5 figures, 1 table, code)
1. Comparison table (plausibility vs faithfulness vs falsifiability)
2. Geometric figure (unit hypersphere)
3. Method flowchart
4. Δ-prediction scatter plot
5. Plausibility gate figure
6. Results summary table
7. Falsification test harness (code)

### Article B (4 figures, 3 tables, templates, code)
1. Requirement → Gap → Protocol table
2. Protocol flowchart
3. Calibration plot
4. Example reports figure
5. Endpoint thresholds table
6. Validation results table
7. Threats to validity table
8. Forensic reporting template
9. Practitioner checklist
10. Protocol implementation (code)

### Article C (2–3 tables, template)
1. Requirement → Current Practice → Gap table
2. Requirement → Minimal Evidence → Method table
3. Compliance template (simplified)
4. Optional: Synthesis flowchart

---

## WHERE TO GET HELP

- **Master plan:** `/home/aaron/projects/xai/PHD_PIPELINE/ARTICLES_OVERVIEW.md`
- **Article A details:** `article_A_theory_method/README.md` + `WORKFLOW.md`
- **Article B details:** `article_B_protocol_thresholds/README.md` + `WORKFLOW.md`
- **Article C details:** `article_C_policy_standards/README.md` + `WORKFLOW.md`
- **Dissertation source:** `falsifiable_attribution_dissertation/chapters/` (DO NOT MODIFY)

---

## FINAL CHECKLIST

Before submission, each article must have:

### Article A
- [ ] 10–12 pages
- [ ] Boxed theorem
- [ ] All 5 figures (high-res)
- [ ] Experimental results integrated
- [ ] Code released (with LICENSE)
- [ ] No scope creep

### Article B
- [ ] 12–15 pages
- [ ] Thresholds pre-registered
- [ ] Forensic template complete
- [ ] All 4 figures + 3 tables (high-res)
- [ ] Limitations honestly disclosed
- [ ] Code released (with LICENSE)

### Article C
- [ ] 6–8 pages
- [ ] Policy-friendly language
- [ ] All tables clear and accurate
- [ ] Compliance template usable
- [ ] Regulatory citations correct

---

## QUICK START COMMANDS

```bash
# View master plan
cat /home/aaron/projects/xai/PHD_PIPELINE/ARTICLES_OVERVIEW.md

# Start with Article A
cd /home/aaron/projects/xai/PHD_PIPELINE/article_A_theory_method
cat README.md && cat WORKFLOW.md

# Start with Article B
cd /home/aaron/projects/xai/PHD_PIPELINE/article_B_protocol_thresholds
cat README.md && cat WORKFLOW.md

# Start with Article C
cd /home/aaron/projects/xai/PHD_PIPELINE/article_C_policy_standards
cat README.md && cat WORKFLOW.md

# View dissertation source (DO NOT MODIFY)
ls -la /home/aaron/projects/xai/PHD_PIPELINE/falsifiable_attribution_dissertation/chapters/
```

---

## EFFORT ESTIMATE

- **Article A:** 35–45 hours
- **Article B:** 40–50 hours
- **Article C:** 20–26 hours
- **Total:** 95–121 hours (8–10 weeks)

---

## CONFIDENCE LEVEL

**High confidence (90%+)** that all three articles are feasible because:
- ✅ All source material exists in dissertation
- ✅ Theory is complete and sound
- ✅ Protocol is designed and ready
- ✅ Regulatory analysis is done
- ✅ Experiments are lean and reproducible
- ✅ Scope is tightly controlled

---

**You're ready. Pick an article and start extracting. 🚀**
