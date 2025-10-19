# Article C: From "Meaningful Information" to Testable Explanations

**Full Title:** From "Meaningful Information" to Testable Explanations: Translating AI Act/GDPR/Daubert into XAI Validation for Face Verification

**Target Venues:** AI & Law, Forensic Science Policy & Management, Communications-style venues
**Article Type:** Policy Synthesis / Short Communications
**Timeline:** Weeks 6–10
**Status:** NOT STARTED

---

## OBJECTIVE

Extract and package the **regulatory/evidentiary analysis** from the dissertation into a concise policy synthesis that shows how legal/forensic requirements translate into concrete XAI validation standards.

---

## WHAT'S ALREADY IN THE DISSERTATION

From `falsifiable_attribution_dissertation/chapters/`:

- **Regulatory analysis** (AI Act Arts. 13–15, GDPR Art. 22) - chapter_02_literature_review.md
- **Daubert/Frye analysis** with explicit "requirements → implications" - chapter_02
- **Evidentiary gap** (why current XAI methods fail legal/forensic standards)
- **Compliance pathway** (how falsifiability + protocol close the gap)

---

## ARTICLE STRUCTURE (6–8 pages, short communications format)

### 1. Introduction (1 page)
**Extract from:** dissertation chapter_01_introduction.md + chapter_02 (regulatory section)

**Content:**
- Problem: AI Act/GDPR/Daubert require "meaningful information" and "testable" explanations
- Gap: No clear translation from legal requirements to technical validation
- Current state: XAI deployed without evidence standards
- Contribution: Mapping from regulatory requirements to minimal validation evidence
- Scope: Face verification (1:1) as case study

**New work needed:**
- [ ] Tighten to 1 page
- [ ] Emphasize policy/legal audience (not technical)
- [ ] Forward-reference to requirements → evidence table

### 2. Regulatory & Evidentiary Requirements (2 pages)
**Extract from:** dissertation chapter_02_literature_review.md (regulatory sections)

**Content:**

**Section 2.1: EU AI Act (Arts. 13–15)**
- Art. 13: High-risk systems must provide "meaningful information"
- Art. 14: Accuracy and error rates must be documented
- Art. 15: Systems must be testable
- **Implication**: XAI must be validated, not just produced

**Section 2.2: GDPR (Art. 22)**
- Right to explanation for automated decision-making
- "Meaningful information" about logic involved
- **Implication**: Explanations must be faithful and verifiable

**Section 2.3: Daubert Standard (Forensic Evidence)**
- Testability: Method must be falsifiable
- Error rates: Known and documented
- Peer review: Published and accepted
- Standards: Controlling standards exist
- **Implication**: XAI evidence must meet scientific rigor

**New work needed:**
- [ ] Condense to 2 pages (currently scattered across dissertation chapter 2)
- [ ] Focus on WHAT is required, not HOW to implement (save "how" for Article B reference)
- [ ] Add brief summary of "evidentiary gap" (current XAI methods don't meet these)

### 3. The Evidentiary Gap (1.5 pages)
**Extract from:** dissertation chapter_02_literature_review.md

**Content:**
- **Current XAI practice**: Produce saliency maps, no validation
- **Why this fails requirements**:
  - Not testable (no falsifiable predictions)
  - No error rates (no quantified accuracy)
  - Not peer-reviewed as evidence (published as methods only)
  - No controlling standards (ad-hoc deployment)
- **Table: Requirement → Current Practice → Gap**
- **Implication**: Need for validation standards

**New work needed:**
- [ ] Condense to 1.5 pages
- [ ] Create "Requirement → Current Practice → Gap" table
- [ ] Cite Article A (theory) and Article B (protocol) as closing the gap (if published)

### 4. Minimal Evidence Requirements (2 pages)
**NEW synthesis section**

**Content:**
- **What evidence is minimally required to satisfy AI Act/GDPR/Daubert?**

**Table: Requirement → Minimal Evidence → Validation Method**
```markdown
| Requirement (Source) | Minimal Evidence | Validation Method |
|---------------------|------------------|-------------------|
| "Meaningful information" (GDPR) | Faithful explanation | Δ-prediction test (Article A/B) |
| Testability (Daubert) | Falsifiable criterion | Counterfactual score prediction |
| Error rates (Daubert, AI Act Art. 14) | CI calibration, failure modes | Statistical validation (Article B) |
| Accuracy (AI Act Art. 13) | Quantified prediction accuracy | Correlation threshold (Article B) |
| Standards (Daubert) | Pre-registered protocol | Published acceptance thresholds |
```

- **Explanation**:
  - "Meaningful information" requires explanations that faithfully predict model behavior
  - "Testability" requires falsifiable claims (not just correlation or subjective assessment)
  - "Error rates" require quantified uncertainty (CIs, calibration)
  - "Accuracy" requires pre-registered thresholds (not post-hoc cherry-picking)
  - "Standards" require published protocols (Article B as example)

**New work needed:**
- [ ] Create "Requirement → Evidence → Method" table
- [ ] Write 1-page explanation of each row
- [ ] Reference Articles A & B (if published) or describe generically

### 5. Compliance Template (1.5 pages)
**Extract from:** Article B's forensic reporting template (adapted for policy audience)

**Content:**
- **Purpose**: Show what compliance documentation looks like
- **Template structure** (condensed from Article B):
  1. Method identification
  2. Validation results (Δ-prediction accuracy, CI calibration)
  3. Error rates and failure modes
  4. Limitations disclosure (dataset, model, demographic scope)
  5. Recommendation (evidence-grade vs rejected)
- **Example**: One filled template (simplified for policy audience)
- **Integration**: How this fits into existing forensic/audit workflows

**New work needed:**
- [ ] Adapt Article B's reporting template for policy audience
- [ ] Simplify technical details
- [ ] Emphasize regulatory compliance (not technical implementation)

### 6. Discussion & Policy Implications (1 page)
**Content:**
- **What this enables**: Evidence-grade XAI for regulated/forensic contexts
- **Who benefits**:
  - Regulators: Clear compliance criteria
  - Auditors: Verifiable validation standards
  - Developers: Roadmap from requirements to evidence
  - Courts: Admissible XAI evidence
- **Remaining gaps**:
  - Demographic stratification (future work)
  - Multi-modal systems (beyond face verification)
  - Human-subjects research (IRB-approved studies)
- **Call to action**: Adopt validation standards, publish protocols

**New work needed:**
- [ ] Write fresh (1 page)
- [ ] Emphasize policy/legal implications
- [ ] Avoid over-claiming (acknowledge limits)

---

## EXTRACTION WORKFLOW

### Phase 1: Content Assembly (Week 6)

**Step 1.1:** Create manuscript skeleton
```bash
cd /home/aaron/projects/xai/PHD_PIPELINE/article_C_policy_standards/manuscript
touch article_C_draft.md
```

**Step 1.2:** Extract introduction
- [ ] Copy relevant sections from `../falsifiable_attribution_dissertation/chapters/chapter_01_introduction.md`
- [ ] Copy regulatory motivation from `chapter_02_literature_review.md`
- [ ] Trim to 1 page
- [ ] Emphasize policy audience

**Step 1.3:** Extract regulatory requirements
- [ ] Copy AI Act analysis from `chapter_02_literature_review.md`
- [ ] Copy GDPR analysis from `chapter_02_literature_review.md`
- [ ] Copy Daubert analysis from `chapter_02_literature_review.md`
- [ ] Condense to 2 pages
- [ ] Focus on WHAT is required (not HOW to implement)

**Step 1.4:** Extract evidentiary gap
- [ ] Copy "gap" discussion from `chapter_02_literature_review.md`
- [ ] Condense to 1.5 pages
- [ ] Create "Requirement → Current Practice → Gap" table

**Step 1.5:** Create minimal evidence section (NEW)
- [ ] Synthesize requirements into "Requirement → Evidence → Method" table
- [ ] Write 1-page explanation
- [ ] Reference Articles A & B (or describe generically)

**Step 1.6:** Create compliance template
- [ ] Adapt Article B's reporting template for policy audience
- [ ] Simplify technical details
- [ ] Fill in one example
- [ ] Emphasize regulatory compliance

**Step 1.7:** Write discussion
- [ ] Policy implications (1 page)
- [ ] Who benefits
- [ ] Remaining gaps
- [ ] Call to action

### Phase 2: Tables & Figures (Week 7)

**Step 2.1:** Create "Requirement → Current Practice → Gap" table
```markdown
| Requirement (Source) | Current XAI Practice | Gap |
|---------------------|---------------------|-----|
| "Meaningful information" (GDPR) | Produce saliency maps | No faithfulness validation |
| Testability (Daubert) | Visual inspection | No falsifiable predictions |
| Error rates (Daubert, AI Act) | Not reported | No quantified uncertainty |
| Accuracy (AI Act) | Assumed high | No acceptance thresholds |
| Standards (Daubert) | Ad-hoc deployment | No published protocols |
```

**Step 2.2:** Create "Requirement → Minimal Evidence → Validation Method" table
```markdown
| Requirement (Source) | Minimal Evidence | Validation Method |
|---------------------|------------------|-------------------|
| "Meaningful information" (GDPR) | Faithful explanation | Δ-prediction test |
| Testability (Daubert) | Falsifiable criterion | Counterfactual score prediction |
| Error rates (Daubert, AI Act) | CI calibration, failure modes | Statistical validation |
| Accuracy (AI Act) | Quantified prediction accuracy | Correlation threshold |
| Standards (Daubert) | Pre-registered protocol | Published acceptance thresholds |
```

**Step 2.3:** Create compliance template (simplified from Article B)
```markdown
## Compliance Template: XAI Validation Report

**1. Method**: [e.g., Grad-CAM]
**2. Validation Results**:
   - Δ-prediction accuracy: ρ = 0.82 (threshold: ρ > 0.7) ✓
   - CI calibration: 94% (threshold: 90–100%) ✓
**3. Error Rates**:
   - Known failure modes: [list]
   - Plausibility rejection rate: 15%
**4. Limitations**:
   - Dataset: LFW (not demographically representative)
   - Model: ArcFace only (not all architectures)
   - Scope: Verification (1:1) only
**5. Recommendation**: COMPLIANT (passes all pre-registered thresholds)
```

**Step 2.4:** Create synthesis figure (optional)
- [ ] Flowchart: Regulatory Requirement → Evidentiary Gap → Validation Method → Compliance
- [ ] Visual representation of the argument

### Phase 3: Polish & Review (Weeks 8–10)

**Step 3.1:** Internal consistency check
- [ ] All regulatory citations are accurate
- [ ] Tables are clear and readable
- [ ] No over-claiming (stay within face verification scope)
- [ ] References to Articles A & B are appropriate (or generic if unpublished)
- [ ] Policy implications are realistic

**Step 3.2:** Writing polish
- [ ] Abstract (150–200 words)
- [ ] Keywords (5–7)
- [ ] Policy-friendly language (minimize jargon)
- [ ] Section transitions
- [ ] Citation formatting (check venue requirements)

**Step 3.3:** Audience targeting
- [ ] Written for policy/legal audience (not technical)
- [ ] Technical details kept to minimum (or moved to appendix)
- [ ] Emphasize practical compliance pathway
- [ ] Clear actionable recommendations

**Step 3.4:** Pre-submission checklist
- [ ] Length: 6–8 pages (check venue limits for short communications)
- [ ] Tables: 2–3 total
- [ ] Figures: 0–1 (optional synthesis figure)
- [ ] References: complete and formatted
- [ ] Author affiliations and contact

---

## KEY DESIGN DECISIONS

### What to INCLUDE
✅ Regulatory requirements (AI Act, GDPR, Daubert)
✅ Evidentiary gap (current XAI practice vs requirements)
✅ Minimal evidence requirements (table)
✅ Compliance template (simplified from Article B)
✅ Policy implications and recommendations

### What to EXCLUDE (save for other articles or dissertation)
❌ Detailed technical implementation → Article B
❌ Theoretical proofs → Article A
❌ Experimental results → Articles A & B
❌ Over-claiming beyond face verification

---

## CONTENT MAPPING: DISSERTATION → ARTICLE C

| Article Section | Dissertation Source | Transformation Needed |
|----------------|---------------------|----------------------|
| Introduction | chapter_01 + chapter_02 (intro) | Trim to 1 page, policy focus |
| Requirements | chapter_02 (regulatory sections) | Condense to 2 pages |
| Gap | chapter_02 (gap discussion) | Condense to 1.5 pages + table |
| Minimal Evidence | NEW synthesis | Create table + 1-page explanation |
| Compliance Template | Article B (adapted) | Simplify for policy audience |
| Discussion | NEW | Write fresh, policy implications |

---

## PROGRESS TRACKING

Use `TodoWrite` tool to track:
- [ ] Content extraction completed
- [ ] Tables created (2–3 total)
- [ ] Compliance template created
- [ ] Draft complete (6–8 pages)
- [ ] Internal review passed
- [ ] Ready for submission

---

## DELIVERABLES CHECKLIST

### Manuscript Files
- [ ] `manuscript/article_C_draft.md` (or .tex)
- [ ] `manuscript/abstract.txt`
- [ ] `manuscript/keywords.txt`

### Tables
- [ ] `tables/table1_requirement_gap.csv`
- [ ] `tables/table2_minimal_evidence.csv`

### Templates
- [ ] `manuscript/compliance_template.md`

### Figures (optional)
- [ ] `figures/fig1_synthesis_flowchart.pdf`

### Bibliography
- [ ] `bibliography/article_C_refs.bib`

### Submission Materials
- [ ] `submission/cover_letter.md`
- [ ] `submission/author_contributions.md`
- [ ] `submission/competing_interests.md`

---

## TIMELINE (5 WEEKS)

| Week | Activity | Deliverable |
|------|----------|------------|
| 6 | Extract requirements, gap, create tables | Draft sections 1–4 |
| 7 | Create compliance template, write discussion | Draft sections 5–6 |
| 8 | Polish for policy audience | Complete draft v1 |
| 9 | Internal review, revisions | Complete draft v2 |
| 10 | Final polish, prepare submission | Submission package |

---

## NEXT STEPS

1. **Start here:** Extract regulatory analysis from dissertation chapter 2
2. **Create:** "Requirement → Current Practice → Gap" table
3. **Create:** "Requirement → Minimal Evidence → Method" table
4. **Simplify:** Article B's reporting template for policy audience

---

## NOTES

- **Article C is the POLICY contribution** → prioritize clarity for non-technical audience
- **Stay concise** → 6–8 pages, short communications format
- **Avoid jargon** → explain technical terms or use simpler language
- **Actionable** → provide clear compliance pathway
- **Realistic** → acknowledge limits, don't over-claim

---

## TARGET VENUES

### Primary Targets (Communications/Short Papers)
- **AI & Law** - Policy focus, legal/regulatory audience
- **Forensic Science Policy & Management** - Forensic/evidentiary focus
- **CACM (Communications of the ACM)** - Technical audience with policy interest

### Secondary Targets (Full Papers)
- **Government Information Quarterly** - Public policy focus
- **Computer Law & Security Review** - Legal/technical intersection
- **Science and Public Policy** - Broader policy audience

Choose venue based on:
- Target audience (legal vs technical vs policy)
- Article length (6–8 pages for communications, longer for full papers)
- Topical fit (AI regulation, forensic science, XAI policy)

---

**Status:** Ready to begin content extraction.
**Next Action:** Extract regulatory analysis from dissertation chapter 2.
