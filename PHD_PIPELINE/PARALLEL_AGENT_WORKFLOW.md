# Parallel Agent Workflow for Article Extraction

**Status:** Active
**Created:** October 15, 2025
**Execution Mode:** Parallel autonomous agents
**Coordination:** This document tracks agent outputs and dependencies

---

## WORKFLOW ARCHITECTURE

```
                    COORDINATOR (You)
                           |
           +---------------+---------------+
           |               |               |
    [AGENT A]         [AGENT B]       [AGENT C]
    Theory/Method     Protocol/       Policy/
                      Thresholds      Standards
           |               |               |
           +-------+-------+-------+-------+
                   |               |
            [AGENT EXP]    [AGENT COORD]
            Experiments    Final
            Planning       Integration
```

---

## AGENT ASSIGNMENTS

### Agent 1: Article A Extraction (Theory/Method)
**Task:** Extract and prepare Article A manuscript from dissertation
**Input:**
- `falsifiable_attribution_dissertation/chapters/chapter_03_theory_COMPLETE.md`
- `falsifiable_attribution_dissertation/chapters/chapter_04_methodology_COMPLETE.md`
- `falsifiable_attribution_dissertation/chapters/chapter_01_introduction.md`
- `falsifiable_attribution_dissertation/chapters/chapter_02_literature_review.md`

**Deliverables:**
1. `article_A_theory_method/manuscript/article_A_draft_sections_1-4.md`
   - Section 1: Introduction (1.5 pages)
   - Section 2: Background (2 pages)
   - Section 3: Theory - with BOXED THEOREM (3 pages)
   - Section 4: Method (2 pages)
2. `article_A_theory_method/manuscript/figures_needed.md` - List of 5 required figures with specifications
3. `article_A_theory_method/manuscript/theorem_box.md` - Executive definition + theorem
4. `article_A_theory_method/manuscript/assumptions_box.md` - Unit hypersphere, geodesics, plausibility

**Instructions:**
- Extract content from specified chapters
- Condense to target page counts
- Promote falsifiability criterion to boxed theorem
- Create assumptions box
- Identify figure needs (don't create figures yet)
- Mark placeholders for experimental section (Section 5)
- DO NOT modify dissertation files

**Dependencies:** None (can run immediately)

---

### Agent 2: Article B Extraction (Protocol/Thresholds)
**Task:** Extract and prepare Article B manuscript from dissertation
**Input:**
- `falsifiable_attribution_dissertation/chapters/chapter_04_methodology_COMPLETE.md`
- `falsifiable_attribution_dissertation/chapters/chapter_02_literature_review.md`
- `falsifiable_attribution_dissertation/chapters/chapter_01_introduction.md`

**Deliverables:**
1. `article_B_protocol_thresholds/manuscript/article_B_draft_sections_1-6.md`
   - Section 1: Introduction (2 pages)
   - Section 2: Background - Evidentiary Requirements (2 pages)
   - Section 3: Operational Protocol (4 pages)
   - Section 4: Pre-Registered Endpoints & Thresholds (2 pages) - NEW SYNTHESIS
   - Section 5: Forensic Reporting Template (2 pages) - NEW
   - Section 6: Risk Analysis & Limitations (1.5 pages)
2. `article_B_protocol_thresholds/manuscript/pre_registration.md` - Frozen thresholds with rationale
3. `article_B_protocol_thresholds/manuscript/forensic_template.md` - 7-field reporting template
4. `article_B_protocol_thresholds/manuscript/practitioner_checklist.md` - How to use protocol
5. `article_B_protocol_thresholds/manuscript/figures_tables_needed.md` - List of required figures/tables

**Instructions:**
- Extract operational protocol from chapter 4
- Extract regulatory analysis from chapter 2
- CREATE pre-registration section (synthesize from methodology)
- CREATE forensic reporting template (based on Daubert analysis)
- Freeze thresholds with justification:
  - Δ-score correlation: ρ > 0.7 (example, justify from literature)
  - LPIPS: < 0.3
  - FID: < 50
  - CI calibration: 90-100%
- Mark placeholders for experimental section (Section 7)
- DO NOT modify dissertation files

**Dependencies:** None (can run immediately)

---

### Agent 3: Article C Extraction (Policy/Standards)
**Task:** Extract and prepare Article C manuscript from dissertation
**Input:**
- `falsifiable_attribution_dissertation/chapters/chapter_02_literature_review.md`
- `falsifiable_attribution_dissertation/chapters/chapter_01_introduction.md`

**Deliverables:**
1. `article_C_policy_standards/manuscript/article_C_draft_complete.md`
   - Section 1: Introduction (1 page)
   - Section 2: Regulatory & Evidentiary Requirements (2 pages)
   - Section 3: The Evidentiary Gap (1.5 pages)
   - Section 4: Minimal Evidence Requirements (2 pages) - NEW SYNTHESIS
   - Section 5: Compliance Template (1.5 pages)
   - Section 6: Discussion & Policy Implications (1 page)
2. `article_C_policy_standards/tables/table1_requirement_gap.md`
3. `article_C_policy_standards/tables/table2_minimal_evidence.md`
4. `article_C_policy_standards/manuscript/compliance_template_simplified.md`

**Instructions:**
- Extract regulatory analysis from chapter 2 (AI Act, GDPR, Daubert)
- Condense to policy-friendly language (minimal jargon)
- CREATE "Requirement → Current Practice → Gap" table
- CREATE "Requirement → Minimal Evidence → Validation Method" table
- CREATE simplified compliance template (adapt from Article B's approach)
- Write for policy/legal audience, not technical
- Complete draft (no experimental section needed)
- DO NOT modify dissertation files

**Dependencies:** Soft dependency on Agent 2 (for template), but can run in parallel

---

### Agent 4: Experiments Planning & Setup
**Task:** Design minimal experiments for Articles A & B validation
**Input:**
- `falsifiable_attribution_dissertation/chapters/chapter_04_methodology_COMPLETE.md`
- Article A and B requirements

**Deliverables:**
1. `PHD_PIPELINE/shared_experiments/experiment_plan.md`
   - Dataset: LFW or CASIA-WebFace (specify which and why)
   - Model: Pretrained ArcFace (specify architecture)
   - Attribution methods: 2-3 methods (specify which)
   - Computational requirements
   - Timeline estimate
2. `PHD_PIPELINE/shared_experiments/experiment_setup.py` - Skeleton code
3. `PHD_PIPELINE/shared_experiments/figures_specifications.md`
   - Article A: Scatter plot spec (predicted vs realized Δ-score)
   - Article A: Results table spec
   - Article B: Calibration plot spec
   - Article B: Example reports spec
4. `PHD_PIPELINE/shared_experiments/requirements.txt` - Python dependencies

**Instructions:**
- Design MINIMAL but DECISIVE experiments
- Use public datasets (LFW recommended for reproducibility)
- Use pretrained models (no training needed)
- Specify 2-3 attribution methods (Grad-CAM recommended + 1-2 others)
- Keep computational cost low
- Create skeleton implementation
- Specify all figure/table outputs
- Ensure reproducibility

**Dependencies:** None (can run immediately)

---

## EXECUTION PLAN

### Phase 1: Parallel Content Extraction (Immediate)
Launch Agents 1, 2, 3, 4 simultaneously:
- Agent 1 → Article A extraction
- Agent 2 → Article B extraction
- Agent 3 → Article C extraction (complete draft)
- Agent 4 → Experiments planning

**Timeline:** Agents will run autonomously for 30-60 minutes

### Phase 2: Review Agent Outputs (After agents complete)
Review deliverables from all agents:
- Check manuscript sections for completeness
- Verify frozen thresholds (Agent 2)
- Review experiment plan (Agent 4)
- Identify gaps or issues

### Phase 3: Figure Creation (Coordinated)
Based on specifications from agents:
- Create comparison table (Article A)
- Create geometric figure (Article A)
- Create method flowchart (Article A)
- Create protocol flowchart (Article B)
- Create calibration plot template (Article B)
- Create policy tables (Article C)

### Phase 4: Experiments Execution (Week 6-8)
Run experiments per Agent 4's plan:
- Generate counterfactuals
- Measure Δ-prediction accuracy
- Evaluate CI calibration
- Create result visualizations

### Phase 5: Integration & Polish (Week 9-10)
- Integrate experimental results into Articles A & B
- Write discussion sections
- Polish all three manuscripts
- Prepare submission packages

---

## AGENT COORDINATION RULES

### DO NOT Modify
- `falsifiable_attribution_dissertation/` folder (read-only source)

### DO Create New Content In
- `article_A_theory_method/manuscript/`
- `article_B_protocol_thresholds/manuscript/`
- `article_C_policy_standards/manuscript/`
- `PHD_PIPELINE/shared_experiments/`

### Shared Resources
- Experiments (Agent 4 → Agents 1 & 2)
- Bibliography (each article maintains separate .bib)

### Conflict Resolution
- If agents need same dissertation content → both can extract (no conflict)
- If agents create overlapping content → merge manually in Phase 2

---

## SUCCESS CRITERIA

### After Phase 1 (Content Extraction)
- [ ] Article A: Sections 1-4 complete (6.5 pages)
- [ ] Article A: Theorem box created
- [ ] Article A: Assumptions box created
- [ ] Article B: Sections 1-6 complete (11.5 pages)
- [ ] Article B: Pre-registration document complete
- [ ] Article B: Forensic template created
- [ ] Article B: Practitioner checklist created
- [ ] Article C: Complete draft (6-8 pages)
- [ ] Article C: Both tables created
- [ ] Article C: Compliance template created
- [ ] Experiments: Plan complete
- [ ] Experiments: Setup code skeleton created

### Quality Gates
- All extractions preserve scientific accuracy
- No over-claiming beyond dissertation scope
- Frozen thresholds have justification (Agent 2)
- Policy language is jargon-free (Agent 3)
- Experiments are reproducible (Agent 4)

---

## MONITORING AGENT PROGRESS

### Agent 1 (Article A) Status
- [ ] Started
- [ ] Introduction extracted
- [ ] Background extracted
- [ ] Theory extracted
- [ ] Theorem box created
- [ ] Method extracted
- [ ] Deliverables ready for review

### Agent 2 (Article B) Status
- [ ] Started
- [ ] Introduction extracted
- [ ] Background extracted
- [ ] Protocol extracted
- [ ] Thresholds frozen
- [ ] Template created
- [ ] Deliverables ready for review

### Agent 3 (Article C) Status
- [ ] Started
- [ ] Requirements extracted
- [ ] Gap analysis extracted
- [ ] Tables created
- [ ] Template created
- [ ] Complete draft ready

### Agent 4 (Experiments) Status
- [ ] Started
- [ ] Experiment plan complete
- [ ] Setup code skeleton created
- [ ] Figure specifications complete
- [ ] Requirements documented

---

## DELIVERABLES SUMMARY

### By Agent 1 (Article A)
- Draft sections 1-4 (~6.5 pages)
- Theorem box
- Assumptions box
- Figure specifications list

### By Agent 2 (Article B)
- Draft sections 1-6 (~11.5 pages)
- Pre-registration document
- Forensic reporting template
- Practitioner checklist
- Figure/table specifications list

### By Agent 3 (Article C)
- Complete draft (~6-8 pages)
- 2 policy tables
- Simplified compliance template

### By Agent 4 (Experiments)
- Experiment plan
- Setup code skeleton
- Figure specifications
- Requirements file

---

## NEXT STEPS AFTER AGENT COMPLETION

1. **Review all agent outputs** (30-60 min)
2. **Create figures based on specifications** (2-4 hours)
3. **Write discussion sections** for Articles A & B (1-2 hours)
4. **Run experiments** per Agent 4's plan (weeks 6-8)
5. **Integrate results** into Articles A & B (2-3 hours)
6. **Final polish** all three articles (3-5 hours)
7. **Prepare submission packages** (2-3 hours)

---

## TIMELINE ESTIMATE

- **Phase 1 (Agent Execution):** 30-60 minutes (autonomous)
- **Phase 2 (Review):** 30-60 minutes (manual)
- **Phase 3 (Figures):** 2-4 hours (manual/AI-assisted)
- **Phase 4 (Experiments):** 2-3 weeks (weeks 6-8, as planned)
- **Phase 5 (Integration):** 1 week (week 9-10, as planned)

**Total to draft completion:** ~10 hours of active work + 2-3 weeks of experiments

---

## STATUS TRACKING

**Started:** October 15, 2025 at [TIME]
**Phase:** 1 - Parallel Content Extraction
**Agents Launched:** 0/4
**Agents Completed:** 0/4

**Last Updated:** [Will be updated as agents complete]

---

**This workflow enables rapid parallel extraction of all three journal articles using autonomous specialized agents. Launch agents now to begin.**
