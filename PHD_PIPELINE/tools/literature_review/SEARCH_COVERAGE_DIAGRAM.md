# Search Coverage Diagram
## Dissertation: "Falsifiable Attribution for Face Verification"

---

## Research Questions Coverage Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RESEARCH QUESTIONS                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  RQ1: Can attribution techniques satisfy formal falsifiability     │
│       criteria?                                                     │
│                                                                     │
│  RQ2: What are theoretical/empirical limits of attribution         │
│       faithfulness in face verification?                           │
│                                                                     │
│  RQ3: How do current methods (Grad-CAM, IG, SHAP, LIME) perform   │
│       under falsifiability testing?                                │
│                                                                     │
│  RQ4: What constitutes sufficient faithfulness for legal/forensic  │
│       deployment?                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       SEARCH COVERAGE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│  ┃ SEARCH 1: XAI in Face Verification                          ┃  │
│  ┃ ------------------------------------------------------------ ┃  │
│  ┃ Primary search - Core intersection                          ┃  │
│  ┃ Expected: 100-300 papers                                    ┃  │
│  ┃ Addresses: RQ1, RQ2, RQ3, RQ4                              ┃  │
│  ┃ Priority: HIGH                                              ┃  │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
│                                                                     │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│  ┃ SEARCH 7: Falsifiability Gap Analysis                       ┃  │
│  ┃ ------------------------------------------------------------ ┃  │
│  ┃ Novelty gap identification                                  ┃  │
│  ┃ Expected: 5-30 papers (FEW PAPERS = NOVELTY)              ┃  │
│  ┃ Addresses: RQ1                                             ┃  │
│  ┃ Priority: HIGH                                              ┃  │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ SEARCH 2: Specific XAI Techniques                            │  │
│  │ ------------------------------------------------------------ │  │
│  │ Grad-CAM, SHAP, LIME, Integrated Gradients                   │  │
│  │ Expected: 200-500 papers                                     │  │
│  │ Addresses: RQ2, RQ3                                          │  │
│  │ Priority: HIGH                                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ SEARCH 3: Faithfulness Evaluation                            │  │
│  │ ------------------------------------------------------------ │  │
│  │ Counterfactuals, sanity checks, fidelity metrics             │  │
│  │ Expected: 100-250 papers                                     │  │
│  │ Addresses: RQ1, RQ2, RQ3                                     │  │
│  │ Priority: HIGH                                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ SEARCH 4: Face Verification Architectures                    │  │
│  │ ------------------------------------------------------------ │  │
│  │ ArcFace, CosFace, metric learning, embeddings                │  │
│  │ Expected: 150-400 papers                                     │  │
│  │ Addresses: RQ2, RQ3                                          │  │
│  │ Priority: HIGH                                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ SEARCH 5: Legal/Forensic Context                             │  │
│  │ ------------------------------------------------------------ │  │
│  │ GDPR, EU AI Act, wrongful arrests, Daubert standard          │  │
│  │ Expected: 80-200 papers                                      │  │
│  │ Addresses: RQ4                                               │  │
│  │ Priority: MEDIUM                                             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ SEARCH 6: Theoretical Foundations                            │  │
│  │ ------------------------------------------------------------ │  │
│  │ Manifold learning, hypersphere, attribution theory           │  │
│  │ Expected: 50-150 papers                                      │  │
│  │ Addresses: RQ1, RQ2                                          │  │
│  │ Priority: MEDIUM                                             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Research Question Coverage Matrix

```
╔═══════════╦═════╦═════╦═════╦═════╗
║  SEARCH   ║ RQ1 ║ RQ2 ║ RQ3 ║ RQ4 ║
╠═══════════╬═════╬═════╬═════╬═════╣
║ Search 1  ║  ✓  ║  ✓  ║  ✓  ║  ✓  ║ (PRIMARY - ALL RQs)
╠═══════════╬═════╬═════╬═════╬═════╣
║ Search 2  ║     ║  ✓  ║  ✓  ║     ║ (XAI techniques)
╠═══════════╬═════╬═════╬═════╬═════╣
║ Search 3  ║  ✓  ║  ✓  ║  ✓  ║     ║ (Faithfulness)
╠═══════════╬═════╬═════╬═════╬═════╣
║ Search 4  ║     ║  ✓  ║  ✓  ║     ║ (Architectures)
╠═══════════╬═════╬═════╬═════╬═════╣
║ Search 5  ║     ║     ║     ║  ✓  ║ (Legal/forensic)
╠═══════════╬═════╬═════╬═════╬═════╣
║ Search 6  ║  ✓  ║  ✓  ║     ║     ║ (Theory)
╠═══════════╬═════╬═════╬═════╬═════╣
║ Search 7  ║  ✓  ║     ║     ║     ║ (GAP - novelty)
╠═══════════╬═════╬═════╬═════╬═════╣
║ COVERAGE  ║  5  ║  5  ║  4  ║  2  ║ (# searches)
╚═══════════╩═════╩═════╩═════╩═════╝

✓ All RQs covered by multiple searches (redundancy = good)
```

---

## Concept Overlap Diagram

```
                      ┌─────────────────────────┐
                      │   XAI METHODS           │
                      │  (Explainability)       │
                      │                         │
                      │  Search 1, 2, 3, 7      │
                      └─────────┬───────────────┘
                                │
                 ┌──────────────┼──────────────┐
                 │              │              │
     ┌───────────▼────┐  ┌──────▼──────┐  ┌───▼──────────┐
     │  EVALUATION     │  │   FACE      │  │  LEGAL/      │
     │  (Faithfulness) │  │ VERIFICATION│  │  FORENSIC    │
     │                 │  │             │  │              │
     │  Search 3, 6    │  │ Search 1,4  │  │  Search 5    │
     └─────────────────┘  └─────────────┘  └──────────────┘
                                │
                                │
                      ┌─────────▼───────────┐
                      │  FALSIFIABILITY     │
                      │  (Novelty Gap)      │
                      │                     │
                      │  Search 7           │
                      └─────────────────────┘
```

---

## Search Strategy Flow

```
START
  │
  ├─► SEARCH 1 (Primary)
  │   └─► XAI in face verification
  │       └─► Establishes core literature base
  │           Expected: 100-300 papers
  │
  ├─► SEARCH 7 (Gap Analysis) ★ CRITICAL ★
  │   └─► Falsifiability in XAI
  │       └─► Confirms novelty (FEW papers expected)
  │           Expected: 5-30 papers
  │           IF >100 papers → Novelty weakened
  │
  ├─► SEARCH 2 (Methods)
  │   └─► Specific XAI techniques (Grad-CAM, SHAP, LIME, IG)
  │       └─► Foundational literature on methods being tested
  │           Expected: 200-500 papers
  │
  ├─► SEARCH 3 (Evaluation)
  │   └─► Faithfulness, fidelity, counterfactual evaluation
  │       └─► Core evaluation methodology literature
  │           Expected: 100-250 papers
  │
  ├─► SEARCH 4 (Architecture)
  │   └─► ArcFace, CosFace, metric learning
  │       └─► Technical background on models being used
  │           Expected: 150-400 papers
  │
  ├─► SEARCH 5 (Context)
  │   └─► Legal/forensic face recognition
  │       └─► Application motivation (GDPR, wrongful arrests)
  │           Expected: 80-200 papers
  │
  └─► SEARCH 6 (Theory)
      └─► Manifold learning, hypersphere, attribution theory
          └─► Deep theoretical foundations
              Expected: 50-150 papers

TOTAL: 685-1,530 papers (before deduplication)
AFTER DEDUPLICATION: 400-800 unique papers
AFTER SCREENING: 80-150 papers for final review
```

---

## Priority Levels Explained

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  HIGH PRIORITY (Execute First)                 ┃
┃  -------------------------------------------- ┃
┃  Search 1: XAI in face verification           ┃
┃  Search 7: Falsifiability gap                 ┃
┃  Search 2: Specific XAI techniques            ┃
┃  Search 3: Faithfulness evaluation            ┃
┃  Search 4: Face verification architectures    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┌──────────────────────────────────────────────┐
│  MEDIUM PRIORITY (Execute Later)             │
│  -------------------------------------------- │
│  Search 5: Legal/forensic context            │
│  Search 6: Theoretical foundations           │
└──────────────────────────────────────────────┘

HIGH priority = Essential for core research
MEDIUM priority = Important for context/motivation
```

---

## Expected Overlap Between Searches

```
                    Search 1 (XAI + Face)
                    ┌─────────────────┐
                    │                 │
          ┌─────────┼────┐            │
          │ Search 2│    │            │
          │ (Methods)    │            │
          │         │    │            │
          │    ┌────┼────┼─────┐      │
          │    │    │    │     │      │
          └────┼────┘    │ S3  │      │
               │         │(Eval)      │
               │         │     │      │
               │    ┌────┼─────┼──┐   │
               │    │    │     │  │   │
               │    │ S4 │     │  │   │
               └────┼────┘     │  │   │
                    │(Arch)    │  │   │
                    │          │  │   │
                    └──────────┘  │   │
                         S7 (Gap) │   │
                         ┌────────┼───┘
                         │        │
                         └────────┘

Overlap = GOOD (validates search strategy)
Expected overlap: 30-50% of papers
Papers in multiple searches = high relevance
```

---

## Date Range Strategy

```
Timeline of Key Developments
════════════════════════════════════════════════════════════

2014  │ DeepFace, FaceNet published
      │ └─► Start date for Search 4, 6
      │
2015  │ Modern deep face verification established
      │
2016  │ LIME published
      │ └─► Start date for Search 1, 2, 5, 7
      │
2017  │ Grad-CAM, SHAP, Integrated Gradients published
      │ SphereFace published
      │
2018  │ "Sanity Checks for Saliency Maps" published
      │ CosFace published
      │ GDPR enforced
      │ └─► Start date for Search 3
      │
2019  │ ArcFace published
      │
2020  │ Face recognition wrongful arrests (Rekognition)
      │
2021  │ EU AI Act proposed
      │
2024  │ EU AI Act finalized
      │
2025  │ Current year (end date for all searches)
      │
      ▼

RATIONALE:
- Search 3 starts 2018 (XAI evaluation emerged after methods)
- Search 4, 6 start 2014 (modern deep face recognition)
- All others start 2016 (XAI methods + GDPR era)
```

---

## Subject Area Coverage

```
╔═══════════════════════════════════════════════════════════╗
║  COMPUTER SCIENCE (COMP)                                  ║
║  ───────────────────────────────────────────────────────  ║
║  All 7 searches - Core discipline                         ║
╚═══════════════════════════════════════════════════════════╝

┌───────────────────────────────────────────────────────────┐
│  ENGINEERING (ENGI)                                       │
│  ───────────────────────────────────────────────────────  │
│  Searches 1-6 - Applied systems                           │
└───────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│  MATHEMATICS (MATH)                                       │
│  ───────────────────────────────────────────────────────  │
│  Searches 1-4, 6, 7 - Theoretical foundations             │
└───────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│  MULTIDISCIPLINARY (MULT)                                 │
│  ───────────────────────────────────────────────────────  │
│  All 7 searches - Cross-cutting work                      │
└───────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│  SOCIAL SCIENCES (SOCI)                                   │
│  ───────────────────────────────────────────────────────  │
│  Search 5 only - Legal/policy papers                      │
└───────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│  DECISION SCIENCES (DECI)                                 │
│  ───────────────────────────────────────────────────────  │
│  Searches 5, 7 - Ethics/decision-making                   │
└───────────────────────────────────────────────────────────┘
```

---

## Expected Results Funnel

```
SCOPUS SEARCHES
═══════════════════════════════════════════════════════════
Search 1:   100-300 papers    ████████████████████
Search 2:   200-500 papers    ████████████████████████████████████
Search 3:   100-250 papers    ████████████████████
Search 4:   150-400 papers    ██████████████████████████
Search 5:    80-200 papers    ██████████████
Search 6:    50-150 papers    ██████████
Search 7:     5-30 papers     ██
─────────────────────────────────────────────────────────
TOTAL:      685-1,530 papers  ████████████████████████████████████████████
═══════════════════════════════════════════════════════════

                      │
                      │ DEDUPLICATION
                      ▼

UNIQUE PAPERS
═══════════════════════════════════════════════════════════
                400-800 papers  ████████████████████████
═══════════════════════════════════════════════════════════

                      │
                      │ TITLE SCREENING
                      ▼

RELEVANT TITLES
═══════════════════════════════════════════════════════════
                250-450 papers  ███████████████
═══════════════════════════════════════════════════════════

                      │
                      │ ABSTRACT SCREENING
                      ▼

RELEVANT ABSTRACTS
═══════════════════════════════════════════════════════════
                150-300 papers  ██████████
═══════════════════════════════════════════════════════════

                      │
                      │ FULL-TEXT SCREENING
                      ▼

FINAL CORPUS
═══════════════════════════════════════════════════════════
                 80-150 papers  █████
═══════════════════════════════════════════════════════════

                      │
                      │ SYNTHESIS
                      ▼

LITERATURE REVIEW CHAPTER (80,000-100,000 words)
```

---

## Validation Checkpoints

```
CHECKPOINT 1: Query Testing
══════════════════════════════════════════════════════════
[ ] All 7 queries tested in Scopus web interface
[ ] Result counts within expected ranges
[ ] First 20 results from each search are relevant
[ ] Key validation papers captured (see list)
[ ] No Boolean syntax errors

CHECKPOINT 2: Search Execution
══════════════════════════════════════════════════════════
[ ] Search date/time documented for each query
[ ] Exact result count recorded
[ ] Results exported in standard format (RIS/BibTeX)
[ ] Any errors or warnings noted

CHECKPOINT 3: Deduplication
══════════════════════════════════════════════════════════
[ ] All results imported to reference manager
[ ] Automatic deduplication run
[ ] Manual check for missed duplicates
[ ] Deduplication rate calculated (30-50% expected)

CHECKPOINT 4: PRISMA Compliance
══════════════════════════════════════════════════════════
[ ] PRISMA flow diagram completed
[ ] Inclusion/exclusion criteria documented
[ ] Screening decisions justified
[ ] All stages tracked with numbers

CHECKPOINT 5: Quality Assurance
══════════════════════════════════════════════════════════
[ ] Inter-rater reliability calculated (if applicable)
[ ] Random sample re-screened for validation
[ ] Excluded papers documented with reasons
[ ] Data extraction template completed
```

---

## Critical Success Factors

```
✓ Search 1 returns 150-250 papers
  └─► Core literature well-covered

✓ Search 7 returns <30 papers
  └─► Novelty gap confirmed

✓ 30-50% overlap between searches
  └─► Good search strategy validation

✓ Key papers (DeepFace, FaceNet, Grad-CAM, LIME, SHAP, IG) captured
  └─► Comprehensive coverage verified

✓ First 20 results from each search are >80% relevant
  └─► Query quality validated

✗ Search 1 returns <50 papers
  └─► Missing key terms - refine keywords

✗ Search 7 returns >100 papers
  └─► Novelty claim weakened - discuss with advisor

✗ <20% overlap between searches
  └─► Searches too narrow or disconnected - reconsider strategy
```

---

## Next Steps

```
1. VALIDATE QUERIES
   └─► Test all 7 in Scopus web interface
   └─► Check result counts and relevance
   └─► Document any modifications needed

2. EXECUTE SEARCHES
   └─► Run in priority order (1, 7, 2, 3, 4, 5, 6)
   └─► Export results to reference manager
   └─► Document execution details

3. DEDUPLICATE
   └─► Import all results
   └─► Remove duplicates
   └─► Calculate deduplication rate

4. SCREEN
   └─► Title screening → Abstract → Full-text
   └─► Apply inclusion/exclusion criteria
   └─► Complete PRISMA flow diagram

5. SYNTHESIZE
   └─► Extract data to synthesis matrix
   └─► Map to research questions
   └─► Write Chapter 2: Literature Review
```

---

**All searches designed. Ready for execution. 🚀**
