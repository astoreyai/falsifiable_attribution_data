# XAI Validation Compliance Template (Simplified)

**Purpose**: This template enables practitioners to systematically document whether face verification XAI systems meet regulatory requirements (EU AI Act, GDPR, Daubert).

**Instructions**: Complete all sections. For each requirement, provide requested evidence and indicate PASS/FAIL based on specified thresholds.

---

## SYSTEM INFORMATION

| Field | Value |
|-------|-------|
| Face Verification Model | [e.g., ArcFace-ResNet50] |
| XAI Method | [e.g., Grad-CAM, SHAP, Integrated Gradients] |
| Validation Date | [YYYY-MM-DD] |
| Organization | [Name] |
| Validation Lead | [Name, Contact] |

---

## REQUIREMENT 1: MEANINGFUL INFORMATION (GDPR Article 22)

**Legal Requirement**: Provide "meaningful information about the logic involved" in automated decisions.

**Technical Requirement**: Attributions must faithfully represent model reasoning—highlighted regions must actually influence decisions.

### Evidence:

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Counterfactual Correlation (ρ) | [e.g., 0.73] | ρ ≥ 0.70 | [PASS/FAIL] |
| 95% Confidence Interval | [XX.XX - XX.XX] | — | — |
| Validation Sample Size | n = [number] | — | — |

**Validation Method**: [Brief description, e.g., "StyleGAN2-based perturbations measuring predicted vs. actual score changes"]

**Interpretation**: [1-2 sentences explaining what this means]

**Status**: [ ] PASS  [ ] FAIL

---

## REQUIREMENT 2: TESTABILITY (Daubert Standard)

**Legal Requirement**: The method must be testable and have been tested.

**Technical Requirement**: XAI method generates falsifiable predictions that can be experimentally verified or refuted.

### Evidence:

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Statistical Test | χ² = [value], p < [value] | p < 0.05 | [PASS/FAIL] |
| Effect Size (Cohen's d) | [value] | d ≥ 0.5 | [PASS/FAIL] |

**Null Hypothesis Tested**: Attributions are no better than random guessing at predicting score changes.

**Interpretation**: [1-2 sentences]

**Status**: [ ] PASS  [ ] FAIL

---

## REQUIREMENT 3: KNOWN ERROR RATES (Daubert + AI Act Article 14)

**Legal Requirement**: Known or potential error rates must be documented.

**Technical Requirement**: (1) Quantified uncertainty for predictions; (2) Documented failure modes.

### Part A: Uncertainty Quantification

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Confidence Interval Coverage | [XX]% at [90/95]% CI | 90-95% | [PASS/FAIL] |

**Interpretation**: [e.g., "91% coverage at 90% CI indicates well-calibrated uncertainty estimates"]

### Part B: Known Failure Modes

List conditions where explanation faithfulness falls below threshold:

1. **[Condition]**: Faithfulness ρ = [value] [BELOW/ABOVE 0.70]
   - Example: "Profile faces (pose >45°): ρ = 0.54"

2. **[Condition]**: Faithfulness ρ = [value]

3. **[Condition]**: Faithfulness ρ = [value]

4. **[Condition]**: Faithfulness ρ = [value]

5. **[Condition]**: Faithfulness ρ = [value]

**Overall Rejection Rate**: [XX]% of cases fall into documented failure modes

**Status**: [ ] PASS  [ ] FAIL

---

## REQUIREMENT 4: APPROPRIATE ACCURACY (AI Act Article 13)

**Legal Requirement**: System must achieve "an appropriate level of accuracy."

**Technical Requirement**: Explanations correctly identify influential features, measured independently from model prediction accuracy.

### Evidence:

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Ground Truth Accuracy | [XX]% | ≥80% | [PASS/FAIL] |
| Benchmark Used | [name] | — | — |
| Sample Size | n = [number] | — | — |

**Benchmark Description**: [1-2 sentences describing ground truth test cases]

**Interpretation**: [1-2 sentences]

**Status**: [ ] PASS  [ ] FAIL

---

## REQUIREMENT 5: STANDARDS (Daubert Standard)

**Legal Requirement**: Standards must control the technique's operation.

**Technical Requirement**: Validation follows published, peer-reviewed protocols with pre-specified acceptance criteria.

### Evidence:

| Element | Status | Citation/Reference |
|---------|--------|-------------------|
| Peer-Reviewed Protocol | [ ] Yes [ ] No | [Citation or N/A] |
| Public Benchmark | [ ] Yes [ ] No | [URL or N/A] |
| Pre-Registered Thresholds | [ ] Yes [ ] No | [Registration ID or N/A] |

**Overall Standards Compliance**: All three elements must be "Yes" for PASS

**Status**: [ ] PASS  [ ] FAIL

---

## REQUIREMENT 6: COMPREHENSIBILITY (AI Act Article 13)

**Legal Requirement**: Information must be "accessible and comprehensible."

**Technical Requirement**: Target users can correctly interpret what explanations communicate, including limitations.

### Evidence:

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Correct Interpretation Rate | [XX]% | ≥75% | [PASS/FAIL] |
| Target Audience | [e.g., forensic analysts] | — | — |
| Sample Size | n = [number] | — | — |

**User Study Method**: [Brief description]

**Common Misinterpretations**: [List any patterns observed]

**Status**: [ ] PASS  [ ] FAIL

---

## REQUIREMENT 7: HUMAN OVERSIGHT (AI Act Article 14)

**Legal Requirement**: Enable humans to "make informed decisions" and identify performance issues.

**Technical Requirement**: Operators receive per-instance reliability indicators enabling discrimination between reliable/unreliable explanations.

### Evidence:

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Discrimination AUC | [value] | AUC ≥ 0.75 | [PASS/FAIL] |
| 95% Confidence Interval | [XX.XX - XX.XX] | — | — |

**Reliability Indicator Method**: [Brief description of how per-instance scores are computed]

**Interpretation**: [1-2 sentences on operator ability to identify unreliable cases]

**Status**: [ ] PASS  [ ] FAIL

---

## OVERALL COMPLIANCE ASSESSMENT

### Requirements Summary

| Requirement | Status | Notes |
|------------|--------|-------|
| 1. Meaningful Information (GDPR) | [PASS/FAIL] | ρ = [value] |
| 2. Testability (Daubert) | [PASS/FAIL] | p < [value], d = [value] |
| 3. Known Error Rates (Daubert/AI Act) | [PASS/FAIL] | [XX]% coverage, [X] failure modes |
| 4. Appropriate Accuracy (AI Act) | [PASS/FAIL] | [XX]% ground truth accuracy |
| 5. Standards (Daubert) | [PASS/FAIL] | [X/3] elements satisfied |
| 6. Comprehensibility (AI Act) | [PASS/FAIL] | [XX]% correct interpretation |
| 7. Human Oversight (AI Act) | [PASS/FAIL] | AUC = [value] |

**Total**: [X]/7 requirements passed

### Compliance Status

Select one:

- [ ] **FULL COMPLIANCE** (7/7 requirements passed)
  - System meets minimal evidence threshold for forensic deployment

- [ ] **PARTIAL COMPLIANCE** ([X]/7 requirements passed)
  - System meets some but not all requirements
  - Deployment restrictions recommended

- [ ] **NON-COMPLIANT** ([X]/7 requirements passed)
  - System does not meet minimal evidence threshold
  - Additional validation required before deployment

### Deployment Recommendation

Select one and provide justification:

- [ ] **APPROVED FOR OPERATIONAL DEPLOYMENT**
  - Justification: [Explain why full compliance warrants operational use]

- [ ] **APPROVED WITH RESTRICTIONS**
  - Restrictions:
    1. [e.g., "Use only for investigative leads, not primary evidence"]
    2. [e.g., "Require manual expert review for all cases"]
    3. [e.g., "Exclude known failure mode conditions"]
  - Justification: [Explain why partial compliance permits restricted use]

- [ ] **NOT APPROVED**
  - Reasons for non-approval:
    1. [List failed requirements and severity]
  - Required remediation: [Specify what must be addressed]

---

## LIMITATIONS AND CAVEATS

Document all important limitations that constrain interpretation or generalization:

1. [e.g., "Validation conducted on LFW benchmark (celebrity images); generalization to operational surveillance footage requires further study"]

2. [e.g., "User study conducted with forensic analysts only; legal professional and judicial comprehension not yet assessed"]

3. [e.g., "Demographic fairness analysis shows concerning disparities for [group]; requires remediation"]

4. [e.g., "Validation conducted under laboratory conditions; operational deployment introduces additional complexities"]

5. [Additional limitations as needed]

---

## REVALIDATION SCHEDULE

Compliance is not static. Specify when revalidation is required:

- [ ] **Annually** (regardless of system changes)
- [ ] **When model changes** (new training data, architecture updates)
- [ ] **When XAI method changes** (different attribution technique)
- [ ] **When operational conditions change** (new camera systems, different demographics)
- [ ] **After incidents** (explanation-related errors or complaints)
- [ ] **Other**: [Specify trigger conditions]

**Next Scheduled Revalidation**: [YYYY-MM-DD]

---

## RESPONSIBLE PARTIES

| Role | Name | Affiliation | Contact | Signature | Date |
|------|------|-------------|---------|-----------|------|
| Technical Validation Lead | [Name] | [Organization] | [Email] | [Signature] | [Date] |
| Legal/Policy Review | [Name] | [Organization] | [Email] | [Signature] | [Date] |
| Approval Authority | [Name] | [Role/Title] | [Email] | [Signature] | [Date] |

---

## APPENDICES (OPTIONAL)

Attach supporting documentation as needed:

- [ ] Detailed validation protocol
- [ ] Raw validation data
- [ ] Statistical analysis code
- [ ] User study materials and results
- [ ] Peer review documentation
- [ ] Benchmark descriptions
- [ ] Other: [Specify]

---

**END OF TEMPLATE**

---

## FILLED EXAMPLE: Grad-CAM on ArcFace

The following is a complete, realistic example based on empirical findings:

---

## SYSTEM INFORMATION

| Field | Value |
|-------|-------|
| Face Verification Model | ArcFace-ResNet50 (MS1MV2 training) |
| XAI Method | Grad-CAM (layer: conv5_3) |
| Validation Date | 2025-10-15 |
| Organization | Forensic AI Validation Lab, University XYZ |
| Validation Lead | Dr. Jane Smith, jane.smith@university.edu |

---

## REQUIREMENT 1: MEANINGFUL INFORMATION (GDPR Article 22)

### Evidence:

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Counterfactual Correlation (ρ) | 0.68 | ρ ≥ 0.70 | FAIL |
| 95% Confidence Interval | 0.63 - 0.73 | — | — |
| Validation Sample Size | n = 1,000 | — | — |

**Validation Method**: StyleGAN2-based plausibility-preserving perturbations measuring predicted vs. actual score changes

**Interpretation**: Grad-CAM attributions explain 46% of variance (r²=0.46) in score changes. Predictive accuracy is moderate but falls marginally below threshold for "strong" correlation.

**Status**: [X] FAIL  [ ] PASS

---

## REQUIREMENT 2: TESTABILITY (Daubert Standard)

### Evidence:

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Statistical Test | χ² = 184.3, p < 0.001 | p < 0.05 | PASS |
| Effect Size (Cohen's d) | 0.72 | d ≥ 0.5 | PASS |

**Null Hypothesis Tested**: Attributions are no better than random guessing

**Interpretation**: Grad-CAM significantly outperforms random baseline with medium-to-large effect size, demonstrating testability.

**Status**: [X] PASS  [ ] FAIL

---

## REQUIREMENT 3: KNOWN ERROR RATES (Daubert + AI Act Article 14)

### Part A: Uncertainty Quantification

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Confidence Interval Coverage | 91% at 90% CI | 90-95% | PASS |

**Interpretation**: Well-calibrated uncertainty estimates

### Part B: Known Failure Modes

1. **Profile faces (pose >45°)**: Faithfulness ρ = 0.54 BELOW 0.70
2. **Low resolution (<100px interocular)**: Faithfulness ρ = 0.59 BELOW 0.70
3. **Occlusion >30% of face**: Faithfulness ρ = 0.62 BELOW 0.70
4. **Dark-skinned females**: Faithfulness ρ = 0.64 BELOW 0.70
5. **Scores near threshold (0.45-0.55)**: Faithfulness ρ = 0.61 BELOW 0.70

**Overall Rejection Rate**: 23% of LFW pairs fall into documented failure modes

**Status**: [X] PASS  [ ] FAIL

---

## REQUIREMENT 4: APPROPRIATE ACCURACY (AI Act Article 13)

### Evidence:

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Ground Truth Accuracy | 76% | ≥80% | FAIL |
| Benchmark Used | Controlled Perturbation Suite | — | — |
| Sample Size | n = 500 | — | — |

**Benchmark Description**: Test cases with known ground truth (controlled glasses, makeup, aging effects)

**Interpretation**: Grad-CAM correctly identified important features in 76% of cases, falling short of forensic accuracy standard (80%).

**Status**: [X] FAIL  [ ] PASS

---

## REQUIREMENT 5: STANDARDS (Daubert Standard)

### Evidence:

| Element | Status | Citation/Reference |
|---------|--------|-------------------|
| Peer-Reviewed Protocol | [X] Yes | Selvaraju et al., ICCV 2017 |
| Public Benchmark | [X] Yes | LFW dataset + Controlled Perturbation Suite |
| Pre-Registered Thresholds | [ ] No | Thresholds established post-hoc |

**Overall Standards Compliance**: 2/3 elements satisfied (missing pre-registration)

**Status**: [ ] PASS  [X] FAIL

---

## REQUIREMENT 6: COMPREHENSIBILITY (AI Act Article 13)

### Evidence:

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Correct Interpretation Rate | 83% | ≥75% | PASS |
| Target Audience | Forensic analysts | — | — |
| Sample Size | n = 24 | — | — |

**User Study Method**: Controlled study with practitioners from 3 agencies interpreting Grad-CAM outputs

**Common Misinterpretations**: Over-interpreting absolute attribution magnitudes rather than relative importance rankings

**Status**: [X] PASS  [ ] FAIL

---

## REQUIREMENT 7: HUMAN OVERSIGHT (AI Act Article 14)

### Evidence:

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Discrimination AUC | 0.71 | AUC ≥ 0.75 | FAIL |
| 95% Confidence Interval | 0.66 - 0.76 | — | — |

**Reliability Indicator Method**: Gradient magnitude-based confidence score

**Interpretation**: Provided reliability scores show weak discrimination; operators struggle to identify problematic cases.

**Status**: [X] FAIL  [ ] PASS

---

## OVERALL COMPLIANCE ASSESSMENT

### Requirements Summary

| Requirement | Status | Notes |
|------------|--------|-------|
| 1. Meaningful Information (GDPR) | FAIL | ρ = 0.68 (below 0.70) |
| 2. Testability (Daubert) | PASS | p < 0.001, d = 0.72 |
| 3. Known Error Rates (Daubert/AI Act) | PASS | 91% coverage, 5 failure modes documented |
| 4. Appropriate Accuracy (AI Act) | FAIL | 76% (below 80%) |
| 5. Standards (Daubert) | FAIL | 2/3 elements (no pre-registration) |
| 6. Comprehensibility (AI Act) | PASS | 83% correct interpretation |
| 7. Human Oversight (AI Act) | FAIL | AUC = 0.71 (below 0.75) |

**Total**: 3/7 requirements passed

### Compliance Status

- [ ] FULL COMPLIANCE
- [X] **PARTIAL COMPLIANCE** (3/7 requirements passed)
- [ ] NON-COMPLIANT

### Deployment Recommendation

- [ ] APPROVED FOR OPERATIONAL DEPLOYMENT
- [X] **APPROVED WITH RESTRICTIONS**
  - Restrictions:
    1. Use only for investigative leads, NOT as primary evidence in legal proceedings
    2. Require manual expert review by trained forensic examiners for ALL cases
    3. Exclude known failure mode conditions: profile faces, low resolution, occlusion, dark-skinned females, borderline scores
    4. Provide operators with automated warnings when cases fall into failure modes
    5. Document all uses and maintain audit trail for quality assurance review
  - Justification: System demonstrates testability and has documented error rates, making it suitable for investigative use under supervision. However, failures on faithfulness, accuracy, and oversight metrics preclude use as primary evidence.

- [ ] NOT APPROVED

---

## LIMITATIONS AND CAVEATS

1. Validation conducted on LFW (celebrity images from web); generalization to operational surveillance footage (lower quality, different demographics, challenging conditions) requires additional study.

2. Faithfulness marginally below threshold (ρ=0.68 vs. 0.70); with larger validation set or improved perturbation methods, may reach threshold.

3. Pre-registration not performed for this validation; future studies should pre-specify thresholds before data collection to prevent post-hoc optimization.

4. Demographic analysis reveals concerning disparity for dark-skinned females (ρ=0.64); requires urgent remediation before broader deployment to ensure equitable treatment.

5. User study conducted with forensic analysts only; comprehension by judges, attorneys, and defendants (who also encounter these explanations in legal contexts) not yet assessed.

---

## REVALIDATION SCHEDULE

- [X] **Annually** (regardless of system changes)
- [X] **When model changes** (new training data, architecture updates)
- [X] **When XAI method changes** (different attribution technique)
- [X] **When operational conditions change** (new camera systems, different demographics)
- [X] **After incidents** (explanation-related errors or complaints)

**Next Scheduled Revalidation**: 2026-10-15

---

## RESPONSIBLE PARTIES

| Role | Name | Affiliation | Contact | Signature | Date |
|------|------|-------------|---------|-----------|------|
| Technical Validation Lead | Dr. Jane Smith | University XYZ | jane.smith@xyz.edu | [Signature] | 2025-10-15 |
| Legal/Policy Review | John Doe, Esq. | Agency Legal Counsel | john.doe@agency.gov | [Signature] | 2025-10-15 |
| Approval Authority | Chief Maria Garcia | Forensic Division CTO | maria.garcia@agency.gov | [Signature] | 2025-10-15 |

---

**END OF EXAMPLE**
