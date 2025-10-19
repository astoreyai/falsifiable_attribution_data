# Table 1: Regulatory Requirements vs. Current XAI Practice

| Requirement (Legal Source) | Current XAI Practice | Gap | Impact |
|---------------------------|---------------------|-----|--------|
| **"Meaningful information about the logic involved"** (GDPR Art. 22(3), Recital 71) | Visual saliency maps (Grad-CAM, SHAP) produced without validation | No empirical verification that highlighted regions actually influenced decision | Individuals cannot effectively contest algorithmic decisions; explanations may be post-hoc rationalizations |
| **"Appropriate level of transparency" and "accurate, accessible, and comprehensible information"** (EU AI Act Art. 13(3)(b)(i-ii)) | Documentation describes XAI method used (e.g., "explanations generated via Grad-CAM") | No evidence that explanations are *accurate* representations of model reasoning | Operators may misinterpret unreliable explanations, leading to incorrect override decisions |
| **Testability** (Daubert standard, Federal Rule of Evidence 702) | XAI methods produce outputs but lack falsifiable hypotheses | Explanations cannot be empirically tested or refuted through controlled experiments | Fails judicial admissibility standards in U.S. federal courts; cannot demonstrate scientific validity |
| **"Known or potential error rates"** (Daubert; EU AI Act Art. 14(4)(b)) | Error rates reported for face verification accuracy, but not for explanation faithfulness | No quantified failure modes of attribution methods; investigators don't know when explanations are unreliable | Cannot assess reliability of specific explanation; may trust misleading explanations in critical cases |
| **"Standards controlling the technique's operation"** (Daubert) | Ad-hoc deployment of XAI methods without published protocols or acceptance thresholds | No consensus standards for when explanation quality is sufficient for forensic use | Inconsistent practices across agencies; no basis for inter-agency comparison or legal challenges |
| **"Appropriate accuracy"** (EU AI Act Art. 13(3)(d)) | Verification models report accuracy metrics (e.g., 99.7% on LFW), but explanation accuracy is assumed, not measured | Attribution methods may systematically misidentify important facial features (empirical studies show 40-69% accuracy) | High model accuracy does not guarantee explanation reliability; false confidence in forensic applications |
| **Human oversight and "informed decision-making"** (EU AI Act Art. 14(4)(a)) | Human operators review XAI outputs without tools to assess explanation quality | Operators lack meta-information about explanation reliability for specific cases | Cannot distinguish reliable from unreliable explanations; oversight becomes pro forma rather than substantive |

## Key Findings from Gap Analysis:

1. **Validation Gap**: Current practice treats explanation generation and explanation validation as separate concerns. Legal frameworks implicitly require both—explanations must exist (GDPR, AI Act) *and be accurate* (AI Act's "appropriate accuracy" language, Daubert's testability requirement).

2. **Quantification Gap**: Regulatory language demands measurable assurance ("error rates," "accuracy," "reliability"), yet current XAI practice provides qualitative heatmaps without quantitative faithfulness metrics or confidence bounds.

3. **Standards Gap**: Forensic science relies on standardized protocols with documented acceptance criteria (e.g., DNA analysis thresholds). XAI deployment lacks comparable standards—no published thresholds for "good enough" explanation quality.

4. **Operationalization Gap**: Legal requirements use broad terms ("meaningful," "appropriate," "comprehensible") that lack technical operationalization. No consensus exists on how to translate these legal concepts into measurable technical requirements.

5. **Deployment Readiness**: The combination of these gaps means that current XAI practice cannot definitively demonstrate compliance with existing regulatory requirements. Systems may be compliant in form (they produce explanations) but not in substance (explanations may be unreliable).

## Implications:

- **For Regulators**: Need to establish technical standards operationalizing vague legal language (what constitutes "meaningful" information?)
- **For Developers**: Must validate explanation methods, not just deploy them—validation protocols needed
- **For Auditors**: Cannot verify compliance without quantitative faithfulness metrics and published acceptance thresholds
- **For Legal System**: Current XAI evidence may fail Daubert scrutiny due to lack of testability and error rate disclosure
