# Table 2: Minimal Evidence Requirements for XAI Compliance

| Requirement (Legal Source) | Minimal Technical Evidence | Validation Method | Acceptance Threshold | Reporting Format |
|---------------------------|---------------------------|-------------------|---------------------|------------------|
| **"Meaningful information about the logic involved"** (GDPR Art. 22) | Faithful attribution map where highlighted regions actually influence model decision | Counterfactual score prediction: Δs_predicted vs. Δs_actual | Pearson ρ ≥ 0.70 between predicted and actual score changes | "Attribution faithfulness: ρ = 0.XX [95% CI: X.XX-X.XX]" |
| **Testability** (Daubert) | Falsifiable hypothesis about feature importance that can be empirically tested | Perturbation experiments with statistical hypothesis testing | p < 0.05 for H₀: attribution is random guessing; effect size Cohen's d ≥ 0.5 | "Explanation testability: χ² = XX, p < 0.001; attributions significantly predict score changes" |
| **"Known or potential error rates"** (Daubert; EU AI Act Art. 14(4)(b)) | (1) Confidence interval calibration for predictions; (2) Documented failure modes | Conformal prediction for CI coverage; stratified evaluation by demographics/conditions | (1) 90-95% coverage for stated confidence intervals; (2) Failure mode documentation complete | "CI calibration: 92% coverage at 90% CI. Known failure modes: [list]. Rejection rate: X%" |
| **"Appropriate accuracy"** (EU AI Act Art. 13(3)(d)) | Quantified explanation accuracy independent of model accuracy | Ground truth test cases where true feature importance is known | Explanation accuracy ≥ 80% on ground truth benchmarks | "Explanation accuracy: 85% correct feature identification [benchmark: controlled perturbation suite]" |
| **Standards** (Daubert) | Pre-registered validation protocol with published acceptance criteria | Peer-reviewed validation study using standardized benchmark | Methods published in peer-reviewed venue; benchmark publicly available | "Validation protocol: [citation]. Benchmark: [name]. Results: [metrics]" |
| **"Accessible and comprehensible information"** (EU AI Act Art. 13(3)(b)(ii)) | Explanation + uncertainty quantification + limitations documentation | User study or expert evaluation of comprehensibility (secondary to technical faithfulness) | Target audience can correctly interpret explanation's meaning and limitations ≥75% of time | "Comprehensibility assessment: XX% correct interpretation by [target audience] in controlled study" |
| **Human oversight** (EU AI Act Art. 14(4)(a)) | Meta-level reliability indicator for each explanation (per-instance quality score) | Prediction confidence calibrated to actual accuracy on held-out validation set | Operator can discriminate between reliable/unreliable explanations with AUC ≥ 0.75 | "Reliability indicator: AUC = 0.XX for predicting explanation error" |

## Operationalization Notes:

### For "Meaningful Information" (GDPR):
- **What it requires**: Explanations must accurately reflect model reasoning, not merely produce plausible-looking heatmaps.
- **How to validate**: Counterfactual score prediction tests whether perturbing attributed regions produces predicted score changes.
- **Minimal threshold**: ρ ≥ 0.70 represents "strong positive correlation" in psychometric literature—attributions correctly predict ≥49% of variance in score changes.

### For Testability (Daubert):
- **What it requires**: The explanation method generates falsifiable predictions that can be experimentally refuted.
- **How to validate**: Perturbation experiments test whether highlighted regions actually influence decisions; statistical tests compare to null hypothesis.
- **Minimal threshold**: p < 0.05 (standard scientific threshold) + effect size d ≥ 0.5 (medium effect) ensures practical significance beyond statistical significance.

### For Error Rates (Daubert + AI Act):
- **What it requires**: Known conditions under which explanations fail or become unreliable.
- **How to validate**: Conformal prediction provides distribution-free confidence intervals; stratified evaluation identifies demographic/pose-specific failures.
- **Minimal threshold**: 90-95% CI coverage is standard in statistical practice; complete documentation means all identifiable failure modes are reported.

### For Accuracy (AI Act):
- **What it requires**: Explanations correctly identify important features, not just correlated features.
- **How to validate**: Ground truth benchmarks with known feature importance (e.g., controlled addition of glasses, makeup).
- **Minimal threshold**: 80% accuracy analogous to forensic science standards in other domains (e.g., fingerprint analysis quality thresholds).

### For Standards (Daubert):
- **What it requires**: Validation follows established protocols with published acceptance criteria, not ad-hoc testing.
- **How to validate**: Peer review of validation methodology; public benchmark enabling independent replication.
- **Minimal threshold**: Publication in peer-reviewed venue ensures methodology scrutiny; public benchmark enables falsifiability.

### For Comprehensibility (AI Act):
- **What it requires**: Target users (forensic analysts, judges, defendants) can understand what explanation communicates.
- **How to validate**: User studies assess interpretation accuracy, not just subjective preference.
- **Minimal threshold**: ≥75% correct interpretation exceeds random chance for most interpretation tasks with ≥3 options.

### For Human Oversight (AI Act):
- **What it requires**: Operators can identify when specific explanations are unreliable and require expert review.
- **How to validate**: Calibration study measuring whether confidence indicators correlate with actual explanation accuracy.
- **Minimal threshold**: AUC ≥ 0.75 represents "acceptable discrimination" in clinical prediction models—operators can distinguish reliable/unreliable cases better than chance.

## Summary Table: Quick Reference

| Legal Requirement | Technical Translation | Validation Method | Pass/Fail Threshold |
|------------------|----------------------|-------------------|---------------------|
| GDPR "meaningful" | Faithful attribution | Counterfactual ρ | ρ ≥ 0.70 |
| Daubert testability | Falsifiable predictions | Perturbation p-value | p < 0.05, d ≥ 0.5 |
| Daubert/AI Act error rates | CI calibration + failure docs | Coverage + completeness | 90-95% coverage |
| AI Act accuracy | Ground truth correctness | Controlled benchmark | ≥80% accuracy |
| Daubert standards | Pre-registered protocol | Peer review + public benchmark | Published + public |
| AI Act comprehensibility | User interpretation | User study | ≥75% correct |
| AI Act oversight | Reliability indicator | Discrimination AUC | AUC ≥ 0.75 |

## Implementation Guidance:

**Minimum Viable Compliance** (for established methods like Grad-CAM):
1. Run counterfactual validation study on representative dataset → report ρ
2. Compute confidence intervals using conformal prediction → report coverage
3. Document known failure modes from validation → list in technical documentation
4. Compare to published ground truth benchmark (or create domain-specific) → report accuracy
5. Publish validation protocol and results → establish standard

**Full Compliance** (for new methods or high-stakes deployment):
1. All minimum viable compliance steps
2. User study with target audience (forensic analysts, legal professionals) → assess comprehensibility
3. Calibration study to develop per-instance reliability indicators → enable operator discrimination
4. Multi-institutional validation → establish generalizability
5. Regular revalidation as models/data evolve → maintain standards

## Gaps Requiring Further Standardization:

1. **Threshold Values**: The thresholds proposed here (ρ ≥ 0.70, 80% accuracy, etc.) are informed by statistical practice and analogous domains, but require community consensus through standards bodies (ISO, NIST).

2. **Ground Truth Benchmarks**: No standardized ground truth benchmark exists for face verification XAI. Community development of shared benchmarks is needed.

3. **Demographic Fairness**: Requirements should extend to include fairness thresholds—e.g., faithfulness ρ must exceed threshold for *all* demographic groups, not just aggregate.

4. **Update Protocols**: Face verification models and XAI methods evolve. Compliance requirements should specify revalidation frequency (e.g., annually, or when model/method changes).

5. **Multi-Method Ensembles**: Guidance needed on whether using multiple XAI methods (Grad-CAM + SHAP) provides additional assurance or merely redundant unreliability.

These gaps represent opportunities for regulatory bodies, standards organizations, and research communities to collaborate on operationalizing legal requirements.
