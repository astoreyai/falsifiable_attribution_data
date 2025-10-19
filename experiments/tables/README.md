# Dissertation Statistical Tables

**Generated:** October 18, 2025
**Status:** Production-Ready
**Data Source:** Real Experimental Results (ZERO SIMULATIONS)

---

## Overview

This directory contains **7 publication-quality LaTeX tables** for the dissertation, generated from real experimental results from Experiments 6.1 through 6.6.

All tables are ready for direct inclusion in the dissertation LaTeX document.

---

## Table Inventory

### Table 6.1: Falsification Rate Comparison
- **File:** `table_6_1.tex`
- **Experiment:** 6.1 - Falsification Rate Comparison
- **Content:** Comparison of FR across 3 attribution methods (Grad-CAM, SHAP, LIME)
- **Columns:** Method | FR (%) | 95% CI | n | p-value | Cohen's h
- **Key Finding:** Grad-CAM has lowest FR (46.1%), but differences not significant
- **Sample Size:** n=200 pairs
- **Data Source:** `/experiments/results_real/exp_6_1/exp_6_1_results_20251018_180300.json`

### Table 6.2: Margin-Stratified Falsification Rate Analysis
- **File:** `table_6_2.tex`
- **Experiment:** 6.2 - Separation Margin Analysis
- **Content:** FR variation across 4 separation margin strata
- **Columns:** Stratum | Margin Range | FR (%) | 95% CI | n
- **Key Finding:** Strong positive correlation (ρ=1.0, p<0.001) between margin and FR
- **Statistical Tests:** Spearman correlation, linear regression (R²=0.985)
- **Strata:**
  - Narrow [0.0, 0.1]: FR = 30.6% (n=14)
  - Moderate [0.1, 0.3]: FR = 35.6% (n=35)
  - Wide [0.3, 0.5]: FR = 44.3% (n=36)
  - Very Wide [0.5, 1.0]: FR = 53.0% (n=115)
- **Data Source:** `/experiments/results_real/exp_6_2/exp_6_2_results_20251018_183607.json`

### Table 6.3: Attribute Falsifiability Rankings
- **File:** `table_6_3.tex`
- **Experiment:** 6.3 - Attribute-Based Validation
- **Content:** Top 10 most falsifiable facial attributes
- **Columns:** Rank | Attribute | Category | FR (%) | 95% CI | n
- **Key Finding:** Expression (Smiling) most falsifiable at 68.7%
- **Hypothesis Test:** H3 supported - 6 of top 10 are occlusion attributes
- **Top 3 Attributes:**
  1. Smiling (Expression): 68.7%
  2. Male (Demographic): 67.3%
  3. Eyeglasses (Occlusion): 65.5%
- **Data Source:** `/experiments/results_real/exp_6_3/exp_6_3_results_20251018_180752.json`

### Table 6.4: Model-Agnostic Testing Results
- **File:** `table_6_4.tex`
- **Experiment:** 6.4 - Model-Agnostic Testing
- **Content:** FR consistency across 3 face recognition models
- **Columns:** Method | ArcFace | CosFace | SphereFace | CV (%) | Model-Agnostic
- **Key Finding:** SHAP is model-agnostic (CV < 15%), Grad-CAM is model-dependent
- **Models Tested:** ArcFace, CosFace, SphereFace
- **Methods Compared:** Grad-CAM vs SHAP
- **Sample Size:** n=500 pairs per model
- **Data Source:** `/experiments/results_real/exp_6_4/exp_6_4_results_20251018_180635.json`

### Table 6.5: Sample Size and Convergence Analysis
- **File:** `table_6_5.tex`
- **Experiment:** 6.5 - Convergence and Sample Size Analysis
- **Content:** Two subtables:
  - (a) Convergence test results
  - (b) Sample size requirements (n=10 to 1000)
- **Key Findings:**
  - H5a CONFIRMED: 97.4% convergence rate (target >95%)
  - H5b VALIDATED: Std follows 1/√n pattern (CLT)
  - Median convergence: 64 iterations (of 100 max)
  - Recommended minimum: n=100 for sufficient power
- **Sample Sizes Tested:** 10, 25, 50, 100, 250, 500, 1000
- **Data Source:** `/experiments/results_real/exp_6_5/exp_6_5_results_20251018_180753.json`

### Table 6.6: Biometric XAI Main Results
- **File:** `table_6_6.tex`
- **Experiment:** 6.6 - Biometric XAI Evaluation
- **Content:** Comparison of standard vs biometric XAI methods
- **Columns:** Method | Standard FR | Biometric FR | Reduction | p-value | Significant
- **Key Finding:** 36.4% average FR reduction (p=0.015, Cohen's d=2.90)
- **Method Pairs:**
  - Grad-CAM: 34.7% → 18.8% (45.8% reduction)
  - SHAP: 35.8% → 21.7% (39.2% reduction)
  - LIME: 45.0% → 34.2% (24.0% reduction)
  - IG: 68.3% → 42.1% (38.4% reduction)
- **Hypothesis:** H6 CONFIRMED - Biometric XAI significantly better
- **Sample Size:** n=1000
- **Data Source:** `/experiments/results_real/exp_6_6/exp_6_6_results_20251018_180753.json`

### Table 6.7: Demographic Fairness Analysis
- **File:** `table_6_7.tex`
- **Experiment:** 6.6 - Biometric XAI Evaluation (Fairness Analysis)
- **Content:** Fairness metrics across gender and age groups
- **Columns:** Method | Gender Gap | Age Gap | DIR_gender | DIR_age | Fair
- **Key Finding:** Biometric methods demonstrate superior fairness
- **Fairness Threshold:** DIR > 0.8 (Disparate Impact Ratio)
- **Standard Methods:** 3 of 4 pass fairness (LIME fails gender fairness)
- **Biometric Methods:** All 4 pass fairness with higher DIR scores
- **Best Performance:** Geodesic IG (DIR_age = 1.00, Age Gap = 0.1%)
- **Data Source:** `/experiments/results_real/exp_6_6/exp_6_6_results_20251018_180753.json`

---

## Usage Instructions

### Including Tables in Dissertation

1. **Copy table file to dissertation directory:**
   ```bash
   cp experiments/tables/table_6_X.tex dissertation/chapters/
   ```

2. **Include in LaTeX document:**
   ```latex
   \input{chapters/table_6_X.tex}
   ```

3. **Or use directly:**
   ```latex
   \input{../experiments/tables/table_6_X.tex}
   ```

### Required LaTeX Packages

All tables use the following packages:

```latex
\usepackage{booktabs}  % For \toprule, \midrule, \bottomrule
\usepackage{array}     % For advanced table formatting
\usepackage{multirow}  % For multirow cells (if needed)
```

---

## Data Provenance

### Experiments Used

| Experiment | Title | Timestamp | Sample Size |
|------------|-------|-----------|-------------|
| 6.1 | Falsification Rate Comparison | 20251018_180300 | n=200 |
| 6.2 | Separation Margin Analysis | 20251018_183607 | n=200 |
| 6.3 | Attribute-Based Validation | 20251018_180752 | n=200 |
| 6.4 | Model-Agnostic Testing | 20251018_180635 | n=500 |
| 6.5 | Convergence & Sample Size | 20251018_180753 | n=10-1000 |
| 6.6 | Biometric XAI Evaluation | 20251018_180753 | n=1000 |

### Data Integrity

- **ZERO SIMULATIONS** - All data from real experiments
- All experimental results stored in `/experiments/results_real/`
- Raw data available in JSON format
- Reproducible via experiment scripts in `/experiments/run_experiment_6_X.py`

---

## Regenerating Tables

To regenerate all tables from experimental results:

```bash
cd /home/aaron/projects/xai/experiments
python3 generate_dissertation_tables.py
```

This will:
1. Read all experimental results from `results_real/`
2. Generate 7 LaTeX tables
3. Save to `tables/` directory
4. Create generation summary

**Output:**
- `table_6_1.tex` through `table_6_7.tex`
- `GENERATION_SUMMARY.txt`

---

## Table Formatting Standards

All tables follow dissertation formatting guidelines:

1. **Caption:** Descriptive title with experiment reference
2. **Label:** Consistent naming `\label{tab:experiment_name}`
3. **Column Headers:** Clear, concise with units
4. **Alignment:** Numeric right-aligned, text left-aligned
5. **Precision:**
   - Percentages: 1 decimal place
   - p-values: 3 decimal places
   - Effect sizes: 2 decimal places
6. **Footnotes:** Explain abbreviations and provide context
7. **Statistical Tests:** Include test type, statistic, and p-value

---

## Quality Assurance

### Checklist for Each Table

- [x] Data sourced from real experiments (not simulated)
- [x] Sample sizes reported accurately
- [x] Confidence intervals at 95% level
- [x] Statistical significance properly indicated
- [x] Effect sizes reported where applicable
- [x] Units clearly specified
- [x] Footnotes explain abbreviations
- [x] Hypothesis test results clearly stated
- [x] LaTeX compiles without errors
- [x] All numbers rounded consistently

---

## Key Statistics Summary

### Overall Dissertation Findings

1. **Best Attribution Method:** Grad-CAM (46.1% FR)
2. **Margin Effect:** FR increases 32.4 percentage points per unit margin
3. **Most Falsifiable Attribute:** Smiling (68.7% FR)
4. **Model-Agnostic Method:** SHAP (CV < 15% across models)
5. **Convergence Rate:** 97.4% (exceeds 95% threshold)
6. **Biometric Improvement:** 36.4% FR reduction (p=0.015)
7. **Fairness Leader:** Biometric methods (all pass DIR > 0.8)

### Hypothesis Test Results

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| H1: Methods differ in FR | PARTIAL | Grad-CAM best, but not significant |
| H2: Margin correlates with FR | CONFIRMED | ρ=1.0, p<0.001, R²=0.985 |
| H3: Occlusion more falsifiable | CONFIRMED | 6 of top 10 are occlusion |
| H4: Methods are model-agnostic | PARTIAL | SHAP yes, Grad-CAM no |
| H5a: Convergence >95% | CONFIRMED | 97.4% convergence rate |
| H5b: Std follows 1/√n | VALIDATED | Empirical matches theoretical |
| H6: Biometric XAI better | CONFIRMED | 36.4% reduction, p=0.015 |

---

## Citation Format

When citing these tables in dissertation text:

```latex
As shown in Table~\ref{tab:falsification_rate_comparison},
Grad-CAM achieved the lowest falsification rate (46.1\%, 95\% CI [39.3, 53.0]).
```

---

## Files in This Directory

```
tables/
├── README.md                    # This file
├── GENERATION_SUMMARY.txt       # Generation log
├── table_6_1.tex               # Falsification Rate Comparison
├── table_6_2.tex               # Margin-Stratified Analysis
├── table_6_3.tex               # Attribute Falsifiability Rankings
├── table_6_4.tex               # Model-Agnostic Testing
├── table_6_5.tex               # Sample Size & Convergence
├── table_6_6.tex               # Biometric XAI Main Results
└── table_6_7.tex               # Demographic Fairness Analysis
```

---

## Version History

- **v1.0** (October 18, 2025) - Initial generation from real experimental results
  - 7 tables created
  - All data from experiments 6.1-6.6
  - ZERO SIMULATIONS

---

## Contact

For questions about table generation or data sources:
- See experiment logs in `/experiments/exp_6_X_real.log`
- Check experimental scripts in `/experiments/run_experiment_6_X.py`
- Review generation script in `/experiments/generate_dissertation_tables.py`

---

**READY FOR DISSERTATION INCLUSION**

All tables are publication-quality and derived from real experimental results.
