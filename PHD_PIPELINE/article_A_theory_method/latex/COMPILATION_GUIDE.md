# IEEE LaTeX Compilation Guide - Article A

## Quick Compilation

To compile the document with bibliography:

```bash
cd /home/aaron/projects/xai/PHD_PIPELINE/article_A_theory_method/latex

# Full compilation sequence
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `main.pdf` (8 pages, IEEE Transactions format)

## Document Structure

```
main.tex                    ← Main document (IEEE format)
├── sections/
│   ├── 01_introduction.tex     ← Section I
│   ├── 02_related_work.tex     ← Section II
│   ├── 03_theory.tex           ← Section III
│   └── 04_method.tex           ← Section IV
└── references.bib          ← Bibliography database
```

## IEEE Format Details

**Document Class**: `\documentclass[journal]{IEEEtran}`
**Style**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
**Format**: Two-column, 10-point Times Roman
**Current Length**: 8 pages (includes placeholders for Sections V-VI)

## Author Information

**Authors**:
- Aaron W. Storey (Student Member, IEEE)
- Masudul H. Imtiaz (Member, IEEE)

**Affiliation**:
Department of Computer Science
Clarkson University
Potsdam, NY 13699, USA

**Contact**:
- storeyaw@clarkson.edu
- mimtiaz@clarkson.edu

## Title

"Falsifiable Attribution for Face Verification via Counterfactual Score Prediction"

## Keywords

Explainable AI, Face Verification, Attribution Methods, Counterfactual Reasoning, Falsifiability, Biometric Systems

## Current Status

✅ **Complete**: Sections I-IV (Introduction, Related Work, Theory, Method)
📝 **Placeholder**: Section V (Experiments)
📝 **Placeholder**: Section VI (Discussion)
✅ **Complete**: Bibliography compilation
✅ **Complete**: IEEE formatting

## Compilation Notes

1. **First Run**: Generates `main.aux` (auxiliary file with citations)
2. **BibTeX**: Processes `references.bib` and generates `main.bbl` (formatted references)
3. **Second Run**: Incorporates bibliography into document
4. **Third Run**: Resolves all cross-references and finalizes PDF

## Common Issues

### Missing References
**Symptom**: "Citation X undefined" warnings
**Solution**: Run the full compilation sequence (4 commands above)

### Wrong Page Numbers
**Symptom**: Cross-references show "??" or wrong numbers
**Solution**: Run `pdflatex main.tex` one more time

### Bibliography Not Appearing
**Symptom**: No references section
**Solution**:
1. Check `references.bib` exists
2. Run `bibtex main` (must be run from latex/ directory)
3. Run `pdflatex main.tex` twice

## Editing Tips

### Adding Content to Sections
Edit the section files in `sections/` directory:
- `01_introduction.tex` for Section I
- `02_related_work.tex` for Section II
- `03_theory.tex` for Section III
- `04_method.tex` for Section IV

### Adding Figures
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/myimage.png}
\caption{Caption text here.}
\label{fig:myimage}
\end{figure}
```

### Adding Tables
```latex
\begin{table}[t]
\centering
\caption{Table caption here.}
\label{tab:mytable}
\begin{tabular}{lcc}
\toprule
Method & Rate & Time \\
\midrule
Grad-CAM & 0.73 & 4s \\
SHAP & 0.89 & 7min \\
\bottomrule
\end{tabular}
\end{table}
```

### Adding References
Add entries to `references.bib`:
```bibtex
@inproceedings{author2025title,
  title={Paper Title},
  author={Author, First and Author, Second},
  booktitle={Conference Name},
  pages={1--10},
  year={2025}
}
```

Then cite with `\cite{author2025title}` in text.

## Next Steps

1. **Complete Section V (Experiments)**
   - Run experiments
   - Generate figures (4-5 expected)
   - Create tables (3 expected)
   - Write results subsections

2. **Complete Section VI (Discussion)**
   - Interpret findings
   - Discuss forensic implications
   - Acknowledge limitations
   - Propose future work

3. **Add Figures and Tables**
   - Create `figures/` directory
   - Generate plots from results
   - Format tables in IEEE style

4. **Final Proofreading**
   - Check all cross-references
   - Verify citations
   - Spell-check
   - Format check

5. **Submission Preparation**
   - Add author photos to biographies
   - De-anonymize acknowledgments
   - Final compilation
   - Generate submission package

## IEEE Submission Requirements

Typical IEEE Transactions submissions require:
- PDF of manuscript
- LaTeX source files (all .tex files)
- Bibliography file (.bib)
- Figure files (all images)
- Copyright form
- Cover letter

This document is ready for all of the above once Sections V-VI are complete.

## Estimated Timeline

- **Experiments** (Section V): 2-3 weeks
- **Writing** (Section VI): 1 week
- **Figures/Tables**: 1 week (concurrent with above)
- **Proofreading**: 3-5 days
- **Total**: ~4 weeks to submission-ready

## Questions?

See `IEEE_FORMATTING_SUMMARY.md` for detailed information about the formatting conversion and IEEE compliance checklist.
