# NOTATION STANDARDIZATION GUIDE

**Agent 4: LaTeX & Quality Agent**
**Date:** October 19, 2025
**Scope:** All dissertation chapters

---

## PURPOSE

Ensure consistent mathematical notation across all 7 chapters to improve readability and professionalism.

---

## STANDARDIZED NOTATION TABLE

### Core Symbols

| Symbol | LaTeX | Meaning | First Defined | Chapters Used |
|--------|-------|---------|---------------|---------------|
| **x** | `\mathbf{x}` | Input face image (vector) | Ch 3 | All |
| **z** | `\mathbf{z}` | Face embedding vector | Ch 3 | All |
| **f** | `f(\cdot)` or `f` | Face verification model | Ch 3 | All |
| **φ** | `\phi(\cdot)` or `\phi` | Embedding function | Ch 3 | All |
| **M** | `\mathcal{M}` | Feature mask (attribution map) | Ch 3 | Ch 3-7 |
| **A** | `A(\cdot)` | Attribution method | Ch 3 | Ch 3-7 |
| **K** | `K` | Number of counterfactuals | Ch 3 | Ch 4-7 |
| **N** | `N` or `n` | Sample size | Ch 4 | Ch 4-7 |
| **T** | `T` | Iteration count | Ch 4 | Ch 4-5 |

### Greek Letters (Thresholds and Margins)

| Symbol | LaTeX | Meaning | First Defined | Note |
|--------|-------|---------|---------------|------|
| **ε** | `\varepsilon` | Separation margin (radians) | Ch 3 | USE varepsilon (not epsilon) |
| **δ** | `\delta` | Target geodesic distance | Ch 4 | Consistent |
| **τ** | `\tau` | Verification threshold | Ch 3 | Consistent |
| **α** | `\alpha` | Learning rate | Ch 4 | Consistent |
| **σ** | `\sigma` | Standard deviation | Ch 6-7 | Consistent |
| **ρ** | `\rho` | Correlation coefficient | Ch 6-7 | Consistent |

### Distances and Metrics

| Symbol | LaTeX | Meaning | First Defined |
|--------|-------|---------|---------------|
| **d_g** | `d_g(\cdot, \cdot)` | Geodesic distance (radians) | Ch 3 |
| **d_E** | `d_E(\cdot, \cdot)` | Euclidean distance | Ch 3 |
| **sim** | `\text{sim}(\cdot, \cdot)` | Cosine similarity | Ch 3 |
| **FR** | `\text{FR}` | Falsification rate | Ch 4 |
| **EER** | `\text{EER}` | Equal error rate | Ch 6 |
| **FAR** | `\text{FAR}` | False accept rate | Ch 6 |
| **FRR** | `\text{FRR}` | False reject rate | Ch 6 |

### Sets and Spaces

| Symbol | LaTeX | Meaning | First Defined |
|--------|-------|---------|---------------|
| **S_high** | `S_{\text{high}}` | High-attribution feature set | Ch 3 |
| **S_low** | `S_{\text{low}}` | Low-attribution feature set | Ch 3 |
| **X** | `\mathcal{X}` | Image space | Ch 3 |
| **Z** | `\mathcal{Z}` | Embedding space | Ch 3 |

---

## INCONSISTENCIES FOUND

### 1. Epsilon Notation (ε)

**Problem:** Mixed usage of `\epsilon` vs `\varepsilon`

**Current State:**
- Chapter 4 uses `\epsilon` in 18 instances (e.g., `\epsilon_{\text{tol}}`, `\epsilon = 0.15`)
- Standard LaTeX typography: `\varepsilon` is preferred for clarity

**Fix:** Replace all `\epsilon` with `\varepsilon` for consistency

**Files to Update:**
```
chapter04.tex (18 instances)
chapter07_results.tex (3 instances)
```

**Search Pattern:**
```bash
grep -rn "\\\\epsilon" chapters/*.tex
```

**Replacement:**
- `\epsilon` → `\varepsilon`
- `\epsilon_{\text{tol}}` → `\varepsilon_{\text{tol}}`

---

### 2. Vector Notation

**Problem:** Some vectors not boldfaced

**Current State:**
- `\mathbf{x}` used consistently (8 occurrences) ✅
- `\mathbf{z}` used only 1 time (likely underused)
- Some instances may use plain `x` or `z` in equations

**Fix:** Verify all vector instances use `\mathbf{}`

**Manual Check Required:** Scan equations for plain `x`, `z`, `\delta` (perturbation)

---

### 3. Function Notation

**Problem:** Inconsistent parentheses in function names

**Current State:**
- `f(\cdot)` (preferred, explicitly shows function)
- `f(x)` (specific evaluation)
- `f` (shorthand, acceptable in prose)

**Fix:** Use `f(\cdot)` when referring to function generally, `f(x)` when evaluating

**No changes needed** (current usage appears correct)

---

### 4. Subscript Text

**Problem:** Mixed usage of `\text{}` in subscripts

**Examples:**
- ✅ Correct: `\varepsilon_{\text{tol}}`, `S_{\text{high}}`, `d_g`
- ❌ Avoid: `\epsilon_{tol}` (non-italic subscripts look wrong)

**Fix:** Ensure all multi-letter subscripts use `\text{}`

**Current Status:** Appears consistent (good)

---

## AUTOMATED CHECKS

### Check 1: Epsilon Consistency

```bash
cd /home/aaron/projects/xai/PHD_PIPELINE/falsifiable_attribution_dissertation/latex/chapters

# Find all epsilon instances (should be varepsilon)
grep -rn "\\\\epsilon" *.tex | grep -v "varepsilon"
```

**Expected Output:** List of lines to fix (approximately 21 instances)

---

### Check 2: Vector Notation

```bash
# Check for non-bold x in equations
grep -rn '\$x\$\|\$x_\|{x}' *.tex | grep -v "\\mathbf{x}" | grep -v "cite" | wc -l
```

**Expected Output:** 0 (if all vectors are boldfaced)

---

### Check 3: Subscript Formatting

```bash
# Check for non-\text subscripts (false positives expected)
grep -rn '_[a-z][a-z]' *.tex | grep -v "\\text{" | head -20
```

**Manual Review Required:** Some false positives (e.g., `d_g` is correct)

---

## PRIORITY FIXES

### High Priority (Must Fix)

1. **Epsilon → Varepsilon** (21 instances)
   - File: `chapter04.tex` (18 instances)
   - File: `chapter07_results.tex` (3 instances)
   - Impact: Typography consistency

**Action:** Use `sed` or manual find-replace

```bash
# Proposed fix (TEST FIRST on backup)
cd /home/aaron/projects/xai/PHD_PIPELINE/falsifiable_attribution_dissertation/latex/chapters

# Backup first
cp chapter04.tex chapter04.tex.backup

# Replace (dry run)
sed 's/\\epsilon/\\varepsilon/g' chapter04.tex | diff - chapter04.tex

# If looks good, apply:
sed -i 's/\\epsilon/\\varepsilon/g' chapter04.tex
sed -i 's/\\epsilon/\\varepsilon/g' chapter07_results.tex
```

---

### Medium Priority (Review Needed)

2. **Vector Boldface Verification**
   - Scan Chapters 3-7 for plain `x`, `z` in equations
   - Verify context (sometimes `x` refers to scalar pixel values, not vectors)
   - Action: Manual review (15 minutes)

---

### Low Priority (Nice to Have)

3. **Function Notation Uniformity**
   - Ensure `f(\cdot)` used consistently when referring to function generally
   - Keep `f(x)` for specific evaluations
   - Action: Spot-check only (appears already consistent)

---

## VERIFICATION CHECKLIST

After applying fixes:

- [ ] Compile LaTeX successfully (no new errors)
- [ ] Check PDF: Does `\varepsilon` render correctly? (should be rounder than `\epsilon`)
- [ ] Grep for remaining `\epsilon` instances (should be 0)
- [ ] Spot-check 5 equations per chapter for vector boldface
- [ ] Verify subscript formatting (all multi-letter subscripts use `\text{}`)

---

## ESTIMATED EFFORT

- **Epsilon fix:** 10 minutes (automated sed)
- **Vector notation check:** 15 minutes (manual review)
- **Verification:** 10 minutes (LaTeX compile + spot-check)
- **Total:** ~35 minutes

---

## IMPACT ON READABILITY

**Before:** Mixed notation (some readers may notice inconsistency)
**After:** Professional, consistent notation (committee approval)

**Defense Benefit:** Shows attention to detail. Committee members who care about typography will notice.

---

## NOTATION REFERENCE CARD (FOR APPENDIX)

**Recommendation:** Add 1-page "Notation Guide" to dissertation appendix

**Template:**
```latex
\chapter*{Notation Guide}
\addcontentsline{toc}{chapter}{Notation Guide}

\section*{Symbols}

\begin{description}
\item[$\mathbf{x}$] Input face image (vector in $\mathbb{R}^{H \times W \times C}$)
\item[$\mathbf{z}$] Face embedding vector (in $\mathbb{R}^d$, typically $d=128$ or $d=512$)
\item[$f(\cdot)$] Face verification model (embedding function $f: \mathcal{X} \to \mathcal{Z}$)
\item[$\mathcal{M}$] Feature mask (attribution map, values in $[0, 1]^{H \times W}$)
\item[$\varepsilon$] Separation margin (radians, used in falsification test thresholds)
\item[$\delta_{\text{target}}$] Target geodesic distance for counterfactual generation
\item[$\tau$] Verification threshold (cosine similarity, typically $\tau = 0.5$)
\item[$d_g(\cdot, \cdot)$] Geodesic distance on hypersphere (radians, $[0, \pi]$)
\item[$\text{sim}(\cdot, \cdot)$] Cosine similarity (range $[-1, 1]$)
\item[$\text{FR}$] Falsification rate (percentage of tests where method fails)
\item[$S_{\text{high}}$] High-attribution feature set (top 20\% by attribution score)
\item[$S_{\text{low}}$] Low-attribution feature set (bottom 20\% by attribution score)
\end{description}

\section*{Conventions}

\begin{itemize}
\item Vectors are typeset in bold: $\mathbf{x}$, $\mathbf{z}$
\item Scalars are italic: $n$, $K$, $\tau$
\item Functions use parentheses: $f(\cdot)$, $A(\cdot)$
\item Sets use calligraphic font: $\mathcal{M}$, $\mathcal{X}$, $\mathcal{Z}$
\item Greek letters: $\varepsilon$ (margin), $\delta$ (target distance), $\tau$ (threshold)
\item Text subscripts use roman font: $S_{\text{high}}$, $\varepsilon_{\text{tol}}$
\end{itemize}
```

**Location to add:** `latex/chapters/appendix_notation.tex`

---

**Report Generated By:** Agent 4 (LaTeX & Quality)
**Status:** Ready for implementation (epsilon fix approved)
**Risk:** Low (sed replacements are safe with backup)
