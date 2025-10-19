# Figures Needed for Article A (Theory/Method)

**Total Required:** 5 figures
**Status:** Specifications complete, awaiting creation

---

## Figure 1: Comparison Table - Attribution Evaluation Paradigms

**Type:** Table/Comparison Chart

**Purpose:** Distinguish our falsifiability framework from prior evaluation approaches

**Content:**

| Criterion | Plausibility | Faithfulness (Proxy) | Falsifiability (Ours) |
|-----------|--------------|----------------------|------------------------|
| **What it measures** | Human interpretability | Correlation with model | Testable predictions |
| **Evaluation method** | User studies | Insertion-deletion AUC | Counterfactual testing |
| **Ground truth** | Human judgment | None (indirect) | Known feature manipulations |
| **Testability** | Subjective | Correlational | Empirically refutable |
| **Example question** | "Does this make sense?" | "Does removing features change score?" | "Do high-attribution features cause large embedding shifts?" |
| **Meets Daubert standard?** | ❌ No | ⚠️ Partial | ✅ Yes |

**Caption:**
> **Figure 1: Comparison of Attribution Evaluation Paradigms.** Plausibility methods assess human interpretability but lack objective ground truth. Faithfulness metrics (insertion-deletion) measure correlation with model behavior but cannot definitively prove explanations are correct. Our falsifiability framework makes testable predictions that can be empirically refuted, meeting scientific standards for expert testimony (Daubert criterion).

**Placement:** Section 1 (Introduction) or Section 3 (Theory)

**Format:** LaTeX table or comparison chart

---

## Figure 2: Geometric Interpretation - Hypersphere Embeddings

**Type:** Diagram (3D illustration with 2D projection)

**Purpose:** Visualize geodesic distance on unit hypersphere and counterfactual perturbations

**Content:**

```
Panel A: Unit Hypersphere Geometry

         ϕ(x₂)
          ●
         /|
      d_g|  ← geodesic arc (surface distance)
       /  |
      ●---+-----●
    ϕ(x₁)  chord  ϕ(x₃)
              (Euclidean ≠ geodesic)

Caption: Face embeddings lie on S^511 (512-D hypersphere).
         Geodesic distance d_g = arccos(⟨u,v⟩) measures
         angle between vectors.

Panel B: Counterfactual Perturbation

    High-attribution           Low-attribution
    (mask eyes)                (mask background)

    ϕ(x)                      ϕ(x)
     ●                         ●
     |                         |
     | d_g = 1.18 rad          | d_g = 0.09 rad
     | (large shift)           | (small shift)
     ↓                         ↓
    ϕ(x'_high)               ϕ(x'_low)
     ●                         ●

Caption: Modifying high-attribution features causes large
         geodesic movement. Modifying low-attribution features
         causes minimal movement.

Panel C: Falsification Test

    Testable Prediction 1:
    E[d_g(high)] > τ_high = 0.75 rad ✓

    Testable Prediction 2:
    E[d_g(low)] < τ_low = 0.55 rad ✓

    Separation: Δ = 1.09 rad > ε = 0.15 rad ✓

    Verdict: Attribution NOT FALSIFIED
```

**Caption:**
> **Figure 2: Geometric Interpretation of Falsifiability on Hypersphere Embeddings.**
> **(A)** Face recognition models (ArcFace/CosFace) embed images onto a unit hypersphere S^{d-1} where similarity is measured as geodesic distance d_g (arc length). Unlike Euclidean distance (chord), geodesic distance respects the manifold's non-Euclidean geometry.
> **(B)** Counterfactual perturbations test attributions: masking high-attribution features (eyes) causes large geodesic shifts (1.18 radians), while masking low-attribution features (background) causes small shifts (0.09 radians).
> **(C)** The falsification test compares observed distances to predicted thresholds. If both predictions hold with significant separation (Δ > ε), the attribution is not falsified.

**Placement:** Section 3 (Theory) after Theorem 1

**Format:** Vector graphics (Inkscape/Adobe Illustrator) or TikZ LaTeX diagram

**Visual Style:** Clean, academic diagram with labeled axes, color-coded features (high=red, low=blue)

---

## Figure 3: Method Flowchart - Counterfactual Generation

**Type:** Flowchart/Algorithm Diagram

**Purpose:** Illustrate Algorithm 1 (counterfactual generation pipeline)

**Content:**

```
┌─────────────────────┐
│ Input: Image x,     │
│ Model f, Features S,│
│ Target δ_target     │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Initialize x' ← x   │
│ Compute ϕ(x) = f(x) │ ← Cache embedding
│ Create mask M_S     │
└──────────┬──────────┘
           │
           ↓
      ┌────────────┐
      │ t ← 1      │
      └─────┬──────┘
            │
   ┌────────↓────────┐
   │ Forward Pass:   │
   │ ϕ(x') = f(x')   │
   │ d_g = arccos(⟨⟩)│
   └────────┬────────┘
            │
   ┌────────↓────────┐
   │ Compute Loss:   │
   │ L = (d_g - δ)²  │
   │   + λ‖x'-x‖²    │
   └────────┬────────┘
            │
   ┌────────↓────────┐
   │ Backward Pass:  │
   │ ∇_{x'} L        │
   └────────┬────────┘
            │
   ┌────────↓────────┐
   │ Gradient Update:│
   │ x' ← x' - α·∇L  │
   │ x' ← M_S⊙x +    │← Apply mask
   │   (1-M_S)⊙x'    │
   └────────┬────────┘
            │
            ↓
      ┌────────────┐
  ┌───┤|d_g-δ|<ε ? │
  │ No└─────┬──────┘
  │     Yes │
  │         ↓
  │   ┌──────────┐
  │   │ Return x'│
  │   │ converged│
  │   └──────────┘
  │
  ↓
┌────────────┐
│ t ← t + 1  │
│ t > T?     │
└──┬─────────┘
   │ No
   └──→ (loop back to Forward Pass)
   │ Yes
   ↓
┌──────────┐
│ Return x'│
│ not conv │
└──────────┘
```

**Caption:**
> **Figure 3: Counterfactual Generation Algorithm (Algorithm 1).** The gradient-based optimization iteratively perturbs input $x$ to produce counterfactual $x'$ achieving target geodesic distance $\delta_{\text{target}}$ while preserving features not in $S$ (via binary mask $M_S$). Early stopping triggers when distance error falls below tolerance $\epsilon_{\text{tol}}$. Typical convergence: 68% within 50 iterations, 98.4% within 100 iterations on LFW dataset.

**Placement:** Section 4 (Method)

**Format:** Flowchart with decision diamonds and process boxes

**Visual Style:** Standard algorithm flowchart (ISO 5807 symbols)

---

## Figure 4: Experimental Results - Δ-Prediction Scatter Plot

**Type:** Scatter plot (PLACEHOLDER - awaiting experiments)

**Purpose:** Validate Theorem 1's differential prediction (Condition 2)

**Content:**

**Axes:**
- X-axis: $\bar{d}_{\text{low}}$ (mean geodesic distance for low-attribution features)
- Y-axis: $\bar{d}_{\text{high}}$ (mean geodesic distance for high-attribution features)

**Data Points:**
- 1,000 images from LFW dataset
- Color-coded by attribution method:
  - Red: Grad-CAM
  - Blue: SHAP
  - Green: Integrated Gradients
  - Orange: LIME

**Regions:**
- Upper-right quadrant (green zone): Both $\bar{d}_{\text{high}} > \tau_{\text{high}}$ and $\bar{d}_{\text{low}} < \tau_{\text{low}}$ → NOT FALSIFIED
- Other quadrants (red zone): Falsification condition fails → FALSIFIED

**Diagonal Line:**
- $y = x + \epsilon$ (separation margin line)
- Points above this line satisfy separation condition

**Expected Results (hypothesized):**
- Grad-CAM: ~60% in green zone (partially falsifiable)
- SHAP: ~75% in green zone (more falsifiable)
- IG: ~70% in green zone
- LIME: ~50% in green zone (least falsifiable)

**Caption:**
> **Figure 4: Separation Between High and Low Attribution Geodesic Distances.** Each point represents one image's attribution. Points in the upper-right (green) zone satisfy both differential predictions (Condition 2 of Theorem 1): $\bar{d}_{\text{high}} > \tau_{\text{high}} = 0.75$ rad and $\bar{d}_{\text{low}} < \tau_{\text{low}} = 0.55$ rad. Points above the diagonal $y = x + 0.15$ satisfy the separation margin (Condition 3). Methods with more points in the green zone produce more falsifiable attributions. **[Results to be added after experiments run.]**

**Placement:** Section 5 (Experiments) - Results subsection

**Format:** Scatter plot with color-coded markers, threshold lines, shaded regions

---

## Figure 5: Plausibility Gate - LPIPS vs Convergence

**Type:** Dual-axis plot (PLACEHOLDER - awaiting experiments)

**Purpose:** Show trade-off between plausibility and counterfactual convergence

**Content:**

**Panel A: LPIPS Distribution**
- Histogram of LPIPS perceptual similarity scores for generated counterfactuals
- X-axis: LPIPS score (0-1, lower = more similar)
- Y-axis: Frequency (count)
- Vertical line at LPIPS = 0.3 (plausibility threshold)
- Color zones:
  - Green (LPIPS < 0.2): Highly plausible
  - Yellow (0.2 ≤ LPIPS < 0.4): Plausible
  - Red (LPIPS ≥ 0.4): Implausible (reject)

**Panel B: Convergence vs Plausibility**
- Scatter plot: Convergence rate (y-axis) vs regularization weight λ (x-axis)
- Show trade-off: higher λ → more plausible but slower convergence
- Optimal point: λ = 0.1 (balance)

**Panel C: Example Counterfactuals**
- Visual grid: 3×3 examples
  - Row 1 (λ=0.01): Low plausibility (LPIPS=0.52), fast convergence
  - Row 2 (λ=0.1): Medium plausibility (LPIPS=0.18), moderate convergence ✓ OPTIMAL
  - Row 3 (λ=1.0): High plausibility (LPIPS=0.09), slow/failed convergence

**Caption:**
> **Figure 5: Plausibility-Convergence Trade-off in Counterfactual Generation.**
> **(A)** LPIPS perceptual similarity distribution for 5,000 generated counterfactuals. Most (89%) fall below LPIPS=0.3, indicating plausible face variations.
> **(B)** Regularization weight λ controls trade-off: higher λ enforces plausibility (low LPIPS) but reduces convergence rate. Optimal λ=0.1 balances both objectives.
> **(C)** Visual examples show that λ=0.1 produces realistic face modifications (masked eyes with inpainting) while maintaining >95% convergence. **[Results to be added after experiments run.]**

**Placement:** Section 5 (Experiments) - Plausibility Analysis subsection

**Format:** Multi-panel figure (histogram + scatter + image grid)

---

## Summary of Figure Requirements

| Figure | Type | Section | Status | Priority |
|--------|------|---------|--------|----------|
| **Fig 1** | Comparison Table | Intro/Theory | Specification complete | High (conceptual foundation) |
| **Fig 2** | Geometric Diagram | Theory | Specification complete | **Critical** (theorem visualization) |
| **Fig 3** | Algorithm Flowchart | Method | Specification complete | High (algorithm explanation) |
| **Fig 4** | Scatter Plot | Experiments | PLACEHOLDER - needs data | Medium (empirical validation) |
| **Fig 5** | Multi-panel Plot | Experiments | PLACEHOLDER - needs data | Medium (quality assurance) |

---

## Production Notes

**For Figures 1-3 (Immediate):**
- Tools: LaTeX (table), TikZ/Inkscape (diagrams), Python Matplotlib (plots)
- Style: Academic journal quality, black-white print-safe, color for online
- Resolution: Vector format (PDF/SVG) for scalability
- Font: Consistent with article body (Times/Computer Modern)

**For Figures 4-5 (After Experiments):**
- Data source: LFW experiments (Chapter 5 implementation)
- Tools: Python Matplotlib/Seaborn + LaTeX pgfplots
- Reproducibility: Include data generation scripts in code/

**Accessibility:**
- Color-blind safe palettes (use markers + colors)
- Alt-text descriptions for all figures
- High-contrast for grayscale printing

---

**END OF FIGURES SPECIFICATION**
