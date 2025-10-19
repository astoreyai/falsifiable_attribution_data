# Boxed Theorem: Falsifiability Criterion for Attribution Methods

**For prominent display in Section 3 (Theory)**

---

## THEOREM 1: Falsifiability Criterion for Attributions

Let $\phi = A(x)$ be an attribution for input $x$ with feature set $M = \{1, \ldots, m\}$. Define:

$$S_{\text{high}} = \{i \in M : |\phi_i| > \theta_{\text{high}}\}$$
$$S_{\text{low}} = \{i \in M : |\phi_i| < \theta_{\text{low}}\}$$

where $\theta_{\text{high}} > \theta_{\text{low}} > 0$ are pre-defined thresholds.

**The attribution $\phi$ is FALSIFIABLE if and only if:**

### Condition 1: Non-Triviality
$$S_{\text{high}} \neq \emptyset \quad \text{and} \quad S_{\text{low}} \neq \emptyset$$

*Both high-attribution and low-attribution feature sets must be non-empty.*

### Condition 2: Differential Prediction
There exist thresholds $\tau_{\text{high}}, \tau_{\text{low}} \in [0, \pi]$ such that:

$$\mathbb{E}_{x' \sim C(x, S_{\text{high}})} [d_g(f(x), f(x'))] > \tau_{\text{high}}$$

$$\mathbb{E}_{x' \sim C(x, S_{\text{low}})} [d_g(f(x), f(x'))] < \tau_{\text{low}}$$

*High-attribution features cause large geodesic embedding shifts; low-attribution features cause small shifts.*

### Condition 3: Separation Margin
$$\tau_{\text{high}} > \tau_{\text{low}} + \epsilon$$

for some margin $\epsilon > 0$ (typical: $\epsilon = 0.15$ radians ≈ 8.6°).

*The difference must be statistically significant, not due to noise.*

---

## Testable Predictions

A falsifiable attribution makes two empirically testable predictions:

1. **Prediction 1 (High-Attribution):**
   Perturbing high-attribution features should cause geodesic distance $> \tau_{\text{high}}$

2. **Prediction 2 (Low-Attribution):**
   Perturbing low-attribution features should cause geodesic distance $< \tau_{\text{low}}$

These predictions can be **empirically tested** by generating counterfactuals $x' = C(x, S)$ and measuring actual geodesic distances $d_g(f(x), f(x'))$.

If empirical measurements contradict these predictions, the attribution is **falsified** (proven wrong).

---

## Geometric Intuition

Face embeddings lie on a 512-dimensional unit hypersphere $\mathbb{S}^{511}$:

```
         ϕ(x')
          ●
         /|
      d_g |  ← geodesic distance (arc length)
       /  |
      ●---+
    ϕ(x)
```

When we modify features (e.g., mask eyes), the embedding moves along a geodesic arc. The criterion requires:
- **High-attribution features** → large movement ($d_g > \tau_{\text{high}}$)
- **Low-attribution features** → small movement ($d_g < \tau_{\text{low}}$)

This makes the attribution **testable**: measure actual movement and compare to predictions.

---

## Connection to Popper's Falsifiability

Karl Popper [1959] defined **falsifiability** as the demarcation criterion for science:

> *A statement is scientific if and only if it makes testable predictions that could be proven wrong through empirical observation.*

**Examples:**
- ✅ **Falsifiable (Scientific):** "All swans are white" → refuted by observing a black swan
- ❌ **Unfalsifiable (Non-scientific):** "God exists" → no empirical test can refute

**Our Extension to XAI:**
- ✅ **Falsifiable Attribution:** "Masking eyes causes $d_g > 0.5$ radians" → testable via counterfactuals
- ❌ **Unfalsifiable Attribution:** "Eyes are important" → vague, no testable prediction

Theorem 1 formalizes when an attribution satisfies Popper's criterion for XAI.

---

## Typical Parameter Values

Based on ArcFace/CosFace verification systems:

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| $\theta_{\text{high}}$ | 0.7 | 70th percentile of $\|\phi\|$ distribution |
| $\theta_{\text{low}}$ | 0.4 | 40th percentile |
| $\tau_{\text{high}}$ | 0.75 radians | ≈ 43° (significant identity shift) |
| $\tau_{\text{low}}$ | 0.55 radians | ≈ 32° (minimal shift) |
| $\epsilon$ | 0.15 radians | ≈ 8.6° (separation margin) |
| $\delta_{\text{target}}$ | 0.8 radians | ≈ 46° (counterfactual target) |

**Verification context:**
- Same identity: $d_g < 0.6$ radians (cos sim > 0.825)
- Different identity: $d_g > 1.0$ radians (cos sim < 0.540)
- Decision boundary: $d_g \approx 0.8$ radians (cos sim ≈ 0.697)

---

**END OF THEOREM BOX**
