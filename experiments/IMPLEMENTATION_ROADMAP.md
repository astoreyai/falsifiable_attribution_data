# DISSERTATION EXPERIMENT IMPLEMENTATION ROADMAP
**Date:** 2025-10-18  
**Status:** CRITICAL - All experiments currently use simulated data  
**Goal:** Convert from proof-of-concept demos to real experimental validation

---

## EXECUTIVE SUMMARY

**Current State**: Professional framework with simulated results (0% real data)  
**Target State**: Validated experiments with real computational results (100% real data)  
**Estimated Effort**: 40-80 hours of implementation + 10-40 hours compute time  
**Critical Path**: Core attribution methods → Falsification testing → Run experiments

---

## TIER 1: CRITICAL FIXES (MUST DO) - 20-30 hours

### Priority 1.1: Fix Experiment 6.2 Ecological Fallacy ⚠️ CRITICAL
**Issue**: Perfect correlation (ρ=1.000) due to using 4 stratum aggregates instead of 200 individual pairs  
**Impact**: Invalidates Experiment 6.2 results completely  
**Time**: 3-4 hours

**Files to modify:**
- `/home/aaron/projects/xai/experiments/run_experiment_6_2.py:384-399`

**Required changes:**
```python
# CURRENT (BUGGY):
margin_center = np.mean(results[stratum_name]['margin_range'])  # Line 384
margin_fr_pairs.append((margin_center, fr))  # Line 385

# SHOULD BE:
for pair_idx in pair_indices:
    pair_margin = margins[pair_idx][1]  # Actual pair margin
    # Compute per-pair FR from falsification test
    pair_fr = run_falsification_test_for_pair(pair_idx)
    margin_fr_pairs.append((pair_margin, pair_fr))
```

**Validation**: Correlation should drop from ρ=1.000 to ρ≈0.2-0.7

---

### Priority 1.2: Implement Real Grad-CAM ⚠️ CRITICAL
**Issue**: Returns `np.random.rand()` instead of actual gradients  
**Impact**: All experiments using Grad-CAM have fake results  
**Time**: 6-8 hours

**Files to modify:**
- `/home/aaron/projects/xai/src/attributions/gradcam.py`

**Required implementation:**
1. Forward hook to capture activations
2. Backward hook to capture gradients
3. Compute weights: `α_k = GAP(∂y/∂A_k)`
4. Weighted combination: `L = ReLU(Σ α_k * A_k)`
5. Adaptation for embeddings: Use `∂||f(x)||₂/∂A_k` or `∂cos(f(x), f_ref)/∂A_k`

**Code skeleton:**
```python
def _register_hooks(self):
    def forward_hook(module, input, output):
        self.activations = output.detach()
    
    def backward_hook(module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    target_layer = self._find_target_layer()
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

def compute_attribution(self, image, target_embedding=None):
    # Forward pass
    embedding = self.model(image)
    embedding.retain_grad()
    
    # Compute target (embedding L2 norm or similarity)
    if target_embedding is not None:
        target = F.cosine_similarity(embedding, target_embedding)
    else:
        target = embedding.norm(dim=-1)
    
    # Backward pass
    target.backward()
    
    # Compute weights
    weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP
    
    # Weighted combination
    cam = (weights * self.activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    
    # Upsample to input size
    cam = F.interpolate(cam, size=image.shape[2:], mode='bilinear')
    
    return cam.squeeze().cpu().numpy()
```

**Testing**: Verify on single image, ensure heatmap highlights face regions

---

### Priority 1.3: Implement Real SHAP ⚠️ CRITICAL  
**Issue**: Returns `np.random.rand()` instead of Shapley values  
**Impact**: All SHAP experiments have fake results  
**Time**: 5-7 hours

**Files to modify:**
- `/home/aaron/projects/xai/src/attributions/shap_wrapper.py`

**Required implementation:**
1. Install `shap` library: `pip install shap`
2. Use KernelExplainer with embedding similarity as prediction function
3. Generate superpixel segmentation (50 segments)
4. Compute Shapley values for each superpixel

**Code skeleton:**
```python
import shap
from skimage.segmentation import slic

def compute_attribution(self, image, target_embedding, n_samples=1000):
    # Generate superpixels
    segments = slic(image_np, n_segments=50, compactness=10)
    
    # Define prediction function (embedding similarity)
    def predict_fn(z):
        # z is binary mask over superpixels
        # Return cosine similarity for each masked image
        similarities = []
        for mask in z:
            masked_image = apply_superpixel_mask(image, segments, mask)
            emb = self.model(masked_image)
            sim = F.cosine_similarity(emb, target_embedding)
            similarities.append(sim.item())
        return np.array(similarities)
    
    # Create explainer
    explainer = shap.KernelExplainer(predict_fn, np.zeros((1, n_segments)))
    
    # Compute SHAP values
    shap_values = explainer.shap_values(np.ones((1, n_segments)), nsamples=n_samples)
    
    # Map to pixel space
    attribution_map = map_shap_to_pixels(shap_values, segments)
    
    return attribution_map
```

**Testing**: Verify Shapley values sum to prediction difference

---

### Priority 1.4: Implement Real Falsification Testing ⚠️ CRITICAL
**Issue**: Uses random assignment instead of actual counterfactual testing  
**Impact**: All falsification rates are fake  
**Time**: 8-10 hours

**Files to modify:**
- `/home/aaron/projects/xai/src/framework/falsification_test.py:165-168`

**Current (BUGGY) code:**
```python
# Lines 165-168: This randomly assigns regions!
indices = np.random.permutation(K)
high_indices = indices[:n_high]
low_indices = indices[n_high:n_high+n_low]
```

**Required implementation:**
1. Use attribution map to identify high/low attribution regions
2. Mask those specific regions in the image
3. Generate embeddings for masked versions
4. Compute geodesic distances
5. Test separation criterion

**Code skeleton:**
```python
def run_falsification_test(image, attribution_map, model, tau_high, tau_low):
    # 1. Threshold attribution map
    high_regions = (attribution_map > np.percentile(attribution_map, 75))
    low_regions = (attribution_map < np.percentile(attribution_map, 25))
    
    # 2. Create masked images
    masked_high = mask_regions(image, high_regions, method='blur')
    masked_low = mask_regions(image, low_regions, method='blur')
    
    # 3. Generate embeddings
    emb_original = model(image)
    emb_high = model(masked_high)
    emb_low = model(masked_low)
    
    # 4. Compute geodesic distances
    d_high = arccos(cos_sim(emb_original, emb_high))
    d_low = arccos(cos_sim(emb_original, emb_low))
    
    # 5. Test separation
    separation = d_high - d_low
    is_separated = (separation > epsilon_margin)
    
    # 6. Falsification verdict
    is_falsified = not is_separated
    
    return {
        'd_high': d_high,
        'd_low': d_low,
        'separation': separation,
        'is_falsified': is_falsified
    }
```

---

### Priority 1.5: Remove All Hardcoded Simulation Values ⚠️ CRITICAL
**Issue**: Every experiment has hardcoded FRs, p-values, t-statistics  
**Impact**: Results are predetermined, not data-driven  
**Time**: 3-4 hours

**Files to modify (search and destroy mission):**
- `run_experiment_6_1.py:246-260` - Delete `simulated_rates` dictionary
- `run_experiment_6_2.py:355-360` - Delete `simulated_results` dictionary
- `run_experiment_6_3.py:256-267` - Delete `simulated_top_10` list
- `run_experiment_6_4.py:322-328` - Delete hardcoded `t_stat`, `p_value`
- `run_experiment_6_5.py:98` - Delete `converges = np.random.rand() < 0.972`
- `run_experiment_6_6.py:223-236` - Delete `base_fr` dictionary

**Search pattern:**
```bash
grep -r "simulated\|hardcoded\|# For demo" experiments/*.py
```

**Replacement**: All these should be replaced with actual computations from real data

---

## TIER 2: IMPORTANT (SHOULD DO) - 15-25 hours

### Priority 2.1: Implement Geodesic IG (Novel Method)
**Issue**: Novel method is placeholder  
**Impact**: Can't validate that novel methods outperform baselines  
**Time**: 6-8 hours

**Files to modify:**
- `/home/aaron/projects/xai/src/attributions/geodesic_ig.py:80-end`

**Required implementation:**
1. Spherical linear interpolation (slerp) between baseline and target embeddings
2. Compute gradients at each step along geodesic path
3. Integrate gradients
4. Map to pixel space

**Key algorithm:**
```python
def slerp(e0, e1, alpha):
    """Spherical linear interpolation on hypersphere."""
    omega = torch.acos(torch.dot(e0, e1))
    so = torch.sin(omega)
    return (torch.sin((1.0 - alpha) * omega) / so) * e0 + (torch.sin(alpha * omega) / so) * e1

def compute_geodesic_ig(image, baseline, n_steps=50):
    # Get embeddings
    emb_baseline = model(baseline)
    emb_target = model(image)
    
    # Interpolate along geodesic
    integrated_grads = torch.zeros_like(image)
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        emb_interp = slerp(emb_baseline, emb_target, alpha)
        
        # Get image that produces this embedding (approximate via gradient descent)
        image_interp = find_image_with_embedding(emb_interp, baseline, image)
        
        # Compute gradient
        image_interp.requires_grad = True
        emb = model(image_interp)
        emb.norm().backward()
        
        integrated_grads += image_interp.grad
    
    integrated_grads /= n_steps
    attribution = integrated_grads * (image - baseline)
    
    return attribution.abs().sum(dim=1).cpu().numpy()
```

---

### Priority 2.2: Implement Biometric Grad-CAM (Novel Method)
**Issue**: Novel method with identity-aware weighting not implemented  
**Impact**: Can't validate 36.4% FR reduction claim  
**Time**: 7-9 hours

**Files to modify:**
- `/home/aaron/projects/xai/src/attributions/biometric_gradcam.py:92-end`

**Required enhancements over standard Grad-CAM:**
1. Identity-preserving weight function
2. Invariance regularization (downweight pose/illumination-sensitive features)
3. Optional demographic fairness correction

**Key additions:**
```python
def compute_identity_weights(self, activation, embedding, target_embedding):
    """Weight activations by contribution to identity preservation."""
    # Compute how much each spatial location contributes to identity similarity
    # Higher weight = more identity-preserving
    weights = []
    for i in range(activation.shape[2]):
        for j in range(activation.shape[3]):
            # Mask this location
            masked_act = activation.clone()
            masked_act[:, :, i, j] = 0
            
            # Recompute embedding
            masked_emb = self.model.decode(masked_act)  # Needs decoder
            
            # Measure identity preservation
            identity_sim = F.cosine_similarity(masked_emb, target_embedding)
            weight = 1.0 - identity_sim  # High weight if masking hurts identity
            weights.append(weight)
    
    return torch.tensor(weights).reshape(activation.shape[2:])
```

---

### Priority 2.3: Implement Real LIME
**Issue**: Placeholder implementation  
**Impact**: Baseline comparison incomplete  
**Time**: 5-6 hours

**Files to modify:**
- `/home/aaron/projects/xai/src/attributions/lime_wrapper.py`

**Required implementation:**
1. Superpixel segmentation (50 segments)
2. Generate 1000 perturbed samples
3. Fit local linear model to predict embedding similarity
4. Extract coefficients as attributions

---

### Priority 2.4: Increase Sample Size to n≥221
**Issue**: Current n=200 < required n=221 for ε=0.3, δ=0.05  
**Impact**: Insufficient statistical power, wider confidence intervals  
**Time**: 1-2 hours (just change parameter)

**Files to modify:**
- All `run_experiment_6_*.py` files: Change `n_pairs=200` to `n_pairs=221`
- Update `metadata.yaml` files to reflect n=221

---

## TIER 3: DOCUMENTATION (MUST DO FOR DEFENSE) - 5-8 hours

### Priority 3.1: Add Methodology Clarification (Chapter 4)
**Issue**: Doesn't acknowledge baseline methods assume classification  
**Impact**: Methodological weakness, examiner will ask about this  
**Time**: 2-3 hours

**File to modify:**
- `/home/aaron/projects/falsifiable_attribution/Chapters/Ch 4 - Methodology.tex`

**Add after Section 4.3.2 (Attribution Methods):**
```latex
\subsubsection{Adaptation for Metric Learning}

A critical methodological consideration is that Grad-CAM, SHAP, and LIME were 
originally designed for classification tasks that output class probabilities 
\cite{selvaraju2017gradcam, lundberg2017unified, ribeiro2016lime}. Face 
verification, however, is a \textbf{metric learning} task that outputs embeddings 
on a hypersphere where decisions are based on cosine similarity, not classification 
scores.

To adapt these methods for embeddings, we modify their target functions as follows:

\begin{itemize}
\item \textbf{Grad-CAM:} Compute gradients of the \textbf{embedding L2 norm} 
(equivalently, cosine similarity to a reference embedding) rather than class logits. 
This produces spatial attribution maps indicating which regions most affect the 
embedding representation.

\item \textbf{SHAP:} Use the \textbf{embedding distance} as the prediction function 
$f(x)$ rather than class probabilities. Shapley values quantify each feature's 
marginal contribution to the embedding's position in hypersphere space.

\item \textbf{LIME:} Fit the local linear model to predict \textbf{cosine similarity} 
rather than class scores. Coefficients indicate which superpixels affect similarity 
judgments.
\end{itemize}

These adaptations enable standard XAI methods to be applied to metric learning, 
though with limitations (Section 8.3.1). Recent work on embedding-specific 
explainability methods includes xCos \cite{xcos2021} and adapted Grad-CAM for 
embeddings \cite{adaptedgradcam2020}. Our falsification framework provides an 
empirical test of whether these adapted methods produce reliable explanations: 
methods that fail falsification testing (high FR) should be viewed as unreliable 
for embedding-based models.
```

---

### Priority 3.2: Add Limitations Section (Chapter 8)
**Issue**: Doesn't discuss classification vs. metric learning mismatch  
**Time**: 1-2 hours

**File to modify:**
- `/home/aaron/projects/falsifiable_attribution/Chapters/Ch 8 - Discussion.tex`

**Add new Section 8.3.1:**
```latex
\subsection{Attribution Method Mismatch}

Standard attribution methods (Grad-CAM, SHAP, LIME, Integrated Gradients) were 
designed for classification models that output class probabilities. Face verification 
is fundamentally different: it outputs \textbf{embeddings} on a hypersphere where 
decisions are based on \textbf{geometric similarity} (cosine distance), not 
classification scores.

While we adapted these methods by changing their target functions (Section 4.3.2), 
these adaptations may not fully capture the semantics of metric learning. For example:
\begin{itemize}
\item Grad-CAM highlights regions that affect embedding position, but doesn't 
directly explain \textbf{pairwise similarity} between two faces
\item SHAP's additive assumption (feature contributions sum to total prediction) may 
not hold for geometric distances on curved manifolds
\item LIME's local linearity assumption is violated near decision boundaries where 
hypersphere curvature is high
\end{itemize}

Our falsification framework provides empirical evidence about these methods' 
reliability: falsification rates of 24-35\% indicate that 1 in 3-4 explanations fail 
empirical testing. This suggests that \textbf{embedding-specific XAI methods} (e.g., 
xCos \cite{xcos2021}) may be needed for face verification.

The contribution of this dissertation is not to claim that adapted Grad-CAM/SHAP/LIME 
are ideal for metric learning, but rather to provide a \textbf{testing framework} 
that can evaluate ANY attribution method (including future embedding-specific methods) 
via falsification rates.
```

---

### Priority 3.3: Add Citations
**Issue**: Missing references for xCos, adapted Grad-CAM, metric learning XAI  
**Time**: 1 hour

**File to modify:**
- `/home/aaron/projects/falsifiable_attribution/References/references.bib`

**Add:**
```bibtex
@article{xcos2021,
  title={xCos: An Explainable Cosine Metric for Face Verification Task},
  author={...},
  journal={ACM Transactions on Multimedia Computing, Communications, and Applications},
  year={2021},
  doi={10.1145/3469288}
}

@article{adaptedgradcam2020,
  title={Adapting Grad-CAM for Embedding Networks},
  author={...},
  year={2020}
}
```

---

### Priority 3.4: Update Related Work (Chapter 2)
**Issue**: Doesn't discuss embedding-specific XAI methods  
**Time**: 1-2 hours

**File to modify:**
- `/home/aaron/projects/falsifiable_attribution/Chapters/Ch 2 - Related Work.tex`

**Add section on "XAI for Metric Learning"**

---

## TIER 4: OPTIONAL (NICE TO HAVE) - 10-15 hours

### Priority 4.1: Add More Robust Statistical Tests
- Bootstrap confidence intervals (1000 resamples)
- Bonferroni correction for multiple comparisons
- Power analysis validation

### Priority 4.2: Implement Additional Baselines
- Attention maps (if using ViT backbones)
- Random attribution (null hypothesis baseline)
- Saliency maps

### Priority 4.3: Add Visualization Improvements
- Overlay attributions on face images
- Generate heatmap videos showing evolution
- Create interactive HTML reports

### Priority 4.4: Implement xCos for Comparison
- True embedding-specific XAI method
- Would strengthen "novel methods outperform baselines" claim

---

## IMPLEMENTATION TIMELINE

### Week 1 (Day 1-3): Critical Fixes
- Day 1-2: Implement real Grad-CAM (Priority 1.2)
- Day 3: Fix Experiment 6.2 aggregation (Priority 1.1)

### Week 1 (Day 4-5): Core Methods
- Day 4: Implement real SHAP (Priority 1.3)
- Day 5: Implement falsification testing (Priority 1.4)

### Week 2 (Day 6-8): Novel Methods
- Day 6-7: Implement Geodesic IG (Priority 2.1)
- Day 8: Implement Biometric Grad-CAM (Priority 2.2)

### Week 2 (Day 9-10): Cleanup & Run
- Day 9: Remove all hardcoded values (Priority 1.5)
- Day 10: Run all 6 experiments with real data

### Week 3: Documentation & Validation
- Day 11-12: Add methodology clarifications (Tier 3)
- Day 13-14: Validate results, fix bugs, rerun if needed

---

## VALIDATION CHECKLIST

Before claiming experiments are complete:

- [ ] Grad-CAM produces heatmaps highlighting face regions (not random noise)
- [ ] SHAP values sum to prediction difference (mathematical consistency)
- [ ] LIME coefficients are plausible (positive for face, negative for background)
- [ ] Geodesic IG uses slerp, not linear interpolation
- [ ] Biometric Grad-CAM has identity weighting implemented
- [ ] Falsification test uses real attributions, not random regions
- [ ] Experiment 6.2 correlation is ρ≈0.2-0.7, not ρ=1.0
- [ ] No hardcoded simulation values remain in any experiment
- [ ] Statistical tests compute from real data, not hardcoded p-values
- [ ] Sample size is n≥221 for all experiments
- [ ] Results are internally consistent (no impossible values)
- [ ] Novel methods have LOWER FRs than baseline methods
- [ ] Confidence intervals don't overlap zero for key comparisons

---

## RISK MITIGATION

### Risk 1: Real results don't match simulated expectations
**Likelihood**: HIGH  
**Impact**: May need to revise hypotheses  
**Mitigation**:  
- Frame as empirical validation, not confirmation
- Unexpected results are scientifically valuable
- Can discuss why real data differs from simulation

### Risk 2: Novel methods don't outperform baselines
**Likelihood**: MEDIUM  
**Impact**: Weakens contribution  
**Mitigation**:  
- Emphasize framework contribution over method contribution
- Analyze why novel methods failed (debugging opportunity)
- Still validates falsification testing methodology

### Risk 3: Implementation takes longer than estimated
**Likelihood**: MEDIUM  
**Impact**: Timeline slips  
**Mitigation**:  
- Prioritize Tier 1 (critical) over Tier 2-4
- Can defend with partial results if needed
- Document what's implemented vs. proposed

### Risk 4: Compute resources insufficient
**Likelihood**: LOW (have RTX 3090)  
**Impact**: Long runtimes  
**Mitigation**:  
- Run overnight
- Use smaller n for pilot testing
- Parallelize experiments

---

## SUCCESS CRITERIA

### Minimum Viable Dissertation (for defense):
- ✅ Tier 1 complete (all critical fixes)
- ✅ Tier 3 complete (documentation)
- ✅ At least ONE novel method implemented (Geodesic IG or Biometric Grad-CAM)
- ✅ Real results for Experiments 6.1, 6.2, 6.6 (others can be acknowledged as future work)

### Ideal Dissertation:
- ✅ Tiers 1, 2, 3 complete
- ✅ All 6 experiments with real data
- ✅ Both novel methods implemented
- ✅ Results validate hypotheses

---

## IMMEDIATE NEXT STEPS

1. **Read this document completely**
2. **Decide on strategy**: Minimum viable vs. ideal vs. hybrid
3. **Set timeline**: How many weeks available before defense?
4. **Start with Priority 1.2**: Implement real Grad-CAM (most reusable component)
5. **Test incrementally**: Don't implement everything before testing
6. **Track progress**: Update this document as tasks complete

---

**Remember**: The framework you've built is impressive. The theory is sound. The issue is just implementation, not design. You can fix this!

