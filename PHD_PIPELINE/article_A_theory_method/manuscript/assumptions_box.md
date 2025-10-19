# Assumptions for Falsifiability Framework

**For display in Section 3 (Theory) or Appendix**

---

## Formal Assumptions

The falsifiability criterion (Theorem 1) and counterfactual generation algorithm (Algorithm 1) rely on the following assumptions:

---

### Assumption 1: Unit-Norm Embeddings (Hypersphere Geometry)

**Statement:**
The face recognition model $f: \mathcal{X} \to \mathbb{S}^{d-1}$ maps inputs to the unit hypersphere:

$$\mathbb{S}^{d-1} = \{u \in \mathbb{R}^d : \|u\|_2 = 1\}$$

All embeddings satisfy $\|\phi(x)\|_2 = \|f(x)\|_2 = 1$ (L2-normalized).

**Satisfied By:**
- ✅ ArcFace [Deng2019]
- ✅ CosFace [Wang2018]
- ✅ SphereFace [Liu2017]
- ✅ Any model with explicit L2 normalization layer before cosine similarity

**NOT Satisfied By:**
- ❌ FaceNet with triplet loss (unnormalized embeddings in Euclidean space) [Schroff2015]
- ❌ VGGFace with softmax classification (no embedding normalization)
- ❌ Models using Euclidean distance instead of cosine similarity

**Justification:**
The geodesic distance metric $d_g(u, v) = \arccos(\langle u, v \rangle)$ is only well-defined on the unit hypersphere. Without L2 normalization, embeddings lie in Euclidean space $\mathbb{R}^d$, requiring different geometric treatment.

**Verification:**
Check model architecture for final layer normalization:
```python
# PyTorch example (ArcFace)
x = self.backbone(img)  # ResNet features
x = F.normalize(x, p=2, dim=1)  # L2 normalize → hypersphere
```

---

### Assumption 2: Geodesic Metric on Hypersphere

**Statement:**
Verification decisions are based on **geodesic distance** $d_g$ (or equivalently, cosine similarity):

$$d_g(u, v) = \arccos(\langle u, v \rangle) \quad \text{(radians)}$$

where $\langle u, v \rangle = \sum_{i=1}^d u_i v_i$ is the inner product.

**Equivalence to Cosine Similarity:**
For unit-normalized vectors, cosine similarity $\cos \theta = \langle u, v \rangle$ and geodesic distance $d_g = \theta$ are monotonically related:

$$\text{High similarity} \iff \cos \theta \approx 1 \iff d_g \approx 0$$
$$\text{Low similarity} \iff \cos \theta \approx 0 \iff d_g \approx \pi/2$$

**Decision Rule:**
Typical ArcFace verification:
```python
cos_sim = torch.sum(emb_A * emb_B, dim=1)  # Inner product
match = (cos_sim > threshold)  # e.g., threshold = 0.3-0.6
```

Equivalently in geodesic space:
```python
d_g = torch.acos(torch.clamp(cos_sim, -1, 1))
match = (d_g < threshold_radians)  # e.g., 0.9-1.2 radians
```

**Why This Matters:**
Our counterfactual generation (Algorithm 1) optimizes:
$$\mathcal{L}(x') = (d_g(f(x), f(x')) - \delta_{\text{target}})^2 + \lambda \|x' - x\|_2^2$$

This loss is geometrically natural for hypersphere embeddings. For Euclidean models, different loss formulations would be required.

---

### Assumption 3: Plausibility Constraints

**Statement:**
Counterfactuals must remain **perceptually plausible** (on the natural face manifold):

$$\|x' - x\|_2 < \epsilon_{\text{pixel}}$$

where $\epsilon_{\text{pixel}}$ is the plausibility bound in pixel space (typically 0.2 L2 norm over RGB [0,1]³).

**Justification:**
If counterfactuals are unrealistic (e.g., adversarial noise, corrupted images), model behavior may be unpredictable due to distribution shift. The falsification test requires that $x'$ represents a realistic face variation.

**Implementation:**
- Feature masking via Grad-CAM spatial maps or superpixel segmentation
- Gradient clipping to prevent large pixel jumps
- Proximity regularization: $\lambda \|x' - x\|_2^2$ in loss function
- Pixel value clamping: $x' \in [0, 1]^{H \times W \times 3}$

**Empirical Validation (PLACEHOLDER):**
Section 5 will report LPIPS perceptual similarity [Zhang2018] and human evaluation to verify counterfactuals are plausible.

---

### Assumption 4: Scope - Verification (1:1) Only

**Statement:**
The framework applies to **pairwise face verification** (1:1 matching):
- Input: Two face images $(x_A, x_B)$
- Output: Similarity score $s = \langle f(x_A), f(x_B) \rangle$ or binary decision (match/non-match)

**NOT applicable to:**
- ❌ Face identification (1:N gallery search)
- ❌ Multi-class classification (softmax over identities)
- ❌ Other biometric modalities (fingerprint, iris, voice)

**Adaptation Required:**
For identification, the framework could be extended by:
1. Treating verification as subroutine (test attribution for top-K matches)
2. Defining counterfactuals that flip rank ordering
3. Adjusting thresholds for gallery-size effects

---

### Assumption 5: Differentiability (Gradient Access)

**Statement:**
The model $f$ is **differentiable** with respect to inputs, enabling gradient-based optimization:

$$\nabla_x f(x) \quad \text{exists and is computable}$$

**Required For:**
- Algorithm 1 (counterfactual generation via gradient descent)
- Integrated Gradients attribution
- Grad-CAM attribution

**NOT Required For:**
- SHAP/LIME (model-agnostic, black-box compatible)

**Limitation:**
Commercial face recognition APIs (e.g., AWS Rekognition, Azure Face API) do not expose gradients. Our framework requires:
- Open-source models (ArcFace, CosFace implementations), OR
- API access to gradients (unlikely for proprietary systems)

For black-box systems, only SHAP-based falsification testing is feasible.

---

## Summary Table

| Assumption | Required By | Check Method |
|------------|-------------|--------------|
| **A1:** Unit-norm embeddings | Geodesic distance metric | Verify $\|\phi(x)\|_2 = 1$ |
| **A2:** Geodesic metric | Falsification criterion | Check if model uses cosine similarity |
| **A3:** Plausibility | Counterfactual validity | LPIPS < 0.3, human evaluation |
| **A4:** Verification (1:1) | Framework scope | Task definition |
| **A5:** Differentiability | Gradient-based methods | Gradient computation test |

---

## Handling Violations

**If Assumption 1 or 2 violated (non-hypersphere models):**
- Framework requires adaptation to Euclidean geometry
- Replace $d_g = \arccos(\langle u, v \rangle)$ with Euclidean distance $\|u - v\|_2$
- Modify Algorithm 1 loss function accordingly

**If Assumption 3 violated (implausible counterfactuals):**
- Increase regularization weight $\lambda$ in Algorithm 1
- Use generative models (StyleGAN) for on-manifold perturbations
- Reject counterfactuals with LPIPS > threshold

**If Assumption 4 violated (identification task):**
- Define attribution per gallery match
- Modify falsification test for rank changes

**If Assumption 5 violated (no gradients):**
- Use model-agnostic methods (SHAP/LIME only)
- Approximate gradients via finite differences (slow)

---

**END OF ASSUMPTIONS BOX**
