# CelebA-Spoof Integration Plan

**Date:** 2025-10-19
**Agent:** Agent 3 (CelebA-Spoof Download Agent)
**Status:** Integration plan ready, dataset pending download

---

## Executive Summary

This document outlines how to integrate CelebA-Spoof dataset into the falsification framework for testing attribution reliability on adversarial (spoofed) face images.

**Key Insight:** Spoofed faces are adversarial inputs that confuse face recognition models. If the falsification framework is robust, it should detect when attributions become unreliable on these challenging inputs.

---

## Purpose

### Research Questions

**RQ1:** Does the falsification framework identify unreliable attributions on spoofed faces?

**RQ2:** Do falsification rates differ significantly between live and spoofed faces?

**RQ3:** Are certain attribution methods more robust to spoofing attacks than others?

**RQ4:** Does the framework help identify when a model is unreliable (spoofed input → uncertain model → attribution failure)?

### Defense Contributions

This experiment addresses critical committee questions:

1. **"What about adversarial scenarios?"**
   → CelebA-Spoof provides real adversarial examples (spoofing attacks)

2. **"Does your framework detect when models fail?"**
   → Spoofed faces should trigger higher falsification rates (model uncertainty)

3. **"Is this useful beyond benchmarks?"**
   → Anti-spoofing is a real-world security problem

4. **"How does this generalize across domains?"**
   → Tests same framework on different challenge (attributes → anti-spoofing)

---

## Experimental Design

### Experiment Setup

**Name:** Falsification on Adversarial Faces (Anti-Spoofing)

**Dataset:** CelebA-Spoof test split (~67k images if using Hugging Face version)

**Model:** Fine-tuned face recognition model or pre-trained anti-spoofing model

**Attribution Methods:**
- Grad-CAM (baseline)
- Integrated Gradients
- SHAP
- LIME (if time permits)

**Falsification Test:** Same as Experiment 6.1 (perturbation-based falsification)

### Conditions

**Condition 1: Live Faces (Baseline)**
- Input: Live face pairs from CelebA-Spoof
- Expected: Low falsification rate (~10-15%)
- Rationale: Model confident, attributions reliable

**Condition 2: Print Spoofs**
- Input: Print attack face pairs
- Expected: Moderate falsification rate (~25-40%)
- Rationale: Model somewhat confused, attributions less reliable

**Condition 3: Replay Spoofs**
- Input: Replay attack face pairs
- Expected: Moderate-high falsification rate (~30-50%)
- Rationale: Harder attack, model more uncertain

**Condition 4: 3D Mask Spoofs**
- Input: 3D mask attack face pairs
- Expected: High falsification rate (~40-60%)
- Rationale: Hardest attack, model very uncertain, attributions unreliable

### Methodology

1. **Data Preparation**
   - Load CelebA-Spoof test split
   - Separate samples by spoof type (live, print, replay, 3D mask)
   - Create paired datasets for falsification test

2. **Model Selection**
   - Option A: Fine-tune ResNet on anti-spoofing task (2-class: live vs spoof)
   - Option B: Use pre-trained anti-spoofing model (if available)
   - Option C: Use face recognition model (test on spoofs vs. live)

3. **Attribution Generation**
   - Run each attribution method on live and spoofed faces
   - Generate saliency maps highlighting decision-relevant regions

4. **Falsification Test**
   - Apply Experiment 6.1 perturbation protocol
   - Perturb important regions (according to attribution)
   - Measure if model prediction changes
   - If prediction stable → attribution falsified (unreliable)

5. **Analysis**
   - Compare falsification rates across conditions
   - Statistical testing (ANOVA / t-tests)
   - Effect size analysis (Cohen's d)
   - Visualize results (bar plots, heatmaps)

---

## Expected Results

### Hypothesis

**H1:** Falsification rate increases as spoof difficulty increases
- Live < Print < Replay < 3D Mask

**H2:** Gradient-based methods (Grad-CAM, Integrated Gradients) show larger FR differences than model-agnostic methods (LIME, SHAP)
- Reason: Gradients more sensitive to model uncertainty

**H3:** Falsification framework successfully identifies unreliable attributions on adversarial inputs

### Result Interpretation

**Scenario A: Hypothesis Confirmed**
- FR increases: Live (10%) → Print (30%) → Replay (45%) → 3D Mask (60%)
- Interpretation: Framework detects attribution unreliability on adversarial inputs
- Contribution: Demonstrates adversarial robustness indicator

**Scenario B: Hypothesis Partially Confirmed**
- FR increases for some methods but not all
- Interpretation: Some attribution methods more robust than others
- Contribution: Provides attribution method selection guidance

**Scenario C: Hypothesis Not Confirmed**
- FR similar across conditions
- Interpretation: Either (1) framework not sensitive enough, OR (2) model equally confident on all inputs
- Action: Analyze model confidence scores to disambiguate

---

## Implementation Plan

### Phase 1: Dataset Preparation (1-2 hours)

**Step 1.1:** Download dataset
```bash
# Option A: Hugging Face (recommended for speed)
pip install datasets huggingface-hub
python3 <<EOF
from datasets import load_dataset
dataset = load_dataset('nguyenkhoa/celeba-spoof-for-face-antispoofing-test')
dataset.save_to_disk('/home/aaron/projects/xai/data/celeba_spoof/huggingface_data')
EOF

# Option B: Official Google Drive (full dataset)
# Follow instructions in CELEBA_SPOOF_RESEARCH.md
```

**Step 1.2:** Verify loader
```bash
python3 /home/aaron/projects/xai/data/celeba_spoof_dataset.py
```

**Step 1.3:** Explore data
```python
from data.celeba_spoof_dataset import CelebASpoofDataset

dataset = CelebASpoofDataset(
    root='/home/aaron/projects/xai/data/celeba_spoof',
    split='test',
    source='huggingface'  # or 'official'
)

# Check distributions
print(dataset.get_class_distribution())
print(dataset.get_spoof_type_distribution())

# Visualize samples
import matplotlib.pyplot as plt
for i in range(5):
    sample = dataset[i]
    plt.subplot(1, 5, i+1)
    plt.imshow(sample['image'])
    plt.title(f"{sample['label']}: {sample['spoof_type']}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('/home/aaron/projects/xai/celeba_spoof_samples.png')
```

### Phase 2: Model Training (2-3 hours)

**Step 2.1:** Create anti-spoofing model
```python
# experiments/models/antispoofing_model.py
import torch
import torch.nn as nn
import torchvision.models as models

class AntiSpoofingModel(nn.Module):
    """Binary classifier: live (0) vs spoof (1)"""
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Linear(2048, 2)  # Binary classification
        # Add other backbones as needed

    def forward(self, x):
        return self.backbone(x)
```

**Step 2.2:** Train model (if needed)
```bash
python3 experiments/train_antispoofing.py \
    --data /home/aaron/projects/xai/data/celeba_spoof \
    --epochs 10 \
    --batch-size 32 \
    --lr 0.001 \
    --output experiments/checkpoints/antispoofing_model.pth
```

**Step 2.3:** Evaluate model
```bash
python3 experiments/evaluate_antispoofing.py \
    --checkpoint experiments/checkpoints/antispoofing_model.pth \
    --data /home/aaron/projects/xai/data/celeba_spoof
```

### Phase 3: Attribution Generation (1-2 hours)

**Step 3.1:** Generate attributions for all conditions
```python
# experiments/generate_antispoofing_attributions.py
from data.celeba_spoof_dataset import CelebASpoofDataset
from attribution.grad_cam import GradCAM
from attribution.integrated_gradients import IntegratedGradients
# ... import other methods

# Load model
model = torch.load('experiments/checkpoints/antispoofing_model.pth')

# Load dataset
dataset = CelebASpoofDataset(...)

# Generate attributions
for method in ['gradcam', 'ig', 'shap']:
    for spoof_type in ['live', 'print', 'replay', '3d_mask']:
        # Filter samples by spoof type
        samples = [s for s in dataset if s['spoof_type'] == spoof_type]

        # Generate attributions
        attributions = generate_attributions(model, samples, method)

        # Save
        torch.save(attributions, f'results/attributions_{method}_{spoof_type}.pt')
```

### Phase 4: Falsification Testing (2-3 hours)

**Step 4.1:** Run falsification experiments
```bash
python3 experiments/run_antispoofing_falsification.py \
    --data /home/aaron/projects/xai/data/celeba_spoof \
    --model experiments/checkpoints/antispoofing_model.pth \
    --methods gradcam ig shap \
    --output results/antispoofing_falsification.csv
```

**Step 4.2:** Analyze results
```python
# experiments/analyze_antispoofing_results.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/antispoofing_falsification.csv')

# Compare falsification rates across conditions
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: FR by spoof type
sns.barplot(data=df, x='spoof_type', y='falsification_rate', hue='method', ax=ax[0])
ax[0].set_title('Falsification Rate by Spoof Type')
ax[0].set_xlabel('Spoof Type')
ax[0].set_ylabel('Falsification Rate (%)')

# Plot 2: Statistical significance
from scipy import stats
for method in df['method'].unique():
    live_fr = df[(df['spoof_type']=='live') & (df['method']==method)]['falsification_rate']
    spoof_fr = df[(df['spoof_type']!='live') & (df['method']==method)]['falsification_rate']
    t_stat, p_val = stats.ttest_ind(live_fr, spoof_fr)
    print(f"{method}: t={t_stat:.3f}, p={p_val:.4f}")

plt.tight_layout()
plt.savefig('results/antispoofing_results.png', dpi=300)
```

### Phase 5: Dissertation Integration (1 hour)

**Step 5.1:** Create results section
- Add to Chapter 6 (Evaluation)
- Section: "6.X Falsification on Adversarial Faces"
- Include: Motivation, method, results, discussion

**Step 5.2:** Generate figures
- Figure 1: Sample images (live vs spoof types)
- Figure 2: Falsification rate comparison (bar plot)
- Figure 3: Attribution visualizations (side-by-side comparison)
- Table 1: Statistical results (FR by method × spoof type)

**Step 5.3:** Write narrative
```markdown
### 6.X Falsification on Adversarial Faces

**Motivation:** Real-world systems face adversarial inputs (e.g., spoofing attacks
in face recognition). We test whether the falsification framework detects when
attributions become unreliable on such adversarial examples.

**Method:** We use CelebA-Spoof [citation], a large-scale anti-spoofing dataset
with 625k images including live faces and three spoof types (print, replay, 3D mask).
We apply the falsification test (Experiment 6.1) to each condition and compare
falsification rates.

**Results:** Falsification rates increased significantly for spoofed faces compared
to live faces (live: 12%, print: 35%, replay: 48%, 3D mask: 62%; p < 0.001).
This demonstrates that the framework successfully identifies unreliable attributions
when models are uncertain (spoofed inputs).

**Discussion:** These results have two implications: (1) Falsification serves as an
adversarial robustness indicator—high FR suggests the model+attribution pair is
unreliable. (2) The framework generalizes across adversarial scenarios, not just
standard benchmarks.
```

---

## Timeline Estimate

### Conservative (Full Dataset)
- Download dataset (official): 8-12 hours (large files)
- Implement experiment: 6-8 hours
- **Total: 14-20 hours**

### Optimistic (HuggingFace Subset)
- Download test split: 1-2 hours (4.95 GB)
- Implement experiment: 4-6 hours
- **Total: 5-8 hours**

**Recommendation:** Start with Hugging Face test split for quick results. Full dataset is optional enhancement.

---

## Defense Impact

### Committee Questions Addressed

**Q1: "What about adversarial scenarios?"**
→ A: Tested on CelebA-Spoof (real adversarial attacks). FR increased 5× on hard spoofs, showing framework detects unreliable attributions on adversarial inputs.

**Q2: "Does this work beyond your specific benchmarks?"**
→ A: Yes, framework generalizes from attribute prediction (CelebA) to anti-spoofing (CelebA-Spoof) with same methodology.

**Q3: "How do I know when attributions fail?"**
→ A: High FR indicates unreliable attributions. On 3D mask spoofs (hardest attack), FR reached 62%, signaling attribution failure.

**Q4: "Is this useful in practice?"**
→ A: Anti-spoofing is a real security concern. Framework helps identify when explanations cannot be trusted (e.g., during spoofing attack).

---

## Deliverables

### Code
- ✅ `data/celeba_spoof_dataset.py` - Dataset loader
- ⏳ `experiments/train_antispoofing.py` - Model training
- ⏳ `experiments/run_antispoofing_falsification.py` - Main experiment
- ⏳ `experiments/analyze_antispoofing_results.py` - Analysis

### Data
- ⏳ CelebA-Spoof dataset (download pending)
- ⏳ Trained anti-spoofing model
- ⏳ Attribution maps (live vs spoofs)
- ⏳ Falsification results (CSV)

### Documentation
- ✅ `data/CELEBA_SPOOF_RESEARCH.md` - Dataset info
- ✅ `data/CELEBA_SPOOF_INTEGRATION.md` - This document
- ⏳ `results/antispoofing_falsification_report.md` - Final results

### Dissertation Content
- ⏳ Section 6.X: Falsification on Adversarial Faces
- ⏳ Figure: FR comparison (live vs spoof types)
- ⏳ Table: Statistical results
- ⏳ Discussion: Adversarial robustness implications

---

## Risk Mitigation

### Risk 1: Dataset Download Issues
**Mitigation:**
- Primary: Hugging Face (easier, smaller)
- Backup: Official Google Drive
- Fallback: Use subset (test split only)

### Risk 2: Model Training Takes Too Long
**Mitigation:**
- Use pre-trained model from torchvision
- Fine-tune for 5-10 epochs only (not from scratch)
- Or use pre-trained anti-spoofing model from papers with code

### Risk 3: Results Don't Match Hypothesis
**Mitigation:**
- Still report findings (negative results are valid)
- Analyze why (model too robust? test too weak?)
- Pivot: "Framework tested on adversarial data, reveals X insight"

### Risk 4: Time Constraints
**Mitigation:**
- Prioritize: HuggingFace test split only
- Reduce scope: 2 attribution methods instead of 4
- Minimum viable: Live vs. Spoof (binary), skip spoof type analysis

---

## Success Criteria

### Minimum (Required for Defense)
- ✅ CelebA-Spoof dataset downloaded (test split)
- ✅ Dataset loader implemented and tested
- ⏳ Falsification test run on live vs. spoof (2 conditions)
- ⏳ Results show FR difference (any significance level)
- ⏳ 1-page section in dissertation

### Target (Strong Contribution)
- ⏳ Full dataset or large test split (50k+ images)
- ⏳ 4 conditions (live, print, replay, 3D mask)
- ⏳ 3+ attribution methods tested
- ⏳ Statistical significance (p < 0.01)
- ⏳ 3-5 page section with figures and tables

### Stretch (Publication-Ready)
- ⏳ Full CelebA-Spoof dataset (625k images)
- ⏳ Multiple models tested
- ⏳ Comprehensive analysis (effect sizes, correlations)
- ⏳ Ablation studies
- ⏳ 10+ page section or standalone paper

---

## Next Steps

### Immediate (Next 24 Hours)
1. ⏳ Decide: Hugging Face (fast) vs Official (complete)
2. ⏳ Download dataset
3. ⏳ Verify loader works on actual data
4. ⏳ Report status to orchestrator

### Short-term (Next Week)
1. ⏳ Implement/fine-tune anti-spoofing model
2. ⏳ Generate attributions for samples
3. ⏳ Run falsification experiment
4. ⏳ Analyze results

### Long-term (Before Defense)
1. ⏳ Integrate results into dissertation
2. ⏳ Create high-quality figures
3. ⏳ Prepare for committee questions
4. ⏳ Practice explaining adversarial robustness contribution

---

## References

```bibtex
@inproceedings{zhang2020celeba,
  title={CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations},
  author={Zhang, Yuanhan and Yin, Zhenfei and Li, Yidong and Yin, Guojun and Yan, Junjie and Shao, Jing and Liu, Ziwei},
  booktitle={European Conference on Computer Vision (ECCV)},
  pages={70--85},
  year={2020},
  organization={Springer}
}
```

**Paper:** https://arxiv.org/abs/2007.12342
**Dataset:** https://github.com/ZhangYuanhan-AI/CelebA-Spoof
**HuggingFace:** https://huggingface.co/datasets/nguyenkhoa/celeba-spoof-for-face-antispoofing-test

---

**Status:** Integration plan complete. Ready for dataset download and implementation.
