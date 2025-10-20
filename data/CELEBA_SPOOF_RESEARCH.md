# CelebA-Spoof Dataset Research

**Date:** 2025-10-19
**Agent:** Agent 3 (CelebA-Spoof Download Agent)

---

## Official Information

- **Paper:** CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations
- **Authors:** Yuanhan Zhang, Zhenfei Yin, Yidong Li, Guojun Yin, Junjie Yan, Jing Shao, Ziwei Liu
- **Publication:** European Conference on Computer Vision (ECCV) 2020
- **Project Page:** https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Spoof.html
- **arXiv:** https://arxiv.org/abs/2007.12342
- **GitHub:** https://github.com/ZhangYuanhan-AI/CelebA-Spoof
- **Papers with Code:** https://paperswithcode.com/dataset/celeba-spoof

---

## Dataset Specifications

### Overall Statistics
- **Total Images:** 625,537 images
- **Subjects:** 10,177 unique identities
- **Attributes:** 43 rich attributes (inherited 40 from CelebA + 3 spoof-specific)
- **Annotations:** Face, illumination, environment, and spoof type labels

### Spoof Types
The dataset includes **10 spoof type annotations**:
1. **Live faces** (genuine/bonafide samples)
2. **Print attacks** (photo prints)
3. **Replay attacks** (digital screen replay)
4. **3D mask attacks**
5. Multiple variations across different materials and conditions

### Capture Conditions
- **Environments:** 2 different environments
- **Illumination:** 4 illumination conditions
- **Sensors:** More than 10 different capture sensors
- **Scenes:** 8 total scenes (2 environments × 4 illumination conditions)

### Splits
Based on the Hugging Face version and common practice:
- **Train:** ~400,000+ images
- **Validation:** ~100,000+ images
- **Test:** ~67,200 images (confirmed on Hugging Face)

---

## Download Sources

### 1. Official Sources (Primary)
**GitHub Repository:**
- URL: https://github.com/ZhangYuanhan-AI/CelebA-Spoof
- Status: Active, Google Drive and Baidu Drive links available
- Access: Free for non-commercial research

**Official Project Page:**
- URL: https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Spoof.html
- Status: Active

### 2. Hugging Face (Recommended Alternative)
**Test Split (Community Upload):**
- Dataset: `nguyenkhoa/celeba-spoof-for-face-antispoofing-test`
- URL: https://huggingface.co/datasets/nguyenkhoa/celeba-spoof-for-face-antispoofing-test
- Size: 4.95 GB (test split only, ~67.2k images)
- Format: Parquet files with preprocessed images
- Features:
  - `image`: Cropped face images
  - `label`: Binary (0=live, 1=spoof)
  - `label_name`: Text label ("live" or "spoof")
- Advantages:
  - Easy download via `datasets` library
  - Already preprocessed and cropped
  - Ready for immediate experimentation
  - No manual Google Drive download needed

**Full Dataset (Community Upload):**
- Dataset: `TrainingDataPro/celeba-spoof-dataset`
- URL: https://huggingface.co/datasets/TrainingDataPro/celeba-spoof-dataset
- Status: Available but size/completeness unclear

### 3. Google Drive (Official - Full Dataset)
- Access via GitHub repository README
- Requires: Google account
- Size: Estimated 50-100+ GB (full dataset with all splits)
- Format: Multiple zip files (CelebA_Spoof.zip.00, .01, .02, etc.)
- Note: Requires concatenation of split zip files before extraction

### 4. Baidu Drive (Alternative)
- Access via GitHub repository README
- Requires: Baidu account (Chinese registration)
- Less convenient for international users

---

## File Size Estimates

### Hugging Face Test Split
- **Downloaded:** 4.95 GB
- **Disk Space Needed:** ~6 GB (with extraction overhead)
- **Images:** 67,200

### Full Official Dataset (Estimated)
- **Downloaded:** 50-100+ GB (compressed)
- **Disk Space Needed:** 100-150+ GB (uncompressed)
- **Images:** 625,537
- **Note:** Exact size requires checking official Google Drive

---

## License and Usage Restrictions

### Non-Commercial Research Only

The CelebA-Spoof dataset is available for **non-commercial research purposes only**.

**Restrictions:**
- ❌ No reproduction, duplication, or copying for commercial purposes
- ❌ No selling, trading, or reselling of images or derived data
- ❌ No commercial exploitation
- ✅ Academic research permitted
- ✅ Educational purposes permitted
- ✅ Publication of research results permitted (with proper citation)

**Important:** This aligns perfectly with PhD dissertation research use case.

---

## Citation

```bibtex
@inproceedings{zhang2020celeba,
  title={CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations},
  author={Zhang, Yuanhan and Yin, Zhenfei and Li, Yidong and Yin, Guojun and Yan, Junjie and Shao, Jing and Liu, Ziwei},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

---

## Research Impact

### Why CelebA-Spoof Matters for This Dissertation

1. **Adversarial Robustness Testing**
   - Spoof images are inherently adversarial (intentionally deceptive)
   - Tests if falsification framework detects unreliable attributions on adversarial inputs
   - Provides defense against committee question: "What about adversarial scenarios?"

2. **Real-World Applicability**
   - Anti-spoofing is a real security concern in face recognition systems
   - Demonstrates framework utility beyond standard benchmarks
   - Shows practical value for security-critical applications

3. **Attribution Reliability Under Uncertainty**
   - Hypothesis: Models uncertain about spoofed faces → attribution methods should fail falsification tests
   - If model can't reliably classify, attribution can't reliably explain
   - Validates core thesis: falsification detects when attributions become unreliable

4. **Complementary to CelebA**
   - Same domain (faces) but different challenge (spoofing vs. attributes)
   - Can compare: "Falsification on normal faces vs. adversarial faces"
   - Demonstrates generalization of framework across related tasks

---

## Download Strategy

### Recommended Approach (Fastest to Complete)

**Phase 1: Quick Start with Hugging Face Test Split**
1. Download `nguyenkhoa/celeba-spoof-for-face-antispoofing-test` (4.95 GB)
2. Run pilot experiments on test split (~67k images)
3. Validate experimental design and falsification framework
4. Generate preliminary results for dissertation

**Phase 2: Full Dataset (If Time Permits)**
1. Download full dataset from official Google Drive
2. Replicate experiments on full train/val/test splits
3. Include more comprehensive results in dissertation
4. More robust statistical validation

**Rationale:**
- Test split alone (67k images) is sufficient for proof-of-concept
- Can demonstrate framework validity without 2-3 day download
- Full dataset is "nice to have" not "must have"
- Time-sensitive dissertation timeline favors quick wins

---

## Technical Specifications

### Image Properties (from CelebA-Spoof paper)
- **Format:** JPEG
- **Face Detection:** Pre-aligned (similar to CelebA)
- **Resolution:** Variable (high resolution available)
- **Preprocessing:** Faces already cropped and aligned in Hugging Face version

### Annotation Format
Based on official GitHub and papers:
- Text files with image filenames and labels
- Binary labels: live (0) vs. spoof (1)
- Multi-class labels: spoof type (print, replay, 3D mask, etc.)
- Attribute labels: Inherited from CelebA (40 attributes like age, gender, glasses, etc.)

---

## Related Datasets (for context)

### Other Anti-Spoofing Datasets
- **OULU-NPU:** 4,950 videos (smaller scale)
- **CASIA-FASD:** 600 videos (much smaller)
- **Replay-Attack:** 1,300 videos (video-based)
- **SiW (Spoofing in the Wild):** 4,478 videos

**CelebA-Spoof Advantage:**
- 100x larger than competitors
- Image-based (easier to integrate with existing framework)
- Rich attributes (can study intersectionality: e.g., "Does falsification differ for spoofed faces with glasses?")

---

## Next Steps

1. ✅ Research completed
2. ⏳ Download test split from Hugging Face
3. ⏳ Create dataset loader (`celeba_spoof_dataset.py`)
4. ⏳ Verify data loading and inspection
5. ⏳ Create integration plan for experiments
6. ⏳ Report status to orchestrator

---

**Status:** Research phase complete. Ready to proceed with download.
