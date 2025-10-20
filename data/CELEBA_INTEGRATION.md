# CelebA Dataset Integration Guide

**Status:** Ready for download and integration
**Last Updated:** October 19, 2025
**Purpose:** Multi-dataset validation for defense readiness (Chapter 8)

---

## Dataset Overview

CelebA (CelebFaces Attributes Dataset) is a large-scale face attributes dataset with rich annotations.

### Statistics

- **Images:** 202,599 aligned & cropped face images (178x218 pixels)
- **Identities:** 10,177 unique celebrities
- **Attributes:** 40 binary labels per image
- **Landmarks:** 5 facial keypoints per image (eyes, nose, mouth corners)
- **Partitions:** Train (162,770), Validation (19,867), Test (19,962)
- **Size:** ~1.5 GB (images) + ~200 MB (annotations)

### Source

- **Official:** http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- **Paper:** Liu et al., "Deep Learning Face Attributes in the Wild", ICCV 2015
- **License:** Non-commercial research only

---

## Dataset Components

### 1. Images: `img_align_celeba/`

- 202,599 JPG images
- Aligned and cropped to 178x218 pixels
- Centered on face region
- Filenames: `000001.jpg` to `202599.jpg`

### 2. Attributes: `list_attr_celeba.txt`

40 binary attributes (-1 = absent, +1 = present):

**Demographic:**
- Male, Young, Attractive, Smiling

**Hair:**
- Bald, Bangs, Black_Hair, Blond_Hair, Brown_Hair, Gray_Hair, Straight_Hair, Wavy_Hair, Receding_Hairline

**Facial Features:**
- Arched_Eyebrows, Bags_Under_Eyes, Bushy_Eyebrows, Big_Lips, Big_Nose, Chubby, Double_Chin, High_Cheekbones, Narrow_Eyes, Pointy_Nose, Rosy_Cheeks

**Accessories:**
- Eyeglasses, Heavy_Makeup, Wearing_Earrings, Wearing_Hat, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie

**Facial Hair:**
- 5_o_Clock_Shadow, Goatee, Mustache, No_Beard, Sideburns

**Other:**
- Blurry, Oval_Face, Pale_Skin, Mouth_Slightly_Open

### 3. Landmarks: `list_landmarks_align_celeba.txt`

5 (x, y) coordinates per image:
- Left eye center
- Right eye center
- Nose tip
- Left mouth corner
- Right mouth corner

### 4. Identities: `identity_CelebA.txt`

- Celebrity identity labels (10,177 unique IDs)
- Format: `image_id celebrity_id`
- Useful for face verification tasks

### 5. Bounding Boxes: `list_bbox_celeba.txt`

- Face bounding boxes: (x, y, width, height)
- Useful for localization tasks

### 6. Partitions: `list_eval_partition.txt`

- Train/val/test split
- Format: `image_id partition` (0=train, 1=val, 2=test)

---

## Download Instructions

### Method 1: Torchvision (Recommended)

Easiest method - handles everything automatically:

```bash
cd /home/aaron/projects/xai
python data/download_celeba.py
```

This will:
- Download all images and annotations
- Organize directory structure
- Verify completeness
- Report statistics

**Time:** 30-60 minutes (depending on network speed)

### Method 2: Kaggle API

If torchvision fails:

```bash
python data/download_celeba.py --method kaggle
```

Requires:
- Kaggle account
- API key configured in `~/.kaggle/kaggle.json`

### Method 3: Manual Download

Show step-by-step instructions:

```bash
python data/download_celeba.py --method manual
```

Follow printed instructions to download from official source.

### Verify Download

Check if dataset is complete:

```bash
python data/download_celeba.py --verify
```

Expected output:
```
✓ Images              : 202,599 images
✓ Attributes          : 25.4 MB
✓ Landmarks           : 12.1 MB
✓ Identities          : 6.8 MB
✓ Bounding Boxes      : 8.2 MB
✓ Partitions          : 4.1 MB
Total disk usage: 1.52 GB
```

### Analyze Statistics

```bash
python data/download_celeba.py --analyze
```

Shows:
- Image count
- Attribute distributions
- Identity statistics
- Partition sizes

---

## Integration with Experiments

### Experiment 6.1: Face Verification

Use identity labels to create same-person and different-person pairs:

```python
from data.celeba_dataset import CelebADataset

# Load CelebA with identities
dataset = CelebADataset(
    root_dir='/home/aaron/projects/xai/data/celeba',
    split='test',
    return_identity=True
)

# Generate verification pairs
pairs = dataset.generate_pairs(n_pairs=1000)
# Returns: [(img1, img2, label)] where label=1 (same), 0 (different)
```

**Use Case:** Cross-dataset validation with LFW and CFP-FP

### Experiment 6.6: Demographic Fairness Analysis

Use attribute labels for demographic subgroup analysis:

```python
# Load CelebA with attributes
dataset = CelebADataset(
    root_dir='/home/aaron/projects/xai/data/celeba',
    split='test',
    return_attributes=True
)

# Filter by demographic attributes
male_samples = dataset.filter_by_attribute('Male', value=1)
female_samples = dataset.filter_by_attribute('Male', value=-1)
young_samples = dataset.filter_by_attribute('Young', value=1)

# Analyze performance by subgroup
results = {
    'male': evaluate_on_subset(male_samples),
    'female': evaluate_on_subset(female_samples),
    'young': evaluate_on_subset(young_samples)
}
```

**Use Case:** Fairness analysis across gender, age, and other attributes

### Attribution Validation with Landmarks

Use landmarks to define facial regions for attribution analysis:

```python
# Load with landmarks
dataset = CelebADataset(
    root_dir='/home/aaron/projects/xai/data/celeba',
    split='test',
    return_landmarks=True
)

# Get image and landmarks
img, landmarks = dataset[0]
# landmarks = [(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5)]

# Define regions
eye_region = define_region(landmarks[0], landmarks[1], margin=10)
nose_region = define_region(landmarks[2], margin=10)
mouth_region = define_region(landmarks[3], landmarks[4], margin=10)

# Compare with attribution heatmaps
attribution_map = generate_attribution(img)
eye_importance = attribution_map[eye_region].mean()
nose_importance = attribution_map[nose_region].mean()
mouth_importance = attribution_map[mouth_region].mean()
```

**Use Case:** Validate that attributions focus on relevant facial features

---

## Comparison to Other Face Datasets

| Feature | LFW | CFP-FP | CelebA |
|---------|-----|--------|--------|
| **Images** | 13,233 | 7,000 | 202,599 |
| **Identities** | 5,749 | 500 | 10,177 |
| **Attributes** | None | None | 40 labels |
| **Landmarks** | No | No | 5 points |
| **Pose Variation** | Low | High (frontal vs. profile) | Low |
| **Diversity** | In-the-wild | Constrained | Celebrity faces |
| **Demographics** | 83% White, 78% Male | Unknown | More balanced |
| **Best For** | Real-world verification | Pose robustness | Attribute analysis |

### Why CelebA is Important

1. **Scale:** 15x more images than LFW, enables robust statistical analysis
2. **Attributes:** Only dataset with rich attribute labels (gender, age, hair, etc.)
3. **Landmarks:** Enables region-based attribution validation
4. **Identities:** Sufficient for same-person vs. different-person analysis
5. **Balance:** Better demographic balance than LFW

### Complementary Strengths

- **LFW:** Real-world in-the-wild conditions
- **CFP-FP:** Extreme pose variations (frontal vs. profile)
- **CelebA:** Large-scale with rich annotations for detailed analysis

Together, these three datasets provide comprehensive multi-dataset validation.

---

## Integration Status

### Completed
- [x] Enhanced download script with 3 methods (torchvision, kaggle, manual)
- [x] Verification function for completeness checking
- [x] Analysis function for statistics
- [x] Integration documentation

### In Progress
- [ ] Download dataset (~1.5 GB, 30-60 minutes)
- [ ] Integrate into `celeba_dataset.py` loader
- [ ] Update `run_multidataset_experiment_6_1.py` to include CelebA
- [ ] Run baseline experiments

### Next Steps

1. **Download Dataset**
   ```bash
   python data/download_celeba.py
   ```

2. **Verify Download**
   ```bash
   python data/download_celeba.py --verify
   ```

3. **Analyze Statistics**
   ```bash
   python data/download_celeba.py --analyze
   ```

4. **Test Dataset Loader**
   ```bash
   python -c "from data.celeba_dataset import CelebADataset; \
              d = CelebADataset('/home/aaron/projects/xai/data/celeba'); \
              print(f'Loaded {len(d)} images')"
   ```

5. **Run Multi-Dataset Experiment**
   ```bash
   python experiments/run_multidataset_experiment_6_1.py \
       --datasets lfw cfp_fp celeba \
       --n-pairs 100 \
       --models facenet vggface
   ```

6. **Analyze Cross-Dataset Results**
   - Compare performance across LFW, CFP-FP, CelebA
   - Identify dataset-specific biases
   - Validate attribution consistency
   - Report findings in Chapter 8 Section 8.2.4

---

## Expected Results Structure

After running experiments, expect:

```
results/experiment_6_1_multidataset/
├── lfw/
│   ├── facenet_results.json
│   ├── vggface_results.json
│   └── attribution_maps/
├── cfp_fp/
│   ├── facenet_results.json
│   ├── vggface_results.json
│   └── attribution_maps/
└── celeba/
    ├── facenet_results.json    # NEW
    ├── vggface_results.json    # NEW
    └── attribution_maps/       # NEW
```

Each `results.json` contains:
- Verification accuracy
- Attribution consistency scores
- Demographic fairness metrics (CelebA only)
- Region importance scores (CelebA only, using landmarks)

---

## Research Questions

CelebA enables answering:

### RQ1: Cross-Dataset Generalization
- Do models trained on LFW generalize to CelebA?
- Are attribution patterns consistent across datasets?

### RQ2: Demographic Fairness
- Does performance vary by gender, age, ethnicity?
- Are attributions equally reliable across subgroups?

### RQ3: Attribute-Based Analysis
- Which facial features matter most for verification?
- Do attributions align with human-interpretable attributes?

### RQ4: Scale Effects
- Does larger dataset (CelebA) improve attribution reliability?
- Are rare attributes handled correctly?

---

## Citation

If using CelebA in dissertation:

```bibtex
@inproceedings{liu2015faceattributes,
  title = {Deep Learning Face Attributes in the Wild},
  author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
  month = {December},
  year = {2015}
}
```

---

## Troubleshooting

### Download Fails
- Try alternate method: `python data/download_celeba.py --method kaggle`
- Check disk space: Need ~2 GB free
- Check network: Large download requires stable connection

### Verification Fails
- Re-download: `python data/download_celeba.py`
- Check file permissions: Should be readable
- Check disk not full

### Dataset Loader Fails
- Verify structure: `python data/download_celeba.py --verify`
- Check imports: `pip install torch torchvision pillow`
- Review error messages in dataset loader

---

## Contact & Resources

- **Official Site:** http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- **Paper:** https://arxiv.org/abs/1411.7766
- **Kaggle:** https://www.kaggle.com/jessicali9530/celeba-dataset
- **PyTorch Docs:** https://pytorch.org/vision/stable/datasets.html#celeba

---

**Ready to download and integrate CelebA for comprehensive multi-dataset validation.**
