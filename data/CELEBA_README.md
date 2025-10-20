# CelebA Dataset Documentation

**Agent:** Agent 2 (CelebA Main Dataset Download)
**Date:** October 19, 2025
**Status:** Download in progress via Kaggle API

---

## Overview

**Full Name:** CelebFaces Attributes Dataset (CelebA)
**Source:** http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
**Paper:** Liu et al. "Deep Learning Face Attributes in the Wild" ICCV 2015
**License:** Non-commercial research purposes only

---

## Dataset Statistics

- **Images:** 202,599 aligned face images
- **Resolution:** 178 x 218 pixels (aligned and cropped)
- **Format:** JPEG
- **Color:** RGB
- **Identities:** 10,177 unique celebrity persons
- **Attributes:** 40 binary attributes per image
- **Landmarks:** 5 facial landmarks (10 coordinates)
- **File Size:** ~1.7 GB total
  - Images: ~1.5 GB
  - Annotations: ~200 MB

---

## Dataset Components

### 1. Images (img_align_celeba/)

202,599 aligned and cropped face images.

**Preprocessing applied by dataset creators:**
- Face detection
- Alignment based on eye positions
- Cropping to 178 x 218 pixels
- No augmentation or filtering

**Image naming:** 000001.jpg to 202599.jpg

### 2. Attributes (list_attr_celeba.txt)

40 binary facial attributes per image.

**Format:** Each line contains image filename followed by 40 values (-1 or +1)

**Attribute List (40 total):**

#### Demographic
- Male
- Young

#### Hair
- Bald
- Bangs
- Black_Hair
- Blond_Hair
- Brown_Hair
- Gray_Hair
- Receding_Hairline
- Straight_Hair
- Wavy_Hair

#### Facial Hair
- Mustache
- Goatee
- Sideburns
- No_Beard
- 5_o_Clock_Shadow

#### Accessories
- Eyeglasses
- Wearing_Hat
- Wearing_Earrings
- Wearing_Necklace
- Wearing_Necktie
- Wearing_Lipstick

#### Facial Features
- Arched_Eyebrows
- Bags_Under_Eyes
- Big_Lips
- Big_Nose
- Bushy_Eyebrows
- Chubby
- Double_Chin
- High_Cheekbones
- Narrow_Eyes
- Oval_Face
- Pale_Skin
- Pointy_Nose
- Rosy_Cheeks

#### Expression & Appearance
- Smiling
- Mouth_Slightly_Open
- Attractive
- Heavy_Makeup
- Blurry

**Distribution (approximate):**
- Most attributes are imbalanced (5-40% positive)
- Male: ~40% positive
- Smiling: ~48% positive
- Young: ~77% positive
- Attractive: ~51% positive

### 3. Landmarks (list_landmarks_align_celeba.txt)

5 facial landmark points per image (x, y coordinates).

**Format:** Each line contains image filename followed by 10 integer values

**Landmark Points:**
1. **Left Eye Center:** (x, y) - Coordinates of left eye
2. **Right Eye Center:** (x, y) - Coordinates of right eye
3. **Nose Tip:** (x, y) - Tip of the nose
4. **Left Mouth Corner:** (x, y) - Left corner of mouth
5. **Right Mouth Corner:** (x, y) - Right corner of mouth

**Coordinate System:**
- Origin: Top-left corner of image
- X-axis: Left to right (0 to 178)
- Y-axis: Top to bottom (0 to 218)

**Use Cases:**
- Face alignment verification
- Geometric feature extraction
- Facial keypoint detection
- Expression analysis

### 4. Identities (identity_CelebA.txt)

Celebrity identity labels for person recognition.

**Format:** Each line contains image filename and person ID

**Statistics:**
- **Unique identities:** 10,177 celebrities
- **Images per person:** Average ~20 images (highly variable)
- **ID range:** 1 to 10,177
- **Min images/person:** 1
- **Max images/person:** ~65

**Distribution:**
- Most celebrities have 10-30 images
- Long-tail distribution (many with few images)
- Enables face verification experiments

**Use Cases:**
- Face verification (same/different person)
- Face recognition training
- Identity-based fairness analysis
- Few-shot learning experiments

### 5. Bounding Boxes (list_bbox_celeba.txt)

Face bounding boxes for each image.

**Format:** Each line contains image filename and 4 values: x_1, y_1, width, height

**Coordinates:**
- x_1, y_1: Top-left corner of face bounding box
- width: Width of box
- height: Height of box

**Use Cases:**
- Face detection evaluation
- Cropping for different aspect ratios
- Region of interest extraction

### 6. Partitions (list_eval_partition.txt)

Train/validation/test split assignments.

**Format:** Each line contains image filename and partition ID (0, 1, or 2)

**Split Distribution:**
- **Train (0):** 162,770 images (80.4%)
- **Validation (1):** 19,867 images (9.8%)
- **Test (2):** 19,962 images (9.8%)

**Split Strategy:**
- Random partition by image (not by person)
- Same person may appear in multiple splits
- Standard split used in most CelebA papers

---

## Directory Structure

After download and extraction:

```
/home/aaron/projects/xai/data/celeba/
└── celeba/
    ├── img_align_celeba/           # 202,599 images
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   ├── 000003.jpg
    │   └── ... (202,599 total)
    ├── list_attr_celeba.txt        # 40 attributes (25 MB)
    ├── list_landmarks_align_celeba.txt  # 5 landmarks (15 MB)
    ├── identity_CelebA.txt         # Person IDs (3 MB)
    ├── list_bbox_celeba.txt        # Bounding boxes (10 MB)
    └── list_eval_partition.txt     # Train/val/test split (5 MB)
```

---

## Usage Examples

### Loading Dataset with CelebADataset

```python
from data.celeba_dataset import CelebADataset
from torchvision import transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load train set
dataset = CelebADataset(
    root_dir='/home/aaron/projects/xai/data/celeba',
    split='train',
    transform=transform
)

print(f"Dataset size: {len(dataset)} images")

# Get sample
image, attributes = dataset[0]
print(f"Image shape: {image.shape}")        # [3, 160, 160]
print(f"Attributes shape: {attributes.shape}")  # [40]
print(f"Attributes: {attributes}")          # [-1, 1, -1, ...]
```

### Querying by Attribute

```python
# Get all male images
male_indices = dataset.get_images_with_attribute('Male', value=1)
print(f"Male images: {len(male_indices)}")

# Get all female images
female_indices = dataset.get_images_with_attribute('Male', value=-1)
print(f"Female images: {len(female_indices)}")

# Get smiling images
smiling_indices = dataset.get_images_with_attribute('Smiling', value=1)
print(f"Smiling images: {len(smiling_indices)}")
```

### Dataset Statistics

```python
stats = dataset.get_dataset_statistics()
print(f"Total images: {stats['n_images']}")
print(f"Number of attributes: {stats['n_attributes']}")

# Attribute distributions
for attr_name, attr_stats in stats['attribute_statistics'].items():
    print(f"{attr_name}: {attr_stats['positive_ratio']*100:.1f}% positive")
```

### Filtering by Multiple Attributes

```python
# Get young smiling females
young_idx = set(dataset.get_images_with_attribute('Young', value=1))
smiling_idx = set(dataset.get_images_with_attribute('Smiling', value=1))
female_idx = set(dataset.get_images_with_attribute('Male', value=-1))

filtered = young_idx & smiling_idx & female_idx
print(f"Young smiling females: {len(filtered)} images")
```

---

## Common Use Cases

### 1. Attribute Classification

Train models to predict attributes from face images.

```python
# Binary classification for each attribute
model = AttributeClassifier(num_attributes=40)
image, attributes = dataset[0]
predictions = model(image)  # [40] binary predictions
loss = binary_crossentropy(predictions, attributes)
```

### 2. Face Verification

Verify if two images are of the same person.

**Approach:**
1. Use identity labels to create positive/negative pairs
2. Train Siamese network or triplet loss
3. Evaluate on test set pairs

**Challenge:** Same person may appear in train and test splits

### 3. Fairness Analysis

Analyze model performance across demographic groups.

```python
# Evaluate separately on male/female subsets
male_subset = Subset(dataset, dataset.get_images_with_attribute('Male', 1))
female_subset = Subset(dataset, dataset.get_images_with_attribute('Male', -1))

male_acc = evaluate(model, male_subset)
female_acc = evaluate(model, female_subset)
fairness_gap = abs(male_acc - female_acc)
```

### 4. Multi-Task Learning

Predict multiple attributes simultaneously.

```python
# Multi-task model predicting all 40 attributes
model = MultiTaskAttributeModel(num_tasks=40)
image, attributes = dataset[0]
predictions = model(image)  # [40] predictions
loss = sum([criterion(pred, label) for pred, label in zip(predictions, attributes)])
```

---

## Citation

If using CelebA dataset in research, please cite:

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

## Known Issues & Limitations

### 1. Celebrity Bias
- All images are of celebrities
- May not generalize to general population
- Different image quality than everyday photos

### 2. Annotation Quality
- Attributes are binary (no "N/A" or uncertainty)
- Some attributes are subjective (e.g., "Attractive")
- Annotation errors exist (~5-10% error rate estimated)

### 3. Demographic Imbalance
- ~60% female, ~40% male
- Limited diversity in age, ethnicity
- Most images are of young adults

### 4. Privacy Considerations
- Images are of public figures
- Use only for non-commercial research
- Follow dataset license terms

### 5. Data Leakage
- Same person can appear in train/val/test splits
- For face recognition: use person-disjoint splits
- For attributes: current split is fine

---

## Verification

After download, verify dataset integrity:

```bash
# Verify all files present
python data/download_celeba.py --verify

# Analyze dataset statistics
python data/download_celeba.py --analyze

# Test dataset loader
python -c "
from data.celeba_dataset import CelebADataset
dataset = CelebADataset(root_dir='data/celeba', split='test')
print(f'Dataset ready: {len(dataset)} images')
"
```

Expected verification output:
```
✓ Images              : 202,599 images
✓ Attributes          : 25.0 MB
✓ Landmarks           : 15.0 MB
✓ Identities          : 3.0 MB
✓ Bounding Boxes      : 10.0 MB
✓ Partitions          : 5.0 MB

Total disk usage: 1.70 GB

✓ Dataset verification PASSED
```

---

## Related Datasets

### CelebA-HQ
- Higher resolution version (1024 x 1024)
- 30,000 images (subset of CelebA)
- Better quality, fewer images

### CelebA-Mask-HQ
- Includes semantic segmentation masks
- 30,000 images
- 19 facial semantic classes

### CelebA-Spoof
- Anti-spoofing dataset
- Same identities as CelebA
- Live vs. spoof labels

---

## Support & Troubleshooting

**Issue:** Images not found
**Solution:** Check directory structure matches expected format

**Issue:** Attribute file parse error
**Solution:** Ensure file is not corrupted, re-download if needed

**Issue:** Out of memory
**Solution:** Use `n_samples` parameter to load subset

**Issue:** Slow loading
**Solution:** Use SSD storage, increase num_workers in DataLoader

---

**Dataset Status:** Ready for experiments after successful download
**Recommended Models:** ResNet, ViT, FaceNet, ArcFace
**Typical Training Time:** 2-4 hours on GPU for attribute classification
