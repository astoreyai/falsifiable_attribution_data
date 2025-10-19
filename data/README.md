# Data Package - Face Verification Dataset Loaders

This package provides PyTorch dataset loaders for face verification experiments.

## Overview

The package includes:

1. **VGGFace2Dataset** - Face verification with identity pairs
2. **CelebADataset** - Face images with 40 attribute annotations
3. **Utilities** - Transforms, metrics, and helper functions

## Installation

The data package is part of the XAI project. No additional installation needed beyond project requirements.

## Dataset Structure

### VGGFace2

Expected directory structure:

```
vggface2/
├── train/
│   ├── n000001/
│   │   ├── 0001_01.jpg
│   │   └── ...
│   └── ...
└── test/
    └── ...
```

**Download:** http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/

### CelebA

Expected directory structure:

```
celeba/
├── img_align_celeba/
│   ├── 000001.jpg
│   └── ...
├── list_attr_celeba.txt       # 40 attributes
└── list_eval_partition.txt    # train/val/test split
```

**Download:** http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## Usage

### VGGFace2 Face Verification Pairs

```python
from data import VGGFace2Dataset, get_insightface_transforms
from torch.utils.data import DataLoader

# Define transforms (112x112 for InsightFace/ArcFace)
transform = get_insightface_transforms(image_size=112)

# Create dataset
dataset = VGGFace2Dataset(
    root_dir='/datasets/vggface2',
    split='test',
    transform=transform,
    n_pairs=200,          # Number of face pairs
    genuine_ratio=0.5,    # 50% genuine, 50% impostor
    seed=42
)

# Create dataloader
from data import collate_verification_pairs

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_verification_pairs
)

# Iterate
for img1, img2, labels in dataloader:
    # img1, img2: [B, 3, 112, 112]
    # labels: [B] (1=same identity, 0=different)
    pass
```

### CelebA with Attributes

```python
from data import CelebADataset, get_default_transforms
from torch.utils.data import DataLoader

# Define transforms
transform = get_default_transforms(image_size=224)

# Create dataset (all 40 attributes)
dataset = CelebADataset(
    root_dir='/datasets/celeba',
    split='test',
    transform=transform,
    n_samples=1000,
    seed=42
)

# Or select specific attributes
dataset = CelebADataset(
    root_dir='/datasets/celeba',
    split='test',
    transform=transform,
    attributes=['Male', 'Smiling', 'Young'],
    n_samples=1000
)

# Create dataloader
from data import collate_attribute_batch

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_attribute_batch
)

# Iterate
for images, attributes in dataloader:
    # images: [B, 3, 224, 224]
    # attributes: [B, N_attrs] with values in {-1, 1}
    pass
```

## Available Transforms

The package provides pre-configured transforms for different models:

```python
from data import (
    get_insightface_transforms,  # 112x112, normalized to [-1, 1]
    get_facenet_transforms,      # 160x160, normalized to [-1, 1]
    get_vggface_transforms,      # 224x224, ImageNet normalization
    get_default_transforms,      # Customizable
)

# Custom transform with augmentation
transform = get_default_transforms(
    image_size=112,
    normalize=True,
    augment=True  # Adds flip, jitter, rotation
)
```

## Dataset Statistics

Both datasets provide statistics methods:

```python
# VGGFace2
stats = dataset.get_dataset_statistics()
print(stats['n_identities'])
print(stats['genuine_ratio'])

# CelebA
stats = dataset.get_dataset_statistics()
print(stats['attribute_statistics']['Male'])
```

## Utilities

### Embedding Metrics

```python
from data import (
    normalize_embeddings,
    cosine_similarity,
    euclidean_distance,
    compute_embedding_statistics,
)

# Normalize embeddings
emb_norm = normalize_embeddings(embeddings)

# Compute similarities
similarities = cosine_similarity(emb1, emb2)
distances = euclidean_distance(emb1, emb2)

# Embedding statistics
stats = compute_embedding_statistics(embeddings)
```

### Verification Metrics

```python
from data import compute_verification_metrics

metrics = compute_verification_metrics(
    similarities=similarity_scores,
    labels=ground_truth_labels
)

print(f"Best threshold: {metrics['best_threshold']}")
print(f"Best accuracy: {metrics['best_accuracy']}")
print(f"TPR at best: {metrics['best_tpr']}")
print(f"FPR at best: {metrics['best_fpr']}")
```

## Dataset Paths

The loaders automatically search common dataset locations:

1. User-provided path
2. `/datasets/<dataset_name>`
3. `~/datasets/<dataset_name>`
4. `/data/<dataset_name>`
5. `~/data/<dataset_name>`

Place your datasets in any of these locations for automatic detection.

## CelebA Attributes

All 40 attributes:

```
5_o_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes,
Bald, Bangs, Big_Lips, Big_Nose, Black_Hair, Blond_Hair, Blurry,
Brown_Hair, Bushy_Eyebrows, Chubby, Double_Chin, Eyeglasses, Goatee,
Gray_Hair, Heavy_Makeup, High_Cheekbones, Male, Mouth_Slightly_Open,
Mustache, Narrow_Eyes, No_Beard, Oval_Face, Pale_Skin, Pointy_Nose,
Receding_Hairline, Rosy_Cheeks, Sideburns, Smiling, Straight_Hair,
Wavy_Hair, Wearing_Earrings, Wearing_Hat, Wearing_Lipstick,
Wearing_Necklace, Wearing_Necktie, Young
```

## Examples

See `examples/dataset_usage.py` for complete examples:

```bash
cd /home/aaron/projects/xai
python examples/dataset_usage.py
```

## Module Structure

```
data/
├── __init__.py           # Package exports
├── datasets.py           # VGGFace2Dataset
├── celeba_dataset.py     # CelebADataset
├── utils.py              # Transforms and utilities
└── README.md             # This file
```

## Error Handling

The loaders provide informative error messages:

- Dataset not found → Shows download URL
- Split not found → Lists available splits
- Missing files → Shows expected paths
- Invalid attributes → Lists valid attribute names

## Notes

1. **Image Sizes:**
   - InsightFace/ArcFace: 112x112
   - FaceNet: 160x160
   - VGGFace: 224x224

2. **Normalization:**
   - Face models: Mean=[0.5, 0.5, 0.5], Std=[0.5, 0.5, 0.5] → [-1, 1]
   - ImageNet: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]

3. **Labels:**
   - VGGFace2: 1=genuine pair, 0=impostor pair
   - CelebA: 1=attribute present, -1=attribute absent

## License

This code is part of the XAI dissertation project. Dataset licenses apply separately.
