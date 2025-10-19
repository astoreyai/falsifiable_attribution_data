# Data Package Quick Reference

## Import Everything You Need

```python
from data import (
    # Datasets
    VGGFace2Dataset,
    CelebADataset,

    # Transforms
    get_insightface_transforms,
    get_facenet_transforms,
    get_vggface_transforms,
    get_default_transforms,

    # Utilities
    collate_verification_pairs,
    collate_attribute_batch,
    compute_verification_metrics,
    normalize_embeddings,
    cosine_similarity,
)
```

## VGGFace2 - 3 Lines

```python
from data import VGGFace2Dataset, get_insightface_transforms, collate_verification_pairs
from torch.utils.data import DataLoader

dataset = VGGFace2Dataset('/datasets/vggface2', split='test',
                          transform=get_insightface_transforms(), n_pairs=200)
loader = DataLoader(dataset, batch_size=16, collate_fn=collate_verification_pairs)
img1, img2, labels = next(iter(loader))  # [16,3,112,112], [16,3,112,112], [16]
```

## CelebA - 3 Lines

```python
from data import CelebADataset, get_default_transforms, collate_attribute_batch
from torch.utils.data import DataLoader

dataset = CelebADataset('/datasets/celeba', split='test',
                        transform=get_default_transforms(224), n_samples=1000)
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_attribute_batch)
images, attrs = next(iter(loader))  # [32,3,224,224], [32,40]
```

## Common Patterns

### Face Verification Experiment

```python
from data import VGGFace2Dataset, get_insightface_transforms
from data import cosine_similarity, compute_verification_metrics

# Load data
dataset = VGGFace2Dataset('/datasets/vggface2', n_pairs=200,
                          transform=get_insightface_transforms())

# Get embeddings from your model
img1, img2, labels = dataset[0]['img1'], dataset[0]['img2'], dataset[0]['label']
emb1 = model(img1.unsqueeze(0))
emb2 = model(img2.unsqueeze(0))

# Compute similarity
similarity = cosine_similarity(emb1, emb2)

# Evaluate performance
metrics = compute_verification_metrics(similarities, labels)
print(f"Accuracy: {metrics['best_accuracy']:.3f}")
```

### Attribute Prediction

```python
from data import CelebADataset, get_default_transforms

# Load specific attributes
dataset = CelebADataset('/datasets/celeba', split='train',
                        attributes=['Male', 'Smiling', 'Young'],
                        transform=get_default_transforms(224))

# Get images with specific attribute
male_indices = dataset.get_images_with_attribute('Male', value=1)
print(f"Found {len(male_indices)} male faces")
```

### Custom Transforms

```python
from data import get_default_transforms

# With augmentation for training
train_transform = get_default_transforms(image_size=112, augment=True)

# No augmentation for testing
test_transform = get_default_transforms(image_size=112, augment=False)
```

## Transform Options

| Model | Transform Function | Image Size |
|-------|-------------------|------------|
| InsightFace/ArcFace | `get_insightface_transforms()` | 112x112 |
| FaceNet | `get_facenet_transforms()` | 160x160 |
| VGGFace | `get_vggface_transforms()` | 224x224 |
| Custom | `get_default_transforms(size)` | Any |

## Dataset Outputs

### VGGFace2

```python
sample = dataset[0]
# sample = {
#     'img1': Tensor[3, 112, 112],
#     'img2': Tensor[3, 112, 112],
#     'label': 1 or 0,  # 1=genuine, 0=impostor
#     'img1_path': '/path/to/img1.jpg',
#     'img2_path': '/path/to/img2.jpg'
# }
```

### CelebA

```python
image, attributes = dataset[0]
# image: Tensor[3, 224, 224]
# attributes: Tensor[40] with values in {-1, 1}
#   -1 = attribute absent
#    1 = attribute present
```

## Verification Metrics

```python
from data import compute_verification_metrics
import numpy as np

similarities = np.array([0.8, 0.3, 0.9, 0.2, 0.7])
labels = np.array([1, 0, 1, 0, 1])

metrics = compute_verification_metrics(similarities, labels)
print(metrics['best_threshold'])   # 0.5
print(metrics['best_accuracy'])    # 1.0
print(metrics['best_tpr'])         # 1.0
print(metrics['best_fpr'])         # 0.0
```

## Statistics

```python
# VGGFace2
stats = dataset.get_dataset_statistics()
print(stats['n_identities'])
print(stats['genuine_ratio'])

# CelebA
stats = dataset.get_dataset_statistics()
print(stats['attribute_statistics']['Male']['positive_ratio'])
```

## Troubleshooting

### Dataset Not Found

Loaders auto-search these paths:
1. Your provided path
2. `/datasets/<dataset_name>/`
3. `~/datasets/<dataset_name>/`
4. `/data/<dataset_name>/`
5. `~/data/<dataset_name>/`

Place datasets in any of these locations.

### No Pairs Generated

For VGGFace2, ensure each identity has at least 2 images.

### Attribute Not Found

Use exact names from the 40 CelebA attributes (case-sensitive):
```python
dataset.get_attribute_names()  # Show available attributes
```

## File Locations

- Package: `/home/aaron/projects/xai/data/`
- Full docs: `/home/aaron/projects/xai/data/README.md`
- Examples: `/home/aaron/projects/xai/examples/dataset_usage.py`
- Summary: `/home/aaron/projects/xai/IMPLEMENTATION_SUMMARY.md`
