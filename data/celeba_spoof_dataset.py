"""
CelebA-Spoof Dataset Loader for Anti-Spoofing Experiments

This module provides a PyTorch dataset class for loading CelebA-Spoof,
a large-scale face anti-spoofing dataset with 625,537 images.

Dataset: CelebA-Spoof (ECCV 2020)
Paper: Zhang et al., "CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset
       with Rich Annotations"
ArXiv: https://arxiv.org/abs/2007.12342

Dataset Structure:
------------------
celeba_spoof/
├── Data/
│   ├── train/         # Training images
│   ├── val/           # Validation images
│   └── test/          # Test images
├── metas/
│   ├── intra_test/
│   │   ├── train_label.json
│   │   ├── test_label.json
│   │   └── ...
│   └── List_of_testing_images.txt
└── README.md

Alternative: Hugging Face Dataset
----------------------------------
If using the Hugging Face version:
Dataset: nguyenkhoa/celeba-spoof-for-face-antispoofing-test
- Preprocessed and cropped faces
- Binary labels (0=live, 1=spoof)
- Easier to download (4.95 GB for test split)

Usage:
------
# Official dataset structure
dataset = CelebASpoofDataset(
    root='/path/to/celeba_spoof',
    split='train',
    transform=transforms
)

# Hugging Face version
dataset = CelebASpoofDataset(
    root='/path/to/celeba_spoof',
    split='test',
    source='huggingface',
    transform=transforms
)

# Iterate
for sample in dataset:
    image = sample['image']          # PIL Image or Tensor
    label = sample['label']          # 0=live, 1=spoof
    spoof_type = sample['spoof_type']  # Type of attack
"""

import os
import json
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
import warnings

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CelebASpoofDataset(Dataset):
    """
    PyTorch dataset for CelebA-Spoof anti-spoofing dataset.

    Supports both official dataset structure and Hugging Face version.

    Attributes:
        root (Path): Root directory of dataset
        split (str): Data split ('train', 'val', 'test')
        source (str): Dataset source ('official' or 'huggingface')
        transform (Callable): Optional transform for images

    Labels:
        0 = Live (genuine/bonafide face)
        1 = Spoof (fake face - print, replay, or 3D mask)

    Spoof Types:
        'live' = Real face
        'print' = Photo print attack
        'replay' = Video replay attack
        '3d_mask' = 3D mask attack
        'unknown' = Type not specified
    """

    # Spoof type mapping (based on CelebA-Spoof paper)
    SPOOF_TYPES = {
        0: 'live',
        1: 'print',
        2: 'replay',
        3: '3d_mask',
        4: 'paper_cut',
        5: 'half_mask',
        6: 'silicone_mask',
        7: 'transparent_mask',
        8: 'mannequin',
        9: 'poster',
        10: 'unknown'
    }

    def __init__(
        self,
        root: str,
        split: str = 'train',
        source: str = 'official',
        transform: Optional[Callable] = None,
        return_path: bool = False
    ):
        """
        Initialize CelebA-Spoof dataset.

        Args:
            root: Root directory containing the dataset
            split: Data split ('train', 'val', 'test')
            source: Dataset source ('official' or 'huggingface')
            transform: Optional transform to apply to images
            return_path: If True, also return image path in sample
        """
        self.root = Path(root)
        self.split = split
        self.source = source.lower()
        self.transform = transform
        self.return_path = return_path

        # Validate inputs
        assert split in ['train', 'val', 'test'], \
            f"split must be 'train', 'val', or 'test', got '{split}'"
        assert source in ['official', 'huggingface'], \
            f"source must be 'official' or 'huggingface', got '{source}'"

        # Load dataset based on source
        if self.source == 'official':
            self._load_official_dataset()
        elif self.source == 'huggingface':
            self._load_huggingface_dataset()

        print(f"CelebA-Spoof Dataset loaded:")
        print(f"  Split: {split}")
        print(f"  Source: {source}")
        print(f"  Samples: {len(self)}")
        if hasattr(self, 'num_live'):
            print(f"  Live: {self.num_live}")
            print(f"  Spoof: {self.num_spoof}")

    def _load_official_dataset(self):
        """Load dataset from official CelebA-Spoof structure."""
        # Image directory
        self.image_dir = self.root / 'Data' / self.split

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}\n"
                f"Please download CelebA-Spoof dataset to {self.root}\n"
                f"See: python data/download_celeba_spoof.py"
            )

        # Metadata directory
        meta_dir = self.root / 'metas' / 'intra_test'
        label_file = meta_dir / f'{self.split}_label.json'

        # Load labels if available
        self.labels_dict = {}
        self.spoof_types_dict = {}

        if label_file.exists():
            with open(label_file, 'r') as f:
                annotations = json.load(f)

            # Parse annotations
            # Format: {"image_name": {"label": 0/1, "spoof_type": X, ...}}
            for img_name, anno in annotations.items():
                self.labels_dict[img_name] = anno.get('label', -1)
                spoof_type_id = anno.get('spoof_type', 10)
                self.spoof_types_dict[img_name] = self.SPOOF_TYPES.get(
                    spoof_type_id, 'unknown'
                )
        else:
            warnings.warn(
                f"Label file not found: {label_file}\n"
                f"Will attempt to infer labels from directory structure"
            )

        # Get all image paths
        self.image_paths = sorted(list(self.image_dir.rglob('*.jpg')))
        if not self.image_paths:
            self.image_paths = sorted(list(self.image_dir.rglob('*.png')))

        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found in {self.image_dir}\n"
                f"Expected .jpg or .png files"
            )

        # Count statistics
        self.num_live = sum(
            1 for p in self.image_paths
            if self.labels_dict.get(p.name, -1) == 0
        )
        self.num_spoof = sum(
            1 for p in self.image_paths
            if self.labels_dict.get(p.name, -1) == 1
        )

    def _load_huggingface_dataset(self):
        """Load dataset from Hugging Face format."""
        # Try loading from Hugging Face datasets cache
        try:
            from datasets import load_from_disk

            hf_data_dir = self.root / 'huggingface_data'

            if not hf_data_dir.exists():
                raise FileNotFoundError(
                    f"Hugging Face data not found: {hf_data_dir}\n"
                    f"Please download first:\n"
                    f"  from datasets import load_dataset\n"
                    f"  dataset = load_dataset('nguyenkhoa/celeba-spoof-for-face-antispoofing-test')\n"
                    f"  dataset.save_to_disk('{hf_data_dir}')"
                )

            dataset = load_from_disk(str(hf_data_dir))

            # Get appropriate split
            if self.split in dataset:
                self.hf_dataset = dataset[self.split]
            elif 'test' in dataset and self.split in ['val', 'test']:
                # Hugging Face version may only have test split
                self.hf_dataset = dataset['test']
                warnings.warn(
                    f"Split '{self.split}' not found, using 'test' split"
                )
            else:
                raise ValueError(
                    f"Split '{self.split}' not found in Hugging Face dataset.\n"
                    f"Available splits: {list(dataset.keys())}"
                )

            # Count statistics
            self.num_live = sum(
                1 for i in range(len(self.hf_dataset))
                if self.hf_dataset[i].get('label', -1) == 0
            )
            self.num_spoof = len(self.hf_dataset) - self.num_live

        except ImportError:
            raise ImportError(
                "Hugging Face datasets library required for source='huggingface'.\n"
                "Install with: pip install datasets"
            )

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        if self.source == 'official':
            return len(self.image_paths)
        else:  # huggingface
            return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - image: PIL Image or Tensor (if transform applied)
                - label: 0 (live) or 1 (spoof)
                - spoof_type: Type of spoofing attack (str)
                - image_path: Path to image file (if return_path=True)
        """
        if self.source == 'official':
            return self._get_official_sample(idx)
        else:  # huggingface
            return self._get_huggingface_sample(idx)

    def _get_official_sample(self, idx: int) -> Dict[str, Any]:
        """Get sample from official dataset structure."""
        img_path = self.image_paths[idx]
        img_name = img_path.name

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Error loading image {img_path}: {e}")

        # Get label
        label = self.labels_dict.get(img_name, -1)

        # Get spoof type
        spoof_type = self.spoof_types_dict.get(img_name, 'unknown')

        # Apply transform
        if self.transform is not None:
            image = self.transform(image)

        sample = {
            'image': image,
            'label': label,
            'spoof_type': spoof_type
        }

        if self.return_path:
            sample['image_path'] = str(img_path)

        return sample

    def _get_huggingface_sample(self, idx: int) -> Dict[str, Any]:
        """Get sample from Hugging Face dataset."""
        hf_sample = self.hf_dataset[idx]

        # Extract image
        image = hf_sample.get('image')
        if isinstance(image, dict) and 'bytes' in image:
            # Handle bytes format
            from io import BytesIO
            image = Image.open(BytesIO(image['bytes'])).convert('RGB')
        elif not isinstance(image, Image.Image):
            # Convert to PIL if needed
            image = Image.fromarray(np.array(image)).convert('RGB')

        # Get label
        label = hf_sample.get('label', -1)

        # Get spoof type (may not be available in HF version)
        label_name = hf_sample.get('label_name', 'unknown')
        spoof_type = 'spoof' if label == 1 else 'live'
        if label_name not in ['live', 'spoof']:
            spoof_type = label_name

        # Apply transform
        if self.transform is not None:
            image = self.transform(image)

        sample = {
            'image': image,
            'label': label,
            'spoof_type': spoof_type
        }

        if self.return_path:
            sample['image_path'] = f"huggingface:{self.split}:{idx}"

        return sample

    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of live vs spoof samples."""
        return {
            'live': self.num_live,
            'spoof': self.num_spoof,
            'total': len(self)
        }

    def get_spoof_type_distribution(self) -> Dict[str, int]:
        """Get distribution of spoof types (official dataset only)."""
        if self.source != 'official':
            warnings.warn(
                "Spoof type distribution only available for official dataset"
            )
            return {}

        from collections import Counter
        spoof_types = [
            self.spoof_types_dict.get(p.name, 'unknown')
            for p in self.image_paths
        ]
        return dict(Counter(spoof_types))


def create_celeba_spoof_dataloaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    source: str = 'official',
    transform_train: Optional[Callable] = None,
    transform_test: Optional[Callable] = None
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train/val/test dataloaders for CelebA-Spoof.

    Args:
        root: Root directory of dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        source: Dataset source ('official' or 'huggingface')
        transform_train: Transform for training set
        transform_test: Transform for val/test sets

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = CelebASpoofDataset(
        root=root,
        split='train',
        source=source,
        transform=transform_train
    )

    val_dataset = CelebASpoofDataset(
        root=root,
        split='val',
        source=source,
        transform=transform_test
    )

    test_dataset = CelebASpoofDataset(
        root=root,
        split='test',
        source=source,
        transform=transform_test
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# Example usage and testing
if __name__ == '__main__':
    import sys

    print("=" * 80)
    print("CelebA-Spoof Dataset Loader Test")
    print("=" * 80)

    # Default root directory
    default_root = '/home/aaron/projects/xai/data/celeba_spoof'
    root_dir = sys.argv[1] if len(sys.argv) > 1 else default_root

    print(f"\nRoot directory: {root_dir}")
    print(f"Testing dataset loader...\n")

    # Try official dataset first
    try:
        print("[1] Testing official dataset structure...")
        dataset = CelebASpoofDataset(
            root=root_dir,
            split='test',
            source='official'
        )

        print(f"\n✓ Official dataset loaded successfully!")
        print(f"  Total samples: {len(dataset)}")

        # Test first sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nFirst sample:")
            print(f"  Image type: {type(sample['image'])}")
            if hasattr(sample['image'], 'size'):
                print(f"  Image size: {sample['image'].size}")
            print(f"  Label: {sample['label']} ({'live' if sample['label'] == 0 else 'spoof'})")
            print(f"  Spoof type: {sample['spoof_type']}")

        # Class distribution
        dist = dataset.get_class_distribution()
        print(f"\nClass distribution:")
        print(f"  Live: {dist['live']} ({dist['live']/dist['total']*100:.1f}%)")
        print(f"  Spoof: {dist['spoof']} ({dist['spoof']/dist['total']*100:.1f}%)")

        # Spoof type distribution
        spoof_dist = dataset.get_spoof_type_distribution()
        if spoof_dist:
            print(f"\nSpoof type distribution:")
            for spoof_type, count in sorted(spoof_dist.items(), key=lambda x: x[1], reverse=True):
                print(f"  {spoof_type}: {count}")

    except FileNotFoundError as e:
        print(f"\n✗ Official dataset not found:")
        print(f"  {e}")
    except Exception as e:
        print(f"\n✗ Error loading official dataset:")
        print(f"  {e}")

    # Try Hugging Face dataset
    print("\n" + "=" * 80)
    try:
        print("[2] Testing Hugging Face dataset structure...")
        dataset_hf = CelebASpoofDataset(
            root=root_dir,
            split='test',
            source='huggingface'
        )

        print(f"\n✓ Hugging Face dataset loaded successfully!")
        print(f"  Total samples: {len(dataset_hf)}")

        # Test first sample
        if len(dataset_hf) > 0:
            sample = dataset_hf[0]
            print(f"\nFirst sample:")
            print(f"  Image type: {type(sample['image'])}")
            if hasattr(sample['image'], 'size'):
                print(f"  Image size: {sample['image'].size}")
            print(f"  Label: {sample['label']} ({'live' if sample['label'] == 0 else 'spoof'})")
            print(f"  Spoof type: {sample['spoof_type']}")

        # Class distribution
        dist = dataset_hf.get_class_distribution()
        print(f"\nClass distribution:")
        print(f"  Live: {dist['live']} ({dist['live']/dist['total']*100:.1f}%)")
        print(f"  Spoof: {dist['spoof']} ({dist['spoof']/dist['total']*100:.1f}%)")

    except FileNotFoundError as e:
        print(f"\n✗ Hugging Face dataset not found:")
        print(f"  {e}")
    except ImportError as e:
        print(f"\n✗ Hugging Face datasets library not installed:")
        print(f"  {e}")
    except Exception as e:
        print(f"\n✗ Error loading Hugging Face dataset:")
        print(f"  {e}")

    print("\n" + "=" * 80)
    print("DOWNLOAD INSTRUCTIONS")
    print("=" * 80)
    print("\nIf dataset not found, download using:")
    print(f"  python /home/aaron/projects/xai/data/download_celeba_spoof.py")
    print("\nOr see research documentation:")
    print(f"  /home/aaron/projects/xai/data/CELEBA_SPOOF_RESEARCH.md")
    print("=" * 80)
