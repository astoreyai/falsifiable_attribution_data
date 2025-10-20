#!/usr/bin/env python3
"""
CelebA Dataset Download Script (Enhanced)

Downloads CelebA dataset for face recognition experiments with multiple methods.

Dataset Details:
- 202,599 face images (aligned & cropped, 178x218 pixels)
- 10,177 unique celebrity identities
- 40 binary attributes per image (gender, age, hair, facial features)
- 5 facial landmarks per image (eyes, nose, mouth corners)
- Bounding boxes and train/val/test split
- Size: ~1.5 GB (images) + ~200 MB (annotations)

Source: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
License: Non-commercial research only

Usage:
    python data/download_celeba.py                         # Auto-download via torchvision
    python data/download_celeba.py --method kaggle         # Download via Kaggle API
    python data/download_celeba.py --method manual         # Show manual instructions
    python data/download_celeba.py --verify                # Verify existing download
    python data/download_celeba.py --analyze               # Analyze dataset statistics
"""

import os
import sys
from pathlib import Path
import argparse
import logging
from typing import Optional, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_via_torchvision(root_dir: str = "/home/aaron/projects/xai/data/celeba") -> bool:
    """
    Download CelebA dataset using PyTorch torchvision (Method 1 - Easiest).

    This is the recommended method as it handles downloads automatically
    and includes all necessary annotations.

    Args:
        root_dir: Directory to download dataset to

    Returns:
        bool: True if successful, False otherwise
    """
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("CelebA Download via Torchvision")
    logger.info("=" * 70)
    logger.info(f"Target directory: {root}")
    logger.info(f"Expected size: ~1.5 GB (images) + ~200 MB (annotations)")
    logger.info(f"Expected time: 30-60 minutes (depending on network speed)")
    logger.info("=" * 70)

    try:
        import torchvision.datasets as datasets

        logger.info("\nDownloading CelebA dataset...")
        logger.info("Components: images + attributes + landmarks + identities + partitions")

        # Download with all target types to ensure we get all files
        celeba = datasets.CelebA(
            root=str(root),
            split='all',  # Download all splits (train/val/test)
            target_type=['attr', 'identity', 'bbox', 'landmarks'],
            download=True
        )

        logger.info(f"\n✓ SUCCESS: Downloaded {len(celeba)} images")
        logger.info(f"✓ Location: {root}/celeba/")

        # Verify complete structure
        expected_files = {
            "Images": root / "celeba" / "img_align_celeba",
            "Attributes": root / "celeba" / "list_attr_celeba.txt",
            "Landmarks": root / "celeba" / "list_landmarks_align_celeba.txt",
            "Identities": root / "celeba" / "identity_CelebA.txt",
            "Bounding Boxes": root / "celeba" / "list_bbox_celeba.txt",
            "Partitions": root / "celeba" / "list_eval_partition.txt"
        }

        logger.info("\nVerifying dataset structure:")
        all_present = True
        for name, path in expected_files.items():
            exists = path.exists()
            status = "✓" if exists else "✗"

            if path.is_dir() and exists:
                count = len(list(path.glob("*.jpg")))
                logger.info(f"  {status} {name:20s}: {count:,} images")
            elif exists:
                size = path.stat().st_size / 1024 / 1024
                logger.info(f"  {status} {name:20s}: {size:.1f} MB")
            else:
                logger.info(f"  {status} {name:20s}: NOT FOUND")
                all_present = False

        if all_present:
            logger.info("\n✓ Dataset structure verified!")
            logger.info("\nDataset includes:")
            logger.info("  - 202,599 aligned face images (178x218 pixels)")
            logger.info("  - 10,177 unique celebrity identities")
            logger.info("  - 40 binary attributes per image")
            logger.info("  - 5 facial landmarks per image")
            logger.info("  - Train/val/test split")
            return True
        else:
            logger.warning("\n⚠ Some files missing. Dataset may be incomplete.")
            return False

    except ImportError:
        logger.error("✗ torchvision not installed")
        logger.info("  Install: pip install torch torchvision")
        return False

    except Exception as e:
        logger.error(f"✗ Torchvision download failed: {e}")
        logger.info("\n[Fallback] Try alternate download methods:")
        logger.info("  python data/download_celeba.py --method kaggle")
        logger.info("  python data/download_celeba.py --method manual")
        return False


def download_via_kaggle(root_dir: str) -> bool:
    """
    Download CelebA dataset using Kaggle API (Method 2).

    Requires Kaggle account and API key configured.

    Args:
        root_dir: Directory to download dataset to

    Returns:
        bool: True if successful, False otherwise
    """
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("CelebA Download via Kaggle API")
    logger.info("=" * 70)

    try:
        import kaggle

        logger.info("Downloading from Kaggle: jessicali9530/celeba-dataset")
        logger.info(f"Target directory: {root}")

        # Download and unzip
        kaggle.api.dataset_download_files(
            'jessicali9530/celeba-dataset',
            path=str(root),
            unzip=True
        )

        logger.info("✓ Kaggle download complete")
        return True

    except ImportError:
        logger.error("✗ Kaggle package not installed")
        logger.info("  Install: pip install kaggle")
        logger.info("  Setup API key: https://www.kaggle.com/docs/api")
        return False

    except Exception as e:
        logger.error(f"✗ Kaggle download failed: {e}")
        logger.info("\nKaggle API Setup:")
        logger.info("  1. Create Kaggle account at https://www.kaggle.com")
        logger.info("  2. Go to https://www.kaggle.com/settings/account")
        logger.info("  3. Click 'Create New API Token'")
        logger.info("  4. Move kaggle.json to ~/.kaggle/kaggle.json")
        logger.info("  5. chmod 600 ~/.kaggle/kaggle.json")
        return False


def print_manual_instructions(root_dir: str):
    """Print manual download instructions (Method 3)."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("MANUAL DOWNLOAD INSTRUCTIONS")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Step 1: Visit Official Source")
    logger.info("  http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    logger.info("")
    logger.info("Step 2: Download Required Files")
    logger.info("  - img_align_celeba.zip              (1.4 GB - aligned face images)")
    logger.info("  - list_attr_celeba.txt              (40 binary attributes)")
    logger.info("  - list_landmarks_align_celeba.txt   (5 facial landmarks)")
    logger.info("  - identity_CelebA.txt               (celebrity identity labels)")
    logger.info("  - list_bbox_celeba.txt              (face bounding boxes)")
    logger.info("  - list_eval_partition.txt           (train/val/test split)")
    logger.info("")
    logger.info("Step 3: Extract and Organize")
    logger.info(f"  Extract img_align_celeba.zip")
    logger.info(f"  Move all files to: {root_dir}/celeba/")
    logger.info("")
    logger.info("Expected Directory Structure:")
    logger.info(f"  {root_dir}/")
    logger.info("    └── celeba/")
    logger.info("        ├── img_align_celeba/")
    logger.info("        │   ├── 000001.jpg")
    logger.info("        │   ├── 000002.jpg")
    logger.info("        │   └── ... (202,599 images)")
    logger.info("        ├── list_attr_celeba.txt")
    logger.info("        ├── list_landmarks_align_celeba.txt")
    logger.info("        ├── identity_CelebA.txt")
    logger.info("        ├── list_bbox_celeba.txt")
    logger.info("        └── list_eval_partition.txt")
    logger.info("")
    logger.info("Step 4: Verify Download")
    logger.info("  python data/download_celeba.py --verify")
    logger.info("")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Alternative Sources:")
    logger.info("")
    logger.info("  Kaggle Dataset:")
    logger.info("    https://www.kaggle.com/jessicali9530/celeba-dataset")
    logger.info("    kaggle datasets download -d jessicali9530/celeba-dataset")
    logger.info("")
    logger.info("  Google Drive (may require manual search):")
    logger.info("    Search: 'CelebA dataset Google Drive'")
    logger.info("")
    logger.info("=" * 70)


def verify_dataset(root_dir: str) -> bool:
    """
    Verify that CelebA dataset is properly downloaded and complete.

    Args:
        root_dir: Root directory of dataset

    Returns:
        bool: True if dataset is valid and complete, False otherwise
    """
    root = Path(root_dir)

    logger.info("=" * 70)
    logger.info("CelebA Dataset Verification")
    logger.info("=" * 70)

    # Check all expected files
    expected_files = {
        "Images": root / "celeba" / "img_align_celeba",
        "Attributes": root / "celeba" / "list_attr_celeba.txt",
        "Landmarks": root / "celeba" / "list_landmarks_align_celeba.txt",
        "Identities": root / "celeba" / "identity_CelebA.txt",
        "Bounding Boxes": root / "celeba" / "list_bbox_celeba.txt",
        "Partitions": root / "celeba" / "list_eval_partition.txt"
    }

    all_present = True
    for name, path in expected_files.items():
        if not path.exists():
            logger.error(f"✗ {name:20s}: NOT FOUND ({path})")
            all_present = False
        elif path.is_dir():
            count = len(list(path.glob("*.jpg")))
            logger.info(f"✓ {name:20s}: {count:,} images")
            if count < 200000:
                logger.warning(f"  ⚠ Expected ~202,599 images, found {count}")
                all_present = False
        else:
            size = path.stat().st_size / 1024 / 1024
            logger.info(f"✓ {name:20s}: {size:.1f} MB")

    # Calculate total disk usage
    celeba_dir = root / "celeba"
    if celeba_dir.exists():
        total_size = sum(f.stat().st_size for f in celeba_dir.rglob('*') if f.is_file())
        total_mb = total_size / 1024 / 1024
        total_gb = total_mb / 1024
        logger.info(f"\nTotal disk usage: {total_gb:.2f} GB ({total_mb:.1f} MB)")

    logger.info("=" * 70)

    if all_present:
        logger.info("✓ Dataset verification PASSED - All components present!")
        logger.info("\nDataset ready for use:")
        logger.info("  from data.celeba_dataset import CelebADataset")
        logger.info(f"  dataset = CelebADataset(root_dir='{root_dir}')")
        return True
    else:
        logger.error("✗ Dataset verification FAILED - Missing components!")
        logger.info("\nTo download missing components:")
        logger.info("  python data/download_celeba.py")
        return False


def analyze_dataset(root_dir: str):
    """
    Analyze CelebA dataset statistics and characteristics.

    Args:
        root_dir: Root directory of dataset
    """
    root = Path(root_dir) / "celeba"

    logger.info("=" * 70)
    logger.info("CelebA Dataset Analysis")
    logger.info("=" * 70)

    # Count images
    img_dir = root / "img_align_celeba"
    if img_dir.exists():
        images = list(img_dir.glob("*.jpg"))
        logger.info(f"\nImages: {len(images):,}")
    else:
        logger.error("Image directory not found!")
        return

    # Analyze attributes
    attr_file = root / "list_attr_celeba.txt"
    if attr_file.exists():
        with open(attr_file, 'r') as f:
            lines = f.readlines()

        # Parse header
        n_images = int(lines[0].strip())
        attr_names = lines[1].strip().split()

        logger.info(f"\nAttributes: {len(attr_names)} binary labels")
        logger.info(f"Attribute names: {', '.join(attr_names[:5])}...")

        # Parse first few samples for distribution
        logger.info("\nAttribute Distribution (first 1000 images):")
        attr_data = []
        for line in lines[2:1002]:  # First 1000 images
            parts = line.strip().split()
            values = [int(v) for v in parts[1:]]
            attr_data.append(values)

        import numpy as np
        attr_array = np.array(attr_data)

        for i, attr in enumerate(attr_names[:10]):  # Show first 10 attributes
            positive = (attr_array[:, i] == 1).sum()
            pct = positive / len(attr_array) * 100
            logger.info(f"  {attr:20s}: {pct:5.1f}% positive")

    # Analyze identities
    id_file = root / "identity_CelebA.txt"
    if id_file.exists():
        with open(id_file, 'r') as f:
            identities = [line.strip().split()[1] for line in f]

        from collections import Counter
        id_counts = Counter(identities)
        n_identities = len(id_counts)

        logger.info(f"\nIdentities: {n_identities:,} unique celebrities")
        logger.info(f"Images per identity: {len(identities) / n_identities:.1f} avg")
        logger.info(f"Min images per identity: {min(id_counts.values())}")
        logger.info(f"Max images per identity: {max(id_counts.values())}")

    # Analyze partitions
    partition_file = root / "list_eval_partition.txt"
    if partition_file.exists():
        with open(partition_file, 'r') as f:
            partitions = [int(line.strip().split()[1]) for line in f]

        from collections import Counter
        partition_counts = Counter(partitions)

        logger.info(f"\nDataset Partitions:")
        logger.info(f"  Train (0): {partition_counts[0]:,} images ({partition_counts[0]/len(partitions)*100:.1f}%)")
        logger.info(f"  Val   (1): {partition_counts[1]:,} images ({partition_counts[1]/len(partitions)*100:.1f}%)")
        logger.info(f"  Test  (2): {partition_counts[2]:,} images ({partition_counts[2]/len(partitions)*100:.1f}%)")

    logger.info("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and manage CelebA dataset for face recognition experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download using torchvision (recommended):
    python data/download_celeba.py

  Download using Kaggle API:
    python data/download_celeba.py --method kaggle

  Show manual download instructions:
    python data/download_celeba.py --method manual

  Verify existing download:
    python data/download_celeba.py --verify

  Analyze dataset statistics:
    python data/download_celeba.py --analyze
        """
    )
    parser.add_argument(
        '--root',
        type=str,
        default='/home/aaron/projects/xai/data/celeba',
        help='Root directory to download dataset (default: %(default)s)'
    )
    parser.add_argument(
        '--method',
        choices=['torchvision', 'kaggle', 'manual'],
        default='torchvision',
        help='Download method (default: torchvision)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify existing dataset without downloading'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze dataset statistics'
    )

    args = parser.parse_args()

    if args.verify:
        # Verify only
        success = verify_dataset(args.root)
        sys.exit(0 if success else 1)

    elif args.analyze:
        # Analyze only
        analyze_dataset(args.root)
        sys.exit(0)

    else:
        # Download using selected method
        if args.method == 'torchvision':
            success = download_via_torchvision(args.root)
        elif args.method == 'kaggle':
            success = download_via_kaggle(args.root)
        elif args.method == 'manual':
            print_manual_instructions(args.root)
            sys.exit(0)

        if success:
            logger.info("\n" + "=" * 70)
            logger.info("✓ CelebA dataset ready!")
            logger.info("=" * 70)
            logger.info("\nNext steps:")
            logger.info("  1. Verify dataset:")
            logger.info("     python data/download_celeba.py --verify")
            logger.info("")
            logger.info("  2. Analyze statistics:")
            logger.info("     python data/download_celeba.py --analyze")
            logger.info("")
            logger.info("  3. Test dataset loader:")
            logger.info("     python -c 'from data.celeba_dataset import CelebADataset; \\")
            logger.info(f"                d = CelebADataset(\"{args.root}\"); print(len(d))'")
            logger.info("")
            logger.info("  4. Run multi-dataset experiments:")
            logger.info("     python experiments/run_multidataset_experiment_6_1.py --datasets celeba")
            logger.info("")
            sys.exit(0)
        else:
            logger.error("\n✗ Download failed.")
            logger.info("Try alternate methods:")
            logger.info("  python data/download_celeba.py --method kaggle")
            logger.info("  python data/download_celeba.py --method manual")
            sys.exit(1)


if __name__ == "__main__":
    main()
