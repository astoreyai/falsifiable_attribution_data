#!/usr/bin/env python3
"""
CelebA Dataset Download Script

Downloads CelebA dataset for face recognition experiments using PyTorch torchvision.

Dataset Details:
- 202,599 face images
- 10,177 identities
- 40 binary attributes per image
- Size: ~1.5 GB

Source: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
License: Non-commercial research only

Usage:
    python data/download_celeba.py
    python data/download_celeba.py --root /path/to/download
"""

import os
import sys
from pathlib import Path
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_celeba(root_dir: str = "/home/aaron/projects/xai/data/celeba"):
    """
    Download CelebA dataset using PyTorch torchvision.

    Args:
        root_dir: Directory to download dataset to

    Returns:
        bool: True if successful, False otherwise
    """
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    logger.info(f"CelebA Download Script")
    logger.info(f"=" * 60)
    logger.info(f"Target directory: {root}")
    logger.info(f"Expected size: ~1.5 GB")
    logger.info(f"Expected time: 30-60 minutes")
    logger.info(f"=" * 60)

    # Method 1: Try torchvision automatic download
    logger.info("\n[Method 1] Attempting torchvision automatic download...")

    try:
        import torchvision.datasets as datasets

        logger.info("Downloading CelebA dataset...")
        logger.info("This may take 30-60 minutes depending on network speed.")

        celeba = datasets.CelebA(
            root=str(root),
            split='all',  # Download all splits
            target_type='identity',
            download=True
        )

        logger.info(f"✓ SUCCESS: Downloaded {len(celeba)} images")
        logger.info(f"✓ Location: {root}/celeba/")

        # Verify structure
        expected_files = [
            root / "celeba" / "img_align_celeba",
            root / "celeba" / "list_attr_celeba.txt",
            root / "celeba" / "list_eval_partition.txt",
            root / "celeba" / "identity_CelebA.txt"
        ]

        logger.info("\nVerifying dataset structure:")
        all_present = True
        for path in expected_files:
            exists = path.exists()
            status = "✓" if exists else "✗"
            logger.info(f"  {status} {path.name}")
            if not exists:
                all_present = False

        if all_present:
            logger.info("\n✓ Dataset structure verified!")
            logger.info(f"\nDataset ready for use:")
            logger.info(f"  from data.celeba_dataset import CelebADataset")
            logger.info(f"  dataset = CelebADataset(root_dir='{root}')")
            return True
        else:
            logger.warning("\n⚠ Some files missing. Dataset may be incomplete.")
            return False

    except ImportError:
        logger.error("✗ torchvision not installed")
        logger.info("  Install: pip install torchvision")
        return False

    except Exception as e:
        logger.error(f"✗ Torchvision download failed: {e}")
        logger.info("\n[Fallback] Manual download instructions:")
        print_manual_instructions(root_dir)
        return False


def print_manual_instructions(root_dir: str):
    """Print manual download instructions."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("MANUAL DOWNLOAD INSTRUCTIONS")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Option A: Official Source")
    logger.info("  1. Visit: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    logger.info("  2. Download files:")
    logger.info("     - img_align_celeba.zip (1.5 GB)")
    logger.info("     - list_attr_celeba.txt")
    logger.info("     - list_eval_partition.txt")
    logger.info("     - identity_CelebA.txt")
    logger.info(f"  3. Extract to: {root_dir}/celeba/")
    logger.info("")
    logger.info("Option B: Kaggle API")
    logger.info("  1. Install: pip install kaggle")
    logger.info("  2. Configure API token (from https://www.kaggle.com/settings)")
    logger.info("  3. Run:")
    logger.info("     kaggle datasets download -d jessicali9530/celeba-dataset")
    logger.info(f"     unzip celeba-dataset.zip -d {root_dir}/")
    logger.info("")
    logger.info("Option C: Google Drive Mirror")
    logger.info("  1. Search for CelebA on Google Drive mirrors")
    logger.info("  2. Download img_align_celeba.zip and annotation files")
    logger.info(f"  3. Extract to: {root_dir}/celeba/")
    logger.info("")
    logger.info("=" * 60)


def verify_dataset(root_dir: str) -> bool:
    """
    Verify that CelebA dataset is properly downloaded.

    Args:
        root_dir: Root directory of dataset

    Returns:
        bool: True if dataset is valid, False otherwise
    """
    root = Path(root_dir)

    # Check required files
    required = [
        root / "celeba" / "img_align_celeba",
        root / "celeba" / "list_attr_celeba.txt",
        root / "celeba" / "list_eval_partition.txt"
    ]

    logger.info("Verifying CelebA dataset...")

    for path in required:
        if not path.exists():
            logger.error(f"✗ Missing: {path}")
            return False
        logger.info(f"✓ Found: {path.name}")

    # Count images
    img_dir = root / "celeba" / "img_align_celeba"
    if img_dir.exists():
        images = list(img_dir.glob("*.jpg"))
        logger.info(f"✓ Found {len(images)} images")

        if len(images) < 200000:
            logger.warning(f"⚠ Expected ~202,599 images, found {len(images)}")
            logger.warning("  Dataset may be incomplete")
            return False

    logger.info("✓ Dataset verification successful!")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download CelebA dataset for face recognition experiments"
    )
    parser.add_argument(
        '--root',
        type=str,
        default='/home/aaron/projects/xai/data/celeba',
        help='Root directory to download dataset (default: %(default)s)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing dataset, do not download'
    )

    args = parser.parse_args()

    if args.verify_only:
        # Just verify
        success = verify_dataset(args.root)
        sys.exit(0 if success else 1)
    else:
        # Download
        success = download_celeba(args.root)

        if success:
            logger.info("\n" + "=" * 60)
            logger.info("✓ CelebA dataset ready!")
            logger.info("=" * 60)
            logger.info("\nNext steps:")
            logger.info("  1. Test dataset loader:")
            logger.info("     python -c 'from data.celeba_dataset import CelebADataset; ")
            logger.info(f"               d = CelebADataset(\"{args.root}\"); print(len(d))'")
            logger.info("")
            logger.info("  2. Run multi-dataset experiments:")
            logger.info("     python experiments/run_multidataset_experiment_6_1.py")
            logger.info("")
            sys.exit(0)
        else:
            logger.error("\n✗ Download failed. See manual instructions above.")
            sys.exit(1)


if __name__ == "__main__":
    main()
