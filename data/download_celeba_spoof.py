#!/usr/bin/env python3
"""
CelebA-Spoof Dataset Download Script

CelebA-Spoof: Large-scale face anti-spoofing dataset
- 625,537 images from 10,177 identities
- Real and spoofed faces (10 spoof types)
- Based on CelebA celebrities

Paper: Zhang et al., "CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset
       with Rich Annotations", ECCV 2020
ArXiv: https://arxiv.org/abs/2007.12342

Dataset Characteristics:
- 625,537 images total
- 10,177 subjects (same as CelebA)
- 10 spoof type annotations
- 40 attribute annotations (inherited from CelebA)
- 8 scenes (2 environments * 4 illumination conditions)
- 10+ sensors used for capture

Spoof Types (4 main categories):
1. Print attacks
2. Paper-cut attacks
3. Replay attacks (video display)
4. 3D/Mask attacks

Usage:
    python data/download_celeba_spoof.py
    python data/download_celeba_spoof.py --verify
"""

import os
import argparse
from pathlib import Path
import urllib.request

def download_celeba_spoof(root_dir="/home/aaron/projects/xai/data/celeba-spoof"):
    """Download CelebA-Spoof dataset (requires manual download)."""
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CelebA-Spoof Dataset Download")
    print("=" * 70)
    print()
    print("Dataset: CelebA-Spoof")
    print("Paper: Zhang et al., ECCV 2020")
    print("Images: 625,537 (10,177 subjects)")
    print("Spoof Types: 10 (4 main categories)")
    print()

    # Official sources
    print("OFFICIAL SOURCES:")
    print("-" * 70)

    github_url = "https://github.com/ZhangYuanhan-AI/CelebA-Spoof"
    print(f"1. GitHub Repository: {github_url}")
    print("   - Official ECCV 2020 dataset")
    print("   - Contains download links (Google Drive, Baidu Drive)")
    print("   - Includes baseline code and benchmarks")
    print()

    project_url = "https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Spoof.html"
    print(f"2. Official Project Page: {project_url}")
    print("   - Dataset description")
    print("   - Paper links (ArXiv, ECCV)")
    print("   - Download instructions")
    print()

    kaggle_url = "https://www.kaggle.com/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing"
    print(f"3. Kaggle (Alternative): {kaggle_url}")
    print("   - Community mirror")
    print("   - Requires Kaggle account")
    print("   - May be easier to download")
    print()

    arxiv_url = "https://arxiv.org/abs/2007.12342"
    print(f"4. Paper (ArXiv): {arxiv_url}")
    print("   - Full dataset description")
    print("   - Benchmark results")
    print()

    # Download instructions
    print("=" * 70)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 70)
    print()
    print("CelebA-Spoof requires manual download or registration.")
    print()
    print("RECOMMENDED APPROACH:")
    print()
    print("Option A: GitHub (Official)")
    print("-" * 70)
    print(f"1. Visit: {github_url}")
    print("2. Check README for download links (Google Drive or Baidu Drive)")
    print("3. Download dataset files:")
    print("   - Images (may be split into parts)")
    print("   - Metadata files")
    print("   - Annotation files")
    print("4. Extract to:", str(root))
    print()

    print("Option B: Kaggle (Easier)")
    print("-" * 70)
    print(f"1. Visit: {kaggle_url}")
    print("2. Sign in with Kaggle account (free)")
    print("3. Click 'Download' button")
    print("4. Extract to:", str(root))
    print()

    # Expected structure
    print("=" * 70)
    print("EXPECTED DIRECTORY STRUCTURE")
    print("=" * 70)
    print()
    print(f"{root}/")
    print("  ├── Data/")
    print("  │   ├── train/")
    print("  │   │   ├── 1/")
    print("  │   │   ├── 2/")
    print("  │   │   └── ...")
    print("  │   ├── val/")
    print("  │   └── test/")
    print("  ├── metas/")
    print("  │   ├── intra_test/")
    print("  │   │   ├── train_label.json")
    print("  │   │   ├── test_label.json")
    print("  │   │   └── ...")
    print("  │   └── List_of_testing_images.txt")
    print("  └── README.md")
    print()

    # Usage terms
    print("=" * 70)
    print("USAGE TERMS")
    print("=" * 70)
    print()
    print("The CelebA-Spoof dataset is available for NON-COMMERCIAL")
    print("RESEARCH PURPOSES ONLY.")
    print()
    print("You agree NOT to:")
    print("- Reproduce, duplicate, copy for commercial purposes")
    print("- Sell, trade, or resell the dataset")
    print("- Use for any commercial applications")
    print()
    print("Citation (if used in research):")
    print()
    print("@inproceedings{zhang2020celeba,")
    print("  title={CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset")
    print("         with Rich Annotations},")
    print("  author={Zhang, Yuanhan and Yin, Zhenfei and Li, Yidong and")
    print("          Yin, Guojun and Yan, Junjie and Shao, Jing and Liu, Ziwei},")
    print("  booktitle={European Conference on Computer Vision (ECCV)},")
    print("  year={2020}")
    print("}")
    print()

    # Save download instructions
    instructions_file = root / "DOWNLOAD_INSTRUCTIONS.txt"
    with open(instructions_file, 'w') as f:
        f.write("CelebA-Spoof Download Instructions\n")
        f.write("=" * 70 + "\n\n")
        f.write("Dataset: CelebA-Spoof (ECCV 2020)\n")
        f.write("Authors: Zhang et al.\n")
        f.write("Images: 625,537 (10,177 subjects)\n")
        f.write("Spoof Types: 10 types (4 main categories)\n\n")
        f.write("Official Sources:\n")
        f.write("-" * 70 + "\n")
        f.write(f"GitHub: {github_url}\n")
        f.write(f"Project Page: {project_url}\n")
        f.write(f"Kaggle: {kaggle_url}\n")
        f.write(f"ArXiv Paper: {arxiv_url}\n\n")
        f.write("Download Method:\n")
        f.write("-" * 70 + "\n")
        f.write("1. Visit GitHub repository for official Google Drive links\n")
        f.write("2. OR use Kaggle for easier download (requires account)\n")
        f.write("3. Download all dataset files\n")
        f.write(f"4. Extract to: {root}\n\n")
        f.write("Expected Structure:\n")
        f.write("-" * 70 + "\n")
        f.write("Data/\n")
        f.write("  ├── train/ (training images)\n")
        f.write("  ├── val/ (validation images)\n")
        f.write("  └── test/ (test images)\n")
        f.write("metas/\n")
        f.write("  ├── intra_test/ (annotations)\n")
        f.write("  └── List_of_testing_images.txt\n\n")
        f.write("Usage Terms:\n")
        f.write("-" * 70 + "\n")
        f.write("NON-COMMERCIAL RESEARCH ONLY\n")
        f.write("Cite: Zhang et al., ECCV 2020\n")

    print(f"✓ Instructions saved to: {instructions_file}")
    print()
    print("After downloading, run: python data/download_celeba_spoof.py --verify")
    print()

    return False

def verify_celeba_spoof(root_dir="/home/aaron/projects/xai/data/celeba-spoof"):
    """Verify CelebA-Spoof dataset structure."""
    root = Path(root_dir)

    print("=" * 70)
    print("Verifying CelebA-Spoof Dataset")
    print("=" * 70)
    print()

    checks = {
        "Root directory": root,
        "Data directory": root / "Data",
        "Train data": root / "Data" / "train",
        "Val data": root / "Data" / "val",
        "Test data": root / "Data" / "test",
        "Metadata": root / "metas",
        "Intra-test metadata": root / "metas" / "intra_test",
    }

    all_ok = True
    total_images = 0

    for name, path in checks.items():
        if path.exists():
            if path.is_dir():
                # Count images
                jpg_files = list(path.rglob("*.jpg"))
                png_files = list(path.rglob("*.png"))
                files = jpg_files + png_files

                if name in ["Train data", "Val data", "Test data"]:
                    total_images += len(files)
                    print(f"✓ {name}: {len(files):,} images")
                elif len(files) > 0:
                    print(f"✓ {name}: {len(files):,} files")
                else:
                    print(f"✓ {name}: exists")
            else:
                print(f"✓ {name}: exists")
        else:
            print(f"✗ {name}: NOT FOUND")
            all_ok = False

    print()
    print("-" * 70)
    print(f"Total images found: {total_images:,}")
    print(f"Expected: ~625,537 images")

    if total_images > 0:
        coverage = (total_images / 625537) * 100
        print(f"Coverage: {coverage:.1f}%")

    print()

    if all_ok and total_images > 600000:
        print("✓ CelebA-Spoof dataset is COMPLETE!")
        print()
        return True
    elif all_ok and total_images > 0:
        print("⚠ CelebA-Spoof dataset is PARTIAL")
        print("  Some images may be missing. Check download completeness.")
        print()
        return False
    else:
        print("✗ CelebA-Spoof dataset is INCOMPLETE or NOT DOWNLOADED")
        print()
        print("To download:")
        print("1. Run: python data/download_celeba_spoof.py")
        print("2. Follow manual download instructions")
        print(f"3. Extract files to: {root}")
        print()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download CelebA-Spoof dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data/download_celeba_spoof.py              # Show download instructions
  python data/download_celeba_spoof.py --verify     # Verify existing download
  python data/download_celeba_spoof.py --root /path/to/data  # Custom directory
        """
    )
    parser.add_argument('--verify', action='store_true',
                        help='Verify existing download')
    parser.add_argument('--root', default='/home/aaron/projects/xai/data/celeba-spoof',
                        help='Root directory for dataset (default: %(default)s)')

    args = parser.parse_args()

    if args.verify:
        success = verify_celeba_spoof(args.root)
        exit(0 if success else 1)
    else:
        download_celeba_spoof(args.root)
        exit(0)

if __name__ == "__main__":
    main()
