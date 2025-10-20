# CelebA Dataset Download Options

**Date:** October 19, 2025
**Agent:** Agent 2 (CelebA Main Dataset Download)
**Dataset:** CelebFaces Attributes (CelebA)

---

## Official Dataset Information

**Source:** http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
**Paper:** Liu et al. "Deep Learning Face Attributes in the Wild" ICCV 2015
**License:** Non-commercial research purposes only

### Dataset Components

- **Images:** 202,599 aligned face images (178x218 pixels)
- **Identities:** 10,177 unique celebrity identities
- **Attributes:** 40 binary attributes per image
- **Landmarks:** 5 facial landmarks (10 coordinates)
- **Bounding Boxes:** Face detection boxes
- **Partitions:** Train/validation/test split

**Total Size:** ~1.5 GB (images) + ~200 MB (annotations) = ~1.7 GB

---

## Download Methods

### Method 1: PyTorch Torchvision (RECOMMENDED)

**Advantages:**
- Fully automated
- Handles all file downloads and extraction
- Includes all metadata (attributes, landmarks, identities, partitions)
- No manual account setup required
- Reliable and well-tested

**Requirements:**
- Python 3.7+
- PyTorch and torchvision installed

**Command:**
```bash
python data/download_celeba.py --method torchvision
```

**What it downloads:**
- img_align_celeba.zip (1.4 GB) → extracted to img_align_celeba/
- list_attr_celeba.txt (25 MB)
- list_landmarks_align_celeba.txt (15 MB)
- identity_CelebA.txt (3 MB)
- list_bbox_celeba.txt (10 MB)
- list_eval_partition.txt (5 MB)

**Expected Time:** 30-60 minutes (depending on network speed)

---

### Method 2: Kaggle API

**Advantages:**
- Alternative source if torchvision fails
- Fast download speeds
- Single archive download

**Disadvantages:**
- Requires Kaggle account
- Requires API key setup

**Requirements:**
- Kaggle account (free)
- Kaggle API key configured

**Setup:**
1. Create account at https://www.kaggle.com
2. Go to https://www.kaggle.com/settings/account
3. Click "Create New API Token" (downloads kaggle.json)
4. Move kaggle.json to ~/.kaggle/kaggle.json
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`
6. Install Kaggle API: `pip install kaggle`

**Command:**
```bash
python data/download_celeba.py --method kaggle
```

**Dataset URL:**
https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

---

### Method 3: Manual Download

**Use when:**
- Automated methods fail
- Need specific dataset version
- Download from official source directly

**Sources:**

#### A. Official Source (Google Drive)
- Visit: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- Look for Google Drive links
- Download individual files:
  - img_align_celeba.zip
  - list_attr_celeba.txt
  - list_landmarks_align_celeba.txt
  - identity_CelebA.txt
  - list_bbox_celeba.txt
  - list_eval_partition.txt

#### B. Baidu Drive (China-based)
- Alternative source on CelebA website
- May require Baidu account
- Same files as Google Drive

**Manual Setup:**
```bash
# Extract images
unzip img_align_celeba.zip -d /home/aaron/projects/xai/data/celeba/celeba/

# Move annotation files
mv list_*.txt /home/aaron/projects/xai/data/celeba/celeba/
mv identity_CelebA.txt /home/aaron/projects/xai/data/celeba/celeba/
```

**Verify structure:**
```bash
python data/download_celeba.py --verify
```

---

## Expected Directory Structure

After successful download:

```
/home/aaron/projects/xai/data/celeba/
└── celeba/
    ├── img_align_celeba/
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   ├── 000003.jpg
    │   └── ... (202,599 total)
    ├── list_attr_celeba.txt              # 40 binary attributes
    ├── list_landmarks_align_celeba.txt   # 5 facial landmarks
    ├── identity_CelebA.txt               # Celebrity IDs
    ├── list_bbox_celeba.txt              # Face bounding boxes
    └── list_eval_partition.txt           # Train/val/test split
```

---

## Verification Commands

### Verify Download Completeness
```bash
python data/download_celeba.py --verify
```

Expected output:
```
✓ Images              : 202,599 images
✓ Attributes          : 25.0 MB
✓ Landmarks           : 15.0 MB
✓ Identities          : 3.0 MB
✓ Bounding Boxes      : 10.0 MB
✓ Partitions          : 5.0 MB

Total disk usage: 1.70 GB

✓ Dataset verification PASSED - All components present!
```

### Analyze Dataset Statistics
```bash
python data/download_celeba.py --analyze
```

Shows:
- Image count
- Attribute distribution
- Identity distribution
- Train/val/test split sizes

---

## Troubleshooting

### Torchvision Download Fails

**Error:** Connection timeout or download interrupted

**Solutions:**
1. Check internet connection
2. Try Kaggle method: `python data/download_celeba.py --method kaggle`
3. Use manual download
4. Check firewall/proxy settings

### Kaggle API Authentication Error

**Error:** "401 Unauthorized" or "Invalid API credentials"

**Solutions:**
1. Verify kaggle.json exists: `ls -la ~/.kaggle/`
2. Check permissions: `chmod 600 ~/.kaggle/kaggle.json`
3. Re-download API token from Kaggle website
4. Verify file format (should be valid JSON)

### Missing Files After Download

**Error:** Verification shows missing components

**Solutions:**
1. Re-run download with same method
2. Try alternative download method
3. Manually download missing files from official source
4. Check disk space: `df -h`

### Disk Space Issues

**Error:** "No space left on device"

**Requirements:** 2 GB free space (1.7 GB dataset + extraction buffer)

**Solutions:**
1. Check available space: `df -h /home/aaron/projects/xai/data/`
2. Clean up temporary files
3. Use different download location (modify --root flag)

---

## Network Requirements

- **Bandwidth:** Broadband connection recommended
- **Data Usage:** ~1.7 GB download
- **Time Estimate:**
  - Fast connection (100 Mbps): 15-20 minutes
  - Medium connection (50 Mbps): 30-40 minutes
  - Slow connection (10 Mbps): 1-2 hours

---

## Citation

If using CelebA dataset in research:

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

## Summary Table

| Method | Speed | Ease | Requirements | Recommended |
|--------|-------|------|--------------|-------------|
| Torchvision | Fast | Easy | Python, PyTorch | ✓ YES |
| Kaggle API | Fast | Medium | Kaggle account + API | Alternative |
| Manual | Slow | Hard | Web browser | Last resort |

**Recommendation:** Use torchvision method unless it fails, then fallback to Kaggle.

---

**Next Steps After Download:**

1. Verify dataset: `python data/download_celeba.py --verify`
2. Analyze statistics: `python data/download_celeba.py --analyze`
3. Test loader: See CELEBA_INTEGRATION.md
4. Run experiments: See experiment scripts
