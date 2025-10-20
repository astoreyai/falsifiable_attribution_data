# CelebA Download Status Report

**Date:** October 19, 2025
**Agent:** CelebA Main Dataset Agent
**Status:** Scripts ready, awaiting installation of dependencies

---

## Summary

Enhanced download scripts have been created with 3 download methods. However, the required Python packages are not currently installed. Below are the options to proceed.

---

## Available Download Methods

### Method 1: Torchvision (Recommended - Easiest)

**Status:** Requires installation
**Command:** `python3 data/download_celeba.py`

**Requirements:**
```bash
pip3 install torch torchvision pillow
```

**Pros:**
- Fully automated download
- Handles all annotations
- Verifies completeness
- Most reliable

**Cons:**
- Requires ~2GB additional disk space for PyTorch
- Large package installation

**Installation:**
```bash
# Install dependencies (one-time)
pip3 install torch torchvision pillow

# Then download CelebA
python3 data/download_celeba.py
```

### Method 2: Kaggle API

**Status:** API key configured, package not installed
**Command:** `python3 data/download_celeba.py --method kaggle`

**Requirements:**
```bash
pip3 install kaggle
```

**Pros:**
- Smaller installation (~10 MB)
- Fast download from Kaggle mirrors
- API already configured at `~/.kaggle/kaggle.json`

**Cons:**
- Requires Kaggle account (already have)
- May need to manually accept dataset terms on Kaggle website

**Installation:**
```bash
# Install Kaggle package (one-time)
pip3 install kaggle

# Verify API access
kaggle datasets list --search celeba

# Download CelebA
python3 data/download_celeba.py --method kaggle
```

### Method 3: Manual Download

**Status:** Always available
**Command:** `python3 data/download_celeba.py --method manual`

**Requirements:** None (manual browser download)

**Pros:**
- No additional packages needed
- Direct from official source
- Full control over download

**Cons:**
- Manual steps required
- Slower (browser download)
- Need to organize files manually

**Steps:**
1. Visit: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
2. Download 6 files (~1.5 GB total)
3. Extract and organize into correct structure
4. Verify with: `python3 data/download_celeba.py --verify`

---

## Current System Status

### Disk Space
```
Available: 753 GB
Required: ~2 GB (1.5 GB dataset + 0.5 GB temporary)
Status: ✓ Sufficient
```

### Python Environment
```
Python: 3.x (available)
NumPy: 1.24.2 (installed)
PyTorch: Not installed
Torchvision: Not installed
Kaggle: Not installed
```

### Kaggle API
```
Configuration: ✓ Present (~/.kaggle/kaggle.json)
Package: ✗ Not installed
Status: Ready to install
```

---

## Recommended Installation Path

**Option A: Full Installation (Recommended for long-term use)**

If you plan to run deep learning experiments:

```bash
# Install full requirements
pip3 install -r requirements.txt

# This includes torch, torchvision, and all experiment dependencies
# Then download CelebA
python3 data/download_celeba.py

# Estimated time: 10-15 minutes install + 30-60 minutes download
```

**Option B: Lightweight Installation (Quick start)**

If you just want to download CelebA now:

```bash
# Install only Kaggle
pip3 install kaggle

# Download CelebA via Kaggle
python3 data/download_celeba.py --method kaggle

# Estimated time: 1 minute install + 20-30 minutes download
```

**Option C: Manual (No installation)**

If you prefer manual control:

```bash
# Show instructions
python3 data/download_celeba.py --method manual

# Follow printed steps
# Estimated time: 60-90 minutes (browser download + manual setup)
```

---

## Created Scripts & Documentation

### 1. Enhanced Download Script
**File:** `/home/aaron/projects/xai/data/download_celeba.py`

**Features:**
- 3 download methods (torchvision, kaggle, manual)
- Automatic verification
- Dataset statistics analysis
- Progress reporting
- Error handling with fallbacks

**Usage:**
```bash
# Download (default: torchvision)
python3 data/download_celeba.py

# Download via Kaggle
python3 data/download_celeba.py --method kaggle

# Show manual instructions
python3 data/download_celeba.py --method manual

# Verify existing download
python3 data/download_celeba.py --verify

# Analyze dataset statistics
python3 data/download_celeba.py --analyze
```

### 2. Integration Guide
**File:** `/home/aaron/projects/xai/data/CELEBA_INTEGRATION.md`

**Contents:**
- Dataset overview (202,599 images, 10,177 identities, 40 attributes)
- Component descriptions (images, attributes, landmarks, identities)
- Integration with experiments (face verification, fairness analysis)
- Comparison to LFW and CFP-FP
- Usage examples
- Research questions
- Citation information

### 3. This Status Report
**File:** `/home/aaron/projects/xai/data/CELEBA_DOWNLOAD_STATUS.md`

---

## Dataset Details

Once downloaded, CelebA will provide:

### Statistics
- **Images:** 202,599 aligned face images (178x218 pixels)
- **Identities:** 10,177 unique celebrities
- **Attributes:** 40 binary labels per image
- **Landmarks:** 5 facial keypoints per image
- **Size:** ~1.5 GB (images) + ~200 MB (annotations)

### Components
1. **img_align_celeba/** - 202,599 JPG images
2. **list_attr_celeba.txt** - 40 binary attributes (gender, age, hair, etc.)
3. **list_landmarks_align_celeba.txt** - 5 (x,y) facial landmarks
4. **identity_CelebA.txt** - Celebrity identity labels
5. **list_bbox_celeba.txt** - Face bounding boxes
6. **list_eval_partition.txt** - Train/val/test split

### Partitions
- Train: 162,770 images (80.3%)
- Validation: 19,867 images (9.8%)
- Test: 19,962 images (9.9%)

---

## Integration with Experiments

### Experiment 6.1: Face Verification
Use CelebA identity labels to create same-person vs. different-person pairs:
- Enables cross-dataset validation with LFW and CFP-FP
- Much larger test set (19,962 images vs. LFW's 13,233)
- More identities (10,177 vs. LFW's 5,749)

### Experiment 6.6: Demographic Fairness
Use CelebA attributes for fairness analysis:
- Gender: Male/Female
- Age: Young/Old
- Other: Pale_Skin, etc.
- Analyze performance across demographic subgroups

### Attribution Validation
Use CelebA landmarks for region-based analysis:
- Define eye, nose, mouth regions using landmarks
- Compare attribution heatmaps with facial regions
- Validate that attributions focus on relevant features

---

## Comparison to Other Datasets

| Feature | LFW | CFP-FP | CelebA |
|---------|-----|--------|--------|
| Images | 13,233 | 7,000 | **202,599** |
| Identities | 5,749 | 500 | **10,177** |
| Attributes | None | None | **40 labels** |
| Landmarks | No | No | **5 points** |
| Size | 173 MB | ~500 MB | **1.5 GB** |

**Key Advantages:**
1. **15x more images** than LFW
2. **Only dataset with rich attribute labels**
3. **Landmarks for region-based analysis**
4. **Better demographic balance** than LFW

---

## Next Steps

### Immediate (Choose one installation method)

**Quick start (Kaggle - Recommended):**
```bash
pip3 install kaggle
python3 data/download_celeba.py --method kaggle
```

**Full installation (PyTorch):**
```bash
pip3 install -r requirements.txt
python3 data/download_celeba.py
```

**Manual (No installation):**
```bash
python3 data/download_celeba.py --method manual
# Follow printed instructions
```

### After Download

1. **Verify completeness:**
   ```bash
   python3 data/download_celeba.py --verify
   ```

2. **Analyze statistics:**
   ```bash
   python3 data/download_celeba.py --analyze
   ```

3. **Test dataset loader:**
   ```bash
   python3 -c "from data.celeba_dataset import CelebADataset; \
               d = CelebADataset('/home/aaron/projects/xai/data/celeba'); \
               print(f'Loaded {len(d)} images')"
   ```

4. **Run multi-dataset experiments:**
   ```bash
   python3 experiments/run_multidataset_experiment_6_1.py \
       --datasets lfw cfp_fp celeba \
       --n-pairs 100
   ```

5. **Update Chapter 8 Section 8.2.4:**
   - Report cross-dataset findings
   - Compare LFW vs. CFP-FP vs. CelebA
   - Discuss demographic fairness results
   - Validate attribution consistency

---

## Troubleshooting

### Installation Issues

**If pip3 is not available:**
```bash
# Install pip
sudo apt-get update
sudo apt-get install python3-pip
```

**If disk space insufficient:**
```bash
# Check space
df -h /home/aaron/projects/xai/

# Clean up if needed
# Then retry download
```

**If Kaggle API fails:**
```bash
# Verify API key
cat ~/.kaggle/kaggle.json

# Check permissions
chmod 600 ~/.kaggle/kaggle.json

# Test API
kaggle datasets list --search celeba
```

### Download Issues

**If download is slow:**
- Be patient (1.5 GB takes time)
- Use background download: `nohup python3 data/download_celeba.py &`
- Check with: `tail -f nohup.out`

**If download fails:**
- Try alternate method (kaggle if torchvision fails)
- Use manual download as last resort
- Check network connection

**If verification fails:**
- Re-download: `python3 data/download_celeba.py`
- Check disk space
- Review error messages

---

## Deliverables Completed

- [x] Enhanced download_celeba.py script (3 methods)
- [x] CELEBA_INTEGRATION.md (comprehensive guide)
- [x] CELEBA_DOWNLOAD_STATUS.md (this report)
- [x] Verified disk space (753 GB available)
- [x] Verified Kaggle API configuration
- [x] Identified missing dependencies
- [x] Provided installation instructions

---

## Estimated Time to Download

Once dependencies are installed:

- **Torchvision method:** 30-60 minutes
- **Kaggle method:** 20-30 minutes
- **Manual method:** 60-90 minutes

Plus installation time:
- **Kaggle only:** ~1 minute
- **PyTorch + torchvision:** ~10-15 minutes
- **Full requirements.txt:** ~10-15 minutes

---

## Recommendations

1. **For immediate download:** Install Kaggle API (quickest)
   ```bash
   pip3 install kaggle
   python3 data/download_celeba.py --method kaggle
   ```

2. **For long-term use:** Install full requirements
   ```bash
   pip3 install -r requirements.txt
   python3 data/download_celeba.py
   ```

3. **For minimal setup:** Manual download
   ```bash
   python3 data/download_celeba.py --method manual
   # Follow instructions
   ```

Choose based on your immediate needs and planned usage.

---

**Status:** Ready to proceed with installation and download.
**Blocker:** Need to install either `kaggle` or `torch/torchvision` packages.
**Recommendation:** Install Kaggle package for quickest download (1 min install + 20-30 min download).
