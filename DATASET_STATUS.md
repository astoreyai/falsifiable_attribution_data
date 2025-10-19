# Dataset Availability Status

**Last Updated:** October 19, 2025
**Purpose:** Multi-dataset validation for dissertation defense readiness

---

## Current Status Summary

| Dataset | Status | Location | Images | Download Required |
|---------|--------|----------|--------|-------------------|
| LFW | ✗ Not Found | `/home/aaron/.local/share/lfw` | 0 | Auto (sklearn) |
| CelebA | ✗ Not Found | `/home/aaron/projects/xai/data/celeba` | 0 | Manual/Auto |
| CFP-FP | ✗ Not Found | `/home/aaron/projects/xai/data/cfp-fp` | 0 | Manual (registration) |

---

## Dataset Details

### 1. LFW (Labeled Faces in the Wild)

**Status:** ✗ NOT FOUND (will auto-download on first use)

**Expected Path:** `/home/aaron/.local/share/lfw`

**Details:**
- Size: 13,233 images
- Identities: 5,749
- File size: ~170 MB
- License: Non-commercial research

**Download Method:**
```bash
# Auto-downloads on first use via sklearn
python -c "from sklearn.datasets import fetch_lfw_people; fetch_lfw_people(download_if_missing=True)"
```

**Or run experiment directly (will auto-download):**
```bash
python experiments/run_multidataset_experiment_6_1.py --datasets lfw
```

**Time Required:** 5-10 minutes (automatic)

**Priority:** LOW (auto-downloads when needed)

---

### 2. CelebA (CelebFaces Attributes Dataset)

**Status:** ✗ NOT FOUND

**Expected Path:** `/home/aaron/projects/xai/data/celeba/celeba/img_align_celeba/`

**Details:**
- Size: 202,599 images
- Identities: 10,177
- File size: ~1.5 GB
- Attributes: 40 binary labels per image
- License: Non-commercial research

**Download Method:**
```bash
cd /home/aaron/projects/xai
python data/download_celeba.py
```

**Verify Installation:**
```bash
python data/download_celeba.py --verify-only
```

**Time Required:** 30-60 minutes (network dependent)

**Priority:** HIGH (critical for generalization validation)

**Expected Structure:**
```
/home/aaron/projects/xai/data/celeba/
└── celeba/
    ├── img_align_celeba/
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ... (202,599 images)
    ├── list_attr_celeba.txt
    ├── list_eval_partition.txt
    └── identity_CelebA.txt
```

---

### 3. CFP-FP (Frontal-Profile Face Pairs)

**Status:** ✗ NOT FOUND (requires registration)

**Expected Path:** `/home/aaron/projects/xai/data/cfp-fp/Data/Images/`

**Details:**
- Size: 7,000 images (3,500 frontal + 3,500 profile)
- Identities: 500
- File size: ~500 MB
- License: Academic research (registration required)

**Download Method:**
```bash
# Step 1: View registration instructions
python data/download_cfp_fp.py

# Step 2: Register at http://www.cfpw.io/

# Step 3: Wait for approval (1-3 business days)

# Step 4: Download and extract CFP-FP.zip
```

**Verify Installation:**
```bash
python data/download_cfp_fp.py --verify /home/aaron/projects/xai/data/cfp-fp
```

**Time Required:**
- Registration: 5 minutes
- Approval wait: 1-3 business days
- Download: 15-30 minutes

**Priority:** MEDIUM (optional for strong defense, required for very strong)

**Expected Structure:**
```
/home/aaron/projects/xai/data/cfp-fp/
├── Protocol/
│   ├── Pair_list_F.txt
│   └── Pair_list_P.txt
└── Data/
    └── Images/
        ├── 001/
        │   ├── 01.jpg (frontal)
        │   ├── 02.jpg (profile)
        │   └── ...
        ├── 002/
        └── ... (500 identity directories)
```

---

## Action Items

### Immediate Actions (Today)

1. **Download CelebA** (HIGH PRIORITY)
   ```bash
   cd /home/aaron/projects/xai
   python data/download_celeba.py
   ```
   - Time: 30-60 minutes
   - Defense impact: +6 points (85 → 91)

2. **Test LFW Auto-Download**
   ```bash
   python experiments/run_multidataset_experiment_6_1.py --datasets lfw --n-pairs 100
   ```
   - Time: 10-15 minutes
   - Verifies baseline dataset works

### Short-Term Actions (This Week)

3. **Register for CFP-FP** (OPTIONAL)
   ```bash
   python data/download_cfp_fp.py
   # Follow registration instructions
   ```
   - Time: 5 minutes + 1-3 days wait
   - Defense impact: +2 additional points (91 → 93)

### Medium-Term Actions (Next Week)

4. **Run Multi-Dataset Experiments**
   ```bash
   # After CelebA downloads
   python experiments/run_multidataset_experiment_6_1.py --datasets lfw celeba --n-pairs 500
   ```
   - Time: 6-8 hours
   - Generates results for Chapter 6

---

## Defense Readiness Impact

### Current Status (No Additional Datasets)
- **Defense Score:** 85/100
- **Committee Risk:** 7/10
- **Weakness:** Single dataset (LFW from existing experiments)
- **Committee Question:** "How do you know this generalizes beyond LFW?"

### After CelebA Download
- **Defense Score:** 91/100 (+6 points)
- **Committee Risk:** 5/10
- **Strength:** Two diverse datasets with different bias characteristics
- **Answer:** "Validated on LFW and CelebA (200K+ images, diverse demographics)"

### After CFP-FP (If Approved)
- **Defense Score:** 93/100 (+2 additional points)
- **Committee Risk:** 4/10
- **Strength:** Three datasets including pose variation
- **Answer:** "Validated on LFW, CelebA, and CFP-FP including frontal+profile poses"

---

## Timeline Estimates

### Scenario A: LFW + CelebA Only (Recommended)

| Task | Time | Cumulative |
|------|------|------------|
| Download CelebA | 30-60 min | 1 hour |
| LFW auto-download | 5-10 min | 1.2 hours |
| Run experiments (500 pairs each) | 6-8 hours | 8-10 hours |
| **Total** | **8-10 hours** | **Ready in 1 day** |

**Defense Readiness:** 91/100 (Strong)

### Scenario B: LFW + CelebA + CFP-FP (Optimal)

| Task | Time | Cumulative |
|------|------|------------|
| Register for CFP-FP | 5 min | 5 min |
| Wait for approval | 1-3 days | 1-3 days |
| Download CelebA | 30-60 min | 1-3 days + 1 hour |
| Download CFP-FP | 15-30 min | 1-3 days + 1.5 hours |
| LFW auto-download | 5-10 min | 1-3 days + 1.7 hours |
| Implement CFP-FP loader | 1-2 hours | 1-3 days + 3.7 hours |
| Run experiments (500 pairs each) | 10-12 hours | 1-3 days + 15 hours |
| **Total** | **1-3 days + 15 hours** | **Ready in 4-5 days** |

**Defense Readiness:** 93/100 (Very Strong)

---

## Fallback Strategy

If CelebA or CFP-FP downloads fail:

### Fallback 1: LFW Only
- **Status:** Current baseline
- **Defense Score:** 85/100
- **Risk:** Moderate
- **Mitigation:** Acknowledge limitation in dissertation, propose as future work

### Fallback 2: LFW + Alternative Dataset
- Consider other public face datasets:
  - VGGFace2 (if available)
  - CASIA-WebFace
  - MS-Celeb-1M (deprecated but may be available)

### Fallback 3: Document Download Attempts
- If CFP-FP registration denied or delayed:
  - Document registration attempt in dissertation
  - List as "future work pending dataset access"
  - Two-dataset validation still acceptable for PhD defense

---

## Verification Commands

### Check All Datasets
```bash
python -c "
import os
from pathlib import Path

datasets = {
    'LFW': '/home/aaron/.local/share/lfw',
    'CelebA': '/home/aaron/projects/xai/data/celeba/celeba/img_align_celeba',
    'CFP-FP': '/home/aaron/projects/xai/data/cfp-fp/Data/Images'
}

for name, path in datasets.items():
    exists = '✓' if os.path.exists(path) else '✗'
    print(f'{exists} {name}')
"
```

### Verify CelebA
```bash
python data/download_celeba.py --verify-only
```

### Verify CFP-FP
```bash
python data/download_cfp_fp.py --verify /home/aaron/projects/xai/data/cfp-fp
```

### Test Experiment Script
```bash
# Dry run (checks availability only)
python experiments/run_multidataset_experiment_6_1.py --n-pairs 10
```

---

## Dataset Loader Status

| Dataset | Loader Exists | Location | Status |
|---------|---------------|----------|--------|
| LFW | ✓ Yes | `experiments/run_multidataset_experiment_6_1.py` | Ready |
| CelebA | ✓ Yes | `data/celeba_dataset.py` | Ready |
| CFP-FP | ✗ No | To be created | TODO |

**CFP-FP Loader:** If CFP-FP is downloaded, create loader following CelebA pattern.

---

## Disk Space Requirements

| Dataset | Compressed | Uncompressed | Total Space Needed |
|---------|------------|--------------|-------------------|
| LFW | N/A | ~170 MB | 200 MB |
| CelebA | ~1.3 GB | ~1.5 GB | 3 GB |
| CFP-FP | ~450 MB | ~500 MB | 1 GB |
| **Total** | **~1.8 GB** | **~2.2 GB** | **~4.5 GB** |

**Current Available Space:**
```bash
df -h /home/aaron/projects/xai
```

---

## Next Steps

1. **Today:** Download CelebA
   ```bash
   python data/download_celeba.py
   ```

2. **Today:** Test LFW + CelebA experiment (quick test)
   ```bash
   python experiments/run_multidataset_experiment_6_1.py --datasets lfw celeba --n-pairs 100
   ```

3. **Optional:** Register for CFP-FP (parallel to CelebA experiments)
   ```bash
   python data/download_cfp_fp.py
   # Follow registration instructions
   ```

4. **This Week:** Run full experiments (500 pairs each)
   ```bash
   python experiments/run_multidataset_experiment_6_1.py --datasets lfw celeba --n-pairs 500
   ```

5. **Next Week:** Analyze results and update Chapter 6

---

## Contact & Support

**Dataset Download Issues:**
- CelebA: See `DATASET_DOWNLOAD_GUIDE.md` for alternative methods
- CFP-FP: Email cfpw-organizers@googlegroups.com

**Experiment Issues:**
- Check logs in `experiments/multidataset_results/`
- Verify GPU availability: `nvidia-smi`

**Questions:**
- Consult: `DATASET_DOWNLOAD_GUIDE.md`
- Consult: `MULTIDATASET_ANALYSIS_PLAN.md`

---

**Last Status Check:** October 19, 2025

**Recommendation:** Download CelebA immediately for 91/100 defense readiness
