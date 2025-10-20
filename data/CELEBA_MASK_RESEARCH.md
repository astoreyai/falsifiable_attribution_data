# CelebA-Mask Dataset Research

**Date:** October 19, 2025
**Agent:** 4 of 4 (Dataset Download - CelebAMask-HQ)
**Purpose:** Semantic segmentation masks for regional attribution validation

---

## Dataset Versions

### CelebAMask-HQ (High Quality) ⭐ RECOMMENDED

- **Paper:** "MaskGAN: Towards Diverse and Interactive Facial Image Manipulation" (CVPR 2020)
- **Authors:** Cheng-Han Lee, Ziwei Liu, Lingyun Wu, Ping Luo (CUHK)
- **Images:** 30,000 high-resolution images (1024×1024)
- **Semantic classes:** 19 classes
- **Classes:**
  - **Facial components:** skin, nose, l_eye, r_eye, l_brow, r_brow, l_ear, r_ear, mouth, u_lip (upper lip), l_lip (lower lip)
  - **Accessories:** eye_g (eyeglasses), ear_r (earring), neck_l (necklace), hat
  - **Other:** hair, neck, cloth, background
- **Source:** Subset of CelebA-HQ
- **Mask resolution:** 512×512 (manually annotated)
- **File size:** ~2.57 GB (Supervisely format), full dataset larger

### CelebA-Mask (Original)

- **Images:** 202,599 (matches full CelebA dataset)
- **Quality:** Lower resolution (178×218)
- **Masks:** Coarser segmentation (binary masks for 40 attributes)
- **File size:** Larger (~3-5 GB)

---

## Download Sources

### 1. Official Repository (GitHub)
- **URL:** https://github.com/switchablenorms/CelebAMask-HQ
- **Status:** ⚠️ Google Drive link reported broken (Issue #74)
- **Alternative:** Baidu Drive (requires Chinese account)
- **License:** Non-commercial research purposes only

### 2. Official CUHK Website
- **URL:** https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html
- **Status:** May redirect to Google Drive/Baidu Drive
- **Note:** Part of CelebA project family

### 3. Kaggle (Community Upload) ✅ ACCESSIBLE
- **URL:** https://www.kaggle.com/datasets/ipythonx/celebamaskhq
- **Status:** ✅ Available for direct download
- **Size:** Full dataset available
- **Method:** `kaggle datasets download -d ipythonx/celebamaskhq`
- **Requirement:** Kaggle API credentials

### 4. Hugging Face
- **URL:** https://huggingface.co/datasets/liusq/CelebAMask-HQ
- **Status:** Available
- **Method:** `datasets.load_dataset("liusq/CelebAMask-HQ")`
- **Note:** May have viewer limitations

### 5. Dataset Ninja
- **URL:** https://datasetninja.com/celebamask-hq
- **Format:** Supervisely format (2.57 GB)
- **Method:** `dataset-tools` package
- **Status:** Alternative download

### 6. TensorFlow Datasets
- **URL:** https://www.tensorflow.org/datasets/catalog/celeb_a_hq
- **Note:** CelebA-HQ images only (no segmentation masks)
- **Status:** Images only, masks separate

---

## Recommended Version

**CelebAMask-HQ** is preferred for the dissertation:

✅ **Advantages:**
- Higher quality masks (manually annotated at 512×512)
- 19 semantic classes (detailed facial parsing)
- Better for attribution validation (precise regional boundaries)
- Smaller dataset size (30K vs 202K images - faster experiments)
- High-resolution images (1024×1024)

❌ **Disadvantages:**
- Download complexity (broken Google Drive links)
- Requires Kaggle API or alternative source
- Non-commercial license (acceptable for PhD research)

---

## Official Links

- **GitHub Repository:** https://github.com/switchablenorms/CelebAMask-HQ
- **CUHK Project Page:** https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html
- **Papers with Code:** https://paperswithcode.com/dataset/celebamask-hq
- **Kaggle Dataset:** https://www.kaggle.com/datasets/ipythonx/celebamaskhq
- **Hugging Face:** https://huggingface.co/datasets/liusq/CelebAMask-HQ
- **Dataset Ninja:** https://datasetninja.com/celebamask-hq

---

## Citation

```bibtex
@inproceedings{CelebAMask-HQ,
  title={MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
  author={Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

---

## Dataset Structure (Expected)

```
CelebAMask-HQ/
├── CelebA-HQ-img/                    # 30,000 images (1024×1024)
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ... (29999.jpg)
│
├── CelebAMask-HQ-mask-anno/          # Segmentation masks (512×512)
│   ├── 0/                            # Images 0-1999
│   │   ├── 00000_skin.png
│   │   ├── 00000_l_brow.png
│   │   ├── ... (19 classes per image)
│   │   └── 01999_background.png
│   ├── 1/                            # Images 2000-3999
│   ├── 2/                            # Images 4000-5999
│   └── ... (15 folders total: 0-14)
│
├── CelebA-HQ-to-CelebA-mapping.txt   # Mapping to original CelebA IDs
├── README.md
└── LICENSE
```

---

## 19 Semantic Classes

| Class ID | Class Name | Description |
|----------|-----------|-------------|
| 0 | skin | Facial skin |
| 1 | l_brow | Left eyebrow |
| 2 | r_brow | Right eyebrow |
| 3 | l_eye | Left eye |
| 4 | r_eye | Right eye |
| 5 | eye_g | Eyeglasses |
| 6 | l_ear | Left ear |
| 7 | r_ear | Right ear |
| 8 | ear_r | Earring |
| 9 | nose | Nose |
| 10 | mouth | Mouth |
| 11 | u_lip | Upper lip |
| 12 | l_lip | Lower lip |
| 13 | neck | Neck |
| 14 | neck_l | Necklace |
| 15 | cloth | Clothing |
| 16 | hair | Hair |
| 17 | hat | Hat |
| 18 | background | Background |

---

## License

**CelebAMask-HQ License:** Non-commercial research purposes only

**Terms:**
- ✅ Use for academic research (PhD dissertation)
- ✅ Internal use within organization
- ❌ Commercial exploitation prohibited
- ❌ Redistribution restricted

**Compliance for Dissertation:** ✅ Acceptable (non-commercial academic research)

---

## Download Recommendation

**Primary Method:** Kaggle API (most reliable)
**Backup Method:** Hugging Face Datasets
**Manual Method:** Dataset Ninja (Supervisely format)

**Decision:** Attempt Kaggle download first. If credentials unavailable, document dataset for future work rather than blocking dissertation progress.

---

## Status Summary

- ✅ Dataset researched and documented
- ✅ Multiple download sources identified
- ⏳ Download attempt pending (Kaggle API)
- ⏳ Dataset structure verification pending
- ⏳ Loader implementation pending
