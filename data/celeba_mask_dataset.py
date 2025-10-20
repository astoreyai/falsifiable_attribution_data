"""
CelebAMask-HQ dataset loader with semantic segmentation masks.

Provides pixel-level face parsing for regional attribution analysis.
Dataset: 30,000 high-resolution images (1024x1024) with 19 semantic classes.

Citation:
    @article{CelebAMask-HQ,
        title={MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
        author={Cheng-Han Lee and Ziwei Liu and Lingyun Wu and Ping Luo},
        journal={Technical Report},
        year={2019}
    }

License: Non-commercial research and educational purposes only.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import numpy as np


class CelebAMaskHQ(Dataset):
    """
    CelebAMask-HQ dataset with 19 semantic segmentation classes.

    Semantic classes (0-18):
        0: skin, 1: l_brow, 2: r_brow, 3: l_eye, 4: r_eye, 5: eye_g (glasses),
        6: l_ear, 7: r_ear, 8: ear_r (earring), 9: nose, 10: mouth, 11: u_lip,
        12: l_lip, 13: neck, 14: neck_l (necklace), 15: cloth, 16: hair,
        17: hat, 18: background

    Args:
        root (str): Root directory containing CelebAMask-HQ dataset
        transform (callable, optional): Transform to apply to images
        return_mask (bool): Whether to load and return segmentation masks
        mask_size (int): Target size for masks (default: 512, original annotation size)
    """

    CLASSES = [
        'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g',
        'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth', 'u_lip',
        'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat', 'background'
    ]

    # Semantic groupings for analysis
    REGION_GROUPS = {
        'eyes': [1, 2, 3, 4, 5],  # brows, eyes, glasses
        'nose': [9],
        'mouth': [10, 11, 12],  # mouth, upper lip, lower lip
        'ears': [6, 7, 8],  # ears, earrings
        'face': [0, 1, 2, 3, 4, 9, 10, 11, 12],  # face components without accessories
        'accessories': [5, 8, 14, 17]  # glasses, earrings, necklace, hat
    }

    def __init__(self, root, transform=None, return_mask=True, mask_size=512):
        self.root = Path(root)
        self.transform = transform
        self.return_mask = return_mask
        self.mask_size = mask_size

        # Image directory
        self.img_dir = self.root / 'CelebAMask-HQ' / 'CelebA-HQ-img'

        # Mask directory (organized by ID folders)
        self.mask_dir = self.root / 'CelebAMask-HQ' / 'CelebAMask-HQ-mask-anno'

        # Check directories exist
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if return_mask and not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        # Get all image IDs (0-29999)
        self.image_ids = sorted([
            int(p.stem) for p in self.img_dir.glob('*.jpg')
        ])

        print(f"Loaded CelebAMask-HQ dataset with {len(self.image_ids)} images")

    def __len__(self):
        return len(self.image_ids)

    def load_mask(self, img_id):
        """
        Load combined semantic mask for image.

        Masks are stored as individual PNG files per class in folders by ID range:
        - Folder 0: images 0-1999
        - Folder 1: images 2000-3999
        - ... etc (15 folders total: 0-14)

        Args:
            img_id (int): Image ID (0-29999)

        Returns:
            np.ndarray: (H, W) array with class IDs 0-18, or 255 for no class
        """
        # Determine folder (2000 images per folder)
        folder_id = img_id // 2000
        mask_folder = self.mask_dir / str(folder_id)

        # Initialize mask (255 = no class assigned)
        combined_mask = np.full((512, 512), 255, dtype=np.uint8)

        # Load individual class masks and combine
        # Note: Not all images have all classes (e.g., not everyone wears glasses)
        for class_id, class_name in enumerate(self.CLASSES):
            mask_file = mask_folder / f'{img_id:05d}_{class_name}.png'
            if mask_file.exists():
                class_mask = np.array(Image.open(mask_file).convert('L'))  # Convert to grayscale
                # Mask pixels are either 0 (background) or 255 (class present)
                combined_mask[class_mask > 0] = class_id

        # Resize if needed
        if self.mask_size != 512:
            combined_mask = np.array(
                Image.fromarray(combined_mask).resize(
                    (self.mask_size, self.mask_size),
                    resample=Image.NEAREST
                )
            )

        return combined_mask

    def __getitem__(self, idx):
        """
        Get image and optionally its segmentation mask.

        Returns:
            dict: {
                'image': PIL Image or transformed tensor,
                'image_id': int,
                'mask': torch.LongTensor (if return_mask=True)
            }
        """
        img_id = self.image_ids[idx]

        # Load image
        img_path = self.img_dir / f'{img_id}.jpg'
        image = Image.open(img_path).convert('RGB')

        result = {'image': image, 'image_id': img_id}

        # Load semantic mask if requested
        if self.return_mask:
            mask = self.load_mask(img_id)
            result['mask'] = torch.from_numpy(mask).long()

        # Apply transforms to image
        if self.transform:
            result['image'] = self.transform(result['image'])

        return result

    def get_class_name(self, class_id):
        """Get semantic class name from ID."""
        if 0 <= class_id < len(self.CLASSES):
            return self.CLASSES[class_id]
        return 'unknown'

    def get_region_mask(self, mask, region_name):
        """
        Extract binary mask for a semantic region.

        Args:
            mask (np.ndarray or torch.Tensor): Segmentation mask
            region_name (str): One of 'eyes', 'nose', 'mouth', 'ears', 'face', 'accessories'

        Returns:
            np.ndarray or torch.Tensor: Binary mask (same type as input)
        """
        if region_name not in self.REGION_GROUPS:
            raise ValueError(f"Unknown region: {region_name}. Choose from {list(self.REGION_GROUPS.keys())}")

        class_ids = self.REGION_GROUPS[region_name]

        if isinstance(mask, torch.Tensor):
            region_mask = torch.zeros_like(mask, dtype=torch.bool)
            for cid in class_ids:
                region_mask |= (mask == cid)
            return region_mask.float()
        else:
            region_mask = np.zeros_like(mask, dtype=bool)
            for cid in class_ids:
                region_mask |= (mask == cid)
            return region_mask.astype(np.float32)

    def visualize_mask(self, mask, save_path=None, show_legend=True):
        """
        Visualize semantic mask with colors.

        Args:
            mask (np.ndarray or torch.Tensor): (H, W) mask with class IDs
            save_path (str, optional): Path to save visualization
            show_legend (bool): Whether to show class legend

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        # Color map for 19 classes + unknown (255)
        colors = plt.cm.tab20(np.linspace(0, 1, 20))

        # Create colored mask
        colored_mask = np.zeros((*mask.shape, 3))
        for i in range(19):
            colored_mask[mask == i] = colors[i][:3]
        colored_mask[mask == 255] = [0.5, 0.5, 0.5]  # Gray for no class

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(colored_mask)
        ax.axis('off')

        # Create legend
        if show_legend:
            patches = [
                mpatches.Patch(color=colors[i], label=self.CLASSES[i])
                for i in range(19)
            ]
            ax.legend(
                handles=patches,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                fontsize=8,
                ncol=1
            )

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)

        return fig

    def compute_region_overlap(self, attribution_map, mask, region_name):
        """
        Compute overlap between attribution map and semantic region.

        This is the key metric for validating attribution methods:
        - High overlap = attributions localize to semantically relevant regions
        - Low overlap = attributions are scattered or incoherent

        Args:
            attribution_map (np.ndarray or torch.Tensor): (H, W) saliency map (0-1)
            mask (np.ndarray or torch.Tensor): (H, W) segmentation mask
            region_name (str): Region to test ('eyes', 'nose', 'mouth', etc.)

        Returns:
            float: Percentage of attribution mass in specified region (0-100)
        """
        # Get region mask
        region_mask = self.get_region_mask(mask, region_name)

        if isinstance(attribution_map, torch.Tensor):
            attribution_in_region = (attribution_map * region_mask).sum()
            total_attribution = attribution_map.sum()
        else:
            attribution_in_region = (attribution_map * region_mask).sum()
            total_attribution = attribution_map.sum()

        if total_attribution > 0:
            overlap = 100 * (attribution_in_region / total_attribution)
            return float(overlap)
        else:
            return 0.0


# Test loader
if __name__ == '__main__':
    import sys

    # Test basic loading
    try:
        dataset = CelebAMaskHQ(
            root='/home/aaron/projects/xai/data/celeba_mask',
            return_mask=True
        )
        print(f"\n✓ Dataset loaded: {len(dataset)} images")

        if len(dataset) > 0:
            # Test first sample
            sample = dataset[0]
            print(f"✓ Image shape: {sample['image'].size}")
            print(f"✓ Mask shape: {sample['mask'].shape}")
            print(f"✓ Image ID: {sample['image_id']}")

            # Check unique classes in mask
            unique_classes = torch.unique(sample['mask']).tolist()
            print(f"✓ Unique classes in first mask: {unique_classes}")
            print(f"  Class names: {[dataset.get_class_name(c) for c in unique_classes if c != 255]}")

            # Test region extraction
            mask_np = sample['mask'].numpy()
            eyes_mask = dataset.get_region_mask(mask_np, 'eyes')
            nose_mask = dataset.get_region_mask(mask_np, 'nose')
            print(f"✓ Eyes region pixels: {eyes_mask.sum():.0f}")
            print(f"✓ Nose region pixels: {nose_mask.sum():.0f}")

            # Test overlap computation
            fake_attribution = np.random.rand(*mask_np.shape)
            overlap = dataset.compute_region_overlap(fake_attribution, mask_np, 'eyes')
            print(f"✓ Random attribution overlap with eyes: {overlap:.2f}%")

            # Optional: Save visualization
            if '--save-viz' in sys.argv:
                fig = dataset.visualize_mask(mask_np, save_path='test_mask_visualization.png')
                print("✓ Saved visualization to test_mask_visualization.png")

            print("\n✓ All tests passed!")

    except FileNotFoundError as e:
        print(f"\n✗ Dataset not found: {e}")
        print("  Download CelebAMask-HQ first")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
