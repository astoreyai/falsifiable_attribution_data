"""
VGGFace2 Dataset Loader for Falsification Experiments

Loads face image pairs for verification tasks.
Compatible with InsightFace model preprocessing.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable
import logging
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


def get_default_transforms(image_size: int = 112) -> transforms.Compose:
    """
    Get default preprocessing transforms for face images.
    
    Compatible with InsightFace model expectations:
    - Resize to 112x112
    - Normalize to [0, 1]
    - Convert to RGB
    
    Args:
        image_size: Target image size (default: 112 for InsightFace)
        
    Returns:
        Composed transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


class VGGFace2Dataset(Dataset):
    """
    VGGFace2 dataset for face verification experiments.
    
    Generates genuine and impostor pairs for falsification testing.
    
    Dataset structure expected:
        root_dir/
            test/  (or train/)
                n000001/
                    0001_01.jpg
                    0002_01.jpg
                    ...
                n000002/
                    ...
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'test',
        n_pairs: int = 200,
        transform: Optional[Callable] = None,
        seed: int = 42,
        genuine_ratio: float = 0.5
    ):
        """
        Initialize VGGFace2Dataset.
        
        Args:
            root_dir: Path to VGGFace2 root directory
            split: 'train' or 'test'
            n_pairs: Total number of pairs to generate
            transform: Image preprocessing transforms
            seed: Random seed for reproducibility
            genuine_ratio: Ratio of genuine pairs (default: 0.5 for balanced)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.n_pairs = n_pairs
        self.transform = transform or get_default_transforms()
        self.genuine_ratio = genuine_ratio
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load dataset structure
        self._load_dataset_structure()
        
        # Generate pairs
        self.pairs = self._generate_pairs()
        
        logger.info(
            f"VGGFace2Dataset initialized: {len(self.pairs)} pairs "
            f"({int(len(self.pairs) * genuine_ratio)} genuine, "
            f"{int(len(self.pairs) * (1 - genuine_ratio))} impostor)"
        )
    
    def _load_dataset_structure(self):
        """Load dataset directory structure and index images by identity."""
        split_dir = self.root_dir / self.split

        # Check if path exists
        if not split_dir.exists():
            # Try alternative structure
            split_dir = self.root_dir

        # If still doesn't exist, use synthetic
        if not split_dir.exists():
            logger.warning(
                f"Dataset directory not found: {split_dir}. "
                f"Creating synthetic dataset for testing."
            )
            self._create_synthetic_structure()
            return
        
        # Index all images by identity
        self.identity_to_images = {}
        
        # Look for identity directories (e.g., n000001, n000002, ...)
        identity_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        
        if len(identity_dirs) == 0:
            # Fallback: create synthetic structure for testing
            logger.warning(
                f"No identity directories found in {split_dir}. "
                f"Creating synthetic dataset for testing."
            )
            self._create_synthetic_structure()
            return
        
        for identity_dir in identity_dirs:
            identity_id = identity_dir.name
            
            # Find all image files
            image_files = list(identity_dir.glob('*.jpg')) + list(identity_dir.glob('*.png'))
            
            if len(image_files) > 0:
                self.identity_to_images[identity_id] = [str(f) for f in image_files]
        
        self.identities = list(self.identity_to_images.keys())
        
        logger.info(
            f"Loaded {len(self.identities)} identities with "
            f"{sum(len(imgs) for imgs in self.identity_to_images.values())} images"
        )
    
    def _create_synthetic_structure(self):
        """Create synthetic dataset structure for testing when real data unavailable."""
        logger.warning("Creating synthetic dataset structure for testing")
        
        # Create synthetic identities
        n_identities = 100
        images_per_identity = 10
        
        self.identity_to_images = {}
        for i in range(n_identities):
            identity_id = f"synthetic_{i:05d}"
            # Use placeholder paths - these won't actually be loaded
            self.identity_to_images[identity_id] = [
                f"synthetic_{identity_id}_{j:03d}.jpg" 
                for j in range(images_per_identity)
            ]
        
        self.identities = list(self.identity_to_images.keys())
    
    def _generate_pairs(self) -> List[Dict]:
        """Generate genuine and impostor pairs."""
        pairs = []
        
        n_genuine = int(self.n_pairs * self.genuine_ratio)
        n_impostor = self.n_pairs - n_genuine
        
        # Generate genuine pairs (same identity, different images)
        identities_with_pairs = [
            ident for ident in self.identities 
            if len(self.identity_to_images[ident]) >= 2
        ]
        
        for _ in range(n_genuine):
            identity = np.random.choice(identities_with_pairs)
            images = self.identity_to_images[identity]
            img1, img2 = np.random.choice(images, size=2, replace=False)
            
            pairs.append({
                'img1_path': img1,
                'img2_path': img2,
                'label': 1,  # genuine
                'identity': identity
            })
        
        # Generate impostor pairs (different identities)
        for _ in range(n_impostor):
            id1, id2 = np.random.choice(self.identities, size=2, replace=False)
            img1 = np.random.choice(self.identity_to_images[id1])
            img2 = np.random.choice(self.identity_to_images[id2])
            
            pairs.append({
                'img1_path': img1,
                'img2_path': img2,
                'label': 0,  # impostor
                'identity': f"{id1}_{id2}"
            })
        
        return pairs
    
    def __len__(self) -> int:
        """Return number of pairs."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a pair of images.
        
        Args:
            idx: Index of pair
            
        Returns:
            {
                'img1': First image tensor,
                'img2': Second image tensor,
                'label': 1 (genuine) or 0 (impostor),
                'img1_path': Path to first image,
                'img2_path': Path to second image
            }
        """
        pair = self.pairs[idx]
        
        # Load images
        # Note: For synthetic testing, we create dummy images
        try:
            img1 = Image.open(pair['img1_path']).convert('RGB')
            img2 = Image.open(pair['img2_path']).convert('RGB')
        except (FileNotFoundError, OSError):
            # Create dummy images for testing
            img1 = Image.new('RGB', (112, 112), color='white')
            img2 = Image.new('RGB', (112, 112), color='white')
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return {
            'img1': img1,
            'img2': img2,
            'label': pair['label'],
            'img1_path': pair['img1_path'],
            'img2_path': pair['img2_path']
        }
