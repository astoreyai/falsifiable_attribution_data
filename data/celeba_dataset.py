"""CelebA dataset loader with attribute annotations."""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, List, Tuple
import numpy as np


class CelebADataset(Dataset):
    """
    CelebA dataset with 40 binary attribute labels.

    Dataset structure expected:
        celeba/
        ├── img_align_celeba/
        │   ├── 000001.jpg
        │   └── ...
        ├── list_attr_celeba.txt  # 40 attributes
        └── list_eval_partition.txt  # train/val/test split

    Attributes (40 total):
        5_o_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes,
        Bald, Bangs, Big_Lips, Big_Nose, Black_Hair, Blond_Hair, Blurry,
        Brown_Hair, Bushy_Eyebrows, Chubby, Double_Chin, Eyeglasses, Goatee,
        Gray_Hair, Heavy_Makeup, High_Cheekbones, Male, Mouth_Slightly_Open,
        Mustache, Narrow_Eyes, No_Beard, Oval_Face, Pale_Skin, Pointy_Nose,
        Receding_Hairline, Rosy_Cheeks, Sideburns, Smiling, Straight_Hair,
        Wavy_Hair, Wearing_Earrings, Wearing_Hat, Wearing_Lipstick,
        Wearing_Necklace, Wearing_Necktie, Young
    """

    # Standard 40 CelebA attributes
    ATTRIBUTES = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
        'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]

    def __init__(
        self,
        root_dir: str,
        split: str = 'test',
        transform: Optional[callable] = None,
        attributes: Optional[List[str]] = None,
        n_samples: Optional[int] = None,
        seed: int = 42
    ):
        """
        Args:
            root_dir: Root directory of CelebA
            split: 'train', 'valid', or 'test'
            transform: Image transformations
            attributes: List of attribute names to load (default: all 40)
            n_samples: Number of samples to use (None = use all)
            seed: Random seed for sampling
        """
        self.root_dir = self._find_dataset_root(root_dir)
        self.split = split
        self.transform = transform
        self.n_samples = n_samples
        self.seed = seed

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(
                f"CelebA dataset not found at {self.root_dir}. "
                f"Please download from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"
            )

        # Use all attributes if not specified
        if attributes is None:
            self.selected_attributes = self.ATTRIBUTES
        else:
            # Validate attributes
            invalid = set(attributes) - set(self.ATTRIBUTES)
            if invalid:
                raise ValueError(f"Invalid attributes: {invalid}")
            self.selected_attributes = attributes

        # Load attribute annotations
        self.attr_df = self._load_attributes()

        # Filter by split
        self.image_paths = []
        self.attributes = []
        self._load_split()

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found for split '{split}'. "
                f"Please check dataset structure."
            )

    def _find_dataset_root(self, root_dir: str) -> str:
        """
        Find CelebA dataset by checking multiple possible locations.

        Args:
            root_dir: User-provided root directory

        Returns:
            Validated root directory path
        """
        # Possible dataset locations
        possible_paths = [
            root_dir,
            '/datasets/celeba',
            os.path.expanduser('~/datasets/celeba'),
            '/data/celeba',
            os.path.expanduser('~/data/celeba'),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Return original if none found (will raise error later)
        return root_dir

    def _load_attributes(self) -> pd.DataFrame:
        """Load attribute annotations from list_attr_celeba.txt."""
        attr_file = os.path.join(self.root_dir, 'list_attr_celeba.txt')

        if not os.path.exists(attr_file):
            # Try alternate location
            attr_file = os.path.join(self.root_dir, 'Anno', 'list_attr_celeba.txt')

        if not os.path.exists(attr_file):
            raise FileNotFoundError(
                f"Attribute file not found. Expected at:\n"
                f"  {os.path.join(self.root_dir, 'list_attr_celeba.txt')}\n"
                f"  or {os.path.join(self.root_dir, 'Anno', 'list_attr_celeba.txt')}"
            )

        # Read attribute file
        # Format: First line is count, second line is header, rest is data
        with open(attr_file, 'r') as f:
            lines = f.readlines()

        # Skip first line (count), use second line as header
        header = lines[1].strip().split()
        data_lines = [line.strip().split() for line in lines[2:]]

        # Create DataFrame
        df = pd.DataFrame(data_lines, columns=['image_id'] + header)

        # Convert attributes to integers (-1, 1) and then to float
        for col in header:
            df[col] = df[col].astype(int).astype(float)

        df.set_index('image_id', inplace=True)

        print(f"Loaded attributes for {len(df)} images")

        return df

    def _load_split(self):
        """Load images for specified split."""
        partition_file = os.path.join(self.root_dir, 'list_eval_partition.txt')

        if not os.path.exists(partition_file):
            # Try alternate location
            partition_file = os.path.join(self.root_dir, 'Eval', 'list_eval_partition.txt')

        # Map split name to partition ID
        split_map = {
            'train': 0,
            'valid': 1,
            'val': 1,
            'test': 2
        }

        if self.split not in split_map:
            raise ValueError(
                f"Invalid split '{self.split}'. "
                f"Must be one of: {list(split_map.keys())}"
            )

        target_partition = split_map[self.split]

        if os.path.exists(partition_file):
            # Load partition file
            partition_df = pd.read_csv(
                partition_file,
                delim_whitespace=True,
                header=None,
                names=['image_id', 'partition']
            )
            partition_df.set_index('image_id', inplace=True)

            # Filter images by partition
            split_images = partition_df[partition_df['partition'] == target_partition].index.tolist()
        else:
            # No partition file - use all images
            print(f"Warning: Partition file not found. Using all images for '{self.split}' split.")
            split_images = self.attr_df.index.tolist()

        # Image directory
        img_dir = os.path.join(self.root_dir, 'img_align_celeba')
        if not os.path.exists(img_dir):
            # Try alternate location
            img_dir = os.path.join(self.root_dir, 'Img', 'img_align_celeba')

        if not os.path.exists(img_dir):
            raise FileNotFoundError(
                f"Image directory not found. Expected at:\n"
                f"  {os.path.join(self.root_dir, 'img_align_celeba')}\n"
                f"  or {os.path.join(self.root_dir, 'Img', 'img_align_celeba')}"
            )

        # Collect valid image paths
        for img_id in split_images:
            img_path = os.path.join(img_dir, img_id)

            if os.path.exists(img_path):
                # Get attributes for this image
                if img_id in self.attr_df.index:
                    attrs = self.attr_df.loc[img_id, self.selected_attributes].values
                    self.image_paths.append(img_path)
                    self.attributes.append(attrs)

        # Sample if requested
        if self.n_samples is not None and self.n_samples < len(self.image_paths):
            np.random.seed(self.seed)
            indices = np.random.choice(
                len(self.image_paths),
                size=self.n_samples,
                replace=False
            )
            self.image_paths = [self.image_paths[i] for i in indices]
            self.attributes = [self.attributes[i] for i in indices]

        print(f"Loaded {len(self.image_paths)} images for split '{self.split}'")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and attribute vector.

        Returns:
            image: Face image tensor
            attributes: Binary attribute vector (N,) with values in {-1, 1}
        """
        img_path = self.image_paths[idx]
        attrs = self.attributes[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)

        # Convert attributes to tensor
        attrs = torch.tensor(attrs, dtype=torch.float32)

        return image, attrs

    def get_attribute_names(self) -> List[str]:
        """Get list of selected attribute names."""
        return self.selected_attributes

    def get_dataset_statistics(self) -> dict:
        """Get statistics about the dataset."""
        attrs_array = np.array(self.attributes)

        # Count positive attributes (value = 1)
        positive_counts = (attrs_array == 1).sum(axis=0)
        negative_counts = (attrs_array == -1).sum(axis=0)

        stats = {
            'n_images': len(self.image_paths),
            'n_attributes': len(self.selected_attributes),
            'split': self.split,
            'attribute_statistics': {}
        }

        for i, attr_name in enumerate(self.selected_attributes):
            stats['attribute_statistics'][attr_name] = {
                'positive_count': int(positive_counts[i]),
                'negative_count': int(negative_counts[i]),
                'positive_ratio': float(positive_counts[i]) / len(self.image_paths)
            }

        return stats

    def get_images_with_attribute(self, attribute: str, value: int = 1) -> List[int]:
        """
        Get indices of images with specific attribute value.

        Args:
            attribute: Attribute name
            value: Attribute value (1 or -1)

        Returns:
            List of image indices
        """
        if attribute not in self.selected_attributes:
            raise ValueError(f"Attribute '{attribute}' not in selected attributes")

        attr_idx = self.selected_attributes.index(attribute)
        attrs_array = np.array(self.attributes)

        indices = np.where(attrs_array[:, attr_idx] == value)[0].tolist()

        return indices
