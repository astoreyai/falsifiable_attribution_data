"""
Example usage of dataset loaders.

This script demonstrates how to use the VGGFace2 and CelebA dataset loaders.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

# Import our data package
from data import (
    VGGFace2Dataset,
    CelebADataset,
    get_insightface_transforms,
    get_default_transforms,
    collate_verification_pairs,
    collate_attribute_batch,
)


def example_vggface2():
    """Example: Load VGGFace2 verification pairs."""
    print("\n" + "="*60)
    print("VGGFace2 Dataset Example")
    print("="*60)

    # Define transforms
    transform = get_insightface_transforms(image_size=112)

    # Create dataset
    try:
        dataset = VGGFace2Dataset(
            root_dir='/datasets/vggface2',  # Will auto-search common paths
            split='test',
            transform=transform,
            n_pairs=200,
            seed=42
        )

        print(f"\nDataset loaded successfully!")
        print(f"Number of pairs: {len(dataset)}")

        # Get statistics
        stats = dataset.get_dataset_statistics()
        print(f"\nDataset Statistics:")
        print(f"  Total images: {stats['n_images']}")
        print(f"  Total identities: {stats['n_identities']}")
        print(f"  Images per identity: {stats['images_per_identity_mean']:.1f} ± {stats['images_per_identity_std']:.1f}")
        print(f"  Genuine pairs: {stats['n_genuine_pairs']}")
        print(f"  Impostor pairs: {stats['n_impostor_pairs']}")

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_verification_pairs
        )

        # Get one batch
        img1, img2, labels = next(iter(dataloader))
        print(f"\nBatch shapes:")
        print(f"  Image 1: {img1.shape}")
        print(f"  Image 2: {img2.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Label distribution: {labels.sum().item()} genuine, {(labels == 0).sum().item()} impostor")

    except FileNotFoundError as e:
        print(f"\nDataset not found: {e}")
        print("Please download VGGFace2 from: http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/")


def example_celeba():
    """Example: Load CelebA with attributes."""
    print("\n" + "="*60)
    print("CelebA Dataset Example")
    print("="*60)

    # Define transforms
    transform = get_default_transforms(image_size=224)

    # Create dataset
    try:
        dataset = CelebADataset(
            root_dir='/datasets/celeba',  # Will auto-search common paths
            split='test',
            transform=transform,
            n_samples=1000,
            seed=42
        )

        print(f"\nDataset loaded successfully!")
        print(f"Number of samples: {len(dataset)}")

        # Get statistics
        stats = dataset.get_dataset_statistics()
        print(f"\nDataset Statistics:")
        print(f"  Total images: {stats['n_images']}")
        print(f"  Number of attributes: {stats['n_attributes']}")

        # Show some attribute statistics
        print(f"\nSample Attribute Statistics:")
        sample_attrs = ['Male', 'Smiling', 'Eyeglasses', 'Young']
        for attr in sample_attrs:
            if attr in stats['attribute_statistics']:
                attr_stats = stats['attribute_statistics'][attr]
                print(f"  {attr}:")
                print(f"    Positive: {attr_stats['positive_count']} ({attr_stats['positive_ratio']:.1%})")
                print(f"    Negative: {attr_stats['negative_count']}")

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_attribute_batch
        )

        # Get one batch
        images, attributes = next(iter(dataloader))
        print(f"\nBatch shapes:")
        print(f"  Images: {images.shape}")
        print(f"  Attributes: {attributes.shape}")

        # Query specific attribute
        male_indices = dataset.get_images_with_attribute('Male', value=1)
        print(f"\nImages with 'Male' attribute: {len(male_indices)}")

    except FileNotFoundError as e:
        print(f"\nDataset not found: {e}")
        print("Please download CelebA from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")


def example_custom_attributes():
    """Example: Load CelebA with specific attributes only."""
    print("\n" + "="*60)
    print("CelebA Custom Attributes Example")
    print("="*60)

    # Select specific attributes
    selected_attrs = ['Male', 'Smiling', 'Eyeglasses', 'Young', 'Attractive']

    transform = get_default_transforms(image_size=224)

    try:
        dataset = CelebADataset(
            root_dir='/datasets/celeba',
            split='test',
            transform=transform,
            attributes=selected_attrs,
            n_samples=500
        )

        print(f"\nDataset loaded with {len(selected_attrs)} attributes:")
        print(f"  {', '.join(selected_attrs)}")
        print(f"Number of samples: {len(dataset)}")

        # Get one sample
        image, attrs = dataset[0]
        print(f"\nSample shapes:")
        print(f"  Image: {image.shape}")
        print(f"  Attributes: {attrs.shape}")
        print(f"  Attribute values: {attrs.tolist()}")

    except FileNotFoundError as e:
        print(f"\nDataset not found: {e}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Dataset Loader Examples")
    print("="*60)

    # Run examples
    example_vggface2()
    example_celeba()
    example_custom_attributes()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60 + "\n")
