"""
Dataset utilities for distributed training
Handles data loading, preprocessing, and distributed sampling
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from pathlib import Path
import json
from typing import Optional, Tuple, Callable


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing without real data
    Generates random tensors on-the-fly
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 10,
        image_size: int = 224,
        transform: Optional[Callable] = None
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transform
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random image
        image = torch.randn(3, self.image_size, self.image_size)
        
        # Generate random label
        label = torch.randint(0, self.num_classes, (1,)).item()
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split: str = 'train', image_size: int = 224):
    """
    Get standard image transforms for training/validation
    
    Args:
        split: 'train' or 'val'
        image_size: Target image size
        
    Returns:
        torchvision.transforms composition
    """
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:  # validation
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # 256 for 224
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def load_dataset(
    data_path: str,
    split: str = 'train',
    transform: Optional[Callable] = None
) -> Dataset:
    """
    Load dataset from path
    Supports ImageFolder format and synthetic data
    
    Args:
        data_path: Path to dataset
        split: 'train' or 'val'
        transform: Optional transform function
        
    Returns:
        PyTorch Dataset
    """
    data_dir = Path(data_path)
    
    # Check if data exists
    if not data_dir.exists():
        print(f"Warning: Data path {data_dir} not found. Using synthetic data.")
        return SyntheticDataset(
            num_samples=1000 if split == 'train' else 200,
            num_classes=10,
            transform=transform
        )
    
    # Check if it's ImageFolder format
    if (data_dir / split).exists():
        data_dir = data_dir / split
    
    # Try to load as ImageFolder
    try:
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        print(f"Loaded {len(dataset)} samples from {data_dir}")
        return dataset
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Warning: Could not load ImageFolder from {data_dir}: {e}")
        print("Using synthetic data instead.")
        return SyntheticDataset(
            num_samples=1000 if split == 'train' else 200,
            num_classes=10,
            transform=transform
        )


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    is_distributed: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    shuffle: bool = True
) -> DataLoader:
    """
    Create DataLoader with optional distributed sampling
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size per GPU
        is_distributed: Whether to use DistributedSampler
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        shuffle: Shuffle data (ignored if using DistributedSampler)
        
    Returns:
        PyTorch DataLoader
    """
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last
        )
        shuffle = False  # Sampler handles shuffling
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0
    )
    
    return loader


def get_dataloader(
    data_path: str,
    split: str = 'train',
    batch_size: int = 32,
    is_distributed: bool = True,
    num_workers: int = 4,
    image_size: int = 224
) -> DataLoader:
    """
    Convenience function to get a ready-to-use dataloader
    
    Args:
        data_path: Path to dataset
        split: 'train' or 'val'
        batch_size: Batch size per GPU
        is_distributed: Use distributed sampler
        num_workers: Number of workers
        image_size: Image size
        
    Returns:
        DataLoader ready for training
    """
    # Get transforms
    transform = get_transforms(split=split, image_size=image_size)
    
    # Load dataset
    dataset = load_dataset(
        data_path=data_path,
        split=split,
        transform=transform
    )
    
    # Create dataloader
    loader = create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        is_distributed=is_distributed,
        num_workers=num_workers,
        shuffle=(split == 'train')
    )
    
    return loader


# Example usage
if __name__ == "__main__":
    import os
    
    # Set up for single GPU
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    
    # Test with synthetic data
    print("Testing with synthetic data...")
    synthetic_loader = get_dataloader(
        data_path='./nonexistent',  # Will use synthetic
        split='train',
        batch_size=32,
        is_distributed=False
    )
    
    # Get one batch
    images, labels = next(iter(synthetic_loader))
    print(f"✓ Synthetic data: {images.shape}, {labels.shape}")
    
    # Test with real data (if exists)
    print("\nTesting with real data...")
    try:
        real_loader = get_dataloader(
            data_path='./data',
            split='train',
            batch_size=32,
            is_distributed=False
        )
        images, labels = next(iter(real_loader))
        print(f"✓ Real data: {images.shape}, {labels.shape}")
    except Exception as e:
        print(f"Real data not available: {e}")
    
    print("\n✓ Dataset utilities working!")
