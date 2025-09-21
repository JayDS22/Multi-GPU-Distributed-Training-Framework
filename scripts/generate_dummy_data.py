#!/usr/bin/env python3
"""
Generate dummy dataset for testing distributed training
Creates synthetic ImageNet-style dataset with images and labels
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
import argparse


def create_synthetic_image(size=(224, 224), class_id=0):
    """
    Create a synthetic image with random noise and class-based pattern
    
    Args:
        size: Image size (height, width)
        class_id: Class ID for pattern generation
    """
    # Create random RGB image
    img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    
    # Add class-specific pattern (simple gradient based on class)
    color_offset = class_id * 10
    for i in range(size[0]):
        img_array[i, :, 0] = np.clip(img_array[i, :, 0] + color_offset, 0, 255)
    
    return Image.fromarray(img_array, 'RGB')


def generate_dataset(
    output_dir: str,
    num_classes: int = 10,
    samples_per_class: int = 100,
    image_size: tuple = (224, 224),
    split: str = "train"
):
    """
    Generate synthetic dataset with class folders
    
    Args:
        output_dir: Output directory path
        num_classes: Number of classes
        samples_per_class: Images per class
        image_size: Image dimensions
        split: Dataset split (train/val)
    """
    output_path = Path(output_dir) / split
    
    print(f"Generating {split} dataset...")
    print(f"  Classes: {num_classes}")
    print(f"  Samples per class: {samples_per_class}")
    print(f"  Total samples: {num_classes * samples_per_class}")
    
    class_names = []
    
    for class_id in range(num_classes):
        class_name = f"class_{class_id:03d}"
        class_names.append(class_name)
        class_dir = output_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for sample_id in range(samples_per_class):
            # Generate synthetic image
            img = create_synthetic_image(size=image_size, class_id=class_id)
            
            # Save image
            img_filename = f"{class_name}_{sample_id:04d}.jpg"
            img_path = class_dir / img_filename
            img.save(img_path, 'JPEG', quality=95)
        
        print(f"  Created {samples_per_class} samples for {class_name}")
    
    # Create metadata
    metadata = {
        "dataset_name": f"synthetic_{split}",
        "num_classes": num_classes,
        "num_samples": num_classes * samples_per_class,
        "samples_per_class": samples_per_class,
        "image_size": list(image_size),
        "class_names": class_names,
        "split": split,
        "format": "ImageFolder",
        "description": "Synthetic dataset for testing distributed training"
    }
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved metadata to {metadata_path}")
    print(f"✓ {split.capitalize()} dataset created at {output_path}")
    
    return len(class_names) * samples_per_class


def generate_flat_dataset(
    output_dir: str,
    num_samples: int = 1000,
    num_classes: int = 10,
    image_size: tuple = (224, 224),
    split: str = "train"
):
    """
    Generate flat dataset (images and labels in separate files)
    
    Args:
        output_dir: Output directory
        num_samples: Total number of samples
        num_classes: Number of classes
        image_size: Image dimensions
        split: Dataset split
    """
    output_path = Path(output_dir) / split
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating flat {split} dataset...")
    print(f"  Samples: {num_samples}")
    print(f"  Classes: {num_classes}")
    
    labels = []
    
    for sample_id in range(num_samples):
        # Random class
        class_id = np.random.randint(0, num_classes)
        labels.append(class_id)
        
        # Generate image
        img = create_synthetic_image(size=image_size, class_id=class_id)
        
        # Save image
        img_filename = f"sample_{sample_id:06d}.jpg"
        img_path = images_dir / img_filename
        img.save(img_path, 'JPEG', quality=95)
    
    # Save labels
    labels_file = labels_dir / "labels.txt"
    with open(labels_file, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    
    # Save labels as numpy array
    labels_npy = labels_dir / "labels.npy"
    np.save(labels_npy, np.array(labels))
    
    print(f"  Saved {num_samples} images to {images_dir}")
    print(f"  Saved labels to {labels_file}")
    print(f"✓ Flat {split} dataset created")
    
    return num_samples


def main():
    parser = argparse.ArgumentParser(description="Generate dummy dataset")
    parser.add_argument("--output-dir", type=str, default="./data",
                       help="Output directory for dataset")
    parser.add_argument("--num-classes", type=int, default=10,
                       help="Number of classes")
    parser.add_argument("--train-samples-per-class", type=int, default=100,
                       help="Training samples per class")
    parser.add_argument("--val-samples-per-class", type=int, default=20,
                       help="Validation samples per class")
    parser.add_argument("--image-size", type=int, default=224,
                       help="Image size (square)")
    parser.add_argument("--format", type=str, default="imagefolder",
                       choices=["imagefolder", "flat"],
                       help="Dataset format")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Generating Dummy Dataset for Distributed Training")
    print("="*60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_size = (args.image_size, args.image_size)
    
    if args.format == "imagefolder":
        # ImageFolder format (class subdirectories)
        train_samples = generate_dataset(
            output_dir=args.output_dir,
            num_classes=args.num_classes,
            samples_per_class=args.train_samples_per_class,
            image_size=image_size,
            split="train"
        )
        
        val_samples = generate_dataset(
            output_dir=args.output_dir,
            num_classes=args.num_classes,
            samples_per_class=args.val_samples_per_class,
            image_size=image_size,
            split="val"
        )
    else:
        # Flat format (images and labels separate)
        train_samples = generate_flat_dataset(
            output_dir=args.output_dir,
            num_samples=args.num_classes * args.train_samples_per_class,
            num_classes=args.num_classes,
            image_size=image_size,
            split="train"
        )
        
        val_samples = generate_flat_dataset(
            output_dir=args.output_dir,
            num_samples=args.num_classes * args.val_samples_per_class,
            num_classes=args.num_classes,
            image_size=image_size,
            split="val"
        )
    
    # Create overall dataset info
    dataset_info = {
        "dataset_name": "synthetic_dataset",
        "format": args.format,
        "num_classes": args.num_classes,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "total_samples": train_samples + val_samples,
        "image_size": [args.image_size, args.image_size],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
    
    info_path = output_dir / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print("\n" + "="*60)
    print("Dataset Generation Complete!")
    print("="*60)
    print(f"Location: {output_dir.absolute()}")
    print(f"Format: {args.format}")
    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {val_samples}")
    print(f"Total samples: {train_samples + val_samples}")
    print("\nDataset is ready for training!")
    print("\nUsage:")
    print(f"  python scripts/production_train.py --batch-size 32 --epochs 2")
    print("="*60)


if __name__ == "__main__":
    main()
