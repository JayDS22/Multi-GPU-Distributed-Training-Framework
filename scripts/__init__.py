
"""
Executable scripts for training and deployment

This package contains command-line scripts:
- production_train.py: Main training script with CLI
- Additional utility scripts
"""

__all__ = []

# Script utilities
def parse_common_args():
    """Common argument parser for scripts"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Distributed Training Framework')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    # Distributed arguments
    parser.add_argument('--strategy', type=str, default='ddp', 
                       choices=['ddp', 'fsdp'], help='Distributed strategy')
    parser.add_argument('--mixed-precision', action='store_true', 
                       help='Enable mixed precision training')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    return parser
