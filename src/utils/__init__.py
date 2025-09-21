
"""
Utility functions and helper modules

This module contains utility functions used across the framework.
"""

# Placeholder for future utility functions
# Example utilities that could be added:
# - Data preprocessing helpers
# - Metric computation utilities
# - Visualization helpers
# - File I/O utilities

__all__ = []

def get_device(local_rank=0):
    """Get PyTorch device for given local rank"""
    import torch
    if torch.cuda.is_available():
        return torch.device(f'cuda:{local_rank}')
    return torch.device('cpu')

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    import torch
    import numpy as np
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_time(seconds):
    """Format seconds to human readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def get_gpu_memory_usage():
    """Get current GPU memory usage"""
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 - allocated
        }
    return None

__all__.extend([
    'get_device',
    'set_seed',
    'count_parameters',
    'format_time',
    'get_gpu_memory_usage',
])
