"""
Utility helper functions for distributed training

Collection of commonly used utility functions for data processing,
metrics computation, and general helpers.
"""

import torch
import torch.distributed as dist
import numpy as np
import random
import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(local_rank: int = 0) -> torch.device:
    """
    Get PyTorch device for given local rank
    
    Args:
        local_rank: Local rank of the process
        
    Returns:
        PyTorch device (cuda:N or cpu)
    """
    if torch.cuda.is_available():
        return torch.device(f'cuda:{local_rank}')
    return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Calculate model size in megabytes
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def format_time(seconds: float) -> str:
    """
    Format seconds to human readable time (HH:MM:SS)
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_gpu_memory_usage(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Get current GPU memory usage
    
    Args:
        device: GPU device (default: current device)
        
    Returns:
        Dictionary with memory statistics in GB
    """
    if not torch.cuda.is_available():
        return {}
    
    if device is None:
        device = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        device = device.index
    
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'max_allocated_gb': max_allocated,
        'total_gb': total,
        'free_gb': total - allocated,
        'utilization_percent': (allocated / total) * 100 if total > 0 else 0
    }


def is_distributed() -> bool:
    """Check if running in distributed mode"""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank"""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes"""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is main process"""
    return get_rank() == 0


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce tensor and compute mean across all ranks
    
    Args:
        tensor: Input tensor
        
    Returns:
        Mean-reduced tensor
    """
    if not is_distributed():
        return tensor
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def save_json(data: Dict[str, Any], filepath: str):
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


class AverageMeter:
    """
    Computes and stores the average and current value
    Useful for tracking training metrics
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update meter with new value
        
        Args:
            val: New value
            n: Number of samples (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class ProgressTracker:
    """Track training progress and estimate time remaining"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
    
    def start(self):
        """Start tracking"""
        import time
        self.start_time = time.time()
    
    def update(self, steps: int = 1):
        """Update progress"""
        self.current_step += steps
    
    def get_eta(self) -> str:
        """Get estimated time remaining"""
        import time
        if self.start_time is None or self.current_step == 0:
            return "Unknown"
        
        elapsed = time.time() - self.start_time
        steps_per_sec = self.current_step / elapsed
        remaining_steps = self.total_steps - self.current_step
        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
        
        return format_time(eta_seconds)
    
    def get_progress(self) -> float:
        """Get progress percentage"""
        return (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0


def create_directories(paths: List[str]):
    """
    Create directories if they don't exist
    
    Args:
        paths: List of directory paths
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def cleanup_old_files(directory: str, pattern: str, keep_last: int = 3):
    """
    Remove old files keeping only the last N files
    
    Args:
        directory: Directory path
        pattern: File pattern (e.g., "checkpoint_*.pt")
        keep_last: Number of files to keep
    """
    files = sorted(Path(directory).glob(pattern))
    
    if len(files) > keep_last:
        for file in files[:-keep_last]:
            file.unlink()


def print_training_summary(
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_acc: float,
    val_acc: float,
    epoch_time: float,
):
    """
    Print formatted training summary
    
    Args:
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        train_acc: Training accuracy
        val_acc: Validation accuracy
        epoch_time: Epoch duration
    """
    print(f"\n{'='*60}")
    print(f"Epoch {epoch} Summary")
    print(f"{'='*60}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Train Acc:  {train_acc:.2f}%  | Val Acc:  {val_acc:.2f}%")
    print(f"Time: {format_time(epoch_time)}")
    print(f"{'='*60}\n")


def compute_accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> List[float]:
    """
    Computes the accuracy over the k top predictions
    
    Args:
        output: Model output (logits)
        target: Ground truth labels
        topk: Tuple of k values for top-k accuracy
        
    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        
        return res


def configure_cudnn(deterministic: bool = False, benchmark: bool = True):
    """
    Configure cuDNN settings
    
    Args:
        deterministic: Enable deterministic mode (slower but reproducible)
        benchmark: Enable benchmark mode (faster but non-deterministic)
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def linear_warmup_cosine_decay(
    current_step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float = 1e-6,
) -> float:
    """
    Learning rate schedule with linear warmup and cosine decay
    
    Args:
        current_step: Current training step
        total_steps: Total training steps
        warmup_steps: Number of warmup steps
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        
    Returns:
        Learning rate for current step
    """
    if current_step < warmup_steps:
        # Linear warmup
        return base_lr * (current_step / warmup_steps)
    else:
        # Cosine decay
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))


def log_gpu_stats():
    """Print GPU statistics"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print(f"\nGPU Statistics:")
    for i in range(torch.cuda.device_count()):
        stats = get_gpu_memory_usage(i)
        print(f"GPU {i} ({torch.cuda.get_device_name(i)}):")
        print(f"  Allocated: {stats['allocated_gb']:.2f} GB")
        print(f"  Reserved:  {stats['reserved_gb']:.2f} GB")
        print(f"  Total:     {stats['total_gb']:.2f} GB")
        print(f"  Free:      {stats['free_gb']:.2f} GB")
        print(f"  Utilization: {stats['utilization_percent']:.1f}%")


def synchronize_processes():
    """Synchronize all distributed processes"""
    if is_distributed():
        dist.barrier()


__all__ = [
    'set_seed',
    'get_device',
    'count_parameters',
    'get_model_size_mb',
    'format_time',
    'get_gpu_memory_usage',
    'is_distributed',
    'get_rank',
    'get_world_size',
    'is_main_process',
    'all_reduce_mean',
    'save_json',
    'load_json',
    'AverageMeter',
    'ProgressTracker',
    'create_directories',
    'cleanup_old_files',
    'print_training_summary',
    'compute_accuracy',
    'configure_cudnn',
    'get_learning_rate',
    'linear_warmup_cosine_decay',
    'log_gpu_stats',
    'synchronize_processes',
]
