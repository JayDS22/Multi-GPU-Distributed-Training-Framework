"""
Core distributed training components

This module contains the fundamental distributed training implementations:
- DistributedTrainer: Base DDP/FSDP trainer
- EnhancedDistributedTrainer: Production trainer with fault tolerance
- CommunicationOptimizer: Communication optimization strategies
"""

from .distributed_training import (
    DistributedTrainer,
    SimpleResNet,
    benchmark_training,
)

from .enhanced_trainer import EnhancedDistributedTrainer

from .communication_optimizer import (
    CommunicationOptimizer,
    GradientAccumulator,
    OverlapCommunicator,
    benchmark_communication,
)

__all__ = [
    # Trainers
    "DistributedTrainer",
    "EnhancedDistributedTrainer",
    
    # Models
    "SimpleResNet",
    
    # Optimization
    "CommunicationOptimizer",
    "GradientAccumulator",
    "OverlapCommunicator",
    
    # Benchmarking
    "benchmark_training",
    "benchmark_communication",
]
