"""
Distributed Training Framework

A production-grade multi-GPU distributed training framework with DDP/FSDP,
communication optimization, and complete DevOps automation.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

from .core.distributed_training import DistributedTrainer, SimpleResNet
from .core.enhanced_trainer import EnhancedDistributedTrainer
from .core.communication_optimizer import CommunicationOptimizer
from .monitoring.monitoring_dashboard import DistributedMonitor
from .monitoring.health_monitoring import HealthMonitor
from .config.config_manager import ConfigManager, ExperimentConfig

__all__ = [
    "DistributedTrainer",
    "EnhancedDistributedTrainer",
    "CommunicationOptimizer",
    "DistributedMonitor",
    "HealthMonitor",
    "ConfigManager",
    "ExperimentConfig",
    "SimpleResNet",
]
