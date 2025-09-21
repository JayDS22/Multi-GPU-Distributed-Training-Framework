
"""
Configuration management

This module handles all configuration aspects:
- Multi-environment configs (dev, staging, production)
- Configuration validation
- CLI override support
"""

from .config_manager import (
    ConfigManager,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    DistributedConfig,
    OptimizationConfig,
    CheckpointConfig,
    MonitoringConfig,
    DataConfig,
    Strategy,
    Precision,
)

__all__ = [
    # Main config manager
    "ConfigManager",
    
    # Configuration classes
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "DistributedConfig",
    "OptimizationConfig",
    "CheckpointConfig",
    "MonitoringConfig",
    "DataConfig",
    
    # Enums
    "Strategy",
    "Precision",
]
