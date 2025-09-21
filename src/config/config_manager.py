#!/usr/bin/env python3
"""
Production configuration management with validation, versioning,
and environment-specific configs
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
import os


class Strategy(Enum):
    DDP = "ddp"
    FSDP = "fsdp"
    DEEPSPEED = "deepspeed"


class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    name: str = "resnet50"
    num_classes: int = 1000
    pretrained: bool = False
    checkpoint_path: Optional[str] = None
    
    def validate(self):
        assert self.num_classes > 0, "num_classes must be positive"


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    
    def validate(self):
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.epochs > 0, "epochs must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 <= self.label_smoothing < 1, "label_smoothing must be in [0, 1)"


@dataclass
class DistributedConfig:
    """Distributed training configuration"""
    strategy: str = "ddp"
    backend: str = "nccl"
    precision: str = "fp16"
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = True
    cpu_offload: bool = False
    activation_checkpointing: bool = False
    sharding_strategy: str = "FULL_SHARD"
    
    def validate(self):
        valid_strategies = ["ddp", "fsdp", "deepspeed"]
        assert self.strategy in valid_strategies, f"strategy must be one of {valid_strategies}"
        assert self.backend in ["nccl", "gloo", "mpi"], "Invalid backend"
        assert self.precision in ["fp32", "fp16", "bf16"], "Invalid precision"


@dataclass
class OptimizationConfig:
    """Communication and compute optimization"""
    enable_gradient_compression: bool = False
    compression_ratio: float = 0.01
    enable_hierarchical_allreduce: bool = False
    bucket_size_mb: int = 25
    async_communication: bool = True
    overlap_computation: bool = True
    
    def validate(self):
        assert 0 < self.compression_ratio <= 1, "compression_ratio must be in (0, 1]"
        assert self.bucket_size_mb > 0, "bucket_size_mb must be positive"


@dataclass
class CheckpointConfig:
    """Checkpoint and model saving configuration"""
    save_dir: str = "./checkpoints"
    save_frequency: int = 1  # Save every N epochs
    keep_last_n: int = 3
    save_best: bool = True
    resume_from: Optional[str] = None
    
    def validate(self):
        assert self.save_frequency > 0, "save_frequency must be positive"
        assert self.keep_last_n > 0, "keep_last_n must be positive"


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    log_dir: str = "./logs"
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: Optional[str] = None
    log_frequency: int = 10
    enable_profiling: bool = False
    alert_webhook: Optional[str] = None
    
    def validate(self):
        assert self.log_frequency > 0, "log_frequency must be positive"
        if self.wandb:
            assert self.wandb_project is not None, "wandb_project required when wandb enabled"


@dataclass
class DataConfig:
    """Dataset configuration"""
    train_path: str = "./data/train"
    val_path: str = "./data/val"
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    def validate(self):
        assert self.num_workers >= 0, "num_workers must be non-negative"
        assert self.prefetch_factor >= 1, "prefetch_factor must be >= 1"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    experiment_name: str = "default_experiment"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def validate(self):
        """Validate all sub-configurations"""
        self.model.validate()
        self.training.validate()
        self.distributed.validate()
        self.optimization.validate()
        self.checkpoint.validate()
        self.monitoring.validate()
        self.data.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_json(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary"""
        return cls(
            experiment_name=config_dict.get('experiment_name', 'default_experiment'),
            seed=config_dict.get('seed', 42),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            distributed=DistributedConfig(**config_dict.get('distributed', {})),
            optimization=OptimizationConfig(**config_dict.get('optimization', {})),
            checkpoint=CheckpointConfig(**config_dict.get('checkpoint', {})),
            monitoring=MonitoringConfig(**config_dict.get('monitoring', {})),
            data=DataConfig(**config_dict.get('data', {})),
        )
    
    def merge_from_cli(self, args):
        """Merge CLI arguments into configuration"""
        if hasattr(args, 'batch_size') and args.batch_size:
            self.training.batch_size = args.batch_size
        if hasattr(args, 'learning_rate') and args.learning_rate:
            self.training.learning_rate = args.learning_rate
        if hasattr(args, 'epochs') and args.epochs:
            self.training.epochs = args.epochs
        if hasattr(args, 'strategy') and args.strategy:
            self.distributed.strategy = args.strategy
        # Add more CLI overrides as needed


class ConfigManager:
    """Manage experiment configurations with versioning and environments"""
    
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.environments = ["development", "staging", "production"]
    
    def create_default_configs(self):
        """Create default configuration files for different environments"""
        
        # Development config - fast iteration
        dev_config = ExperimentConfig(
            experiment_name="development",
            training=TrainingConfig(
                batch_size=16,
                epochs=2,
                learning_rate=0.001,
            ),
            monitoring=MonitoringConfig(
                tensorboard=True,
                enable_profiling=True,
            )
        )
        dev_config.to_yaml(self.config_dir / "dev.yaml")
        
        # Staging config - full validation
        staging_config = ExperimentConfig(
            experiment_name="staging",
            training=TrainingConfig(
                batch_size=32,
                epochs=10,
                learning_rate=0.001,
            ),
            distributed=DistributedConfig(
                strategy="ddp",
                precision="fp16",
            )
        )
        staging_config.to_yaml(self.config_dir / "staging.yaml")
        
        # Production config - full training
        prod_config = ExperimentConfig(
            experiment_name="production",
            training=TrainingConfig(
                batch_size=64,
                epochs=100,
                learning_rate=0.001,
                warmup_epochs=5,
            ),
            distributed=DistributedConfig(
                strategy="fsdp",
                precision="fp16",
                activation_checkpointing=True,
            ),
            optimization=OptimizationConfig(
                enable_gradient_compression=True,
                enable_hierarchical_allreduce=True,
            ),
            checkpoint=CheckpointConfig(
                save_frequency=5,
                keep_last_n=5,
            )
        )
        prod_config.to_yaml(self.config_dir / "production.yaml")
    
    def load_config(self, environment: str = "development") -> ExperimentConfig:
        """Load configuration for specific environment"""
        config_file = self.config_dir / f"{environment}.yaml"
        
        if not config_file.exists():
            print(f"Config {config_file} not found, creating defaults...")
            self.create_default_configs()
        
        config = ExperimentConfig.from_yaml(config_file)
        config.validate()
        return config
    
    def save_config(self, config: ExperimentConfig, name: str):
        """Save configuration with specific name"""
        config.validate()
        config.to_yaml(self.config_dir / f"{name}.yaml")
        config.to_json(self.config_dir / f"{name}.json")


if __name__ == "__main__":
    # Example usage
    config_manager = ConfigManager()
    config_manager.create_default_configs()
    
    # Load production config
    config = config_manager.load_config("production")
    print("Production Config:")
    print(yaml.dump(config.to_dict(), default_flow_style=False))
