# Configuration Files

This directory contains environment-specific configuration files for distributed training.

## Available Configurations

### 1. Development (`dev.yaml`)
**Purpose:** Fast iteration and debugging

**Key Settings:**
- Small batch size (16)
- Few epochs (2)
- FP32 precision for easier debugging
- Profiling enabled
- Minimal optimization

**Use Cases:**
- Testing new features
- Debugging issues
- Quick experiments
- Local development

**Example:**
```bash
python scripts/production_train.py --config configs/dev.yaml
```

### 2. Staging (`staging.yaml`)
**Purpose:** Pre-production validation

**Key Settings:**
- Standard batch size (32)
- Partial training (10 epochs)
- Mixed precision enabled
- Some optimizations enabled
- Full monitoring

**Use Cases:**
- Integration testing
- Performance validation
- Configuration testing
- Pre-deployment checks

**Example:**
```bash
python scripts/production_train.py --config configs/staging.yaml
```

### 3. Production (`production.yaml`)
**Purpose:** Full-scale production training

**Key Settings:**
- Large batch size (64)
- Full training (100 epochs)
- FSDP strategy
- All optimizations enabled
- Complete monitoring & alerting

**Use Cases:**
- Production model training
- Large-scale experiments
- Final model deployment
- Multi-node training

**Example:**
```bash
python scripts/production_train.py --config configs/production.yaml
```

## Configuration Structure

Each config file contains:

```yaml
# Experiment metadata
experiment_name: production
seed: 42

# Model configuration
model:
  name: resnet50
  num_classes: 1000

# Training hyperparameters
training:
  batch_size: 64
  epochs: 100
  learning_rate: 0.001

# Distributed settings
distributed:
  strategy: fsdp  # or ddp
  precision: fp16  # or fp32, bf16

# Optimization settings
optimization:
  enable_gradient_compression: true
  enable_hierarchical_allreduce: true

# Checkpoint settings
checkpoint:
  save_dir: ./checkpoints
  save_frequency: 5

# Monitoring settings
monitoring:
  tensorboard: true
  log_frequency: 10
```

## Creating Custom Configurations

### Option 1: Copy Existing Config
```bash
cp configs/production.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml
python scripts/production_train.py --config configs/my_experiment.yaml
```

### Option 2: Generate Programmatically
```python
from src.config.config_manager import ConfigManager, ExperimentConfig

# Create custom config
config = ExperimentConfig(
    experiment_name="my_experiment",
    training=TrainingConfig(batch_size=128, epochs=50),
    distributed=DistributedConfig(strategy="ddp")
)

# Save
config.to_yaml("configs/my_experiment.yaml")
```

## Configuration Validation

All configs are automatically validated on load:

```python
from src.config.config_manager import ConfigManager

manager = ConfigManager()
config = manager.load_config("production")  # Auto-validates
```

Validation checks:
- ✅ Positive values for batch_size, epochs, etc.
- ✅ Valid strategy ('ddp', 'fsdp')
- ✅ Valid precision ('fp32', 'fp16', 'bf16')
- ✅ Compatible settings (e.g., cpu_offload only with FSDP)

## Environment-Specific Settings

### Development
```yaml
training:
  batch_size: 16      # Small for fast iteration
  epochs: 2           # Quick validation
distributed:
  precision: fp32     # Easier debugging
monitoring:
  enable_profiling: true
```

### Staging
```yaml
training:
  batch_size: 32      # Standard size
  epochs: 10          # Partial training
distributed:
  precision: fp16     # Test mixed precision
monitoring:
  wandb: true         # Track experiments
```

### Production
```yaml
training:
  batch_size: 64      # Optimized for throughput
  epochs: 100         # Full training
distributed:
  strategy: fsdp      # Best scalability
  precision: fp16     # 2x speedup
optimization:
  enable_gradient_compression: true
  enable_hierarchical_allreduce: true
monitoring:
  alert_webhook: "https://hooks.slack.com/..."
```

## CLI Overrides

You can override any config value from command line:

```bash
# Override batch size
python scripts/production_train.py \
  --config configs/production.yaml \
  --batch-size 128

# Override strategy
python scripts/production_train.py \
  --config configs/production.yaml \
  --strategy ddp

# Override multiple values
python scripts/production_train.py \
  --config configs/production.yaml \
  --batch-size 128 \
  --learning-rate 0.0005 \
  --epochs 50
```

## Configuration Best Practices

### 1. Start Small, Scale Up
```
dev.yaml → staging.yaml → production.yaml
```

### 2. Version Control
```bash
git add configs/my_experiment.yaml
git commit -m "Add experiment config for model v2"
```

### 3. Document Changes
```yaml
# my_experiment.yaml
# Purpose: Test new data augmentation
# Changes from production:
#   - Increased batch size to 128
#   - Added mixup augmentation
```

### 4. Use Descriptive Names
```
configs/
├── dev.yaml
├── staging.yaml
├── production.yaml
├── resnet101_imagenet.yaml
├── vit_large_finetune.yaml
└── llama_7b_pretrain.yaml
```

## Troubleshooting

### Config Not Found
```python
# Error: FileNotFoundError
# Solution: Create config or check path
from src.config.config_manager import ConfigManager
manager = ConfigManager()
manager.create_default_configs()  # Creates dev, staging, production
```

### Validation Error
```python
# Error: AssertionError: batch_size must be positive
# Solution: Fix invalid values
config.training.batch_size = 32  # Must be > 0
config.validate()
```

### Missing Required Fields
```yaml
# Error: Missing 'experiment_name'
# Solution: Add required fields
experiment_name: my_experiment  # Required
seed: 42                        # Required
```

## Quick Reference

| Config | Batch | Epochs | Strategy | Precision | Use Case |
|--------|-------|--------|----------|-----------|----------|
| dev    | 16    | 2      | DDP      | FP32      | Development |
| staging | 32   | 10     | DDP/FSDP | FP16      | Validation |
| production | 64 | 100   | FSDP     | FP16      | Production |

## Additional Resources

- Full schema: `src/config/config_manager.py`
- Examples: `tests/test_integration.py`
- Documentation: `DOCS/SETUP_GUIDE.md`
