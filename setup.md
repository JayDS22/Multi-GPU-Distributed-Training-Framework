# Complete Setup Guide - From Git Clone to Running

## ✅ Prerequisites

### Hardware Requirements
- **Minimum:** 1 GPU (NVIDIA with CUDA support)
- **Recommended:** 4-8 GPUs
- **Tested:** NVIDIA V100, A100, RTX 3090/4090
- **Memory:** 16GB+ RAM, 10GB+ GPU memory per GPU

### Software Requirements
- **OS:** Linux (Ubuntu 20.04+ recommended)
- **CUDA:** 11.8 or 12.1+
- **Python:** 3.8, 3.9, or 3.10
- **NCCL:** 2.15+ (usually comes with CUDA)

## 🚀 Step-by-Step Setup

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/distributed-training-framework.git
cd distributed-training-framework

# Check your setup
nvidia-smi  # Verify GPUs are visible
python --version  # Should be 3.8+
```

### Step 2: Create Python Environment

```bash
# Option A: Using conda (recommended)
conda create -n dist-training python=3.10
conda activate dist-training

# Option B: Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA support first
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Step 4: Quick Verification Test

```bash
# Run unit tests (no GPU needed for some tests)
pytest test_distributed.py::TestDistributedTraining::test_communication_optimizer -v

# Run single GPU test
python -c "
from distributed_training import SimpleResNet
import torch
model = SimpleResNet()
x = torch.randn(2, 3, 224, 224)
out = model(x)
print(f'✓ Model works! Output shape: {out.shape}')
"
```

### Step 5: Run Your First Training

#### Option A: Single GPU (Easiest)
```bash
# Basic training on 1 GPU
python production_train.py \
    --strategy ddp \
    --batch-size 32 \
    --epochs 2 \
    --mixed-precision
```

#### Option B: Multi-GPU (Same Machine)
```bash
# Training on 4 GPUs
torchrun \
    --nproc_per_node=4 \
    production_train.py \
    --strategy ddp \
    --batch-size 32 \
    --epochs 2 \
    --mixed-precision

# Or use the launcher script
chmod +x launch_training.sh
./launch_training.sh 4 ddp 32
```

#### Option C: Multi-GPU with FSDP (Large Models)
```bash
torchrun \
    --nproc_per_node=4 \
    production_train.py \
    --strategy fsdp \
    --batch-size 64 \
    --epochs 2 \
    --mixed-precision \
    --activation-checkpointing
```

### Step 6: Monitor Training

```bash
# In another terminal, start TensorBoard
tensorboard --logdir=./logs --port=6006

# Open browser to http://localhost:6006
# You'll see real-time metrics!
```

### Step 7: Run Benchmarks

```bash
# Quick benchmark (1, 2, 4 GPUs)
python run_benchmark.py \
    --gpus 1 2 4 \
    --strategies ddp \
    --batch-sizes 32

# Full benchmark suite
python run_benchmark.py \
    --gpus 1 2 4 8 \
    --strategies ddp fsdp \
    --batch-sizes 32 64
```

## 🐛 Troubleshooting

### Issue 1: "CUDA out of memory"
```bash
# Solution: Reduce batch size
python production_train.py --batch-size 16

# Or use auto batch sizing
python production_train.py --auto-batch-size

# Or enable CPU offload (FSDP only)
python production_train.py --strategy fsdp --cpu-offload
```

### Issue 2: "NCCL Error" or Communication Timeout
```bash
# Set environment variables
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800

# Then retry training
torchrun --nproc_per_node=4 production_train.py ...
```

### Issue 3: "No module named 'distributed_training'"
```bash
# Make sure you're in the project directory
cd distributed-training-framework

# Reinstall in development mode
pip install -e .
```

### Issue 4: Port Already in Use (29500)
```bash
# Change master port
torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    production_train.py ...
```

## 📊 Verify Everything Works

### Complete Verification Script

```bash
#!/bin/bash
echo "=== Distributed Training Framework Verification ==="

# 1. Check GPU
echo -e "\n1. Checking GPUs..."
nvidia-smi --query-gpu=name,memory.total --format=csv

# 2. Check Python packages
echo -e "\n2. Checking Python packages..."
python -c "
import torch
import torch.distributed as dist
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'GPU Count: {torch.cuda.device_count()}')
"

# 3. Run unit tests
echo -e "\n3. Running unit tests..."
pytest test_distributed.py -v --tb=short || echo "Some tests require multi-GPU"

# 4. Test single GPU training
echo -e "\n4. Testing single GPU training..."
python production_train.py \
    --batch-size 16 \
    --epochs 1 \
    --mixed-precision || echo "Single GPU test failed"

# 5. Test multi-GPU (if available)
if [ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -gt 1 ]; then
    echo -e "\n5. Testing multi-GPU training..."
    torchrun --nproc_per_node=2 production_train.py \
        --batch-size 16 \
        --epochs 1 \
        --strategy ddp || echo "Multi-GPU test failed"
fi

echo -e "\n✓ Verification complete!"
```

Save this as `verify_setup.sh`, then run:
```bash
chmod +x verify_setup.sh
./verify_setup.sh
```

## 🎯 Expected Output

### Successful Training Output
```
Launching distributed training with:
  GPUs: 4
  Strategy: DDP
  Batch Size: 32
  Iterations: 100

Epoch 0 | Step 0/100 | Loss: 6.9089 | Time: 45.23ms
Epoch 0 | Step 100/100 | Loss: 6.2341 | Time: 42.18ms

Epoch 0 Summary:
  Loss: 6.3452
  Time: 4.32s
  Throughput: 1,148.3 samples/s/GPU
  Communication Overhead: 11.2%
  GPU Memory Peak: 8.45 GB

✓ Training complete!
```

### Successful Benchmark Output
```
Benchmark Results:
Strategy: ddp
GPUs: 4
Batch Size: 32
Images/sec: 4,592.00
Time/iteration: 27.84ms
Mixed Precision: True
Scaling Efficiency: 94.5%
```

## 🚀 Quick Start Examples

### Example 1: Minimal Training
```bash
# Just start training with defaults
python production_train.py
```

### Example 2: Production Training
```bash
# Full featured production setup
python production_train.py \
    --strategy fsdp \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --gradient-clip 1.0 \
    --mixed-precision \
    --activation-checkpointing \
    --checkpoint-dir ./checkpoints
```

### Example 3: Resume Training
```bash
# Resume from checkpoint
python production_train.py \
    --resume ./checkpoints/checkpoint_epoch_50.pt \
    --strategy ddp
```

### Example 4: Multi-Node Training (Advanced)
```bash
# Node 0 (master)
export MASTER_ADDR=192.168.1.100
export NODE_RANK=0
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    production_train.py --strategy ddp

# Node 1 (worker)
export MASTER_ADDR=192.168.1.100
export NODE_RANK=1
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    production_train.py --strategy ddp
```

## 📁 Project Files Overview

After cloning, you should see:
```
distributed-training-framework/
├── distributed_training.py      # ← Core DDP/FSDP
├── enhanced_trainer.py           # ← Production trainer
├── production_train.py           # ← Main training script
├── communication_optimizer.py    # ← Optimizations
├── monitoring_dashboard.py       # ← Metrics & TensorBoard
├── run_benchmark.py             # ← Benchmarking
├── test_distributed.py          # ← Tests
├── launch_training.sh           # ← Easy launcher
├── requirements.txt             # ← Dependencies
├── setup.py                     # ← Package setup
├── Dockerfile                   # ← Container
├── k8s-deployment.yaml          # ← Kubernetes
├── README.md                    # ← Main docs
├── IMPLEMENTATION_CHECKLIST.md  # ← Status report
├── QUICK_REFERENCE.md           # ← Quick guide
└── SETUP_GUIDE.md              # ← This file
```

## ✅ Final Checklist

Before running in production:

- [ ] GPU drivers installed (nvidia-smi works)
- [ ] CUDA 11.8+ or 12.1+ installed
- [ ] Python 3.8-3.10 installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Package installed (`pip install -e .`)
- [ ] Tests pass (`pytest test_distributed.py -v`)
- [ ] Single GPU works (`python production_train.py`)
- [ ] Multi-GPU works (`torchrun --nproc_per_node=2 production_train.py`)
- [ ] TensorBoard shows metrics (`tensorboard --logdir=./logs`)
- [ ] Benchmarks complete (`python run_benchmark.py`)

## 🎉 You're Ready!

If all steps completed successfully, you now have a fully functional distributed training framework!

**Next Steps:**
1. Replace `SimpleResNet` with your actual model
2. Replace dummy dataset with your real data
3. Tune hyperparameters for your task
4. Run full training
5. Deploy on your cluster

**Need Help?**
- Check `QUICK_REFERENCE.md` for common commands
- See `README.md` for detailed documentation
- Review `test_distributed.py` for usage examples

---
