import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.cuda.amp import autocast, GradScaler
import os
import time
from typing import Optional, Tuple
import argparse
from contextlib import nullcontext


class DistributedTrainer:
    """
    Multi-GPU Distributed Training Framework
    Supports DDP, FSDP, and mixed precision training
    """
    
    def __init__(
        self,
        model: nn.Module,
        strategy: str = "ddp",
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
    ):
        self.strategy = strategy.lower()
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Initialize process group
        self._init_distributed()
        
        # Setup device
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        
        # Move model to device and wrap with distributed strategy
        self.model = self._setup_model(model)
        
        # Setup mixed precision
        self.scaler = GradScaler() if mixed_precision else None
        self.autocast_context = autocast if mixed_precision else nullcontext
        
    def _init_distributed(self):
        """Initialize distributed training environment"""
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
            )
        
        # Synchronize all processes
        dist.barrier()
        
    def _setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model with chosen distributed strategy"""
        model = model.to(self.device)
        
        if self.strategy == "ddp":
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
                static_graph=True,
            )
        elif self.strategy == "fsdp":
            auto_wrap_policy = size_based_auto_wrap_policy
            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=None,  # We handle mixed precision separately
                device_id=self.local_rank,
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return model
    
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        step: int,
    ) -> float:
        """Execute single training step with gradient accumulation"""
        inputs, targets = batch
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        
        # Scale loss by accumulation steps
        scale_factor = 1.0 / self.gradient_accumulation_steps
        
        with self.autocast_context():
            outputs = self.model(inputs)
            loss = criterion(outputs, targets) * scale_factor
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step (only on accumulation boundary)
        if (step + 1) % self.gradient_accumulation_steps == 0:
            if self.scaler:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        return loss.item() * self.gradient_accumulation_steps
    
    def reduce_loss(self, loss: float) -> float:
        """Average loss across all processes"""
        loss_tensor = torch.tensor(loss, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return (loss_tensor / self.world_size).item()
    
    def barrier(self):
        """Synchronize all processes"""
        dist.barrier()
    
    def is_main_process(self) -> bool:
        """Check if current process is main"""
        return self.rank == 0
    
    def cleanup(self):
        """Cleanup distributed training"""
        dist.destroy_process_group()


def benchmark_training(
    model: nn.Module,
    strategy: str,
    num_gpus: int,
    batch_size: int,
    num_iterations: int = 100,
    mixed_precision: bool = True,
) -> dict:
    """
    Benchmark training performance
    """
    trainer = DistributedTrainer(
        model=model,
        strategy=strategy,
        mixed_precision=mixed_precision,
    )
    
    # Create dummy data
    input_size = (batch_size, 3, 224, 224)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Warmup
    for _ in range(10):
        dummy_input = torch.randn(*input_size, device=trainer.device)
        dummy_target = torch.randint(0, 1000, (batch_size,), device=trainer.device)
        _ = trainer.train_step(
            (dummy_input, dummy_target),
            optimizer,
            criterion,
            step=0,
        )
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for step in range(num_iterations):
        dummy_input = torch.randn(*input_size, device=trainer.device)
        dummy_target = torch.randint(0, 1000, (batch_size,), device=trainer.device)
        loss = trainer.train_step(
            (dummy_input, dummy_target),
            optimizer,
            criterion,
            step=step,
        )
    
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    # Calculate metrics
    images_per_second = (num_iterations * batch_size * num_gpus) / elapsed_time
    
    trainer.cleanup()
    
    return {
        "strategy": strategy,
        "num_gpus": num_gpus,
        "batch_size": batch_size,
        "images_per_second": images_per_second,
        "time_per_iteration": elapsed_time / num_iterations,
        "mixed_precision": mixed_precision,
    }


class SimpleResNet(nn.Module):
    """Simple ResNet-like model for benchmarking"""
    
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4)
        self.layer3 = self._make_layer(256, 512, 6)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Distributed Training Framework")
    parser.add_argument("--strategy", type=str, default="ddp", choices=["ddp", "fsdp"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--mixed-precision", action="store_true", default=True)
    
    args = parser.parse_args()
    
    model = SimpleResNet(num_classes=1000)
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    
    results = benchmark_training(
        model=model,
        strategy=args.strategy,
        num_gpus=num_gpus,
        batch_size=args.batch_size,
        num_iterations=args.iterations,
        mixed_precision=args.mixed_precision,
    )
    
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"\nBenchmark Results:")
        print(f"Strategy: {results['strategy']}")
        print(f"GPUs: {results['num_gpus']}")
        print(f"Batch Size: {results['batch_size']}")
        print(f"Images/sec: {results['images_per_second']:.2f}")
        print(f"Time/iteration: {results['time_per_iteration']*1000:.2f}ms")
        print(f"Mixed Precision: {results['mixed_precision']}")
