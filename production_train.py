#!/usr/bin/env python3
"""
Production training script with all enterprise features:
- Checkpoint/resume
- TensorBoard monitoring  
- Auto-tuning
- Dynamic batch sizing
- Fault tolerance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
import argparse
import os
import json
from pathlib import Path
from enhanced_trainer import EnhancedDistributedTrainer
from monitoring_dashboard import DistributedMonitor, ScalingEfficiencyTracker
from distributed_training import SimpleResNet


def create_dataloader(dataset, batch_size: int, is_distributed: bool = True):
    """Create distributed dataloader"""
    sampler = DistributedSampler(dataset) if is_distributed else None
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    return loader


def find_optimal_batch_size(
    model: nn.Module,
    device: torch.device,
    max_batch_size: int = 512,
) -> int:
    """Automatically find optimal batch size that fits in GPU memory"""
    batch_size = 2
    
    while batch_size <= max_batch_size:
        try:
            # Test forward and backward pass
            dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
            dummy_target = torch.randint(0, 1000, (batch_size,), device=device)
            
            output = model(dummy_input)
            loss = nn.functional.cross_entropy(output, dummy_target)
            loss.backward()
            
            # Clear gradients and cache
            model.zero_grad()
            torch.cuda.empty_cache()
            
            # Double batch size for next iteration
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                # Return the last successful batch size
                return batch_size // 2
            else:
                raise e
    
    return batch_size // 2


class DynamicBatchSizer:
    """Dynamically adjust batch size based on GPU memory usage"""
    
    def __init__(self, initial_batch_size: int, min_batch_size: int = 8, max_batch_size: int = 512):
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.oom_count = 0
    
    def adjust(self, oom_occurred: bool = False):
        """Adjust batch size based on OOM events"""
        if oom_occurred:
            self.oom_count += 1
            # Reduce batch size by 25%
            self.batch_size = max(self.min_batch_size, int(self.batch_size * 0.75))
        elif self.oom_count == 0:
            # Gradually increase if no OOM for a while
            self.batch_size = min(self.max_batch_size, int(self.batch_size * 1.1))
        
        return self.batch_size


def train_epoch(
    trainer: EnhancedDistributedTrainer,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    monitor: DistributedMonitor,
    epoch: int,
    scaler: ScalingEfficiencyTracker = None,
):
    """Train for one epoch"""
    trainer.model.train()
    total_loss = 0
    num_batches = 0
    
    for step, (inputs, targets) in enumerate(dataloader):
        step_start = time.time()
        
        # Training step
        metrics = trainer.train_step(
            batch=(inputs, targets),
            optimizer=optimizer,
            criterion=criterion,
            step=step,
        )
        
        total_loss += metrics['loss']
        num_batches += 1
        
        # Monitor metrics
        step_time = time.time() - step_start
        monitor.log_training_step(
            loss=metrics['loss'],
            batch_size=inputs.size(0),
            step_time=step_time,
            comm_time=metrics.get('communication_time', 0),
        )
        
        # Log progress
        if trainer.is_main_process() and step % 100 == 0:
            print(f"Epoch {epoch} | Step {step}/{len(dataloader)} | "
                  f"Loss: {metrics['loss']:.4f} | "
                  f"Time: {step_time*1000:.2f}ms")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Production Distributed Training")
    parser.add_argument("--strategy", type=str, default="ddp", choices=["ddp", "fsdp"])
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto if not specified)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--mixed-precision", action="store_true", default=True)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--auto-batch-size", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize model
    model = SimpleResNet(num_classes=1000)
    
    # Auto-find batch size if requested
    if args.auto_batch_size and args.batch_size is None:
        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        model_test = SimpleResNet(num_classes=1000).to(device)
        args.batch_size = find_optimal_batch_size(model_test, device)
        del model_test
        torch.cuda.empty_cache()
        print(f"Auto-selected batch size: {args.batch_size}")
    elif args.batch_size is None:
        args.batch_size = 32
    
    # Initialize trainer
    trainer = EnhancedDistributedTrainer(
        model=model,
        strategy=args.strategy,
        mixed_precision=args.mixed_precision,
        gradient_clip_val=args.gradient_clip,
        enable_activation_checkpointing=args.activation_checkpointing,
        cpu_offload=args.cpu_offload,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Initialize monitoring
    monitor = DistributedMonitor(log_dir="./logs")
    
    # Optimizer
    optimizer = optim.AdamW(trainer.model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = trainer.load_checkpoint(args.resume, optimizer)
        print(f"Resumed from epoch {start_epoch}")
    
    # Create dummy dataset (replace with real dataset)
    from torch.utils.data import TensorDataset
    num_samples = 10000
    dummy_data = torch.randn(num_samples, 3, 224, 224)
    dummy_labels = torch.randint(0, 1000, (num_samples,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    
    # Create dataloader
    dataloader = create_dataloader(dataset, args.batch_size, is_distributed=True)
    
    # Scaling efficiency tracker
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    tracker = ScalingEfficiencyTracker(baseline_throughput=450, baseline_gpus=1)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Set epoch for DistributedSampler
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        
        # Train epoch
        avg_loss = train_epoch(
            trainer=trainer,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            monitor=monitor,
            epoch=epoch,
        )
        
        epoch_time = time.time() - epoch_start
        
        # Log epoch summary
        monitor.log_epoch_summary(epoch, avg_loss, epoch_time)
        
        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        
        trainer.save_checkpoint(
            epoch=epoch,
            optimizer=optimizer,
            loss=avg_loss,
            is_best=is_best,
        )
        
        # Print summary
        if trainer.is_main_process():
            perf_metrics = trainer.get_performance_metrics()
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Throughput: {perf_metrics['samples_per_second_per_gpu']:.1f} samples/s/GPU")
            print(f"  Communication Overhead: {perf_metrics['communication_overhead_percent']:.1f}%")
            print(f"  GPU Memory Peak: {perf_metrics['gpu_memory_peak_gb']:.2f} GB")
    
    # Final metrics
    if trainer.is_main_process():
        final_metrics = trainer.get_performance_metrics()
        summary_stats = monitor.get_summary_stats()
        
        print("\n" + "="*60)
        print("FINAL PERFORMANCE METRICS")
        print("="*60)
        print(f"Total Training Time: {final_metrics['total_training_time']:.2f}s")
        print(f"Samples/sec/GPU: {final_metrics['samples_per_second_per_gpu']:.1f}")
        print(f"Communication Overhead: {final_metrics['communication_overhead_percent']:.1f}%")
        print(f"Peak GPU Memory: {final_metrics['gpu_memory_peak_gb']:.2f} GB")
        print(f"Average Loss: {summary_stats['avg_loss']:.4f}")
        print(f"Average GPU Utilization: {summary_stats['avg_gpu_util']:.1f}%")
        
        # Calculate scaling efficiency
        throughput = final_metrics['samples_per_second_per_gpu'] * world_size
        measurement = tracker.record_measurement(
            num_gpus=world_size,
            throughput=throughput,
            time_per_iteration=final_metrics['total_training_time'] / summary_stats['total_steps']
        )
        print(f"Scaling Efficiency: {measurement['scaling_efficiency_%']:.1f}%")
        
        # Export metrics
        monitor.export_metrics("training_metrics.json")
        
        # Generate report
        with open("scaling_report.md", "w") as f:
            f.write(tracker.generate_report())
        
        print("\n✓ Training complete!")
    
    # Cleanup
    monitor.close()
    trainer.cleanup()


if __name__ == "__main__":
    import time
    main()
