import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import nullcontext


class EnhancedDistributedTrainer:
    """
    Production-ready distributed training framework with all enterprise features
    """
    
    def __init__(
        self,
        model: nn.Module,
        strategy: str = "ddp",
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        gradient_clip_val: float = 1.0,
        enable_activation_checkpointing: bool = False,
        cpu_offload: bool = False,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
    ):
        self.strategy = strategy.lower()
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_val = gradient_clip_val
        self.enable_activation_checkpointing = enable_activation_checkpointing
        
        # Distributed setup
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Initialize process group
        self._init_distributed()
        
        # Device setup
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        
        # Setup model with optimizations
        self.model = self._setup_model(model, cpu_offload)
        
        # Mixed precision setup
        self.scaler = GradScaler() if mixed_precision and strategy == "ddp" else None
        self.autocast_context = autocast if mixed_precision else nullcontext
        
        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Monitoring
        if self.is_main_process():
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
        
        # Metrics tracking
        self.metrics = {
            'training_time': 0,
            'communication_time': 0,
            'samples_processed': 0,
            'gpu_memory_allocated': 0,
            'gpu_memory_reserved': 0,
        }
        
        self.global_step = 0
    
    def _init_distributed(self):
        """Initialize distributed training environment"""
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
            )
        dist.barrier()
    
    def _setup_model(self, model: nn.Module, cpu_offload: bool = False) -> nn.Module:
        """Setup model with chosen distributed strategy and optimizations"""
        model = model.to(self.device)
        
        # Apply activation checkpointing if enabled
        if self.enable_activation_checkpointing:
            self._apply_activation_checkpointing(model)
        
        if self.strategy == "ddp":
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
                static_graph=True,
                find_unused_parameters=False,
            )
        elif self.strategy == "fsdp":
            # FSDP mixed precision policy
            mp_policy = None
            if self.mixed_precision:
                mp_policy = MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                )
            
            # CPU offload configuration
            cpu_offload_config = CPUOffload(offload_params=True) if cpu_offload else None
            
            model = FSDP(
                model,
                auto_wrap_policy=size_based_auto_wrap_policy,
                mixed_precision=mp_policy,
                device_id=self.local_rank,
                cpu_offload=cpu_offload_config,
                sharding_strategy=self._get_sharding_strategy(),
            )
        
        return model
    
    def _get_sharding_strategy(self):
        """Get optimal FSDP sharding strategy"""
        from torch.distributed.fsdp import ShardingStrategy
        
        # FULL_SHARD for maximum memory savings
        # HYBRID_SHARD for better performance on multi-node
        if self.world_size > 8:
            return ShardingStrategy.HYBRID_SHARD
        return ShardingStrategy.FULL_SHARD
    
    def _apply_activation_checkpointing(self, model: nn.Module):
        """Apply activation checkpointing to reduce memory"""
        # Find modules to checkpoint (typically large blocks)
        for module in model.modules():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                module.forward = lambda *args, m=module, **kwargs: checkpoint(m._original_forward, *args, **kwargs)
                module._original_forward = module.forward
    
    def train_step(
        self,
        batch: tuple,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        step: int,
    ) -> Dict[str, float]:
        """Enhanced training step with metrics tracking"""
        inputs, targets = batch
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        
        # Track computation time
        torch.cuda.synchronize()
        compute_start = time.time()
        
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
        
        torch.cuda.synchronize()
        compute_time = time.time() - compute_start
        
        # Communication and optimizer step
        if (step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.gradient_clip_val > 0:
                if self.scaler:
                    self.scaler.unscale_(optimizer)
                
                if self.strategy == "ddp":
                    torch.nn.utils.clip_grad_norm_(
                        self.model.module.parameters(),
                        self.gradient_clip_val
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val
                    )
            
            # Track communication time
            comm_start = time.time()
            
            if self.scaler:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            
            torch.cuda.synchronize()
            comm_time = time.time() - comm_start
            
            self.metrics['communication_time'] += comm_time
        else:
            comm_time = 0
        
        self.metrics['training_time'] += compute_time
        self.metrics['samples_processed'] += inputs.size(0) * self.world_size
        
        # Update global step
        self.global_step += 1
        
        # Log metrics
        if self.writer and (step + 1) % 100 == 0:
            self._log_metrics(loss.item() * self.gradient_accumulation_steps, compute_time, comm_time)
        
        return {
            'loss': loss.item() * self.gradient_accumulation_steps,
            'compute_time': compute_time,
            'communication_time': comm_time,
        }
    
    def _log_metrics(self, loss: float, compute_time: float, comm_time: float):
        """Log metrics to TensorBoard"""
        if not self.writer:
            return
        
        # Memory metrics
        memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        
        self.writer.add_scalar('Training/Loss', loss, self.global_step)
        self.writer.add_scalar('Performance/ComputeTime', compute_time * 1000, self.global_step)  # ms
        self.writer.add_scalar('Performance/CommunicationTime', comm_time * 1000, self.global_step)
        self.writer.add_scalar('Memory/Allocated_GB', memory_allocated, self.global_step)
        self.writer.add_scalar('Memory/Reserved_GB', memory_reserved, self.global_step)
        
        # Calculate ratios
        total_time = compute_time + comm_time
        if total_time > 0:
            comm_overhead = (comm_time / total_time) * 100
            self.writer.add_scalar('Performance/CommunicationOverhead_%', comm_overhead, self.global_step)
    
    def save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        loss: float,
        is_best: bool = False,
    ):
        """Save training checkpoint with fault tolerance"""
        if not self.is_main_process():
            return
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': self.metrics,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
        
        # Keep only last 3 checkpoints
        self._cleanup_old_checkpoints(keep_last=3)
    
    def load_checkpoint(self, checkpoint_path: str, optimizer: Optional[torch.optim.Optimizer] = None):
        """Load checkpoint for resuming training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.metrics = checkpoint.get('metrics', self.metrics)
        
        return checkpoint['epoch'], checkpoint['loss']
    
    def _cleanup_old_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoints keeping only the last N"""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        
        if len(checkpoints) > keep_last:
            for ckpt in checkpoints[:-keep_last]:
                ckpt.unlink()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate and return performance metrics"""
        total_time = self.metrics['training_time'] + self.metrics['communication_time']
        
        metrics = {
            'samples_per_second': self.metrics['samples_processed'] / total_time if total_time > 0 else 0,
            'samples_per_second_per_gpu': (self.metrics['samples_processed'] / total_time / self.world_size) if total_time > 0 else 0,
            'communication_overhead_percent': (self.metrics['communication_time'] / total_time * 100) if total_time > 0 else 0,
            'total_training_time': total_time,
            'gpu_memory_peak_gb': torch.cuda.max_memory_allocated(self.device) / 1024**3,
        }
        
        return metrics
    
    def is_main_process(self) -> bool:
        """Check if current process is main"""
        return self.rank == 0
    
    def barrier(self):
        """Synchronize all processes"""
        dist.barrier()
    
    def cleanup(self):
        """Cleanup distributed training"""
        if self.writer:
            self.writer.close()
        dist.destroy_process_group()
