#!/usr/bin/env python3
"""
Production health monitoring and auto-recovery system
"""

import torch
import torch.distributed as dist
import psutil
import time
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import subprocess


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthMetrics:
    """System health metrics"""
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_utilization: float
    cpu_percent: float
    ram_percent: float
    disk_usage_percent: float
    iteration_time_ms: float
    communication_time_ms: float
    status: HealthStatus = HealthStatus.HEALTHY
    
    def to_dict(self) -> Dict:
        return {
            "gpu_memory_used_gb": self.gpu_memory_used,
            "gpu_memory_total_gb": self.gpu_memory_total,
            "gpu_memory_percent": (self.gpu_memory_used / self.gpu_memory_total * 100) if self.gpu_memory_total > 0 else 0,
            "gpu_utilization_percent": self.gpu_utilization,
            "cpu_percent": self.cpu_percent,
            "ram_percent": self.ram_percent,
            "disk_usage_percent": self.disk_usage_percent,
            "iteration_time_ms": self.iteration_time_ms,
            "communication_time_ms": self.communication_time_ms,
            "status": self.status.value,
        }


class HealthMonitor:
    """Monitor system health and trigger auto-recovery"""
    
    def __init__(
        self,
        check_interval: int = 60,
        gpu_memory_threshold: float = 0.95,
        iteration_time_threshold_ms: float = 1000,
        cpu_threshold: float = 90.0,
    ):
        self.check_interval = check_interval
        self.gpu_memory_threshold = gpu_memory_threshold
        self.iteration_time_threshold_ms = iteration_time_threshold_ms
        self.cpu_threshold = cpu_threshold
        
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else None
        
        self.metrics_history: List[HealthMetrics] = []
        self.last_check_time = time.time()
    
    def collect_metrics(self, iteration_time_ms: float = 0, comm_time_ms: float = 0) -> HealthMetrics:
        """Collect current system health metrics"""
        
        # GPU metrics
        if torch.cuda.is_available() and self.device is not None:
            gpu_memory_used = torch.cuda.memory_allocated(self.device) / 1024**3
            gpu_memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            
            # Try to get GPU utilization
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                gpu_util = gpus[self.device].load * 100 if self.device < len(gpus) else 0
            except:
                gpu_util = 0
        else:
            gpu_memory_used = 0
            gpu_memory_total = 0
            gpu_util = 0
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram_percent = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        metrics = HealthMetrics(
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_utilization=gpu_util,
            cpu_percent=cpu_percent,
            ram_percent=ram_percent,
            disk_usage_percent=disk_usage,
            iteration_time_ms=iteration_time_ms,
            communication_time_ms=comm_time_ms,
        )
        
        # Determine health status
        metrics.status = self._determine_health_status(metrics)
        
        return metrics
    
    def _determine_health_status(self, metrics: HealthMetrics) -> HealthStatus:
        """Determine overall health status from metrics"""
        
        # Critical conditions
        if metrics.gpu_memory_total > 0:
            gpu_memory_ratio = metrics.gpu_memory_used / metrics.gpu_memory_total
            if gpu_memory_ratio > 0.98:
                return HealthStatus.CRITICAL
        
        if metrics.cpu_percent > 98:
            return HealthStatus.CRITICAL
        
        # Unhealthy conditions
        if metrics.gpu_memory_total > 0:
            gpu_memory_ratio = metrics.gpu_memory_used / metrics.gpu_memory_total
            if gpu_memory_ratio > self.gpu_memory_threshold:
                return HealthStatus.UNHEALTHY
        
        if metrics.iteration_time_ms > self.iteration_time_threshold_ms * 2:
            return HealthStatus.UNHEALTHY
        
        # Degraded conditions
        if metrics.cpu_percent > self.cpu_threshold:
            return HealthStatus.DEGRADED
        
        if metrics.iteration_time_ms > self.iteration_time_threshold_ms:
            return HealthStatus.DEGRADED
        
        if metrics.ram_percent > 90:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def check_health(self, iteration_time_ms: float = 0, comm_time_ms: float = 0) -> HealthMetrics:
        """Perform health check and return metrics"""
        current_time = time.time()
        
        if current_time - self.last_check_time < self.check_interval:
            return self.metrics_history[-1] if self.metrics_history else None
        
        metrics = self.collect_metrics(iteration_time_ms, comm_time_ms)
        self.metrics_history.append(metrics)
        
        # Keep only last 100 metrics
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
        
        self.last_check_time = current_time
        
        return metrics
    
    def get_health_summary(self) -> Dict:
        """Get summary of recent health metrics"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 checks
        
        return {
            "current_status": self.metrics_history[-1].status.value,
            "avg_gpu_utilization": sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics),
            "avg_iteration_time_ms": sum(m.iteration_time_ms for m in recent_metrics) / len(recent_metrics),
            "peak_gpu_memory_gb": max(m.gpu_memory_used for m in recent_metrics),
            "health_checks": len(self.metrics_history),
        }


class AutoRecovery:
    """Automatic recovery mechanisms for training failures"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
    
    def attempt_recovery(self, error_type: str, trainer, optimizer) -> bool:
        """Attempt to recover from failure"""
        self.recovery_attempts += 1
        
        if self.recovery_attempts > self.max_recovery_attempts:
            print(f"Max recovery attempts ({self.max_recovery_attempts}) reached. Failing...")
            return False
        
        print(f"Attempting recovery from {error_type} (attempt {self.recovery_attempts})")
        
        if error_type == "oom":
            return self._recover_from_oom(trainer, optimizer)
        elif error_type == "communication":
            return self._recover_from_communication_failure()
        elif error_type == "checkpoint":
            return self._recover_from_checkpoint_failure(trainer, optimizer)
        else:
            return False
    
    def _recover_from_oom(self, trainer, optimizer) -> bool:
        """Recover from out-of-memory error"""
        print("OOM detected. Clearing cache and reducing batch size...")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Reduce batch size by 25%
        if hasattr(trainer, 'batch_size'):
            new_batch_size = max(8, int(trainer.batch_size * 0.75))
            print(f"Reducing batch size: {trainer.batch_size} -> {new_batch_size}")
            trainer.batch_size = new_batch_size
        
        # Enable gradient checkpointing if not already
        if hasattr(trainer, 'enable_activation_checkpointing'):
            trainer.enable_activation_checkpointing = True
            print("Enabled activation checkpointing")
        
        return True
    
    def _recover_from_communication_failure(self) -> bool:
        """Recover from distributed communication failure"""
        print("Communication failure detected. Reinitializing process group...")
        
        try:
            # Destroy existing process group
            if dist.is_initialized():
                dist.destroy_process_group()
            
            # Wait a bit
            time.sleep(5)
            
            # Reinitialize
            dist.init_process_group(backend="nccl")
            print("Process group reinitialized successfully")
            return True
        except Exception as e:
            print(f"Failed to recover from communication failure: {e}")
            return False
    
    def _recover_from_checkpoint_failure(self, trainer, optimizer) -> bool:
        """Recover from checkpoint loading failure"""
        print("Checkpoint failure detected. Attempting to load from backup...")
        
        # Try to find latest valid checkpoint
        import glob
        checkpoints = sorted(glob.glob(f"{self.checkpoint_dir}/checkpoint_epoch_*.pt"))
        
        for ckpt in reversed(checkpoints):
            try:
                print(f"Trying checkpoint: {ckpt}")
                trainer.load_checkpoint(ckpt, optimizer)
                print(f"Successfully loaded checkpoint: {ckpt}")
                return True
            except Exception as e:
                print(f"Failed to load {ckpt}: {e}")
                continue
        
        print("No valid checkpoint found. Starting from scratch.")
        return True  # Start fresh


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                print("Circuit breaker: Attempting half-open state")
            else:
                raise Exception("Circuit breaker is OPEN. Too many failures.")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
                print("Circuit breaker: Recovered to CLOSED state")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                print(f"Circuit breaker: OPENED after {self.failure_count} failures")
            
            raise e


class GracefulShutdown:
    """Handle graceful shutdown with cleanup"""
    
    def __init__(self, trainer, checkpoint_dir: str = "./checkpoints"):
        self.trainer = trainer
        self.checkpoint_dir = checkpoint_dir
        self.shutdown_requested = False
        
        # Register signal handlers
        import signal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def check_and_handle(self, epoch: int, optimizer) -> bool:
        """Check if shutdown requested and handle cleanup"""
        if self.shutdown_requested:
            print("Performing cleanup before shutdown...")
            
            # Save emergency checkpoint
            emergency_checkpoint = f"{self.checkpoint_dir}/emergency_checkpoint_epoch_{epoch}.pt"
            self.trainer.save_checkpoint(
                epoch=epoch,
                optimizer=optimizer,
                loss=0.0,
                is_best=False,
            )
            print(f"Emergency checkpoint saved: {emergency_checkpoint}")
            
            # Cleanup distributed
            if hasattr(self.trainer, 'cleanup'):
                self.trainer.cleanup()
            
            print("Graceful shutdown complete.")
            return True
        
        return False


class MetricsAggregator:
    """Aggregate and analyze metrics across ranks"""
    
    def __init__(self):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    def aggregate_health_metrics(self, local_metrics: HealthMetrics) -> Dict:
        """Gather health metrics from all ranks"""
        if not dist.is_initialized():
            return local_metrics.to_dict()
        
        # Convert to tensor for all_gather
        metrics_tensor = torch.tensor([
            local_metrics.gpu_memory_used,
            local_metrics.gpu_utilization,
            local_metrics.cpu_percent,
            local_metrics.iteration_time_ms,
        ], device=torch.cuda.current_device())
        
        # Gather from all ranks
        gathered_tensors = [torch.zeros_like(metrics_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_tensors, metrics_tensor)
        
        # Aggregate statistics
        all_metrics = torch.stack(gathered_tensors)
        
        return {
            "avg_gpu_memory_gb": all_metrics[:, 0].mean().item(),
            "max_gpu_memory_gb": all_metrics[:, 0].max().item(),
            "min_gpu_memory_gb": all_metrics[:, 0].min().item(),
            "avg_gpu_utilization": all_metrics[:, 1].mean().item(),
            "avg_cpu_percent": all_metrics[:, 2].mean().item(),
            "avg_iteration_time_ms": all_metrics[:, 3].mean().item(),
            "max_iteration_time_ms": all_metrics[:, 3].max().item(),
        }
    
    def detect_stragglers(self, iteration_times: List[float], threshold: float = 1.5) -> List[int]:
        """Detect slow workers (stragglers)"""
        if not dist.is_initialized():
            return []
        
        avg_time = sum(iteration_times) / len(iteration_times)
        stragglers = [
            i for i, t in enumerate(iteration_times)
            if t > avg_time * threshold
        ]
        
        return stragglers


if __name__ == "__main__":
    # Example usage
    import os
    
    if "RANK" in os.environ:
        # In distributed mode
        health_monitor = HealthMonitor()
        auto_recovery = AutoRecovery()
        circuit_breaker = CircuitBreaker()
        
        # Simulate health check
        metrics = health_monitor.check_health(iteration_time_ms=50.0)
        print(f"Health Status: {metrics.status.value}")
        print(f"Metrics: {metrics.to_dict()}")
        
        # Test recovery
        # auto_recovery.attempt_recovery("oom", None, None)
    else:
        print("Run with torchrun for distributed health monitoring")
