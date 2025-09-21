#!/usr/bin/env python3
"""
Real-time monitoring dashboard for distributed training
"""

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import psutil
import GPUtil
from typing import Dict, List
import time
from collections import deque
import json


class DistributedMonitor:
    """
    Monitor distributed training metrics across all GPUs
    """
    
    def __init__(self, log_dir: str = "./logs", window_size: int = 100):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # TensorBoard writer (only on main process)
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
        
        # Metric history with sliding window
        self.window_size = window_size
        self.metrics_history = {
            'throughput': deque(maxlen=window_size),
            'loss': deque(maxlen=window_size),
            'gpu_util': deque(maxlen=window_size),
            'gpu_memory': deque(maxlen=window_size),
            'communication_time': deque(maxlen=window_size),
        }
        
        self.start_time = time.time()
        self.step = 0
    
    def log_training_step(
        self,
        loss: float,
        batch_size: int,
        step_time: float,
        comm_time: float = 0,
    ):
        """Log metrics for a single training step"""
        self.step += 1
        
        # Calculate throughput
        throughput = batch_size * self.world_size / step_time
        
        # Get GPU metrics
        gpu_metrics = self._get_gpu_metrics()
        
        # Update history
        self.metrics_history['throughput'].append(throughput)
        self.metrics_history['loss'].append(loss)
        self.metrics_history['gpu_util'].append(gpu_metrics['utilization'])
        self.metrics_history['gpu_memory'].append(gpu_metrics['memory_used_gb'])
        self.metrics_history['communication_time'].append(comm_time)
        
        # Synchronize metrics across all ranks
        synchronized_metrics = self._sync_metrics_across_ranks({
            'loss': loss,
            'throughput': throughput,
            'gpu_util': gpu_metrics['utilization'],
            'gpu_memory': gpu_metrics['memory_used_gb'],
            'comm_time': comm_time,
        })
        
        # Log to TensorBoard (main process only)
        if self.writer:
            self._log_to_tensorboard(synchronized_metrics)
        
        return synchronized_metrics
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get current GPU metrics"""
        device = torch.cuda.current_device()
        
        # PyTorch memory stats
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
        
        # GPU utilization (using GPUtil if available)
        utilization = 0
        try:
            gpus = GPUtil.getGPUs()
            if device < len(gpus):
                utilization = gpus[device].load * 100
        except:
            pass
        
        return {
            'memory_allocated_gb': memory_allocated,
            'memory_reserved_gb': memory_reserved,
            'memory_used_gb': memory_allocated,
            'max_memory_gb': max_memory,
            'utilization': utilization,
        }
    
    def _sync_metrics_across_ranks(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Average metrics across all ranks"""
        if not dist.is_initialized() or self.world_size == 1:
            return metrics
        
        synchronized = {}
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=torch.cuda.current_device())
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            synchronized[key] = (tensor / self.world_size).item()
        
        return synchronized
    
    def _log_to_tensorboard(self, metrics: Dict[str, float]):
        """Log metrics to TensorBoard"""
        for key, value in metrics.items():
            self.writer.add_scalar(f'Training/{key}', value, self.step)
        
        # Calculate and log derived metrics
        if len(self.metrics_history['communication_time']) > 0:
            avg_comm_time = sum(self.metrics_history['communication_time']) / len(self.metrics_history['communication_time'])
            total_time = time.time() - self.start_time
            comm_overhead = (avg_comm_time / (total_time / self.step)) * 100 if self.step > 0 else 0
            
            self.writer.add_scalar('Performance/CommunicationOverhead_%', comm_overhead, self.step)
        
        if len(self.metrics_history['throughput']) > 0:
            avg_throughput = sum(self.metrics_history['throughput']) / len(self.metrics_history['throughput'])
            self.writer.add_scalar('Performance/AvgThroughput', avg_throughput, self.step)
    
    def log_epoch_summary(self, epoch: int, avg_loss: float, epoch_time: float):
        """Log epoch-level summary"""
        if not self.writer:
            return
        
        self.writer.add_scalar('Epoch/Loss', avg_loss, epoch)
        self.writer.add_scalar('Epoch/Time_seconds', epoch_time, epoch)
        
        # Calculate epoch metrics
        if len(self.metrics_history['throughput']) > 0:
            avg_throughput = sum(self.metrics_history['throughput']) / len(self.metrics_history['throughput'])
            self.writer.add_scalar('Epoch/AvgThroughput', avg_throughput, epoch)
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics"""
        total_time = time.time() - self.start_time
        
        stats = {
            'total_steps': self.step,
            'total_time_seconds': total_time,
            'avg_throughput': sum(self.metrics_history['throughput']) / len(self.metrics_history['throughput']) if self.metrics_history['throughput'] else 0,
            'avg_loss': sum(self.metrics_history['loss']) / len(self.metrics_history['loss']) if self.metrics_history['loss'] else 0,
            'avg_gpu_util': sum(self.metrics_history['gpu_util']) / len(self.metrics_history['gpu_util']) if self.metrics_history['gpu_util'] else 0,
            'avg_gpu_memory_gb': sum(self.metrics_history['gpu_memory']) / len(self.metrics_history['gpu_memory']) if self.metrics_history['gpu_memory'] else 0,
            'max_gpu_memory_gb': torch.cuda.max_memory_allocated() / 1024**3,
        }
        
        return stats
    
    def export_metrics(self, output_path: str):
        """Export metrics to JSON"""
        stats = self.get_summary_stats()
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def close(self):
        """Close monitoring"""
        if self.writer:
            self.writer.close()


class ScalingEfficiencyTracker:
    """
    Track and calculate scaling efficiency metrics
    """
    
    def __init__(self, baseline_throughput: float, baseline_gpus: int = 1):
        self.baseline_throughput = baseline_throughput
        self.baseline_gpus = baseline_gpus
        self.measurements = []
    
    def record_measurement(self, num_gpus: int, throughput: float, time_per_iteration: float):
        """Record a scaling measurement"""
        ideal_throughput = self.baseline_throughput * (num_gpus / self.baseline_gpus)
        scaling_efficiency = (throughput / ideal_throughput) * 100 if ideal_throughput > 0 else 0
        
        measurement = {
            'num_gpus': num_gpus,
            'throughput': throughput,
            'time_per_iteration_ms': time_per_iteration * 1000,
            'ideal_throughput': ideal_throughput,
            'scaling_efficiency_%': scaling_efficiency,
        }
        
        self.measurements.append(measurement)
        return measurement
    
    def generate_report(self) -> str:
        """Generate scaling efficiency report"""
        report = "# Scaling Efficiency Report\n\n"
        report += "| GPUs | Throughput (img/s) | Ideal | Efficiency | Time/iter (ms) |\n"
        report += "|------|-------------------|-------|------------|----------------|\n"
        
        for m in sorted(self.measurements, key=lambda x: x['num_gpus']):
            report += f"| {m['num_gpus']} | {m['throughput']:.1f} | {m['ideal_throughput']:.1f} | "
            report += f"{m['scaling_efficiency_%']:.1f}% | {m['time_per_iteration_ms']:.2f} |\n"
        
        return report
    
    def get_communication_overhead(self, num_gpus: int) -> float:
        """Estimate communication overhead based on scaling efficiency"""
        for m in self.measurements:
            if m['num_gpus'] == num_gpus:
                # Communication overhead = 100% - scaling efficiency
                return 100 - m['scaling_efficiency_%']
        return 0


if __name__ == "__main__":
    # Example usage
    import os
    
    if "RANK" in os.environ:
        monitor = DistributedMonitor()
        tracker = ScalingEfficiencyTracker(baseline_throughput=1000, baseline_gpus=1)
        
        # Simulate training
        for step in range(100):
            loss = 0.5 - (step * 0.001)
            
            metrics = monitor.log_training_step(
                loss=loss,
                batch_size=32,
                step_time=0.1,
                comm_time=0.02,
            )
            
            time.sleep(0.1)
        
        # Export results
        stats = monitor.get_summary_stats()
        print(json.dumps(stats, indent=2))
        
        monitor.close()
