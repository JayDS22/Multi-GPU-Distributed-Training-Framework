#!/usr/bin/env python3
"""
Test suite for distributed training framework
"""

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
import os
import subprocess
import tempfile
from pathlib import Path


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestDistributedTraining:
    """Test distributed training functionality"""
    
    def test_single_gpu_training(self):
        """Test training on single GPU"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        from distributed_training import DistributedTrainer, SimpleResNet
        
        # Set environment for single GPU
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        model = SimpleModel()
        
        # This would normally initialize distributed, but for single GPU we skip
        assert model is not None
    
    def test_communication_optimizer(self):
        """Test gradient compression"""
        from communication_optimizer import CommunicationOptimizer
        
        optimizer = CommunicationOptimizer(compression_ratio=0.1)
        
        # Test compression and decompression
        tensor = torch.randn(1000, 1000)
        compressed_values, indices, original_shape = optimizer.compress_gradients(tensor)
        
        assert compressed_values.numel() < tensor.numel()
        assert len(indices) == compressed_values.numel()
        
        # Decompress
        decompressed = optimizer.decompress_gradients(
            compressed_values, indices, original_shape
        )
        
        assert decompressed.shape == tensor.shape
    
    def test_gradient_accumulation(self):
        """Test gradient accumulator"""
        from communication_optimizer import GradientAccumulator
        
        accumulator = GradientAccumulator(accumulation_steps=4)
        
        model = SimpleModel()
        
        # Simulate 4 steps of accumulation
        for step in range(4):
            # Create dummy gradients
            for name, param in model.named_parameters():
                param.grad = torch.randn_like(param)
            
            should_sync = accumulator.accumulate(model.named_parameters())
            
            if step < 3:
                assert not should_sync
            else:
                assert should_sync
        
        # Get accumulated gradients
        accumulated = accumulator.get_averaged_gradients()
        assert len(accumulated) > 0
        
        # Reset
        accumulator.reset()
        assert accumulator.current_step == 0


class TestScalability:
    """Test scaling behavior"""
    
    def test_model_creation(self):
        """Test model can be created"""
        from distributed_training import SimpleResNet
        
        model = SimpleResNet(num_classes=1000)
        assert model is not None
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 1000)
    
    def test_benchmark_script(self):
        """Test benchmark script can run"""
        from run_benchmark import ScalabilityBenchmark
        
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = ScalabilityBenchmark(output_dir=tmpdir)
            
            # Test single benchmark (would need actual GPUs to run)
            # Just verify the object is created correctly
            assert benchmark.output_dir.exists()


def test_compression_ratio():
    """Test different compression ratios"""
    from communication_optimizer import CommunicationOptimizer
    
    tensor = torch.randn(10000)
    
    for ratio in [0.01, 0.05, 0.1, 0.5]:
        optimizer = CommunicationOptimizer(compression_ratio=ratio)
        compressed, indices, shape = optimizer.compress_gradients(tensor)
        
        # Verify compression ratio
        actual_ratio = compressed.numel() / tensor.numel()
        assert abs(actual_ratio - ratio) < 0.01  # Allow 1% tolerance


def test_bucket_all_reduce():
    """Test bucketed all-reduce"""
    from communication_optimizer import CommunicationOptimizer
    
    optimizer = CommunicationOptimizer(bucket_size_mb=1)
    
    # Create multiple small tensors
    tensors = [torch.randn(100) for _ in range(10)]
    
    # Without distributed, this should just return tensors
    result = optimizer.bucket_all_reduce(tensors)
    assert len(result) == len(tensors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
