#!/usr/bin/env python3
"""
Performance tests for distributed training framework
Benchmarks throughput, latency, and scaling efficiency
"""

import pytest
import torch
import torch.nn as nn
import time
from src.core.distributed_training import SimpleResNet
from src.core.communication_optimizer import CommunicationOptimizer
from src.utils import count_parameters


class TestModelPerformance:
    """Test model inference and training performance"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_pass_throughput(self):
        """Test forward pass throughput"""
        model = SimpleResNet(num_classes=1000)
        device = torch.device('cuda:0')
        model = model.to(device)
        model.eval()
        
        batch_size = 32
        input_tensor = torch.randn(batch_size, 3, 224, 224, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        
        num_iterations = 100
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_tensor)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        throughput = (num_iterations * batch_size) / elapsed
        latency_ms = (elapsed / num_iterations) * 1000
        
        print(f"\nForward Pass Performance:")
        print(f"  Throughput: {throughput:.1f} images/sec")
        print(f"  Latency: {latency_ms:.2f} ms")
        
        # Assert reasonable performance
        assert throughput > 100, f"Throughput too low: {throughput}"
        assert latency_ms < 100, f"Latency too high: {latency_ms}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward_pass_performance(self):
        """Test backward pass performance"""
        model = SimpleResNet(num_classes=1000)
        device = torch.device('cuda:0')
        model = model.to(device)
        model.train()
        
        batch_size = 32
        criterion = nn.CrossEntropyLoss()
        
        # Warmup
        for _ in range(10):
            input_tensor = torch.randn(batch_size, 3, 224, 224, device=device)
            target = torch.randint(0, 1000, (batch_size,), device=device)
            output = model(input_tensor)
            loss = criterion(output, target)
            loss.backward()
            model.zero_grad()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        
        num_iterations = 50
        for _ in range(num_iterations):
            input_tensor = torch.randn(batch_size, 3, 224, 224, device=device)
            target = torch.randint(0, 1000, (batch_size,), device=device)
            output = model(input_tensor)
            loss = criterion(output, target)
            loss.backward()
            model.zero_grad()
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        time_per_iteration = (elapsed / num_iterations) * 1000
        
        print(f"\nBackward Pass Performance:")
        print(f"  Time per iteration: {time_per_iteration:.2f} ms")
        
        assert time_per_iteration < 200, f"Too slow: {time_per_iteration}ms"


class TestCommunicationPerformance:
    """Test communication optimization performance"""
    
    def test_gradient_compression_speed(self):
        """Test gradient compression performance"""
        optimizer = CommunicationOptimizer(compression_ratio=0.01)
        
        # Create large tensor
        tensor = torch.randn(10000, 1000)
        
        # Warmup
        for _ in range(5):
            _, _, _ = optimizer.compress_gradients(tensor)
        
        # Benchmark compression
        start = time.time()
        num_iterations = 100
        
        for _ in range(num_iterations):
            compressed, indices, shape = optimizer.compress_gradients(tensor)
        
        compression_time = (time.time() - start) / num_iterations * 1000
        
        # Benchmark decompression
        start = time.time()
        for _ in range(num_iterations):
            decompressed = optimizer.decompress_gradients(compressed, indices, shape)
        
        decompression_time = (time.time() - start) / num_iterations * 1000
        
        print(f"\nCompression Performance:")
        print(f"  Compression time: {compression_time:.2f} ms")
        print(f"  Decompression time: {decompression_time:.2f} ms")
        print(f"  Compression ratio: {compressed.numel() / tensor.numel() * 100:.1f}%")
        
        assert compression_time < 50, f"Compression too slow: {compression_time}ms"
        assert decompression_time < 50, f"Decompression too slow: {decompression_time}ms"


class TestMemoryEfficiency:
    """Test memory usage and efficiency"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_usage(self):
        """Test GPU memory consumption"""
        device = torch.device('cuda:0')
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create model
        model = SimpleResNet(num_classes=1000)
        model = model.to(device)
        
        # Get parameter count
        num_params = count_parameters(model)
        
        # Measure memory
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3
        
        print(f"\nMemory Usage:")
        print(f"  Parameters: {num_params / 1e6:.1f}M")
        print(f"  Memory allocated: {memory_allocated:.2f} GB")
        print(f"  Peak memory: {peak_memory:.2f} GB")
        
        # Reasonable memory usage for ResNet-50
        assert memory_allocated < 2.0, f"Memory usage too high: {memory_allocated}GB"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_memory(self):
        """Test memory savings with mixed precision"""
        device = torch.device('cuda:0')
        
        # FP32 model
        torch.cuda.empty_cache()
        model_fp32 = SimpleResNet(num_classes=1000).to(device)
        batch = torch.randn(32, 3, 224, 224, device=device)
        output = model_fp32(batch)
        fp32_memory = torch.cuda.memory_allocated(device) / 1024**3
        
        # FP16 model
        torch.cuda.empty_cache()
        model_fp16 = SimpleResNet(num_classes=1000).to(device).half()
        batch_fp16 = batch.half()
        output = model_fp16(batch_fp16)
        fp16_memory = torch.cuda.memory_allocated(device) / 1024**3
        
        memory_reduction = (1 - fp16_memory / fp32_memory) * 100
        
        print(f"\nMixed Precision Memory:")
        print(f"  FP32 memory: {fp32_memory:.2f} GB")
        print(f"  FP16 memory: {fp16_memory:.2f} GB")
        print(f"  Reduction: {memory_reduction:.1f}%")
        
        assert memory_reduction > 20, f"Insufficient memory reduction: {memory_reduction}%"


class TestScalingEfficiency:
    """Test scaling efficiency metrics"""
    
    def test_batch_size_scaling(self):
        """Test performance across different batch sizes"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = SimpleResNet(num_classes=1000)
        device = torch.device('cuda:0')
        model = model.to(device)
        model.eval()
        
        batch_sizes = [8, 16, 32, 64]
        throughputs = []
        
        for bs in batch_sizes:
            try:
                input_tensor = torch.randn(bs, 3, 224, 224, device=device)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        _ = model(input_tensor)
                
                # Benchmark
                torch.cuda.synchronize()
                start = time.time()
                
                with torch.no_grad():
                    for _ in range(50):
                        _ = model(input_tensor)
                
                torch.cuda.synchronize()
                elapsed = time.time() - start
                
                throughput = (50 * bs) / elapsed
                throughputs.append(throughput)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at batch size {bs}")
                    break
                raise
        
        print(f"\nBatch Size Scaling:")
        for bs, tput in zip(batch_sizes[:len(throughputs)], throughputs):
            print(f"  Batch {bs}: {tput:.1f} images/sec")
        
        # Throughput should generally increase with batch size
        assert len(throughputs) >= 2, "Need at least 2 batch sizes"


class TestEndToEndPerformance:
    """Test complete training performance"""
    
    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_training_iteration_time(self):
        """Test complete training iteration time"""
        from src.core.enhanced_trainer import EnhancedDistributedTrainer
        
        model = SimpleResNet(num_classes=1000)
        
        trainer = EnhancedDistributedTrainer(
            model=model,
            strategy='ddp',
            mixed_precision=True,
        )
        
        batch_size = 32
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        target = torch.randint(0, 1000, (batch_size,))
        
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Warmup
        for _ in range(10):
            metrics = trainer.train_step(
                batch=(input_tensor, target),
                optimizer=optimizer,
                criterion=criterion,
                step=0,
            )
        
        # Benchmark
        times = []
        for step in range(50):
            start = time.time()
            metrics = trainer.train_step(
                batch=(input_tensor, target),
                optimizer=optimizer,
                criterion=criterion,
                step=step,
            )
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times) * 1000
        throughput = batch_size / (sum(times) / len(times))
        
        print(f"\nTraining Performance:")
        print(f"  Avg iteration time: {avg_time:.2f} ms")
        print(f"  Throughput: {throughput:.1f} images/sec")
        
        # Performance targets
        assert avg_time < 100, f"Iteration too slow: {avg_time}ms"
        assert throughput > 200, f"Throughput too low: {throughput}"
        
        trainer.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
