import torch
import torch.distributed as dist
from typing import List, Optional
import time


class CommunicationOptimizer:
    """
    Optimize communication overhead in distributed training
    Implements gradient compression, bucketing, and overlapping
    """
    
    def __init__(
        self,
        compression_ratio: float = 0.01,
        bucket_size_mb: int = 25,
        enable_overlap: bool = True,
    ):
        self.compression_ratio = compression_ratio
        self.bucket_size_mb = bucket_size_mb
        self.enable_overlap = enable_overlap
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
    def compress_gradients(self, gradients: torch.Tensor) -> tuple:
        """
        Top-K gradient compression
        Only communicate top-k% of gradients by magnitude
        """
        flat_grad = gradients.flatten()
        numel = flat_grad.numel()
        k = max(1, int(numel * self.compression_ratio))
        
        # Get top-k values and indices
        top_values, top_indices = torch.topk(flat_grad.abs(), k)
        
        # Get signs
        signs = torch.sign(flat_grad[top_indices])
        compressed_values = top_values * signs
        
        return compressed_values, top_indices, gradients.shape
    
    def decompress_gradients(
        self,
        compressed_values: torch.Tensor,
        indices: torch.Tensor,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """Decompress gradients back to original shape"""
        numel = torch.prod(torch.tensor(original_shape)).item()
        decompressed = torch.zeros(numel, device=compressed_values.device)
        decompressed[indices] = compressed_values
        return decompressed.reshape(original_shape)
    
    def all_reduce_compressed(
        self,
        tensor: torch.Tensor,
        async_op: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        All-reduce with gradient compression
        """
        if not dist.is_initialized():
            return tensor
        
        # Compress
        compressed_values, indices, original_shape = self.compress_gradients(tensor)
        
        # All-reduce compressed values
        dist.all_reduce(compressed_values, async_op=async_op)
        
        # Average
        compressed_values /= self.world_size
        
        # Decompress
        return self.decompress_gradients(compressed_values, indices, original_shape)
    
    def bucket_all_reduce(
        self,
        tensors: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Bucket small tensors together for efficient communication
        """
        if not dist.is_initialized():
            return tensors
        
        bucket_size_bytes = self.bucket_size_mb * 1024 * 1024
        buckets = []
        current_bucket = []
        current_size = 0
        
        # Group tensors into buckets
        for tensor in tensors:
            tensor_size = tensor.numel() * tensor.element_size()
            
            if current_size + tensor_size > bucket_size_bytes and current_bucket:
                buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
            
            current_bucket.append(tensor)
            current_size += tensor_size
        
        if current_bucket:
            buckets.append(current_bucket)
        
        # All-reduce each bucket
        results = []
        for bucket in buckets:
            # Flatten bucket
            flat_bucket = torch.cat([t.flatten() for t in bucket])
            
            # All-reduce
            dist.all_reduce(flat_bucket)
            flat_bucket /= self.world_size
            
            # Unflatten
            offset = 0
            for tensor in bucket:
                numel = tensor.numel()
                results.append(flat_bucket[offset:offset+numel].reshape(tensor.shape))
                offset += numel
        
        return results
    
    def hierarchical_all_reduce(
        self,
        tensor: torch.Tensor,
        intra_node_group: Optional[dist.ProcessGroup] = None,
        inter_node_group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """
        Two-level hierarchical all-reduce
        1. Reduce within nodes (fast NVLink/PCIe)
        2. Reduce across nodes (slower network)
        3. Broadcast within nodes
        """
        if not dist.is_initialized():
            return tensor
        
        # Step 1: Intra-node reduce
        if intra_node_group:
            dist.reduce(tensor, dst=0, group=intra_node_group)
        
        # Step 2: Inter-node all-reduce (only node leaders)
        if inter_node_group and self.rank == 0:
            dist.all_reduce(tensor, group=inter_node_group)
            tensor /= dist.get_world_size(inter_node_group)
        
        # Step 3: Intra-node broadcast
        if intra_node_group:
            dist.broadcast(tensor, src=0, group=intra_node_group)
        
        return tensor


class GradientAccumulator:
    """
    Accumulate gradients across multiple steps
    Reduces communication frequency
    """
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.accumulated_gradients = {}
    
    def accumulate(self, named_parameters) -> bool:
        """
        Accumulate gradients. Returns True when ready to sync.
        """
        for name, param in named_parameters:
            if param.grad is None:
                continue
            
            if name not in self.accumulated_gradients:
                self.accumulated_gradients[name] = param.grad.clone()
            else:
                self.accumulated_gradients[name] += param.grad
        
        self.current_step += 1
        
        if self.current_step >= self.accumulation_steps:
            return True
        return False
    
    def get_averaged_gradients(self):
        """Get accumulated and averaged gradients"""
        for name in self.accumulated_gradients:
            self.accumulated_gradients[name] /= self.accumulation_steps
        return self.accumulated_gradients
    
    def reset(self):
        """Reset accumulator"""
        self.current_step = 0
        self.accumulated_gradients.clear()


class OverlapCommunicator:
    """
    Overlap computation and communication
    Uses CUDA streams for async operations
    """
    
    def __init__(self):
        self.compute_stream = torch.cuda.Stream()
        self.comm_stream = torch.cuda.Stream()
        self.handles = []
    
    def async_all_reduce(self, tensor: torch.Tensor):
        """Launch async all-reduce"""
        with torch.cuda.stream(self.comm_stream):
            handle = dist.all_reduce(tensor, async_op=True)
            self.handles.append(handle)
    
    def wait_all(self):
        """Wait for all async operations"""
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        torch.cuda.synchronize()


def benchmark_communication(
    tensor_size: int = 1024 * 1024 * 100,  # 100M parameters
    num_iterations: int = 50,
) -> dict:
    """
    Benchmark different communication strategies
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    device = torch.device(f"cuda:{dist.get_rank()}")
    optimizer = CommunicationOptimizer()
    
    # Create test tensor
    tensor = torch.randn(tensor_size, device=device)
    
    results = {}
    
    # Baseline: Standard all-reduce
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        dist.all_reduce(tensor.clone())
    torch.cuda.synchronize()
    results["standard_allreduce"] = (time.time() - start) / num_iterations
    
    # Compressed all-reduce
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        optimizer.all_reduce_compressed(tensor.clone())
    torch.cuda.synchronize()
    results["compressed_allreduce"] = (time.time() - start) / num_iterations
    
    # Bucketed all-reduce
    tensors = [tensor[i*1000:(i+1)*1000].clone() for i in range(1000)]
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        optimizer.bucket_all_reduce(tensors)
    torch.cuda.synchronize()
    results["bucketed_allreduce"] = (time.time() - start) / num_iterations
    
    return results


if __name__ == "__main__":
    import os
    
    if "RANK" in os.environ:
        results = benchmark_communication()
        
        if dist.get_rank() == 0:
            print("\nCommunication Benchmark Results:")
            for method, time_ms in results.items():
                print(f"{method}: {time_ms*1000:.2f}ms")
    else:
        print("Run with torchrun for distributed benchmarking")
