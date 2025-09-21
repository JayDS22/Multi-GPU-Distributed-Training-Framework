
"""
Benchmarking suite for distributed training


This package contains performance benchmarking tools:
- Scalability benchmarks (1-256 GPUs)
- Communication overhead analysis
- Throughput measurements
- Efficiency metrics
"""

from .run_benchmark import (
    ScalabilityBenchmark,
    benchmark_training,
)

__all__ = [
    'ScalabilityBenchmark',
    'benchmark_training',
]

# Benchmark utilities
def format_benchmark_results(results):
    """Format benchmark results for display"""
    output = []
    output.append("\n" + "="*60)
    output.append("BENCHMARK RESULTS")
    output.append("="*60)
    
    for result in results:
        if result.get('success', False):
            output.append(f"\n{result['num_gpus']} GPUs ({result['strategy'].upper()}):")
            output.append(f"  Throughput: {result.get('throughput', 0):.1f} samples/s")
            output.append(f"  Latency: {result.get('latency_ms', 0):.2f} ms/iter")
            if 'scaling_efficiency' in result:
                output.append(f"  Efficiency: {result['scaling_efficiency']:.1f}%")
    
    output.append("\n" + "="*60)
    return "\n".join(output)
