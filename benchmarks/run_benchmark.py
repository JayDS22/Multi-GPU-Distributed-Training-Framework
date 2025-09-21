#!/usr/bin/env python3
"""
Scalability benchmark script for 1-256 GPUs
Tests DDP and FSDP with various configurations
"""

import os
import json
import subprocess
from typing import List, Dict
import argparse
from pathlib import Path


class ScalabilityBenchmark:
    """Run scalability benchmarks across multiple GPU configurations"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def run_single_benchmark(
        self,
        num_gpus: int,
        strategy: str,
        batch_size: int,
        mixed_precision: bool = True,
        num_iterations: int = 100,
    ) -> Dict:
        """Run benchmark on specified number of GPUs"""
        
        # Calculate nodes needed (8 GPUs per node)
        num_nodes = (num_gpus + 7) // 8
        gpus_per_node = min(8, num_gpus)
        
        # Build torchrun command
        cmd = [
            "torchrun",
            f"--nproc_per_node={gpus_per_node}",
            f"--nnodes={num_nodes}",
            "--node_rank=0",
            "--master_addr=localhost",
            "--master_port=29500",
            "distributed_training.py",
            f"--strategy={strategy}",
            f"--batch-size={batch_size}",
            f"--iterations={num_iterations}",
        ]
        
        if mixed_precision:
            cmd.append("--mixed-precision")
        
        print(f"\nRunning benchmark: {num_gpus} GPUs, {strategy}, batch_size={batch_size}")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            
            # Parse results from output
            output_lines = result.stdout.split('\n')
            benchmark_result = {
                'num_gpus': num_gpus,
                'strategy': strategy,
                'batch_size': batch_size,
                'mixed_precision': mixed_precision,
                'success': result.returncode == 0,
            }
            
            # Extract metrics from output
            for line in output_lines:
                if 'Images/sec:' in line:
                    benchmark_result['throughput'] = float(line.split(':')[1].strip())
                elif 'Time/iteration:' in line:
                    benchmark_result['latency_ms'] = float(line.split(':')[1].strip().rstrip('ms'))
            
            return benchmark_result
            
        except subprocess.TimeoutExpired:
            print(f"Benchmark timed out for {num_gpus} GPUs")
            return {
                'num_gpus': num_gpus,
                'strategy': strategy,
                'batch_size': batch_size,
                'success': False,
                'error': 'timeout',
            }
        except Exception as e:
            print(f"Error running benchmark: {e}")
            return {
                'num_gpus': num_gpus,
                'strategy': strategy,
                'batch_size': batch_size,
                'success': False,
                'error': str(e),
            }
    
    def run_scaling_study(
        self,
        gpu_configs: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
        strategies: List[str] = ['ddp', 'fsdp'],
        batch_sizes: List[int] = [32, 64, 128],
    ):
        """Run comprehensive scaling study"""
        
        for num_gpus in gpu_configs:
            for strategy in strategies:
                for batch_size in batch_sizes:
                    result = self.run_single_benchmark(
                        num_gpus=num_gpus,
                        strategy=strategy,
                        batch_size=batch_size,
                    )
                    self.results.append(result)
                    
                    # Save intermediate results
                    self.save_results()
    
    def save_results(self):
        """Save benchmark results to JSON"""
        output_file = self.output_dir / "scalability_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    def generate_report(self):
        """Generate markdown report from results"""
        report_file = self.output_dir / "benchmark_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Multi-GPU Distributed Training Benchmark Results\n\n")
            
            # Scaling efficiency table
            f.write("## Scaling Efficiency\n\n")
            f.write("| GPUs | Strategy | Batch Size | Throughput (img/s) | Latency (ms) | Efficiency |\n")
            f.write("|------|----------|------------|-------------------|--------------|------------|\n")
            
            baseline_throughput = {}
            
            for result in self.results:
                if not result.get('success', False):
                    continue
                
                key = f"{result['strategy']}_{result['batch_size']}"
                
                if result['num_gpus'] == 1:
                    baseline_throughput[key] = result.get('throughput', 0)
                
                throughput = result.get('throughput', 0)
                latency = result.get('latency_ms', 0)
                
                # Calculate efficiency
                if key in baseline_throughput and baseline_throughput[key] > 0:
                    ideal_throughput = baseline_throughput[key] * result['num_gpus']
                    efficiency = (throughput / ideal_throughput) * 100 if ideal_throughput > 0 else 0
                else:
                    efficiency = 100 if result['num_gpus'] == 1 else 0
                
                f.write(f"| {result['num_gpus']} | {result['strategy'].upper()} | "
                       f"{result['batch_size']} | {throughput:.1f} | {latency:.2f} | "
                       f"{efficiency:.1f}% |\n")
            
            # Communication overhead analysis
            f.write("\n## Communication Overhead\n\n")
            f.write("As the number of GPUs increases, communication overhead becomes the bottleneck.\n")
            f.write("FSDP typically shows better scaling for very large models due to reduced memory per GPU.\n")
        
        print(f"\nReport generated: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Run scalability benchmarks")
    parser.add_argument(
        "--gpus",
        type=int,
        nargs='+',
        default=[1, 2, 4, 8],
        help="GPU configurations to test"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs='+',
        default=['ddp', 'fsdp'],
        choices=['ddp', 'fsdp'],
        help="Strategies to benchmark"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs='+',
        default=[32, 64],
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    benchmark = ScalabilityBenchmark(output_dir=args.output_dir)
    benchmark.run_scaling_study(
        gpu_configs=args.gpus,
        strategies=args.strategies,
        batch_sizes=args.batch_sizes,
    )
    benchmark.generate_report()
    
    print("\n✓ Benchmarking complete!")


if __name__ == "__main__":
    main()
