"""Performance benchmarking suite for TEMPO."""

import time
import json
import argparse
import statistics
from typing import List, Dict, Any
from pathlib import Path
import torch
import numpy as np

from src.application.services.generation_service import GenerationService
from src.utils.config_manager import ConfigManager


class TempoPerformanceBenchmark:
    """Benchmarking suite for TEMPO generation performance."""
    
    def __init__(self, device: str = "auto"):
        """Initialize benchmark suite."""
        self.device = device
        self.service = GenerationService()
        self.results = []
        
    def benchmark_generation(
        self,
        prompt: str,
        max_tokens: int,
        selection_threshold: float,
        use_retroactive_removal: bool = False,
        runs: int = 5
    ) -> Dict[str, Any]:
        """Benchmark a single generation configuration."""
        timings = []
        tokens_per_second_list = []
        memory_usage = []
        
        config = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "selection_threshold": selection_threshold,
            "use_retroactive_removal": use_retroactive_removal,
            "debug_mode": False
        }
        
        # Warm-up run
        _ = self.service.generate(config)
        
        for run in range(runs):
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                # MPS doesn't have explicit cache clearing
                pass
                
            start_time = time.time()
            
            # Track memory before generation
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated()
            else:
                mem_before = 0
                
            result = self.service.generate(config)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Track memory after generation
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated()
                memory_usage.append((mem_after - mem_before) / (1024 ** 2))  # MB
            
            # Calculate metrics
            if result.get("generated_text"):
                num_tokens = len(result.get("token_ids", []))
                tokens_per_second = num_tokens / elapsed if elapsed > 0 else 0
                tokens_per_second_list.append(tokens_per_second)
            
            timings.append(elapsed)
        
        # Calculate statistics
        return {
            "config": config,
            "runs": runs,
            "timings": {
                "mean": statistics.mean(timings),
                "median": statistics.median(timings),
                "stdev": statistics.stdev(timings) if len(timings) > 1 else 0,
                "min": min(timings),
                "max": max(timings)
            },
            "tokens_per_second": {
                "mean": statistics.mean(tokens_per_second_list) if tokens_per_second_list else 0,
                "median": statistics.median(tokens_per_second_list) if tokens_per_second_list else 0,
                "stdev": statistics.stdev(tokens_per_second_list) if len(tokens_per_second_list) > 1 else 0
            },
            "memory_mb": {
                "mean": statistics.mean(memory_usage) if memory_usage else 0,
                "max": max(memory_usage) if memory_usage else 0
            }
        }
    
    def benchmark_threshold_sweep(
        self,
        prompt: str,
        max_tokens: int = 100,
        thresholds: List[float] = None,
        runs: int = 3
    ) -> List[Dict[str, Any]]:
        """Benchmark performance across different selection thresholds."""
        if thresholds is None:
            thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        
        results = []
        for threshold in thresholds:
            print(f"Benchmarking threshold: {threshold}")
            result = self.benchmark_generation(
                prompt=prompt,
                max_tokens=max_tokens,
                selection_threshold=threshold,
                runs=runs
            )
            results.append(result)
        
        return results
    
    def benchmark_sequence_length(
        self,
        prompt: str,
        lengths: List[int] = None,
        selection_threshold: float = 0.1,
        runs: int = 3
    ) -> List[Dict[str, Any]]:
        """Benchmark performance across different sequence lengths."""
        if lengths is None:
            lengths = [50, 100, 200, 500, 1000]
        
        results = []
        for length in lengths:
            print(f"Benchmarking sequence length: {length}")
            result = self.benchmark_generation(
                prompt=prompt,
                max_tokens=length,
                selection_threshold=selection_threshold,
                runs=runs
            )
            results.append(result)
        
        return results
    
    def benchmark_features(
        self,
        prompt: str,
        max_tokens: int = 100,
        runs: int = 3
    ) -> Dict[str, Any]:
        """Benchmark different feature combinations."""
        features = [
            {"name": "baseline", "config": {"use_retroactive_removal": False}},
            {"name": "with_retroactive", "config": {"use_retroactive_removal": True}},
            {"name": "high_threshold", "config": {"selection_threshold": 0.3}},
            {"name": "low_threshold", "config": {"selection_threshold": 0.01}},
        ]
        
        results = {}
        for feature in features:
            print(f"Benchmarking feature: {feature['name']}")
            config = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "selection_threshold": 0.1,
                **feature["config"]
            }
            
            result = self.benchmark_generation(
                **config,
                runs=runs
            )
            results[feature["name"]] = result
        
        return results
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        output_path = Path("benchmark") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of benchmark results."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        if isinstance(results, list):
            # Threshold or sequence length sweep
            for result in results:
                config = result["config"]
                timings = result["timings"]
                tps = result["tokens_per_second"]
                
                print(f"\nThreshold: {config['selection_threshold']}, "
                      f"Max Tokens: {config['max_tokens']}")
                print(f"  Time (s): {timings['mean']:.3f} ± {timings['stdev']:.3f}")
                print(f"  Tokens/s: {tps['mean']:.1f} ± {tps['stdev']:.1f}")
                
        elif isinstance(results, dict):
            # Feature comparison
            for name, result in results.items():
                timings = result["timings"]
                tps = result["tokens_per_second"]
                
                print(f"\n{name.upper()}")
                print(f"  Time (s): {timings['mean']:.3f} ± {timings['stdev']:.3f}")
                print(f"  Tokens/s: {tps['mean']:.1f} ± {tps['stdev']:.1f}")
                if result["memory_mb"]["max"] > 0:
                    print(f"  Memory (MB): {result['memory_mb']['max']:.1f}")


def main():
    """Run benchmarks."""
    parser = argparse.ArgumentParser(description="TEMPO Performance Benchmark")
    parser.add_argument("--prompt", type=str, 
                       default="The future of artificial intelligence is",
                       help="Prompt to use for benchmarking")
    parser.add_argument("--runs", type=int, default=5,
                       help="Number of runs per configuration")
    parser.add_argument("--suite", type=str, default="all",
                       choices=["all", "threshold", "length", "features"],
                       help="Which benchmark suite to run")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    benchmark = TempoPerformanceBenchmark(device=args.device)
    
    if args.suite in ["all", "threshold"]:
        print("\nRunning threshold sweep benchmark...")
        threshold_results = benchmark.benchmark_threshold_sweep(
            prompt=args.prompt,
            runs=args.runs
        )
        benchmark.results.append({
            "suite": "threshold_sweep",
            "results": threshold_results
        })
        benchmark.print_summary(threshold_results)
    
    if args.suite in ["all", "length"]:
        print("\nRunning sequence length benchmark...")
        length_results = benchmark.benchmark_sequence_length(
            prompt=args.prompt,
            runs=args.runs
        )
        benchmark.results.append({
            "suite": "sequence_length",
            "results": length_results
        })
        benchmark.print_summary(length_results)
    
    if args.suite in ["all", "features"]:
        print("\nRunning feature comparison benchmark...")
        feature_results = benchmark.benchmark_features(
            prompt=args.prompt,
            runs=args.runs
        )
        benchmark.results.append({
            "suite": "feature_comparison",
            "results": feature_results
        })
        benchmark.print_summary(feature_results)
    
    # Save all results
    benchmark.save_results()


if __name__ == "__main__":
    main()