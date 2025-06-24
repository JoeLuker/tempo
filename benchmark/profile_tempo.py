"""Profiling utilities for TEMPO to identify performance bottlenecks."""

import cProfile
import pstats
import io
import time
from pathlib import Path
from memory_profiler import profile
import torch

from src.application.services.generation_service import GenerationService


class TempoProfiler:
    """Profiling tools for TEMPO generation."""
    
    def __init__(self):
        """Initialize profiler."""
        self.service = GenerationService()
    
    def profile_generation(
        self,
        prompt: str,
        max_tokens: int = 100,
        selection_threshold: float = 0.1,
        output_file: str = "profile_results.txt"
    ):
        """Profile a single generation run."""
        config = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "selection_threshold": selection_threshold,
            "debug_mode": False
        }
        
        # CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        result = self.service.generate(config)
        end_time = time.time()
        
        profiler.disable()
        
        # Save profiling results
        output_path = Path("benchmark") / output_file
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(f"Generation completed in {end_time - start_time:.2f} seconds\n")
            f.write(f"Generated {len(result.get('token_ids', []))} tokens\n\n")
            
            # Write CPU profiling stats
            f.write("="*60 + "\n")
            f.write("CPU PROFILING RESULTS\n")
            f.write("="*60 + "\n\n")
            
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(50)  # Top 50 functions
            f.write(s.getvalue())
            
            # Write call graph
            f.write("\n" + "="*60 + "\n")
            f.write("CALL GRAPH (Top 20)\n")
            f.write("="*60 + "\n\n")
            
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_callers(20)
            f.write(s.getvalue())
        
        print(f"Profile results saved to {output_path}")
        
        # Also create a binary profile for further analysis
        profiler.dump_stats(str(Path("benchmark") / "profile.pstats"))
        print("Binary profile saved to benchmark/profile.pstats")
        print("You can analyze it with: python -m pstats benchmark/profile.pstats")
    
    @profile
    def memory_profile_generation(
        self,
        prompt: str,
        max_tokens: int = 100,
        selection_threshold: float = 0.1
    ):
        """Profile memory usage during generation."""
        config = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "selection_threshold": selection_threshold,
            "debug_mode": False
        }
        
        # Track GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
        
        result = self.service.generate(config)
        
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            print(f"\nGPU Memory Usage:")
            print(f"  Start: {start_memory / 1024**2:.1f} MB")
            print(f"  End: {end_memory / 1024**2:.1f} MB")
            print(f"  Peak: {peak_memory / 1024**2:.1f} MB")
            print(f"  Allocated: {(end_memory - start_memory) / 1024**2:.1f} MB")
    
    def profile_hotspots(
        self,
        prompt: str,
        max_tokens: int = 50,
        selection_threshold: float = 0.1
    ):
        """Identify performance hotspots in the generation pipeline."""
        config = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "selection_threshold": selection_threshold,
            "debug_mode": False
        }
        
        # Profile key components
        components = [
            "token_generation",
            "rope_modification",
            "attention_computation",
            "pruning",
            "text_formatting"
        ]
        
        timings = {}
        
        # Instrument the generation process
        # This would require adding timing hooks to the actual code
        # For now, we'll do overall profiling
        
        print("Profiling key components...")
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = self.service.generate(config)
        
        profiler.disable()
        
        # Extract timing for key functions
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        print("\nTop time-consuming functions:")
        print("="*60)
        
        # Print functions that take more than 1% of total time
        total_time = stats.total_tt
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            if tt / total_time > 0.01:  # More than 1% of total time
                percentage = (tt / total_time) * 100
                print(f"{func[2]:40s} {percentage:6.1f}% ({tt:.3f}s)")


def main():
    """Run profiling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TEMPO Performance Profiler")
    parser.add_argument("--prompt", type=str,
                       default="The future of artificial intelligence is",
                       help="Prompt to use for profiling")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--threshold", type=float, default=0.1,
                       help="Selection threshold")
    parser.add_argument("--mode", type=str, default="cpu",
                       choices=["cpu", "memory", "hotspots"],
                       help="Profiling mode")
    
    args = parser.parse_args()
    
    profiler = TempoProfiler()
    
    if args.mode == "cpu":
        print("Running CPU profiling...")
        profiler.profile_generation(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            selection_threshold=args.threshold
        )
    elif args.mode == "memory":
        print("Running memory profiling...")
        profiler.memory_profile_generation(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            selection_threshold=args.threshold
        )
    elif args.mode == "hotspots":
        print("Identifying performance hotspots...")
        profiler.profile_hotspots(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            selection_threshold=args.threshold
        )


if __name__ == "__main__":
    main()