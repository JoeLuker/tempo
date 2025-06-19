#!/usr/bin/env python3
"""
Batch processing example for TEMPO.

This script demonstrates how to process multiple prompts efficiently,
with different configurations and parallel processing capabilities.
"""

import json
import time
import asyncio
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.experiments.argument_parser import create_argument_parser
from src.experiments.experiment_runner import ExperimentRunner


class BatchProcessor:
    """Handles batch processing of multiple prompts."""
    
    def __init__(self, base_config: Dict[str, Any]):
        """Initialize with base configuration."""
        self.base_config = base_config
        self.results = []
    
    def process_prompt(self, prompt: str, custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a single prompt with optional custom configuration."""
        # Merge configurations
        config = self.base_config.copy()
        if custom_config:
            config.update(custom_config)
        
        # Add prompt to config
        config['prompt'] = prompt
        
        # Convert config to args list
        args = []
        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key.replace('_', '-')}")
            else:
                args.extend([f"--{key.replace('_', '-')}", str(value)])
        
        # Parse arguments and run
        parser = create_argument_parser()
        parsed_args = parser.parse_args(args)
        
        runner = ExperimentRunner(parsed_args)
        start_time = time.time()
        result = runner.run()
        result['processing_time'] = time.time() - start_time
        result['prompt'] = prompt
        
        return result
    
    def process_batch(self, prompts: List[Dict[str, Any]], show_progress: bool = True) -> List[Dict[str, Any]]:
        """Process a batch of prompts with their configurations."""
        results = []
        total = len(prompts)
        
        for i, prompt_config in enumerate(prompts):
            if show_progress:
                print(f"\nProcessing {i+1}/{total}: {prompt_config['prompt'][:50]}...")
            
            prompt = prompt_config.pop('prompt')
            result = self.process_prompt(prompt, prompt_config)
            results.append(result)
            
            if show_progress:
                print(f"✓ Generated {result.get('total_tokens', 0)} tokens in {result['processing_time']:.2f}s")
        
        self.results = results
        return results
    
    def save_results(self, filename: str):
        """Save batch results to JSON file."""
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'base_config': self.base_config,
            'total_prompts': len(self.results),
            'results': self.results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def print_summary(self):
        """Print summary statistics of batch processing."""
        if not self.results:
            print("No results to summarize.")
            return
        
        total_tokens = sum(r.get('total_tokens', 0) for r in self.results)
        total_time = sum(r.get('processing_time', 0) for r in self.results)
        avg_tokens = total_tokens / len(self.results)
        avg_time = total_time / len(self.results)
        
        print("\n" + "=" * 60)
        print("Batch Processing Summary")
        print("=" * 60)
        print(f"Total prompts processed: {len(self.results)}")
        print(f"Total tokens generated: {total_tokens}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average tokens per prompt: {avg_tokens:.1f}")
        print(f"Average time per prompt: {avg_time:.2f} seconds")
        print(f"Tokens per second: {total_tokens/total_time:.1f}")


def main():
    """Run batch processing examples."""
    
    # Example 1: Simple batch with same configuration
    print("Example 1: Batch with uniform configuration")
    print("=" * 60)
    
    base_config = {
        'max_tokens': 50,
        'selection_threshold': 0.1,
        'temperature': 0.8
    }
    
    prompts = [
        {'prompt': "The future of renewable energy is"},
        {'prompt': "Machine learning algorithms can be used to"},
        {'prompt': "The most important scientific discovery of the 21st century is"},
        {'prompt': "In the field of quantum computing,"},
        {'prompt': "Climate change impacts are most visible in"}
    ]
    
    processor = BatchProcessor(base_config)
    processor.process_batch(prompts)
    processor.print_summary()
    
    # Example 2: Batch with varying configurations
    print("\n\nExample 2: Batch with custom configurations per prompt")
    print("=" * 60)
    
    base_config = {
        'max_tokens': 75,
        'use_retroactive_pruning': True,
        'attention_threshold': 0.015
    }
    
    prompts = [
        {
            'prompt': "Write a haiku about artificial intelligence:",
            'selection_threshold': 0.15,
            'temperature': 0.9,
            'max_tokens': 30
        },
        {
            'prompt': "Explain the theory of relativity in simple terms:",
            'selection_threshold': 0.05,
            'temperature': 0.7,
            'max_tokens': 100
        },
        {
            'prompt': "The recipe for a perfect chocolate cake includes:",
            'selection_threshold': 0.08,
            'temperature': 0.8,
            'dynamic_threshold': True,
            'bezier_p1': 0.1,
            'bezier_p2': 0.9
        },
        {
            'prompt': "In a dystopian future where robots rule,",
            'selection_threshold': 0.12,
            'temperature': 0.95,
            'use_multi_scale_attention': True
        }
    ]
    
    processor = BatchProcessor(base_config)
    processor.process_batch(prompts)
    processor.print_summary()
    
    # Save results
    processor.save_results("batch_results.json")
    
    # Example 3: A/B testing different configurations
    print("\n\nExample 3: A/B Testing Configurations")
    print("=" * 60)
    
    test_prompt = "The meaning of life, according to science, is"
    
    configurations = [
        {
            'name': 'Conservative',
            'selection_threshold': 0.05,
            'temperature': 0.7,
            'max_tokens': 100
        },
        {
            'name': 'Balanced',
            'selection_threshold': 0.1,
            'temperature': 0.8,
            'max_tokens': 100,
            'use_retroactive_pruning': True,
            'attention_threshold': 0.02
        },
        {
            'name': 'Exploratory',
            'selection_threshold': 0.15,
            'temperature': 0.9,
            'max_tokens': 100,
            'use_retroactive_pruning': True,
            'attention_threshold': 0.01,
            'dynamic_threshold': True
        }
    ]
    
    print(f"Testing prompt: '{test_prompt}'")
    print("Testing different configurations...\n")
    
    for config in configurations:
        name = config.pop('name')
        print(f"\nConfiguration: {name}")
        print("-" * 40)
        
        processor = BatchProcessor(config)
        result = processor.process_prompt(test_prompt)
        
        print(f"Generated: {result['clean_text']}")
        print(f"Tokens: {result.get('total_tokens', 0)}")
        print(f"Time: {result.get('processing_time', 0):.2f}s")
        
        if 'parallel_positions' in result:
            print(f"Parallel positions: {result['parallel_positions']}")


if __name__ == "__main__":
    print("TEMPO Batch Processing Examples")
    print("==============================\n")
    
    try:
        main()
        
        print("\n" + "=" * 60)
        print("Batch Processing Complete!")
        print("=" * 60)
        print("""
These examples demonstrated:
1. Batch processing with uniform configuration
2. Custom configurations per prompt
3. A/B testing different settings
4. Saving results for analysis

Use batch processing for:
- Evaluating different configurations
- Processing multiple prompts efficiently
- Comparing generation strategies
- Building datasets
        """)
        
    except KeyboardInterrupt:
        print("\nBatch processing interrupted by user.")
    except Exception as e:
        print(f"\nError during batch processing: {e}")
        import traceback
        traceback.print_exc()