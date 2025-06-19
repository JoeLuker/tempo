#!/usr/bin/env python3
"""
TEMPO API Client Examples
Shows how to interact with the TEMPO API programmatically
"""

import requests
import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional
import time


class TempoAPIClient:
    """Simple client for TEMPO API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def generate(self, 
                prompt: str,
                selection_threshold: float = 0.1,
                max_tokens: int = 100,
                **kwargs) -> Dict[str, Any]:
        """Generate text using TEMPO API"""
        
        payload = {
            "prompt": prompt,
            "selection_threshold": selection_threshold,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        response.raise_for_status()
        
        return response.json()
        
    def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False


async def async_generate(client: aiohttp.ClientSession,
                        base_url: str,
                        prompt: str,
                        **kwargs) -> Dict[str, Any]:
    """Async generation for concurrent requests"""
    
    payload = {
        "prompt": prompt,
        "selection_threshold": kwargs.get("selection_threshold", 0.1),
        "max_tokens": kwargs.get("max_tokens", 100),
        **kwargs
    }
    
    async with client.post(f"{base_url}/api/generate", json=payload) as response:
        return await response.json()


def example_basic_generation():
    """Basic API usage example"""
    print("\n" + "="*60)
    print("Example 1: Basic API Generation")
    print("="*60)
    
    client = TempoAPIClient()
    
    # Check API health
    if not client.health_check():
        print("❌ API is not running. Please start it with: uvicorn api:app --reload")
        return
        
    # Generate text
    result = client.generate(
        prompt="The meaning of life is",
        selection_threshold=0.1,
        max_tokens=50
    )
    
    print(f"Prompt: {result['prompt']}")
    print(f"Clean text: {result['clean_text']}")
    print(f"Generation time: {result['generation_time']:.2f}s")
    print(f"Tokens generated: {result['total_tokens']}")


def example_with_pruning():
    """API usage with pruning enabled"""
    print("\n" + "="*60)
    print("Example 2: Generation with Pruning")
    print("="*60)
    
    client = TempoAPIClient()
    
    if not client.health_check():
        print("❌ API is not running")
        return
        
    result = client.generate(
        prompt="Artificial intelligence will",
        selection_threshold=0.15,
        max_tokens=100,
        use_retroactive_pruning=True,
        attention_threshold=0.01,
        dynamic_threshold=True,
        bezier_p1=0.1,
        bezier_p2=0.9
    )
    
    print(f"Clean output:\n{result['clean_text']}")
    print(f"\nWith alternatives:\n{result['generated_text']}")


def example_batch_processing():
    """Process multiple prompts efficiently"""
    print("\n" + "="*60)
    print("Example 3: Batch Processing")
    print("="*60)
    
    client = TempoAPIClient()
    
    if not client.health_check():
        print("❌ API is not running")
        return
        
    prompts = [
        "The future of technology is",
        "In a world without gravity,",
        "The secret to happiness is",
        "Once upon a midnight dreary,"
    ]
    
    print("Processing batch of prompts...\n")
    
    results = []
    start_time = time.time()
    
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Processing: {prompt[:30]}...")
        
        result = client.generate(
            prompt=prompt,
            selection_threshold=0.12,
            max_tokens=40
        )
        results.append(result)
        
    total_time = time.time() - start_time
    
    print(f"\nBatch processing complete in {total_time:.2f}s")
    print("\nResults:")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['prompt']}")
        print(f"   → {result['clean_text']}")


async def example_concurrent_requests():
    """Concurrent API requests using asyncio"""
    print("\n" + "="*60)
    print("Example 4: Concurrent Requests")
    print("="*60)
    
    base_url = "http://localhost:8000"
    
    # Check API health first
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as response:
                if response.status != 200:
                    print("❌ API is not running")
                    return
    except:
        print("❌ API is not running")
        return
        
    prompts = [
        "The meaning of life is",
        "To be or not to be",
        "In the beginning was",
        "All happy families are",
        "It was the best of times"
    ]
    
    print(f"Sending {len(prompts)} concurrent requests...\n")
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            async_generate(
                session,
                base_url,
                prompt,
                selection_threshold=0.1,
                max_tokens=30
            )
            for prompt in prompts
        ]
        
        results = await asyncio.gather(*tasks)
        
    total_time = time.time() - start_time
    
    print(f"All requests completed in {total_time:.2f}s")
    print(f"Average time per request: {total_time/len(prompts):.2f}s\n")
    
    for result in results:
        print(f"• {result['prompt'][:30]}... → {result['clean_text'][:50]}...")


def example_parameter_comparison():
    """Compare different parameter settings"""
    print("\n" + "="*60)
    print("Example 5: Parameter Comparison")
    print("="*60)
    
    client = TempoAPIClient()
    
    if not client.health_check():
        print("❌ API is not running")
        return
        
    prompt = "The recipe for success includes"
    
    configs = [
        {"name": "Low threshold", "selection_threshold": 0.05},
        {"name": "Medium threshold", "selection_threshold": 0.15},
        {"name": "High threshold", "selection_threshold": 0.25},
        {"name": "With pruning", "selection_threshold": 0.15, 
         "use_retroactive_pruning": True, "attention_threshold": 0.01}
    ]
    
    print(f"Prompt: '{prompt}'\n")
    print("Comparing different configurations:")
    print("-" * 60)
    
    for config in configs:
        name = config.pop("name")
        result = client.generate(
            prompt=prompt,
            max_tokens=40,
            **config
        )
        
        print(f"\n{name}:")
        print(f"  Config: {config}")
        print(f"  Output: {result['clean_text']}")
        print(f"  Parallel tokens: {result.get('parallel_token_count', 'N/A')}")


def example_error_handling():
    """Demonstrate error handling"""
    print("\n" + "="*60)
    print("Example 6: Error Handling")
    print("="*60)
    
    client = TempoAPIClient()
    
    # Test various error conditions
    test_cases = [
        {
            "name": "Invalid threshold",
            "params": {"prompt": "Test", "selection_threshold": 2.0}
        },
        {
            "name": "Negative max_tokens",
            "params": {"prompt": "Test", "max_tokens": -10}
        },
        {
            "name": "Empty prompt",
            "params": {"prompt": "", "max_tokens": 50}
        }
    ]
    
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        try:
            result = client.generate(**test['params'])
            print(f"  ✓ Success: {result.get('clean_text', 'No output')}")
        except requests.exceptions.HTTPError as e:
            print(f"  ✗ HTTP Error: {e}")
            if e.response.text:
                print(f"    Details: {e.response.text}")
        except Exception as e:
            print(f"  ✗ Error: {type(e).__name__}: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TEMPO API Client Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
These examples demonstrate how to use the TEMPO API programmatically.

Make sure the API is running:
  uvicorn api:app --reload --port 8000

Examples:
  python examples/api_client.py --example 1  # Basic generation
  python examples/api_client.py --example 4  # Concurrent requests
  python examples/api_client.py --all        # Run all examples
        """
    )
    
    parser.add_argument(
        "--example",
        type=int,
        choices=range(1, 7),
        help="Run specific example (1-6)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all examples"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="API base URL"
    )
    
    args = parser.parse_args()
    
    # Update base URL if provided
    if args.base_url != "http://localhost:8000":
        TempoAPIClient.base_url = args.base_url
    
    examples = {
        1: ("Basic Generation", example_basic_generation),
        2: ("With Pruning", example_with_pruning),
        3: ("Batch Processing", example_batch_processing),
        4: ("Concurrent Requests", lambda: asyncio.run(example_concurrent_requests())),
        5: ("Parameter Comparison", example_parameter_comparison),
        6: ("Error Handling", example_error_handling)
    }
    
    if args.all:
        print("\nRunning all API examples...")
        for num, (name, func) in examples.items():
            try:
                func()
                if num < len(examples):
                    input("\nPress Enter for next example...")
            except KeyboardInterrupt:
                print("\n\nInterrupted.")
                break
                
    elif args.example:
        name, func = examples[args.example]
        print(f"\nRunning Example {args.example}: {name}")
        func()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()