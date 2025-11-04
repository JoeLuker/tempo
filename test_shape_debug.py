#!/usr/bin/env python3
"""Direct test of attention matrix extraction without full server."""

import sys
sys.path.insert(0, '/Users/jluker/tempo')

import torch
from src.experiments.experiment_runner import run_experiment
from src.infrastructure.cache.cache_manager import CacheManager
import logging

logging.basicConfig(level=logging.INFO)

# Simple test
print("Running minimal generation to test attention extraction...")

config = {
    'model_name': 'deepcogito/cogito-v1-preview-llama-3B',
    'prompt': 'Hi',
    'max_tokens': 2,
    'selection_threshold': 0.7,  # High threshold = minimal parallel tokens
    'device': 'mps',
    'seed': 42
}

result = run_experiment(config)

if 'attention_matrix' in result and result['attention_matrix']:
    attn = result['attention_matrix']
    print(f"\n✓ Got attention_matrix!")
    print(f"  Type: {type(attn)}")
    print(f"  Shape: {len(attn)}×{len(attn[0]) if attn and len(attn) > 0 else '?'}")
    print(f"  Expected: NxN where N is sequence length")
    print(f"\n  Matrix preview:")
    for i, row in enumerate(attn[:min(3, len(attn))]):
        print(f"    Row {i}: {row[:5] if len(row) > 5 else row}")
else:
    print("\n✗ No attention_matrix in result")
    print(f"  Result keys: {result.keys()}")
