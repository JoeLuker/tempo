#!/usr/bin/env python3
"""Test that attention weights are extracted and returned."""

import requests
import json
import time

print("Waiting for server to start...")
time.sleep(5)

print("Testing generation with attention extraction...")
response = requests.post('http://localhost:8765/api/generate', json={
    "prompt": "The cat",
    "max_tokens": 5,
    "selection_threshold": 0.25,
    "seed": 42
}, timeout=180)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Generated {len(data['nodes'])} tokens")

    if data.get('attention_matrix'):
        attn = data['attention_matrix']
        print(f"✓ Attention matrix extracted: {len(attn)}x{len(attn[0])}")
        print(f"  Sample attention weights from token 0:")
        print(f"    {attn[0][:5]}")  # First 5 weights from first token
    else:
        print("✗ No attention matrix returned")
else:
    print(f"Error: {response.text}")
