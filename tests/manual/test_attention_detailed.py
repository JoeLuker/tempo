#!/usr/bin/env python3
"""Detailed test of attention extraction showing full matrix structure."""

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
        print(f"\n✓ Attention matrix extracted:")
        print(f"  Type: {type(attn)}")
        print(f"  Length (rows): {len(attn)}")
        if len(attn) > 0:
            print(f"  First row type: {type(attn[0])}")
            print(f"  First row length (cols): {len(attn[0]) if isinstance(attn[0], list) else 'N/A'}")
            print(f"\n  Full matrix shape: {len(attn)}x{len(attn[0]) if isinstance(attn[0], list) else '?'}")
            print(f"\n  First 3 rows (showing first 5 values each):")
            for i in range(min(3, len(attn))):
                if isinstance(attn[i], list):
                    print(f"    Row {i}: {attn[i][:5]}")
                else:
                    print(f"    Row {i}: {attn[i]} (not a list!)")
    else:
        print("✗ No attention matrix returned")
else:
    print(f"Error: {response.text}")
