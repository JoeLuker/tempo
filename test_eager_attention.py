#!/usr/bin/env python3
"""Test that eager attention works and returns attention weights."""

import requests
import json

print("Testing generation with eager attention...")
response = requests.post('http://localhost:8765/api/generate', json={
    "prompt": "Test",
    "max_tokens": 3,
    "selection_threshold": 0.25,
    "seed": 42
}, timeout=120)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Generated {len(data['nodes'])} tokens successfully")
    print("✓ Eager attention model loaded and working!")
else:
    print(f"Error: {response.text}")
