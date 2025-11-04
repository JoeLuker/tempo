#!/usr/bin/env python3
"""Test the API to see what data is returned."""

import requests
import json

response = requests.post('http://localhost:8765/api/generate', json={
    "prompt": "Test",
    "max_tokens": 5,
    "selection_threshold": 0.25,
    "seed": 42
})

print(f"Status: {response.status_code}")
print(f"\nResponse JSON:")
data = response.json()
print(json.dumps(data, indent=2))

print(f"\nNumber of nodes: {len(data['nodes'])}")
print(f"Number of edges: {len(data['edges'])}")

if data['nodes']:
    print(f"\nFirst 3 nodes:")
    for node in data['nodes'][:3]:
        print(f"  {node}")
