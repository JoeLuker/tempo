#!/usr/bin/env python3
"""Test API with higher threshold to see convergence."""

import requests
import json

# Use higher threshold to get more parallel tokens
response = requests.post('http://localhost:8765/api/generate', json={
    "prompt": "The cat",
    "max_tokens": 15,
    "selection_threshold": 0.40,  # Higher threshold = more parallel tokens
    "seed": 42
})

data = response.json()

print(f"Status: {response.status_code}")
print(f"Nodes: {len(data['nodes'])}")
print(f"Edges: {len(data['edges'])}")

# Find convergence patterns
convergence_nodes = []
for node in data['nodes']:
    if len(node['parent_ids']) > 1:
        convergence_nodes.append(node)

print(f"\nConvergence nodes (multiple parents): {len(convergence_nodes)}")
for node in convergence_nodes:
    print(f"  {node['id']}: '{node['text']}' has {len(node['parent_ids'])} parents: {node['parent_ids']}")

# Find divergence patterns
divergence_steps = {}
for node in data['nodes']:
    step = node['logical_step']
    if step not in divergence_steps:
        divergence_steps[step] = []
    divergence_steps[step].append(node)

print(f"\nDivergence (parallel tokens per step):")
for step in sorted(divergence_steps.keys()):
    nodes_at_step = divergence_steps[step]
    if len(nodes_at_step) > 1:
        tokens_text = [n['text'] for n in nodes_at_step]
        print(f"  Step {step}: {len(nodes_at_step)} parallel tokens: {tokens_text}")
