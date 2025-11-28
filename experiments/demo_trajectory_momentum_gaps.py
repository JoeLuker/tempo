"""
Trajectory Momentum Gap Filling

Compute the DIRECTIONAL FLOW (derivative/tangent) of the embedding trajectory
as position increases, then extrapolate that momentum to fill gaps.

Not the expected value (average), but the DIRECTION the space is moving.
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepcogito/cogito-v1-preview-llama-3B"
device = "mps"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

embed_layer = model.get_input_embeddings()

prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
prompt_length = input_ids.shape[1]

print(f"\nPrompt: '{prompt}' (length {prompt_length})")
print()

threshold = 0.02

print("="*80)
print("TRAJECTORY MOMENTUM GAP FILLING")
print("="*80)
print()

# Get embeddings for the prompt tokens
with torch.no_grad():
    prompt_embeds = embed_layer(input_ids)  # [1, prompt_length, hidden_size]

print(f"Prompt embeddings shape: {prompt_embeds.shape}")
print()

# Compute the MOMENTUM - directional flow of embeddings through the prompt
# This is the derivative/tangent: how embeddings change from one position to the next
print("Computing trajectory momentum...")

# Calculate differences between consecutive positions
diffs = []
for i in range(1, prompt_length):
    diff = prompt_embeds[0, i] - prompt_embeds[0, i-1]  # Direction from i-1 to i
    diffs.append(diff)

# The momentum is the average direction of change
# This gives us the "velocity vector" of the embedding trajectory
momentum = torch.stack(diffs).mean(dim=0)  # [hidden_size]

print(f"Momentum vector shape: {momentum.shape}")
print(f"Momentum magnitude: {torch.norm(momentum).item():.4f}")
print()

# Extrapolate positions using momentum
# Position N = last_prompt_position + (N - last_pos) * momentum
last_prompt_embed = prompt_embeds[0, -1]  # Last token of prompt
last_prompt_pos = prompt_length - 1

print("Extrapolating future positions using momentum...")
extrapolated_embeds = {}

for future_pos in range(prompt_length, 15):
    # How many steps beyond the prompt?
    steps_ahead = future_pos - last_prompt_pos

    # Extrapolate: move along the momentum vector
    extrapolated = last_prompt_embed + (steps_ahead * momentum)
    extrapolated_embeds[future_pos] = extrapolated.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]

    if future_pos <= 6 or future_pos >= 13:
        print(f"Position {future_pos}: {steps_ahead} steps * momentum")

print()
print(f"✓ Extrapolated positions 5-14 using trajectory momentum")
print()

# Now generate with TEMPO using momentum-extrapolated gaps
print("="*80)
print("GENERATING WITH TEMPO USING MOMENTUM-FILLED GAPS")
print("="*80)
print()

results = {
    "prompt": prompt,
    "threshold": threshold,
    "num_positions": 10,
    "method": "trajectory_momentum_gaps",
    "tokens": []
}

total_tokens = 0

for target_pos in range(5, 15):
    # Build sequence: prompt + momentum-extrapolated gaps
    seq_embeds = prompt_embeds.clone()
    seq_positions = list(range(prompt_length))

    # Fill gaps with momentum-extrapolated embeddings
    for fill_pos in range(5, target_pos):
        seq_embeds = torch.cat([seq_embeds, extrapolated_embeds[fill_pos]], dim=1)
        seq_positions.append(fill_pos)

    # Add target position (also momentum-extrapolated)
    seq_embeds = torch.cat([seq_embeds, extrapolated_embeds[target_pos]], dim=1)
    seq_positions.append(target_pos)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            inputs_embeds=seq_embeds,
            position_ids=torch.tensor([seq_positions], device=device),
            return_dict=True,
            use_cache=False
        )

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        # Apply TEMPO threshold
        viable = probs >= threshold
        viable_ids = torch.where(viable)[0]
        viable_probs = probs[viable_ids]

        # Sort by probability
        sorted_indices = torch.argsort(viable_probs, descending=True)
        viable_ids = viable_ids[sorted_indices]
        viable_probs = viable_probs[sorted_indices]

        tokens = []
        for tid, prob in zip(viable_ids, viable_probs):
            tokens.append({
                "token_id": tid.item(),
                "text": tokenizer.decode([tid.item()]),
                "probability": round(prob.item(), 4)
            })

        gap_positions = list(range(prompt_length, target_pos))

        results["tokens"].append({
            "logical_position": target_pos,
            "gaps": gap_positions,
            "num_tokens": len(tokens),
            "tokens": tokens
        })

        total_tokens += len(tokens)

        print(f"Position {target_pos} (gaps at {gap_positions if gap_positions else 'none'}):")
        print(f"  {len(tokens)} TEMPO tokens:")
        for tok in tokens[:8]:
            print(f"    '{tok['text']}' (p={tok['probability']:.4f})")
        if len(tokens) > 8:
            print(f"    ... and {len(tokens) - 8} more")
        print()

print("="*80)
print(f"Generated {len(results['tokens'])} positions with momentum-filled gaps!")
print(f"Total TEMPO tokens: {total_tokens}")
print("="*80)

# Save results
output_file = "/tmp/trajectory_momentum_gaps.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results to {output_file}")
