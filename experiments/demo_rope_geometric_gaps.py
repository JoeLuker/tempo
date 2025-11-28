"""
RoPE Geometric Gap Filling

Instead of dummy tokens or position ID gaps, compute the geometric trajectory
that RoPE creates in embedding space and fill gaps with embeddings positioned
along the natural geodesic of the positional manifold.
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

model_name = "deepcogito/cogito-v1-preview-llama-3B"
device = "mps"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
prompt_length = input_ids.shape[1]

print(f"\nPrompt: '{prompt}' (length {prompt_length})")
print()

# Get model config
config = model.config
hidden_size = config.hidden_size
num_heads = config.num_attention_heads
head_dim = hidden_size // num_heads

print(f"Hidden size: {hidden_size}")
print(f"Num heads: {num_heads}")
print(f"Head dim: {head_dim}")
print()

# RoPE parameters
rope_theta = getattr(config, 'rope_theta', 10000.0)

print(f"RoPE theta: {rope_theta}")
print()

# Compute RoPE frequencies
def compute_rope_freqs(dim, theta=10000.0):
    """Compute RoPE frequency for each dimension pair."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    return freqs

freqs = compute_rope_freqs(head_dim, rope_theta).to(device)
print(f"RoPE frequencies (first 5): {freqs[:5].tolist()}")
print()

# Compute RoPE embedding for a position
def compute_rope_embedding(position, freqs):
    """Compute the RoPE rotational embedding for a given position.

    Returns cos and sin components for applying rotation.
    """
    t = position * freqs  # [dim/2]
    # For each pair of dimensions, we get cos and sin
    cos = torch.cos(t)
    sin = torch.sin(t)
    return cos, sin

# Compute geometric trajectory
print("="*80)
print("COMPUTING GEOMETRIC TRAJECTORY")
print("="*80)
print()

positions = list(range(15))  # 0 to 14
trajectory = []

for pos in positions:
    cos, sin = compute_rope_embedding(pos, freqs)
    # The "position" in embedding space is defined by these rotation angles
    trajectory.append({
        "position": pos,
        "cos": cos.cpu().tolist(),
        "sin": sin.cpu().tolist(),
        "angles": (pos * freqs).cpu().tolist()[:5]  # First 5 angles
    })

    if pos <= 5 or pos >= 13:
        print(f"Position {pos}:")
        print(f"  Angles (first 5): {trajectory[-1]['angles']}")

print()
print(f"Computed trajectory for {len(trajectory)} positions")
print()

# Now create a sequence with geometric gap filling
print("="*80)
print("METHOD: GEOMETRIC GAP FILLING")
print("="*80)
print()

num_copies = 10
threshold = 0.02

# Get embeddings for the prompt
with torch.no_grad():
    # First get the input embeddings
    embed_layer = model.get_input_embeddings()
    prompt_embeds = embed_layer(input_ids)  # [1, prompt_length, hidden_size]

print(f"Prompt embeddings shape: {prompt_embeds.shape}")
print()

# Build sequences with geometric gap filling
sequences = []
position_ids_list = []
attention_masks = []

for copy_idx in range(num_copies):
    target_position = prompt_length + copy_idx  # Positions 5-14

    # Start with prompt embeddings and positions
    seq_embeds = prompt_embeds.clone()
    seq_positions = list(range(prompt_length))

    # Fill gaps with PURE GEOMETRIC embeddings
    # Zero semantic content - only positional information from RoPE rotations
    num_gaps = copy_idx
    for gap_idx in range(num_gaps):
        gap_position = prompt_length + gap_idx

        # Pure geometric placeholder: ZERO vector
        # RoPE will apply rotations to position it geometrically in the manifold
        # No semantic content whatsoever - pure positional structure
        placeholder = torch.zeros(1, 1, hidden_size, device=device)

        seq_embeds = torch.cat([seq_embeds, placeholder], dim=1)
        seq_positions.append(gap_position)

    # Add final position (also zero - predicting from pure geometry)
    final_placeholder = torch.zeros(1, 1, hidden_size, device=device)
    seq_embeds = torch.cat([seq_embeds, final_placeholder], dim=1)
    seq_positions.append(target_position)

    sequences.append(seq_embeds)
    position_ids_list.append(seq_positions)

    # Attention mask: all positions visible
    attn_mask = torch.ones(1, len(seq_positions), device=device)
    attention_masks.append(attn_mask)

print(f"Copy 0 (position 5): {len(position_ids_list[0])} tokens")
print(f"Copy 9 (position 14): {len(position_ids_list[9])} tokens")
print()

# Concatenate all sequences for batch processing
max_len = max(len(p) for p in position_ids_list)

# Pad sequences
batch_embeds = []
batch_positions = []
batch_masks = []

for embeds, positions, mask in zip(sequences, position_ids_list, attention_masks):
    seq_len = embeds.shape[1]
    if seq_len < max_len:
        # Pad embeddings
        padding = torch.zeros(1, max_len - seq_len, hidden_size, device=device)
        embeds = torch.cat([embeds, padding], dim=1)
        # Pad positions (use last position)
        positions = positions + [positions[-1]] * (max_len - len(positions))
        # Pad mask with zeros
        mask_padding = torch.zeros(1, max_len - mask.shape[1], device=device)
        mask = torch.cat([mask, mask_padding], dim=1)

    batch_embeds.append(embeds)
    batch_positions.append(positions)
    batch_masks.append(mask)

# Stack into batch
batch_embeds = torch.cat(batch_embeds, dim=0)  # [num_copies, max_len, hidden_size]
batch_positions = torch.tensor(batch_positions, device=device)  # [num_copies, max_len]
batch_masks = torch.cat(batch_masks, dim=0)  # [num_copies, max_len]

print(f"Batch shape: {batch_embeds.shape}")
print(f"Position IDs shape: {batch_positions.shape}")
print(f"Attention mask shape: {batch_masks.shape}")
print()

print("Running forward pass with geometric gap filling...")
with torch.no_grad():
    # Pass embeddings directly instead of token IDs
    outputs = model(
        inputs_embeds=batch_embeds,
        attention_mask=batch_masks,
        position_ids=batch_positions,
        return_dict=True,
        use_cache=False
    )

print("✓ Complete\n")
print("="*80)

# Extract predictions
results = {
    "prompt": prompt,
    "threshold": threshold,
    "num_positions": num_copies,
    "method": "geometric_gap_filling",
    "tokens": []
}

total_tokens = 0
for copy_idx in range(num_copies):
    target_position = prompt_length + copy_idx
    seq_len = len(position_ids_list[copy_idx])

    # Last token position in this sequence
    phys_pos = seq_len - 1

    logits = outputs.logits[copy_idx, phys_pos, :]
    probs = torch.softmax(logits, dim=-1)
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

    gap_positions = list(range(prompt_length, target_position))

    results["tokens"].append({
        "logical_position": target_position,
        "gaps": gap_positions,
        "num_tokens": len(tokens),
        "tokens": tokens
    })

    total_tokens += len(tokens)

    print(f"Position {target_position} (gaps at {gap_positions if gap_positions else 'none'}):")
    print(f"  {len(tokens)} TEMPO tokens:")
    for tok in tokens[:8]:
        print(f"    '{tok['text']}' (p={tok['probability']:.4f})")
    if len(tokens) > 8:
        print(f"    ... and {len(tokens) - 8} more")
    print()

print("="*80)
print(f"Generated {num_copies} positions with geometric gap filling!")
print(f"Total TEMPO tokens: {total_tokens}")
print("="*80)

# Save results
output_file = "/tmp/rope_geometric_gaps.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results to {output_file}")
