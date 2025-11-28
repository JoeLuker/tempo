"""
Expected Embedding Gap Filling

Fill gaps with the EXPECTED embedding - weighted average of all possible tokens
by their probabilities. This gives a heuristic "where the model thinks it's going"
without committing to specific tokens.
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
vocab_size = embed_layer.num_embeddings

prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
prompt_length = input_ids.shape[1]

print(f"\nPrompt: '{prompt}' (length {prompt_length})")
print(f"Vocab size: {vocab_size}")
print()

threshold = 0.02

print("="*80)
print("EXPECTED EMBEDDING GAP FILLING")
print("="*80)
print()

# First, get predictions for position 5 to compute expected embeddings for gaps
print("Computing expected embeddings for future positions...")

# Get prompt embeddings
with torch.no_grad():
    prompt_embeds = embed_layer(input_ids)  # [1, prompt_length, hidden_size]

# Compute logits for position 5 (first position after prompt)
with torch.no_grad():
    outputs = model(
        inputs_embeds=prompt_embeds,
        position_ids=torch.arange(prompt_length, device=device).unsqueeze(0),
        return_dict=True,
        use_cache=False
    )

    logits_5 = outputs.logits[0, -1, :]  # Logits for position 5
    probs_5 = torch.softmax(logits_5, dim=-1)  # [vocab_size]

# Compute EXPECTED embedding for position 5
# This is the weighted average of all token embeddings by their probability
all_embeddings = embed_layer.weight  # [vocab_size, hidden_size]
expected_embed_5 = torch.matmul(probs_5.unsqueeze(0), all_embeddings)  # [1, hidden_size]
expected_embed_5 = expected_embed_5.unsqueeze(0)  # [1, 1, hidden_size]

print(f"✓ Computed expected embedding for position 5")
print(f"  Top 5 tokens: {[(tokenizer.decode([i]), probs_5[i].item()) for i in torch.topk(probs_5, 5).indices.tolist()]}")
print()

# Now recursively compute expected embeddings for positions 6-14
expected_embeddings = {5: expected_embed_5}

for pos in range(6, 15):
    # Build sequence: prompt + expected embeddings for positions 5 to pos-1
    seq_embeds = prompt_embeds.clone()
    seq_positions = list(range(prompt_length))

    for fill_pos in range(5, pos):
        seq_embeds = torch.cat([seq_embeds, expected_embeddings[fill_pos]], dim=1)
        seq_positions.append(fill_pos)

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

        # Compute expected embedding for this position
        expected_embed = torch.matmul(probs.unsqueeze(0), all_embeddings).unsqueeze(0)
        expected_embeddings[pos] = expected_embed

        if pos <= 7 or pos >= 13:
            top_tokens = [(tokenizer.decode([i]), probs[i].item()) for i in torch.topk(probs, 3).indices.tolist()]
            print(f"Position {pos}: top tokens {top_tokens}")

print()
print(f"✓ Computed expected embeddings for positions 5-14")
print()

# Now use these expected embeddings to generate with TEMPO at each position
print("="*80)
print("GENERATING WITH TEMPO USING EXPECTED EMBEDDINGS")
print("="*80)
print()

results = {
    "prompt": prompt,
    "threshold": threshold,
    "num_positions": 10,
    "method": "expected_embedding_gaps",
    "tokens": []
}

total_tokens = 0

for target_pos in range(5, 15):
    # Build sequence with expected embeddings filling gaps
    seq_embeds = prompt_embeds.clone()
    seq_positions = list(range(prompt_length))

    # Fill gaps with expected embeddings
    for fill_pos in range(5, target_pos):
        seq_embeds = torch.cat([seq_embeds, expected_embeddings[fill_pos]], dim=1)
        seq_positions.append(fill_pos)

    # Add the target position (use expected embedding as placeholder)
    seq_embeds = torch.cat([seq_embeds, expected_embeddings[target_pos]], dim=1)
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
print(f"Generated {len(results['tokens'])} positions with expected embedding gaps!")
print(f"Total TEMPO tokens: {total_tokens}")
print("="*80)

# Save results
output_file = "/tmp/expected_embedding_gaps.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results to {output_file}")
