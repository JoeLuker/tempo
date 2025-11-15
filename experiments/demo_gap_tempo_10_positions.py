"""
TEMPO + Gap Mechanism: Generate 10 future positions in ONE forward pass
Uses very low threshold (0.02) to capture many token alternatives
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

prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
prompt_length = input_ids.shape[1]

print(f"\nPrompt: '{prompt}' (length {prompt_length})\n")

# Number of future positions to generate
num_positions = 10
threshold = 0.02

# Create sequence: prompt repeated num_positions times (NO extra tokens)
sequence = torch.cat([input_ids[0]] * num_positions).unsqueeze(0)

# Position IDs with gaps
# Copy 1: [0,1,2,3,4] -> predicts position 5
# Copy 2: [0,1,2,3,5] -> predicts position 6 (gap at 5)
# Copy 3: [0,1,2,3,6] -> predicts position 7 (gaps at 5,6)
# etc.
position_list = []
for i in range(num_positions):
    # Each copy: [0, 1, 2, 3, 4+i]
    copy_positions = list(range(prompt_length - 1)) + [prompt_length - 1 + i]
    position_list.extend(copy_positions)

position_ids = torch.tensor([position_list], device=device)

# Attention mask: each copy only attends to itself
total_length = prompt_length * num_positions
attention_mask = torch.zeros(1, total_length, device=device)

for i in range(num_positions):
    start = i * prompt_length
    end = start + prompt_length
    attention_mask[0, start:end] = 1

print(f"Generating {num_positions} positions in ONE forward pass")
print(f"TEMPO threshold: {threshold}")
print(f"Total sequence length: {total_length}")
print(f"Position IDs (first 3 copies): {position_list[:15]}")
print()

# ONE forward pass
print("Running forward pass...")
with torch.no_grad():
    outputs = model(
        input_ids=sequence,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        use_cache=False
    )

print("✓ Forward pass complete!\n")
print("="*80)

# Collect results
results = {
    "prompt": prompt,
    "threshold": threshold,
    "num_positions": num_positions,
    "tokens": []
}

# Extract TEMPO tokens for each position
for i in range(num_positions):
    # Physical position: last token of each copy
    phys_pos = (i + 1) * prompt_length - 1
    # Logical position: what position we're predicting
    logical_pos = prompt_length + i

    logits = outputs.logits[0, phys_pos, :]
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

    gap_positions = list(range(prompt_length, logical_pos))

    results["tokens"].append({
        "logical_position": logical_pos,
        "gaps": gap_positions,
        "num_tokens": len(tokens),
        "tokens": tokens
    })

    print(f"Position {logical_pos} (gaps at {gap_positions if gap_positions else 'none'}):")
    print(f"  {len(tokens)} TEMPO tokens:")
    for tok in tokens[:5]:  # Show top 5
        print(f"    '{tok['text']}' (p={tok['probability']:.4f})")
    if len(tokens) > 5:
        print(f"    ... and {len(tokens) - 5} more")
    print()

print("="*80)
print(f"Generated {num_positions} positions with TEMPO in ONE forward pass!")
print(f"Total TEMPO tokens: {sum(len(p['tokens']) for p in results['tokens'])}")
print("="*80)

# Save to JSON
output_file = "/tmp/gap_tempo_10_positions.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results to {output_file}")
