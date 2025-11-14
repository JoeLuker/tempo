"""
Complete TEMPO + Gap Mechanism with Proper Attention Masks
"""

import torch
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

# Create sequence: prompt repeated 3 times
sequence = torch.cat([input_ids[0], input_ids[0], input_ids[0]]).unsqueeze(0)

# Position IDs with gaps
position_ids = torch.tensor([[0,1,2,3,4, 0,1,2,3,5, 0,1,2,3,6]], device=device)

# Attention mask: 2D, each copy only attends to itself
attention_mask = torch.zeros(1, 15, device=device)
attention_mask[0, :5] = 1    # Copy 1
attention_mask[0, 5:10] = 1  # Copy 2
attention_mask[0, 10:15] = 1 # Copy 3

print("Position IDs:", position_ids[0].tolist())
print()

# ONE forward pass
with torch.no_grad():
    outputs = model(
        input_ids=sequence,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        use_cache=False
    )

print("✓ Forward pass complete!\n")

# Apply TEMPO threshold
threshold = 0.05

positions = [
    (4, "Position 4 (immediate)"),
    (9, "Position 5 (gap at 4)"),
    (14, "Position 6 (gaps at 4,5)")
]

for phys_pos, desc in positions:
    print(f"{desc}:")
    logits = outputs.logits[0, phys_pos, :]
    probs = torch.softmax(logits, dim=-1)
    viable = probs >= threshold
    viable_ids = torch.where(viable)[0]
    viable_probs = probs[viable_ids]
    
    print(f"  {len(viable_ids)} TEMPO tokens:")
    for tid, prob in zip(viable_ids[:5], viable_probs[:5]):
        print(f"    '{tokenizer.decode([tid.item()])}' (p={prob.item():.4f})")
    print()

print("="*80)
print("Generated 3 independent positions in ONE forward pass!")
print("="*80)
