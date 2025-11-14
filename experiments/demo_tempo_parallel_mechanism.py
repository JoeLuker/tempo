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

prompt = "The answer to whether a hotdog is a sandwich or not is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
prompt_length = input_ids.shape[1]

print(f"\nPrompt: '{prompt}' (length {prompt_length})\n")

# Normal TEMPO: just the prompt, no gaps
sequence = input_ids

# Position IDs: normal consecutive positions
position_ids = torch.arange(prompt_length, device=device).unsqueeze(0)

# Attention mask: standard causal
attention_mask = torch.ones(1, prompt_length, device=device)

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

# Apply TEMPO threshold to get next tokens
threshold = 0.05

# Get predictions after the prompt
logits = outputs.logits[0, -1, :]
probs = torch.softmax(logits, dim=-1)
viable = probs >= threshold
viable_ids = torch.where(viable)[0]
viable_probs = probs[viable_ids]

print(f"TEMPO threshold: {threshold}")
print(f"Found {len(viable_ids)} viable next tokens:")
for tid, prob in zip(viable_ids, viable_probs):
    print(f"  '{tokenizer.decode([tid.item()])}' (p={prob.item():.4f})")
print()

# Now continue sequentially with each TEMPO token
print("="*80)
print("SEQUENTIAL CONTINUATION WITH TEMPO TOKENS")
print("="*80)
print()

max_tokens_to_show = min(3, len(viable_ids))
for i in range(max_tokens_to_show):
    token_id = viable_ids[i]
    token_text = tokenizer.decode([token_id.item()])
    token_prob = viable_probs[i].item()

    # Continue this path
    print(f"Path {i+1}: '{token_text}' (p={token_prob:.4f})")

    current_sequence = torch.cat([input_ids[0], token_id.unsqueeze(0)])

    # Generate next 3 tokens sequentially
    for step in range(3):
        seq_input = current_sequence.unsqueeze(0)
        pos_ids = torch.arange(len(current_sequence), device=device).unsqueeze(0)

        with torch.no_grad():
            out = model(input_ids=seq_input, position_ids=pos_ids, return_dict=True, use_cache=False)

        next_logits = out.logits[0, -1, :]
        next_probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.argmax(next_probs)

        current_sequence = torch.cat([current_sequence, next_token.unsqueeze(0)])

    full_text = tokenizer.decode(current_sequence[prompt_length:])
    print(f"  → {full_text}")
    print()

print("="*80)
