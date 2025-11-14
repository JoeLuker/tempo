"""
TEMPO with Superposition Continuation

Generate TEMPO tokens in superposition at position N, then generate position N+1
that attends to ALL the superposition tokens simultaneously.

Uses RoPE + attention mask hacks from the library.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.rope.embedding_modifier import modify_positions_for_parallel_tokens

model_name = "deepcogito/cogito-v1-preview-llama-3B"
device = "mps"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

prompt = "The answer to whether a hotdog is a sandwich or not is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
prompt_length = input_ids.shape[1]

print(f"\nPrompt: '{prompt}'\n")

# Step 1: Get TEMPO tokens
threshold = 0.05

with torch.no_grad():
    outputs = model(input_ids, return_dict=True, use_cache=False)
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    viable = probs >= threshold
    viable_ids = torch.where(viable)[0]
    viable_probs = probs[viable_ids]

print(f"TEMPO threshold: {threshold}")
print(f"Found {len(viable_ids)} viable tokens:")
for tid, prob in zip(viable_ids, viable_probs):
    print(f"  '{tokenizer.decode([tid.item()])}' (p={prob.item():.4f})")
print()

# Step 2: Create sequence with TEMPO tokens in superposition + next token
# Structure: [prompt, tok1, tok2, ..., tokN, next_tok]
# Position:  [0...N-1, N,   N,   ..., N,    N+1]

# We'll add a placeholder for the next token (we'll generate logits for it)
# Use BOS token as placeholder
next_token_placeholder = tokenizer.bos_token_id

# Build sequence
tempo_tokens = viable_ids.tolist()
sequence = input_ids[0].tolist() + tempo_tokens + [next_token_placeholder]
sequence_tensor = torch.tensor(sequence, device=device).unsqueeze(0)

# Build position map
position_map = {}
# Prompt: normal positions
for i in range(prompt_length):
    position_map[i] = i

# TEMPO tokens: all at same logical position (prompt_length)
for i in range(len(tempo_tokens)):
    physical_pos = prompt_length + i
    position_map[physical_pos] = prompt_length

# Next token: at logical position prompt_length + 1
next_token_physical_pos = prompt_length + len(tempo_tokens)
position_map[next_token_physical_pos] = prompt_length + 1

# Create position IDs
physical_positions = torch.arange(len(sequence), device=device)
logical_positions = modify_positions_for_parallel_tokens(
    physical_positions.unsqueeze(0),
    position_map,
    torch.device(device)
)

print("Sequence structure:")
print(f"  Prompt: positions 0-{prompt_length-1}")
print(f"  TEMPO tokens: {len(tempo_tokens)} tokens at logical position {prompt_length}")
print(f"  Next token: logical position {prompt_length + 1}")
print()
print(f"Position mapping:")
print(f"  Physical: {list(range(len(sequence)))}")
print(f"  Logical:  {logical_positions[0].tolist()}")
print()

# Create attention mask - DON'T isolate parallel tokens!
# Let them all see each other so the next token can aggregate
seq_len = len(sequence)
attention_mask = torch.ones(1, seq_len, device=device)

print("NOTE: Parallel tokens are NOT isolated - they can see each other")
print("This allows the next token to aggregate information from all of them\n")
print("Running forward pass...")
with torch.no_grad():
    outputs = model(
        input_ids=sequence_tensor,
        attention_mask=attention_mask,
        position_ids=logical_positions,
        return_dict=True,
        use_cache=False
    )

print("✓ Complete!\n")

# Get logits for the next token position
next_token_logits = outputs.logits[0, next_token_physical_pos, :]
next_token_probs = torch.softmax(next_token_logits, dim=-1)
top_next = torch.topk(next_token_probs, k=5)

print(f"{'='*80}")
print("NEXT TOKEN PREDICTIONS (attending to superposition)")
print(f"{'='*80}")
print()
print("The next token sees ALL these TEMPO tokens simultaneously:")
for tid, prob in zip(viable_ids, viable_probs):
    print(f"  '{tokenizer.decode([tid.item()])}' (p={prob.item():.4f})")
print()
print("And predicts:")
for tid, prob in zip(top_next.indices, top_next.values):
    print(f"  '{tokenizer.decode([tid.item()])}' (p={prob.item():.4f})")
print()

# Continue the sequence
best_next = top_next.indices[0].item()
best_next_text = tokenizer.decode([best_next])

print(f"{'='*80}")
print(f"Full continuation:")
print(f"  Prompt: '{prompt}'")
print(f"  Superposition: {[tokenizer.decode([t]) for t in tempo_tokens]}")
print(f"  Next: '{best_next_text}'")
print(f"{'='*80}")
