# TEMPO Algorithm Details

## Core Innovation: RoPE Modification for Parallel Processing

### The Problem
Traditional autoregressive generation processes one token at a time. Beam search maintains multiple sequences but requires separate model states. TEMPO enables processing multiple tokens at the same position within a single model state.

### The Solution: Modified Positional Embeddings

#### Standard RoPE
```python
# Traditional: each token gets unique position
position_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
tokens =      [A, B, C, D, E, F, G, H, I]
```

#### TEMPO RoPE
```python
# TEMPO: multiple tokens share logical positions
position_ids = [0, 1, 2, 2, 2, 3, 4, 4, 5]  # Physical positions
logical_pos =  [0, 1, 2, 2, 2, 3, 4, 4, 5]  # Logical positions
tokens =      [A, B, C1,C2,C3,D, E1,E2,F]   # C1,C2,C3 are parallel at position 2
```

### Implementation Details

#### 1. Position Mapping
```python
class RoPEModifier:
    def build_position_map(self, token_indices, position_offset):
        position_map = {}
        logical_pos = position_offset
        
        for physical_pos, token_idx in enumerate(token_indices):
            if token_idx == -1:  # Parallel set separator
                logical_pos += 1
            else:
                position_map[physical_pos + position_offset] = logical_pos
        
        return position_map
```

#### 2. Attention Masking
Parallel tokens at the same position cannot attend to each other (configurable):
```python
def create_attention_mask(self, parallel_sets):
    mask = torch.ones(seq_len, seq_len)
    
    for start, end in parallel_sets:
        if self.isolate_parallel_tokens:
            # Tokens in same set cannot see each other
            mask[start:end, start:end] = 0
            # Restore self-attention
            mask.fill_diagonal_(1)
    
    return mask
```

#### 3. KV Cache Management
Maintains consistency when positions are non-monotonic:
```python
def update_kv_cache(self, new_keys, new_values, position_ids):
    for i, pos in enumerate(position_ids):
        # Update cache at logical position, not physical
        cache.key_cache[:, :, pos] = new_keys[:, :, i]
        cache.value_cache[:, :, pos] = new_values[:, :, i]
```

## Retroactive Attention-Based Pruning

### Core Insight
Tokens that receive little attention from future tokens are likely incoherent continuations.

### Algorithm Steps

1. **Attention Extraction**
```python
def extract_attention_to_position(self, attention_weights, target_positions):
    # Average attention across all layers and heads
    attention = attention_weights.mean(dim=[0, 1])  # [seq_len, seq_len]
    
    # Get attention from future to past positions
    attention_to_targets = attention[target_positions:, :target_positions]
    
    return attention_to_targets.mean(dim=0)
```

2. **Pruning Decision**
```python
def should_prune_token(self, attention_score, threshold):
    # Sigmoid-based soft decision
    if self.use_sigmoid_threshold:
        pruning_prob = 1 / (1 + torch.exp(-self.sigmoid_steepness * 
                            (threshold - attention_score)))
        return pruning_prob > 0.5
    else:
        return attention_score < threshold
```

3. **Multi-Scale Attention Aggregation**
```python
def aggregate_attention_across_layers(self, layer_attentions):
    # Weight later layers more heavily
    weights = torch.linspace(0.5, 1.0, len(layer_attentions))
    weights = weights / weights.sum()
    
    weighted_attention = sum(w * attn for w, attn in 
                           zip(weights, layer_attentions))
    return weighted_attention
```

## Dynamic Threshold Adjustment

### Bezier Curve Thresholding
Adjusts pruning aggressiveness over generation length:

```python
def bezier_curve(t, p1, p2):
    # Cubic Bezier with control points (0,0), (p1,p1), (p2,p2), (1,1)
    return 3*t*(1-t)**2*p1 + 3*t**2*(1-t)*p2 + t**3

def compute_dynamic_threshold(self, step, total_steps):
    t = step / total_steps
    threshold_multiplier = bezier_curve(t, self.bezier_p1, self.bezier_p2)
    return self.base_threshold * threshold_multiplier
```

## MCTS Integration

### Node Expansion
Each node represents a partial sequence with multiple possible continuations:

```python
class MCTSNode:
    def __init__(self, tokens, logprobs, parent=None):
        self.tokens = tokens
        self.logprobs = logprobs
        self.visits = 0
        self.value = 0.0
        self.children = []
        
    def uct_score(self, c_puct):
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = c_puct * sqrt(log(self.parent.visits) / self.visits)
        return exploitation + exploration
```

### Simulation
Uses model confidence as value estimate:

```python
def simulate(self, node, depth):
    if depth == 0:
        # Use average token probability as value estimate
        return torch.exp(torch.tensor(node.logprobs)).mean().item()
    
    # Generate next tokens
    tokens = self.select_tokens_above_threshold()
    
    # Recursively evaluate
    values = []
    for token in tokens:
        child_value = self.simulate(child_node, depth - 1)
        values.append(child_value)
    
    return max(values) if values else 0.0
```

## Performance Optimizations

### 1. Batched Processing
Process all parallel tokens in single forward pass:
```python
# Instead of:
for token in parallel_tokens:
    output = model(token)

# TEMPO does:
batched_tokens = torch.stack(parallel_tokens)
outputs = model(batched_tokens)  # Single forward pass
```

### 2. Sparse Attention Patterns
Only compute attention where needed:
```python
def compute_sparse_attention(self, Q, K, V, mask):
    # Only compute attention for non-zero mask positions
    active_positions = mask.nonzero()
    sparse_scores = torch.zeros_like(mask)
    
    for i, j in active_positions:
        sparse_scores[i, j] = torch.matmul(Q[i], K[j].T)
    
    return torch.matmul(F.softmax(sparse_scores, dim=-1), V)
```

### 3. Incremental KV Cache Updates
Only update changed positions:
```python
def update_cache_incremental(self, cache, new_kv, positions):
    # Track which positions actually changed
    changed_positions = set()
    
    for pos, new_k, new_v in zip(positions, new_kv):
        if not torch.equal(cache.k[pos], new_k):
            cache.k[pos] = new_k
            cache.v[pos] = new_v
            changed_positions.add(pos)
    
    return changed_positions
```