"""Custom attention implementation that exposes attention weights.

MLX's mx.fast.scaled_dot_product_attention is fused and doesn't expose
attention weights. For Hebbian consolidation, we need access to these
weights to compute importance scores.

This module provides a manual attention implementation that returns both
the attention output and the attention weights.
"""

import mlx.core as mx


def attention_with_weights(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: mx.array | None = None,
    scale: float | None = None,
) -> tuple[mx.array, mx.array]:
    """Compute scaled dot-product attention with attention weight output.

    Args:
        q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, n_kv_heads, seq_len, head_dim)
        v: Value tensor of shape (batch, n_kv_heads, seq_len, head_dim)
        mask: Optional additive attention mask (negative values for masking)
        scale: Optional scale factor (defaults to 1/sqrt(head_dim))

    Returns:
        output: Attention output of shape (batch, n_heads, seq_len, head_dim)
        attn_weights: Attention weights of shape (batch, n_heads, seq_len, seq_len)
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    # Compute attention scores: Q @ K^T
    # (batch, n_heads, seq_q, head_dim) @ (batch, n_heads, head_dim, seq_k)
    # -> (batch, n_heads, seq_q, seq_k)
    scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale

    # Apply mask if provided (additive mask with large negative for masked positions)
    if mask is not None:
        scores = scores + mask

    # Softmax to get attention weights
    attn_weights = mx.softmax(scores, axis=-1)

    # Apply attention weights to values
    # (batch, n_heads, seq_q, seq_k) @ (batch, n_heads, seq_k, head_dim)
    # -> (batch, n_heads, seq_q, head_dim)
    output = mx.matmul(attn_weights, v)

    return output, attn_weights


def create_causal_mask(seq_len: int, offset: int = 0) -> mx.array:
    """Create an additive causal attention mask.

    Args:
        seq_len: Length of the sequence
        offset: Position offset for KV cache

    Returns:
        mask: Additive mask of shape (seq_len, offset + seq_len) with
              -inf for positions that should not attend
    """
    # Row indices: what we're attending FROM (query positions)
    row_indices = mx.arange(offset, offset + seq_len)

    # Column indices: what we're attending TO (key positions)
    col_indices = mx.arange(offset + seq_len)

    # Causal: can only attend to positions <= current position
    # Shape: (seq_len, offset + seq_len)
    mask = row_indices[:, None] < col_indices[None, :]

    # Convert to additive mask: False -> 0, True -> -inf
    return mx.where(mask, mx.array(-1e9), mx.array(0.0))
