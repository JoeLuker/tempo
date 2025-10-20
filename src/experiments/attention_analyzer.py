"""Attention pattern analysis for mechanistic interpretability.

This module provides tools to capture, analyze, and visualize attention patterns
during TEMPO generation, especially focusing on how parallel tokens attend to
each other.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


class AttentionAnalyzer:
    """Analyzes attention patterns during parallel token generation."""

    def __init__(self, tokenizer, debug_mode: bool = False):
        """Initialize the attention analyzer.

        Args:
            tokenizer: Tokenizer for decoding token IDs
            debug_mode: Enable debug logging
        """
        self.tokenizer = tokenizer
        self.debug_mode = debug_mode
        self.attention_history = []
        self.token_history = []
        self.position_map = {}

    def capture_attention(
        self,
        attention_weights: torch.Tensor,
        token_ids: List[int],
        logical_step: int,
        physical_positions: List[int]
    ):
        """Capture attention weights for a generation step.

        Args:
            attention_weights: Attention tensor [batch, heads, seq_len, seq_len]
            token_ids: Token IDs at current positions
            logical_step: Current logical generation step
            physical_positions: Physical positions in sequence
        """
        if attention_weights is None:
            return

        # Average across heads for simplicity
        if attention_weights.dim() == 4:
            # [batch, heads, seq_len, seq_len] -> [seq_len, seq_len]
            avg_attention = attention_weights[0].mean(dim=0).cpu().numpy()
        else:
            avg_attention = attention_weights.cpu().numpy()

        # Decode tokens
        decoded_tokens = []
        for tid in token_ids:
            token_text = self.tokenizer.decode_tokens([tid])
            decoded_tokens.append(token_text[0] if isinstance(token_text, list) else token_text)

        # Store attention data
        self.attention_history.append({
            'logical_step': logical_step,
            'physical_positions': physical_positions,
            'token_ids': token_ids,
            'tokens': decoded_tokens,
            'attention_matrix': avg_attention,
        })

        # Track position mapping
        for phys_pos in physical_positions:
            self.position_map[phys_pos] = logical_step

    def analyze_parallel_attention(self, logical_step: int) -> Dict:
        """Analyze how parallel tokens at a given step attend to previous tokens.

        Args:
            logical_step: The logical step to analyze

        Returns:
            Dictionary with attention statistics for parallel tokens
        """
        # Find attention data for this step
        step_data = None
        for data in self.attention_history:
            if data['logical_step'] == logical_step:
                step_data = data
                break

        if step_data is None:
            return {'error': f'No data for logical step {logical_step}'}

        positions = step_data['physical_positions']
        tokens = step_data['tokens']
        attention = step_data['attention_matrix']

        # Analyze attention from each parallel token to all previous tokens
        parallel_attention = {}
        for i, (pos, token) in enumerate(zip(positions, tokens)):
            # Get attention from this position to all previous positions
            attn_scores = attention[pos, :pos]

            # Find top attended positions
            top_k = min(5, len(attn_scores))
            top_indices = np.argsort(attn_scores)[-top_k:][::-1]

            parallel_attention[f"{token} (pos {pos})"] = {
                'attention_to_previous': attn_scores.tolist(),
                'top_attended_positions': top_indices.tolist(),
                'top_attention_scores': attn_scores[top_indices].tolist(),
            }

        return {
            'logical_step': logical_step,
            'parallel_tokens': tokens,
            'attention_patterns': parallel_attention,
        }

    def analyze_cross_parallel_attention(self, logical_step: int) -> Dict:
        """Analyze how parallel tokens attend to each other.

        Args:
            logical_step: The logical step to analyze

        Returns:
            Attention matrix between parallel tokens
        """
        step_data = None
        for data in self.attention_history:
            if data['logical_step'] == logical_step:
                step_data = data
                break

        if step_data is None:
            return {'error': f'No data for logical step {logical_step}'}

        positions = step_data['physical_positions']
        tokens = step_data['tokens']
        attention = step_data['attention_matrix']

        # Extract sub-matrix of attention between parallel tokens
        if len(positions) > 1:
            # Get attention from parallel positions to parallel positions
            cross_attn = np.zeros((len(positions), len(positions)))
            for i, pos_i in enumerate(positions):
                for j, pos_j in enumerate(positions):
                    cross_attn[i, j] = attention[pos_i, pos_j]

            return {
                'logical_step': logical_step,
                'tokens': tokens,
                'cross_attention_matrix': cross_attn.tolist(),
                'should_be_isolated': True,  # TEMPO isolates parallel tokens
                'max_cross_attention': float(np.max(cross_attn)),
            }
        else:
            return {
                'logical_step': logical_step,
                'tokens': tokens,
                'note': 'Only one token at this step',
            }

    def export_to_json(self, output_path: Path):
        """Export all captured attention data to JSON.

        Args:
            output_path: Path to save JSON file
        """
        export_data = {
            'position_map': self.position_map,
            'steps': []
        }

        for data in self.attention_history:
            # Convert numpy arrays to lists for JSON serialization
            step_export = {
                'logical_step': data['logical_step'],
                'physical_positions': data['physical_positions'],
                'token_ids': data['token_ids'],
                'tokens': data['tokens'],
                'attention_matrix': data['attention_matrix'].tolist(),
            }
            export_data['steps'].append(step_export)

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

    def get_attention_summary(self) -> Dict:
        """Get summary statistics of all attention patterns.

        Returns:
            Dictionary with summary statistics
        """
        if not self.attention_history:
            return {'error': 'No attention data captured'}

        total_steps = len(self.attention_history)
        parallel_steps = sum(
            1 for d in self.attention_history if len(d['physical_positions']) > 1
        )

        max_parallel_width = max(
            len(d['physical_positions']) for d in self.attention_history
        )

        return {
            'total_steps': total_steps,
            'parallel_steps': parallel_steps,
            'max_parallel_width': max_parallel_width,
            'average_parallel_width': np.mean([
                len(d['physical_positions']) for d in self.attention_history
            ]),
        }

    def reset(self):
        """Clear all captured attention data."""
        self.attention_history = []
        self.token_history = []
        self.position_map = {}
