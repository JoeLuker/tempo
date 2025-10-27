"""Pruning decision analysis for mechanistic interpretability.

Analyzes why certain tokens are pruned during retroactive removal.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


class PruningAnalyzer:
    """Analyzes pruning decisions during TEMPO generation."""

    def __init__(self, tokenizer, debug_mode: bool = False):
        """Initialize the pruning analyzer.

        Args:
            tokenizer: Tokenizer for decoding token IDs
            debug_mode: Enable debug logging
        """
        self.tokenizer = tokenizer
        self.debug_mode = debug_mode
        self.pruning_history = []

    def capture_pruning_decision(
        self,
        logical_step: int,
        original_tokens: List[Tuple[int, float]],
        pruned_tokens: List[Tuple[int, float]],
        attention_scores: Optional[np.ndarray] = None,
        pruning_threshold: float = 0.0,
        pruning_reason: str = "unknown"
    ):
        """Capture a pruning decision for analysis.

        Args:
            logical_step: Logical generation step
            original_tokens: Original token set [(token_id, prob), ...]
            pruned_tokens: Remaining tokens after pruning
            attention_scores: Attention scores used for pruning
            pruning_threshold: Threshold used for pruning
            pruning_reason: Reason for pruning (attention, relative, etc.)
        """
        # Decode tokens
        original_decoded = []
        for tid, prob in original_tokens:
            text = self.tokenizer.decode_tokens([tid])
            token_text = text[0] if isinstance(text, list) else text
            original_decoded.append((token_text, tid, prob))

        pruned_decoded = []
        for tid, prob in pruned_tokens:
            text = self.tokenizer.decode_tokens([tid])
            token_text = text[0] if isinstance(text, list) else text
            pruned_decoded.append((token_text, tid, prob))

        # Identify removed tokens
        pruned_ids = {tid for _, tid, _ in pruned_decoded}
        removed_tokens = [
            (text, tid, prob)
            for text, tid, prob in original_decoded
            if tid not in pruned_ids
        ]

        # Store decision
        decision = {
            'logical_step': logical_step,
            'original_count': len(original_tokens),
            'pruned_count': len(pruned_tokens),
            'removed_count': len(removed_tokens),
            'original_tokens': original_decoded,
            'pruned_tokens': pruned_decoded,
            'removed_tokens': removed_tokens,
            'pruning_threshold': pruning_threshold,
            'pruning_reason': pruning_reason,
        }

        if attention_scores is not None:
            decision['attention_scores'] = attention_scores.tolist() if isinstance(attention_scores, np.ndarray) else attention_scores

        self.pruning_history.append(decision)

    def analyze_pruning_pattern(self) -> Dict:
        """Analyze overall pruning patterns across generation.

        Returns:
            Dictionary with pruning statistics
        """
        if not self.pruning_history:
            return {'error': 'No pruning data captured'}

        total_removed = sum(d['removed_count'] for d in self.pruning_history)
        total_original = sum(d['original_count'] for d in self.pruning_history)

        pruning_rates = [
            d['removed_count'] / d['original_count'] if d['original_count'] > 0 else 0
            for d in self.pruning_history
        ]

        return {
            'total_pruning_events': len(self.pruning_history),
            'total_tokens_removed': total_removed,
            'total_tokens_processed': total_original,
            'overall_pruning_rate': total_removed / total_original if total_original > 0 else 0,
            'average_pruning_rate_per_step': np.mean(pruning_rates),
            'max_pruning_rate': np.max(pruning_rates) if pruning_rates else 0,
            'min_pruning_rate': np.min(pruning_rates) if pruning_rates else 0,
        }

    def analyze_removed_tokens(self) -> Dict:
        """Analyze characteristics of removed tokens.

        Returns:
            Analysis of what kinds of tokens get pruned
        """
        if not self.pruning_history:
            return {'error': 'No pruning data captured'}

        all_removed = []
        for decision in self.pruning_history:
            all_removed.extend(decision['removed_tokens'])

        # Analyze by probability
        removed_probs = [prob for _, _, prob in all_removed]

        return {
            'total_removed_tokens': len(all_removed),
            'removed_token_probability_stats': {
                'mean': float(np.mean(removed_probs)) if removed_probs else 0,
                'median': float(np.median(removed_probs)) if removed_probs else 0,
                'min': float(np.min(removed_probs)) if removed_probs else 0,
                'max': float(np.max(removed_probs)) if removed_probs else 0,
                'std': float(np.std(removed_probs)) if removed_probs else 0,
            },
            'most_removed_tokens': self._get_most_removed_tokens(all_removed),
        }

    def _get_most_removed_tokens(self, removed_tokens: List[Tuple[str, int, float]], top_k: int = 10) -> List[Dict]:
        """Get most frequently removed tokens.

        Args:
            removed_tokens: List of (text, token_id, prob) tuples
            top_k: Number of top tokens to return

        Returns:
            List of token statistics
        """
        token_counts = {}
        for text, tid, prob in removed_tokens:
            if text not in token_counts:
                token_counts[text] = {'count': 0, 'total_prob': 0.0, 'token_id': tid}
            token_counts[text]['count'] += 1
            token_counts[text]['total_prob'] += prob

        # Sort by count
        sorted_tokens = sorted(
            token_counts.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:top_k]

        return [
            {
                'token': token,
                'removal_count': stats['count'],
                'average_probability': stats['total_prob'] / stats['count'],
                'token_id': stats['token_id'],
            }
            for token, stats in sorted_tokens
        ]

    def get_step_details(self, logical_step: int) -> Dict:
        """Get detailed pruning information for a specific step.

        Args:
            logical_step: The logical step to analyze

        Returns:
            Detailed pruning decision data
        """
        for decision in self.pruning_history:
            if decision['logical_step'] == logical_step:
                return decision

        return {'error': f'No pruning data for step {logical_step}'}

    def export_to_json(self, output_path: Path):
        """Export all pruning decisions to JSON.

        Args:
            output_path: Path to save JSON file
        """
        export_data = {
            'pruning_history': self.pruning_history,
            'summary': self.analyze_pruning_pattern(),
            'removed_token_analysis': self.analyze_removed_tokens(),
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

    def reset(self):
        """Clear all captured pruning data."""
        self.pruning_history = []
