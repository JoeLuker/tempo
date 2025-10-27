"""Visualization tools for TEMPO attention patterns."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class AttentionVisualizer:
    """Creates text-based visualizations of attention patterns."""

    @staticmethod
    def visualize_attention_matrix(
        attention_matrix: np.ndarray,
        source_tokens: List[str],
        target_tokens: List[str],
        width: int = 60
    ) -> str:
        """Create ASCII heatmap of attention matrix.

        Args:
            attention_matrix: Attention weights [source, target]
            source_tokens: Source token labels
            target_tokens: Target token labels
            width: Character width for visualization

        Returns:
            ASCII art heatmap
        """
        chars = " .:-=+*#%@"

        lines = []
        lines.append("Attention Matrix (source→target):")
        lines.append("")

        # Normalize to 0-1 range
        attn_min = attention_matrix.min()
        attn_max = attention_matrix.max()
        if attn_max > attn_min:
            normalized = (attention_matrix - attn_min) / (attn_max - attn_min)
        else:
            normalized = attention_matrix

        # Header
        header = " " * 20 + "".join([f"{t[:5]:>6}" for t in target_tokens])
        lines.append(header)

        # Rows
        for i, src_token in enumerate(source_tokens):
            row = f"{src_token[:18]:>18}: "
            for j in range(len(target_tokens)):
                val = normalized[i, j]
                char_idx = int(val * (len(chars) - 1))
                row += f"  {chars[char_idx]}   "
            lines.append(row)

        lines.append("")
        lines.append(f"Legend: {chars[0]}=low  {chars[-1]}=high")

        return "\n".join(lines)

    @staticmethod
    def visualize_parallel_attention(analysis_result: Dict) -> str:
        """Visualize how parallel tokens attend to context.

        Args:
            analysis_result: Result from AttentionAnalyzer.analyze_parallel_attention

        Returns:
            Formatted text visualization
        """
        if 'error' in analysis_result:
            return f"Error: {analysis_result['error']}"

        lines = []
        lines.append(f"Parallel Token Attention (Step {analysis_result['logical_step']}):")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Tokens: {analysis_result['parallel_tokens']}")
        lines.append("")

        for token_key, attn_data in analysis_result['attention_patterns'].items():
            lines.append(f"Token: {token_key}")
            lines.append("  Top attended positions:")

            for pos, score in zip(
                attn_data['top_attended_positions'],
                attn_data['top_attention_scores']
            ):
                bar_length = int(score * 40)
                bar = "█" * bar_length
                lines.append(f"    Position {pos:3d}: {bar} {score:.4f}")

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def visualize_cross_parallel(analysis_result: Dict) -> str:
        """Visualize cross-parallel-token attention.

        Args:
            analysis_result: Result from AttentionAnalyzer.analyze_cross_parallel_attention

        Returns:
            Formatted text visualization
        """
        if 'error' in analysis_result or 'note' in analysis_result:
            return str(analysis_result)

        lines = []
        lines.append(f"Cross-Parallel Attention (Step {analysis_result['logical_step']}):")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Tokens: {analysis_result['tokens']}")
        lines.append(f"Should be isolated: {analysis_result['should_be_isolated']}")
        lines.append(f"Max cross-attention: {analysis_result['max_cross_attention']:.6f}")
        lines.append("")

        matrix = np.array(analysis_result['cross_attention_matrix'])
        tokens = analysis_result['tokens']

        # Show matrix
        vis = AttentionVisualizer.visualize_attention_matrix(
            matrix, tokens, tokens
        )
        lines.append(vis)

        # Interpretation
        lines.append("")
        if analysis_result['max_cross_attention'] < 0.01:
            lines.append("✓ Parallel tokens are properly isolated (attention < 0.01)")
        else:
            lines.append("⚠ Parallel tokens may be attending to each other")

        return "\n".join(lines)

    @staticmethod
    def export_attention_heatmap_data(
        attention_data_path: Path,
        output_path: Path
    ):
        """Export attention data in format suitable for external plotting.

        Args:
            attention_data_path: Path to JSON attention data
            output_path: Path to save processed data
        """
        with open(attention_data_path, 'r') as f:
            data = json.load(f)

        # Process into plottable format
        export = {
            'steps': [],
            'summary': {
                'total_steps': len(data['steps']),
                'position_map': data['position_map'],
            }
        }

        for step in data['steps']:
            step_data = {
                'logical_step': step['logical_step'],
                'tokens': step['tokens'],
                'attention_matrix': step['attention_matrix'],
                'positions': step['physical_positions'],
            }
            export['steps'].append(step_data)

        with open(output_path, 'w') as f:
            json.dump(export, f, indent=2)

        print(f"Exported heatmap data to: {output_path}")
