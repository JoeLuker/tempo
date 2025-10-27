"""Visualization tools for comparing isolation modes."""

from typing import Dict, List, Tuple
import difflib


class IsolationVisualizer:
    """Visualizes differences between isolated and visible modes."""

    @staticmethod
    def show_text_diff(isolated_text: str, visible_text: str) -> str:
        """Show character-level diff between texts.

        Args:
            isolated_text: Text from isolated mode
            visible_text: Text from visible mode

        Returns:
            Formatted diff visualization
        """
        lines = []
        lines.append("TEXT DIFF (Isolated vs Visible):")
        lines.append("=" * 70)

        # Split into words for better readability
        isolated_words = isolated_text.split()
        visible_words = visible_text.split()

        diff = difflib.unified_diff(
            isolated_words,
            visible_words,
            lineterm='',
            fromfile='isolated',
            tofile='visible'
        )

        for line in diff:
            lines.append(line)

        if len(lines) == 2:
            lines.append("No differences found")

        return "\n".join(lines)

    @staticmethod
    def show_token_divergence(isolated_tokens: List[str], visible_tokens: List[str]) -> str:
        """Show where token sequences diverge.

        Args:
            isolated_tokens: Tokens from isolated mode
            visible_tokens: Tokens from visible mode

        Returns:
            Formatted divergence visualization
        """
        lines = []
        lines.append("TOKEN DIVERGENCE ANALYSIS:")
        lines.append("=" * 70)

        # Find first divergence point
        divergence_idx = None
        min_len = min(len(isolated_tokens), len(visible_tokens))

        for i in range(min_len):
            if isolated_tokens[i] != visible_tokens[i]:
                divergence_idx = i
                break

        if divergence_idx is None and len(isolated_tokens) == len(visible_tokens):
            lines.append("✓ Token sequences are IDENTICAL")
            return "\n".join(lines)

        # Show context around divergence
        if divergence_idx is not None:
            lines.append(f"First divergence at position {divergence_idx}:")
            lines.append("")

            # Show 3 tokens before and after divergence
            start = max(0, divergence_idx - 3)
            end = min(min_len, divergence_idx + 4)

            lines.append("ISOLATED: " + " ".join(isolated_tokens[start:end]))
            lines.append("VISIBLE:  " + " ".join(visible_tokens[start:end]))
            lines.append(" " * 10 + "^" * 10 + " (divergence)")
        else:
            lines.append(f"Sequences match up to position {min_len}")
            lines.append(f"Length difference: Isolated={len(isolated_tokens)}, Visible={len(visible_tokens)}")

        lines.append("")
        lines.append(f"Total tokens - Isolated: {len(isolated_tokens)}, Visible: {len(visible_tokens)}")

        return "\n".join(lines)

    @staticmethod
    def show_parallel_token_influence(
        isolated_parallel_sets: Dict[int, List[Tuple[str, float]]],
        visible_parallel_sets: Dict[int, List[Tuple[str, float]]]
    ) -> str:
        """Show how parallel token visibility affects token selection.

        Args:
            isolated_parallel_sets: Parallel token sets from isolated mode
            visible_parallel_sets: Parallel token sets from visible mode

        Returns:
            Formatted analysis
        """
        lines = []
        lines.append("PARALLEL TOKEN INFLUENCE ANALYSIS:")
        lines.append("=" * 70)

        all_steps = sorted(set(isolated_parallel_sets.keys()) | set(visible_parallel_sets.keys()))

        for step in all_steps[:10]:  # Show first 10 steps
            lines.append(f"\nStep {step}:")

            iso_tokens = isolated_parallel_sets.get(step, [])
            vis_tokens = visible_parallel_sets.get(step, [])

            if iso_tokens:
                iso_str = ", ".join([f"{tok}({prob:.3f})" for tok, prob in iso_tokens[:5]])
                lines.append(f"  Isolated: [{iso_str}]")

            if vis_tokens:
                vis_str = ", ".join([f"{tok}({prob:.3f})" for tok, prob in vis_tokens[:5]])
                lines.append(f"  Visible:  [{vis_str}]")

            # Check if same tokens selected
            if iso_tokens and vis_tokens:
                iso_set = {tok for tok, _ in iso_tokens}
                vis_set = {tok for tok, _ in vis_tokens}

                if iso_set == vis_set:
                    lines.append("  ✓ Same tokens selected")
                else:
                    only_iso = iso_set - vis_set
                    only_vis = vis_set - iso_set

                    if only_iso:
                        lines.append(f"  → Only in isolated: {only_iso}")
                    if only_vis:
                        lines.append(f"  → Only in visible: {only_vis}")

        return "\n".join(lines)

    @staticmethod
    def create_side_by_side(isolated_text: str, visible_text: str, width: int = 60) -> str:
        """Create side-by-side comparison.

        Args:
            isolated_text: Text from isolated mode
            visible_text: Text from visible mode
            width: Width of each column

        Returns:
            Side-by-side formatted text
        """
        lines = []
        lines.append("SIDE-BY-SIDE COMPARISON:")
        lines.append("=" * (width * 2 + 5))

        header = f"{'ISOLATED':<{width}} | {'VISIBLE':<{width}}"
        lines.append(header)
        lines.append("-" * (width * 2 + 5))

        # Split into lines of approximately equal width
        iso_lines = IsolationVisualizer._wrap_text(isolated_text, width)
        vis_lines = IsolationVisualizer._wrap_text(visible_text, width)

        max_lines = max(len(iso_lines), len(vis_lines))

        for i in range(max_lines):
            iso_line = iso_lines[i] if i < len(iso_lines) else ""
            vis_line = vis_lines[i] if i < len(vis_lines) else ""

            # Highlight differences
            if iso_line != vis_line:
                marker = " ⚠"
            else:
                marker = "  "

            lines.append(f"{iso_line:<{width}} |{marker} {vis_line:<{width}}")

        return "\n".join(lines)

    @staticmethod
    def _wrap_text(text: str, width: int) -> List[str]:
        """Wrap text to specified width.

        Args:
            text: Text to wrap
            width: Maximum line width

        Returns:
            List of wrapped lines
        """
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space

            if current_length + word_length > width:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_length = word_length
                else:
                    # Single word longer than width
                    lines.append(word[:width])
                    current_line = []
                    current_length = 0
            else:
                current_line.append(word)
                current_length += word_length

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    @staticmethod
    def summarize_differences(results: Dict) -> str:
        """Create executive summary of differences.

        Args:
            results: Comparison results dictionary

        Returns:
            Summary string
        """
        lines = []
        lines.append("EXECUTIVE SUMMARY:")
        lines.append("=" * 70)

        comp = results.get('comparison', {})

        # Key findings
        if comp.get('texts_identical'):
            lines.append("✓ IDENTICAL: Both modes produced identical output")
            lines.append("  → Parallel token visibility had no effect on this prompt")
        else:
            lines.append("⚠ DIVERGENT: Modes produced different outputs")
            lines.append("  → Parallel token visibility influenced generation")

        lines.append("")

        # Performance
        time_diff = comp.get('time_difference', 0)
        if abs(time_diff) < 0.1:
            lines.append("✓ PERFORMANCE: Similar generation time")
        elif time_diff > 0:
            lines.append(f"⚠ PERFORMANCE: Visible mode {time_diff:.2f}s slower")
        else:
            lines.append(f"✓ PERFORMANCE: Visible mode {-time_diff:.2f}s faster")

        lines.append("")

        # Length difference
        len_diff = comp.get('visible_length', 0) - comp.get('isolated_length', 0)
        if len_diff == 0:
            lines.append("✓ LENGTH: Same output length")
        else:
            lines.append(f"  LENGTH: Visible mode {len_diff:+d} chars")

        lines.append("")
        lines.append("HYPOTHESIS:")
        if comp.get('texts_identical'):
            lines.append("  The model's top token selections were strong enough that")
            lines.append("  parallel token visibility didn't change the outcome.")
        else:
            lines.append("  Allowing parallel tokens to see each other likely caused")
            lines.append("  different token probabilities, leading to divergence.")

        return "\n".join(lines)
