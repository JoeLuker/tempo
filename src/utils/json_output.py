"""Rich JSON output formatter for TEMPO generation results.

Provides detailed, explorable JSON output for mechanistic interpretability,
including token alternatives, probabilities, branching statistics, and more.
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class TokenChoice:
    """A single token choice at a generation step."""
    token_id: int
    token_text: str
    probability: float
    logit: float
    rank: int  # 1 = highest probability
    selected: bool


@dataclass
class GenerationStep:
    """Complete information about a single generation step."""
    step: int
    position: int  # Logical position in output
    prompt_tokens_so_far: int

    # Token choices
    num_candidates: int
    selected_tokens: List[TokenChoice]
    rejected_tokens: List[TokenChoice]  # Top rejected for comparison

    # Statistics
    total_probability_mass_selected: float
    entropy: float  # Information-theoretic entropy
    branching_factor: int  # How many tokens selected this step

    # Timing
    generation_time_ms: float

    # Optional: attention patterns if captured
    attention_summary: Optional[Dict[str, Any]] = None


@dataclass
class GenerationStatistics:
    """Overall statistics for the generation."""
    total_steps: int
    total_tokens_generated: int
    total_time_seconds: float
    tokens_per_second: float

    # Branching stats
    avg_branching_factor: float
    max_branching_factor: int
    min_branching_factor: int
    total_branches_explored: int

    # Probability stats
    avg_selected_probability: float
    min_selected_probability: float
    max_selected_probability: float

    # Efficiency
    avg_entropy: float
    total_probability_mass_used: float


@dataclass
class GenerationTree:
    """Tree structure showing the exploration of token space."""
    prompt: str
    final_text: str

    # Configuration
    selection_threshold: float
    max_tokens: int
    temperature: float
    model_name: str

    # The generation timeline
    steps: List[GenerationStep]

    # Overall statistics
    statistics: GenerationStatistics

    # Metadata
    timestamp: str
    seed: Optional[int] = None


class TempoJsonFormatter:
    """Formats TEMPO generation results into rich, explorable JSON."""

    def __init__(self, indent: int = 2):
        """Initialize formatter.

        Args:
            indent: JSON indentation level (use None for compact)
        """
        self.indent = indent

    def format_generation(
        self,
        tree: GenerationTree,
        include_attention: bool = False
    ) -> str:
        """Format a complete generation as JSON.

        Args:
            tree: The generation tree to format
            include_attention: Whether to include attention data

        Returns:
            Pretty-printed JSON string
        """
        data = asdict(tree)

        # Add some computed fields for easy exploration
        data['meta'] = {
            'version': '1.0',
            'format': 'tempo-generation-tree',
            'explorable_in': [
                'jq',
                'Python json module',
                'JavaScript',
                'any JSON explorer'
            ],
            'interesting_paths': {
                'all_selected_tokens': '.steps[].selected_tokens[]',
                'branching_points': '.steps[] | select(.branching_factor > 1)',
                'high_entropy_steps': '.steps[] | select(.entropy > 2.0)',
                'low_confidence': '.steps[] | select(.selected_tokens[].probability < 0.3)',
            }
        }

        return json.dumps(data, indent=self.indent, ensure_ascii=False)

    def save_generation(
        self,
        tree: GenerationTree,
        output_path: Path,
        include_attention: bool = False
    ) -> None:
        """Save generation to JSON file.

        Args:
            tree: The generation tree
            output_path: Where to save the JSON
            include_attention: Whether to include attention data
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        json_str = self.format_generation(tree, include_attention)
        output_path.write_text(json_str, encoding='utf-8')

        print(f"\n✨ Generation saved to: {output_path}")
        print(f"📊 {tree.statistics.total_steps} steps, {tree.statistics.total_tokens_generated} tokens")
        print(f"🌳 {tree.statistics.total_branches_explored} branches explored")
        print(f"⚡ {tree.statistics.tokens_per_second:.2f} tokens/sec")
        print(f"\n💡 Explore with: jq '.steps[] | select(.branching_factor > 1)' {output_path}")

    def create_comparison_report(
        self,
        generations: List[GenerationTree],
        output_path: Path
    ) -> None:
        """Create a comparison report of multiple generations.

        Args:
            generations: List of generation trees to compare
            output_path: Where to save the comparison
        """
        comparison = {
            'meta': {
                'num_generations': len(generations),
                'format': 'tempo-comparison-report'
            },
            'generations': [asdict(g) for g in generations],
            'comparison': {
                'avg_branching_factor': sum(g.statistics.avg_branching_factor for g in generations) / len(generations),
                'avg_entropy': sum(g.statistics.avg_entropy for g in generations) / len(generations),
                'avg_tokens_per_second': sum(g.statistics.tokens_per_second for g in generations) / len(generations),
            }
        }

        output_path.write_text(
            json.dumps(comparison, indent=self.indent, ensure_ascii=False),
            encoding='utf-8'
        )

        print(f"\n📈 Comparison saved to: {output_path}")


def create_example_queries() -> Dict[str, str]:
    """Return example jq queries for exploring TEMPO JSON output."""
    return {
        "Show all branching points":
            ".steps[] | select(.branching_factor > 1) | {step, branching_factor, selected_tokens}",

        "Find high-entropy decisions":
            ".steps[] | select(.entropy > 2.0) | {step, entropy, selected_tokens}",

        "Extract just the selected text":
            ".steps[].selected_tokens[] | select(.selected) | .token_text",

        "Show probability distribution":
            ".steps[] | {step, probs: [.selected_tokens[].probability]}",

        "Find low-confidence selections":
            ".steps[] | select(.selected_tokens[].probability < 0.3)",

        "Show branching statistics":
            ".statistics | {avg_branching_factor, max_branching_factor, total_branches_explored}",

        "Compare selected vs rejected tokens":
            ".steps[] | {step, selected: .selected_tokens[0].token_text, rejected: .rejected_tokens[0].token_text}",

        "Timeline of probabilities":
            "[.steps[] | {step, prob: .total_probability_mass_selected}]",
    }


def print_example_usage():
    """Print example usage for the JSON output."""
    examples = create_example_queries()

    print("\n" + "="*60)
    print("🔍 TEMPO JSON EXPLORATION EXAMPLES")
    print("="*60)

    for description, query in examples.items():
        print(f"\n{description}:")
        print(f"  jq '{query}' output.json")

    print("\n" + "="*60)
    print("💡 TIP: Use 'jq -C' for colored output")
    print("💡 TIP: Pipe to 'less -R' for paging")
    print("="*60 + "\n")
