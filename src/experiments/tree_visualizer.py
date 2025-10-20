"""Parallel token tree visualization for TEMPO generation.

Visualizes the branching structure of parallel token generation as a tree.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


class TokenTreeNode:
    """Node in the parallel token tree."""

    def __init__(self, token: str, token_id: int, probability: float, logical_step: int, physical_position: int):
        """Initialize a tree node.

        Args:
            token: Token text
            token_id: Token ID
            probability: Token probability
            logical_step: Logical generation step
            physical_position: Physical position in sequence
        """
        self.token = token
        self.token_id = token_id
        self.probability = probability
        self.logical_step = logical_step
        self.physical_position = physical_position
        self.children = []
        self.pruned = False
        self.attention_score = None

    def add_child(self, child: 'TokenTreeNode'):
        """Add a child node."""
        self.children.append(child)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            'token': self.token,
            'token_id': self.token_id,
            'probability': self.probability,
            'logical_step': self.logical_step,
            'physical_position': self.physical_position,
            'pruned': self.pruned,
            'attention_score': self.attention_score,
            'children': [child.to_dict() for child in self.children],
        }


class TokenTreeVisualizer:
    """Visualizes parallel token generation as a tree structure."""

    def __init__(self, tokenizer):
        """Initialize the tree visualizer.

        Args:
            tokenizer: Tokenizer for decoding tokens
        """
        self.tokenizer = tokenizer
        self.root = None
        self.all_nodes = []

    def build_tree(self, generation_result: Dict):
        """Build tree from generation result.

        Args:
            generation_result: Result containing token sets and layout
        """
        # TODO: Parse generation result and build tree structure
        # For now, this is a framework
        pass

    def visualize_ascii(self, max_depth: Optional[int] = None) -> str:
        """Create ASCII art tree visualization.

        Args:
            max_depth: Maximum depth to display (None for all)

        Returns:
            ASCII tree representation
        """
        if self.root is None:
            return "No tree built"

        lines = []
        self._render_node(self.root, "", True, lines, 0, max_depth)
        return "\n".join(lines)

    def _render_node(
        self,
        node: TokenTreeNode,
        prefix: str,
        is_last: bool,
        lines: List[str],
        depth: int,
        max_depth: Optional[int]
    ):
        """Recursively render a node and its children.

        Args:
            node: Current node
            prefix: Prefix for this line
            is_last: Whether this is the last child
            lines: Output lines list
            depth: Current depth
            max_depth: Maximum depth to render
        """
        if max_depth is not None and depth > max_depth:
            return

        # Current node
        connector = "└── " if is_last else "├── "
        token_repr = f"{node.token} (p={node.probability:.3f})"
        if node.pruned:
            token_repr += " [PRUNED]"

        lines.append(prefix + connector + token_repr)

        # Children
        if node.children:
            extension = "    " if is_last else "│   "
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                self._render_node(
                    child,
                    prefix + extension,
                    is_last_child,
                    lines,
                    depth + 1,
                    max_depth
                )

    def get_tree_stats(self) -> Dict:
        """Get statistics about the tree.

        Returns:
            Dictionary with tree statistics
        """
        if not self.all_nodes:
            return {'error': 'No tree built'}

        total_nodes = len(self.all_nodes)
        pruned_nodes = sum(1 for node in self.all_nodes if node.pruned)
        leaf_nodes = sum(1 for node in self.all_nodes if not node.children)

        depths = []
        for node in self.all_nodes:
            depths.append(node.logical_step)

        return {
            'total_nodes': total_nodes,
            'pruned_nodes': pruned_nodes,
            'active_nodes': total_nodes - pruned_nodes,
            'leaf_nodes': leaf_nodes,
            'max_depth': max(depths) if depths else 0,
            'branching_factor_avg': total_nodes / (max(depths) + 1) if depths else 0,
        }

    def export_to_json(self, output_path: Path):
        """Export tree structure to JSON.

        Args:
            output_path: Path to save JSON file
        """
        if self.root is None:
            return

        export_data = {
            'tree': self.root.to_dict(),
            'stats': self.get_tree_stats(),
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

    def export_for_d3(self, output_path: Path):
        """Export in D3.js-compatible format for web visualization.

        Args:
            output_path: Path to save JSON file
        """
        if self.root is None:
            return

        # D3 hierarchical format
        d3_data = {
            'name': 'root',
            'children': self._node_to_d3(self.root)
        }

        with open(output_path, 'w') as f:
            json.dump(d3_data, f, indent=2)

    def _node_to_d3(self, node: TokenTreeNode) -> List[Dict]:
        """Convert node to D3 format.

        Args:
            node: Node to convert

        Returns:
            List of D3-compatible child nodes
        """
        d3_children = []
        for child in node.children:
            d3_child = {
                'name': child.token,
                'value': child.probability,
                'pruned': child.pruned,
            }
            if child.children:
                d3_child['children'] = self._node_to_d3(child)
            d3_children.append(d3_child)

        return d3_children

    def find_paths_to_leaves(self) -> List[List[TokenTreeNode]]:
        """Find all paths from root to leaf nodes.

        Returns:
            List of paths (each path is list of nodes)
        """
        if self.root is None:
            return []

        all_paths = []
        self._find_paths_recursive(self.root, [], all_paths)
        return all_paths

    def _find_paths_recursive(
        self,
        node: TokenTreeNode,
        current_path: List[TokenTreeNode],
        all_paths: List[List[TokenTreeNode]]
    ):
        """Recursively find paths to leaves.

        Args:
            node: Current node
            current_path: Current path being built
            all_paths: All paths found so far
        """
        current_path = current_path + [node]

        if not node.children:
            # Leaf node
            all_paths.append(current_path)
        else:
            for child in node.children:
                self._find_paths_recursive(child, current_path, all_paths)

    def visualize_paths(self, max_paths: int = 10) -> str:
        """Visualize generation paths through the tree.

        Args:
            max_paths: Maximum number of paths to show

        Returns:
            Formatted path visualization
        """
        paths = self.find_paths_to_leaves()[:max_paths]

        lines = []
        lines.append(f"Generation Paths (showing {len(paths)} of {len(self.find_paths_to_leaves())}):")
        lines.append("=" * 60)

        for i, path in enumerate(paths, 1):
            tokens = " → ".join([node.token for node in path])
            avg_prob = sum(node.probability for node in path) / len(path)
            has_pruned = any(node.pruned for node in path)

            status = " [CONTAINS PRUNED]" if has_pruned else ""
            lines.append(f"\nPath {i} (avg prob: {avg_prob:.3f}){status}:")
            lines.append(f"  {tokens}")

        return "\n".join(lines)
