"""Text formatting and visualization for generation output.

This module handles formatting generated text with parallel token
visualization and clean text extraction.
"""

import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from ...domain.entities.parallel_generation import LogicalPosition
from ...utils.logging_utils import LoggingMixin


@dataclass
class FormattingOptions:
    """Options for text formatting."""
    show_token_ids: bool = False
    show_probabilities: bool = False
    bracket_style: str = "[/]"  # Format: "[/]" or "{/}" or "(/)" 
    highlight_parallel: bool = True
    max_probability_decimals: int = 3


class GenerationFormatter(LoggingMixin):
    """Formats generated text with parallel token visualization."""
    
    def __init__(self, tokenizer: Any, debug_mode: bool = False):
        """Initialize the generation formatter.
        
        Args:
            tokenizer: Tokenizer for decoding tokens
            debug_mode: Whether to enable debug logging
        """
        super().__init__()
        self.setup_logging("generation_formatter", "formatter.log", debug_mode)
        self.tokenizer = tokenizer
    
    def format_using_layout(
        self,
        prompt: str,
        input_ids: List[int],
        logical_layout: List[LogicalPosition],
        prompt_length: int,
        all_original_token_sets: Dict[int, List[Tuple[int, float]]],
        options: Optional[FormattingOptions] = None
    ) -> str:
        """Format text using logical layout information.
        
        Args:
            prompt: Original prompt text
            input_ids: Full sequence of token IDs
            logical_layout: List of logical positions
            prompt_length: Length of prompt in tokens
            all_original_token_sets: Original token sets at each step
            options: Formatting options
            
        Returns:
            Formatted text with parallel token visualization
        """
        if options is None:
            options = FormattingOptions()
        
        # Start with the prompt
        formatted_parts = [prompt]
        
        # Extract bracket characters
        open_bracket = options.bracket_style[0]
        close_bracket = options.bracket_style[-1]
        separator = options.bracket_style[1:-1]
        
        # Process each logical position
        for logical_pos, start_idx, end_idx in logical_layout:
            if logical_pos == 0:  # Skip prompt
                continue
            
            # Get tokens at this position
            position_tokens = input_ids[start_idx:end_idx + 1]
            
            # Decode tokens
            decoded_tokens = []
            for token_id in position_tokens:
                decoded_text = self.tokenizer.decode([token_id])
                decoded_tokens.append((token_id, decoded_text))
            
            # Format based on number of tokens
            if len(decoded_tokens) == 1:
                # Single token - no special formatting
                formatted_parts.append(decoded_tokens[0][1])
            else:
                # Multiple tokens - use bracket notation
                token_parts = []
                
                for i, (token_id, text) in enumerate(decoded_tokens):
                    if options.show_token_ids:
                        token_part = f"{text}({token_id})"
                    else:
                        token_part = text
                    
                    # Add probability if available and requested
                    if options.show_probabilities and logical_pos - 1 in all_original_token_sets:
                        token_set = all_original_token_sets[logical_pos - 1]
                        # Find probability for this token
                        for tid, prob in token_set:
                            if tid == token_id:
                                prob_str = f"{prob:.{options.max_probability_decimals}f}"
                                token_part += f":{prob_str}"
                                break
                    
                    token_parts.append(token_part)
                
                # Join with separator
                formatted_content = separator.join(token_parts)
                formatted_parts.append(f"{open_bracket}{formatted_content}{close_bracket}")
        
        return "".join(formatted_parts)
    
    def extract_clean_text(self, formatted_text: str) -> str:
        """Extract clean text without formatting.
        
        Args:
            formatted_text: Text with parallel token formatting
            
        Returns:
            Clean text with first token from each parallel set
        """
        # Pattern to match any bracket style: [...], {...}, (...)
        # Captures content between brackets
        pattern = r'[\[\{\(]([^\]\}\)]+)[\]\}\)]'
        
        def replace_bracket(match):
            content = match.group(1)
            # Split by common separators
            tokens = re.split(r'[/|,;]', content)
            if tokens:
                # Take the first token and clean it
                first_token = tokens[0]
                # Remove token IDs if present (e.g., "word(123)" -> "word")
                first_token = re.sub(r'\(\d+\)', '', first_token)
                # Remove probabilities if present (e.g., "word:0.95" -> "word")
                first_token = re.sub(r':[0-9.]+$', '', first_token)
                return first_token.strip()
            return ''
        
        # Replace all bracket expressions with first token
        clean_text = re.sub(pattern, replace_bracket, formatted_text)
        
        # Clean up any double spaces
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        return clean_text.strip()
    
    def format_parallel_tokens(
        self,
        tokens: List[Tuple[int, str, float]],
        options: Optional[FormattingOptions] = None
    ) -> str:
        """Format a list of parallel tokens.
        
        Args:
            tokens: List of (token_id, text, probability) tuples
            options: Formatting options
            
        Returns:
            Formatted string representation
        """
        if options is None:
            options = FormattingOptions()
        
        if not tokens:
            return ""
        
        if len(tokens) == 1:
            token_id, text, prob = tokens[0]
            if options.show_token_ids and options.show_probabilities:
                return f"{text}({token_id}):{prob:.{options.max_probability_decimals}f}"
            elif options.show_token_ids:
                return f"{text}({token_id})"
            elif options.show_probabilities:
                return f"{text}:{prob:.{options.max_probability_decimals}f}"
            else:
                return text
        
        # Multiple tokens
        open_bracket = options.bracket_style[0]
        close_bracket = options.bracket_style[-1]
        separator = options.bracket_style[1:-1]
        
        formatted_tokens = []
        for token_id, text, prob in tokens:
            if options.show_token_ids and options.show_probabilities:
                formatted = f"{text}({token_id}):{prob:.{options.max_probability_decimals}f}"
            elif options.show_token_ids:
                formatted = f"{text}({token_id})"
            elif options.show_probabilities:
                formatted = f"{text}:{prob:.{options.max_probability_decimals}f}"
            else:
                formatted = text
            formatted_tokens.append(formatted)
        
        content = separator.join(formatted_tokens)
        return f"{open_bracket}{content}{close_bracket}"
    
    def format_visualization_data(
        self,
        token_sets: List[Tuple[int, Tuple[List[int], List[float]], Tuple[List[int], List[float]]]],
        tokenizer: Any
    ) -> List[Dict[str, Any]]:
        """Format token sets for visualization.
        
        Args:
            token_sets: List of (step, original_tokens, removed_tokens) tuples
            tokenizer: Tokenizer for decoding
            
        Returns:
            List of formatted visualization data
        """
        viz_data = []
        
        for step, (original_ids, original_probs), (removed_ids, removed_probs) in token_sets:
            # Decode tokens
            original_texts = [tokenizer.decode([tid]) for tid in original_ids]
            removed_texts = [tokenizer.decode([tid]) for tid in removed_ids]
            
            # Build visualization entry
            entry = {
                "step": step,
                "original_tokens": [
                    {
                        "id": tid,
                        "text": text,
                        "probability": prob
                    }
                    for tid, text, prob in zip(original_ids, original_texts, original_probs)
                ],
                "removed_tokens": [
                    {
                        "id": tid,
                        "text": text,
                        "probability": prob
                    }
                    for tid, text, prob in zip(removed_ids, removed_texts, removed_probs)
                ],
                "num_original": len(original_ids),
                "num_removed": len(removed_ids),
                "num_surviving": len(original_ids) - len(removed_ids)
            }
            
            viz_data.append(entry)
        
        return viz_data
    
    def create_tree_structure(
        self,
        logical_layout: List[LogicalPosition],
        token_sets: Dict[int, List[Tuple[int, float]]]
    ) -> Dict[str, Any]:
        """Create tree structure for visualization.
        
        Args:
            logical_layout: Logical layout of tokens
            token_sets: Token sets at each step
            
        Returns:
            Tree structure for visualization
        """
        nodes = []
        edges = []
        
        # Create root node
        root_id = "root"
        nodes.append({
            "id": root_id,
            "label": "[START]",
            "step": -1,
            "is_root": True
        })
        
        # Process each logical step
        for logical_pos, start_idx, end_idx in logical_layout:
            if logical_pos == 0:  # Skip prompt
                continue
            
            # Get tokens at this step
            if logical_pos - 1 in token_sets:
                tokens = token_sets[logical_pos - 1]
                
                for i, (token_id, prob) in enumerate(tokens):
                    node_id = f"step{logical_pos}_token{i}"
                    
                    # Decode token
                    text = self.tokenizer.decode([token_id])
                    
                    # Create node
                    nodes.append({
                        "id": node_id,
                        "label": text,
                        "token_id": token_id,
                        "probability": prob,
                        "step": logical_pos,
                        "is_root": False
                    })
                    
                    # Create edge from previous step
                    if logical_pos == 1:
                        # Connect to root
                        edges.append({
                            "source": root_id,
                            "target": node_id,
                            "weight": prob
                        })
                    else:
                        # Connect to all tokens from previous step
                        if logical_pos - 2 in token_sets:
                            prev_tokens = token_sets[logical_pos - 2]
                            for j, _ in enumerate(prev_tokens):
                                prev_node_id = f"step{logical_pos-1}_token{j}"
                                edges.append({
                                    "source": prev_node_id,
                                    "target": node_id,
                                    "weight": prob
                                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "num_steps": len(logical_layout) - 1,  # Exclude prompt
            "total_nodes": len(nodes),
            "total_edges": len(edges)
        }
