"""Data capture system for mechanistic interpretability experiments."""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


class ExperimentDataCapture:
    """Captures attention weights, logits, and other data during generation."""

    def __init__(self, experiment_config: Dict, output_dir: Path):
        """Initialize data capture.

        Args:
            experiment_config: Experiment configuration dictionary
            output_dir: Directory to save captured data
        """
        self.config = experiment_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.attention_weights = []  # List of attention tensors per step
        self.logits_history = []  # List of logit tensors per step
        self.token_history = []  # List of token IDs per step
        self.position_map = {}  # Physical position -> logical position
        self.parallel_sets = []  # List of parallel token sets per step

        # Flags
        self.capture_attention = experiment_config.get('capture_attention', False)
        self.capture_logits = experiment_config.get('capture_logits', False)
        self.capture_kv_cache = experiment_config.get('capture_kv_cache', False)
        self.capture_rope_positions = experiment_config.get('capture_rope_positions', False)

    def capture_step_data(
        self,
        logical_step: int,
        physical_positions: List[int],
        token_ids: List[int],
        logits: Optional[torch.Tensor] = None,
        attention: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None
    ):
        """Capture data for a single generation step.

        Args:
            logical_step: Logical generation step
            physical_positions: Physical positions in sequence
            token_ids: Token IDs generated at this step
            logits: Full logit tensor (optional)
            attention: Attention tensor (optional)
            kv_cache: KV cache state (optional)
        """
        step_data = {
            'logical_step': logical_step,
            'physical_positions': physical_positions,
            'token_ids': token_ids,
        }

        # Capture attention weights
        if self.capture_attention and attention is not None:
            # Store as numpy for analysis
            if isinstance(attention, (tuple, list)):
                # Multiple layers - store all as a stacked array
                attn_np = torch.stack([a for a in attention]).cpu().numpy()
            else:
                attn_np = attention.cpu().numpy()

            self.attention_weights.append({
                'step': logical_step,
                'positions': physical_positions,
                'attention': attn_np,
            })

        # Capture logits
        if self.capture_logits and logits is not None:
            logits_np = logits.cpu().numpy()
            self.logits_history.append({
                'step': logical_step,
                'positions': physical_positions,
                'logits': logits_np,
            })

        # Track tokens and positions
        self.token_history.append(step_data)
        for phys_pos in physical_positions:
            self.position_map[phys_pos] = logical_step

        # Track parallel sets
        if len(token_ids) > 1:
            self.parallel_sets.append({
                'step': logical_step,
                'count': len(token_ids),
                'tokens': token_ids,
                'positions': physical_positions,
            })

    def save_attention_weights(self):
        """Save captured attention weights to file."""
        if not self.attention_weights:
            return

        output_file = self.output_dir / self.config.get('attention_output_file', 'attention_weights.npz')

        # Prepare data for saving
        save_dict = {}
        for i, step_data in enumerate(self.attention_weights):
            save_dict[f'step_{i}_attention'] = step_data['attention']
            save_dict[f'step_{i}_positions'] = np.array(step_data['positions'])
            save_dict[f'step_{i}_logical'] = step_data['step']

        # Save metadata
        save_dict['metadata'] = np.array([{
            'num_steps': len(self.attention_weights),
            'config': self.config.get('experiment_name', 'unknown'),
        }], dtype=object)

        np.savez_compressed(output_file, **save_dict)
        print(f"Saved attention weights to: {output_file}")

    def save_logits(self):
        """Save captured logits to file."""
        if not self.logits_history:
            return

        output_file = self.output_dir / self.config.get('logits_output_file', 'logits_distributions.npz')

        # Prepare data
        save_dict = {}
        for i, step_data in enumerate(self.logits_history):
            save_dict[f'step_{i}_logits'] = step_data['logits']
            save_dict[f'step_{i}_positions'] = np.array(step_data['positions'])
            save_dict[f'step_{i}_logical'] = step_data['step']

        np.savez_compressed(output_file, **save_dict)
        print(f"Saved logits to: {output_file}")

    def save_position_map(self):
        """Save RoPE position mapping."""
        if not self.capture_rope_positions:
            return

        output_file = self.output_dir / self.config.get('rope_positions_file', 'rope_positions.json')

        with open(output_file, 'w') as f:
            json.dump(self.position_map, f, indent=2)

        print(f"Saved RoPE positions to: {output_file}")

    def save_parallel_sets(self):
        """Save parallel token set information."""
        output_file = self.output_dir / 'parallel_sets.json'

        with open(output_file, 'w') as f:
            json.dump({
                'parallel_sets': self.parallel_sets,
                'summary': {
                    'total_parallel_steps': len(self.parallel_sets),
                    'max_parallel_width': max([s['count'] for s in self.parallel_sets]) if self.parallel_sets else 0,
                }
            }, f, indent=2)

        print(f"Saved parallel sets to: {output_file}")

    def save_all(self):
        """Save all captured data."""
        if self.capture_attention:
            self.save_attention_weights()

        if self.capture_logits:
            self.save_logits()

        if self.capture_rope_positions:
            self.save_position_map()

        self.save_parallel_sets()

        # Save experiment metadata
        meta_file = self.output_dir / 'experiment_metadata.json'
        with open(meta_file, 'w') as f:
            json.dump({
                'config': self.config,
                'num_steps': len(self.token_history),
                'total_tokens': sum(len(s['token_ids']) for s in self.token_history),
            }, f, indent=2)

        print(f"Experiment data saved to: {self.output_dir}")

    def get_summary(self) -> Dict:
        """Get summary of captured data.

        Returns:
            Dictionary with summary statistics
        """
        return {
            'total_steps': len(self.token_history),
            'attention_steps_captured': len(self.attention_weights),
            'logits_steps_captured': len(self.logits_history),
            'parallel_steps': len(self.parallel_sets),
            'max_parallel_width': max([s['count'] for s in self.parallel_sets]) if self.parallel_sets else 0,
        }
