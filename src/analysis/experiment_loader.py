"""Experiment data loader for analysis.

Loads captured experiment data from disk into structured format.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AttentionData:
    """Container for attention weight data from an experiment."""
    steps: Dict[int, np.ndarray] = field(default_factory=dict)  # step -> attention tensor
    positions: Dict[int, List[int]] = field(default_factory=dict)  # step -> physical positions
    logical_steps: Dict[int, int] = field(default_factory=dict)  # step -> logical step
    num_layers: int = 0
    num_heads: int = 0

    def __post_init__(self):
        """Extract metadata from first step."""
        if self.steps:
            first_step = self.steps[0]
            self.num_layers = first_step.shape[0]
            if len(first_step.shape) >= 3:
                self.num_heads = first_step.shape[2]


@dataclass
class LogitsData:
    """Container for logits distribution data from an experiment."""
    steps: Dict[int, np.ndarray] = field(default_factory=dict)  # step -> logits tensor
    positions: Dict[int, List[int]] = field(default_factory=dict)  # step -> physical positions
    vocab_size: int = 0

    def __post_init__(self):
        """Extract vocab size from first step."""
        if self.steps:
            first_step = self.steps[0]
            self.vocab_size = first_step.shape[-1]


@dataclass
class ParallelSetData:
    """Container for parallel token set data."""
    sets: List[Dict] = field(default_factory=list)
    total_parallel_steps: int = 0
    max_parallel_width: int = 0


@dataclass
class ExperimentData:
    """Complete experiment data container."""
    experiment_name: str
    prompt: str
    config: Dict
    attention: Optional[AttentionData] = None
    logits: Optional[LogitsData] = None
    parallel_sets: Optional[ParallelSetData] = None
    rope_positions: Optional[Dict[int, int]] = None  # physical -> logical

    @property
    def has_attention(self) -> bool:
        """Check if experiment has attention data."""
        return self.attention is not None and len(self.attention.steps) > 0

    @property
    def has_logits(self) -> bool:
        """Check if experiment has logits data."""
        return self.logits is not None and len(self.logits.steps) > 0

    @property
    def num_steps(self) -> int:
        """Get number of generation steps."""
        if self.has_attention:
            return len(self.attention.steps)
        elif self.has_logits:
            return len(self.logits.steps)
        return 0


class ExperimentLoader:
    """Loads experiment data from disk."""

    def __init__(self, results_dir: Path = Path("experiments/results")):
        """Initialize experiment loader.

        Args:
            results_dir: Root directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        assert self.results_dir.exists(), f"Results directory not found: {results_dir}"
        logger.info(f"ExperimentLoader initialized with results_dir: {self.results_dir}")

    def load_experiment(self, experiment_name: str) -> ExperimentData:
        """Load all data for a single experiment.

        Args:
            experiment_name: Name of experiment subdirectory

        Returns:
            ExperimentData containing all loaded data
        """
        exp_dir = self.results_dir / experiment_name

        if not exp_dir.exists():
            raise ValueError(f"Experiment directory not found: {exp_dir}")

        logger.info(f"Loading experiment: {experiment_name}")

        # Load metadata
        metadata_file = exp_dir / "experiment_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        experiment_data = ExperimentData(
            experiment_name=experiment_name,
            prompt=metadata['config'].get('prompt', ''),
            config=metadata['config']
        )

        # Load attention weights if present
        attention_file = exp_dir / "attention_weights.npz"
        if attention_file.exists():
            experiment_data.attention = self._load_attention(attention_file)
            logger.info(f"Loaded attention data: {len(experiment_data.attention.steps)} steps")

        # Load logits if present
        logits_file = exp_dir / "logits_distributions.npz"
        if logits_file.exists():
            experiment_data.logits = self._load_logits(logits_file)
            logger.info(f"Loaded logits data: {len(experiment_data.logits.steps)} steps")

        # Load parallel sets
        parallel_file = exp_dir / "parallel_sets.json"
        if parallel_file.exists():
            with open(parallel_file, 'r') as f:
                parallel_data = json.load(f)
            experiment_data.parallel_sets = ParallelSetData(
                sets=parallel_data['parallel_sets'],
                total_parallel_steps=parallel_data['summary']['total_parallel_steps'],
                max_parallel_width=parallel_data['summary']['max_parallel_width']
            )
            logger.info(f"Loaded parallel sets: {experiment_data.parallel_sets.total_parallel_steps} parallel steps")

        # Load RoPE positions if present
        rope_file = exp_dir / "rope_positions.json"
        if rope_file.exists():
            with open(rope_file, 'r') as f:
                rope_data = json.load(f)
            # Convert string keys to integers
            experiment_data.rope_positions = {int(k): v for k, v in rope_data.items()}
            logger.info(f"Loaded RoPE positions: {len(experiment_data.rope_positions)} mappings")

        return experiment_data

    def _load_attention(self, attention_file: Path) -> AttentionData:
        """Load attention weights from NPZ file.

        Args:
            attention_file: Path to attention_weights.npz

        Returns:
            AttentionData container
        """
        data = np.load(attention_file, allow_pickle=True)

        attention_data = AttentionData()

        # Extract step data
        step_keys = [k for k in data.files if k.endswith('_attention')]

        for key in step_keys:
            step_num = int(key.split('_')[1])
            attention_data.steps[step_num] = data[key]

            # Get corresponding positions and logical step
            pos_key = f'step_{step_num}_positions'
            logical_key = f'step_{step_num}_logical'

            if pos_key in data.files:
                attention_data.positions[step_num] = data[pos_key].tolist()

            if logical_key in data.files:
                attention_data.logical_steps[step_num] = int(data[logical_key])

        return attention_data

    def _load_logits(self, logits_file: Path) -> LogitsData:
        """Load logits distributions from NPZ file.

        Args:
            logits_file: Path to logits_distributions.npz

        Returns:
            LogitsData container
        """
        data = np.load(logits_file, allow_pickle=True)

        logits_data = LogitsData()

        # Extract step data
        step_keys = [k for k in data.files if k.endswith('_logits')]

        for key in step_keys:
            step_num = int(key.split('_')[1])
            logits_data.steps[step_num] = data[key]

            # Get corresponding positions
            pos_key = f'step_{step_num}_positions'
            if pos_key in data.files:
                logits_data.positions[step_num] = data[pos_key].tolist()

        return logits_data

    def list_experiments(self) -> List[str]:
        """List all available experiments.

        Returns:
            List of experiment directory names
        """
        experiments = []
        for item in self.results_dir.iterdir():
            if item.is_dir() and (item / "experiment_metadata.json").exists():
                experiments.append(item.name)

        return sorted(experiments)

    def load_multiple(self, experiment_names: List[str]) -> Dict[str, ExperimentData]:
        """Load multiple experiments at once.

        Args:
            experiment_names: List of experiment names to load

        Returns:
            Dictionary mapping experiment names to ExperimentData
        """
        experiments = {}
        for name in experiment_names:
            try:
                experiments[name] = self.load_experiment(name)
            except Exception as e:
                logger.error(f"Failed to load experiment {name}: {e}")

        return experiments
