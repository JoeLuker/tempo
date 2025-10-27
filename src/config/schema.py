"""Configuration schema for TEMPO runs.

Defines the structure for YAML config files that control TEMPO generation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "deepcogito/cogito-v1-preview-llama-3B"
    revision: str = "main"
    device: Optional[str] = None  # Auto-detect if None


@dataclass
class GenerationConfig:
    """Core generation parameters."""
    prompt: str
    max_tokens: int = 150
    selection_threshold: float = 0.12
    min_steps: int = 0
    seed: int = 42


@dataclass
class PhaseConfig:
    """Configuration for a single generation phase."""
    name: str
    max_positions: int
    threshold: float
    description: str = ""


@dataclass
class MultiPhaseConfig:
    """Multi-phase generation configuration."""
    enabled: bool = False
    phases: List[PhaseConfig] = field(default_factory=list)


@dataclass
class ExtensionConfig:
    """Extension system configuration."""
    # Built-in extensions
    confidence_surfing: bool = False
    genealogy_tracking: bool = False
    entropy_watching: bool = False

    # Two-phase
    two_phase: bool = False
    dynamic_phase: bool = False
    phase1_steps: int = 25
    max_positions: int = 100
    phase2_threshold: float = 1.0

    # Custom extension pipeline (advanced)
    custom_pipeline: Optional[str] = None  # Python code to eval


@dataclass
class PruningConfig:
    """Retroactive pruning configuration."""
    enabled: bool = False
    attention_threshold: float = 0.01


@dataclass
class OutputConfig:
    """Output configuration."""
    json_output: bool = True
    json_file: Optional[str] = None  # Auto-generate if None
    include_attention: bool = False
    include_logits: bool = False
    save_checkpoints: bool = False
    checkpoint_every: int = 50


@dataclass
class DebugConfig:
    """Debug and profiling options."""
    debug_mode: bool = False
    show_token_ids: bool = False
    profile: bool = False
    verbose: bool = False


@dataclass
class TEMPOConfig:
    """Complete TEMPO run configuration."""
    name: str  # Descriptive name for this run
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=lambda: GenerationConfig(prompt=""))
    multi_phase: MultiPhaseConfig = field(default_factory=MultiPhaseConfig)
    extensions: ExtensionConfig = field(default_factory=ExtensionConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TEMPOConfig':
        """Load config from YAML file.

        Args:
            yaml_path: Path to YAML config file

        Returns:
            TEMPOConfig instance
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TEMPOConfig':
        """Create config from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            TEMPOConfig instance
        """
        # Extract nested configs
        model = ModelConfig(**data.get('model', {}))
        generation = GenerationConfig(**data.get('generation', {'prompt': ''}))

        # Multi-phase
        multi_phase_data = data.get('multi_phase', {})
        phases = [
            PhaseConfig(**p) for p in multi_phase_data.get('phases', [])
        ]
        multi_phase = MultiPhaseConfig(
            enabled=multi_phase_data.get('enabled', False),
            phases=phases
        )

        extensions = ExtensionConfig(**data.get('extensions', {}))
        pruning = PruningConfig(**data.get('pruning', {}))
        output = OutputConfig(**data.get('output', {}))
        debug = DebugConfig(**data.get('debug', {}))

        return cls(
            name=data['name'],
            model=model,
            generation=generation,
            multi_phase=multi_phase,
            extensions=extensions,
            pruning=pruning,
            output=output,
            debug=debug,
            description=data.get('description', ''),
            tags=data.get('tags', [])
        )

    def to_args_dict(self) -> Dict[str, Any]:
        """Convert to ArgumentParser-compatible dict.

        Returns:
            Dictionary compatible with TEMPO's argument parser
        """
        args = {
            # Model
            'model': self.model.name,

            # Generation
            'prompt': self.generation.prompt,
            'max_tokens': self.generation.max_tokens,
            'selection_threshold': self.generation.selection_threshold,
            'min_steps': self.generation.min_steps,
            'seed': self.generation.seed,

            # Extensions
            'enable_extensions': (
                self.extensions.confidence_surfing or
                self.extensions.genealogy_tracking or
                self.extensions.entropy_watching
            ),
            'two_phase': self.extensions.two_phase,
            'dynamic_phase': self.extensions.dynamic_phase,
            'phase1_steps': self.extensions.phase1_steps,
            'max_positions': self.extensions.max_positions,
            'phase2_threshold': self.extensions.phase2_threshold,

            # Pruning
            'use_retroactive_pruning': self.pruning.enabled,
            'attention_threshold': self.pruning.attention_threshold,

            # Output
            'output_json': self.output.json_output,
            'json_output_file': self.output.json_file,

            # Debug
            'debug_mode': self.debug.debug_mode,
            'show_token_ids': self.debug.show_token_ids,
            'profile': self.debug.profile,

            # Defaults
            'use_retroactive_removal': self.pruning.enabled,
            'disable_kv_cache': False,
            'isolate_parallel_tokens': True,
            'default_mode': False,
            'use_custom_rope': True,
        }

        return args

    def to_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file.

        Args:
            yaml_path: Path to save YAML file
        """
        data = {
            'name': self.name,
            'description': self.description,
            'tags': self.tags,
            'model': {
                'name': self.model.name,
                'revision': self.model.revision,
            },
            'generation': {
                'prompt': self.generation.prompt,
                'max_tokens': self.generation.max_tokens,
                'selection_threshold': self.generation.selection_threshold,
                'min_steps': self.generation.min_steps,
                'seed': self.generation.seed,
            },
            'extensions': {
                'confidence_surfing': self.extensions.confidence_surfing,
                'genealogy_tracking': self.extensions.genealogy_tracking,
                'entropy_watching': self.extensions.entropy_watching,
                'two_phase': self.extensions.two_phase,
                'dynamic_phase': self.extensions.dynamic_phase,
                'phase1_steps': self.extensions.phase1_steps,
                'max_positions': self.extensions.max_positions,
                'phase2_threshold': self.extensions.phase2_threshold,
            },
            'pruning': {
                'enabled': self.pruning.enabled,
                'attention_threshold': self.pruning.attention_threshold,
            },
            'output': {
                'json_output': self.output.json_output,
                'json_file': self.output.json_file,
            },
            'debug': {
                'debug_mode': self.debug.debug_mode,
                'show_token_ids': self.debug.show_token_ids,
                'profile': self.debug.profile,
            },
        }

        if self.multi_phase.enabled:
            data['multi_phase'] = {
                'enabled': True,
                'phases': [
                    {
                        'name': p.name,
                        'max_positions': p.max_positions,
                        'threshold': p.threshold,
                        'description': p.description,
                    }
                    for p in self.multi_phase.phases
                ]
            }

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
