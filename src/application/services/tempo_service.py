"""
TEMPO Service - Clean programmatic API for TEMPO generation.

This service provides a facade over the complex TEMPO architecture,
hiding initialization details and providing a simple generate() method.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class TEMPOService:
    """
    Service layer for TEMPO text generation.

    Handles model loading, initialization, and provides clean API.
    Uses singleton pattern for model to avoid reloading.
    """

    _instance = None
    _model = None
    _tokenizer = None
    _device = None

    def __new__(cls):
        """Singleton pattern - only one instance with loaded model."""
        if cls._instance is None:
            cls._instance = super(TEMPOService, cls).__new__(cls)
        return cls._instance

    def _ensure_initialized(self, model_name: str = "deepcogito/cogito-v1-preview-llama-3B"):
        """Lazy initialization of model and tokenizer."""
        if self._model is None:
            logger.info(f"Loading TEMPO model: {model_name}")

            # Import here to avoid circular dependencies
            from src.utils.model_utils import load_model_and_tokenizer
            from src.modeling.model_wrapper import TEMPOModelWrapper

            # Load model and tokenizer
            model, tokenizer, device = load_model_and_tokenizer(
                model_name=model_name,
                device="mps"  # Auto-detect in model_utils
            )

            # Wrap in TEMPO wrapper
            self._model = TEMPOModelWrapper(model)
            self._tokenizer = tokenizer
            self._device = device

            logger.info(f"Model loaded successfully on {device}")

    def generate(
        self,
        prompt: str,
        selection_threshold: float = 0.25,
        max_tokens: int = 20,
        isolate: bool = False,
        seed: int = 42,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using TEMPO.

        Args:
            prompt: Input text prompt
            selection_threshold: Probability threshold for token selection
            max_tokens: Maximum tokens to generate
            isolate: Whether to isolate parallel tokens
            seed: Random seed for reproducibility
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with generation results including:
            - generated_text: Formatted text with parallel tokens shown
            - raw_generated_text: Plain text output
            - generation_time: Time taken
            - Other metadata
        """
        # Ensure model is loaded
        model_name = kwargs.get('model', 'deepcogito/cogito-v1-preview-llama-3B')
        self._ensure_initialized(model_name)

        # Import ExperimentRunner
        from src.experiments.experiment_runner import ExperimentRunner

        # Create runner with loaded model/tokenizer
        runner = ExperimentRunner(
            model=self._model,
            tokenizer=self._tokenizer,
            device=self._device,
            skip_wrapping=True  # Already wrapped
        )

        # Build args dict
        args = {
            'prompt': prompt,
            'max_tokens': max_tokens,
            'selection_threshold': selection_threshold,
            'isolate': isolate,
            'seed': seed,
            'output_dir': './temp_output',
            'model': model_name,
            'use_custom_rope': True,
            'debug_mode': False,
            'use_retroactive_removal': False,
            'disable_kv_cache': False,
            'enable_thinking': False,
            'use_mcts': False,
            **kwargs  # Allow overrides
        }

        # Run generation
        try:
            result = runner.run_experiment(args)

            # Clean up temp output
            import shutil
            output_path = Path(args['output_dir'])
            if output_path.exists():
                shutil.rmtree(output_path, ignore_errors=True)

            return result

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise

    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            logger.info("Unloading TEMPO model")
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None

            # Force garbage collection
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
