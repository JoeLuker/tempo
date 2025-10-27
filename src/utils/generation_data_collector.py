"""Collects detailed generation data for JSON output and analysis."""

import time
import math
from typing import List, Dict, Any, Optional
from datetime import datetime

from .json_output import (
    TokenChoice,
    GenerationStep,
    GenerationStatistics,
    GenerationTree,
)


class GenerationDataCollector:
    """Collects detailed data during TEMPO generation for rich JSON output."""

    def __init__(self, prompt: str, config: Dict[str, Any], model_name: str, seed: Optional[int] = None):
        """Initialize the collector.

        Args:
            prompt: The input prompt
            config: Generation configuration
            model_name: Name of the model being used
            seed: Random seed if set
        """
        self.prompt = prompt
        self.config = config
        self.model_name = model_name
        self.seed = seed

        self.steps: List[GenerationStep] = []
        self.start_time = time.time()

        # Track statistics as we go
        self.total_branches = 0
        self.branching_factors: List[int] = []
        self.selected_probabilities: List[float] = []
        self.entropies: List[float] = []

    def add_step(
        self,
        step_num: int,
        position: int,
        prompt_tokens_so_far: int,
        selected_tokens: List[tuple],  # [(token_id, text, prob, logit), ...]
        rejected_tokens: List[tuple],  # Top N rejected for comparison
        generation_time_ms: float,
        attention_summary: Optional[Dict[str, Any]] = None
    ):
        """Add a generation step.

        Args:
            step_num: The logical step number
            position: Logical position in output
            prompt_tokens_so_far: Number of prompt tokens processed
            selected_tokens: List of (token_id, text, probability, logit) tuples
            rejected_tokens: List of top rejected tokens for comparison
            generation_time_ms: Time taken for this step
            attention_summary: Optional attention pattern summary
        """
        # Calculate entropy from probabilities
        probs = [t[2] for t in selected_tokens + rejected_tokens]
        entropy = self._calculate_entropy(probs)

        # Create TokenChoice objects
        selected_choices = [
            TokenChoice(
                token_id=t[0],
                token_text=t[1],
                probability=t[2],
                logit=t[3],
                rank=i + 1,
                selected=True
            )
            for i, t in enumerate(selected_tokens)
        ]

        rejected_choices = [
            TokenChoice(
                token_id=t[0],
                token_text=t[1],
                probability=t[2],
                logit=t[3],
                rank=len(selected_tokens) + i + 1,
                selected=False
            )
            for i, t in enumerate(rejected_tokens)
        ]

        # Calculate statistics
        total_prob_selected = sum(t[2] for t in selected_tokens)
        branching_factor = len(selected_tokens)

        # Create step
        step = GenerationStep(
            step=step_num,
            position=position,
            prompt_tokens_so_far=prompt_tokens_so_far,
            num_candidates=len(selected_tokens) + len(rejected_tokens),
            selected_tokens=selected_choices,
            rejected_tokens=rejected_choices,
            total_probability_mass_selected=total_prob_selected,
            entropy=entropy,
            branching_factor=branching_factor,
            generation_time_ms=generation_time_ms,
            attention_summary=attention_summary
        )

        self.steps.append(step)

        # Update running statistics
        self.total_branches += branching_factor
        self.branching_factors.append(branching_factor)
        self.selected_probabilities.extend([t[2] for t in selected_tokens])
        self.entropies.append(entropy)

    def finalize(self, final_text: str) -> GenerationTree:
        """Create the final generation tree with statistics.

        Args:
            final_text: The complete generated text

        Returns:
            Complete GenerationTree ready for JSON output
        """
        total_time = time.time() - self.start_time

        # Calculate statistics
        stats = GenerationStatistics(
            total_steps=len(self.steps),
            total_tokens_generated=self.total_branches,
            total_time_seconds=total_time,
            tokens_per_second=self.total_branches / total_time if total_time > 0 else 0,
            avg_branching_factor=sum(self.branching_factors) / len(self.branching_factors) if self.branching_factors else 0,
            max_branching_factor=max(self.branching_factors) if self.branching_factors else 0,
            min_branching_factor=min(self.branching_factors) if self.branching_factors else 0,
            total_branches_explored=self.total_branches,
            avg_selected_probability=sum(self.selected_probabilities) / len(self.selected_probabilities) if self.selected_probabilities else 0,
            min_selected_probability=min(self.selected_probabilities) if self.selected_probabilities else 0,
            max_selected_probability=max(self.selected_probabilities) if self.selected_probabilities else 1.0,
            avg_entropy=sum(self.entropies) / len(self.entropies) if self.entropies else 0,
            total_probability_mass_used=sum(self.selected_probabilities)
        )

        tree = GenerationTree(
            prompt=self.prompt,
            final_text=final_text,
            selection_threshold=self.config.get('selection_threshold', 0.1),
            max_tokens=self.config.get('max_tokens', 100),
            temperature=self.config.get('temperature', 1.0),
            model_name=self.model_name,
            steps=self.steps,
            statistics=stats,
            timestamp=datetime.now().isoformat(),
            seed=self.seed
        )

        return tree

    def _calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate Shannon entropy of probability distribution.

        Args:
            probabilities: List of probabilities

        Returns:
            Entropy in bits
        """
        if not probabilities:
            return 0.0

        # Normalize probabilities
        total = sum(probabilities)
        if total == 0:
            return 0.0

        normalized = [p / total for p in probabilities]

        # Calculate entropy: H = -sum(p * log2(p))
        entropy = 0.0
        for p in normalized:
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy
