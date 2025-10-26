"""Unit tests for GenerationOrchestrator domain service."""

import pytest
import torch
from unittest.mock import Mock, MagicMock, call

from src.domain.services.generation_orchestrator import GenerationOrchestrator
from src.domain.entities.generation_state import GenerationState
from src.domain.entities.parallel_generation import GenerationConfig
from src.domain.entities.logits import TokenLogits
from src.domain.entities.token import Token, TokenSet


class TestGenerationOrchestrator:
    """Tests for GenerationOrchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""
        return GenerationOrchestrator(debug_mode=False)

    @pytest.fixture
    def mock_strategy(self):
        """Mock generation strategy."""
        strategy = MagicMock()
        # Return a token set with one token
        strategy.select_tokens.return_value = TokenSet(
            tokens=[Token(id=42, text="test", logit=2.5, probability=0.8, position=0)],
            position=0
        )
        strategy.should_terminate.return_value = False
        return strategy

    @pytest.fixture
    def mock_token_generator(self):
        """Mock token generator."""
        generator = MagicMock()
        # Return logits and updated state
        logits = TokenLogits(
            tensor=torch.randn(1, 1000),
            sequence_position=0,
            batch_index=0
        )
        state = GenerationState(
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            attention_mask=torch.ones((1, 4)),
            sequence_length=4
        )
        generator.generate_logits_with_cache.return_value = (logits, state)
        return generator

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert orchestrator.logical_layout == []
        assert orchestrator.sequence_tracker is not None
        assert orchestrator.removal_coordinator is not None

    def test_single_step_generation(
        self,
        orchestrator,
        basic_generation_state,
        basic_generation_config,
        mock_strategy,
        mock_token_generator
    ):
        """Test single step generation without parallel tokens."""
        # Set max_tokens to 1 for single step
        config = GenerationConfig(
            max_tokens=1,
            selection_threshold=0.5,
            isolate_parallel_tokens=True
        )

        result, final_state = orchestrator.orchestrate_generation(
            initial_state=basic_generation_state,
            config=config,
            strategy=mock_strategy,
            token_generator=mock_token_generator
        )

        # Verify token generator was called
        assert mock_token_generator.generate_logits_with_cache.called
        # Verify strategy was called
        assert mock_strategy.select_tokens.called
        # Verify result contains generated tokens
        assert result is not None
        assert hasattr(result, 'token_sets')

    def test_multiple_step_generation(
        self,
        orchestrator,
        basic_generation_state,
        mock_strategy,
        mock_token_generator
    ):
        """Test multiple step generation."""
        config = GenerationConfig(
            max_tokens=3,
            selection_threshold=0.5,
            isolate_parallel_tokens=False
        )

        # Make strategy return terminate after 3 steps
        call_count = [0]

        def should_terminate_side_effect(*args, **kwargs):
            call_count[0] += 1
            return call_count[0] >= 3

        mock_strategy.should_terminate.side_effect = should_terminate_side_effect

        result, final_state = orchestrator.orchestrate_generation(
            initial_state=basic_generation_state,
            config=config,
            strategy=mock_strategy,
            token_generator=mock_token_generator
        )

        # Verify multiple steps were executed
        assert mock_token_generator.generate_logits_with_cache.call_count >= 2

    def test_attention_mask_integration(
        self,
        orchestrator,
        basic_generation_state,
        basic_generation_config,
        mock_strategy,
        mock_token_generator,
        mock_attention_manager
    ):
        """Test attention manager integration."""
        config = GenerationConfig(
            max_tokens=1,
            selection_threshold=0.5,
            isolate_parallel_tokens=True
        )

        result, final_state = orchestrator.orchestrate_generation(
            initial_state=basic_generation_state,
            config=config,
            strategy=mock_strategy,
            token_generator=mock_token_generator,
            attention_manager=mock_attention_manager
        )

        # Verify attention manager was called to build masks
        assert mock_attention_manager.build_attention_mask.called

    def test_parallel_token_registration(
        self,
        orchestrator,
        basic_generation_state,
        mock_strategy,
        mock_token_generator,
        mock_attention_manager
    ):
        """Test parallel token set registration."""
        config = GenerationConfig(
            max_tokens=1,
            selection_threshold=0.1,
            isolate_parallel_tokens=True
        )

        # Make strategy return multiple tokens
        mock_strategy.select_tokens.return_value = TokenSet(
            tokens=[
                Token(id=42, text="hello", logit=1.5, probability=0.5, position=0),
                Token(id=100, text="world", logit=1.0, probability=0.3, position=0)
            ],
            position=0
        )

        result, final_state = orchestrator.orchestrate_generation(
            initial_state=basic_generation_state,
            config=config,
            strategy=mock_strategy,
            token_generator=mock_token_generator,
            attention_manager=mock_attention_manager
        )

        # Verify attention manager registered the parallel set
        assert mock_attention_manager.register_parallel_set.called
