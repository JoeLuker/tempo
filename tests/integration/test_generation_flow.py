"""Integration tests for end-to-end generation flow."""

import pytest
import torch
from unittest.mock import MagicMock, Mock

from src.application.use_cases.generate_text import GenerateTextUseCase
from src.domain.entities.parallel_generation import GenerationConfig, GenerationResult
from src.domain.entities.generation_state import GenerationState, TokenizationResult
from src.infrastructure.generation.token_generator_impl import TokenGeneratorImpl
from src.infrastructure.generation.standard_generation_strategy import StandardGenerationStrategy
from src.application.services.sequence_manager import SequenceManager
from src.application.services.attention_service import AttentionService


@pytest.mark.integration
class TestGenerationFlow:
    """Integration tests for the complete generation pipeline."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that returns reasonable outputs."""
        model = MagicMock()

        def forward_impl(*args, **kwargs):
            # Get input shape
            input_ids = kwargs.get('input_ids')
            batch_size = input_ids.shape[0] if input_ids is not None else 1
            seq_len = input_ids.shape[1] if input_ids is not None else 1
            vocab_size = 1000

            # Create output
            output = MagicMock()
            # Return logits with predictable high-probability tokens
            logits = torch.randn(batch_size, seq_len, vocab_size)
            # Make token 42 consistently high probability
            logits[:, :, 42] = 10.0
            output.logits = logits
            output.attentions = None
            output.past_key_values = None

            return output

        model.forward = forward_impl
        return model

    @pytest.fixture
    def mock_tokenizer_adapter(self):
        """Create a mock tokenizer adapter."""
        tokenizer = MagicMock()

        def tokenize_prompt_impl(prompt):
            # Simple tokenization mock
            input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
            attention_mask = torch.ones((1, 4), dtype=torch.long)
            return TokenizationResult(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_count=4
            )

        def decode_impl(token_ids):
            return "test generated output"

        tokenizer.tokenize_prompt = tokenize_prompt_impl
        tokenizer.decode = decode_impl
        tokenizer.eos_token_id = 2

        return tokenizer

    @pytest.fixture
    def token_generator(self, mock_model):
        """Create token generator with mocked model."""
        from src.infrastructure.model.model_adapter import ModelAdapter

        # Wrap mock model in adapter
        model_adapter = ModelAdapter(
            model=mock_model,
            device="cpu"
        )

        return TokenGeneratorImpl(
            model_adapter=model_adapter,
            debug_mode=False
        )

    @pytest.fixture
    def generation_strategy(self):
        """Create standard generation strategy."""
        return StandardGenerationStrategy()

    @pytest.fixture
    def sequence_manager(self, mock_tokenizer_adapter):
        """Create sequence manager."""
        return SequenceManager(
            tokenizer=mock_tokenizer_adapter
        )

    @pytest.fixture
    def use_case(
        self,
        token_generator,
        mock_tokenizer_adapter,
        generation_strategy,
        sequence_manager
    ):
        """Create the generate text use case."""
        return GenerateTextUseCase(
            token_generator=token_generator,
            tokenizer=mock_tokenizer_adapter,
            generation_strategy=generation_strategy,
            sequence_manager=sequence_manager,
            debug_mode=False
        )

    def test_basic_generation_completes(self, use_case):
        """Test that basic generation completes without errors."""
        config = GenerationConfig(
            max_tokens=5,
            selection_threshold=0.5,
            isolate_parallel_tokens=False,
            disable_kv_cache=True  # Simpler for testing
        )

        result = use_case.execute(
            prompt="Test prompt",
            config=config
        )

        assert result is not None
        assert isinstance(result, GenerationResult)
        assert result.generated_text is not None

    def test_isolated_vs_visible_modes(
        self,
        token_generator,
        mock_tokenizer_adapter,
        generation_strategy,
        sequence_manager
    ):
        """Test that isolated and visible modes can both complete."""
        # Test isolated mode
        isolated_attention = AttentionService(
            isolate_parallel_tokens=True,
            device="cpu"
        )

        isolated_use_case = GenerateTextUseCase(
            token_generator=token_generator,
            tokenizer=mock_tokenizer_adapter,
            generation_strategy=generation_strategy,
            sequence_manager=sequence_manager,
            attention_manager=isolated_attention,
            debug_mode=False
        )

        isolated_config = GenerationConfig(
            max_tokens=3,
            selection_threshold=0.5,
            isolate_parallel_tokens=True,
            disable_kv_cache=True
        )

        isolated_result = isolated_use_case.execute(
            prompt="Test",
            config=isolated_config
        )

        assert isolated_result is not None

        # Test visible mode
        visible_attention = AttentionService(
            isolate_parallel_tokens=False,
            device="cpu"
        )

        visible_use_case = GenerateTextUseCase(
            token_generator=token_generator,
            tokenizer=mock_tokenizer_adapter,
            generation_strategy=generation_strategy,
            sequence_manager=sequence_manager,
            attention_manager=visible_attention,
            debug_mode=False
        )

        visible_config = GenerationConfig(
            max_tokens=3,
            selection_threshold=0.5,
            isolate_parallel_tokens=False,
            disable_kv_cache=True
        )

        visible_result = visible_use_case.execute(
            prompt="Test",
            config=visible_config
        )

        assert visible_result is not None

    def test_attention_manager_called_in_isolated_mode(
        self,
        token_generator,
        mock_tokenizer_adapter,
        generation_strategy,
        sequence_manager,
        mock_attention_manager
    ):
        """Test that attention manager is properly invoked in isolated mode."""
        use_case = GenerateTextUseCase(
            token_generator=token_generator,
            tokenizer=mock_tokenizer_adapter,
            generation_strategy=generation_strategy,
            sequence_manager=sequence_manager,
            attention_manager=mock_attention_manager,
            debug_mode=False
        )

        config = GenerationConfig(
            max_tokens=2,
            selection_threshold=0.5,
            isolate_parallel_tokens=True,
            disable_kv_cache=True
        )

        result = use_case.execute(
            prompt="Test",
            config=config
        )

        # Verify attention manager was used
        assert mock_attention_manager.build_attention_mask.called
