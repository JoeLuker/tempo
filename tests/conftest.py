"""Pytest configuration and shared fixtures."""

import pytest
import torch
from unittest.mock import Mock, MagicMock
from typing import Any

from src.domain.entities.generation_state import GenerationState
from src.domain.entities.logits import TokenLogits
from src.domain.entities.token import Token, TokenSet
from src.domain.entities.parallel_generation import GenerationConfig


@pytest.fixture
def mock_device():
    """Mock device for testing."""
    return "cpu"


@pytest.fixture
def sample_input_ids():
    """Sample input_ids tensor for testing."""
    return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)


@pytest.fixture
def sample_attention_mask():
    """Sample attention mask tensor for testing."""
    return torch.ones((1, 4), dtype=torch.long)


@pytest.fixture
def basic_generation_state(sample_input_ids, sample_attention_mask):
    """Basic generation state fixture."""
    return GenerationState(
        input_ids=sample_input_ids,
        attention_mask=sample_attention_mask,
        sequence_length=4
    )


@pytest.fixture
def sample_logits():
    """Sample logits tensor for testing."""
    # Create logits with known probabilities
    vocab_size = 1000
    logits_tensor = torch.randn(1, vocab_size)
    # Make token 42 have high probability
    logits_tensor[0, 42] = 10.0
    # Make token 100 have medium probability
    logits_tensor[0, 100] = 5.0
    return TokenLogits(
        tensor=logits_tensor,
        sequence_position=3,
        batch_index=0
    )


@pytest.fixture
def sample_token():
    """Sample token fixture."""
    return Token(id=42, text="test", logit=2.5, probability=0.8, position=0)


@pytest.fixture
def sample_token_set():
    """Sample token set fixture."""
    tokens = [
        Token(id=42, text="hello", logit=1.5, probability=0.5, position=0),
        Token(id=100, text="world", logit=1.0, probability=0.3, position=0),
        Token(id=200, text="test", logit=0.8, probability=0.2, position=0)
    ]
    return TokenSet(tokens=tokens, position=0)


@pytest.fixture
def basic_generation_config():
    """Basic generation configuration fixture."""
    return GenerationConfig(
        max_tokens=50,
        selection_threshold=0.1,
        isolate_parallel_tokens=True,
        disable_kv_cache=False,
        use_retroactive_removal=False
    )


@pytest.fixture
def mock_model_adapter():
    """Mock model adapter for testing."""
    mock = MagicMock()

    # Setup forward method to return reasonable outputs
    def mock_forward(*args, **kwargs):
        output = MagicMock()
        output.logits = torch.randn(1, 1, 1000)
        output.attentions = None
        output.past_key_values = None
        return output

    mock.forward = mock_forward
    return mock


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    mock = MagicMock()
    mock.encode.return_value = [1, 2, 3, 4]
    mock.decode.return_value = "test output"
    mock.eos_token_id = 2
    return mock


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager for testing."""
    mock = MagicMock()
    mock.cache_attention = MagicMock()
    mock.get_cached_attention = MagicMock(return_value=None)
    return mock


@pytest.fixture
def mock_performance_tracker():
    """Mock performance tracker for testing."""
    mock = MagicMock()
    mock.track_model_call = MagicMock()
    mock.get_stats = MagicMock(return_value={})
    return mock


@pytest.fixture
def mock_attention_manager():
    """Mock attention manager for testing."""
    mock = MagicMock()
    mock.build_attention_mask = MagicMock(return_value=torch.zeros(10, 10))
    mock.register_parallel_set = MagicMock()
    mock.reset = MagicMock()
    return mock
