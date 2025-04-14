import pytest
import torch
import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import after setting up the path
from src.modeling.model_wrapper import TEMPOModelWrapper

@pytest.fixture
def mock_device():
    """Return a mock device string for testing."""
    return "cpu"

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    
    # Mock encode method
    tokenizer.encode.return_value = [101, 2023, 2003, 1037, 4937, 102]
    
    # Mock decode method
    tokenizer.decode.return_value = "This is a test"
    
    # Mock batch_decode method
    tokenizer.batch_decode.return_value = ["This is a test"]
    
    # Mock call method for tokenization
    tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 2023, 2003, 1037, 4937, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]])
    }
    
    # Mock essential attributes
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "[EOS]"
    tokenizer.eos_token_id = 102
    tokenizer.bos_token = "[BOS]"
    tokenizer.bos_token_id = 101
    tokenizer.unk_token = "[UNK]"
    tokenizer.unk_token_id = 100
    
    # Mock vocabulary size
    tokenizer.vocab_size = 50000
    
    # Mock convert_tokens_to_ids and convert_ids_to_tokens
    token_to_id_map = {
        "[PAD]": 0,
        "[BOS]": 101,
        "[EOS]": 102,
        "[UNK]": 100,
        "This": 2023,
        "is": 2003,
        "a": 1037,
        "test": 4937
    }
    id_to_token_map = {v: k for k, v in token_to_id_map.items()}
    
    tokenizer.convert_tokens_to_ids.side_effect = lambda tokens: [token_to_id_map.get(t, 100) for t in tokens]
    tokenizer.convert_ids_to_tokens.side_effect = lambda ids: [id_to_token_map.get(i, "[UNK]") for i in ids]
    
    return tokenizer

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    
    # Mock forward method
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.rand((1, 6, 50000))  # (batch, seq, vocab)
    mock_outputs.past_key_values = [(torch.rand(1, 12, 6, 64), torch.rand(1, 12, 6, 64)) for _ in range(12)]
    mock_outputs.attentions = tuple(torch.rand(1, 12, 6, 6) for _ in range(12))
    model.return_value = mock_outputs
    
    # Mock generate method
    model.generate.return_value = torch.tensor([[101, 2023, 2003, 1037, 4937, 102, 2001, 2004, 2005]])
    
    # Mock attributes
    model.config = MagicMock()
    model.config.model_type = "test_model"
    model.config.vocab_size = 50000
    
    # Set up mock named modules
    attention_module = MagicMock()
    rope_module = MagicMock()
    lm_head_module = MagicMock()
    
    # Create a mock hook to be returned by register_forward_hook
    mock_hook = MagicMock()
    mock_hook.remove = MagicMock()  # Ensure remove method exists
    
    # Add register_forward_hook method to each module
    def mock_register_hook(hook_fn):
        # Call the hook function with dummy inputs to ensure it works
        module = MagicMock()
        inputs = (torch.rand(1, 1),)
        outputs = torch.rand(1, 1)
        hook_fn(module, inputs, outputs)
        return mock_hook
    
    attention_module.register_forward_hook = MagicMock(side_effect=mock_register_hook)
    rope_module.register_forward_hook = MagicMock(side_effect=mock_register_hook)
    lm_head_module.register_forward_hook = MagicMock(side_effect=mock_register_hook)
    
    # Configure named_modules to return our mock modules
    mock_modules = {
        "transformer.attention": attention_module,
        "transformer.rope": rope_module,
        "lm_head": lm_head_module
    }
    
    def named_modules_side_effect():
        return mock_modules.items()
    
    model.named_modules = MagicMock(side_effect=named_modules_side_effect)
    
    # Create a mock parameter with a device attribute
    mock_param = torch.nn.Parameter(torch.rand(1))
    # Make parameters() return an iterator that can be called multiple times
    model.parameters = MagicMock(return_value=iter([mock_param]))
    
    return model

@pytest.fixture
def mock_wrapped_model(mock_model, mock_tokenizer):
    """Create a mock wrapped model for testing."""
    # Use a custom class that raises AttributeError for missing_attr
    class CustomMockWrapper(MagicMock):
        def __getattribute__(self, name):
            if name == "missing_attr":
                raise AttributeError("missing_attr not found")
            return super().__getattribute__(name)
    
    # Create the mock with our custom class
    wrapped_model = CustomMockWrapper(spec=TEMPOModelWrapper)
    
    # Set up basic attributes
    wrapped_model.model = mock_model
    wrapped_model.tokenizer = mock_tokenizer
    wrapped_model.device = torch.device("cpu")
    wrapped_model.intermediate_values = {}
    wrapped_model.debug_mode = False
    wrapped_model.config = mock_model.config
    
    # Copy any generation config from the model if it exists
    if hasattr(mock_model, 'generation_config'):
        wrapped_model.generation_config = mock_model.generation_config
    
    # Create mock hooks and activation hooks
    wrapped_model.hooks = {
        "transformer.attention": MagicMock(),
        "transformer.rope": MagicMock(),
        "lm_head": MagicMock()
    }
    
    # Create mock activation hooks with working remove methods
    hook1 = MagicMock()
    hook1.remove = MagicMock()
    hook2 = MagicMock()
    hook2.remove = MagicMock()
    hook3 = MagicMock()
    hook3.remove = MagicMock()
    wrapped_model.activation_hooks = [hook1, hook2, hook3]
    
    # _ensure_device implementation
    def ensure_device(tensor):
        if isinstance(tensor, torch.Tensor) and tensor.device != wrapped_model.device:
            return tensor.to(wrapped_model.device)
        return tensor
    wrapped_model._ensure_device.side_effect = ensure_device
    
    # Forward method implementation
    def forward(*args, **kwargs):
        # Clear intermediate values
        wrapped_model.intermediate_values.clear()
        
        # Process args and kwargs through _ensure_device
        processed_args = [wrapped_model._ensure_device(arg) for arg in args]
        processed_kwargs = {k: wrapped_model._ensure_device(v) for k, v in kwargs.items()}
        
        return mock_model(*processed_args, **processed_kwargs)
    wrapped_model.forward.side_effect = forward
    
    # Generate method implementation
    def generate(*args, **kwargs):
        wrapped_model.intermediate_values = {}
        return mock_model.generate(*args, **kwargs)
    wrapped_model.generate.side_effect = generate
    
    # Unwrap method implementation
    def unwrap():
        for hook in wrapped_model.activation_hooks:
            hook.remove()
        wrapped_model.activation_hooks = []
        return mock_model
    wrapped_model.unwrap.side_effect = unwrap
    
    # Set debug mode implementation
    def set_debug_mode(enabled=False):
        assert isinstance(enabled, bool), "Debug mode must be a boolean"
        wrapped_model.debug_mode = enabled
    wrapped_model.set_debug_mode.side_effect = set_debug_mode
    
    # Add frequently accessed model attributes directly to the wrapper
    for attr_name in ["vocab_size", "hidden_size", "num_hidden_layers", "model_type"]:
        if hasattr(mock_model.config, attr_name):
            setattr(wrapped_model, attr_name, getattr(mock_model.config, attr_name))
    
    # Set up special_attr for test_getattr
    wrapped_model.special_attr = "special value"
    
    return wrapped_model

@pytest.fixture
def mock_token_generator(mock_wrapped_model, mock_tokenizer, mock_device):
    """Create a mock token generator."""
    from src.generation.token_generator import TokenGenerator
    
    # Create with mocks but patch the setup_logger method
    with patch.object(TokenGenerator, '_setup_logger'):
        token_generator = TokenGenerator(mock_wrapped_model, mock_tokenizer, mock_device)
    
    return token_generator

@pytest.fixture
def mock_token_selector(mock_tokenizer):
    """Create a mock token selector."""
    from src.generation.token_selector import TokenSelector
    
    token_selector = TokenSelector(mock_tokenizer)
    
    return token_selector

@pytest.fixture
def mock_text_formatter(mock_tokenizer):
    """Create a mock text formatter."""
    from src.generation.text_formatter import TextFormatter
    
    text_formatter = TextFormatter(mock_tokenizer)
    
    return text_formatter

@pytest.fixture
def mock_parallel_generator(mock_wrapped_model, mock_tokenizer, mock_device):
    """Create a mock parallel generator."""
    from src.generation.parallel_generator import ParallelGenerator
    
    # Create mock instances for each component
    mock_rope = MagicMock()
    mock_rope.install.return_value = True
    
    mock_token_gen = MagicMock()
    mock_token_selector = MagicMock()
    mock_text_formatter = MagicMock()
    mock_attention_mgr = MagicMock()
    
    # Create generator without calling __init__
    with patch.object(ParallelGenerator, '__init__', return_value=None):
        generator = ParallelGenerator.__new__(ParallelGenerator)
    
    # Set up required attributes directly
    generator.model = mock_wrapped_model
    generator.tokenizer = mock_tokenizer
    generator.pruner = None
    generator.device = mock_device
    generator.has_custom_attention = True
    generator.use_custom_rope = True
    generator.debug_mode = False
    generator.is_qwen_model = False
    
    # Set mock components
    generator.rope_modifier = mock_rope
    generator.token_generator = mock_token_gen
    generator.token_selector = mock_token_selector
    generator.text_formatter = mock_text_formatter
    generator.attention_manager = mock_attention_mgr
    
    # Add sequence tracking attributes
    generator.sequence_lengths = {}
    generator._total_length = 0
    generator._current_beam_ids = None
    
    # Add logger
    generator.logger = MagicMock()
    
    return generator 