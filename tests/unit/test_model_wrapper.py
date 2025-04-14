import pytest
import torch
from unittest.mock import patch, MagicMock
from src.modeling.model_wrapper import TEMPOModelWrapper

class TestTEMPOModelWrapper:
    """Test suite for the TEMPOModelWrapper class."""

    def test_initialization(self, mock_model):
        """Test that wrapper initializes correctly with valid model."""
        wrapper = TEMPOModelWrapper(mock_model)
        assert wrapper.model == mock_model
        assert hasattr(wrapper, "intermediate_values")
        assert hasattr(wrapper, "activation_hooks")
        assert wrapper.config == mock_model.config
        assert wrapper.device is not None

    def test_initialization_with_invalid_model(self):
        """Test that wrapper raises appropriate errors with invalid model."""
        with pytest.raises(AssertionError, match="Model cannot be None"):
            TEMPOModelWrapper(None)
        
        invalid_model = MagicMock()
        delattr(invalid_model, "forward")
        with pytest.raises(AssertionError, match="Model must have a forward method"):
            TEMPOModelWrapper(invalid_model)

    def test_register_hooks(self, mock_model):
        """Test that hooks are properly registered on model components."""
        # Create a model with specific components to hook into
        model = MagicMock()
        attention_module = MagicMock()
        rope_module = MagicMock()
        output_module = MagicMock()
        
        # Setup named_modules to return our mocked modules
        def mock_named_modules():
            yield "transformer.attention", attention_module
            yield "transformer.rope", rope_module
            yield "lm_head", output_module
        
        model.named_modules = mock_named_modules
        model.config = MagicMock()
        params_iter = iter([torch.nn.Parameter(torch.rand(1))])
        model.parameters = lambda: params_iter
        
        # Create wrapper with our custom model
        wrapper = TEMPOModelWrapper(model)
        
        # Check that register_forward_hook was called on each module
        assert attention_module.register_forward_hook.called
        assert rope_module.register_forward_hook.called
        assert output_module.register_forward_hook.called
        
        # Check that activation hooks were stored
        assert len(wrapper.activation_hooks) == 3

    def test_ensure_device(self, mock_wrapped_model):
        """Test that tensors are properly moved to the correct device."""
        # Create tensor on CPU
        cpu_tensor = torch.tensor([1, 2, 3])
        
        # Mock device to be CUDA
        mock_wrapped_model.device = torch.device("cuda:0")
        
        # Patch torch.Tensor.to to avoid actual device movement
        with patch.object(torch.Tensor, "to", return_value=torch.tensor([1, 2, 3])) as mock_to:
            result = mock_wrapped_model._ensure_device(cpu_tensor)
            mock_to.assert_called_once_with(mock_wrapped_model.device)
        
        # Test with non-tensor input
        non_tensor = "not a tensor"
        result = mock_wrapped_model._ensure_device(non_tensor)
        assert result == non_tensor

    def test_forward(self, mock_wrapped_model, mock_model):
        """Test forward pass with device management."""
        # Setup input tensors
        input_tensor = torch.tensor([1, 2, 3])
        kwargs = {"attention_mask": torch.tensor([1, 1, 1])}
        
        # Patch _ensure_device
        with patch.object(mock_wrapped_model, "_ensure_device", side_effect=lambda x: x) as mock_ensure:
            mock_wrapped_model.forward(input_tensor, **kwargs)
            
            # Check that _ensure_device was called for args and kwargs
            assert mock_ensure.call_count >= 2
            
            # Check that model's forward was called with correct args
            mock_model.assert_called_once_with(input_tensor, **kwargs)
            
            # Check that intermediate values were cleared
            assert mock_wrapped_model.intermediate_values == {}

    def test_generate(self, mock_wrapped_model, mock_model):
        """Test generate method properly delegates to model."""
        # Setup input tensors
        input_tensor = torch.tensor([1, 2, 3])
        kwargs = {"max_length": 10}
        
        # Call generate
        result = mock_wrapped_model.generate(input_tensor, **kwargs)
        
        # Check that model's generate was called with correct args
        mock_model.generate.assert_called_once_with(input_tensor, **kwargs)
        
        # Check that intermediate values were reset
        assert mock_wrapped_model.intermediate_values == {}
        
        # Check that result is as expected
        assert result is mock_model.generate.return_value

    def test_unwrap(self, mock_wrapped_model, mock_model):
        """Test unwrapping returns original model and cleans up hooks."""
        # Add some mock hooks
        hook1 = MagicMock()
        hook2 = MagicMock()
        mock_wrapped_model.activation_hooks = [hook1, hook2]
        
        # Call unwrap
        result = mock_wrapped_model.unwrap()
        
        # Check that hooks were removed
        hook1.remove.assert_called_once()
        hook2.remove.assert_called_once()
        
        # Check that activation_hooks was cleared
        assert mock_wrapped_model.activation_hooks == []
        
        # Check that original model was returned
        assert result is mock_model

    def test_set_debug_mode(self, mock_wrapped_model):
        """Test setting debug mode."""
        # Enable debug mode
        mock_wrapped_model.set_debug_mode(True)
        assert mock_wrapped_model.debug_mode is True
        
        # Disable debug mode
        mock_wrapped_model.set_debug_mode(False)
        assert mock_wrapped_model.debug_mode is False
        
        # Test with invalid value
        with pytest.raises(AssertionError, match="Debug mode must be a boolean"):
            mock_wrapped_model.set_debug_mode("not a boolean")

    def test_getattr(self, mock_wrapped_model, mock_model):
        """Test attribute access is forwarded to wrapped model."""
        # Add attribute to mock model
        mock_model.special_attr = "special value"
        
        # Access via wrapper
        assert mock_wrapped_model.special_attr == "special value"
        
        # Test missing attribute
        with pytest.raises(AttributeError, match="missing_attr"):
            # Try to access a non-existent attribute
            mock_wrapped_model.missing_attr 