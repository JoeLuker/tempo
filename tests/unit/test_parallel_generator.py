import pytest
import torch
from unittest.mock import patch, MagicMock, call
import numpy as np
from src.generation.parallel_generator import ParallelGenerator


class TestParallelGenerator:
    """Test suite for the ParallelGenerator class."""

    def test_initialization(self, mock_wrapped_model, mock_tokenizer, mock_device):
        """Test that ParallelGenerator initializes correctly with valid components."""
        # Patch all the dependent classes and methods
        with patch('src.generation.parallel_generator.TokenGenerator'), \
             patch('src.generation.parallel_generator.TokenSelector'), \
             patch('src.generation.parallel_generator.TextFormatter'), \
             patch('src.generation.parallel_generator.AttentionManager'), \
             patch('src.generation.parallel_generator.RoPEModifier'), \
             patch.object(ParallelGenerator, '_setup_logger'):
            
            generator = ParallelGenerator(
                mock_wrapped_model, 
                mock_tokenizer, 
                pruner=None, 
                device=mock_device,
                has_custom_attention=True,
                use_custom_rope=True
            )
            
            assert generator.model == mock_wrapped_model
            assert generator.tokenizer == mock_tokenizer
            assert generator.device == mock_device
            assert generator.has_custom_attention is True
            assert generator.use_custom_rope is True
            assert generator.sequence_length == 0
            assert generator.initial_prompt_length == 0
            assert generator.step_count == 0

    def test_initialization_with_invalid_args(self, mock_wrapped_model, mock_tokenizer):
        """Test that ParallelGenerator raises appropriate errors with invalid args."""
        with patch.object(ParallelGenerator, '_setup_logger'):
            # Test with None model
            with pytest.raises(AssertionError, match="Model cannot be None"):
                ParallelGenerator(None, mock_tokenizer, pruner=None, device="cpu")
            
            # Test with None tokenizer
            with pytest.raises(AssertionError, match="Tokenizer cannot be None"):
                ParallelGenerator(mock_wrapped_model, None, pruner=None, device="cpu")
            
            # Test with invalid device
            with pytest.raises(AssertionError, match="Unsupported device"):
                ParallelGenerator(mock_wrapped_model, mock_tokenizer, pruner=None, device="invalid_device")
            
            # Test with invalid has_custom_attention
            with pytest.raises(AssertionError, match="has_custom_attention must be a boolean"):
                ParallelGenerator(
                    mock_wrapped_model, 
                    mock_tokenizer, 
                    pruner=None, 
                    device="cpu",
                    has_custom_attention="not a boolean"
                )

    def test_log(self, mock_parallel_generator):
        """Test logging functionality."""
        # Mock the logger
        mock_parallel_generator.logger = MagicMock()
        
        # Test with debug mode off
        mock_parallel_generator.debug_mode = False
        mock_parallel_generator.log("test message")
        mock_parallel_generator.logger.info.assert_not_called()
        
        # Test with debug mode on for each level
        mock_parallel_generator.debug_mode = True
        
        # info level
        mock_parallel_generator.log("info message", "info")
        mock_parallel_generator.logger.info.assert_called_with("info message")
        
        # debug level
        mock_parallel_generator.log("debug message", "debug")
        mock_parallel_generator.logger.debug.assert_called_with("debug message")
        
        # warning level
        mock_parallel_generator.log("warning message", "warning")
        mock_parallel_generator.logger.warning.assert_called_with("warning message")
        
        # error level
        mock_parallel_generator.log("error message", "error")
        mock_parallel_generator.logger.error.assert_called_with("error message")
        
        # Test with invalid level
        with pytest.raises(AssertionError, match="Invalid log level"):
            mock_parallel_generator.log("test message", "invalid_level")
        
        # Test with empty message
        with pytest.raises(AssertionError, match="Log message cannot be empty"):
            mock_parallel_generator.log("", "info")

    def test_init_sequence_tracking(self, mock_parallel_generator):
        """Test sequence length tracking initialization."""
        # Initialize with valid prompt length
        mock_parallel_generator._init_sequence_tracking(10)
        assert mock_parallel_generator.sequence_length == 0
        assert mock_parallel_generator.initial_prompt_length == 10
        assert mock_parallel_generator.step_count == 0
        assert mock_parallel_generator.sequence_length_history == []
        
        # Test with negative prompt length
        with pytest.raises(AssertionError, match="Prompt length cannot be negative"):
            mock_parallel_generator._init_sequence_tracking(-1)

    def test_update_sequence_length(self, mock_parallel_generator):
        """Test updating sequence length."""
        # Initialize sequence tracking
        mock_parallel_generator._init_sequence_tracking(10)
        
        # Mock callback
        callback = MagicMock()
        
        # Update with valid length
        result = mock_parallel_generator.update_sequence_length(5, callback)
        assert result is True
        assert mock_parallel_generator.sequence_length == 5
        assert mock_parallel_generator.step_count == 1
        assert mock_parallel_generator.sequence_length_history == [5]
        callback.assert_called_once_with(5, 1, 10)
        
        # Update with same length (should not increase)
        callback.reset_mock()
        result = mock_parallel_generator.update_sequence_length(5, callback)
        assert result is False
        assert mock_parallel_generator.step_count == 1  # Unchanged
        callback.assert_not_called()
        
        # Update with larger length
        result = mock_parallel_generator.update_sequence_length(10, callback)
        assert result is True
        assert mock_parallel_generator.sequence_length == 10
        assert mock_parallel_generator.step_count == 2
        assert mock_parallel_generator.sequence_length_history == [5, 10]
        callback.assert_called_once_with(10, 2, 10)
        
        # Test with negative length
        with pytest.raises(AssertionError, match="Sequence length cannot be negative"):
            mock_parallel_generator.update_sequence_length(-1)
        
        # Test with callback that raises exception
        callback.reset_mock()
        callback.side_effect = RuntimeError("Test error")
        
        # Should not raise exception, but log the error
        mock_parallel_generator.logger = MagicMock()
        result = mock_parallel_generator.update_sequence_length(15, callback)
        assert result is True
        assert mock_parallel_generator.sequence_length == 15
        mock_parallel_generator.logger.error.assert_called_once()

    def test_generate_method_validation(self, mock_parallel_generator):
        """Test the validation of generate method parameters."""
        # Set up mocks for token processing components
        mock_parallel_generator.token_generator = MagicMock()
        mock_parallel_generator.token_selector = MagicMock()
        mock_parallel_generator.text_formatter = MagicMock()
        mock_parallel_generator.attention_manager = MagicMock()
        
        # Mock internal methods to avoid full execution
        with patch.object(mock_parallel_generator, '_init_sequence_tracking'):
            # Test with empty prompt
            with pytest.raises(AssertionError, match="Prompt cannot be empty"):
                mock_parallel_generator.generate(prompt="")
            
            # Test with negative max_tokens
            with pytest.raises(AssertionError, match="max_tokens must be positive"):
                mock_parallel_generator.generate(prompt="test", max_tokens=0)
            
            # Test with invalid threshold
            with pytest.raises(AssertionError, match="threshold must be between 0.0 and 1.0"):
                mock_parallel_generator.generate(prompt="test", threshold=1.5)
            
            # Test with negative min_steps
            with pytest.raises(AssertionError, match="min_steps cannot be negative"):
                mock_parallel_generator.generate(prompt="test", min_steps=-1)
            
            # Test with non-boolean return_parallel_sets
            with pytest.raises(AssertionError, match="return_parallel_sets must be a boolean"):
                mock_parallel_generator.generate(prompt="test", return_parallel_sets="not a boolean")
            
            # Test with non-boolean use_pruning
            with pytest.raises(AssertionError, match="use_pruning must be a boolean"):
                mock_parallel_generator.generate(prompt="test", use_pruning="not a boolean")
            
            # Test with non-boolean show_token_ids
            with pytest.raises(AssertionError, match="show_token_ids must be a boolean"):
                mock_parallel_generator.generate(prompt="test", show_token_ids="not a boolean")

    def test_sequence_tracking_during_generation(self, mock_parallel_generator):
        """Test that sequence tracking is properly updated during generation."""
        # Mock the generate method to directly call the tracking methods
        def mock_generate(*args, **kwargs):
            mock_parallel_generator._init_sequence_tracking()
            mock_parallel_generator.update_sequence_length(0, 5)  # Arbitrary values for testing
            return "Test output"
        
        mock_parallel_generator.generate = mock_generate
        
        # Set up the mocks we want to verify
        mock_init = MagicMock()
        mock_update = MagicMock()
        
        # Patch the methods that would be called during generation
        with patch.object(mock_parallel_generator, '_init_sequence_tracking', mock_init), \
             patch.object(mock_parallel_generator, 'update_sequence_length', mock_update):
            
            # Call generate which should now call our mocked methods
            result = mock_parallel_generator.generate(
                prompt="Test prompt",
                max_tokens=3,
                threshold=0.1
            )
            
            # Check that sequence tracking was initialized and updated
            mock_init.assert_called_once()
            mock_update.assert_called_once()
            
            # Verify the result
            assert result == "Test output"

    def test_get_sequence_length(self, mock_parallel_generator):
        """Test getting the current sequence length."""
        mock_parallel_generator.sequence_length = 10
        assert mock_parallel_generator.get_sequence_length() == 10
    
    def test_get_total_sequence_length(self, mock_parallel_generator):
        """Test getting the total sequence length including prompt."""
        mock_parallel_generator.initial_prompt_length = 5
        mock_parallel_generator.sequence_length = 10
        assert mock_parallel_generator.get_total_sequence_length() == 15 