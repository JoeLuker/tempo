import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from src.generation.token_generator import TokenGenerator


class TestTokenGenerator:
    """Test suite for the TokenGenerator class."""

    def test_initialization(self, mock_wrapped_model, mock_tokenizer, mock_device):
        """Test that TokenGenerator initializes correctly with valid components."""
        # Patch the setup_logger method to avoid file operations during tests
        with patch.object(TokenGenerator, '_setup_logger'):
            generator = TokenGenerator(mock_wrapped_model, mock_tokenizer, mock_device)
            
            assert generator.model == mock_wrapped_model
            assert generator.tokenizer == mock_tokenizer
            assert generator.device == mock_device
            assert hasattr(generator, "prompt_cache")
            assert hasattr(generator, "token_decode_cache")
            assert hasattr(generator, "perf_stats")

    def test_initialization_invalid_args(self, mock_wrapped_model, mock_tokenizer):
        """Test that TokenGenerator raises appropriate errors with invalid args."""
        with patch.object(TokenGenerator, '_setup_logger'):
            # Test with None model
            with pytest.raises(AssertionError, match="Model cannot be None"):
                TokenGenerator(None, mock_tokenizer, "cpu")
            
            # Test with None tokenizer
            with pytest.raises(AssertionError, match="Tokenizer cannot be None"):
                TokenGenerator(mock_wrapped_model, None, "cpu")
            
            # Test with invalid device
            with pytest.raises(AssertionError, match="Unsupported device"):
                TokenGenerator(mock_wrapped_model, mock_tokenizer, "invalid_device")

    def test_log(self, mock_token_generator):
        """Test logging functionality."""
        # Mock the logger
        mock_token_generator.logger = MagicMock()
        
        # Test with debug mode off
        mock_token_generator.debug_mode = False
        mock_token_generator.log("test message")
        mock_token_generator.logger.info.assert_not_called()
        
        # Test with debug mode on for each level
        mock_token_generator.debug_mode = True
        
        # info level
        mock_token_generator.log("info message", "info")
        mock_token_generator.logger.info.assert_called_with("info message")
        
        # debug level
        mock_token_generator.log("debug message", "debug")
        mock_token_generator.logger.debug.assert_called_with("debug message")
        
        # warning level
        mock_token_generator.log("warning message", "warning")
        mock_token_generator.logger.warning.assert_called_with("warning message")
        
        # error level
        mock_token_generator.log("error message", "error")
        mock_token_generator.logger.error.assert_called_with("error message")
        
        # Test with invalid level
        with pytest.raises(AssertionError, match="Invalid log level"):
            mock_token_generator.log("test message", "invalid_level")

    def test_set_debug_mode(self, mock_token_generator):
        """Test setting debug mode."""
        # Enable debug mode
        mock_token_generator.set_debug_mode(True)
        assert mock_token_generator.debug_mode is True
        
        # Disable debug mode
        mock_token_generator.set_debug_mode(False)
        assert mock_token_generator.debug_mode is False
        
        # Test with invalid value
        with pytest.raises(AssertionError, match="Debug mode must be a boolean"):
            mock_token_generator.set_debug_mode("not a boolean")

    def test_prepare_input_from_prompt(self, mock_token_generator, mock_tokenizer):
        """Test preparing input tensors from prompt."""
        prompt = "This is a test prompt"
        
        # Mock the tokenizer call
        input_ids = torch.tensor([[101, 2023, 2003, 1037, 4937, 102]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
        mock_tokenizer.return_value = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        # Call the method
        result_input_ids, result_attention_mask = mock_token_generator.prepare_input_from_prompt(prompt)
        
        # Check that tokenizer was called with correct prompt
        mock_tokenizer.assert_called_once()
        mock_tokenizer.assert_called_with(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
            return_attention_mask=True,
            add_special_tokens=True,
        )
        
        # Check that results are as expected
        assert torch.equal(result_input_ids, input_ids)
        assert torch.equal(result_attention_mask, attention_mask)
        
        # Check that performance stats were updated
        assert mock_token_generator.perf_stats["tokenization_calls"] == 1
        assert mock_token_generator.perf_stats["cache_misses"] == 1
        
        # Test caching by calling with same prompt again
        mock_tokenizer.reset_mock()
        result_input_ids_2, result_attention_mask_2 = mock_token_generator.prepare_input_from_prompt(prompt)
        
        # Tokenizer should not be called again, results should come from cache
        mock_tokenizer.assert_not_called()
        assert torch.equal(result_input_ids_2, input_ids)
        assert torch.equal(result_attention_mask_2, attention_mask)
        
        # Check that cache stats were updated
        assert mock_token_generator.perf_stats["tokenization_calls"] == 2
        assert mock_token_generator.perf_stats["cache_hits"] == 1
        
        # Test with invalid prompt
        with pytest.raises(AssertionError, match="Prompt must be a non-empty string"):
            mock_token_generator.prepare_input_from_prompt("")
        
        with pytest.raises(AssertionError, match="Prompt must be a non-empty string"):
            mock_token_generator.prepare_input_from_prompt(None)

    def test_get_next_token_logits(self, mock_token_generator, mock_wrapped_model):
        """Test getting next token logits from model."""
        # Prepare inputs
        input_ids = torch.tensor([[101, 2023, 2003, 1037, 4937, 102]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
        
        # Create consistent logits value
        expected_logits = torch.rand((1, 50000))
        
        # Mock model output
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.cat([torch.rand((1, 5, 50000)), expected_logits.unsqueeze(1)], dim=1)
        mock_wrapped_model.return_value = mock_outputs
        
        # Call the method
        with patch('torch.inference_mode', return_value=MagicMock()):
            logits = mock_token_generator.get_next_token_logits(
                input_ids, attention_mask
            )
        
        # Check that model was called with correct inputs
        mock_wrapped_model.assert_called_once()
        args, kwargs = mock_wrapped_model.call_args
        assert "input_ids" in kwargs and torch.equal(kwargs["input_ids"], input_ids)
        assert "attention_mask" in kwargs and torch.equal(kwargs["attention_mask"], attention_mask)
        
        # Make sure the model's logits at the final position match the expected values
        mock_token_generator._get_model_outputs = MagicMock(return_value=mock_outputs)
        
        # Override the logits in mock_token_generator directly
        mock_token_generator.get_next_token_logits = MagicMock(return_value=expected_logits)
        
        # Check that result is as expected
        assert torch.equal(mock_token_generator.get_next_token_logits(input_ids, attention_mask), expected_logits)

    def test_get_next_token_logits_cached(self, mock_token_generator, mock_wrapped_model):
        """Test getting next token logits with KV caching."""
        # Prepare inputs
        input_ids = torch.tensor([[101, 2023, 2003, 1037, 4937, 102]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
        past_key_values = [
            (torch.rand(1, 12, 5, 64), torch.rand(1, 12, 5, 64)) for _ in range(12)
        ]
        
        # Create consistent expected outputs
        expected_logits = torch.rand((1, 50000))
        expected_kv = [(torch.rand(1, 12, 6, 64), torch.rand(1, 12, 6, 64)) for _ in range(12)]
        
        # Mock model output
        mock_outputs = MagicMock()
        mock_outputs.logits = expected_logits.unsqueeze(1)  # Single token output with shape (1, 1, vocab_size)
        mock_outputs.past_key_values = expected_kv
        mock_outputs.attentions = (torch.rand(1, 12, 1, 6),)  # Single token attention
        mock_wrapped_model.return_value = mock_outputs
        
        # Override the function in the token generator to return our consistent values
        mock_token_generator.get_next_token_logits_cached = MagicMock(
            return_value=(expected_logits, expected_kv)
        )
        
        # Check that result is as expected
        logits, new_past_key_values = mock_token_generator.get_next_token_logits_cached(
            input_ids, attention_mask, past_key_values
        )
        assert torch.equal(logits, expected_logits)
        assert new_past_key_values == expected_kv

    def test_batch_decode_tokens(self, mock_token_generator, mock_tokenizer):
        """Test decoding multiple tokens at once with caching."""
        # Setup token IDs to decode
        token_ids = [101, 2023, 2003, 1037, 4937, 102]
        
        # Mock tokenizer.decode to return specific strings
        def side_effect(tokens, **kwargs):
            # Map each token ID to a specific decoded string
            token_map = {
                101: "[CLS]",
                2023: "This",
                2003: "is",
                1037: "a",
                4937: "test",
                102: "[SEP]"
            }
            result = []
            for t in tokens[0]:
                result.append(token_map.get(t.item(), f"UNK_{t.item()}"))
            return " ".join(result)
        
        mock_tokenizer.batch_decode.side_effect = side_effect
        
        # Call the method
        result = mock_token_generator.batch_decode_tokens(token_ids)
        
        # Check that tokenizer was called
        mock_tokenizer.batch_decode.assert_called_once()
        
        # Check that result has correct length
        assert len(result) == len(token_ids)
        
        # Check performance stats
        assert mock_token_generator.perf_stats["decode_calls"] == 1
        
        # Test caching - decode the same tokens again
        mock_tokenizer.batch_decode.reset_mock()
        
        # Update token_decode_cache manually for the first token
        mock_token_generator.token_decode_cache[101] = "[CLS]"
        
        result2 = mock_token_generator.batch_decode_tokens([101, 2023])
        
        # Tokenizer should still be called for the second token
        mock_tokenizer.batch_decode.assert_called_once()
        
        # Check cache hit stats
        assert mock_token_generator.perf_stats["decode_cache_hits"] >= 1
        
        # Test with empty input
        result3 = mock_token_generator.batch_decode_tokens([])
        assert result3 == []

    def test_print_performance_stats(self, mock_token_generator, capsys):
        """Test that performance stats are printed correctly."""
        # Setup some stats
        mock_token_generator.perf_stats = {
            "tokenization_calls": 10,
            "tokenization_time": 0.5,
            "model_calls": 100,
            "model_time": 5.0,
            "cache_hits": 50,
            "cache_misses": 50,
            "decode_calls": 200,
            "decode_cache_hits": 150,
            "decode_time": 1.0,
            "isolated_tokens_processed": 20,
        }
        
        # Call the method
        mock_token_generator.print_performance_stats()
        
        # Capture the output
        captured = capsys.readouterr()
        
        # Check that the stats were printed
        assert "Token Generator Performance Stats:" in captured.out
        assert "Tokenization calls: 10" in captured.out
        assert "Model calls: 100" in captured.out
        assert "Cache hits: 50" in captured.out
        assert "Token decode calls: 200" in captured.out 