import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import torch
import json
import numpy as np

from api import (
    app,
    ModelSingleton,
    GenerationResponse,
    TimingInfo,
    ModelInfo,
    StepInfo,
    TokenInfo,
)
from src.generation.parallel_generator import ParallelGenerator


@pytest.fixture
def mock_generator_components():
    """Create mocks for model, tokenizer, and generator."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_generator = MagicMock(spec=ParallelGenerator)

    # Setup mock generator.generate to return a valid dictionary
    mock_generator.generate.return_value = {
        "generated_text": "[Mock] This is a test response",
        "raw_generated_text": "This is a test response",
        "token_sets": [
            (
                0,  # logical_step
                ([101, 2023], [0.9, 0.8]),  # original (ids, probs)
                ([2023], [0.8]),  # removed (ids, probs)
            )
        ],
        "all_original_token_sets": {0: [(101, 0.9), (2023, 0.8)]},
        "all_surviving_token_sets": {0: [(101, 0.9)]},
        "position_to_tokens": {"1": [" test"], "2": [" response"]},
        "original_parallel_positions": {1},
        "generation_time": 0.45,
        "removal_time": 0.05,
        "is_qwen_model": False,
        "had_repetition_loop": False,
        "logical_layout": [(0, 0, 5), (1, 6, 6), (2, 7, 7)],
    }

    # Mock tokenizer decode
    def mock_decode(token_ids, skip_special_tokens=True):
        return " ".join([f"tok{tid}" for tid in token_ids])

    mock_tokenizer.decode = mock_decode

    # Set device attribute for health check
    mock_generator.device = "cpu"

    return mock_model, mock_tokenizer, mock_generator


@pytest.fixture
def client(mock_generator_components):
    """Create a test client with mocked ModelSingleton."""
    mock_model, mock_tokenizer, mock_generator = mock_generator_components

    with patch(
        "api.ModelSingleton.get_instance",
        return_value=(mock_model, mock_tokenizer, mock_generator),
    ):
        ModelSingleton.initialized = True
        yield TestClient(app)
    ModelSingleton.initialized = False


class TestAPI:
    """Integration tests for the API."""

    def test_root_endpoint(self, client):
        """Test the root endpoint returns a valid response."""
        response = client.get("/api/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert "TEMPO API is running" in response.json()["message"]

    def test_health_check(self, client, mock_generator_components):
        """Test the health check endpoint returns a valid response."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] is True
        assert response.json()["model_name"] == "deepcogito/cogito-v1-preview-llama-3B"
        assert response.json()["device"] == "cpu"

    def test_generate_valid_request(self, client, mock_generator_components):
        """Test text generation with a valid request."""
        mock_model, mock_tokenizer, mock_generator = mock_generator_components

        request_data = {
            "prompt": "This is a test prompt",
            "max_tokens": 50,
            "threshold": 0.1,
            "use_retroactive_removal": True,
            "attention_threshold": 0.05,
            "min_steps": 0,
            "use_custom_rope": True,
            "disable_kv_cache": False,
            "show_token_ids": False,
            "enable_thinking": False,
            "debug_mode": False,
            "allow_intraset_token_visibility": False,
            "no_preserve_isolated_tokens": False,
            "no_relative_attention": False,
            "relative_threshold": 0.5,
            "no_multi_scale_attention": False,
            "no_sigmoid_threshold": False,
            "sigmoid_steepness": 10.0,
            "complete_removal_mode": "keep_token",
            "disable_kv_cache_consistency": False,
        }

        response = client.post("/api/generate", json=request_data)
        assert response.status_code == 200, response.text

        response_data = response.json()
        assert "generated_text" in response_data
        assert "raw_generated_text" in response_data
        assert "timing" in response_data
        assert "model_info" in response_data
        assert "steps" in response_data

        assert "generation_time" in response_data["timing"]
        if request_data["use_retroactive_removal"]:
            assert "retroactive_removal" in response_data
            assert (
                response_data["retroactive_removal"]["attention_threshold"]
                == request_data["attention_threshold"]
            )

        mock_generator.generate.assert_called_once()
        args, kwargs = mock_generator.generate.call_args

        assert kwargs["prompt"] == request_data["prompt"]
        assert kwargs["max_tokens"] == request_data["max_tokens"]
        assert kwargs["threshold"] == request_data["threshold"]
        assert (
            kwargs["use_retroactive_removal"] == request_data["use_retroactive_removal"]
        )
        assert kwargs["debug_mode"] == request_data["debug_mode"]
        assert kwargs["retroactive_remover"] is not None

    def test_generate_invalid_request(self, client):
        """Test text generation with invalid parameters."""
        # Test empty prompt
        response = client.post(
            "/api/generate",
            json={
                "prompt": "",
                "max_tokens": 50,
                "threshold": 0.1,
                "use_retroactive_removal": False,
                "debug_mode": False,
            },
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        if isinstance(detail, list):
            detail = detail[0]["msg"]
        assert "prompt cannot be empty" in detail.lower()

        # Test threshold out of range
        response = client.post(
            "/api/generate",
            json={
                "prompt": "Test",
                "max_tokens": 50,
                "threshold": 2.0,
                "use_retroactive_removal": False,
                "debug_mode": False,
            },
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        if isinstance(detail, list):
            detail = detail[0]["msg"]
        assert "input should be less than or equal to 1" in detail.lower()

    def test_generate_with_removal_error(self, client, mock_generator_components):
        """Test error handling when retroactive removal initialization fails."""
        mock_model, mock_tokenizer, mock_generator = mock_generator_components

        # Make generator raise an error during removal setup
        def raise_error(*args, **kwargs):
            raise RuntimeError("Failed to initialize retroactive removal")

        mock_generator.generate.side_effect = raise_error

        request_data = {
            "prompt": "This is a test prompt",
            "max_tokens": 50,
            "threshold": 0.1,
            "use_retroactive_removal": True,
            "attention_threshold": 0.05,
            "min_steps": 0,
            "use_custom_rope": True,
            "disable_kv_cache": False,
            "show_token_ids": False,
            "enable_thinking": False,
            "debug_mode": False,
            "allow_intraset_token_visibility": False,
            "no_preserve_isolated_tokens": False,
            "no_relative_attention": False,
            "relative_threshold": 0.5,
            "no_multi_scale_attention": False,
            "no_sigmoid_threshold": False,
            "sigmoid_steepness": 10.0,
            "complete_removal_mode": "keep_token",
            "disable_kv_cache_consistency": False,
        }

        response = client.post("/api/generate", json=request_data)
        assert response.status_code == 500
        assert "Failed to initialize retroactive removal" in response.json()["detail"]

    def test_generate_with_cuda_error(self, client, mock_generator_components):
        """Test handling of CUDA/MPS out of memory errors."""
        mock_model, mock_tokenizer, mock_generator = mock_generator_components

        # Make generator raise OOM error
        def raise_oom_error(*args, **kwargs):
            if torch.backends.mps.is_available():
                raise RuntimeError("MPS out of memory")
            else:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        mock_generator.generate.side_effect = raise_oom_error

        request_data = {
            "prompt": "This is a test prompt",
            "max_tokens": 50,
            "threshold": 0.1,
            "min_steps": 0,
            "use_retroactive_removal": True,
            "attention_threshold": 0.05,
            "use_custom_rope": True,
            "disable_kv_cache": False,
            "show_token_ids": False,
            "enable_thinking": False,
            "debug_mode": False,
            "allow_intraset_token_visibility": False,
            "no_preserve_isolated_tokens": False,
            "no_relative_attention": False,
            "relative_threshold": 0.5,
            "no_multi_scale_attention": False,
            "no_sigmoid_threshold": False,
            "sigmoid_steepness": 10.0,
            "complete_removal_mode": "keep_token",
            "disable_kv_cache_consistency": False,
        }

        response = client.post("/api/generate", json=request_data)
        assert response.status_code == 500
        error_detail = response.json()["detail"]
        assert any(
            msg in error_detail
            for msg in [
                "Generation failed: CUDA out of memory",
                "Generation failed: MPS out of memory",
            ]
        )

    def test_generate_with_value_error(self, client, mock_generator_components):
        """Test handling of ValueError during generation."""
        mock_model, mock_tokenizer, mock_generator = mock_generator_components

        # Make generator raise ValueError
        def raise_value_error(*args, **kwargs):
            raise ValueError("Invalid parameter combination")

        mock_generator.generate.side_effect = raise_value_error

        request_data = {
            "prompt": "This is a test prompt",
            "max_tokens": 50,
            "threshold": 0.1,
            "use_retroactive_removal": True,
            "attention_threshold": 0.05,
            "min_steps": 0,
            "use_custom_rope": True,
            "disable_kv_cache": False,
            "show_token_ids": False,
            "enable_thinking": False,
            "debug_mode": False,
            "allow_intraset_token_visibility": False,
            "no_preserve_isolated_tokens": False,
            "no_relative_attention": False,
            "relative_threshold": 0.5,
            "no_multi_scale_attention": False,
            "no_sigmoid_threshold": False,
            "sigmoid_steepness": 10.0,
            "complete_removal_mode": "keep_token",
            "disable_kv_cache_consistency": False,
        }

        response = client.post("/api/generate", json=request_data)
        assert response.status_code == 500
        assert (
            "Invalid generation parameters: Invalid parameter combination"
            in response.json()["detail"]
        )