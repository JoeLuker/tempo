import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import torch
import json
from api import app, ModelSingleton


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_model_singleton():
    """Create a mock for the ModelSingleton class."""
    # Save original get_instance method to restore later
    original_get_instance = ModelSingleton.get_instance

    # Create mocks for model components
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_generator = MagicMock()

    # Setup mock generator.generate to return valid response
    mock_generator.generate.return_value = {
        "generated_text": "This is a test response",
        "raw_generated_text": "This is a test response",
        "token_sets": [
            (
                0,
                (torch.tensor([101, 2023]), torch.tensor([0.9, 0.8])),
                (torch.tensor([101]), torch.tensor([0.9])),
            )
        ],
        "generation_time": 0.5,
        "pruning_time": 0.1,
        "is_qwen_model": False,
    }

    # Mock the get_instance method
    ModelSingleton.get_instance = MagicMock(
        return_value=(mock_model, mock_tokenizer, mock_generator)
    )
    ModelSingleton.initialized = True
    ModelSingleton.retroactive_pruner = MagicMock()

    yield (mock_model, mock_tokenizer, mock_generator)

    # Restore original method
    ModelSingleton.get_instance = original_get_instance


class TestAPI:
    """Integration tests for the API."""

    def test_root_endpoint(self, client):
        """Test the root endpoint returns a valid response."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert "TEMPO API is running" in response.json()["message"]

    def test_health_check(self, client, mock_model_singleton):
        """Test the health check endpoint returns a valid response."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] is True

    def test_generate_valid_request(self, client, mock_model_singleton):
        """Test text generation with a valid request."""
        mock_model, mock_tokenizer, mock_generator = mock_model_singleton

        # Setup mock token decoding
        mock_tokenizer.decode.return_value = "token"

        # Prepare valid request
        request_data = {
            "prompt": "This is a test prompt",
            "max_tokens": 50,
            "threshold": 0.1,
            "use_pruning": True,
            "min_steps": 0,
        }

        # Send request
        response = client.post("/generate", json=request_data)

        # Check response
        assert response.status_code == 200
        assert "generated_text" in response.json()
        assert "raw_generated_text" in response.json()
        assert "timing" in response.json()

        # Check that generator was called with correct parameters
        mock_generator.generate.assert_called_once()
        args, kwargs = mock_generator.generate.call_args

        assert kwargs["prompt"] == request_data["prompt"]
        assert kwargs["max_tokens"] == request_data["max_tokens"]
        assert kwargs["threshold"] == request_data["threshold"]
        assert kwargs["use_pruning"] == request_data["use_pruning"]

    def test_generate_invalid_request(self, client):
        """Test text generation with invalid parameters."""
        # Prepare invalid request - empty prompt
        request_data = {
            "prompt": "",  # Empty prompt
            "max_tokens": 50,
            "threshold": 0.1,
        }

        # Send request
        response = client.post("/generate", json=request_data)

        # Check that request was rejected
        assert response.status_code == 422  # Unprocessable Entity

        # Prepare another invalid request - threshold out of range
        request_data = {
            "prompt": "This is a test prompt",
            "max_tokens": 50,
            "threshold": 2.0,  # Out of range
        }

        # Send request
        response = client.post("/generate", json=request_data)

        # Check that request was rejected
        assert response.status_code == 422  # Unprocessable Entity

    def test_generate_with_retroactive_pruning(self, client, mock_model_singleton):
        """Test text generation with retroactive pruning enabled."""
        mock_model, mock_tokenizer, mock_generator = mock_model_singleton

        # Setup mock token decoding
        mock_tokenizer.decode.return_value = "token"

        # Prepare request with retroactive pruning
        request_data = {
            "prompt": "This is a test prompt",
            "max_tokens": 50,
            "threshold": 0.1,
            "use_pruning": True,
            "use_retroactive_pruning": True,
            "attention_threshold": 0.05,
        }

        # Send request
        response = client.post("/generate", json=request_data)

        # Check response
        assert response.status_code == 200

        # Check that retroactive pruner was used
        mock_generator.generate.assert_called_once()
        args, kwargs = mock_generator.generate.call_args

        assert "retroactive_pruner" in kwargs
        assert kwargs["retroactive_pruner"] is not None

    def test_generate_with_error(self, client, mock_model_singleton):
        """Test error handling during generation."""
        mock_model, mock_tokenizer, mock_generator = mock_model_singleton

        # Make generator raise an exception
        mock_generator.generate.side_effect = RuntimeError("Test error")

        # Prepare request
        request_data = {
            "prompt": "This is a test prompt",
            "max_tokens": 50,
            "threshold": 0.1,
        }

        # Send request
        response = client.post("/generate", json=request_data)

        # Check that error response was returned
        assert response.status_code == 500
        assert "detail" in response.json()
        assert "Test error" in response.json()["detail"]
