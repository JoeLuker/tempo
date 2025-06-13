"""Integration tests for the monadic API."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

from src.domain.monads import Ok, Err
from src.presentation.api.models.responses import GenerationResponse


@pytest.fixture
def mock_service():
    """Create a mock monadic generation service."""
    with patch('api_monadic.MonadicGenerationService') as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def client(mock_service):
    """Create test client with mocked service."""
    from api_monadic import app, ServiceSingleton
    
    # Set the mock service
    ServiceSingleton.service = mock_service
    
    return TestClient(app)


class TestMonadicAPIEndpoints:
    """Integration tests for monadic API endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["service"] == "TEMPO Monadic API"
    
    def test_health_check_success(self, client):
        """Test health check when service is available."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_health_check_failure(self, client):
        """Test health check when service initialization fails."""
        with patch('api_monadic.ServiceSingleton.get_service') as mock_get:
            mock_get.return_value = Err("Service initialization failed")
            
            response = client.get("/health")
            assert response.status_code == 503
            assert "Service unavailable" in response.json()["detail"]
    
    def test_generate_success(self, client, mock_service):
        """Test successful text generation."""
        # Mock successful generation
        mock_response = GenerationResponse(
            generated_text="Generated text with [token1/token2] options",
            clean_text="Generated text with options",
            raw_generated_text="Generated text with [token1/token2] options",
            token_count=5,
            generation_time=1.5,
            parallel_sets=[{
                "step": 1,
                "tokens": [
                    {"text": "token1", "prob": 0.6},
                    {"text": "token2", "prob": 0.4}
                ]
            }],
            metadata={"model": "test-model", "device": "cpu"}
        )
        
        mock_service.generate_text.return_value = Ok(mock_response)
        
        request_data = {
            "prompt": "Test prompt",
            "max_tokens": 100,
            "selection_threshold": 0.1
        }
        
        response = client.post("/generate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["generated_text"] == mock_response.generated_text
        assert data["clean_text"] == mock_response.clean_text
        assert data["token_count"] == mock_response.token_count
        assert data["generation_time"] == mock_response.generation_time
    
    def test_generate_invalid_request(self, client, mock_service):
        """Test generation with invalid request."""
        request_data = {
            "prompt": "",  # Empty prompt
            "max_tokens": -10,  # Invalid max tokens
            "selection_threshold": 2.0  # Invalid threshold
        }
        
        response = client.post("/generate", json=request_data)
        assert response.status_code == 422  # Pydantic validation error
    
    def test_generate_service_error(self, client, mock_service):
        """Test generation when service returns error."""
        mock_service.generate_text.return_value = Err("Generation failed: Model error")
        
        request_data = {
            "prompt": "Test prompt",
            "max_tokens": 100
        }
        
        response = client.post("/generate", json=request_data)
        assert response.status_code == 400
        assert "Generation failed" in response.json()["detail"]
    
    def test_generate_batch_success(self, client, mock_service):
        """Test successful batch generation."""
        # Mock responses for batch
        mock_responses = [
            GenerationResponse(
                generated_text=f"Response {i}",
                clean_text=f"Response {i}",
                raw_generated_text=f"Response {i}",
                token_count=10,
                generation_time=1.0,
                parallel_sets=[],
                metadata={}
            )
            for i in range(3)
        ]
        
        mock_service.generate_text.side_effect = [Ok(resp) for resp in mock_responses]
        
        request_data = [
            {"prompt": f"Prompt {i}", "max_tokens": 50}
            for i in range(3)
        ]
        
        response = client.post("/generate/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 3
        for i, item in enumerate(data):
            assert item["generated_text"] == f"Response {i}"
    
    def test_generate_batch_validation_error(self, client):
        """Test batch generation with validation errors."""
        request_data = [
            {"prompt": "Valid prompt", "max_tokens": 50},
            {"prompt": "", "max_tokens": 50},  # Invalid
            {"prompt": "Another valid", "max_tokens": 50}
        ]
        
        response = client.post("/generate/batch", json=request_data)
        assert response.status_code == 422  # Pydantic validation
    
    def test_generate_batch_partial_failure(self, client, mock_service):
        """Test batch generation with partial failures."""
        # First succeeds, second fails
        mock_service.generate_text.side_effect = [
            Ok(GenerationResponse(
                generated_text="Success",
                clean_text="Success",
                raw_generated_text="Success",
                token_count=5,
                generation_time=1.0,
                parallel_sets=[],
                metadata={}
            )),
            Err("Second generation failed")
        ]
        
        request_data = [
            {"prompt": "Prompt 1", "max_tokens": 50},
            {"prompt": "Prompt 2", "max_tokens": 50}
        ]
        
        response = client.post("/generate/batch", json=request_data)
        assert response.status_code == 500
        assert "Generation failed" in response.json()["detail"]
    
    def test_generate_stream_success(self, client, mock_service):
        """Test streaming generation endpoint."""
        mock_response = GenerationResponse(
            generated_text="Hello world test",
            clean_text="Hello world test",
            raw_generated_text="Hello world test",
            token_count=3,
            generation_time=1.0,
            parallel_sets=[],
            metadata={}
        )
        
        mock_service.generate_text.return_value = Ok(mock_response)
        
        request_data = {
            "prompt": "Test prompt",
            "max_tokens": 50
        }
        
        response = client.post("/generate/stream", json=request_data, stream=True)
        assert response.status_code == 200
        
        # Collect streamed events
        events = []
        for line in response.iter_lines():
            if line and line.startswith("data: "):
                event_data = json.loads(line[6:])  # Skip "data: " prefix
                events.append(event_data)
        
        # Verify stream structure
        assert events[0]["status"] == "started"
        
        # Check tokens are streamed
        token_events = [e for e in events if "token" in e]
        assert len(token_events) == 3  # "Hello", "world", "test"
        
        # Check completion
        completion_events = [e for e in events if e.get("status") == "completed"]
        assert len(completion_events) == 1
        assert "full_response" in completion_events[0]
    
    def test_generate_stream_error(self, client, mock_service):
        """Test streaming with generation error."""
        mock_service.generate_text.return_value = Err("Stream generation failed")
        
        request_data = {
            "prompt": "Test prompt",
            "max_tokens": 50
        }
        
        response = client.post("/generate/stream", json=request_data, stream=True)
        assert response.status_code == 200
        
        # Collect error event
        events = []
        for line in response.iter_lines():
            if line and line.startswith("data: "):
                event_data = json.loads(line[6:])
                events.append(event_data)
        
        # Should have error event
        error_events = [e for e in events if "error" in e]
        assert len(error_events) == 1
        assert "Stream generation failed" in error_events[0]["error"]
    
    def test_analyze_endpoint(self, client, mock_service):
        """Test the analyze generation endpoint."""
        request_data = {
            "prompt": "Test prompt",
            "max_tokens": 150,
            "selection_threshold": 0.15,
            "use_retroactive_removal": True
        }
        
        response = client.post("/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "threshold_analysis" in data
        assert "tokens_analysis" in data
        assert "recommendations" in data
        
        # Check specific recommendations
        assert "Selective - balanced creativity and coherence" in data["threshold_analysis"]
        assert "Medium generation" in data["tokens_analysis"]
    
    def test_exception_handler(self, client):
        """Test general exception handling."""
        with patch('api_monadic.ServiceSingleton.get_service') as mock_get:
            mock_get.side_effect = RuntimeError("Unexpected error")
            
            response = client.get("/health")
            assert response.status_code == 500
            
            # In debug mode, would include traceback
            data = response.json()
            assert "error" in data
            assert data["details"]["type"] == "RuntimeError"


class TestMonadicAPIUtilities:
    """Tests for API utility decorators and functions."""
    
    @pytest.mark.asyncio
    async def test_with_timeout_decorator(self):
        """Test timeout decorator."""
        from api_monadic import with_timeout
        import asyncio
        
        @with_timeout(0.1)
        async def slow_function():
            await asyncio.sleep(1.0)
            return "completed"
        
        with pytest.raises(Exception) as exc_info:
            await slow_function()
        
        assert "504" in str(exc_info.value)
        assert "timeout" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_with_rate_limit_decorator(self):
        """Test rate limiting decorator."""
        from api_monadic import with_rate_limit
        from unittest.mock import Mock
        
        call_count = [0]
        
        @with_rate_limit(max_calls=2, window_seconds=1.0)
        async def limited_function(request):
            call_count[0] += 1
            return "success"
        
        # Mock request with client info
        mock_request = Mock()
        mock_request.client.host = "127.0.0.1"
        
        # First two calls should succeed
        assert await limited_function(mock_request) == "success"
        assert await limited_function(mock_request) == "success"
        
        # Third call should be rate limited
        with pytest.raises(Exception) as exc_info:
            await limited_function(mock_request)
        
        assert "429" in str(exc_info.value)
        assert "Rate limit exceeded" in str(exc_info.value)