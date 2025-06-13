"""Unit tests for monadic generation service."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import time

from src.domain.monads import Result, Ok, Err
from src.application.services.monadic_generation_service import (
    MonadicGenerationService,
    GenerationDependencies,
    GenerationContext,
    validate_request,
    with_retry,
    log_result
)
from src.presentation.api.models.requests import GenerationRequest
from src.presentation.api.models.responses import GenerationResponse


class TestMonadicGenerationService:
    """Tests for the monadic generation service."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        deps = GenerationDependencies(
            model_repository=Mock(),
            response_formatter=Mock(),
            debug_mode=False
        )
        
        # Mock model components
        model_wrapper = Mock()
        tokenizer = Mock()
        generator = Mock()
        token_generator = Mock()
        
        deps.model_repository.get_model_components.return_value = (
            model_wrapper, tokenizer, generator, token_generator
        )
        
        return deps
    
    @pytest.fixture
    def valid_request(self):
        """Create a valid generation request."""
        return GenerationRequest(
            prompt="Test prompt",
            max_tokens=100,
            selection_threshold=0.1,
            use_retroactive_removal=False,
            debug_mode=False
        )
    
    def test_validate_request_success(self, valid_request):
        """Test successful request validation."""
        result = validate_request(valid_request)
        assert result.is_ok()
        assert result.unwrap() == valid_request
    
    def test_validate_request_empty_prompt(self):
        """Test validation with empty prompt."""
        request = GenerationRequest(prompt="", max_tokens=100)
        result = validate_request(request)
        assert result.is_err()
        assert "Prompt cannot be empty" in result.unwrap_err()
    
    def test_validate_request_invalid_max_tokens(self):
        """Test validation with invalid max tokens."""
        request = GenerationRequest(prompt="Test", max_tokens=-10)
        result = validate_request(request)
        assert result.is_err()
        assert "Max tokens must be positive" in result.unwrap_err()
    
    def test_validate_request_invalid_threshold(self):
        """Test validation with invalid selection threshold."""
        request = GenerationRequest(prompt="Test", selection_threshold=1.5)
        result = validate_request(request)
        assert result.is_err()
        assert "Selection threshold must be between 0 and 1" in result.unwrap_err()
    
    @patch('src.application.services.monadic_generation_service.logger')
    def test_create_context_success(self, mock_logger, mock_dependencies, valid_request):
        """Test successful context creation."""
        service = MonadicGenerationService()
        service.model_repository = mock_dependencies.model_repository
        
        context_reader = service._create_context(valid_request)
        result = context_reader.run(mock_dependencies)
        
        assert result.is_ok()
        context = result.unwrap()
        assert isinstance(context, GenerationContext)
        assert context.request == valid_request
        assert context.retroactive_remover is None
        assert context.system_content is None
    
    def test_create_context_failure(self, mock_dependencies, valid_request):
        """Test context creation failure."""
        service = MonadicGenerationService()
        mock_dependencies.model_repository.get_model_components.side_effect = Exception("Model load failed")
        
        context_reader = service._create_context(valid_request)
        result = context_reader.run(mock_dependencies)
        
        assert result.is_err()
        assert "Failed to create context" in result.unwrap_err()
    
    def test_set_debug_mode(self, mock_dependencies, valid_request):
        """Test debug mode setting."""
        service = MonadicGenerationService()
        
        # Create context with mocked components
        context = GenerationContext(
            request=valid_request,
            model_wrapper=Mock(),
            tokenizer=Mock(),
            generator=Mock(),
            token_generator=Mock(),
            retroactive_remover=None,
            system_content=None,
            start_time=time.time()
        )
        
        debug_reader = service._set_debug_mode(context)
        result = debug_reader.run(mock_dependencies)
        
        assert result.is_ok()
        context.token_generator.set_debug_mode.assert_called_once_with(False)
    
    def test_prepare_system_content_default(self, mock_dependencies, valid_request):
        """Test system content preparation without enable_thinking."""
        service = MonadicGenerationService()
        
        context = GenerationContext(
            request=valid_request,
            model_wrapper=Mock(),
            tokenizer=Mock(),
            generator=Mock(),
            token_generator=Mock(),
            retroactive_remover=None,
            system_content=None,
            start_time=time.time()
        )
        
        content_reader = service._prepare_system_content(context)
        result = content_reader.run(mock_dependencies)
        
        assert result.is_ok()
        assert result.unwrap().system_content is None
    
    def test_prepare_system_content_with_thinking(self, mock_dependencies):
        """Test system content preparation with enable_thinking."""
        service = MonadicGenerationService()
        
        request = GenerationRequest(
            prompt="Test",
            enable_thinking=True,
            system_content=None
        )
        
        context = GenerationContext(
            request=request,
            model_wrapper=Mock(),
            tokenizer=Mock(),
            generator=Mock(),
            token_generator=Mock(),
            retroactive_remover=None,
            system_content=None,
            start_time=time.time()
        )
        
        content_reader = service._prepare_system_content(context)
        result = content_reader.run(mock_dependencies)
        
        assert result.is_ok()
        assert result.unwrap().system_content == "Enable deep thinking subroutine."
    
    @patch('src.application.services.monadic_generation_service.RetroactiveRemover')
    def test_create_retroactive_remover_enabled(self, mock_remover_class, mock_dependencies):
        """Test retroactive remover creation when enabled."""
        service = MonadicGenerationService()
        
        request = GenerationRequest(
            prompt="Test",
            use_retroactive_removal=True,
            attention_threshold=0.02
        )
        
        mock_generator = Mock()
        mock_generator.device = "cpu"
        
        context = GenerationContext(
            request=request,
            model_wrapper=Mock(),
            tokenizer=Mock(),
            generator=mock_generator,
            token_generator=Mock(),
            retroactive_remover=None,
            system_content=None,
            start_time=time.time()
        )
        
        mock_remover = Mock()
        mock_remover_class.return_value = mock_remover
        
        remover_reader = service._create_retroactive_remover(context)
        result = remover_reader.run(mock_dependencies)
        
        assert result.is_ok()
        assert result.unwrap().retroactive_remover == mock_remover
        mock_remover_class.assert_called_once()
    
    def test_perform_generation_success(self, mock_dependencies, valid_request):
        """Test successful generation."""
        service = MonadicGenerationService()
        
        mock_generator = Mock()
        mock_generation_result = {
            'generated_text': 'Test output',
            'num_tokens': 10,
            'parallel_sets': []
        }
        mock_generator.generate.return_value = mock_generation_result
        
        context = GenerationContext(
            request=valid_request,
            model_wrapper=Mock(),
            tokenizer=Mock(),
            generator=mock_generator,
            token_generator=Mock(),
            retroactive_remover=None,
            system_content=None,
            start_time=time.time()
        )
        
        generation_reader = service._perform_generation(context)
        result = generation_reader.run(mock_dependencies)
        
        assert result.is_ok()
        assert result.unwrap().generation_result == mock_generation_result
        mock_generator.generate.assert_called_once()
    
    def test_perform_generation_failure(self, mock_dependencies, valid_request):
        """Test generation failure."""
        service = MonadicGenerationService()
        
        mock_generator = Mock()
        mock_generator.generate.side_effect = Exception("Generation failed")
        
        context = GenerationContext(
            request=valid_request,
            model_wrapper=Mock(),
            tokenizer=Mock(),
            generator=mock_generator,
            token_generator=Mock(),
            retroactive_remover=None,
            system_content=None,
            start_time=time.time()
        )
        
        generation_reader = service._perform_generation(context)
        result = generation_reader.run(mock_dependencies)
        
        assert result.is_err()
        assert "Error during generation" in result.unwrap_err()
    
    def test_format_response_success(self, mock_dependencies, valid_request):
        """Test successful response formatting."""
        service = MonadicGenerationService()
        
        mock_response = GenerationResponse(
            generated_text="Test output",
            clean_text="Test output",
            raw_generated_text="Test output",
            token_count=10,
            generation_time=1.0,
            parallel_sets=[],
            metadata={}
        )
        
        mock_dependencies.response_formatter.format_response.return_value = mock_response
        
        context = GenerationContext(
            request=valid_request,
            model_wrapper=Mock(),
            tokenizer=Mock(),
            generator=Mock(device="cpu"),
            token_generator=Mock(),
            retroactive_remover=None,
            system_content=None,
            start_time=time.time() - 1.0,
            generation_result={'generated_text': 'Test output'}
        )
        
        format_reader = service._format_response(context)
        result = format_reader.run(mock_dependencies)
        
        assert result.is_ok()
        assert result.unwrap() == mock_response
    
    @patch('src.application.services.monadic_generation_service.time.time')
    def test_full_generation_pipeline(self, mock_time, mock_dependencies, valid_request):
        """Test the full generation pipeline."""
        # Mock time for consistent timing
        mock_time.side_effect = [1000.0, 1001.0]  # Start and end times
        
        service = MonadicGenerationService()
        service.model_repository = mock_dependencies.model_repository
        service.response_formatter = mock_dependencies.response_formatter
        
        # Setup mocks
        mock_generator = Mock()
        mock_generator.use_custom_rope = False
        mock_generator.device = "cpu"
        mock_generator.generate.return_value = {
            'generated_text': 'Test output',
            'num_tokens': 10
        }
        
        mock_dependencies.model_repository.get_model_components.return_value = (
            Mock(),  # model_wrapper
            Mock(),  # tokenizer
            mock_generator,
            Mock()   # token_generator
        )
        
        mock_response = GenerationResponse(
            generated_text="Test output",
            clean_text="Test output",
            raw_generated_text="Test output",
            token_count=10,
            generation_time=1.0,
            parallel_sets=[],
            metadata={}
        )
        mock_dependencies.response_formatter.format_response.return_value = mock_response
        
        # Run the pipeline
        result = service.generate_text(valid_request)
        
        assert result.is_ok()
        assert result.unwrap() == mock_response
        
        # Verify components were called
        mock_dependencies.model_repository.get_model_components.assert_called_once()
        mock_generator.generate.assert_called_once()
        mock_dependencies.response_formatter.format_response.assert_called_once()


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_with_retry_success_first_try(self):
        """Test retry with immediate success."""
        counter = [0]
        
        def operation():
            counter[0] += 1
            return Ok("success")
        
        result = with_retry(operation, max_retries=3)
        assert result.is_ok()
        assert result.unwrap() == "success"
        assert counter[0] == 1
    
    @patch('time.sleep')
    def test_with_retry_success_after_failures(self, mock_sleep):
        """Test retry with eventual success."""
        counter = [0]
        
        def operation():
            counter[0] += 1
            if counter[0] < 3:
                return Err("failure")
            return Ok("success")
        
        result = with_retry(operation, max_retries=3, delay=0.1)
        assert result.is_ok()
        assert result.unwrap() == "success"
        assert counter[0] == 3
        
        # Check exponential backoff
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(0.1)  # First retry
        mock_sleep.assert_any_call(0.2)  # Second retry (exponential)
    
    def test_with_retry_all_failures(self):
        """Test retry with all failures."""
        counter = [0]
        
        def operation():
            counter[0] += 1
            return Err("failure")
        
        result = with_retry(operation, max_retries=3)
        assert result.is_err()
        assert result.unwrap_err() == "failure"
        assert counter[0] == 3
    
    @patch('src.application.services.monadic_generation_service.logger')
    def test_log_result_success(self, mock_logger):
        """Test logging successful result."""
        result = Ok("success")
        logged = log_result(result)
        
        assert logged == result
        mock_logger.debug.assert_called_once_with("Operation succeeded")
        mock_logger.error.assert_not_called()
    
    @patch('src.application.services.monadic_generation_service.logger')
    def test_log_result_failure(self, mock_logger):
        """Test logging failed result."""
        result = Err("error message")
        logged = log_result(result)
        
        assert logged == result
        mock_logger.error.assert_called_once_with("Operation failed: error message")
        mock_logger.debug.assert_not_called()