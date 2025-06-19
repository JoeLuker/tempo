# TEMPO Development Guide

This guide provides comprehensive information for developers working on or extending TEMPO.

## Development Environment Setup

### Prerequisites

- Python 3.8+ (3.10 recommended)
- Node.js 16+ (for frontend)
- Git
- Virtual environment tool (venv, conda, etc.)
- 16GB+ RAM
- CUDA-capable GPU (optional but recommended)

### Initial Setup

1. **Clone the repository:**
```bash
git clone https://github.com/JoeLuker/tempo.git
cd tempo
```

2. **Create and activate virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-test.txt

# Install package in development mode
pip install -e .
```

4. **Setup pre-commit hooks:**
```bash
pip install pre-commit
pre-commit install
```

5. **Frontend setup:**
```bash
cd frontend
npm install
cd ..
```

## Code Structure

### Project Layout

```
tempo/
├── src/                    # Core Python implementation
│   ├── domain/            # Business logic & interfaces
│   ├── application/       # Use cases & services
│   ├── infrastructure/    # External implementations
│   └── presentation/      # API layer
├── frontend/              # Svelte web interface
├── tests/                 # Test suites
├── docs/                  # Documentation
├── examples/              # Example scripts
└── scripts/               # Utility scripts
```

### Key Modules

#### Core Generation Pipeline
- `parallel_generator.py` - Main generation orchestrator
- `token_generator.py` - Token-level generation logic
- `rope_modifier.py` - Position embedding modifications
- `attention_manager.py` - Attention mask management

#### Pruning System
- `retroactive_pruner.py` - Attention-based pruning
- `dynamic_threshold.py` - Threshold adjustment curves

#### Model Integration
- `model_wrapper.py` - Unified model interface
- `model_adapter.py` - Architecture-specific adapters

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
# ... edit files ...

# Run tests
python run_tests.py

# Run linting
black src/
isort src/
flake8 src/

# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push to remote
git push origin feature/your-feature-name
```

### 2. Testing

#### Unit Tests
```bash
# Run all unit tests
python run_tests.py --unit-only

# Run specific test file
pytest tests/unit/test_token_generator.py

# Run with coverage
pytest --cov=src --cov-report=html
```

#### Integration Tests
```bash
# Run integration tests
python run_tests.py --integration-only

# Run with real model (slow)
pytest tests/integration/test_api.py --use-real-model
```

#### Frontend Tests
```bash
cd frontend

# Run unit tests
npm run test:unit

# Run E2E tests
npm run test:e2e

# Run with UI
npm run test:ui
```

### 3. Debugging

#### Enable Debug Logging
```python
# In code
from src.utils import config
config.debug.global_debug = True
config.debug.module_debug["token_generator"] = True

# Via environment
export TEMPO_DEBUG_GLOBAL_DEBUG=true
export TEMPO_DEBUG_MODULE_DEBUG_TOKEN_GENERATOR=true
```

#### Using Debugger
```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use IDE debugger with launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug TEMPO",
            "type": "python",
            "request": "launch",
            "program": "run_tempo.py",
            "args": ["--prompt", "Test", "--debug-mode"],
            "console": "integratedTerminal"
        }
    ]
}
```

#### Memory Profiling
```bash
# Install memory profiler
pip install memory_profiler

# Run with memory profiling
python -m memory_profiler run_tempo.py --prompt "Test"
```

## Coding Standards

### Python Style Guide

Follow PEP 8 with these additions:

1. **Line length**: 88 characters (Black default)
2. **Imports**: Use `isort` for ordering
3. **Type hints**: Required for public functions
4. **Docstrings**: Google style for all public APIs

Example:
```python
from typing import List, Optional, Tuple

from src.domain.entities import Token
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TokenProcessor:
    """Processes tokens for parallel generation.
    
    This class handles the core logic for processing multiple token
    candidates in parallel, including selection and pruning.
    
    Attributes:
        threshold: Minimum probability for token selection.
        max_tokens: Maximum number of parallel tokens.
    """
    
    def __init__(self, threshold: float = 0.1, max_tokens: int = 5) -> None:
        """Initialize the token processor.
        
        Args:
            threshold: Minimum probability threshold (0.0-1.0).
            max_tokens: Maximum parallel tokens per position.
            
        Raises:
            ValueError: If threshold is not in range [0, 1].
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
            
        self.threshold = threshold
        self.max_tokens = max_tokens
        logger.debug(f"Initialized with threshold={threshold}")
    
    def process(
        self, 
        logits: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[List[Token], float]:
        """Process logits to select token candidates.
        
        Args:
            logits: Raw model output logits.
            temperature: Sampling temperature.
            
        Returns:
            A tuple of (selected_tokens, total_probability).
            
        Example:
            >>> processor = TokenProcessor(threshold=0.1)
            >>> tokens, prob = processor.process(logits)
            >>> print(f"Selected {len(tokens)} tokens")
        """
        # Implementation here
        pass
```

### Frontend Style Guide

Follow Svelte and TypeScript best practices:

```typescript
// Use TypeScript interfaces
interface GenerationParams {
  prompt: string;
  maxTokens: number;
  selectionThreshold: number;
}

// Component example
<script lang="ts">
  import { onMount } from 'svelte';
  import type { GenerationResult } from '$lib/types';
  
  export let params: GenerationParams;
  
  let result: GenerationResult | null = null;
  let loading = false;
  
  async function generate(): Promise<void> {
    loading = true;
    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      result = await response.json();
    } catch (error) {
      console.error('Generation failed:', error);
    } finally {
      loading = false;
    }
  }
  
  onMount(() => {
    // Component initialization
  });
</script>
```

## Adding New Features

### 1. Adding a New Pruning Strategy

```python
# src/domain/interfaces/pruning_strategy.py
from abc import ABC, abstractmethod
from typing import List

from src.domain.entities import Token, GenerationContext


class IPruningStrategy(ABC):
    """Interface for token pruning strategies."""
    
    @abstractmethod
    def prune(
        self, 
        tokens: List[Token], 
        context: GenerationContext
    ) -> List[Token]:
        """Prune tokens based on strategy logic."""
        pass


# src/pruning/my_pruning_strategy.py
class MyPruningStrategy(IPruningStrategy):
    """Custom pruning implementation."""
    
    def __init__(self, config: dict):
        self.config = config
    
    def prune(
        self, 
        tokens: List[Token], 
        context: GenerationContext
    ) -> List[Token]:
        # Implement pruning logic
        return pruned_tokens


# Register in parallel_generator.py
pruning_strategies = {
    "my_strategy": MyPruningStrategy,
    # ... other strategies
}
```

### 2. Adding a New Model Adapter

```python
# src/infrastructure/models/my_model_adapter.py
from src.infrastructure.models.model_adapter import ModelAdapter


class MyModelAdapter(ModelAdapter):
    """Adapter for MyModel architecture."""
    
    @staticmethod
    def detect_architecture(model) -> bool:
        """Check if model is MyModel type."""
        return hasattr(model, "my_model_specific_attribute")
    
    def install_rope_modifier(self, model, modifier):
        """Install RoPE modifier for MyModel."""
        # Implementation specific to MyModel
        pass
    
    def get_attention_layers(self, model):
        """Get attention layers for MyModel."""
        # Return list of attention modules
        pass


# Register in model_wrapper.py
ADAPTERS = [
    MyModelAdapter,
    # ... other adapters
]
```

### 3. Adding API Endpoints

```python
# src/presentation/api/routes/my_endpoint.py
from fastapi import APIRouter, Depends
from src.application.services import MyService

router = APIRouter(prefix="/my-feature", tags=["my-feature"])


@router.post("/process")
async def process_data(
    data: MyRequestModel,
    service: MyService = Depends(get_my_service)
):
    """Process data with my feature."""
    result = await service.process(data)
    return MyResponseModel(result=result)


# Register in api.py
from src.presentation.api.routes import my_endpoint
app.include_router(my_endpoint.router)
```

## Testing Best Practices

### 1. Unit Test Structure

```python
# tests/unit/test_my_feature.py
import pytest
from unittest.mock import Mock, patch

from src.my_module import MyClass


class TestMyClass:
    """Test suite for MyClass."""
    
    @pytest.fixture
    def instance(self):
        """Create instance for testing."""
        return MyClass(config={"key": "value"})
    
    def test_initialization(self, instance):
        """Test proper initialization."""
        assert instance.config["key"] == "value"
    
    def test_process_valid_input(self, instance):
        """Test processing with valid input."""
        result = instance.process("valid input")
        assert result.success
        assert len(result.data) > 0
    
    def test_process_invalid_input(self, instance):
        """Test processing with invalid input."""
        with pytest.raises(ValueError):
            instance.process("")
    
    @patch("src.my_module.external_function")
    def test_with_mock(self, mock_func, instance):
        """Test with mocked external dependency."""
        mock_func.return_value = "mocked result"
        result = instance.process_with_external()
        assert result == "mocked result"
        mock_func.assert_called_once()
```

### 2. Integration Test Structure

```python
# tests/integration/test_generation_flow.py
import pytest
from src.application.services import GenerationService
from src.infrastructure.models import ModelRepository


@pytest.mark.integration
class TestGenerationFlow:
    """Integration tests for generation flow."""
    
    @pytest.fixture
    def service(self):
        """Create service with real dependencies."""
        repo = ModelRepository()
        return GenerationService(repo)
    
    def test_end_to_end_generation(self, service):
        """Test complete generation flow."""
        request = GenerationRequest(
            prompt="Test prompt",
            max_tokens=50,
            selection_threshold=0.1
        )
        
        result = service.generate(request)
        
        assert result.success
        assert len(result.generated_text) > 0
        assert result.token_count <= 50
```

## Performance Optimization

### 1. Profiling

```bash
# CPU profiling
python -m cProfile -o profile.stats run_tempo.py --prompt "Test"
python -m pstats profile.stats

# Line profiling
pip install line_profiler
kernprof -l -v run_tempo.py

# Memory profiling
pip install memory_profiler
python -m memory_profiler run_tempo.py
```

### 2. Optimization Techniques

```python
# Use torch.no_grad() for inference
with torch.no_grad():
    logits = model(input_ids)

# Reuse tensors when possible
buffer = torch.empty(size, device=device)
# Reuse buffer instead of creating new tensors

# Batch operations
tokens = torch.stack(token_list)  # Process together
results = model(tokens)  # Single forward pass

# Cache expensive computations
@lru_cache(maxsize=128)
def expensive_computation(key):
    return compute_result(key)
```

## Release Process

### 1. Version Bumping

```bash
# Update version in src/__init__.py
__version__ = "0.2.0"

# Update CHANGELOG.md
# Add release notes

# Commit version bump
git add .
git commit -m "chore: bump version to 0.2.0"
```

### 2. Creating Release

```bash
# Create and push tag
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0

# Build distribution
python -m build

# Upload to PyPI (if applicable)
python -m twine upload dist/*
```

### 3. Post-Release

1. Create GitHub release with changelog
2. Update documentation
3. Announce in relevant channels
4. Monitor for issues

## Troubleshooting Development Issues

### Common Issues

1. **Import errors**: Ensure package is installed with `pip install -e .`
2. **Model loading fails**: Check CUDA/GPU availability
3. **Tests fail**: Ensure test dependencies are installed
4. **Frontend build fails**: Check Node.js version

### Debug Commands

```bash
# Check Python environment
python -c "import sys; print(sys.path)"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check installed packages
pip list | grep -E "torch|transformers"

# Clear caches
rm -rf .cache/
rm -rf __pycache__/
rm -rf src/__pycache__/
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.

### Quick Contribution Checklist

- [ ] Fork and clone repository
- [ ] Create feature branch
- [ ] Write tests for new functionality
- [ ] Ensure all tests pass
- [ ] Add documentation
- [ ] Run linting tools
- [ ] Submit pull request
- [ ] Respond to code review feedback

## Resources

- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Best Practices](https://pytorch.org/tutorials/beginner/best_practices.html)
- [Svelte Tutorial](https://svelte.dev/tutorial)
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)