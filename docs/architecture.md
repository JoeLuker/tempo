# TEMPO Architecture Documentation

## System Overview

TEMPO (Threshold-Enabled Multipath Parallel Output) implements a novel approach to language model text generation by processing multiple token candidates simultaneously within a single sequence state. This document details the system architecture, component interactions, and design decisions.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │   CLI       │  │   Web UI     │  │    REST API        │    │
│  │(run_tempo.py)│  │  (Svelte)   │  │  (FastAPI)        │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      Application Layer                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐ │
│  │ Generation      │  │ Model Service   │  │ Response       │ │
│  │ Service         │  │                 │  │ Formatter      │ │
│  └─────────────────┘  └─────────────────┘  └────────────────┘ │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                        Domain Layer                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐ │
│  │ Parallel        │  │ Token           │  │ Pruning        │ │
│  │ Generator       │  │ Generator       │  │ Strategies     │ │
│  └─────────────────┘  └─────────────────┘  └────────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐ │
│  │ RoPE            │  │ Attention       │  │ MCTS           │ │
│  │ Modifier        │  │ Manager         │  │ Generator      │ │
│  └─────────────────┘  └─────────────────┘  └────────────────┘ │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                     Infrastructure Layer                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐ │
│  │ Model           │  │ Cache           │  │ Performance    │ │
│  │ Wrapper         │  │ Manager         │  │ Tracker        │ │
│  └─────────────────┘  └─────────────────┘  └────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Parallel Generator (`src/generation/parallel_generator.py`)

The heart of TEMPO's parallel token processing system.

**Responsibilities:**
- Orchestrates the generation process
- Manages logical position tracking
- Coordinates with RoPE modifier and attention manager
- Implements the main generation loop

**Key Methods:**
- `generate()`: Main entry point for text generation
- `_process_step()`: Handles a single generation step
- `_apply_pruning()`: Applies various pruning strategies

### 2. RoPE Modifier (`src/generation/rope_modifier.py`)

Modifies Rotary Position Embeddings to enable parallel token processing.

**Key Innovation:**
- Assigns the same positional embedding to tokens at the same logical position
- Enables the model to process multiple tokens "simultaneously"

**Implementation:**
```python
# Simplified concept
for physical_idx, logical_pos in enumerate(logical_layout):
    position_ids[physical_idx] = logical_pos
```

### 3. Token Generator (`src/generation/token_generator.py`)

Handles the core token generation logic and model interactions.

**Features:**
- Prompt caching for efficiency
- KV cache management
- Support for different model architectures
- Performance tracking

**Key Classes:**
- `TokenGenerator`: Main generation class
- `PromptCache`: Caches tokenized prompts
- `KVCacheManager`: Manages key-value caches

### 4. Attention Manager (`src/generation/attention_manager.py`)

Controls attention patterns between tokens, especially parallel tokens.

**Capabilities:**
- Standard causal masking
- Custom masks for parallel token visibility
- Integration with RoPE modifications

**Attention Patterns:**
```
Standard:  [1 0 0 0]    Parallel:  [1 0 0 0]
           [1 1 0 0]               [1 1 1 0]
           [1 1 1 0]               [1 1 1 0]
           [1 1 1 1]               [1 1 1 1]
```

### 5. Pruning System

Multiple strategies for refining token sets:

#### Retroactive Pruner (`src/pruning/retroactive_pruner.py`)
- Prunes previously processed tokens based on attention from future tokens
- Uses attention patterns to identify less relevant tokens

#### Dynamic Threshold (`src/pruning/dynamic_threshold.py`)
- Adjusts pruning aggressiveness over time
- Supports Bezier curves and ReLU transitions

### 6. Model Wrapper (`src/modeling/model_wrapper.py`)

Provides a unified interface to different model architectures.

**Features:**
- Architecture detection (Qwen, Llama, etc.)
- Custom RoPE installation
- Model-specific optimizations

## Data Flow

### 1. Generation Request Flow

```
User Input
    ↓
Request Validation
    ↓
Model Initialization (if needed)
    ↓
Tokenization & Prompt Caching
    ↓
Generation Loop:
    ├─> Get Logits
    ├─> Apply Selection Threshold
    ├─> Apply Pruning (optional)
    ├─> Update Logical Layout
    ├─> Modify RoPE (if enabled)
    ├─> Update Attention Mask
    └─> Append Tokens
    ↓
Format Response
    ↓
Return to User
```

### 2. Parallel Token Processing

```
Step N: "The"
    ↓
Get next token probabilities
    ↓
Select tokens above threshold:
    ["future": 0.4, "world": 0.3, "next": 0.2]
    ↓
Apply pruning strategies
    ↓
Append all selected tokens: "The future world next"
    ↓
Update logical layout: [0, 1, 1, 1]
    ↓
RoPE assigns same position to parallel tokens
    ↓
Process Step N+1 with parallel context
```

## Extension Points

### 1. Custom Pruning Strategies

Implement the `IPruningStrategy` interface:

```python
class CustomPruningStrategy:
    def prune(self, tokens: List[Token], context: GenerationContext) -> List[Token]:
        # Custom pruning logic
        return pruned_tokens
```

### 2. Model Adapters

Add support for new model architectures:

```python
class NewModelAdapter(ModelAdapter):
    def detect_architecture(self, model) -> str:
        # Detection logic
    
    def install_rope_modifier(self, model, modifier):
        # Model-specific RoPE installation
```

### 3. Generation Strategies

Implement alternative generation approaches:

```python
class CustomGenerationStrategy:
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        # Custom generation logic
```

## Monadic API Design

The system also includes a functional programming approach using monads for better error handling and composition.

### Monad Types Used:
- **Result**: For operations that can fail
- **Maybe**: For optional values
- **Reader**: For dependency injection
- **IO**: For side effects
- **State**: For stateful computations

### Example Pipeline:
```python
generation_pipeline = (
    validate_request(request)
    .flat_map(create_context)
    .flat_map(perform_generation)
    .flat_map(format_response)
)
```

## Performance Considerations

### 1. Memory Management
- KV cache reuse for efficiency
- Prompt caching to avoid re-tokenization
- Careful tensor management to prevent memory leaks

### 2. Computational Optimization
- Batch processing where possible
- Early exit conditions
- Efficient attention mask computation

### 3. Scalability
- Stateless API design
- Model singleton for resource sharing
- Configurable concurrency limits

## Configuration System

Hierarchical configuration with multiple sources:

1. Default values in code
2. Configuration file (`config.json`)
3. Environment variables (`TEMPO_*`)
4. Runtime parameters

Example:
```python
# Priority: Runtime > Environment > File > Default
config.model.device = args.device or 
                     os.getenv("TEMPO_MODEL_DEVICE") or 
                     config_file.get("device") or 
                     "cuda"
```

## Testing Architecture

### Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Focus on business logic

### Integration Tests
- Test component interactions
- Use real models for critical paths
- Validate end-to-end flows

### Performance Tests
- Profile generation speed
- Monitor memory usage
- Benchmark against baselines

## Deployment Considerations

### Containerization
- Separate containers for API and model
- Volume mounts for model cache
- Environment-based configuration

### Monitoring
- Structured logging with levels
- Performance metrics collection
- Health check endpoints

### Scaling
- Horizontal scaling for API layer
- Model server pooling
- Queue-based request handling

## Future Architecture Enhancements

1. **Plugin System**: Dynamic loading of custom components
2. **Distributed Generation**: Multi-GPU/multi-node support
3. **Stream Processing**: Real-time token streaming
4. **Model Zoo**: Easy switching between models
5. **Caching Layer**: Redis/Memcached integration
6. **Metrics Dashboard**: Prometheus/Grafana integration

## Contributing to Architecture

When adding new features:

1. Follow the layered architecture pattern
2. Maintain clear interfaces between components
3. Add appropriate tests
4. Update this documentation
5. Consider backward compatibility

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.