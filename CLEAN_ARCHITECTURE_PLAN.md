# TEMPO Clean Architecture Refactoring Plan

## Overview

This document outlines a comprehensive plan to refactor the TEMPO codebase following clean architecture principles. The refactoring aims to break up large files, establish clear boundaries between layers, and ensure each module has a single, well-defined responsibility.

## Architecture Layers

### 1. Domain Layer (Core Business Logic)
**Location**: `src/domain/`
- No external dependencies
- Pure Python classes and interfaces
- Business rules and entities

### 2. Application Layer (Use Cases)
**Location**: `src/application/`
- Orchestrates domain objects
- Implements use cases
- Depends only on Domain layer

### 3. Infrastructure Layer (External Systems)
**Location**: `src/infrastructure/`
- Model implementations
- File I/O
- External API integrations
- Depends on Domain and Application layers

### 4. Presentation Layer (User Interface)
**Location**: `src/presentation/` and `api/`
- API endpoints
- Request/response handling
- User interface logic
- Depends on all other layers

## Dependency Flow

```
Presentation → Application → Domain
     ↓              ↓
Infrastructure ←────┘
```

Dependencies flow inward. Domain has no dependencies, Application depends on Domain, Infrastructure and Presentation depend on both Domain and Application.

## File Breakdown Plan

### 1. Breaking up `parallel_generator.py` (987 lines)

#### Domain Layer Components:
- `src/domain/entities/token.py`
  - `Token` class with id and probability
  - `TokenSet` class for managing collections of tokens
  - `LogicalPosition` class for tracking positions

- `src/domain/entities/generation_state.py`
  - `GenerationState` class for tracking generation progress
  - `SequenceLayout` class for logical/physical position mapping

- `src/domain/interfaces/token_selector.py`
  - `ITokenSelector` interface

- `src/domain/interfaces/pruning_strategy.py`
  - `IPruningStrategy` interface

#### Application Layer Components:
- `src/application/use_cases/generate_text.py`
  - `GenerateTextUseCase` class (main generation logic)
  - Orchestrates the generation process

- `src/application/use_cases/apply_mcts.py`
  - `ApplyMCTSUseCase` class for MCTS-based generation

- `src/application/services/threshold_calculator.py`
  - `ThresholdCalculator` for dynamic threshold logic

- `src/application/services/repetition_detector.py`
  - `RepetitionDetector` for detecting text patterns

#### Infrastructure Layer Components:
- `src/infrastructure/generation/parallel_generator_impl.py`
  - Concrete implementation that wires everything together
  - Handles model interactions

### 2. Breaking up `token_generator.py` (1513 lines)

#### Domain Layer Components:
- `src/domain/entities/logits.py`
  - `Logits` value object
  - `LogitDistribution` for probability distributions

- `src/domain/interfaces/model_interface.py`
  - `ILanguageModel` interface for model abstraction

- `src/domain/services/tokenization.py`
  - `ITokenizer` interface

#### Application Layer Components:
- `src/application/services/cache_manager.py`
  - `PromptCache` for caching tokenized prompts
  - `KVCacheManager` for managing key-value caches

- `src/application/services/performance_tracker.py`
  - `PerformanceTracker` for tracking metrics

#### Infrastructure Layer Components:
- `src/infrastructure/models/token_generator_impl.py`
  - Concrete implementation for token generation
  - Model-specific logic

- `src/infrastructure/models/model_adapter.py`
  - Adapters for different model types (Qwen, Llama, etc.)

### 3. Breaking up `api.py` (912 lines)

#### Domain Layer Components:
- `src/domain/entities/generation_request.py`
  - `GenerationRequest` entity with all parameters
  - `GenerationResult` entity for results

#### Application Layer Components:
- `src/application/use_cases/initialize_model.py`
  - `InitializeModelUseCase` for model setup

- `src/application/services/request_validator.py`
  - `RequestValidator` for validating generation requests

- `src/application/services/result_formatter.py`
  - `ResultFormatter` for formatting generation results

#### Presentation Layer Components:
- `src/presentation/api/routes/generation.py`
  - FastAPI route handlers

- `src/presentation/api/middleware/cors.py`
  - CORS configuration

- `src/presentation/api/middleware/error_handler.py`
  - Global error handling

- `src/presentation/api/schemas/requests.py`
  - Pydantic models for API requests

- `src/presentation/api/schemas/responses.py`
  - Pydantic models for API responses

#### Infrastructure Layer Components:
- `src/infrastructure/models/model_singleton.py`
  - Singleton pattern for model management

### 4. Breaking up `attention_manager.py` (1225 lines)

#### Domain Layer Components:
- `src/domain/entities/attention_mask.py`
  - `AttentionMask` value object
  - `AttentionPattern` for different masking patterns

- `src/domain/interfaces/attention_strategy.py`
  - `IAttentionStrategy` interface

#### Application Layer Components:
- `src/application/services/mask_builder.py`
  - `MaskBuilder` for creating attention masks

- `src/application/services/position_coordinator.py`
  - `PositionCoordinator` for coordinating with RoPE

#### Infrastructure Layer Components:
- `src/infrastructure/attention/attention_manager_impl.py`
  - Concrete implementation

- `src/infrastructure/attention/mask_cache.py`
  - `MaskCache` for caching attention masks

## New Module Structure

```
src/
├── domain/
│   ├── __init__.py
│   ├── entities/
│   │   ├── __init__.py
│   │   ├── token.py
│   │   ├── generation_state.py
│   │   ├── logits.py
│   │   ├── attention_mask.py
│   │   └── generation_request.py
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── token_selector.py
│   │   ├── pruning_strategy.py
│   │   ├── model_interface.py
│   │   └── attention_strategy.py
│   └── services/
│       ├── __init__.py
│       └── tokenization.py
│
├── application/
│   ├── __init__.py
│   ├── use_cases/
│   │   ├── __init__.py
│   │   ├── generate_text.py
│   │   ├── apply_mcts.py
│   │   └── initialize_model.py
│   └── services/
│       ├── __init__.py
│       ├── threshold_calculator.py
│       ├── repetition_detector.py
│       ├── cache_manager.py
│       ├── performance_tracker.py
│       ├── request_validator.py
│       ├── result_formatter.py
│       ├── mask_builder.py
│       └── position_coordinator.py
│
├── infrastructure/
│   ├── __init__.py
│   ├── generation/
│   │   ├── __init__.py
│   │   └── parallel_generator_impl.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── token_generator_impl.py
│   │   ├── model_adapter.py
│   │   └── model_singleton.py
│   └── attention/
│       ├── __init__.py
│       ├── attention_manager_impl.py
│       └── mask_cache.py
│
└── presentation/
    ├── __init__.py
    └── api/
        ├── __init__.py
        ├── routes/
        │   ├── __init__.py
        │   └── generation.py
        ├── middleware/
        │   ├── __init__.py
        │   ├── cors.py
        │   └── error_handler.py
        └── schemas/
            ├── __init__.py
            ├── requests.py
            └── responses.py
```

## Migration Strategy

### Phase 1: Create Domain Layer (Week 1)
1. Define all entities and value objects
2. Create interfaces for external dependencies
3. Move pure business logic to domain services

### Phase 2: Build Application Layer (Week 2)
1. Implement use cases by extracting orchestration logic
2. Create application services for cross-cutting concerns
3. Ensure use cases only depend on domain interfaces

### Phase 3: Refactor Infrastructure (Week 3)
1. Create concrete implementations of domain interfaces
2. Move all external system interactions to infrastructure
3. Implement adapters for different model types

### Phase 4: Clean Up Presentation (Week 4)
1. Move API endpoints to presentation layer
2. Create proper request/response schemas
3. Implement middleware and error handling

### Phase 5: Testing and Integration (Week 5)
1. Update all imports and dependencies
2. Ensure tests still pass
3. Add integration tests for new architecture

## Key Design Patterns to Apply

1. **Dependency Injection**: Use interfaces in domain, inject implementations
2. **Repository Pattern**: For model and cache access
3. **Strategy Pattern**: For different pruning and attention strategies
4. **Factory Pattern**: For creating domain entities
5. **Adapter Pattern**: For different model types
6. **Observer Pattern**: For progress callbacks

## Benefits of This Architecture

1. **Testability**: Each layer can be tested independently
2. **Maintainability**: Clear responsibilities and boundaries
3. **Flexibility**: Easy to swap implementations
4. **Scalability**: Can add new features without affecting existing code
5. **Readability**: Smaller, focused files with clear purposes

## Next Steps

1. Review and approve this plan
2. Create the new directory structure
3. Begin Phase 1 implementation
4. Set up CI/CD to ensure no regressions during migration