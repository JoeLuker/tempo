# TEMPO Project Structure

This document describes the organization of the TEMPO repository to maintain research-level quality and clarity.

## Root Directory

**Essential Files Only:**
```
tempo/
├── README.md                    # Main project overview
├── CLAUDE.md                    # Development guidelines
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # MIT License
├── requirements.txt             # Production dependencies
├── requirements-test.txt        # Testing dependencies
├── setup.py                     # Package installation
│
├── run_tempo.py                 # Primary CLI entry point
├── test_memory_controls.py      # Memory system validation
├── playground_server.py         # Interactive playground server
├── check_requirements.py        # Dependency checker
├── generate_config.py           # Configuration generator
├── create_generation_viz.py     # Visualization tool
└── setup_models.py              # Model setup utility
```

## Source Code (`src/`)

**Domain-Driven Architecture:**
```
src/
├── domain/                      # Core business logic
│   ├── entities/               # Domain entities
│   ├── interfaces/             # Contracts and protocols
│   └── services/               # Domain services
│
├── application/                 # Application layer
│   ├── services/               # Application services
│   └── use_cases/              # Use case implementations
│
├── infrastructure/              # External integrations
│   ├── cache/                  # Caching implementations
│   ├── generation/             # Generation strategies
│   ├── model/                  # Model adapters
│   ├── performance/            # Performance tracking
│   ├── search/                 # Search algorithms (MCTS)
│   ├── selection/              # Token selection
│   └── tokenization/           # Tokenizer adapters
│
├── algorithms/                  # Core TEMPO algorithms
│   ├── attention/              # Attention analysis
│   ├── generation/             # Generation pipeline
│   ├── pruning/                # Pruning strategies
│   └── rope/                   # RoPE modifications
│
├── modeling/                    # Model wrappers
├── experiments/                 # Experiment runners
├── visualization/               # Visualization tools
└── utils/                       # Shared utilities
```

## Documentation (`docs/`)

**Comprehensive Guides:**
```
docs/
├── quickstart.md               # Getting started guide
├── algorithm.md                # TEMPO algorithm explanation
├── architecture.md             # System architecture
├── configuration-guide.md      # Configuration options
├── memory_management.md        # Memory controls guide
├── mechanistic_interpretability.md  # Analysis tools
│
└── development/                # Development documentation
    ├── ddd_test_results.md    # DDD refactor results
    ├── isolation_analysis_findings.md
    ├── MEMORY_CONTROLS_SUMMARY.md
    ├── MEMORY_CONTROLS_TEST_RESULTS.md
    ├── FRONTEND_TEST_REPORT.md
    ├── QUICK_TEST_GUIDE.md
    └── SCALING_PLAN.md
```

## Examples (`examples/`)

**Usage Examples:**
```
examples/
├── basic_generation.py         # Simple generation
├── advanced_pruning.py         # Pruning features
├── api_client.py               # API usage
├── configs/                    # Example configurations
└── capture_playground.py       # Playground utilities
```

## Experiments (`experiments/`)

**Research and Analysis:**
```
experiments/
├── baseline_runner.py          # Baseline experiments
├── simple_persistent.py        # Layer 1 optimization
├── parallel_suite.py           # Layer 2 optimization
├── batched_runner.py           # Layer 3 optimization
├── experiment_runner.py        # Unified runner
│
└── analysis/                   # Analysis scripts
    ├── analyze_attention_by_set_size.py
    ├── analyze_attention_mech_interp.py
    ├── analyze_cross_parallel_attention.py
    ├── analyze_total_parallel_attention.py
    └── statistical_confidence_analysis.py
```

## Testing (`tests/`)

**Test Organization:**
```
tests/
├── unit/                       # Unit tests
├── integration/                # Integration tests
│
└── manual/                     # Manual testing scripts
    ├── test_attention_detailed.py
    ├── test_attention_extraction.py
    ├── test_convergence.py
    ├── test_eager_attention.py
    ├── test_shape_debug.py
    ├── test_layer1_simple.py
    ├── test_layer1_ultra_simple.py
    └── test_api.py
```

## Benchmarking (`benchmark/`)

**Performance Benchmarks:**
```
benchmark/
├── profile_tempo.py            # Profiling tools
├── benchmark_tempo.py          # Performance benchmarks
└── benchmark_all_layers.py     # Layer comparison
```

## Frontend (`frontend/`)

**Web Interface:**
```
frontend/
├── src/
│   ├── routes/                 # SvelteKit routes
│   ├── lib/
│   │   ├── components/        # Svelte components
│   │   ├── data/              # Data and configs
│   │   └── utils/             # Frontend utilities
│   └── app.html
├── static/                     # Static assets
├── package.json
└── svelte.config.js
```

## Organization Principles

### 1. **Root Directory Minimalism**
- Only essential entry points and configuration
- No test scripts (→ `tests/`)
- No analysis scripts (→ `experiments/analysis/`)
- No temporary documentation (→ `docs/development/`)

### 2. **Clear Separation of Concerns**
- **Domain**: Business logic independent of frameworks
- **Application**: Use cases and orchestration
- **Infrastructure**: External dependencies
- **Algorithms**: Core TEMPO implementations

### 3. **Documentation Hierarchy**
- **User-facing**: `docs/` root (quickstart, guides)
- **Developer-facing**: `docs/development/` (test results, plans)
- **API docs**: Generated from code

### 4. **Test Organization**
- **Unit tests**: Isolated component testing
- **Integration tests**: System integration
- **Manual tests**: Development/debugging utilities

### 5. **Research Artifacts**
- **Experiments**: `experiments/` for runners and analysis
- **Benchmarks**: `benchmark/` for performance
- **Examples**: `examples/` for usage patterns

## File Naming Conventions

- **Python scripts**: `snake_case.py`
- **Documentation**: `kebab-case.md` or `UPPER_CASE.md` for root
- **Directories**: `snake_case/` or `kebab-case/`
- **Config files**: `config.yaml`, `settings.json`

## What Goes Where?

| Item | Location | Reason |
|------|----------|--------|
| CLI entry point | Root | Primary interface |
| Model utilities | `src/utils/` | Shared infrastructure |
| Analysis scripts | `experiments/analysis/` | Research artifacts |
| Test scripts | `tests/manual/` | Development tools |
| User guides | `docs/` | Documentation |
| Dev notes | `docs/development/` | Internal documentation |
| Performance tests | `benchmark/` | Performance analysis |
| Usage examples | `examples/` | User education |
| Web UI | `frontend/` | Separate frontend |

## Maintenance Rules

1. **Root cleanup**: If >10 Python files in root, move to appropriate directory
2. **Documentation**: User-facing in `docs/`, development in `docs/development/`
3. **Tests**: Production tests in `tests/`, debugging scripts in `tests/manual/`
4. **Research**: Experiments and analysis in `experiments/`
5. **Examples**: Keep `examples/` minimal and well-documented

## Quality Standards

### Research-Level Organization:
- ✅ Clear separation of source, tests, docs, examples
- ✅ Minimal root directory clutter
- ✅ Logical grouping of related files
- ✅ Consistent naming conventions
- ✅ Comprehensive documentation

### Professional Structure:
- ✅ Domain-driven design in `src/`
- ✅ Test organization by type
- ✅ Documentation hierarchy
- ✅ Example and experiment separation
- ✅ Performance benchmarking tools

This structure ensures TEMPO maintains research-quality organization while remaining accessible and maintainable.
