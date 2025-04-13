# TEMPO: Threshold-Enabled Multiple Parallel Outputs

[![GitHub](https://img.shields.io/github/license/JoeLuker/tempo)](https://github.com/JoeLuker/tempo/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/JoeLuker/tempo)](https://github.com/JoeLuker/tempo/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/JoeLuker/tempo)](https://github.com/JoeLuker/tempo/issues)

This project implements and evaluates a non-autoregressive text generation mechanism called "TEMPO" (Threshold-Enabled Multiple Parallel Outputs) using Mistral-7B on Apple Silicon with MPS. It includes both a command-line interface and a modern web interface for interactive exploration.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

Standard autoregressive text generation may constrain a model's ability to express potentially concurrent internal states. TEMPO explores how language models process sequential information with concurrent possibilities by:

- Using a **Threshold** mechanism that controls token selection
- **Enabling** functionality that transforms standard generation
- Generating **Multiple** tokens simultaneously at each step
- Implementing a **Parallel** generation process
- Producing **Outputs** that demonstrate model uncertainty

The experiment tests generating multiple tokens simultaneously based on a probability threshold, with advanced features like Monte Carlo Tree Search (MCTS) and deep thinking mode.

## Features

- **Web Interface**: Modern Svelte-based UI for interactive exploration
- **Multiple Generation Strategies**:
  - Basic parallel generation with threshold control
  - Monte Carlo Tree Search (MCTS) for optimized path selection
  - Deep thinking mode for more thoughtful responses
- **Advanced Pruning**:
  - Coherence-based pruning for focused outputs
  - Diversity-based pruning for exploring alternatives
  - Dynamic thresholding with customizable Bezier curves
- **Visualization Tools**:
  - Token distribution analysis
  - Position-based visualization
  - Interactive exploration of parallel paths

## System Requirements

- Python 3.8 or higher
- Node.js 16 or higher
- Apple Silicon Mac (M1/M2) for optimal performance
- At least 16GB RAM
- 20GB free disk space for model and dependencies

## Installation

### Backend Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn api:app --reload
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The web interface will be available at `http://localhost:5173` and the API at `http://localhost:8000`.

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
MODEL_PATH=/path/to/model
DEVICE=mps  # or cuda for NVIDIA GPUs
LOG_LEVEL=INFO
API_PORT=8000
FRONTEND_PORT=5173
```

### Model Configuration

The system supports various model configurations through the API:

- Model size and architecture
- Generation parameters
- Pruning strategies
- Visualization options

See the API documentation at `http://localhost:8000/docs` for detailed configuration options.

## Usage

### Command Line Interface

```bash
# Basic generation
python run_tempo.py --prompt "Your prompt here" --threshold 0.1

# With MCTS enabled
python run_tempo.py --prompt "Your prompt here" --threshold 0.1 --use-mcts --mcts-simulations 10

# With deep thinking mode
python run_tempo.py --prompt "Your prompt here" --threshold 0.1 --enable-thinking

# With custom pruning strategy
python run_tempo.py --prompt "Your prompt here" --threshold 0.1 --use-pruning --coherence-threshold 0.7 --diversity-clusters 3
```

### API Usage

The FastAPI backend provides the following endpoints:

- `GET /`: API documentation
- `GET /health`: Health check endpoint
- `POST /generate`: Main generation endpoint with extensive configuration options

Example API call:

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Your prompt here", "threshold": 0.1}'
```

## Development

### Project Structure

```
.
├── api.py                 # FastAPI backend implementation
├── run_tempo.py          # Command-line interface
├── requirements.txt      # Python dependencies
├── frontend/            # Svelte frontend
│   ├── src/            # Frontend source code
│   ├── static/         # Static assets
│   └── package.json    # Frontend dependencies
├── src/                # Core implementation
│   ├── modeling/       # Model wrapper and utilities
│   ├── generation/     # Generation strategies
│   ├── search/        # MCTS implementation
│   ├── pruning/       # Pruning strategies
│   ├── visualization/ # Visualization tools
│   └── experiments/   # Experimental features and research
├── output/            # Generated outputs
└── images/            # Project images and visualizations
```

### Development Guidelines

1. Follow PEP 8 style guide for Python code
2. Use type hints for all function parameters and return values
3. Write docstrings for all public functions and classes
4. Keep commits atomic and well-documented
5. Create feature branches for new development
6. Update documentation when adding new features

## Testing

### Backend Testing

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run specific test file
pytest tests/test_generation.py

# Run with coverage
pytest --cov=src
```

### Frontend Testing

```bash
cd frontend
npm test
```

## Security

### Best Practices

1. Never commit API keys or sensitive credentials
2. Use environment variables for configuration
3. Validate all user input
4. Implement rate limiting for API endpoints
5. Keep dependencies updated
6. Use HTTPS in production

### Security Headers

The API includes security headers by default:
- CORS protection
- XSS protection
- Content Security Policy
- HSTS (in production)

## Troubleshooting

### Common Issues

1. **Model Loading Issues**
   - Ensure sufficient disk space
   - Check model path in configuration
   - Verify GPU/CPU compatibility

2. **API Connection Problems**
   - Check if both backend and frontend servers are running
   - Verify port configurations
   - Check firewall settings

3. **Generation Errors**
   - Adjust threshold values
   - Check available memory
   - Verify model compatibility

4. **Visualization Issues**
   - Clear browser cache
   - Check browser compatibility
   - Verify data format

### Getting Help

- Check the [issues](https://github.com/JoeLuker/tempo/issues) page
- Create a new issue with detailed error information
- Include system information and error logs
- Follow [@JoeLuker](https://github.com/JoeLuker) on GitHub for updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

Please ensure your PR:
- Follows the project's coding style
- Includes tests for new features
- Updates documentation
- Has a clear description

## License

TEMPO is licensed under the MIT License.

The MIT License is a permissive license that allows you to:
- Use the software for any purpose
- Modify and distribute the software
- Use the software commercially
- Keep modifications private

The only requirements are:
- Include the original copyright notice
- Include the license text

See the [LICENSE](LICENSE) file for full terms and conditions.