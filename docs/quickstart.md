# TEMPO Quick Start Guide

Get up and running with TEMPO in 5 minutes! This guide covers the essential steps to start generating text with parallel token exploration.

## Prerequisites

- Python 3.8 or higher
- 16GB+ RAM (for model loading)
- ~20GB disk space

## Installation

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/JoeLuker/tempo.git
cd tempo

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Model (First Run)

The default model (`deepcogito/cogito-v1-preview-llama-3B`) will download automatically on first use (~6GB).

## Quick Examples

### Basic Generation

Generate text with default TEMPO settings:

```bash
python run_tempo.py --prompt "Explain quantum computing in simple terms" \
    --selection-threshold 0.05 \
    --max-tokens 100
```

### Enable Retroactive Pruning

Improve coherence by pruning less relevant tokens:

```bash
python run_tempo.py --prompt "Write a haiku about artificial intelligence" \
    --selection-threshold 0.1 \
    --use-retroactive-pruning \
    --attention-threshold 0.02 \
    --max-tokens 50
```

### Creative Writing Mode

Use higher thresholds for more creative outputs:

```bash
python run_tempo.py --prompt "Once upon a time in a digital realm" \
    --selection-threshold 0.15 \
    --use-retroactive-pruning \
    --attention-threshold 0.01 \
    --dynamic-threshold \
    --bezier-p1 0.1 \
    --bezier-p2 0.9 \
    --max-tokens 200
```

## Web Interface

For an interactive experience with visualizations:

```bash
# Terminal 1: Start the API server
uvicorn api:app --reload --port 8000

# Terminal 2: Start the frontend
cd frontend
npm install  # First time only
npm run dev
```

Open http://localhost:5174 in your browser.

## Understanding the Output

TEMPO's output shows parallel token exploration:

- `[tokenA/tokenB]` - Multiple tokens were considered at this position
- Clean text is shown in the web UI
- Tree visualization shows branching structure

Example output:
```
The [future/world] of [AI/technology] is [fascinating/exciting].
```

## Key Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `--selection-threshold` | Minimum probability for token selection | 0.05 - 0.2 |
| `--attention-threshold` | Threshold for retroactive pruning | 0.01 - 0.05 |
| `--max-tokens` | Maximum tokens to generate | 50 - 500 |

## Common Use Cases

### 1. Technical Explanations
```bash
python run_tempo.py --prompt "Explain how neural networks learn" \
    --selection-threshold 0.08 \
    --max-tokens 150
```

### 2. Creative Story Generation
```bash
python run_tempo.py --prompt "The robot discovered something unusual" \
    --selection-threshold 0.12 \
    --use-retroactive-pruning \
    --attention-threshold 0.015 \
    --max-tokens 300
```

### 3. Code Documentation
```bash
python run_tempo.py --prompt "Document this Python function: def fibonacci(n):" \
    --selection-threshold 0.06 \
    --enable-thinking \
    --max-tokens 200
```

## Tips for Best Results

1. **Start with lower thresholds** (0.05-0.1) for more focused generation
2. **Enable retroactive pruning** for better coherence in longer texts
3. **Use the web UI** to visualize token selection and understand the model's decisions
4. **Experiment with dynamic thresholds** for varying exploration over time
5. **Save interesting outputs** using `--output-dir` for analysis

## Next Steps

- Read the [full documentation](../README.md) for advanced features
- Try [MCTS mode](../README.md#mcts-parameters-cli-only) for enhanced exploration
- Explore [configuration options](configuration-guide.md) for fine-tuning
- Check [examples](examples/) for specific use cases

## Troubleshooting

### Model Loading Issues
- Ensure you have sufficient RAM (16GB+)
- Check internet connection for model download
- Try `--model "gpt2"` for testing with a smaller model

### Generation Errors
- Lower the `--selection-threshold` if getting too many parallel tokens
- Enable `--debug-mode` for detailed logs
- Check `logs/` directory for error details

### Performance Issues
- Use `--disable-kv-cache` if experiencing memory issues
- Reduce `--max-tokens` for faster generation
- Consider using CPU with `--device cpu` if GPU issues occur

## Getting Help

- Check the [FAQ](../README.md#troubleshooting)
- Open an [issue](https://github.com/JoeLuker/tempo/issues) on GitHub
- Join discussions in the repository

Ready to explore parallel token generation? Start with the examples above and experiment with different parameters to see how TEMPO reveals the model's decision-making process!