# TEMPO: Threshold-Enabled Multipath Parallel Output

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/JoeLuker/tempo)](https://github.com/JoeLuker/tempo/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-green)](https://github.com/JoeLuker/tempo/actions)
[![GitHub stars](https://img.shields.io/github/stars/JoeLuker/tempo)](https://github.com/JoeLuker/tempo/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/JoeLuker/tempo)](https://github.com/JoeLuker/tempo/issues)

**TEMPO** is an experimental approach to language model generation that explores processing multiple token possibilities simultaneously. Unlike traditional beam search, TEMPO modifies Rotary Position Embeddings (RoPE) to simulate parallel processing within a single sequence state, providing unique insights into model uncertainty and decision-making.

🚧 **Status:** Experimental research project focused on the `deepcogito/cogito-v1-preview-llama-3B` model.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 16GB+ RAM
- ~20GB free disk space
- (Optional) Node.js 16+ for web interface

### One-Command Setup

```bash
# Clone and setup TEMPO in one command
git clone https://github.com/JoeLuker/tempo.git && cd tempo && ./quickstart.sh
```

Or manually:

```bash
# Clone the repository
git clone https://github.com/JoeLuker/tempo.git
cd tempo

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the model (first run will do this automatically)
python3 setup_models.py  # Optional: pre-download model

# Run your first generation!
python3 run_tempo.py --prompt "Once upon a time" --selection-threshold 0.1 --max-tokens 50
```

### 🎯 Try These Examples

```bash
# Basic generation
python3 run_tempo.py --prompt "Explain quantum computing simply" --selection-threshold 0.05

# Creative writing with multiple paths
python3 run_tempo.py --prompt "Write a haiku about AI" --selection-threshold 0.15 --use-retroactive-pruning

# See the branching visualization
python3 run_tempo.py --prompt "The future of technology is" --selection-threshold 0.2 --save-visualization
```

### 🌐 Web Interface (Optional)

```bash
# In one terminal: Start the backend
uvicorn api:app --reload --port 8000

# In another terminal: Start the frontend
cd frontend && npm install && npm run dev
```

Visit `http://localhost:5174` for an interactive UI with real-time visualization!

## 📖 What is TEMPO?

TEMPO enables language models to explore multiple generation paths simultaneously, revealing:

- **🔍 Model Uncertainty**: See where models are confident vs. uncertain
- **🧠 Attention Patterns**: Understand which tokens truly matter for coherence  
- **🌳 Branching Logic**: Visualize the model's internal decision tree

### Example Output

```
The future of artificial intelligence [is/will be/lies] [fascinating/transformative/unpredictable]
```

TEMPO shows you the `[alternative/tokens]` the model considered at each step!

## 📚 Documentation

- **[Quick Start Guide](docs/quickstart.md)** - Get up and running in 5 minutes
- **[Architecture Overview](docs/architecture.md)** - Technical deep dive
- **[API Reference](docs/api-reference.md)** - Complete API documentation
- **[Configuration Guide](docs/configuration-guide.md)** - All configuration options
- **[Examples](docs/examples/)** - Code examples and tutorials
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions

## 🎯 Key Features

- **🔀 Parallel Token Processing**: Explore multiple generation paths simultaneously
- **✂️ Smart Pruning**: Retroactively refine paths based on attention patterns
- **📊 Visualization**: See token probabilities and branching in real-time
- **🌐 Web Interface**: Interactive UI for experimentation
- **🧪 Research Tools**: MCTS search, dynamic thresholding, attention analysis
- **⚡ Performance**: Optimized for Apple Silicon (MPS) and CUDA

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.10+ |
| RAM | 8GB | 16GB+ |
| Disk Space | 15GB | 20GB+ |
| GPU | Optional | CUDA 11.0+ or Apple MPS |
| OS | Linux/macOS/Windows | macOS (Apple Silicon) or Linux |

**Tested Model**: `deepcogito/cogito-v1-preview-llama-3B` (3B parameters)

## 🏗️ How TEMPO Works

1. **Token Selection**: At each step, identify tokens above a probability threshold
2. **Parallel Processing**: Process multiple tokens as if they occupy the same position
3. **RoPE Modification**: Modify positional embeddings to enable parallel processing
4. **Attention Analysis**: Track how future tokens attend to parallel alternatives
5. **Smart Pruning**: Remove less-attended paths retroactively
6. **Visualization**: Show the branching structure in the output

<details>
<summary>📊 Technical Details</summary>

TEMPO modifies the transformer's Rotary Position Embeddings (RoPE) to assign identical positional encodings to tokens that should be processed in parallel. This allows the model to maintain multiple hypotheses within a single forward pass, unlike beam search which requires separate sequence states.

Key components:
- `RoPEModifier`: Patches position embeddings
- `ParallelGenerator`: Manages multi-token generation
- `RetroactivePruner`: Refines token sets based on attention
- `AttentionManager`: Controls token visibility

</details>

## 🛠️ Installation

### Option 1: Quick Install (Recommended)

```bash
# Clone and setup in one command
git clone https://github.com/JoeLuker/tempo.git && cd tempo && ./quickstart.sh
```

### Option 2: Manual Installation

<details>
<summary>Click for manual steps</summary>

```bash
# 1. Clone the repository
git clone https://github.com/JoeLuker/tempo.git
cd tempo

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt  # Optional: for testing

# 4. (Optional) Pre-download the model
python3 setup_models.py

# 5. Verify installation
python3 check_requirements.py
```

</details>

### Option 3: Docker (Coming Soon)

```bash
# Build and run with Docker
docker build -t tempo .
docker run -it -p 8000:8000 tempo
```

## 📖 Usage Guide

### Command Line Interface

#### Basic Examples

```bash
# Simple generation
python3 run_tempo.py --prompt "The meaning of life is"

# Adjust threshold for more/fewer parallel tokens
python3 run_tempo.py --prompt "AI will" --selection-threshold 0.2

# Enable smart pruning
python3 run_tempo.py --prompt "Write a story" --use-retroactive-pruning --attention-threshold 0.01
```

#### Advanced Features

```bash
# Dynamic thresholding with Bezier curves
python3 run_tempo.py --prompt "Complex reasoning about" \
  --selection-threshold 0.1 \
  --use-retroactive-pruning \
  --dynamic-threshold \
  --bezier-p1 0.1 --bezier-p2 0.9

# MCTS exploration
python3 run_tempo.py --prompt "Creative story:" \
  --use-mcts \
  --mcts-simulations 20 \
  --mcts-depth 8

# Cogito deep thinking mode
python3 run_tempo.py --prompt "Explain quantum mechanics" \
  --enable-thinking \
  --selection-threshold 0.15
```

### Web Interface

```bash
# Start both servers with one command
./start_tempo.sh

# Or manually:
# Terminal 1: Backend
uvicorn api:app --reload

# Terminal 2: Frontend  
cd frontend && npm run dev
```

Open `http://localhost:5174` for:
- 🎨 Interactive text generation
- 📊 Real-time token visualization
- 🌳 Branching tree view
- ⚙️ Easy parameter tuning

### API Usage

<details>
<summary>📡 REST API Examples</summary>

```python
# Python example
import requests

response = requests.post(
    "http://localhost:8000/api/generate",
    json={
        "prompt": "The future of AI is",
        "selection_threshold": 0.1,
        "use_retroactive_pruning": True,
        "max_tokens": 100
    }
)

result = response.json()
print(result["clean_text"])  # Human-readable output
print(result["generated_text"])  # Output with [token/alternatives]
```

```javascript
// JavaScript example
const response = await fetch('http://localhost:8000/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        prompt: "Once upon a time",
        selection_threshold: 0.15,
        max_tokens: 150
    })
});

const data = await response.json();
console.log(data.clean_text);
```

</details>

## 🧪 Configuration

TEMPO offers flexible configuration through multiple methods:

### 1. Environment Variables
```bash
export TEMPO_MODEL_ID="mistralai/Mistral-7B-v0.1"  # Change model
export TEMPO_DEBUG=true                            # Enable debug mode
export TEMPO_GENERATION_MAX_LENGTH=500            # Set max length
```

### 2. Configuration File
```bash
# Copy template and edit
cp config.example.json config.json
vim config.json

# Or use the interactive generator
python3 generate_config.py
```

### 3. Command Line Arguments
```bash
python3 run_tempo.py --selection-threshold 0.1 --max-tokens 200
```

**Priority**: CLI args > Environment vars > Config file > Defaults

See [Configuration Guide](docs/configuration-guide.md) for all options.

## 🔬 Development

### Running Tests

```bash
# Quick test
python3 run_tests.py

# With coverage
python3 run_tests.py --cov

# Specific tests
python3 run_tests.py --unit-only
python3 run_tests.py --integration-only
```

### Project Structure

```
tempo/
├── src/               # Core implementation
│   ├── generation/    # Token generation logic
│   ├── pruning/       # Pruning strategies
│   └── modeling/      # Model wrappers
├── frontend/          # Svelte web UI
├── api.py            # FastAPI backend
├── run_tempo.py      # CLI interface
└── docs/             # Documentation
```

### Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## ❓ Troubleshooting

### Common Issues

**Model won't load**
```bash
# Check system requirements
python3 check_requirements.py

# Pre-download model
python3 setup_models.py
```

**Port already in use**
```bash
# Use the start script (handles port conflicts)
./start_tempo.sh

# Or manually kill the process
lsof -ti:8000 | xargs kill -9
```

**Out of memory**
- Reduce `--max-tokens`
- Lower `--selection-threshold` 
- Use CPU instead: `export TEMPO_DEVICE=cpu`

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more solutions.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built on HuggingFace Transformers
- Inspired by mechanistic interpretability research
- Special thanks to the Cogito model team

---

<p align="center">
  <strong>Ready to explore multiple futures? 🚀</strong><br>
  <code>python3 run_tempo.py --prompt "Your idea here"</code>
</p>

