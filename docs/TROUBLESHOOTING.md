# TEMPO Troubleshooting Guide

This guide helps you resolve common issues when using TEMPO.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Model Loading Problems](#model-loading-problems)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [API and Frontend Issues](#api-and-frontend-issues)
- [Generation Quality Issues](#generation-quality-issues)
- [Common Error Messages](#common-error-messages)

## Installation Issues

### Python Version Error

**Problem**: `Python 3.8+ required`

**Solution**:
```bash
# Check your Python version
python3 --version

# If too old, install Python 3.8+ using:
# macOS
brew install python@3.10

# Ubuntu/Debian
sudo apt update
sudo apt install python3.10

# Windows
# Download from https://www.python.org/downloads/
```

### Dependency Installation Fails

**Problem**: `pip install -r requirements.txt` fails

**Solutions**:

1. **Update pip**:
   ```bash
   python3 -m pip install --upgrade pip
   ```

2. **Install missing system dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-dev build-essential

   # macOS
   xcode-select --install
   ```

3. **Use conda instead**:
   ```bash
   conda create -n tempo python=3.10
   conda activate tempo
   pip install -r requirements.txt
   ```

### Virtual Environment Issues

**Problem**: Virtual environment not activating

**Solution**:
```bash
# Make sure you created it
python3 -m venv .venv

# Correct activation commands:
# Linux/macOS
source .venv/bin/activate

# Windows Command Prompt
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

## Model Loading Problems

### Model Download Fails

**Problem**: Can't download the model

**Solutions**:

1. **Check disk space**:
   ```bash
   # Need ~8GB for 3B model
   df -h  # Linux/macOS
   ```

2. **Pre-download model**:
   ```bash
   python3 setup_models.py
   ```

3. **Use different cache directory**:
   ```bash
   export HF_HOME=/path/with/more/space/.cache/huggingface
   ```

4. **Authentication required**:
   ```bash
   # Some models need HuggingFace login
   pip install huggingface-hub
   huggingface-cli login
   ```

### Out of Memory (OOM)

**Problem**: `RuntimeError: CUDA out of memory` or system freezes

**Solutions**:

1. **Reduce batch size/length**:
   ```bash
   python3 run_tempo.py --max-tokens 50  # Reduce from default
   ```

2. **Use CPU instead**:
   ```bash
   export TEMPO_MODEL_DEVICE=cpu
   python3 run_tempo.py --prompt "Test"
   ```

3. **Enable quantization** (for 7B+ models):
   ```python
   # In config.json
   {
     "model": {
       "quantization": "4bit"
     }
   }
   ```

4. **Close other applications** to free memory

### Wrong Device Error

**Problem**: `RuntimeError: MPS backend not available`

**Solution**:
```bash
# Force CPU usage
export TEMPO_MODEL_DEVICE=cpu

# Or in your command
python3 run_tempo.py --prompt "Test" --device cpu
```

## Runtime Errors

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Make sure you're in the TEMPO root directory
cd /path/to/tempo

# Add to Python path if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### RoPE Patching Fails

**Problem**: `Failed to install RoPE hook`

**Solutions**:

1. **Check model compatibility**:
   ```bash
   # RoPE modification only works with Llama-based models
   # Use --no-custom-rope for other architectures
   python3 run_tempo.py --prompt "Test" --no-custom-rope
   ```

2. **Update transformers**:
   ```bash
   pip install --upgrade transformers
   ```

### Attention Manager Errors

**Problem**: `RuntimeError in CoherencePruningStrategy`

**Solution**:
```bash
# Disable pruning temporarily
python3 run_tempo.py --prompt "Test" --no-retroactive-pruning

# Or adjust thresholds
python3 run_tempo.py --prompt "Test" --attention-threshold 0.05
```

## Performance Issues

### Slow Generation

**Problem**: Generation takes too long

**Solutions**:

1. **Reduce parallel tokens**:
   ```bash
   # Lower threshold = fewer parallel paths
   python3 run_tempo.py --selection-threshold 0.05
   ```

2. **Disable visualization**:
   ```bash
   python3 run_tempo.py --no-save-visualization
   ```

3. **Use GPU acceleration**:
   ```bash
   # Check GPU availability
   python3 check_requirements.py

   # Ensure using GPU
   export TEMPO_MODEL_DEVICE=cuda  # or mps for Apple Silicon
   ```

4. **Profile to find bottlenecks**:
   ```bash
   python3 run_tempo.py --prompt "Test" --profile --max-tokens 20
   ```

### High Memory Usage

**Problem**: System becomes unresponsive

**Solutions**:

1. **Limit generation length**:
   ```bash
   python3 run_tempo.py --max-tokens 50
   ```

2. **Disable KV cache** (slower but less memory):
   ```bash
   python3 run_tempo.py --disable-kv-cache
   ```

3. **Monitor memory usage**:
   ```bash
   # In another terminal
   watch -n 1 free -h  # Linux
   top  # macOS/Linux
   ```

## API and Frontend Issues

### Port Already in Use

**Problem**: `[Errno 48] Address already in use`

**Solutions**:

1. **Kill existing process**:
   ```bash
   # Find process
   lsof -ti:8000  # macOS/Linux
   netstat -ano | findstr :8000  # Windows

   # Kill it
   kill -9 $(lsof -ti:8000)  # macOS/Linux
   ```

2. **Use different port**:
   ```bash
   uvicorn api:app --port 8001
   ```

3. **Use the start script** (handles this automatically):
   ```bash
   ./start_tempo.sh
   ```

### Frontend Won't Connect to API

**Problem**: Frontend shows connection errors

**Solutions**:

1. **Check both services are running**:
   ```bash
   # Terminal 1
   uvicorn api:app --reload --port 8000

   # Terminal 2
   cd frontend && npm run dev
   ```

2. **Verify API URL** in frontend:
   ```javascript
   // frontend/.env
   VITE_API_URL=http://localhost:8000
   ```

3. **Check CORS settings**:
   ```python
   # In api.py or config.json
   "cors_origins": ["http://localhost:5173", "http://localhost:5174"]
   ```

### npm Install Fails

**Problem**: Frontend dependencies won't install

**Solutions**:

1. **Clear npm cache**:
   ```bash
   npm cache clean --force
   rm -rf node_modules package-lock.json
   npm install
   ```

2. **Use correct Node version**:
   ```bash
   # Need Node 16+
   node --version

   # Install via nvm
   nvm install 18
   nvm use 18
   ```

## Generation Quality Issues

### Too Many Parallel Tokens

**Problem**: Output is cluttered with alternatives

**Solution**:
```bash
# Lower the selection threshold
python3 run_tempo.py --selection-threshold 0.05

# Enable aggressive pruning
python3 run_tempo.py --use-retroactive-pruning --attention-threshold 0.02
```

### No Branching Occurring

**Problem**: Output looks like standard generation

**Solutions**:

1. **Increase selection threshold**:
   ```bash
   python3 run_tempo.py --selection-threshold 0.2
   ```

2. **Check RoPE is enabled**:
   ```bash
   python3 run_tempo.py --use-custom-rope  # Should be default
   ```

3. **Verify TEMPO mode**:
   ```bash
   # Make sure not in default mode
   python3 run_tempo.py --prompt "Test"  # NOT --default-mode
   ```

### Incoherent Output

**Problem**: Generated text doesn't make sense

**Solutions**:

1. **Enable pruning**:
   ```bash
   python3 run_tempo.py --use-retroactive-pruning
   ```

2. **Adjust temperature**:
   ```bash
   python3 run_tempo.py --temperature 0.7  # Lower = more focused
   ```

3. **Use dynamic thresholding**:
   ```bash
   python3 run_tempo.py --dynamic-threshold --bezier-p1 0.1 --bezier-p2 0.9
   ```

## Common Error Messages

### "RuntimeError: Expected all tensors to be on the same device"

**Cause**: Mixed GPU/CPU tensors

**Fix**:
```bash
# Force single device
export TEMPO_MODEL_DEVICE=cpu  # or cuda or mps
```

### "ValueError: --attention-threshold must be between 0 and 1"

**Cause**: Invalid parameter value

**Fix**:
```bash
# Check parameter ranges
python3 run_tempo.py --help
```

### "FileNotFoundError: config.json"

**Cause**: Optional config file not found

**Fix**:
```bash
# Create config from template
cp config.example.json config.json

# Or use generator
python3 generate_config.py
```

### "AssertionError in debug mode"

**Cause**: Internal consistency check failed

**Fix**:
```bash
# Disable debug mode
export TEMPO_DEBUG=false

# Report issue with details
```

## Getting Help

If you're still stuck:

1. **Check existing issues**: https://github.com/JoeLuker/tempo/issues
2. **Run diagnostics**:
   ```bash
   python3 check_requirements.py --export diagnostics.json
   ```
3. **Enable debug logging**:
   ```bash
   export TEMPO_DEBUG=true
   export TEMPO_LOGGING_LEVEL=DEBUG
   ```
4. **Create minimal reproduction**:
   ```bash
   # Simplest command that shows the problem
   python3 run_tempo.py --prompt "Test" --max-tokens 10 --debug-mode
   ```

## Quick Fixes Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated  
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Enough disk space (20GB+)
- [ ] Enough RAM (8GB minimum, 16GB recommended)
- [ ] Model downloaded (`python3 setup_models.py`)
- [ ] In TEMPO root directory when running
- [ ] API running for web interface (`uvicorn api:app`)
- [ ] Correct ports available (8000, 5174)

## Debug Commands

```bash
# System check
python3 check_requirements.py

# Test basic generation
python3 run_tempo.py --prompt "Hello" --max-tokens 10 --debug-mode

# Test API
curl http://localhost:8000/health

# Check model cache
ls ~/.cache/huggingface/hub/models--deepcogito*
```