#!/bin/bash

# TEMPO Quick Start Script
# This script sets up TEMPO and runs a test generation in one go

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_status() {
    echo -e "${BLUE}[TEMPO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Welcome message
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           TEMPO Quick Start Installation             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python version
print_status "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.8+ required, but $PYTHON_VERSION found"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.8 or higher"
    exit 1
fi

# Check if we're in the TEMPO directory
if [ ! -f "run_tempo.py" ]; then
    print_error "This script must be run from the TEMPO directory"
    print_error "Please cd into the tempo directory and try again"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv .venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip -q

# Install dependencies
print_status "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt -q
print_success "Dependencies installed"

# Check if model needs to be downloaded
print_status "Checking model availability..."
if [ -f "setup_models.py" ]; then
    python3 setup_models.py --check-only 2>/dev/null || {
        print_warning "Model will be downloaded on first run"
    }
else
    print_warning "Model will be downloaded on first run"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p output logs
print_success "Directories created"

# Check available memory
print_status "Checking system resources..."
if command -v free &> /dev/null; then
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM" -lt 8 ]; then
        print_warning "System has ${TOTAL_MEM}GB RAM. 16GB+ recommended for optimal performance"
    else
        print_success "System has ${TOTAL_MEM}GB RAM"
    fi
elif command -v sysctl &> /dev/null; then
    # macOS
    TOTAL_MEM=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
    if [ "$TOTAL_MEM" -lt 8 ]; then
        print_warning "System has ${TOTAL_MEM}GB RAM. 16GB+ recommended for optimal performance"
    else
        print_success "System has ${TOTAL_MEM}GB RAM"
    fi
fi

# Success message
echo ""
print_success "TEMPO setup complete!"
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  Ready to use TEMPO!                 ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# Provide next steps
echo "Try these commands to get started:"
echo ""
echo -e "${YELLOW}1. Basic generation:${NC}"
echo "   python3 run_tempo.py --prompt \"Once upon a time\" --selection-threshold 0.1"
echo ""
echo -e "${YELLOW}2. Creative writing with branching:${NC}"
echo "   python3 run_tempo.py --prompt \"Write a haiku about AI\" --selection-threshold 0.15"
echo ""
echo -e "${YELLOW}3. Advanced with pruning:${NC}"
echo "   python3 run_tempo.py --prompt \"The future of technology\" --use-retroactive-pruning"
echo ""
echo -e "${YELLOW}4. Start the web interface:${NC}"
echo "   ./start_tempo.sh"
echo ""
echo -e "${BLUE}For more examples, see the README.md or run: python3 run_tempo.py --help${NC}"
echo ""

# Optional: Run a test generation
read -p "Would you like to run a test generation now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    print_status "Running test generation..."
    echo ""
    python3 run_tempo.py --prompt "The meaning of life is" --selection-threshold 0.1 --max-tokens 50
    echo ""
    print_success "Test generation complete!"
    echo ""
    echo "Check the 'output' directory for visualizations!"
fi

echo ""
print_status "Virtual environment is activated. When you're done, run: deactivate"
echo ""