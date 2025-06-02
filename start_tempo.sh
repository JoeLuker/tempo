#!/bin/bash

# TEMPO Development Startup Script (Safe Mode - No Docker)
# Starts both the FastAPI backend and the frontend development server locally

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[TEMPO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "api.py" ]; then
    print_error "api.py not found. Please run this script from the TEMPO root directory."
    exit 1
fi

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -i :$port > /dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to kill process on port
kill_port() {
    local port=$1
    print_warning "Port $port is in use. Attempting to free it..."
    lsof -ti :$port | xargs kill -9 2>/dev/null || true
    sleep 2
}

print_status "Starting TEMPO development environment (Safe Mode - No Docker)..."

# Check for Python virtual environment
if [ ! -d ".venv" ]; then
    print_warning "Virtual environment not found. Creating one..."
    python3 -m venv .venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Check if requirements are installed by testing key imports
if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    print_success "Python dependencies installed"
fi

# Function to cleanup on exit
cleanup() {
    print_status "Shutting down TEMPO servers..."
    
    # Kill any running processes
    if [ ! -z "$BACKEND_PID" ]; then
        if kill -0 $BACKEND_PID 2>/dev/null; then
            kill $BACKEND_PID
            print_success "Backend server stopped"
        fi
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            kill $FRONTEND_PID
            print_success "Frontend server stopped"
        fi
    fi
    
    print_success "TEMPO development environment stopped"
}

# Set up signal handlers for cleanup
trap cleanup EXIT INT TERM

# Start Backend
print_status "Starting FastAPI backend server..."

# Check if port 8000 is in use
if check_port 8000; then
    kill_port 8000
fi

# Start the API server
print_status "Starting backend in development mode..."
uvicorn api:app --reload --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
print_status "Waiting for backend to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_success "Backend server is running on http://localhost:8000"
        print_success "API documentation available at http://localhost:8000/docs"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        print_warning "Backend health check failed, but process may still be starting..."
        if kill -0 $BACKEND_PID 2>/dev/null; then
            print_success "Backend process is running on http://localhost:8000"
        else
            print_error "Backend process failed to start"
            exit 1
        fi
    fi
done

# Start Frontend
print_status "Starting frontend development server..."

cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    print_status "Installing frontend dependencies..."
    npm install
    print_success "Frontend dependencies installed"
fi

# Check if port 5174 is in use
if check_port 5174; then
    kill_port 5174
fi

print_status "Starting frontend in development mode..."
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start
print_status "Waiting for frontend to start..."
for i in {1..30}; do
    if curl -s http://localhost:5174 > /dev/null 2>&1; then
        print_success "Frontend server is running on http://localhost:5174"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        print_warning "Frontend health check failed, but process may still be starting..."
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            print_success "Frontend process is running on http://localhost:5174"
        else
            print_error "Frontend process failed to start"
            cd ..
            exit 1
        fi
    fi
done

cd ..

# Print final status
echo ""
print_success "TEMPO development environment is running!"
echo ""
print_status "Available URLs:"
echo "  Frontend: http://localhost:5174"
echo "  Backend API: http://localhost:8000"
echo "  API Documentation: http://localhost:8000/docs"
echo ""
print_status "Press Ctrl+C to stop all servers"

# Wait for user interrupt
wait