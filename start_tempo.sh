#!/bin/bash

# TEMPO Development Startup Script
# Starts both the FastAPI backend and the frontend development server

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

# Parse command line arguments
BACKEND_ONLY=false
FRONTEND_ONLY=false
USE_DOCKER=false
PRODUCTION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend-only)
            BACKEND_ONLY=true
            shift
            ;;
        --frontend-only)
            FRONTEND_ONLY=true
            shift
            ;;
        --docker)
            USE_DOCKER=true
            shift
            ;;
        --production)
            PRODUCTION=true
            shift
            ;;
        --help)
            echo "TEMPO Development Startup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --backend-only   Start only the FastAPI backend server"
            echo "  --frontend-only  Start only the frontend development server"
            echo "  --docker         Use Docker for frontend (requires Docker)"
            echo "  --production     Start in production mode"
            echo "  --help           Show this help message"
            echo ""
            echo "Default: Starts both backend and frontend in development mode"
            echo ""
            echo "URLs:"
            echo "  Frontend: http://localhost:5173 (dev) or http://localhost:3000 (prod)"
            echo "  Backend API: http://localhost:8000"
            echo "  API Docs: http://localhost:8000/docs"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

print_status "Starting TEMPO development environment..."

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

# Function to start backend
start_backend() {
    print_status "Starting FastAPI backend server..."
    
    # Check if port 8000 is in use
    if check_port 8000; then
        kill_port 8000
    fi
    
    # Start the API server with virtual environment activated
    if [ "$PRODUCTION" = true ]; then
        print_status "Starting backend in production mode..."
        uvicorn api:app --host 0.0.0.0 --port 8000 &
    else
        print_status "Starting backend in development mode..."
        uvicorn api:app --reload --port 8000 &
    fi
    
    BACKEND_PID=$!
    echo $BACKEND_PID > .backend.pid
    
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
            # Check if the process is still running
            if kill -0 $BACKEND_PID 2>/dev/null; then
                print_success "Backend process is running on http://localhost:8000"
                print_success "API documentation available at http://localhost:8000/docs"
            else
                print_error "Backend process failed to start"
                return 1
            fi
            break
        fi
    done
}

# Function to start frontend
start_frontend() {
    if [ "$USE_DOCKER" = true ]; then
        start_frontend_docker
    else
        start_frontend_local
    fi
}

# Function to start frontend locally
start_frontend_local() {
    print_status "Starting frontend development server..."
    
    cd frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        print_status "Installing frontend dependencies..."
        npm install
        print_success "Frontend dependencies installed"
    fi
    
    # Check if port 5173 (dev) or 3000 (prod) is in use
    if [ "$PRODUCTION" = true ]; then
        if check_port 3000; then
            kill_port 3000
        fi
        print_status "Building and starting frontend in production mode..."
        npm run build
        npm run preview &
        FRONTEND_PID=$!
        FRONTEND_URL="http://localhost:3000"
    else
        if check_port 5173; then
            kill_port 5173
        fi
        print_status "Starting frontend in development mode..."
        npm run dev &
        FRONTEND_PID=$!
        FRONTEND_URL="http://localhost:5173"
    fi
    
    echo $FRONTEND_PID > ../.frontend.pid
    
    # Wait for frontend to start
    print_status "Waiting for frontend to start..."
    for i in {1..30}; do
        if curl -s $FRONTEND_URL > /dev/null 2>&1; then
            print_success "Frontend server is running on $FRONTEND_URL"
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            print_warning "Frontend health check failed, but process may still be starting..."
            # Check if the process is still running
            if kill -0 $FRONTEND_PID 2>/dev/null; then
                print_success "Frontend process is running on $FRONTEND_URL"
            else
                print_error "Frontend process failed to start"
                cd ..
                return 1
            fi
            break
        fi
    done
    
    cd ..
}

# Function to start frontend with Docker
start_frontend_docker() {
    print_status "Starting frontend with Docker..."
    
    cd frontend
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        cd ..
        return 1
    fi
    
    # Check if docker-compose.yml exists
    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml not found in frontend directory"
        cd ..
        return 1
    fi
    
    # Stop any existing containers first to avoid conflicts
    print_status "Checking for existing containers..."
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Small delay to let Docker clean up
    sleep 2
    
    if [ "$PRODUCTION" = true ]; then
        print_status "Starting frontend in production mode with Docker..."
        # Use docker compose (v2) instead of docker-compose if available
        if command -v "docker compose" > /dev/null 2>&1; then
            docker compose up frontend-prod -d
        else
            docker-compose up frontend-prod -d
        fi
        FRONTEND_URL="http://localhost:3000"
    else
        print_status "Starting frontend in development mode with Docker..."
        # Use docker compose (v2) instead of docker-compose if available
        if command -v "docker compose" > /dev/null 2>&1; then
            docker compose up frontend-dev -d
        else
            docker-compose up frontend-dev -d
        fi
        FRONTEND_URL="http://localhost:5173"
    fi
    
    # Wait for frontend to start
    print_status "Waiting for frontend to start..."
    for i in {1..60}; do
        if curl -s $FRONTEND_URL > /dev/null 2>&1; then
            print_success "Frontend server is running on $FRONTEND_URL"
            break
        fi
        sleep 1
        if [ $i -eq 60 ]; then
            print_warning "Frontend health check failed, checking Docker status..."
            if command -v "docker compose" > /dev/null 2>&1; then
                docker compose ps
            else
                docker-compose ps
            fi
            cd ..
            return 1
        fi
    done
    
    cd ..
}

# Function to cleanup on exit
cleanup() {
    print_status "Shutting down TEMPO servers..."
    
    if [ -f ".backend.pid" ]; then
        BACKEND_PID=$(cat .backend.pid)
        if kill -0 $BACKEND_PID 2>/dev/null; then
            kill $BACKEND_PID
            print_success "Backend server stopped"
        fi
        rm -f .backend.pid
    fi
    
    if [ -f ".frontend.pid" ]; then
        FRONTEND_PID=$(cat .frontend.pid)
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            kill $FRONTEND_PID
            print_success "Frontend server stopped"
        fi
        rm -f .frontend.pid
    fi
    
    if [ "$USE_DOCKER" = true ]; then
        cd frontend 2>/dev/null || true
        print_status "Stopping Docker containers..."
        # Use docker compose (v2) instead of docker-compose if available
        if command -v "docker compose" > /dev/null 2>&1; then
            docker compose down --remove-orphans 2>/dev/null || true
        else
            docker-compose down --remove-orphans 2>/dev/null || true
        fi
        cd .. 2>/dev/null || true
        print_success "Docker containers stopped"
    fi
    
    print_success "TEMPO development environment stopped"
}

# Set up signal handlers for cleanup
trap cleanup EXIT INT TERM

# Start services based on options
if [ "$BACKEND_ONLY" = true ]; then
    start_backend
    if [ $? -eq 0 ]; then
        print_success "Backend-only mode: Press Ctrl+C to stop"
        wait
    fi
elif [ "$FRONTEND_ONLY" = true ]; then
    start_frontend
    if [ $? -eq 0 ]; then
        print_success "Frontend-only mode: Press Ctrl+C to stop"
        wait
    fi
else
    # Start both backend and frontend
    start_backend
    if [ $? -eq 0 ]; then
        start_frontend
        if [ $? -eq 0 ]; then
            # Print final status
            echo ""
            print_success "TEMPO development environment is running!"
            echo ""
            print_status "Available URLs:"
            if [ "$PRODUCTION" = true ]; then
                echo "  Frontend: http://localhost:3000"
            else
                echo "  Frontend: http://localhost:5173"
            fi
            echo "  Backend API: http://localhost:8000"
            echo "  API Documentation: http://localhost:8000/docs"
            echo ""
            print_status "Press Ctrl+C to stop all servers"
            
            # Wait for user interrupt
            wait
        else
            print_error "Failed to start frontend"
            exit 1
        fi
    else
        print_error "Failed to start backend. Aborting."
        exit 1
    fi
fi