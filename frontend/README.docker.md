# TEMPO Frontend - Docker Setup

This guide explains how to run the TEMPO frontend in Docker while keeping the backend/model running natively on your Mac (to access GPU acceleration).

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│  Docker         │    │  Native Mac     │
│  Frontend       │────│  Backend + Model│
│  (Port 5173)    │    │  (Port 8000)    │
└─────────────────┘    └─────────────────┘
```

## Prerequisites

- Docker Desktop for Mac
- Backend API running on your Mac (port 8000)

## Quick Start

### Development Mode (Hot Reload)

```bash
# Build and start development container
npm run docker:dev

# Or manually
docker-compose up frontend-dev
```

The frontend will be available at: http://localhost:5173

### Production Mode

```bash
# Build and start production container
npm run docker:prod

# Or manually
docker-compose up frontend-prod
```

The frontend will be available at: http://localhost:3000

## Commands

| Command | Description |
|---------|-------------|
| `npm run docker:dev` | Start development container with hot reload |
| `npm run docker:prod` | Start production container |
| `npm run docker:build` | Build Docker images |
| `npm run docker:down` | Stop and remove containers |
| `npm run docker:logs` | View container logs |
| `npm run docker:shell` | Access container shell |

## How It Works

### Network Configuration

The Docker container uses `host.docker.internal` to communicate with services running on your Mac:

- **Development**: Vite dev server proxies `/api/*` to `http://host.docker.internal:8000`
- **Production**: Static files served by `serve` with proxy configuration

### Environment Variables

- `DOCKER_ENV=true`: Enables Docker-specific networking
- `NODE_ENV=development|production`: Sets environment mode

### Volume Mounting

In development mode, your source code is mounted into the container for hot reload:

```yaml
volumes:
  - .:/app
  - /app/node_modules  # Prevent overwriting
```

## Starting the Backend

Before running the frontend container, ensure your backend is running:

```bash
# In the main TEMPO directory
cd /path/to/tempo
source .venv/bin/activate
uvicorn api:app --reload --port 8000
```

## Troubleshooting

### Backend Connection Issues

1. **Check backend is running**: Visit http://localhost:8000/docs
2. **Check Docker networking**: 
   ```bash
   docker-compose exec frontend-dev nslookup host.docker.internal
   ```
3. **Check proxy configuration**: Look for proxy logs in container output

### Port Conflicts

- Development: Port 5173
- Production: Port 3000
- Backend: Port 8000

If ports are busy, modify `docker-compose.yml`:

```yaml
ports:
  - "5174:5173"  # Use different external port
```

### Build Issues

```bash
# Clean rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Container Logs

```bash
# View logs
npm run docker:logs

# Follow specific service
docker-compose logs -f frontend-dev
```

## Development Workflow

1. **Start backend** on your Mac (for GPU access)
2. **Start frontend** in Docker:
   ```bash
   npm run docker:dev
   ```
3. **Edit code** - changes auto-reload in container
4. **Access app** at http://localhost:5173

## Production Deployment

For production, the container serves pre-built static files:

1. **Build production image**:
   ```bash
   npm run docker:build
   ```

2. **Start production container**:
   ```bash
   npm run docker:prod
   ```

3. **Access app** at http://localhost:3000

## Health Checks

Both containers include health checks:

```bash
# Check container health
docker-compose ps
```

Healthy containers show "Up" status with "(healthy)" indicator.

## Performance Notes

- **Development**: Slower due to volume mounting and Node.js dev server
- **Production**: Fast static file serving with optimized builds
- **Native backend**: Full GPU acceleration on Mac

## Security

- Containers run as non-root user (`frontend:nodejs`)
- No sensitive data stored in containers
- Communication over localhost only