# TEMPO Deployment Guide

This guide covers deploying TEMPO in production environments, including containerization, scaling, monitoring, and best practices.

## Deployment Options

### 1. Standalone Server
- Single machine deployment
- Suitable for small-scale usage
- Easy to set up and maintain

### 2. Containerized Deployment
- Docker/Kubernetes deployment
- Better resource isolation
- Easier scaling and management

### 3. Cloud Deployment
- AWS, GCP, or Azure
- Managed container services
- Auto-scaling capabilities

## Docker Deployment

### Creating a Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 tempo && chown -R tempo:tempo /app
USER tempo

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TEMPO_API_HOST=0.0.0.0
ENV TEMPO_API_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Multi-stage Build for Smaller Images

```dockerfile
# Dockerfile.multistage
# Build stage
FROM python:3.10-slim as builder

RUN apt-get update && apt-get install -y \
    build-essential \
    git

WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Runtime stage
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy application
COPY . .
RUN pip install -e .

# Create user and set permissions
RUN useradd -m -u 1000 tempo && chown -R tempo:tempo /app
USER tempo

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  tempo-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - TEMPO_MODEL_MODEL_ID=deepcogito/cogito-v1-preview-llama-3B
      - TEMPO_MODEL_DEVICE=cuda
      - TEMPO_API_MAX_CONCURRENT_REQUESTS=5
      - TEMPO_LOGGING_LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  tempo-frontend:
    build: ./frontend
    ports:
      - "5174:5174"
    depends_on:
      - tempo-api
    environment:
      - VITE_API_URL=http://tempo-api:8000

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

volumes:
  redis-data:
  prometheus-data:
```

### Building and Running

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f tempo-api

# Scale API instances
docker-compose up -d --scale tempo-api=3

# Stop services
docker-compose down
```

## Kubernetes Deployment

### Deployment Configuration

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tempo-api
  labels:
    app: tempo
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tempo-api
  template:
    metadata:
      labels:
        app: tempo-api
    spec:
      containers:
      - name: tempo
        image: your-registry/tempo:latest
        ports:
        - containerPort: 8000
        env:
        - name: TEMPO_MODEL_MODEL_ID
          value: "deepcogito/cogito-v1-preview-llama-3B"
        - name: TEMPO_MODEL_DEVICE
          value: "cuda"
        - name: TEMPO_API_MAX_CONCURRENT_REQUESTS
          value: "5"
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
        - name: config
          mountPath: /app/config
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: config
        configMap:
          name: tempo-config
```

### Service Configuration

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: tempo-api
spec:
  selector:
    app: tempo-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### ConfigMap for Configuration

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: tempo-config
data:
  config.json: |
    {
      "logging": {
        "log_level": "INFO",
        "enable_file_logging": true
      },
      "model": {
        "model_id": "deepcogito/cogito-v1-preview-llama-3B",
        "device": "cuda",
        "torch_dtype": "float16"
      },
      "api": {
        "max_concurrent_requests": 5,
        "rate_limit": 100
      }
    }
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tempo-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tempo-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Production Configuration

### Environment Variables

```bash
# Production environment variables
export TEMPO_ENVIRONMENT=production
export TEMPO_MODEL_MODEL_ID=deepcogito/cogito-v1-preview-llama-3B
export TEMPO_MODEL_DEVICE=cuda
export TEMPO_MODEL_QUANTIZATION=4bit  # For memory efficiency
export TEMPO_API_HOST=0.0.0.0
export TEMPO_API_PORT=8000
export TEMPO_API_WORKERS=4
export TEMPO_API_MAX_CONCURRENT_REQUESTS=10
export TEMPO_API_REQUEST_TIMEOUT=300
export TEMPO_API_RATE_LIMIT=1000
export TEMPO_LOGGING_LOG_LEVEL=WARNING
export TEMPO_LOGGING_LOG_DIR=/var/log/tempo
export TEMPO_CACHE_ENABLE_PROMPT_CACHE=true
export TEMPO_CACHE_CACHE_DIR=/var/cache/tempo
export TEMPO_PERFORMANCE_MEMORY_EFFICIENT_MODE=true
```

### Nginx Configuration

```nginx
# /etc/nginx/sites-available/tempo
upstream tempo_backend {
    least_conn;
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name api.tempo.example.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.tempo.example.com;
    
    ssl_certificate /etc/ssl/certs/tempo.crt;
    ssl_certificate_key /etc/ssl/private/tempo.key;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # API rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # WebSocket support
    map $http_upgrade $connection_upgrade {
        default upgrade;
        '' close;
    }
    
    location / {
        proxy_pass http://tempo_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        # Buffer settings
        proxy_buffering off;
        proxy_request_buffering off;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://tempo_backend;
    }
    
    # Static files (if any)
    location /static {
        alias /var/www/tempo/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

## Monitoring and Logging

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'tempo-api'
    static_configs:
      - targets: ['tempo-api:9090']
    metrics_path: '/metrics'

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'gpu'
    static_configs:
      - targets: ['gpu-exporter:9835']
```

### Grafana Dashboard

Create dashboards monitoring:
- Request rate and latency
- Token generation rate
- GPU utilization
- Memory usage
- Error rates
- Cache hit rates

### Logging Configuration

```python
# logging_config.py
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/tempo/api.log',
            'maxBytes': 104857600,  # 100MB
            'backupCount': 10,
            'formatter': 'json'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file', 'console']
    }
}
```

## Scaling Strategies

### Horizontal Scaling

1. **API Layer**: Scale API instances behind load balancer
2. **Model Serving**: Use model server pools
3. **Caching**: Distributed cache with Redis/Memcached
4. **Queue-based**: Use message queues for async processing

### Vertical Scaling

1. **GPU Upgrade**: Use more powerful GPUs (A100, H100)
2. **Memory**: Increase RAM for larger batch sizes
3. **CPU**: More cores for preprocessing
4. **Storage**: Fast SSDs for model loading

### Model Optimization

1. **Quantization**: Use 4-bit or 8-bit quantization
2. **Model Sharding**: Distribute model across GPUs
3. **Compilation**: Use torch.compile() for speedup
4. **Flash Attention**: Enable for memory efficiency

## Security Considerations

### API Security

```python
# security.py
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.JWTError:
        raise HTTPException(status_code=403, detail="Invalid token")

# Use in endpoints
@app.post("/api/generate")
async def generate(request: GenerationRequest, user=Depends(verify_token)):
    # Authorized request processing
    pass
```

### Network Security

1. **TLS/SSL**: Encrypt all traffic
2. **Firewall**: Restrict access to necessary ports
3. **VPN**: Use for internal communications
4. **API Keys**: Implement key-based authentication
5. **Rate Limiting**: Prevent abuse
6. **Input Validation**: Sanitize all inputs

## Performance Tuning

### System Tuning

```bash
# Increase file descriptor limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize TCP settings
echo "net.core.somaxconn = 65536" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65536" >> /etc/sysctl.conf
sysctl -p
```

### Application Tuning

```python
# Async optimization
import asyncio
import uvloop

# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Connection pooling
from aiohttp import ClientSession, TCPConnector

connector = TCPConnector(
    limit=100,
    limit_per_host=30,
    ttl_dns_cache=300
)
session = ClientSession(connector=connector)
```

## Backup and Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backup/tempo"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup models
tar -czf "$BACKUP_DIR/models_$DATE.tar.gz" /app/models

# Backup configurations
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" /app/config

# Backup logs
tar -czf "$BACKUP_DIR/logs_$DATE.tar.gz" /var/log/tempo

# Clean old backups (keep 7 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
```

### Recovery Procedure

1. Stop services
2. Restore from backup
3. Verify configurations
4. Test functionality
5. Resume services

## Maintenance

### Health Checks

```python
# health.py
@app.get("/health/detailed")
async def detailed_health():
    checks = {
        "api": "healthy",
        "model_loaded": model_manager.is_loaded(),
        "gpu_available": torch.cuda.is_available(),
        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "active_requests": request_counter.value,
        "cache_hit_rate": cache_manager.hit_rate(),
        "uptime_seconds": time.time() - start_time
    }
    
    status = "healthy" if all([
        checks["model_loaded"],
        checks["memory_usage_mb"] < 15000,  # 15GB threshold
        checks["active_requests"] < 100
    ]) else "degraded"
    
    return {"status": status, "checks": checks}
```

### Rolling Updates

```bash
# rolling_update.sh
#!/bin/bash
INSTANCES=3
SERVICE="tempo-api"

for i in $(seq 1 $INSTANCES); do
    echo "Updating instance $i..."
    
    # Stop instance
    docker-compose stop ${SERVICE}_$i
    
    # Update and start
    docker-compose pull ${SERVICE}
    docker-compose up -d ${SERVICE}_$i
    
    # Wait for health check
    until curl -f http://localhost:800$i/health; do
        sleep 5
    done
    
    echo "Instance $i updated successfully"
    sleep 30  # Wait before next instance
done
```

## Troubleshooting Production Issues

### Common Issues

1. **Out of Memory**: Reduce batch size, enable quantization
2. **Slow Response**: Check GPU utilization, enable caching
3. **Connection Errors**: Verify network settings, check timeouts
4. **Model Loading**: Ensure sufficient disk space, check permissions

### Debug Commands

```bash
# Check GPU status
nvidia-smi

# Monitor system resources
htop

# Check docker logs
docker logs tempo-api --tail 100 -f

# Analyze API performance
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/api/generate

# Check disk usage
df -h

# Monitor network connections
netstat -tulpn | grep 8000
```

## Best Practices

1. **Always use environment-specific configs**
2. **Implement comprehensive monitoring**
3. **Set up automated backups**
4. **Use health checks and readiness probes**
5. **Implement graceful shutdown**
6. **Document your deployment process**
7. **Test disaster recovery procedures**
8. **Keep dependencies updated**
9. **Use CI/CD for deployments**
10. **Monitor costs in cloud environments**