# CRISPR-UniPredict Deployment Guide

## Overview

Complete guide for deploying CRISPR-UniPredict using Docker and Docker Compose with optional Nginx reverse proxy.

---

## Prerequisites

### Required
- Docker (≥20.10)
- Docker Compose (≥1.29)
- NVIDIA Docker Runtime (for GPU support)
- 8GB+ RAM
- GPU with CUDA 11.8 support (optional but recommended)

### Installation

**Ubuntu/Debian:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**macOS:**
```bash
# Install Docker Desktop (includes Docker and Compose)
brew install --cask docker

# Start Docker Desktop
open /Applications/Docker.app
```

**Windows:**
```bash
# Install Docker Desktop for Windows
# Download from: https://www.docker.com/products/docker-desktop
```

---

## Quick Start

### 1. Build Docker Image

```bash
# Navigate to project directory
cd /path/to/CRISPR-UniPredict

# Build image
docker-compose build

# Or build with specific tag
docker build -t crispr-unipredict:latest .
```

### 2. Run Container

```bash
# Start services
docker-compose up

# Or run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 3. Access Services

- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health
- **Web UI**: http://localhost:8501

---

## Detailed Deployment

### Option 1: Docker Compose (Recommended)

**Basic Setup:**

```bash
# Build and start
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f crispr_unipredict

# Stop
docker-compose down
```

**With GPU Support:**

```bash
# Ensure NVIDIA Docker is installed
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi

# Start with GPU
docker-compose up -d

# Verify GPU access
docker exec crispr-unipredict python -c "import torch; print(torch.cuda.is_available())"
```

**With Volume Mounts:**

```bash
# Mount local directories
docker-compose up -d

# Models will be at: ./models/
# Results will be at: ./results/
# Logs will be at: ./logs/
```

### Option 2: Docker Run (Manual)

```bash
# Build image
docker build -t crispr-unipredict:latest .

# Run with GPU
docker run -d \
  --name crispr-unipredict \
  --gpus all \
  -p 8000:8000 \
  -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  -e CUDA_VISIBLE_DEVICES=0 \
  crispr-unipredict:latest

# Run without GPU
docker run -d \
  --name crispr-unipredict \
  -p 8000:8000 \
  -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  crispr-unipredict:latest
```

### Option 3: Kubernetes Deployment

**Create `k8s-deployment.yaml`:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crispr-unipredict
spec:
  replicas: 2
  selector:
    matchLabels:
      app: crispr-unipredict
  template:
    metadata:
      labels:
        app: crispr-unipredict
    spec:
      containers:
      - name: crispr-unipredict
        image: crispr-unipredict:latest
        ports:
        - containerPort: 8000
        - containerPort: 8501
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: models
          mountPath: /app/models
        - name: results
          mountPath: /app/results
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: results
        persistentVolumeClaim:
          claimName: results-pvc
```

**Deploy:**

```bash
kubectl apply -f k8s-deployment.yaml
kubectl get pods
kubectl logs -f deployment/crispr-unipredict
```

---

## Production Setup

### 1. SSL/TLS Configuration

**Generate Self-Signed Certificate:**

```bash
mkdir -p ssl
openssl req -x509 -newkey rsa:4096 -nodes -out ssl/cert.pem -keyout ssl/key.pem -days 365
```

**Or Use Let's Encrypt:**

```bash
# Install certbot
sudo apt-get install certbot

# Generate certificate
sudo certbot certonly --standalone -d your-domain.com

# Copy to ssl directory
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/key.pem
```

### 2. Nginx Reverse Proxy

**Enable Nginx:**

```bash
# Uncomment nginx service in docker-compose.yml
# Then start
docker-compose up -d nginx
```

**Access via Nginx:**

```
https://localhost/docs      # API docs
https://localhost/          # Web UI
https://localhost/api/...   # API endpoints
```

### 3. Environment Configuration

**Create `.env` file:**

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Web Configuration
WEB_HOST=0.0.0.0
WEB_PORT=8501

# Model Configuration
MODEL_CHECKPOINT=models/checkpoints/best.pt
DEVICE=cuda

# Logging
LOG_LEVEL=INFO
```

**Use in docker-compose:**

```yaml
services:
  crispr_unipredict:
    env_file: .env
```

### 4. Monitoring and Logging

**Docker Logs:**

```bash
# View logs
docker-compose logs -f

# View specific service
docker-compose logs -f crispr_unipredict

# Save logs
docker-compose logs > deployment.log
```

**Health Monitoring:**

```bash
# Check health
curl http://localhost:8000/health

# Monitor in loop
watch -n 5 'curl http://localhost:8000/health'
```

---

## Common Tasks

### Update Model Checkpoint

```bash
# Copy new checkpoint
cp /path/to/new/checkpoint.pt models/checkpoints/best.pt

# Restart container
docker-compose restart crispr_unipredict
```

### Scale Services

```bash
# Run multiple API instances
docker-compose up -d --scale api=3

# With load balancing (requires custom docker-compose)
```

### Backup Data

```bash
# Backup results
docker cp crispr-unipredict:/app/results ./backup/results_$(date +%Y%m%d)

# Backup models
docker cp crispr-unipredict:/app/models ./backup/models_$(date +%Y%m%d)
```

### Clean Up

```bash
# Remove container
docker-compose down

# Remove images
docker rmi crispr-unipredict:latest

# Remove volumes
docker volume prune

# Full cleanup
docker system prune -a
```

---

## Troubleshooting

### Issue: GPU Not Detected

**Solution:**

```bash
# Check NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi

# Verify in container
docker exec crispr-unipredict nvidia-smi

# Check CUDA availability
docker exec crispr-unipredict python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Port Already in Use

**Solution:**

```bash
# Find process using port
lsof -i :8000
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use different ports
docker-compose up -d -p 8001:8000 -p 8502:8501
```

### Issue: Out of Memory

**Solution:**

```bash
# Increase Docker memory limit
# Edit docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 16G

# Or restart with memory limit
docker run -m 16g crispr-unipredict:latest
```

### Issue: Model Not Loading

**Solution:**

```bash
# Check model file exists
docker exec crispr-unipredict ls -la models/checkpoints/

# Check logs
docker-compose logs crispr_unipredict | grep -i error

# Verify model path in config
docker exec crispr-unipredict cat configs/model_config.yaml
```

---

## Performance Optimization

### 1. GPU Optimization

```yaml
# In docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=0
  - TORCH_CUDA_MAX_MEMORY_ALLOCATED=8000000000
```

### 2. API Workers

```bash
# Increase workers for better throughput
uvicorn api.inference_api:app --workers 4 --host 0.0.0.0 --port 8000
```

### 3. Caching

```bash
# Enable model caching
docker exec crispr-unipredict python -c "
from api.inference_api import model_manager
model_manager.load_model()
"
```

---

## Security Best Practices

### 1. Use Secrets

```bash
# Create secret file
echo "your-secret-key" > .secrets

# Use in docker-compose
secrets:
  api_key:
    file: .secrets
```

### 2. Network Isolation

```yaml
# In docker-compose.yml
networks:
  internal:
    internal: true
  external:
```

### 3. Rate Limiting

```bash
# Already configured in nginx.conf
# API: 10 requests/second
# Web: 30 requests/second
```

---

## Summary

The deployment provides:
- ✅ **Easy setup** with Docker Compose
- ✅ **GPU support** with NVIDIA Docker
- ✅ **Reverse proxy** with Nginx
- ✅ **SSL/TLS** support
- ✅ **Health checks** and monitoring
- ✅ **Volume management** for persistence
- ✅ **Production-ready** configuration
- ✅ **Kubernetes** support

Perfect for research, production, and cloud deployment!
