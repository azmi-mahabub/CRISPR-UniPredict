# CRISPR-UniPredict Inference API Guide

## Overview

The `inference_api.py` provides a FastAPI-based REST API for CRISPR-UniPredict model predictions with comprehensive documentation, input validation, and error handling.

---

## Quick Start

### Installation

```bash
pip install fastapi uvicorn pydantic torch
```

### Launch API

```bash
# From project root
uvicorn api.inference_api:app --host 0.0.0.0 --port 8000

# Or with auto-reload
uvicorn api.inference_api:app --host 0.0.0.0 --port 8000 --reload
```

### Access API

- **API Docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## API Endpoints

### 1. Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "timestamp": "2025-11-22T01:06:00"
}
```

### 2. Model Information

```bash
GET /model/info
```

**Response:**
```json
{
  "model_name": "CRISPR-UniPredict",
  "device": "cuda",
  "model_loaded": true,
  "parameters": 1992565,
  "input_length": 23,
  "tasks": ["on_target", "off_target"]
}
```

### 3. On-Target Prediction

```bash
POST /predict/on_target
```

**Request:**
```json
{
  "sgrna": "GCTAGCTAGCTAGCTAGCTAGCT",
  "target": null
}
```

**Response:**
```json
{
  "efficiency_score": 0.8234,
  "confidence": 0.92,
  "sgrna": "GCTAGCTAGCTAGCTAGCTAGCT"
}
```

### 4. Off-Target Prediction

```bash
POST /predict/off_target
```

**Request:**
```json
{
  "sgrna": "GCTAGCTAGCTAGCTAGCTAGCT",
  "target": "ATGCATGCATGCATGCATGCATG"
}
```

**Response:**
```json
{
  "off_target_prob": 0.1234,
  "is_off_target": false,
  "risk_level": "low",
  "sgrna": "GCTAGCTAGCTAGCTAGCTAGCT"
}
```

### 5. Comprehensive Prediction

```bash
POST /predict/comprehensive
```

**Request:**
```json
{
  "sgrna": "GCTAGCTAGCTAGCTAGCTAGCT",
  "target": "ATGCATGCATGCATGCATGCATG"
}
```

**Response:**
```json
{
  "on_target_score": 0.8234,
  "off_target_risk": 0.1234,
  "comprehensive_score": 0.7248,
  "recommendation": "Good sgRNA",
  "sgrna": "GCTAGCTAGCTAGCTAGCTAGCT"
}
```

### 6. Batch Prediction

```bash
POST /batch_predict
```

**Request:**
```json
{
  "sgrnas": [
    "GCTAGCTAGCTAGCTAGCTAGCT",
    "ATGCATGCATGCATGCATGCATG",
    "CCGGCCGGCCGGCCGGCCGGCCG"
  ],
  "targets": [
    "ATGCATGCATGCATGCATGCATG",
    "GCTAGCTAGCTAGCTAGCTAGCT",
    "TTAATTAATTAATTAATTAATTAA"
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "on_target_score": 0.8234,
      "off_target_risk": 0.1234,
      "comprehensive_score": 0.7248,
      "recommendation": "Good sgRNA",
      "sgrna": "GCTAGCTAGCTAGCTAGCTAGCT"
    },
    ...
  ],
  "total_processed": 3,
  "processing_time_ms": 245.32
}
```

---

## Python Client Examples

### Using requests library

```python
import requests

BASE_URL = "http://localhost:8000"

# Single on-target prediction
response = requests.post(
    f"{BASE_URL}/predict/on_target",
    json={"sgrna": "GCTAGCTAGCTAGCTAGCTAGCT"}
)
print(response.json())

# Comprehensive prediction
response = requests.post(
    f"{BASE_URL}/predict/comprehensive",
    json={
        "sgrna": "GCTAGCTAGCTAGCTAGCTAGCT",
        "target": "ATGCATGCATGCATGCATGCATG"
    }
)
result = response.json()
print(f"Score: {result['comprehensive_score']:.4f}")
print(f"Recommendation: {result['recommendation']}")
```

### Using httpx (async)

```python
import httpx
import asyncio

async def predict():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/predict/comprehensive",
            json={
                "sgrna": "GCTAGCTAGCTAGCTAGCTAGCT",
                "target": "ATGCATGCATGCATGCATGCATG"
            }
        )
        return response.json()

result = asyncio.run(predict())
print(result)
```

### Batch predictions

```python
import requests

sgrnas = [
    "GCTAGCTAGCTAGCTAGCTAGCT",
    "ATGCATGCATGCATGCATGCATG",
    "CCGGCCGGCCGGCCGGCCGGCCG"
]

response = requests.post(
    "http://localhost:8000/batch_predict",
    json={"sgrnas": sgrnas}
)

results = response.json()
for pred in results['predictions']:
    print(f"{pred['sgrna']}: {pred['recommendation']}")
```

---

## cURL Examples

### On-target prediction

```bash
curl -X POST "http://localhost:8000/predict/on_target" \
  -H "Content-Type: application/json" \
  -d '{"sgrna": "GCTAGCTAGCTAGCTAGCTAGCT"}'
```

### Comprehensive prediction

```bash
curl -X POST "http://localhost:8000/predict/comprehensive" \
  -H "Content-Type: application/json" \
  -d '{
    "sgrna": "GCTAGCTAGCTAGCTAGCTAGCT",
    "target": "ATGCATGCATGCATGCATGCATG"
  }'
```

### Batch prediction

```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sgrnas": ["GCTAGCTAGCTAGCTAGCTAGCT", "ATGCATGCATGCATGCATGCATG"],
    "targets": ["ATGCATGCATGCATGCATGCATG", "GCTAGCTAGCTAGCTAGCTAGCT"]
  }'
```

---

## Input Validation

### sgRNA Validation
- Length: 20-25 bp
- Characters: Only ACGT (case-insensitive)
- Required: Yes

### Target Validation
- Characters: Only ACGT (case-insensitive)
- Required: No (optional for on-target only)
- Required: Yes (for off-target prediction)

### Batch Validation
- Maximum sequences: 1000 per batch
- Minimum sequences: 1

---

## Error Handling

### Common Errors

**400 Bad Request**: Invalid input
```json
{
  "detail": "sgRNA must contain only ACGT nucleotides"
}
```

**503 Service Unavailable**: Model not loaded
```json
{
  "detail": "Model not loaded"
}
```

**500 Internal Server Error**: Prediction failed
```json
{
  "detail": "Prediction failed: [error details]"
}
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build
docker build -t crispr-unipredict-api .

# Run
docker run -p 8000:8000 crispr-unipredict-api
```

### Production (Gunicorn + Uvicorn)

```bash
pip install gunicorn

gunicorn api.inference_api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Performance Tips

### 1. Batch Processing
Use batch endpoint for multiple predictions:
```python
# Good: Batch 100 sequences
response = requests.post(
    "http://localhost:8000/batch_predict",
    json={"sgrnas": [sgrna1, sgrna2, ..., sgrna100]}
)

# Avoid: Individual requests
for sgrna in sgrnas:
    requests.post("http://localhost:8000/predict/on_target", ...)
```

### 2. Connection Pooling
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)

response = session.post("http://localhost:8000/predict/comprehensive", ...)
```

### 3. GPU Acceleration
API automatically uses GPU if available. Check with:
```bash
curl http://localhost:8000/model/info
```

---

## Summary

The API provides:
- ✅ **RESTful endpoints** for all prediction tasks
- ✅ **Input validation** for robustness
- ✅ **Error handling** with informative messages
- ✅ **Batch processing** for efficiency
- ✅ **Automatic documentation** (Swagger UI)
- ✅ **CORS support** for cross-origin requests
- ✅ **Production-ready** with proper logging

Perfect for integration with web applications and pipelines!
