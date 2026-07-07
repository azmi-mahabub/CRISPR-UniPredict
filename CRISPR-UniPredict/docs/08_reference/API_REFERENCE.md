# CRISPR-UniPredict API Reference

Complete API documentation for CRISPR-UniPredict including Python API, REST API, and CLI.

---

## Table of Contents

1. [Python API](#python-api)
2. [REST API](#rest-api)
3. [CLI](#cli)
4. [Data Types](#data-types)
5. [Error Handling](#error-handling)
6. [Examples](#examples)

---

## Python API

### Model Class: CRISPRUniPredict

#### Initialization

```python
from models.crispr_unipredict import CRISPRUniPredict

# Initialize new model
model = CRISPRUniPredict(
    device='cuda',           # 'cuda' or 'cpu'
    msc_channels=64,         # MSC output channels
    mhsa_embed_dim=256,      # MHSA embedding dimension
    bigru_hidden=128,        # BiGRU hidden dimension
    dropout=0.35             # Dropout rate
)
```

#### Load Pretrained Model

```python
# Load from checkpoint
model = CRISPRUniPredict.load_pretrained(
    checkpoint_path='models/checkpoints/best.pt',
    device='cuda'
)

# Or load state dict
checkpoint = torch.load('models/checkpoints/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

#### Prediction Methods

##### predict_on_target()

Predict on-target efficiency for a single sgRNA.

```python
score = model.predict_on_target(
    sgrna="GCTAGCTAGCTAGCTAGCTAGCT",
    target=None  # optional
)
# Returns: float (0-1)
# Example: 0.8234
```

**Parameters:**
- `sgrna` (str): sgRNA sequence (20-23 bp, ACGT only)
- `target` (str, optional): Target sequence (ACGT only)

**Returns:**
- float: On-target efficiency score (0-1)

**Raises:**
- ValueError: If sequence format invalid

##### predict_off_target()

Predict off-target risk for sgRNA-target pair.

```python
result = model.predict_off_target(
    sgrna="GCTAGCTAGCTAGCTAGCTAGCT",
    target="ATGCATGCATGCATGCATGCATG"  # required
)
# Returns: {
#     "probability": 0.1234,
#     "is_off_target": False,
#     "risk_level": "low"
# }
```

**Parameters:**
- `sgrna` (str): sgRNA sequence (20-23 bp)
- `target` (str): Target sequence (required)

**Returns:**
- dict: Off-target prediction with keys:
  - `probability` (float): Off-target probability (0-1)
  - `is_off_target` (bool): Whether high off-target risk
  - `risk_level` (str): "low", "medium", or "high"

##### predict_comprehensive()

Comprehensive evaluation combining both tasks.

```python
result = model.predict_comprehensive(
    sgrna="GCTAGCTAGCTAGCTAGCTAGCT",
    target="ATGCATGCATGCATGCATGCATG"
)
# Returns: {
#     "on_target_score": 0.8234,
#     "off_target_risk": 0.1234,
#     "off_target_safety": 0.8766,
#     "comprehensive_score": 0.7218,
#     "recommendation": "Good sgRNA"
# }
```

**Parameters:**
- `sgrna` (str): sgRNA sequence (20-23 bp)
- `target` (str): Target sequence

**Returns:**
- dict: Comprehensive prediction with keys:
  - `on_target_score` (float): On-target efficiency (0-1)
  - `off_target_risk` (float): Off-target probability (0-1)
  - `off_target_safety` (float): 1 - off_target_risk
  - `comprehensive_score` (float): on_target × off_target_safety
  - `recommendation` (str): "Excellent", "Good", "Acceptable", or "Poor"

#### Batch Processing

##### batch_predict()

Process multiple sgRNA-target pairs.

```python
results = model.batch_predict(
    sgrnas=["GCTAGCTAGCTAGCTAGCTAGCT", "ATGCATGCATGCATGCATGCATG"],
    targets=["ATGCATGCATGCATGCATGCATG", "GCTAGCTAGCTAGCTAGCTAGCT"],
    batch_size=32,
    return_comprehensive=True
)
# Returns: List[dict]
```

**Parameters:**
- `sgrnas` (List[str]): List of sgRNA sequences
- `targets` (List[str]): List of target sequences
- `batch_size` (int): Batch size for processing (default: 32)
- `return_comprehensive` (bool): Return comprehensive scores (default: True)

**Returns:**
- List[dict]: List of prediction dictionaries

#### Attention Visualization

##### get_attention_weights()

Extract attention weights from model.

```python
attention_weights = model.get_attention_weights(
    sgrna="GCTAGCTAGCTAGCTAGCTAGCT",
    target="ATGCATGCATGCATGCATGCATG"
)
# Returns: torch.Tensor (shape: [4, 23])
```

**Parameters:**
- `sgrna` (str): sgRNA sequence
- `target` (str): Target sequence

**Returns:**
- torch.Tensor: Attention weights (nucleotides × positions)

##### plot_attention()

Visualize attention weights.

```python
model.plot_attention(
    sgrna="GCTAGCTAGCTAGCTAGCTAGCT",
    target="ATGCATGCATGCATGCATGCATG",
    save_path="attention.png",
    figsize=(10, 6)
)
```

**Parameters:**
- `sgrna` (str): sgRNA sequence
- `target` (str): Target sequence
- `save_path` (str, optional): Path to save figure
- `figsize` (tuple): Figure size (default: (10, 6))

#### Model Information

##### get_model_info()

Get model configuration and parameters.

```python
info = model.get_model_info()
# Returns: {
#     "name": "CRISPRUniPredict",
#     "parameters": 1992565,
#     "device": "cuda",
#     "branches": ["MSC", "BiGRU", "RNA-FM"],
#     "tasks": ["on_target", "off_target"]
# }
```

**Returns:**
- dict: Model information

##### get_trainable_params()

Get number of trainable parameters.

```python
trainable = model.get_trainable_params()
# Returns: 1234567
```

**Returns:**
- int: Number of trainable parameters

---

## REST API

### Base URL

```
http://localhost:8000
```

### Authentication

No authentication required for basic endpoints. Rate limiting applies (see Deployment Guide).

### Common Response Format

**Success Response:**
```json
{
  "status": "success",
  "data": {...},
  "timestamp": "2025-11-22T01:18:00Z"
}
```

**Error Response:**
```json
{
  "status": "error",
  "error": "Error message",
  "code": 400,
  "timestamp": "2025-11-22T01:18:00Z"
}
```

### Endpoints

#### GET /health

Health check endpoint.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "timestamp": "2025-11-22T01:18:00Z"
}
```

#### GET /model/info

Get model information.

**Request:**
```bash
curl http://localhost:8000/model/info
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

#### POST /predict/on_target

Predict on-target efficiency.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/on_target" \
  -H "Content-Type: application/json" \
  -d '{
    "sgrna": "GCTAGCTAGCTAGCTAGCTAGCT",
    "target": null
  }'
```

**Request Body:**
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

**Parameters:**
- `sgrna` (string, required): sgRNA sequence (20-23 bp)
- `target` (string, optional): Target sequence

**Returns:**
- `efficiency_score` (number): On-target efficiency (0-1)
- `confidence` (number): Prediction confidence (0-1)
- `sgrna` (string): Input sgRNA sequence

#### POST /predict/off_target

Predict off-target risk.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/off_target" \
  -H "Content-Type: application/json" \
  -d '{
    "sgrna": "GCTAGCTAGCTAGCTAGCTAGCT",
    "target": "ATGCATGCATGCATGCATGCATG"
  }'
```

**Request Body:**
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

**Parameters:**
- `sgrna` (string, required): sgRNA sequence (20-23 bp)
- `target` (string, required): Target sequence

**Returns:**
- `off_target_prob` (number): Off-target probability (0-1)
- `is_off_target` (boolean): Whether high off-target risk
- `risk_level` (string): "low", "medium", or "high"
- `sgrna` (string): Input sgRNA sequence

#### POST /predict/comprehensive

Comprehensive sgRNA evaluation.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/comprehensive" \
  -H "Content-Type: application/json" \
  -d '{
    "sgrna": "GCTAGCTAGCTAGCTAGCTAGCT",
    "target": "ATGCATGCATGCATGCATGCATG"
  }'
```

**Request Body:**
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
  "comprehensive_score": 0.7218,
  "recommendation": "Good sgRNA",
  "sgrna": "GCTAGCTAGCTAGCTAGCTAGCT"
}
```

**Parameters:**
- `sgrna` (string, required): sgRNA sequence (20-23 bp)
- `target` (string, required): Target sequence

**Returns:**
- `on_target_score` (number): On-target efficiency (0-1)
- `off_target_risk` (number): Off-target probability (0-1)
- `comprehensive_score` (number): Combined score (0-1)
- `recommendation` (string): "Excellent", "Good", "Acceptable", or "Poor"
- `sgrna` (string): Input sgRNA sequence

#### POST /batch_predict

Batch prediction for multiple sequences.

**Request:**
```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sgrnas": ["GCTAGCTAGCTAGCTAGCTAGCT", "ATGCATGCATGCATGCATGCATG"],
    "targets": ["ATGCATGCATGCATGCATGCATG", "GCTAGCTAGCTAGCTAGCTAGCT"]
  }'
```

**Request Body:**
```json
{
  "sgrnas": [
    "GCTAGCTAGCTAGCTAGCTAGCT",
    "ATGCATGCATGCATGCATGCATG"
  ],
  "targets": [
    "ATGCATGCATGCATGCATGCATG",
    "GCTAGCTAGCTAGCTAGCTAGCT"
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
      "comprehensive_score": 0.7218,
      "recommendation": "Good sgRNA",
      "sgrna": "GCTAGCTAGCTAGCTAGCTAGCT"
    },
    {
      "on_target_score": 0.7856,
      "off_target_risk": 0.2345,
      "comprehensive_score": 0.6089,
      "recommendation": "Acceptable sgRNA",
      "sgrna": "ATGCATGCATGCATGCATGCATG"
    }
  ],
  "total_processed": 2,
  "processing_time_ms": 245.32
}
```

**Parameters:**
- `sgrnas` (array, required): List of sgRNA sequences (max 1000)
- `targets` (array, optional): List of target sequences

**Returns:**
- `predictions` (array): List of prediction objects
- `total_processed` (integer): Number of sequences processed
- `processing_time_ms` (number): Processing time in milliseconds

---

## CLI

### predict.py

Command-line prediction script.

**Usage:**
```bash
python scripts/predict.py [OPTIONS]
```

**Options:**
```
--sgrna TEXT              sgRNA sequence (20-23 bp)
--target TEXT             Target sequence (optional)
--input FILE              Input CSV file
--output FILE             Output CSV file
--batch-size INTEGER      Batch size (default: 32)
--device TEXT             Device: cuda or cpu (default: cuda)
--help                    Show help message
```

**Examples:**

Single prediction:
```bash
python scripts/predict.py --sgrna GCTAGCTAGCTAGCTAGCTAGCT --target ATGCATGCATGCATGCATGCATG
```

Batch prediction:
```bash
python scripts/predict.py --input predictions.csv --output results.csv --batch-size 64
```

---

## Data Types

### Sequence Format

**sgRNA:**
- Length: 20-23 bp
- Characters: A, C, G, T (case-insensitive)
- Example: `GCTAGCTAGCTAGCTAGCTAGCT`

**Target:**
- Characters: A, C, G, T (case-insensitive)
- Example: `ATGCATGCATGCATGCATGCATG`

### Score Ranges

| Score | Range | Interpretation |
|-------|-------|-----------------|
| On-Target | 0-1 | 0=low efficiency, 1=high efficiency |
| Off-Target | 0-1 | 0=low risk, 1=high risk |
| Comprehensive | 0-1 | 0=poor, 1=excellent |

### Recommendation Levels

| Score | Recommendation |
|-------|-----------------|
| ≥0.7 | Excellent sgRNA |
| 0.6-0.7 | Good sgRNA |
| 0.4-0.6 | Acceptable sgRNA |
| <0.4 | Poor sgRNA - Not recommended |

---

## Error Handling

### Common Errors

**400 Bad Request:**
```json
{
  "detail": "sgRNA must contain only ACGT nucleotides"
}
```

**422 Unprocessable Entity:**
```json
{
  "detail": [
    {
      "loc": ["body", "sgrna"],
      "msg": "ensure this value has at least 20 characters",
      "type": "value_error.any_str.min_length"
    }
  ]
}
```

**503 Service Unavailable:**
```json
{
  "detail": "Model not loaded"
}
```

### Error Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid input) |
| 422 | Unprocessable entity (validation error) |
| 500 | Internal server error |
| 503 | Service unavailable (model not loaded) |

---

## Examples

### Python Example

```python
import torch
from models.crispr_unipredict import CRISPRUniPredict
from models.encoding import SequenceEncoder

# Load model
model = CRISPRUniPredict.load_pretrained('models/checkpoints/best.pt')
encoder = SequenceEncoder(device='cuda')

# Single prediction
sgrna = "GCTAGCTAGCTAGCTAGCTAGCT"
target = "ATGCATGCATGCATGCATGCATG"

result = model.predict_comprehensive(sgrna, target)
print(f"On-target: {result['on_target_score']:.4f}")
print(f"Off-target: {result['off_target_risk']:.4f}")
print(f"Recommendation: {result['recommendation']}")

# Batch prediction
sgrnas = ["GCTAGCTAGCTAGCTAGCTAGCT", "ATGCATGCATGCATGCATGCATG"]
targets = ["ATGCATGCATGCATGCATGCATG", "GCTAGCTAGCTAGCTAGCTAGCT"]

results = model.batch_predict(sgrnas, targets)
for r in results:
    print(f"{r['sgrna']}: {r['recommendation']}")
```

### REST API Example

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict/comprehensive" \
  -H "Content-Type: application/json" \
  -d '{
    "sgrna": "GCTAGCTAGCTAGCTAGCTAGCT",
    "target": "ATGCATGCATGCATGCATGCATG"
  }' | jq

# Batch prediction
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sgrnas": ["GCTAGCTAGCTAGCTAGCTAGCT", "ATGCATGCATGCATGCATGCATG"],
    "targets": ["ATGCATGCATGCATGCATGCATG", "GCTAGCTAGCTAGCTAGCTAGCT"]
  }' | jq
```

### Python Requests Example

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Single prediction
response = requests.post(
    f"{BASE_URL}/predict/comprehensive",
    json={
        "sgrna": "GCTAGCTAGCTAGCTAGCTAGCT",
        "target": "ATGCATGCATGCATGCATGCATG"
    }
)

result = response.json()
print(f"Comprehensive Score: {result['comprehensive_score']:.4f}")
print(f"Recommendation: {result['recommendation']}")

# Batch prediction
response = requests.post(
    f"{BASE_URL}/batch_predict",
    json={
        "sgrnas": ["GCTAGCTAGCTAGCTAGCTAGCT", "ATGCATGCATGCATGCATGCATG"],
        "targets": ["ATGCATGCATGCATGCATGCATG", "GCTAGCTAGCTAGCTAGCTAGCT"]
    }
)

results = response.json()
print(f"Processed: {results['total_processed']} sequences")
print(f"Time: {results['processing_time_ms']:.2f}ms")
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-22 | Initial release |

---

**For more information, see the [main documentation](../README.md) or [API Guide](../API_GUIDE.md).**
