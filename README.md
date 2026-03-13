# Indonesian Sentiment Analysis API

Quantized BERT model serving Indonesian text sentiment analysis via FastAPI, deployed on Railway.

**Model**: [`ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa`](https://huggingface.co/ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa) — dynamic INT8 quantized ONNX.

## Quick Start

### 1. Generate the quantized model

```bash
pip install -r requirements-quantize.txt
python scripts/quantize.py
```

This exports the model to ONNX, quantizes it to INT8, and saves everything to `models/`.

### 2. Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Run with Docker

```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

## API Endpoints

### `GET /health`

Health check.

### `POST /predict`

```json
{
  "text": "Film ini sangat bagus dan menarik!"
}
```

**Python requests example:**

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    headers={"x-api-key": "demo-key-gl-sentiment-2024"},
    json={"text": "Film ini sangat bagus dan menarik!"},
)
print(response.json())
```

Response:

```json
{
  "text": "Film ini sangat bagus dan menarik!",
  "sentiment": "positive",
  "confidence": 0.987,
  "scores": [
    {"label": "positive", "score": 0.987},
    {"label": "neutral", "score": 0.008},
    {"label": "negative", "score": 0.005}
  ]
}
```

### `POST /predict/batch`

```json
{
  "texts": ["Pelayanannya sangat buruk", "Biasa saja"]
}
```

**Python requests example:**

```python
import requests

response = requests.post(
    "http://localhost:8000/predict/batch",
    headers={"x-api-key": "demo-key-gl-sentiment-2024"},
    json={"texts": ["Pelayanannya sangat buruk", "Biasa saja"]},
)
print(response.json())
```

## Deploy on Railway

1. Push this repo to GitHub (with model files via Git LFS)
2. Connect the repo in [Railway](https://railway.app)
3. Railway auto-detects the Dockerfile and deploys

Railway sets the `PORT` environment variable automatically.

## Project Structure

```
├── app/
│   ├── main.py          # FastAPI application
│   ├── model.py         # ONNX Runtime inference
│   ├── schemas.py       # Request/response models
│   └── config.py        # Configuration
├── scripts/
│   └── quantize.py      # Quantization & evaluation script
├── models/              # Quantized ONNX model (Git LFS)
├── Dockerfile
├── requirements.txt     # Serving dependencies
└── requirements-quantize.txt  # Quantization dependencies
```
