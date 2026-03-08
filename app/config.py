import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
ONNX_MODEL_PATH = MODEL_DIR / "model_quantized.onnx"
TOKENIZER_PATH = MODEL_DIR

# Label mapping (from model config.json)
ID2LABEL = {0: "positive", 1: "neutral", 2: "negative"}

# Inference config
MAX_LENGTH = 128

# Server (Railway injects PORT at runtime; defaults to 8000 locally)
PORT = int(os.environ.get("PORT", 8000))
