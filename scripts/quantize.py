"""
BERT Indonesian Sentiment Analysis — Quantize

Exports the HuggingFace model to ONNX and applies dynamic INT8 quantization.

Usage:
    pip install -r requirements-quantize.txt
    python scripts/quantize.py
"""

import os
import shutil

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoConfig, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_ID = "ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa"
ONNX_DIR = "./model_onnx"
QUANTIZED_DIR = "./models"

# Step 1: Export to ONNX
print("Step 1: Exporting model to ONNX...")
model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID, export=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
config = AutoConfig.from_pretrained(MODEL_ID)
model.save_pretrained(ONNX_DIR)
tokenizer.save_pretrained(ONNX_DIR)
config.save_pretrained(ONNX_DIR)
print(f"  -> {ONNX_DIR}")

# Step 2: Quantize (dynamic INT8, CPU)
print("Step 2: Quantizing to INT8...")
quantizer = ORTQuantizer.from_pretrained(ONNX_DIR)
quantization_config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
quantizer.quantize(save_dir=QUANTIZED_DIR, quantization_config=quantization_config)

for fname in ["tokenizer_config.json", "vocab.txt", "special_tokens_map.json", "tokenizer.json", "config.json"]:
    src = os.path.join(ONNX_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(QUANTIZED_DIR, fname))

print(f"  -> {QUANTIZED_DIR}")
print("Done!")
