"""
BERT Indonesian Sentiment Analysis — Quantize & Evaluate

Exports the HuggingFace model to ONNX, applies dynamic INT8 quantization,
evaluates both versions, and generates comparison charts.

Usage:
    pip install -r requirements-quantize.txt
    python scripts/quantize.py
"""

import os
import shutil
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = "ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa"
ONNX_DIR = "./model_onnx"
QUANTIZED_DIR = "./models"  # Output directly to models/ for the API
DATASET_NAME = "indonlp/indonlu"
SUBSET = "smsa"
SPLIT = "test"
BATCH_SIZE = 32
MAX_LENGTH = 128


# ── Helpers ─────────────────────────────────────────────────────────────────
def get_size_mb(model_dir, filename):
    path = os.path.join(model_dir, filename)
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0


def evaluate(model, tokenizer, dataset):
    all_preds, all_labels = [], []
    start = time.time()
    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch["text"],
            return_tensors="np",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        preds = np.argmax(model(**inputs).logits, axis=-1).tolist()
        all_preds.extend(preds)
        all_labels.extend(batch["label"])
    elapsed = time.time() - start
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
        "precision_macro": precision_score(all_labels, all_preds, average="macro"),
        "recall_macro": recall_score(all_labels, all_preds, average="macro"),
        "elapsed": elapsed,
        "throughput": len(dataset) / elapsed,
        "preds": all_preds,
        "labels": all_labels,
    }


# ── Step 1: Export to ONNX ──────────────────────────────────────────────────
print("Step 1: Exporting model to ONNX...")
model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID, export=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model.save_pretrained(ONNX_DIR)
tokenizer.save_pretrained(ONNX_DIR)
print(f"  -> {ONNX_DIR}")


# ── Step 2: Quantize (dynamic INT8, CPU) ────────────────────────────────────
print("Step 2: Quantizing to INT8...")
quantizer = ORTQuantizer.from_pretrained(ONNX_DIR)
quantization_config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
quantizer.quantize(save_dir=QUANTIZED_DIR, quantization_config=quantization_config)

# Copy tokenizer & config files to quantized dir
for fname in [
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "config.json",
]:
    src = os.path.join(ONNX_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(QUANTIZED_DIR, fname))
print(f"  -> {QUANTIZED_DIR}")


# ── Step 3: Evaluate both models ────────────────────────────────────────────
dataset = load_dataset(DATASET_NAME, SUBSET, split=SPLIT)
print(f"\nStep 3: Evaluating on {len(dataset)} test samples...")

print("  Evaluating original ONNX...")
orig_model = ORTModelForSequenceClassification.from_pretrained(ONNX_DIR)
orig_tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)
orig_metrics = evaluate(orig_model, orig_tokenizer, dataset)
id2label = orig_model.config.id2label
del orig_model

print("  Evaluating quantized ONNX...")
quant_model = ORTModelForSequenceClassification.from_pretrained(QUANTIZED_DIR)
quant_tokenizer = AutoTokenizer.from_pretrained(QUANTIZED_DIR)
quant_metrics = evaluate(quant_model, quant_tokenizer, dataset)
del quant_model

orig_size = get_size_mb(ONNX_DIR, "model.onnx")
quant_size = get_size_mb(QUANTIZED_DIR, "model_quantized.onnx")
print(f"\n  Original:  {orig_size:.1f} MB")
print(f"  Quantized: {quant_size:.1f} MB")
print(f"  Reduction: {(1 - quant_size / orig_size) * 100:.1f}%")


# ── Step 4: Classification reports ──────────────────────────────────────────
target_names = [id2label.get(i, f"LABEL_{i}") for i in sorted(id2label.keys())]

print("\n--- Original ONNX ---")
print(classification_report(orig_metrics["labels"], orig_metrics["preds"], target_names=target_names))

print("--- Quantized ONNX (INT8) ---")
print(classification_report(quant_metrics["labels"], quant_metrics["preds"], target_names=target_names))


# ── Step 5: Charts (academic style) ─────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.edgecolor": "#cccccc",
    }
)

C_ORIG = "#2166ac"
C_QUANT = "#b2182b"


def add_value_label(ax, bar, text, offset=0, fontsize=10):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + offset,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )


# Chart 1: Model Size
fig, ax = plt.subplots(figsize=(5, 4))
labels = ["Original\n(FP32)", "Quantized\n(INT8)"]
sizes = [orig_size, quant_size]
bars = ax.bar(labels, sizes, color=[C_ORIG, C_QUANT], width=0.45, edgecolor="black", linewidth=0.6)
for bar, s in zip(bars, sizes):
    add_value_label(ax, bar, f"{s:.1f} MB", offset=orig_size * 0.02)
reduction = (1 - quant_size / orig_size) * 100
ax.annotate(
    "",
    xy=(1, quant_size + orig_size * 0.01),
    xytext=(0, orig_size + orig_size * 0.01),
    arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.2),
)
ax.text(
    0.5,
    orig_size + orig_size * 0.05,
    f"\u2193 {reduction:.1f}%",
    ha="center",
    va="bottom",
    fontsize=11,
    fontweight="bold",
    color="#333333",
)
ax.set_ylabel("Model Size (MB)")
ax.set_title("(a) Model Size Comparison")
ax.set_ylim(0, orig_size * 1.3)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
plt.tight_layout()
plt.savefig("chart_model_size.png", dpi=300, bbox_inches="tight")
plt.close()

# Chart 2: Performance Metrics
fig, ax = plt.subplots(figsize=(7, 4))
metric_labels = ["Accuracy", "F1 (macro)", "F1 (weighted)", "Precision", "Recall"]
metric_keys = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
orig_vals = [orig_metrics[k] for k in metric_keys]
quant_vals = [quant_metrics[k] for k in metric_keys]
x = np.arange(len(metric_labels))
w = 0.3
bars1 = ax.bar(x - w / 2, orig_vals, w, label="Original (FP32)", color=C_ORIG, edgecolor="black", linewidth=0.6)
bars2 = ax.bar(x + w / 2, quant_vals, w, label="Quantized (INT8)", color=C_QUANT, edgecolor="black", linewidth=0.6)
for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Score")
ax.set_title("(b) Classification Performance Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
all_vals = orig_vals + quant_vals
y_min = min(all_vals)
y_floor = max(0, round(y_min - 0.03, 2))
ax.set_ylim(y_floor, 1.005)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.legend(loc="lower left")
ax.spines[["top", "right"]].set_visible(False)
for y in np.arange(y_floor, 1.01, 0.02):
    ax.axhline(y=y, color="#dddddd", linewidth=0.5, zorder=0)
plt.tight_layout()
plt.savefig("chart_performance.png", dpi=300, bbox_inches="tight")
plt.close()

# Chart 3: Inference Speed
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
bar_labels = ["Original\n(FP32)", "Quantized\n(INT8)"]

throughputs = [orig_metrics["throughput"], quant_metrics["throughput"]]
b1 = ax1.bar(bar_labels, throughputs, color=[C_ORIG, C_QUANT], width=0.45, edgecolor="black", linewidth=0.6)
for bar, val in zip(b1, throughputs):
    add_value_label(ax1, bar, f"{val:.1f}", offset=max(throughputs) * 0.02)
speedup = throughputs[1] / throughputs[0] if throughputs[0] > 0 else 0
ax1.set_ylabel("Samples / Second")
ax1.set_title("(c) Inference Throughput")
ax1.set_ylim(0, max(throughputs) * 1.25)
ax1.spines[["top", "right"]].set_visible(False)
ax1.yaxis.set_major_locator(mticker.MaxNLocator(5))
ax1.text(
    0.97,
    0.95,
    f"{speedup:.2f}x speedup",
    transform=ax1.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    fontstyle="italic",
    color="#333333",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#cccccc"),
)

times = [orig_metrics["elapsed"], quant_metrics["elapsed"]]
b2 = ax2.bar(bar_labels, times, color=[C_ORIG, C_QUANT], width=0.45, edgecolor="black", linewidth=0.6)
for bar, val in zip(b2, times):
    add_value_label(ax2, bar, f"{val:.1f}s", offset=max(times) * 0.02)
ax2.set_ylabel("Time (seconds)")
ax2.set_title("(d) Total Inference Time")
ax2.set_ylim(0, max(times) * 1.25)
ax2.spines[["top", "right"]].set_visible(False)
ax2.yaxis.set_major_locator(mticker.MaxNLocator(5))
time_saved = times[0] - times[1]
ax2.text(
    0.97,
    0.95,
    f"\u2193 {time_saved:.1f}s",
    transform=ax2.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    fontstyle="italic",
    color="#333333",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#cccccc"),
)

plt.tight_layout(w_pad=3)
plt.savefig("chart_speed.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nDone! Charts saved: chart_model_size.png, chart_performance.png, chart_speed.png")
print(f"Quantized model saved to: {QUANTIZED_DIR}/")
