import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from app.config import ONNX_MODEL_PATH, TOKENIZER_PATH, ID2LABEL, MAX_LENGTH


def _softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class SentimentModel:
    def __init__(self):
        self.tokenizer = None
        self.session = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(ONNX_MODEL_PATH),
            sess_options,
            providers=["CPUExecutionProvider"],
        )

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": inputs["token_type_ids"].astype(np.int64),
        }
        logits = self.session.run(None, ort_inputs)[0]
        probs = _softmax(logits[0])
        scores = {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))}
        top_label = max(scores, key=scores.get)
        return {"sentiment": top_label, "confidence": scores[top_label], "scores": scores}

    def predict_batch(self, texts: list[str]) -> list[dict]:
        inputs = self.tokenizer(
            texts,
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": inputs["token_type_ids"].astype(np.int64),
        }
        logits = self.session.run(None, ort_inputs)[0]
        results = []
        for i in range(len(texts)):
            probs = _softmax(logits[i])
            scores = {ID2LABEL[j]: float(probs[j]) for j in range(len(probs))}
            top_label = max(scores, key=scores.get)
            results.append({"sentiment": top_label, "confidence": scores[top_label], "scores": scores})
        return results
