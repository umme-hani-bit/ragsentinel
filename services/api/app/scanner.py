from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("models/injection_classifier/saved_model")

_TOKENIZER = None
_MODEL = None


def _get_scanner_model():
    global _TOKENIZER, _MODEL
    if _MODEL is None:
        if not MODEL_DIR.exists():
            raise RuntimeError(f"Scanner model missing at {MODEL_DIR}. Run train.py first.")
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _MODEL.eval()
    return _TOKENIZER, _MODEL


def scan_chunks(
    chunks: List[Dict[str, Any]],
    threshold: float = 0.65,
    enabled: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Scores retrieved chunks for prompt injection.
    Returns: (processed_chunks, taint_flags)
    """
    if not enabled or not chunks:
        return chunks, []

    tokenizer, model = _get_scanner_model()
    inputs = tokenizer(
        [c["text"] for c in chunks],
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[:, 1].tolist()

    processed, flags = [], []
    for chunk, prob in zip(chunks, probs):
        c = dict(chunk)
        c["injection_prob"] = round(prob, 4)
        c["is_tainted"] = prob >= threshold
        if c["is_tainted"]:
            flags.append(f"TAINT_INJECTION_DETECTED: chunk_{chunk['chunk_id']} (prob={prob:.4f})")
        processed.append(c)

    return processed, flags