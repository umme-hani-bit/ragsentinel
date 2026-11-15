from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"
DEFAULT_THRESHOLD = 0.5
MAX_PREMISES = 3        # cap cited chunks so sentence x premise pairs stay small on CPU
MIN_SENT_WORDS = 4      # skip trivial fragments like "OK."

_TOKENIZER = None
_MODEL = None
_MODEL_NAME: Optional[str] = None
_ENTAIL_ID: Optional[int] = None


def _get_model(model_name: str = DEFAULT_MODEL):
    """Lazy-load the NLI model once and reuse it across requests."""
    global _TOKENIZER, _MODEL, _MODEL_NAME, _ENTAIL_ID
    if _MODEL is None or _MODEL_NAME != model_name:
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
        _MODEL.eval()
        _MODEL_NAME = model_name
        # Find the entailment label index from the model config instead of hardcoding.
        id2label = {int(k): v.upper() for k, v in _MODEL.config.id2label.items()}
        _ENTAIL_ID = next((i for i, lbl in id2label.items() if "ENTAIL" in lbl), 2)
    return _TOKENIZER, _MODEL, _ENTAIL_ID


def split_answer_sentences(answer: str) -> List[str]:
    """Split on sentence boundaries; keep only sentences with enough words to score."""
    raw = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [s.strip() for s in raw if len(s.split()) >= MIN_SENT_WORDS]


def _entailment_scores(sentences: List[str], premises: List[str], model_name: str) -> List[List[float]]:
    """
    Returns a matrix: for each sentence, the entailment probability against each premise.
    We take the max over premises later (a sentence is supported if ANY cited chunk entails it).
    """
    tokenizer, model, entail_id = _get_model(model_name)

    pairs = [(prem, sent) for sent in sentences for prem in premises]
    if not pairs:
        return [[] for _ in sentences]

    scores = torch.zeros(len(pairs))
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            enc = tokenizer(
                [p for p, _ in batch],
                [s for _, s in batch],
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)[:, entail_id]
            scores[i : i + len(batch)] = probs.cpu()

    # reshape flat pairs back into [sentence][premise]
    matrix: List[List[float]] = []
    idx = 0
    for _ in sentences:
        row = [float(scores[idx + j]) for j in range(len(premises))]
        matrix.append(row)
        idx += len(premises)
    return matrix


def verify_faithfulness(
    answer: str,
    cited_chunks: List[Dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Score each answer sentence against the cited chunks.
    A sentence counts as SUPPORTED if max entailment over cited chunks >= threshold.
    """
    sentences = split_answer_sentences(answer)
    premises = [c["text"][:1500] for c in cited_chunks[:MAX_PREMISES] if c.get("text")]

    if not sentences:
        return {"sentences_total": 0, "sentences_supported": 0,
                "unsupported_claim_rate": None, "verdicts": [], "reason": "no answer sentences to score"}
    if not premises:
        return {"sentences_total": len(sentences), "sentences_supported": 0,
                "unsupported_claim_rate": None, "verdicts": [], "reason": "no cited chunks to check against"}

    matrix = _entailment_scores(sentences, premises, model_name)

    verdicts = []
    supported_count = 0
    for sent, row in zip(sentences, matrix):
        best = max(row) if row else 0.0
        is_supported = best >= threshold
        supported_count += 1 if is_supported else 0
        verdicts.append({"sentence": sent, "max_entailment": round(best, 4), "supported": is_supported})

    total = len(sentences)
    return {
        "sentences_total": total,
        "sentences_supported": supported_count,
        "unsupported_claim_rate": round((total - supported_count) / total, 4),
        "verdicts": verdicts,
        "reason": "ok",
    }