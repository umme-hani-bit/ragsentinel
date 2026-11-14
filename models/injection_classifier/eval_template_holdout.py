#!/usr/bin/env python3
"""Evaluate classifier generalization to injection payloads held out by template."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pyarrow.parquet as pq
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

INJ_SPANS = Path("data/metadata/injection_spans.parquet")
AUG_INJ = Path("data/processed/enron_emails_aug_inj.parquet")

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5
MAX_LEN = 256
SEED = 42
BENIGN_POOL_SIZE = 4096
RESULTS_PATH = Path(
    "models/injection_classifier/saved_model/template_holdout_results.json"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        token_cache: Dict[str, Dict[str, List[int]]],
    ):
        self.encodings: List[Dict[str, List[int]]] = []
        self.labels = labels
        for text in texts:
            encoding = token_cache.get(text)
            if encoding is None:
                tokenized = tokenizer(
                    text,
                    truncation=True,
                    max_length=MAX_LEN,
                    padding=False,
                )
                encoding = {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                }
                token_cache[text] = encoding
            self.encodings.append(encoding)

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            **self.encodings[idx],
            "labels": self.labels[idx],
        }


def collate_batch(batch, tokenizer) -> Dict[str, torch.Tensor]:
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    model_inputs = tokenizer.pad(
        [
            {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
            }
            for item in batch
        ],
        padding=True,
        return_tensors="pt",
    )
    return {**model_inputs, "labels": labels}


def load_payload_texts(
) -> Tuple[Dict[str, List[str]], Dict[str, int], Set[str]]:
    spans = pq.read_table(
        INJ_SPANS,
        columns=["doc_id", "block_start", "block_end", "payload_id", "tier"],
    ).to_pylist()
    injected_doc_ids = {row["doc_id"] for row in spans}
    if not injected_doc_ids:
        raise ValueError(f"No injection spans found in {INJ_SPANS}.")
    payload_ids_by_doc: Dict[str, Set[str]] = defaultdict(set)
    for row in spans:
        if row["payload_id"] is not None:
            payload_ids_by_doc[row["doc_id"]].add(row["payload_id"])
    mixed_payload_docs = [
        doc_id for doc_id, payload_ids in payload_ids_by_doc.items()
        if len(payload_ids) > 1
    ]
    if mixed_payload_docs:
        raise ValueError(
            "Template holdout requires each document to contain one payload ID; "
            f"{len(mixed_payload_docs)} documents contain multiple payload IDs."
        )

    augmented = pq.read_table(
        AUG_INJ,
        columns=["doc_id", "augmented_body"],
        filters=[("doc_id", "in", list(injected_doc_ids))],
    )
    augmented_by_doc = dict(
        zip(augmented["doc_id"].to_pylist(), augmented["augmented_body"].to_pylist())
    )

    payload_texts: Dict[str, List[str]] = defaultdict(list)
    payload_tiers: Dict[str, int] = {}
    for row in spans:
        payload_id = row["payload_id"]
        if payload_id is None:
            raise ValueError("Injection span is missing payload_id.")
        full_text = augmented_by_doc.get(row["doc_id"]) or ""
        start, end = row["block_start"], row["block_end"]
        if 0 <= start < end <= len(full_text):
            snippet = full_text[start:end].strip()
            if snippet:
                payload_texts[payload_id].append(snippet)
                tier = int(row["tier"])
                previous_tier = payload_tiers.setdefault(payload_id, tier)
                if previous_tier != tier:
                    raise ValueError(f"Payload {payload_id} has inconsistent tiers.")

    if not payload_texts:
        raise ValueError("No non-empty injected snippets could be loaded.")
    return dict(payload_texts), payload_tiers, injected_doc_ids


def load_disjoint_benign_pools(
    injected_doc_ids: Set[str],
    injection_texts: Set[str],
) -> Tuple[List[str], List[str]]:
    """Reservoir-sample benign documents, then split unique texts disjointly."""
    rng = random.Random(SEED)
    reservoir: List[Tuple[str, str]] = []
    eligible_count = 0

    parquet = pq.ParquetFile(AUG_INJ)
    for batch in parquet.iter_batches(
        columns=["doc_id", "clean_body"],
        batch_size=4096,
    ):
        for doc_id, clean_body in zip(
            batch.column("doc_id").to_pylist(),
            batch.column("clean_body").to_pylist(),
        ):
            if doc_id in injected_doc_ids or not clean_body:
                continue
            text = str(clean_body)[:MAX_LEN].strip()
            if not text or text in injection_texts:
                continue

            eligible_count += 1
            item = (doc_id, text)
            if len(reservoir) < BENIGN_POOL_SIZE:
                reservoir.append(item)
            else:
                replacement = rng.randrange(eligible_count)
                if replacement < BENIGN_POOL_SIZE:
                    reservoir[replacement] = item

    unique_texts = list(dict.fromkeys(text for _, text in reservoir))
    rng.shuffle(unique_texts)
    split_at = (len(unique_texts) * 4) // 5
    train_pool = unique_texts[:split_at]
    test_pool = unique_texts[split_at:]
    if not train_pool or not test_pool:
        raise ValueError("Not enough unique benign samples for disjoint evaluation.")

    overlap = set(train_pool) & set(test_pool)
    if overlap:
        raise RuntimeError("Benign train/test pools unexpectedly overlap.")
    return train_pool, test_pool


def train_and_evaluate(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    tokenizer,
    token_cache: Dict[str, Dict[str, List[int]]],
    seed: int,
) -> Tuple[float, float, float]:
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        TextDataset(train_texts, train_labels, tokenizer, token_cache),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer),
    )
    if not train_texts:
        raise ValueError("Training fold is empty.")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    model.train()
    for epoch in range(1, EPOCHS + 1):
        print(f"  training epoch {epoch}/{EPOCHS}", flush=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if batch_index % 10 == 0 or batch_index == len(train_loader):
                print(
                    f"    batch {batch_index}/{len(train_loader)}",
                    flush=True,
                )

    test_loader = DataLoader(
        TextDataset(test_texts, test_labels, tokenizer, token_cache),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, tokenizer),
    )
    model.eval()
    all_predictions: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            predictions = torch.argmax(outputs.logits, dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(batch["labels"].tolist())

    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    return precision, recall, f1


def save_results(results: List[Dict[str, object]]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps({"completed_folds": results}, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(RESULTS_PATH)


def main() -> int:
    print("Loading payload texts...", flush=True)
    payload_texts, payload_tiers, injected_doc_ids = load_payload_texts()
    all_injection_texts = {
        text for texts in payload_texts.values() for text in texts
    }
    benign_train_pool, benign_test_pool = load_disjoint_benign_pools(
        injected_doc_ids,
        all_injection_texts,
    )

    payload_ids = sorted(payload_texts)
    print(
        f"Found {len(payload_ids)} payload templates from "
        f"{len(injected_doc_ids)} documents.",
        flush=True,
    )
    print(
        f"Disjoint benign pools: train={len(benign_train_pool)}, "
        f"test={len(benign_test_pool)}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    token_cache: Dict[str, Dict[str, List[int]]] = {}

    results: List[Dict[str, object]] = []
    for fold, held_out_id in enumerate(payload_ids):
        held_out_texts = payload_texts[held_out_id]
        held_out_set = set(held_out_texts)
        other_payload_texts = [
            text
            for payload_id, texts in payload_texts.items()
            if payload_id != held_out_id
            for text in texts
            if text not in held_out_set
        ]
        if not other_payload_texts:
            raise ValueError(f"No positive training samples for {held_out_id}.")

        rng = random.Random(SEED + fold)
        train_benign_count = min(len(other_payload_texts) * 2, len(benign_train_pool))
        test_benign_count = min(len(held_out_texts) * 2, len(benign_test_pool))
        train_benign = rng.sample(benign_train_pool, train_benign_count)
        test_benign = rng.sample(benign_test_pool, test_benign_count)

        train_texts = other_payload_texts + train_benign
        train_labels = [1] * len(other_payload_texts) + [0] * len(train_benign)
        test_texts = held_out_texts + test_benign
        test_labels = [1] * len(held_out_texts) + [0] * len(test_benign)
        if set(train_texts) & set(test_texts):
            raise RuntimeError(f"Exact text overlap in fold for {held_out_id}.")

        print(
            f"Fold {fold + 1}/{len(payload_ids)}: hold out {held_out_id} "
            f"(tier {payload_tiers[held_out_id]}, train={len(train_texts)}, "
            f"test={len(test_texts)})",
            flush=True,
        )
        precision, recall, f1 = train_and_evaluate(
            train_texts,
            train_labels,
            test_texts,
            test_labels,
            tokenizer,
            token_cache,
            seed=SEED + fold,
        )
        results.append({
                "payload_id": held_out_id,
                "tier": payload_tiers[held_out_id],
                "test_positive_count": len(held_out_texts),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })
        save_results(results)
        print(
            f"  precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}",
            flush=True,
        )

    print("\n--- Cross-template generalization ---", flush=True)
    for metric in ("precision", "recall", "f1"):
        average = sum(result[metric] for result in results) / len(results)
        print(f"Macro {metric}: {average:.4f}", flush=True)

    for tier in (1, 2, 3):
        tier_results = [result for result in results if result["tier"] == tier]
        if tier_results:
            average_f1 = sum(float(result["f1"]) for result in tier_results) / len(tier_results)
            print(
                f"Tier {tier} macro F1: {average_f1:.4f} "
                f"({len(tier_results)} templates)",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
