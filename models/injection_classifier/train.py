from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow.parquet as pq
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

INJ_SPANS = Path("data/metadata/injection_spans.parquet")
INJ_SPLITS = Path("data/metadata/injection_doc_splits.parquet")
AUG_INJ = Path("data/processed/enron_emails_aug_inj.parquet")
MODEL_OUT_DIR = Path("models/injection_classifier/saved_model")
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5
MAX_LEN = 256
SEED = 42


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def prepare_data_splits() -> Tuple[Dict[str, List[str]], Dict[str, List[int]]]:
    splits_tbl = pq.read_table(INJ_SPLITS)
    doc_splits = dict(zip(splits_tbl["doc_id"].to_pylist(), splits_tbl["split"].to_pylist()))

    spans_tbl = pq.read_table(INJ_SPANS)
    aug_tbl = pq.read_table(AUG_INJ, columns=["doc_id", "augmented_body", "clean_body"])
    aug_docs = dict(zip(aug_tbl["doc_id"].to_pylist(), aug_tbl["augmented_body"].to_pylist()))
    clean_docs = dict(zip(aug_tbl["doc_id"].to_pylist(), aug_tbl["clean_body"].to_pylist()))

    texts_by_split: Dict[str, List[str]] = {"train": [], "dev": [], "test": []}
    labels_by_split: Dict[str, List[int]] = {"train": [], "dev": [], "test": []}

    # Positives: injected snippets (label 1)
    for doc_id, start, end in zip(
        spans_tbl["doc_id"].to_pylist(),
        spans_tbl["block_start"].to_pylist(),
        spans_tbl["block_end"].to_pylist(),
    ):
        split = doc_splits.get(doc_id, "train")
        full_text = aug_docs.get(doc_id, "")
        if full_text and start < len(full_text):
            snippet = full_text[start:end].strip()
            if snippet:
                texts_by_split[split].append(snippet)
                labels_by_split[split].append(1)

    # Negatives: benign snippets (label 0), 2:1 ratio
    rng = random.Random(SEED)
    all_clean = list(clean_docs.items())
    rng.shuffle(all_clean)

    for split in ["train", "dev", "test"]:
        needed = len(labels_by_split[split]) * 2
        added = 0
        for doc_id, clean_text in all_clean:
            if added >= needed:
                break
            if doc_id not in doc_splits and clean_text.strip():
                snippet = clean_text[:MAX_LEN].strip()
                if snippet:
                    texts_by_split[split].append(snippet)
                    labels_by_split[split].append(0)
                    added += 1

    return texts_by_split, labels_by_split


def train() -> None:
    set_seed(SEED)
    print("Preparing training dataset...", flush=True)
    texts, labels = prepare_data_splits()
    split_sizes = ", ".join(
        f"{split}: {len(texts[split])}" for split in ("train", "dev", "test")
    )
    print(f"Split sizes: {split_sizes}", flush=True)
    if not texts["train"]:
        raise ValueError("Training split is empty; cannot train the classifier.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}", flush=True)
    model.to(device)

    train_ds = TextClassificationDataset(texts["train"], labels["train"], tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    model.train()
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS} starting...", flush=True)
        total_loss = 0.0
        samples_seen = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            print(
                f"Epoch {epoch}/{EPOCHS} batch {batch_index}/{len(train_loader)}",
                flush=True,
            )
            optimizer.zero_grad()
            outputs = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_size = batch["labels"].size(0)
            total_loss += loss.item() * batch_size
            samples_seen += batch_size

        average_loss = total_loss / samples_seen
        print(
            f"Epoch {epoch}/{EPOCHS} complete - training loss: {average_loss:.4f}",
            flush=True,
        )

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_OUT_DIR)
    tokenizer.save_pretrained(MODEL_OUT_DIR)
    print(f"Training finished successfully. Model saved to {MODEL_OUT_DIR}", flush=True)


if __name__ == "__main__":
    train()