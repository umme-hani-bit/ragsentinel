from __future__ import annotations

from pathlib import Path

import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __package__:
    from .train import TextClassificationDataset, prepare_data_splits
else:
    from train import TextClassificationDataset, prepare_data_splits

MODEL_DIR = Path("models/injection_classifier/saved_model")


def evaluate() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_DIR}. Run train.py first.")

    texts_dict, labels_dict = prepare_data_splits()  # FIX: one call, not two
    test_texts, test_labels = texts_dict["test"], labels_dict["test"]
    print(f"Evaluating on held-out test split ({len(test_texts)} samples)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loader = DataLoader(
        TextClassificationDataset(test_texts, test_labels, tokenizer),
        batch_size=16,
        shuffle=False,
    )

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            all_probs.extend(torch.softmax(outputs.logits, dim=1)[:, 1].cpu().tolist())
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().tolist())
            all_labels.extend(batch["labels"].tolist())

    print("\n--- DistilBERT Injection Classifier (Test Split) ---")
    print(f"Precision: {precision_score(all_labels, all_preds):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds):.4f}")
    print(f"F1 Score:  {f1_score(all_labels, all_preds):.4f}")
    print(f"ROC AUC:   {roc_auc_score(all_labels, all_probs):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Benign", "Injection"]))


if __name__ == "__main__":
    evaluate()