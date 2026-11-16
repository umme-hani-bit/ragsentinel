from __future__ import annotations

from typing import Dict, Set

import pyarrow.parquet as pq

if __package__:
    from .train import INJ_SPLITS, prepare_data_splits
else:
    from train import INJ_SPLITS, prepare_data_splits


def check_overlap() -> None:
    split_rows = pq.read_table(
        INJ_SPLITS,
        columns=["doc_id", "split"],
    ).to_pylist()
    doc_ids: Dict[str, Set[str]] = {"train": set(), "test": set()}
    for row in split_rows:
        split = row["split"]
        if split in doc_ids:
            doc_ids[split].add(row["doc_id"])

    doc_overlap = doc_ids["train"] & doc_ids["test"]
    print(f"Train/test injection document ID overlap: {len(doc_overlap)}")

    texts, labels = prepare_data_splits()
    text_sets = {
        split: {
            label: {
                text
                for text, sample_label in zip(texts[split], labels[split])
                if sample_label == label
            }
            for label in (0, 1)
        }
        for split in ("train", "test")
    }

    total_overlap = set(texts["train"]) & set(texts["test"])
    print(f"Exact text overlap between train and test: {len(total_overlap)}")
    for label, name in ((0, "benign"), (1, "injection")):
        overlap = text_sets["train"][label] & text_sets["test"][label]
        print(f"Exact {name} text overlap: {len(overlap)}")

    if doc_overlap or total_overlap:
        print("LEAKAGE DETECTED: train/test data is not independent.")
    else:
        print("No train/test document-ID or exact-text overlap detected.")


if __name__ == "__main__":
    check_overlap()
