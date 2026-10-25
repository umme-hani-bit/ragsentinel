#!/usr/bin/env python3
"""Find the most balanced tenant partition seed."""
import hashlib
from collections import Counter

import pandas as pd
import yaml


def get_tenant(emp_id, n, seed):
    key = f"{seed}|{emp_id or 'unknown'}".encode("utf-8", "ignore")
    h = hashlib.sha1(key).hexdigest()
    idx = int(h, 16) % n
    return chr(ord("A") + idx)


def main():
    with open("configs/corpus.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    tenants = int(cfg.get("tenants", 3))

    print("Loading employee email counts...")
    df = pd.read_json("data/processed/enron_emails.jsonl", lines=True)[["employee_id"]]
    employee_counts = df["employee_id"].astype(str).value_counts().to_dict()

    best_seed = None
    best_diff = float("inf")

    print("Searching seeds 1-500...")
    for seed in range(1, 501):
        counts = Counter()
        for emp, count in employee_counts.items():
            counts[get_tenant(emp, tenants, seed)] += count

        diff = max(counts.values()) - min(counts.values())
        total = sum(counts.values())

        if diff < best_diff:
            best_diff = diff
            best_seed = seed
            print(
                f"  seed {seed:>3}: "
                f"A={counts['A']:>6,}  B={counts['B']:>6,}  C={counts['C']:>6,}  "
                f"diff={diff:>5,} ({diff/total*100:4.2f}%)"
            )

        if diff == 0:
            break

    print(f"\nBest seed: {best_seed}  (diff = {best_diff:,} emails)")


if __name__ == "__main__":
    main()
