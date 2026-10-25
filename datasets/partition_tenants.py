#!/usr/bin/env python3
"""
Phase 2: Deterministic tenant split (hash-based).
Creates data/metadata/doc_tenants.parquet mapping each doc to A/B/C.
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
import yaml


def load_cfg(path: str = "configs/corpus.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    seeds = y.get("seeds") or {}
    return {
        "tenants": int(y.get("tenants", 3)),
        "seed": int(seeds.get("partition", 42)),
        "processed_dir": y.get("processed_dir", "data/processed"),
        "metadata_dir": y.get("metadata_dir", "data/metadata"),
    }


def tenant_labels(n: int):
    return [chr(ord("A") + i) for i in range(n)]


def get_tenant(emp_id: str, n: int, seed: int) -> str:
    key = f"{seed}|{emp_id or 'unknown'}".encode("utf-8", "ignore")
    h = hashlib.sha1(key).hexdigest()
    idx = int(h, 16) % n
    return tenant_labels(n)[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", default=None)
    ap.add_argument("--in-parquet", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_cfg()
    proc = Path(cfg["processed_dir"])
    meta = Path(cfg["metadata_dir"])
    meta.mkdir(parents=True, exist_ok=True)

    in_parq = Path(args.in_parquet) if args.in_parquet else proc / "enron_emails.parquet"
    in_jsonl = Path(args.in_jsonl) if args.in_jsonl else proc / "enron_emails.jsonl"
    outp = Path(args.out) if args.out else meta / "doc_tenants.parquet"

    if in_parq.exists():
        print(f"Loading Parquet: {in_parq}")
        df = pd.read_parquet(in_parq, columns=["doc_id", "employee_id"])
    elif in_jsonl.exists():
        print(f"Loading JSONL: {in_jsonl}")
        df = pd.read_json(in_jsonl, lines=True)[["doc_id", "employee_id"]]
    else:
        raise SystemExit(
            f"ERROR: No input found. Tried:\n  - {in_parq}\n  - {in_jsonl}"
        )

    n = cfg["tenants"]
    seed = cfg["seed"]

    df["employee_id"] = df["employee_id"].astype(str)
    df["tenant_id"] = [get_tenant(emp, n, seed) for emp in df["employee_id"]]
    df["method"] = "hash_mod"
    df["seed"] = seed

    cols = ["doc_id", "employee_id", "tenant_id", "method", "seed"]
    df[cols].to_parquet(outp, index=False)

    counts = df["tenant_id"].value_counts().sort_index()
    total = len(df)

    print(f"\nWrote: {outp}  ({total:,} rows)")
    print(f"Tenants: {n}  |  Seed: {seed}")
    for t, c in counts.items():
        print(f"  {t}: {c:>8,}  ({c / total * 100:5.2f}%)")


if __name__ == "__main__":
    main()
