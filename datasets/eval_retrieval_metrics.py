#!/usr/bin/env python3
"""Evaluate hybrid retrieval against generated QA pairs."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.retrieval.hybrid_retriever import HybridRetriever


QA_PATH = Path("data/metadata/qas_enron.parquet")
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ragsvc:ragsvc@localhost:5432/ragsentinel",
)


def load_qas(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    required_columns = {"qa_id", "tenant_id", "doc_id", "question", "answer", "qtype"}
    missing_columns = required_columns - set(table.column_names)
    if missing_columns:
        raise ValueError(f"QA Parquet is missing columns: {sorted(missing_columns)}")

    qas = [
        {
            "qa_id": qa_id,
            "tenant_id": tenant_id,
            "doc_id": doc_id,
            "question": question,
            "answer": answer,
            "qtype": qtype,
        }
        for qa_id, tenant_id, doc_id, question, answer, qtype in zip(
            table.column("qa_id").to_pylist(),
            table.column("tenant_id").to_pylist(),
            table.column("doc_id").to_pylist(),
            table.column("question").to_pylist(),
            table.column("answer").to_pylist(),
            table.column("qtype").to_pylist(),
        )
    ]

    if limit:
        qas = qas[:limit]
    return qas


def compute_metrics(
    retriever: HybridRetriever,
    qas: list[dict[str, Any]],
    ks: list[int] | None = None,
) -> dict[str, Any]:
    if ks is None:
        ks = [1, 3, 5, 10, 20]
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    if len(set(ks)) != len(ks):
        raise ValueError("ks must not contain duplicates")

    hits_at_k = {k: 0 for k in ks}
    rr_at_k = {k: 0.0 for k in ks}
    timing_sums: defaultdict[str, float] = defaultdict(float)
    total = 0
    skipped = 0
    max_k = max(ks)

    for i, qa in enumerate(qas):
        result = retriever.hybrid_search(
            query=qa["question"],
            tenant_id=qa["tenant_id"],
            top_k_fts=50,
            top_k_dense=50,
            top_n=max_k,
        )
        total += 1

        for key, value in result.timings.items():
            timing_sums[key] += value

        rank = next(
            (
                position
                for position, chunk in enumerate(result.chunks, start=1)
                if chunk.doc_id == qa["doc_id"]
            ),
            None,
        )
        for k in ks:
            if rank is not None and rank <= k:
                hits_at_k[k] += 1
                rr_at_k[k] += 1.0 / rank

        if (i + 1) % 10 == 0:
            print(f"  evaluated {i + 1}/{len(qas)} QAs...")

    hit_rate = {k: hits_at_k[k] / total if total else 0.0 for k in ks}
    mrr = {k: rr_at_k[k] / total if total else 0.0 for k in ks}
    avg_timings = {
        key: round(value / total, 1) if total else 0.0
        for key, value in timing_sums.items()
    }

    return {
        "total_evaluated": total,
        "skipped": skipped,
        "hit_rate_at_k": hit_rate,
        "mrr_at_k": mrr,
        "avg_timings_ms": avg_timings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on QA pairs")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of QAs to evaluate; use 0 for all (default: 50)",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="1,3,5,10,20",
        help="Comma-separated positive k values",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be zero or greater")
    try:
        ks = [int(value.strip()) for value in args.ks.split(",")]
    except ValueError:
        parser.error("--ks must be a comma-separated list of positive integers")
    if not ks or any(k < 1 for k in ks) or len(set(ks)) != len(ks):
        parser.error("--ks must contain unique positive integers")

    if not QA_PATH.exists():
        print(f"ERROR: {QA_PATH} not found. Run generate_enron_qas.py first.")
        return 2

    qas = load_qas(QA_PATH, limit=args.limit)
    print(f"Loaded {len(qas)} QA pairs from {QA_PATH}")
    print(f"Tenants in QAs: {sorted({qa['tenant_id'] for qa in qas})}")

    retriever = HybridRetriever(DB_URL)
    print("Running evaluation...")
    try:
        metrics = compute_metrics(retriever, qas, ks=ks)
    finally:
        retriever.close()

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    print(
        f"\nEvaluated: {metrics['total_evaluated']}  |  "
        f"Skipped: {metrics['skipped']}"
    )
    print(f"\n{'k':>5}  {'Hit Rate@k':>12}  {'MRR@k':>10}")
    print("-" * 35)
    for k in ks:
        hit_rate = metrics["hit_rate_at_k"][k]
        mrr = metrics["mrr_at_k"][k]
        print(f"{k:>5}  {hit_rate:>11.1%}  {mrr:>10.4f}")

    print(f"\nHeadline: Recall@5 = {metrics['hit_rate_at_k'].get(5, 0):.1%}")
    print("\nAverage timings per query:")
    for stage, milliseconds in metrics["avg_timings_ms"].items():
        print(f"  {stage:>12}: {milliseconds:>8.1f} ms")
    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
