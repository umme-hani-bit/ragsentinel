#!/usr/bin/env python3
"""Chunk augmented documents, generate BGE embeddings, and store them in PostgreSQL."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterable

import pandas as pd
import psycopg2
import pyarrow.parquet as pq
from fastembed import TextEmbedding
from tqdm import tqdm


AUG_INJ_PATH = Path("data/processed/enron_emails_aug_inj.parquet")
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ragsvc:ragsvc@localhost:5432/ragsentinel",
)
MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSIONS = 1024
PARQUET_BATCH_SIZE = 500
BALANCED_SAMPLE_SEED = 0
INGEST_COLUMNS = (
    "doc_id",
    "tenant_id",
    "employee_id",
    "date",
    "subject",
    "raw_body",
    "clean_body",
    "augmented_body",
    "sentences",
    "inserted_secrets_count",
    "inserted_injections_count",
)


def _balanced_quotas(tenant_counts: dict[str, int], limit: int) -> dict[str, int]:
    quotas = {tenant_id: 0 for tenant_id in tenant_counts}
    remaining = min(limit, sum(tenant_counts.values()))

    while remaining:
        active = sorted(
            tenant_id
            for tenant_id, count in tenant_counts.items()
            if quotas[tenant_id] < count
        )
        per_tenant = max(1, remaining // len(active))
        for tenant_id in active:
            allocation = min(
                tenant_counts[tenant_id] - quotas[tenant_id],
                per_tenant,
                remaining,
            )
            quotas[tenant_id] += allocation
            remaining -= allocation
            if remaining == 0:
                break

    return quotas


def _value(row: pd.Series, column: str, default: object = None) -> object:
    value = row.get(column, default)
    if value is None or (
        pd.api.types.is_scalar(value) and pd.isna(value)
    ):
        return default
    return value


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _optional_text(row: pd.Series, column: str) -> str | None:
    value = _value(row, column)
    return None if value is None else str(value)


class Ingestor:
    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self.model: TextEmbedding | None = None

    def connect(self) -> psycopg2.extensions.connection:
        return psycopg2.connect(self.db_url)

    @staticmethod
    def chunk_text(
        sentences: Iterable[str],
        max_tokens: int = 500,
        overlap: int = 50,
    ) -> list[str]:
        """Pack sentences into word-count-limited chunks with word overlap."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if overlap < 0 or overlap >= max_tokens:
            raise ValueError("overlap must be between 0 and max_tokens - 1")

        chunks: list[str] = []
        current: list[str] = []

        for sentence in sentences:
            pending = str(sentence).split()
            while pending:
                available = max_tokens - len(current)
                take = min(available, len(pending))
                current.extend(pending[:take])
                del pending[:take]

                if pending:
                    chunks.append(" ".join(current))
                    current = current[-overlap:] if overlap else []

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _ingest_document(
        self,
        cur: psycopg2.extensions.cursor,
        tenant_id: str,
        row: pd.Series,
        max_tokens: int,
        overlap: int,
    ) -> None:
        doc_id = _text(_value(row, "doc_id", ""))
        raw_body = _text(_value(row, "raw_body", ""))
        clean_body = _text(_value(row, "clean_body", ""))
        augmented_body = _text(_value(row, "augmented_body", ""))

        cur.execute(
            """
            INSERT INTO documents (
                doc_id, tenant_id, employee_id, date_str, subject,
                raw_body, clean_body, augmented_body,
                inserted_secrets_count, inserted_injections_count
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                employee_id = EXCLUDED.employee_id,
                date_str = EXCLUDED.date_str,
                subject = EXCLUDED.subject,
                raw_body = EXCLUDED.raw_body,
                clean_body = EXCLUDED.clean_body,
                augmented_body = EXCLUDED.augmented_body,
                inserted_secrets_count = EXCLUDED.inserted_secrets_count,
                inserted_injections_count = EXCLUDED.inserted_injections_count,
                updated_at = now()
            WHERE documents.tenant_id = EXCLUDED.tenant_id;
            """,
            (
                doc_id,
                tenant_id,
                _optional_text(row, "employee_id"),
                _optional_text(row, "date"),
                _optional_text(row, "subject"),
                raw_body,
                clean_body,
                augmented_body,
                int(_value(row, "inserted_secrets_count", 0) or 0),
                int(_value(row, "inserted_injections_count", 0) or 0),
            ),
        )
        if cur.rowcount != 1:
            raise ValueError(
                f"Document {doc_id!r} already exists under a different tenant"
            )

        sentences = _value(row, "sentences")
        if isinstance(sentences, (list, tuple)):
            sentence_list = [
                str(sentence)
                for sentence in sentences
                if sentence is not None
                and not (pd.api.types.is_scalar(sentence) and pd.isna(sentence))
            ]
        else:
            sentence_list = []
        if not sentence_list:
            sentence_list = augmented_body.splitlines()

        chunks = self.chunk_text(sentence_list, max_tokens, overlap)

        # Re-running ingestion replaces this document's chunks and cascades to its
        # old embeddings, avoiding duplicates while remaining safe under tenant RLS.
        cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))

        if not chunks:
            return

        model = self.model
        if model is None:
            print(f"Loading embedding model {MODEL_NAME}...")
            model = TextEmbedding(model_name=MODEL_NAME)
            self.model = model
        embeddings = list(model.embed(chunks))
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Embedding model returned {len(embeddings)} vectors "
                f"for {len(chunks)} chunks of document {doc_id!r}"
            )

        for chunk_index, (text, embedding) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """
                INSERT INTO chunks (doc_id, tenant_id, chunk_index, text)
                VALUES (%s, %s, %s, %s)
                RETURNING chunk_id;
                """,
                (doc_id, tenant_id, chunk_index, text),
            )
            chunk_id = cur.fetchone()[0]

            vector = (
                embedding.tolist()
                if hasattr(embedding, "tolist")
                else list(embedding)
            )
            if len(vector) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"Expected {EMBEDDING_DIMENSIONS}-dimensional embedding, "
                    f"got {len(vector)} for document {doc_id!r}"
                )
            vector_literal = (
                "[" + ",".join(f"{float(value):.8g}" for value in vector) + "]"
            )
            cur.execute(
                """
                INSERT INTO embeddings (chunk_id, tenant_id, vec)
                VALUES (%s, %s, %s::vector);
                """,
                (chunk_id, tenant_id, vector_literal),
            )

    def run(
        self,
        input_path: Path = AUG_INJ_PATH,
        limit: int | None = None,
        max_tokens: int = 500,
        overlap: int = 50,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if overlap < 0 or overlap >= max_tokens:
            raise ValueError("overlap must be between 0 and max_tokens - 1")
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1")
        if not input_path.is_file():
            raise FileNotFoundError(f"Input Parquet file not found: {input_path}")

        print(f"Loading data from {input_path}...")
        parquet_file = pq.ParquetFile(input_path)
        required_columns = {"doc_id", "tenant_id", "augmented_body"}
        parquet_columns = set(parquet_file.schema_arrow.names)
        missing_columns = required_columns - parquet_columns
        if missing_columns:
            raise ValueError(
                f"Input Parquet is missing required columns: {sorted(missing_columns)}"
            )
        columns = [column for column in INGEST_COLUMNS if column in parquet_columns]

        selected_positions: dict[str, set[int]] | None = None
        if limit is not None:
            tenant_counts: dict[str, int] = {}
            for batch in parquet_file.iter_batches(
                batch_size=PARQUET_BATCH_SIZE,
                columns=["tenant_id"],
            ):
                tenant_ids = batch.column(0).to_pandas()
                if tenant_ids.isna().any():
                    raise ValueError(
                        "Input contains a document with a missing ID or tenant"
                    )
                tenant_ids = tenant_ids.astype(str)
                if (tenant_ids.str.strip() == "").any():
                    raise ValueError("Input contains an empty document ID or tenant")
                for tenant_id, count in tenant_ids.value_counts().items():
                    tenant_counts[tenant_id] = (
                        tenant_counts.get(tenant_id, 0) + int(count)
                    )

            quotas = _balanced_quotas(tenant_counts, limit)
            rng = random.Random(BALANCED_SAMPLE_SEED)
            selected_positions = {
                tenant_id: set(rng.sample(range(tenant_counts[tenant_id]), quota))
                for tenant_id, quota in quotas.items()
            }
            print(
                "Balanced sample: "
                + ", ".join(
                    f"{tenant_id}={quotas[tenant_id]:,}"
                    for tenant_id in sorted(quotas)
                )
            )

        conn = self.connect()
        processed_count = 0
        tenant_positions: dict[str, int] = {}
        try:
            for batch in parquet_file.iter_batches(
                batch_size=PARQUET_BATCH_SIZE,
                columns=columns,
            ):
                df = batch.to_pandas()

                if df["doc_id"].isna().any() or df["tenant_id"].isna().any():
                    raise ValueError(
                        "Input contains a document with a missing ID or tenant"
                    )
                df["doc_id"] = df["doc_id"].astype(str)
                df["tenant_id"] = df["tenant_id"].astype(str)
                if (df["doc_id"].str.strip() == "").any() or (
                    df["tenant_id"].str.strip() == ""
                ).any():
                    raise ValueError("Input contains an empty document ID or tenant")

                for tenant_id, group in df.groupby("tenant_id", sort=True):
                    if selected_positions is not None:
                        start = tenant_positions.get(tenant_id, 0)
                        tenant_positions[tenant_id] = start + len(group)
                        selected = selected_positions[tenant_id]
                        group = group.iloc[
                            [
                                offset
                                for offset, position in enumerate(
                                    range(start, start + len(group))
                                )
                                if position in selected
                            ]
                        ]
                        if group.empty:
                            continue

                    print(
                        f"\nProcessing tenant {tenant_id} "
                        f"({len(group)} documents in this batch)..."
                    )
                    with conn:
                        with conn.cursor() as cur:
                            # Transaction-local setting avoids leaking tenant context
                            # between transactions on this connection.
                            cur.execute(
                                "SELECT set_config('app.tenant_id', %s, true)",
                                (tenant_id,),
                            )
                            for _, row in tqdm(
                                group.iterrows(),
                                total=len(group),
                                desc=f"Tenant {tenant_id}",
                            ):
                                self._ingest_document(
                                    cur,
                                    tenant_id,
                                    row,
                                    max_tokens,
                                    overlap,
                                )
                                processed_count += 1
                if limit is not None and processed_count >= limit:
                    break
        finally:
            conn.close()

        if processed_count == 0:
            print("No documents to ingest.")
        else:
            print(f"\nIngestion complete. Processed {processed_count:,} documents.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=AUG_INJ_PATH)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ingest a reproducible, balanced sample across tenants",
    )
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    Ingestor(DB_URL).run(
        input_path=args.input,
        limit=args.limit,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
