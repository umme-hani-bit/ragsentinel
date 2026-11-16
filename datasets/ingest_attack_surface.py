#!/usr/bin/env python3
"""Top-up ingestion: guarantee every attack-surface document is in the DB.

The initial corpus sample (--limit 5000) captured only ~1% of the planted
secret/injection docs, so attack-harness queries were targeting documents
that do not exist in Postgres. This script ingests exactly the missing
attack-surface docs, reusing ingest_pipeline.Ingestor so chunking
(500 tokens / 50 overlap), the bge-large embedding model, and all INSERT
paths are byte-identical to the original ingestion.

Safety properties:
- Missing-only: docs already in the DB are skipped (no re-chunking,
  no re-embedding, no duplicates).
- Existence checks use the superuser admin DSN: ragsvc under FORCE RLS
  with no tenant context sees zero rows and would falsely report every
  existing document as missing.
- Ingestion runs as ragsvc with transaction-local app.tenant_id per
  tenant group, exactly mirroring the original pipeline.
- Deterministic: the target set is sorted; --limit takes a stable prefix.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    from datasets.ingest_pipeline import (
        AUG_INJ_PATH,
        DB_URL,
        INGEST_COLUMNS,
        PARQUET_BATCH_SIZE,
        Ingestor,
    )
except ImportError:  # running as `python datasets/ingest_attack_surface.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ingest_pipeline import (  # type: ignore[no-redef]
        AUG_INJ_PATH,
        DB_URL,
        INGEST_COLUMNS,
        PARQUET_BATCH_SIZE,
        Ingestor,
    )

SECRET_SPANS_PATH = Path("data/metadata/secret_spans.parquet")
INJECTION_SPANS_PATH = Path("data/metadata/injection_spans.parquet")

# Admin DSN is used ONLY for existence checks / verification (reads as
# superuser, bypassing RLS). Ingestion uses DB_URL (ragsvc) like the
# original pipeline.
ADMIN_DB_URL = os.getenv(
    "ADMIN_DATABASE_URL",
    "postgresql://ragsentinel:ragsentinel@localhost:5432/ragsentinel",
)


def load_target_doc_ids() -> set[str]:
    """Union of doc_ids carrying planted secrets or injections."""
    targets: set[str] = set()
    for path in (SECRET_SPANS_PATH, INJECTION_SPANS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required metadata file not found: {path}")
        frame = pd.read_parquet(path, columns=["doc_id"])
        targets.update(frame["doc_id"].astype(str))
    return targets


def fetch_ingested_doc_ids() -> set[str]:
    """All doc_ids currently in documents. Admin connection: bypasses RLS."""
    conn = psycopg2.connect(ADMIN_DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT doc_id FROM documents;")
            return {str(row[0]) for row in cur.fetchall()}
    finally:
        conn.close()


class AttackSurfaceIngestor(Ingestor):
    """Ingestor restricted to an explicit set of target doc_ids."""

    def run_targets(
        self,
        target_ids: set[str],
        input_path: Path = AUG_INJ_PATH,
        max_tokens: int = 500,
        overlap: int = 50,
        dry_run: bool = False,
    ) -> set[str]:
        """Ingest every target doc found in the parquet. Returns found ids."""
        if not target_ids:
            print("No target documents to ingest.")
            return set()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input Parquet file not found: {input_path}")

        parquet_file = pq.ParquetFile(input_path)
        parquet_columns = set(parquet_file.schema_arrow.names)
        required_columns = {"doc_id", "tenant_id", "augmented_body"}
        missing_columns = required_columns - parquet_columns
        if missing_columns:
            raise ValueError(
                f"Input Parquet is missing required columns: {sorted(missing_columns)}"
            )
        columns = [c for c in INGEST_COLUMNS if c in parquet_columns]

        found: set[str] = set()
        conn = None if dry_run else self.connect()
        progress = tqdm(total=len(target_ids), desc="Attack-surface docs")
        try:
            for batch in parquet_file.iter_batches(
                batch_size=PARQUET_BATCH_SIZE,
                columns=columns,
            ):
                df = batch.to_pandas()
                df["doc_id"] = df["doc_id"].astype(str)
                df = df[df["doc_id"].isin(target_ids)]
                if df.empty:
                    continue

                if df["tenant_id"].isna().any():
                    raise ValueError("Target document has a missing tenant")
                df["tenant_id"] = df["tenant_id"].astype(str)
                if (df["tenant_id"].str.strip() == "").any():
                    raise ValueError("Target document has an empty tenant")

                for tenant_id, group in df.groupby("tenant_id", sort=True):
                    if dry_run:
                        found.update(group["doc_id"])
                        progress.update(len(group))
                        continue
                    assert conn is not None
                    with conn:
                        with conn.cursor() as cur:
                            # Transaction-local tenant context (same as the
                            # original pipeline) so RLS permits the writes.
                            cur.execute(
                                "SELECT set_config('app.tenant_id', %s, true)",
                                (tenant_id,),
                            )
                            for _, row in group.iterrows():
                                self._ingest_document(
                                    cur, tenant_id, row, max_tokens, overlap
                                )
                                found.add(str(row["doc_id"]))
                                progress.update(1)
        finally:
            progress.close()
            if conn is not None:
                conn.close()

        absent = sorted(target_ids - found)
        if absent:
            print(
                f"\nWARNING: {len(absent)} target doc_ids were NOT found in "
                f"{input_path.name} (metadata/corpus mismatch). First few: "
                f"{absent[:5]}"
            )
        return found


def verify_coverage(target_ids: set[str]) -> bool:
    """Report target coverage; flag ingested docs with zero chunks."""
    conn = psycopg2.connect(ADMIN_DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.doc_id,
                       (SELECT count(*) FROM chunks c WHERE c.doc_id = d.doc_id)
                FROM documents d
                WHERE d.doc_id = ANY(%s);
                """,
                (sorted(target_ids),),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    present = {row[0] for row in rows}
    empty = sorted(doc_id for doc_id, n_chunks in rows if n_chunks == 0)
    missing = sorted(target_ids - present)

    print("\n--- COVERAGE VERIFICATION ---")
    print(f"Target attack-surface docs : {len(target_ids)}")
    print(f"Present in documents       : {len(present)}")
    print(f"Still missing              : {len(missing)}")
    print(f"Present but ZERO chunks    : {len(empty)}  (unretrievable; "
          f"exclude from harness targets)")
    if missing:
        print(f"  missing sample: {missing[:5]}")
    if empty:
        print(f"  zero-chunk sample: {empty[:5]}")

    ok = not missing
    print(f"[{'PASS' if ok else 'FAIL'}] All attack-surface docs ingested")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=AUG_INJ_PATH)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ingest only the first N missing docs (stable order; for testing)",
    )
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only; no database writes, no model load",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.overlap < 0 or args.overlap >= args.max_tokens:
        parser.error("--overlap must be between 0 and --max-tokens - 1")

    targets = load_target_doc_ids()
    print(f"Attack-surface docs in metadata: {len(targets)}")

    ingested = fetch_ingested_doc_ids()
    missing = sorted(targets - ingested)
    print(f"Already ingested: {len(targets) - len(missing)} | missing: {len(missing)}")

    if not missing:
        return 0 if verify_coverage(targets) else 1

    wanted = set(missing[: args.limit]) if args.limit else set(missing)
    print(f"Docs to ingest this run: {len(wanted)}"
          + (" (dry-run)" if args.dry_run else ""))

    AttackSurfaceIngestor(DB_URL).run_targets(
        wanted,
        input_path=args.input,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("Dry run complete — no writes performed.")
        return 0

    return 0 if verify_coverage(targets) else 1


if __name__ == "__main__":
    raise SystemExit(main())