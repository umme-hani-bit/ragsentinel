#!/usr/bin/env python3
"""Hybrid retrieval engine: FTS + dense vector search + reranking."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import psycopg2
from fastembed import TextEmbedding
from sentence_transformers import CrossEncoder

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ragsentinel:ragsentinel@localhost:5432/ragsentinel",
)


@dataclass
class ChunkResult:
    chunk_id: int
    doc_id: str
    tenant_id: str
    chunk_index: int
    text: str
    fts_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: float = 0.0
    source: str = ""


@dataclass
class RetrievalResult:
    query: str
    tenant_id: str
    top_n: int
    chunks: list[ChunkResult]
    timings: dict[str, float] = field(default_factory=dict)
    fts_count: int = 0
    dense_count: int = 0
    merged_count: int = 0


class HybridRetriever:
    def __init__(self, db_url: str = DB_URL):
        self.db_url = db_url
        self._conn: psycopg2.extensions.connection | None = None
        self._current_tenant: str | None = None
        self._embed_model: TextEmbedding | None = None
        self._reranker: CrossEncoder | None = None

    @property
    def embed_model(self) -> TextEmbedding:
        if self._embed_model is None:
            print("Loading embedding model (BAAI/bge-large-en-v1.5)...")
            self._embed_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
        return self._embed_model

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            print("Loading reranker (BAAI/bge-reranker-base)...")
            self._reranker = CrossEncoder("BAAI/bge-reranker-base")
        return self._reranker

    def _ensure_conn(self, tenant_id: str) -> psycopg2.extensions.connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.db_url)
            self._conn.autocommit = True
            self._current_tenant = None

        if self._current_tenant != tenant_id:
            with self._conn.cursor() as cur:
                cur.execute("SET app.tenant_id = %s", (tenant_id,))
            self._current_tenant = tenant_id

        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None
        self._current_tenant = None

    def fts_search(self, query: str, tenant_id: str, top_k: int = 50) -> list[ChunkResult]:
        conn = self._ensure_conn(tenant_id)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.chunk_id, c.doc_id, c.tenant_id, c.chunk_index, c.text,
                       ts_rank(c.tsv, plainto_tsquery('english', %s)) AS rank
                FROM chunks c
                WHERE c.tsv @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
                """,
                (query, query, top_k),
            )
            rows = cur.fetchall()

        return [
            ChunkResult(
                chunk_id=row[0],
                doc_id=row[1],
                tenant_id=row[2],
                chunk_index=row[3],
                text=row[4],
                fts_score=float(row[5]),
                source="fts",
            )
            for row in rows
        ]

    def dense_search(self, query: str, tenant_id: str, top_k: int = 50) -> list[ChunkResult]:
        query_vec = list(self.embed_model.embed([query]))[0].tolist()
        vec_str = "[" + ",".join(str(x) for x in query_vec) + "]"

        conn = self._ensure_conn(tenant_id)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.chunk_id, c.doc_id, c.tenant_id, c.chunk_index, c.text,
                       1 - (e.vec <=> %s::vector) AS similarity
                FROM embeddings e
                JOIN chunks c ON c.chunk_id = e.chunk_id
                ORDER BY e.vec <=> %s::vector
                LIMIT %s
                """,
                (vec_str, vec_str, top_k),
            )
            rows = cur.fetchall()

        return [
            ChunkResult(
                chunk_id=row[0],
                doc_id=row[1],
                tenant_id=row[2],
                chunk_index=row[3],
                text=row[4],
                dense_score=float(row[5]),
                source="dense",
            )
            for row in rows
        ]

    def _merge_results(
        self,
        fts_results: list[ChunkResult],
        dense_results: list[ChunkResult],
    ) -> list[ChunkResult]:
        seen: dict[int, ChunkResult] = {}

        for candidate in fts_results:
            seen[candidate.chunk_id] = candidate

        for candidate in dense_results:
            if candidate.chunk_id in seen:
                seen[candidate.chunk_id].dense_score = candidate.dense_score
                seen[candidate.chunk_id].source = "both"
            else:
                seen[candidate.chunk_id] = candidate

        return list(seen.values())

    def _rerank(self, query: str, candidates: list[ChunkResult], top_n: int) -> list[ChunkResult]:
        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.reranker.predict(pairs)

        for i, candidate in enumerate(candidates):
            candidate.rerank_score = float(scores[i])

        candidates.sort(key=lambda item: item.rerank_score, reverse=True)
        return candidates[:top_n]

    def hybrid_search(
        self,
        query: str,
        tenant_id: str,
        top_k_fts: int = 50,
        top_k_dense: int = 50,
        top_n: int = 20,
    ) -> RetrievalResult:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        fts_results = self.fts_search(query, tenant_id, top_k_fts)
        timings["fts_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        dense_results = self.dense_search(query, tenant_id, top_k_dense)
        timings["dense_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        merged = self._merge_results(fts_results, dense_results)
        timings["merge_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        final = self._rerank(query, merged, top_n)
        timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        timings["total_ms"] = round(
            timings["fts_ms"] + timings["dense_ms"] + timings["merge_ms"] + timings["rerank_ms"],
            1,
        )

        return RetrievalResult(
            query=query,
            tenant_id=tenant_id,
            top_n=top_n,
            chunks=final,
            timings=timings,
            fts_count=len(fts_results),
            dense_count=len(dense_results),
            merged_count=len(merged),
        )

    def log_retrieval(self, result: RetrievalResult, defense_profile: str = "none") -> None:
        conn = self._ensure_conn(result.tenant_id)
        qhash = hashlib.sha256(result.query.encode("utf-8")).hexdigest()[:16]

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO retrieval_logs
                    (tenant_id, qhash, query, top_k, timings_ms, defense_profile)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    result.tenant_id,
                    qhash,
                    result.query,
                    result.top_n,
                    json.dumps(result.timings),
                    defense_profile,
                ),
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Test hybrid retrieval")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--tenant", type=str, default="A", help="Tenant ID (A/B/C)")
    parser.add_argument("--top-n", type=int, default=10, help="Final results to return")
    parser.add_argument("--fts-k", type=int, default=50, help="FTS candidates")
    parser.add_argument("--dense-k", type=int, default=50, help="Dense candidates")
    args = parser.parse_args()

    retriever = HybridRetriever()

    print(f"\nQuery: {args.query}")
    print(f"Tenant: {args.tenant}")
    print(f"Parameters: FTS top-{args.fts_k}, Dense top-{args.dense_k}, Final top-{args.top_n}")
    print("-" * 80)

    result = retriever.hybrid_search(
        query=args.query,
        tenant_id=args.tenant,
        top_k_fts=args.fts_k,
        top_k_dense=args.dense_k,
        top_n=args.top_n,
    )

    print(f"\nFTS candidates: {result.fts_count}")
    print(f"Dense candidates: {result.dense_count}")
    print(f"After merge: {result.merged_count}")
    print(f"After rerank: {len(result.chunks)}")
    print(f"\nTimings: {result.timings}")

    print(f"\nTop {len(result.chunks)} results:")
    print("-" * 80)
    for i, chunk in enumerate(result.chunks, 1):
        preview = chunk.text[:120].replace("\n", " ")
        print(
            f"  {i:2d}. [{chunk.source:5s}] doc={chunk.doc_id[:30]:30s} "
            f"rerank={chunk.rerank_score:+.4f}  {preview}..."
        )

    retriever.log_retrieval(result)
    print("\nLogged to retrieval_logs table.")

    retriever.close()


if __name__ == "__main__":
    main()
