#!/usr/bin/env python3
"""
Hybrid Retrieval Engine: FTS + Dense (pgvector) + Reranking.

Security model:
- RLS is FORCED on all tables; the app role is non-superuser.
- Tenant context is set per connection, and connections are request-scoped.
- SQL also filters on tenant_id explicitly (defence in depth).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import psycopg2
from fastembed import TextEmbedding


# ---------------- configuration ----------------
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ragsentinel:ragsentinel@localhost:5432/ragsentinel",
)

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")

# bge-reranker-base is the target model; the smaller ones are fallbacks
# because this runs on CPU.
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_FALLBACK_MODELS = (
    "Xenova/ms-marco-MiniLM-L-12-v2",
    "Xenova/ms-marco-MiniLM-L-6-v2",
)


# ---------------- module-level model cache ----------------
_MODEL_LOCK = threading.Lock()
_EMBED_MODEL: Optional[TextEmbedding] = None
_RERANKER = None
_RERANKER_LOAD_ATTEMPTED = False
_RERANKER_ERROR = ""


def get_embed_model() -> TextEmbedding:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        with _MODEL_LOCK:
            if _EMBED_MODEL is None:
                print(f"Loading embedding model ({EMBED_MODEL_NAME})...")
                _EMBED_MODEL = TextEmbedding(model_name=EMBED_MODEL_NAME)
                print("Embedding model ready.")
    return _EMBED_MODEL


def _load_cross_encoder():
    """Load a fastembed ONNX cross-encoder, trying several model names."""
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError:
        from fastembed import TextCrossEncoder

    candidates = (RERANK_MODEL,) + RERANK_FALLBACK_MODELS
    last_error: Optional[Exception] = None

    for name in candidates:
        try:
            print(f"Loading reranker ({name})...")
            model = TextCrossEncoder(model_name=name)
            print(f"Reranker ready ({name}).")
            return model, name
        except Exception as exc:
            print(f"  reranker '{name}' unavailable: {type(exc).__name__}: {exc}")
            last_error = exc

    raise RuntimeError(f"No reranker model could be loaded: {last_error}")


def get_reranker():
    """Returns the cross-encoder, or None if unavailable."""
    global _RERANKER, _RERANKER_LOAD_ATTEMPTED, _RERANKER_ERROR

    if _RERANKER is not None:
        return _RERANKER

    with _MODEL_LOCK:
        if _RERANKER is not None:
            return _RERANKER
        if _RERANKER_LOAD_ATTEMPTED:
            return None

        _RERANKER_LOAD_ATTEMPTED = True
        try:
            _RERANKER, _ = _load_cross_encoder()
        except Exception as exc:
            _RERANKER_ERROR = f"{type(exc).__name__}: {exc}"
            print(f"[warn] reranker disabled, using heuristic score: {_RERANKER_ERROR}")
            _RERANKER = None

    return _RERANKER


def reranker_status() -> Dict[str, object]:
    return {
        "loaded": _RERANKER is not None,
        "attempted": _RERANKER_LOAD_ATTEMPTED,
        "error": _RERANKER_ERROR,
    }


# ---------------- result types ----------------
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
    chunks: List[ChunkResult]
    timings: Dict[str, float] = field(default_factory=dict)
    fts_count: int = 0
    dense_count: int = 0
    merged_count: int = 0
    reranked: bool = False


class HybridRetriever:
    def __init__(self, db_url: str = DB_URL):
        self.db_url = db_url
        self._conn: Optional[psycopg2.extensions.connection] = None
        self._current_tenant: Optional[str] = None

    # ---- models (module-cached, no torch) ----
    @property
    def embed_model(self) -> TextEmbedding:
        return get_embed_model()

    @property
    def reranker(self):
        return get_reranker()

    # ---- connection management ----
    def _ensure_conn(self, tenant_id: str) -> psycopg2.extensions.connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.db_url)
            self._conn.autocommit = True
            self._current_tenant = None

        if self._current_tenant != tenant_id:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT set_config('app.tenant_id', %s, false)",
                    (tenant_id,),
                )
            self._current_tenant = tenant_id

        return self._conn

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
        self._conn = None
        self._current_tenant = None

    # ---- FTS ----
    def fts_search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 20,
    ) -> List[ChunkResult]:
        conn = self._ensure_conn(tenant_id)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.chunk_id, c.doc_id, c.tenant_id, c.chunk_index, c.text,
                       ts_rank(c.tsv, plainto_tsquery('english', %s)) AS rank
                FROM chunks c
                WHERE c.tenant_id = %s
                  AND c.tsv @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
                """,
                (query, tenant_id, query, top_k),
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

    # ---- Dense ----
    def dense_search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 20,
    ) -> List[ChunkResult]:
        query_vec = list(self.embed_model.embed([query]))[0].tolist()
        vec_str = "[" + ",".join(f"{value:.8f}" for value in query_vec) + "]"

        conn = self._ensure_conn(tenant_id)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.chunk_id, c.doc_id, c.tenant_id, c.chunk_index, c.text,
                       1 - (e.vec <=> %s::vector) AS similarity
                FROM embeddings e
                JOIN chunks c ON c.chunk_id = e.chunk_id
                WHERE c.tenant_id = %s
                  AND e.tenant_id = %s
                ORDER BY e.vec <=> %s::vector
                LIMIT %s
                """,
                (vec_str, tenant_id, tenant_id, vec_str, top_k),
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

    # ---- Merge ----
    def _merge_results(
        self,
        fts_results: List[ChunkResult],
        dense_results: List[ChunkResult],
    ) -> List[ChunkResult]:
        merged: Dict[int, ChunkResult] = {}

        for candidate in fts_results:
            merged[candidate.chunk_id] = candidate

        for candidate in dense_results:
            if candidate.chunk_id in merged:
                merged[candidate.chunk_id].dense_score = candidate.dense_score
                merged[candidate.chunk_id].source = "both"
            else:
                merged[candidate.chunk_id] = candidate

        return list(merged.values())

    # ---- Rerank with graceful fallback ----
    @staticmethod
    def _heuristic_score(candidate: ChunkResult) -> float:
        """Used when the cross-encoder is unavailable."""
        if candidate.source == "both":
            return 1.0 + (0.5 * candidate.dense_score) + candidate.fts_score
        if candidate.source == "dense":
            return candidate.dense_score
        return candidate.fts_score

    def _rerank(
        self,
        query: str,
        candidates: List[ChunkResult],
        top_n: int,
    ) -> tuple:
        """Returns (ranked_candidates, used_cross_encoder)."""
        if not candidates:
            return [], False

        model = self.reranker

        if model is None:
            for candidate in candidates:
                candidate.rerank_score = self._heuristic_score(candidate)
            candidates.sort(key=lambda candidate: candidate.rerank_score, reverse=True)
            return candidates[:top_n], False

        texts = [candidate.text for candidate in candidates]
        try:
            results = list(model.rerank(query, texts))
            scores = [0.0] * len(candidates)
            for result in results:
                index = int(getattr(result, "index"))
                if 0 <= index < len(candidates):
                    scores[index] = float(getattr(result, "score"))
        except Exception as exc:
            print(
                f"[warn] rerank failed, using heuristic: "
                f"{type(exc).__name__}: {exc}"
            )
            for candidate in candidates:
                candidate.rerank_score = self._heuristic_score(candidate)
            candidates.sort(key=lambda candidate: candidate.rerank_score, reverse=True)
            return candidates[:top_n], False

        for index, candidate in enumerate(candidates):
            candidate.rerank_score = scores[index]

        candidates.sort(key=lambda candidate: candidate.rerank_score, reverse=True)
        return candidates[:top_n], True

    # ---- Hybrid ----
    def hybrid_search(
        self,
        query: str,
        tenant_id: str,
        top_k_fts: int = 20,
        top_k_dense: int = 20,
        top_n: int = 5,
    ) -> RetrievalResult:
        timings: Dict[str, float] = {}

        start = time.perf_counter()
        fts_results = self.fts_search(query, tenant_id, top_k_fts)
        timings["fts_ms"] = round((time.perf_counter() - start) * 1000, 1)

        start = time.perf_counter()
        dense_results = self.dense_search(query, tenant_id, top_k_dense)
        timings["dense_ms"] = round((time.perf_counter() - start) * 1000, 1)

        start = time.perf_counter()
        merged = self._merge_results(fts_results, dense_results)
        timings["merge_ms"] = round((time.perf_counter() - start) * 1000, 1)

        start = time.perf_counter()
        final, used_reranker = self._rerank(query, merged, top_n)
        timings["rerank_ms"] = round((time.perf_counter() - start) * 1000, 1)

        timings["total_ms"] = round(
            timings["fts_ms"]
            + timings["dense_ms"]
            + timings["merge_ms"]
            + timings["rerank_ms"],
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
            reranked=used_reranker,
        )

    # ---- Logging ----
    def log_retrieval(
        self,
        result: RetrievalResult,
        defense_profile: str = "none",
    ) -> None:
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


# ---------------- standalone test ----------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Test hybrid retrieval")
    parser.add_argument("--query", required=True)
    parser.add_argument("--tenant", default="A")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--fts-k", type=int, default=20)
    parser.add_argument("--dense-k", type=int, default=20)
    args = parser.parse_args()

    retriever = HybridRetriever()
    try:
        result = retriever.hybrid_search(
            args.query,
            args.tenant,
            top_k_fts=args.fts_k,
            top_k_dense=args.dense_k,
            top_n=args.top_n,
        )
        print(
            f"FTS={result.fts_count} Dense={result.dense_count} "
            f"Merged={result.merged_count} Final={len(result.chunks)} "
            f"Reranked={result.reranked}"
        )
        print(f"Timings: {result.timings}")
        print(f"Reranker status: {reranker_status()}")
        for index, chunk in enumerate(result.chunks, 1):
            preview = chunk.text[:110].replace("\n", " ")
            print(
                f"{index:2d}. [{chunk.source:5s}] {chunk.doc_id} | "
                f"{chunk.rerank_score:+.4f} | {preview}"
            )
        retriever.log_retrieval(result, defense_profile="standalone_test")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
