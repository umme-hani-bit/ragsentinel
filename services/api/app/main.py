from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import ConfigError, load_bundle
from app.prompting import (
    build_rag_prompt,
    detect_context_taint,
    parse_llm_json_response,
)
from app.schemas import Citation, QueryRequest, QueryResponse, ToolCall
try:
    from retrieval.hybrid_retriever import (
        HybridRetriever,
        get_embed_model,
        get_reranker,
    )
except ImportError:
    from services.retrieval.hybrid_retriever import (
        HybridRetriever,
        get_embed_model,
        get_reranker,
    )


app = FastAPI(title="RAGsentinel API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_DIR = os.getenv("CONFIG_DIR", "/app/configs")
FTS_K = int(os.getenv("RETRIEVAL_FTS_K", "10"))
DENSE_K = int(os.getenv("RETRIEVAL_DENSE_K", "10"))
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ragsentinel:ragsentinel@localhost:5432/ragsentinel",
)

try:
    BUNDLE = load_bundle(CONFIG_DIR)
except ConfigError as exc:
    raise RuntimeError(f"Config load failed: {exc}") from exc


def _fingerprint() -> str:
    return str(getattr(BUNDLE, "fingerprint", "unknown"))


def _allowed_tenants() -> set[str]:
    corpus = getattr(BUNDLE, "corpus", None)
    raw = getattr(corpus, "tenants", None)
    if isinstance(raw, int):
        return {chr(ord("A") + index) for index in range(raw)}
    if isinstance(raw, (list, tuple, set)):
        return {str(tenant) for tenant in raw}
    return {"A", "B", "C"}


def warmup_models() -> None:
    get_embed_model()
    get_reranker()


def search_hybrid_and_rerank(
    query: str,
    tenant_id: str,
    top_k: int,
    rls_enabled: bool = True,
    defense_profile: str = "rls_only",
) -> Dict[str, Any]:
    if not rls_enabled:
        raise ValueError("RLS is required for retrieval")

    retriever = HybridRetriever(DATABASE_URL)
    try:
        result = retriever.hybrid_search(
            query=query,
            tenant_id=tenant_id,
            top_k_fts=FTS_K,
            top_k_dense=DENSE_K,
            top_n=top_k,
        )
        rows: List[Dict[str, Any]] = []
        for chunk in result.chunks:
            if chunk.tenant_id != tenant_id:
                raise RuntimeError("cross_tenant_chunk_in_retrieval_result")
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "tenant_id": chunk.tenant_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "score": float(chunk.rerank_score),
                    "fts_score": float(chunk.fts_score),
                    "dense_score": float(chunk.dense_score),
                    "source": chunk.source,
                }
            )

        try:
            retriever.log_retrieval(result, defense_profile=defense_profile)
        except Exception as exc:
            print(f"[warn] retrieval log failed: {type(exc).__name__}: {exc}")

        return {
            "chunks": rows,
            "timings_ms": result.timings,
            "fts_count": result.fts_count,
            "dense_count": result.dense_count,
            "merged_count": result.merged_count,
            "reranked": bool(getattr(result, "reranked", False)),
        }
    finally:
        retriever.close()


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/configz")
def configz() -> Dict[str, Any]:
    corpus = getattr(BUNDLE, "corpus", None)
    defenses = getattr(BUNDLE, "defenses", None)

    return {
        "dataset": getattr(corpus, "dataset", "unknown"),
        "dataset_version": getattr(corpus, "dataset_version", "unknown"),
        "tenants": getattr(corpus, "tenants", ["A", "B", "C"]),
        "defenses": {
            "rls_enabled": bool(getattr(defenses, "rls_enabled", True)),
            "injection_scanner_enabled": bool(
                getattr(defenses, "injection_scanner_enabled", False)
            ),
            "tool_guard_enabled": bool(
                getattr(defenses, "tool_guard_enabled", False)
            ),
        },
        "fingerprint": _fingerprint(),
    }


@app.post("/warmup")
def warmup() -> Dict[str, str]:
    try:
        warmup_models()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model warmup failed: {type(exc).__name__}",
        ) from exc
    return {"status": "models_ready"}


def _stub_llm(
    req: QueryRequest,
    contexts: List[Citation],
) -> Dict[str, Any]:
    """Task 15 placeholder. Task 16 replaces this with a hosted model."""
    if contexts:
        top = contexts[0]
        payload = {
            "answer": (
                f"[STUB] Retrieved {len(contexts)} chunk(s) for tenant "
                f"{req.tenant_id}. Best match (doc {top.doc_id}): "
                f"{top.text[:280]}"
            ),
            "citations": [citation.chunk_id for citation in contexts[:2]],
            "tool_calls": [],
            "safety_notes": "stub_generation_task15",
        }
    else:
        payload = {
            "answer": "I don't know based on the available context.",
            "citations": [],
            "tool_calls": [],
            "safety_notes": "stub_generation_no_context",
        }

    return parse_llm_json_response(json.dumps(payload))


@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    timings: Dict[str, float] = {}

    allowed = _allowed_tenants()
    if req.tenant_id not in allowed:
        raise HTTPException(
            status_code=403,
            detail={"error": "unknown_tenant", "allowed": sorted(allowed)},
        )

    if not req.defense_profile.rls_enabled:
        raise HTTPException(
            status_code=400,
            detail="rls_enabled is enforced server-side and cannot be disabled.",
        )

    retrieval_start = time.perf_counter()
    try:
        retrieval = search_hybrid_and_rerank(
            query=req.query,
            tenant_id=req.tenant_id,
            top_k=req.top_k,
            rls_enabled=True,
            defense_profile="rls_only",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Retrieval unavailable: {type(exc).__name__}",
        ) from exc
    timings["retrieval_ms"] = round(
        (time.perf_counter() - retrieval_start) * 1000,
        2,
    )

    for key, value in (retrieval.get("timings_ms") or {}).items():
        timings[f"retrieval_{key}"] = float(value)

    raw_chunks = retrieval["chunks"]
    contexts = [
        Citation(
            chunk_id=chunk["chunk_id"],
            doc_id=chunk["doc_id"],
            tenant_id=chunk["tenant_id"],
            text=chunk["text"],
            score=float(chunk["score"]),
        )
        for chunk in raw_chunks
    ]

    prompt_start = time.perf_counter()
    prompt = build_rag_prompt(
        tenant_id=req.tenant_id,
        query=req.query,
        chunks=raw_chunks,
    )
    timings["prompt_build_ms"] = round(
        (time.perf_counter() - prompt_start) * 1000,
        2,
    )
    timings["prompt_chars"] = float(len(prompt))

    llm_start = time.perf_counter()
    parsed = _stub_llm(req, contexts)
    timings["llm_generation_ms"] = round(
        (time.perf_counter() - llm_start) * 1000,
        2,
    )

    cited_ids = set(parsed.get("citations", []))
    citations = [context for context in contexts if context.chunk_id in cited_ids]

    tool_calls: List[ToolCall] = []
    for item in parsed.get("tool_calls", []):
        if isinstance(item, dict) and item.get("tool_name"):
            tool_calls.append(
                ToolCall(
                    tool_name=str(item["tool_name"]),
                    arguments=item.get("arguments") or {},
                    status=str(item.get("status", "allowed")),
                )
            )

    timings["total_ms"] = round((time.perf_counter() - started) * 1000, 2)

    return QueryResponse(
        tenant_id=req.tenant_id,
        query=req.query,
        answer=parsed.get("answer", ""),
        citations=citations,
        contexts=contexts,
        tool_calls=tool_calls,
        safety_notes=parsed.get("safety_notes", ""),
        taint_flags=detect_context_taint(raw_chunks),
        timings_ms=timings,
        config_fingerprint=_fingerprint(),
        llm_mode="stub",
    )
