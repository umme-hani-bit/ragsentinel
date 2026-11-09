from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class DefenseProfile(BaseModel):
    rls_enabled: bool = True
    injection_scanner_enabled: bool = False
    tool_guard_enabled: bool = False


class QueryRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=32)
    query: str = Field(..., min_length=1, max_length=4000)
    defense_profile: DefenseProfile = Field(default_factory=DefenseProfile)
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    chunk_id: int
    doc_id: str
    tenant_id: str
    text: str
    score: float


class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    status: str = "allowed"


class LLMStructuredOutput(BaseModel):
    answer: str
    citations: List[int] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    safety_notes: str = ""


class QueryResponse(BaseModel):
    tenant_id: str
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    contexts: List[Citation] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    safety_notes: str = ""
    taint_flags: List[str] = Field(default_factory=list)
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    config_fingerprint: str = ""
    llm_mode: str = "stub"
