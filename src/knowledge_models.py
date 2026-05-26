from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

TaskType = Literal["qa", "structured_extraction", "energy_summary", "report_generation"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass(slots=True)
class DocumentRecord:
    doc_id: str
    building_id: str
    source_type: str
    title: str
    path: str
    parsed_md_path: str
    tags: list[str] = field(default_factory=list)
    uploaded_at: str = field(default_factory=utc_now_iso)
    status: str = "ready"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentRecord":
        return cls(
            doc_id=str(payload.get("doc_id", "")),
            building_id=str(payload.get("building_id", "")),
            source_type=str(payload.get("source_type", "")),
            title=str(payload.get("title", "")),
            path=str(payload.get("path", "")),
            parsed_md_path=str(payload.get("parsed_md_path", "")),
            tags=[str(item) for item in payload.get("tags", [])],
            uploaded_at=str(payload.get("uploaded_at", utc_now_iso())),
            status=str(payload.get("status", "ready")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    building_id: str
    source_type: str
    title: str
    text: str
    path: str
    score_hint: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChunkRecord":
        return cls(
            chunk_id=str(payload.get("chunk_id", "")),
            doc_id=str(payload.get("doc_id", "")),
            building_id=str(payload.get("building_id", "")),
            source_type=str(payload.get("source_type", "")),
            title=str(payload.get("title", "")),
            text=str(payload.get("text", "")),
            path=str(payload.get("path", "")),
            score_hint=float(payload.get("score_hint", 0.0) or 0.0),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class AnalysisRequest:
    building_id: str
    task_type: TaskType
    user_query: str
    selected_docs: list[str] = field(default_factory=list)
    selected_csvs: list[str] = field(default_factory=list)
    request_id: str = field(default_factory=lambda: new_id("req"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnalysisResult:
    answer_markdown: str
    extracted_json: dict[str, Any]
    cited_chunks: list[dict[str, Any]]
    confidence: float
    followups: list[str]
    adapter_name: str
    used_fallback: bool = False
    warnings: list[str] = field(default_factory=list)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryEntry:
    memory_id: str
    building_id: str
    title: str
    summary: str
    created_at: str = field(default_factory=utc_now_iso)
    source_trace_id: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryEntry":
        return cls(
            memory_id=str(payload.get("memory_id", new_id("mem"))),
            building_id=str(payload.get("building_id", "")),
            title=str(payload.get("title", "")),
            summary=str(payload.get("summary", "")),
            created_at=str(payload.get("created_at", utc_now_iso())),
            source_trace_id=str(payload.get("source_trace_id", "")),
            tags=[str(item) for item in payload.get("tags", [])],
        )


@dataclass(slots=True)
class CuratedTraceRecord:
    trace_id: str
    request: dict[str, Any]
    result: dict[str, Any]
    reviewer_notes: str = ""
    approved: bool = True
    saved_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
