from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from src.knowledge_analysis import (
    AnalysisContext,
    CloudAdapterError,
    CloudModelAdapter,
    HeuristicAnalysisAdapter,
    LocalLLMAdapter,
    LocalMCPAnalysisAdapter,
)
from src.knowledge_base import KnowledgeWorkbench
from src.knowledge_models import AnalysisRequest, AnalysisResult


@dataclass(slots=True)
class CampusAssistantSnapshot:
    building_id: str
    building_name: str
    year: int
    meter_name: str
    source: str
    metrics: dict[str, Any]
    campus_metrics: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    scope: str = "building"

    def summary_markdown(self) -> str:
        lines = [
            f"# {self.building_name}",
            "",
            f"- Scope: {self.scope}",
            f"- Year: {self.year}",
            f"- Source: {self.source}",
        ]
        if self.meter_name:
            lines.append(f"- Meter: {self.meter_name}")
        for key, value in self.metrics.items():
            if value in (None, "", []):
                continue
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)


def _harness_memory_enabled() -> bool:
    raw = os.getenv("ENERGY_HARNESS_MEMORY_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off", "disabled"}


def resolve_harness_memory_config() -> dict[str, str] | None:
    if not _harness_memory_enabled():
        return None

    root = Path(os.getenv("ENERGY_HARNESS_MCP_ROOT", r"C:\Users\User\Documents\lm-studio-mcp"))
    python_exe = Path(
        os.getenv(
            "ENERGY_HARNESS_MCP_PYTHON",
            str(root / ".venv" / "Scripts" / "python.exe"),
        )
    )
    server_script = Path(os.getenv("ENERGY_HARNESS_MCP_SERVER", str(root / "server.py")))
    config_path = Path(os.getenv("ENERGY_HARNESS_MCP_CONFIG", str(root / "mcp.config.json")))

    if not (python_exe.is_file() and server_script.is_file() and config_path.is_file()):
        return None

    return {
        "python": str(python_exe),
        "server": str(server_script),
        "cwd": str(root),
        "config": str(config_path),
    }


class CampusAssistantService:
    def __init__(
        self,
        *,
        workbench: KnowledgeWorkbench | None = None,
        cloud_adapter: CloudModelAdapter | None = None,
        fallback_adapter: HeuristicAnalysisAdapter | None = None,
        local_mcp_adapter: LocalMCPAnalysisAdapter | None = None,
        local_llm_adapter: LocalLLMAdapter | None = None,
    ) -> None:
        self.workbench = workbench or KnowledgeWorkbench()
        self.cloud_adapter = cloud_adapter or CloudModelAdapter()
        self.fallback_adapter = fallback_adapter or HeuristicAnalysisAdapter()
        self.local_mcp_adapter = local_mcp_adapter or LocalMCPAnalysisAdapter()
        self.local_llm_adapter = local_llm_adapter or LocalLLMAdapter()

    def analyze(
        self,
        *,
        query: str,
        task_type: str,
        snapshot: CampusAssistantSnapshot,
        force_local_mcp: bool = False,
    ) -> AnalysisResult:
        request = AnalysisRequest(
            building_id=snapshot.building_id,
            task_type=task_type,  # type: ignore[arg-type]
            user_query=query,
        )
        context = self._build_context(query=query, snapshot=snapshot)
        if force_local_mcp:
            return self.local_mcp_adapter.force_analyze(request, context)

        # 1. Local MCP structured tools (deterministic)
        local_result = self.local_mcp_adapter.maybe_analyze(request, context)
        if local_result is not None:
            return local_result

        # 2. Local LLM (priority over cloud)
        if self.local_llm_adapter.available():
            try:
                return self.local_llm_adapter.analyze(request, context)
            except CloudAdapterError:
                pass  # fall through to cloud

        # 3. Cloud (Gemini)
        if self.cloud_adapter.configured():
            try:
                return self.cloud_adapter.analyze(request, context)
            except CloudAdapterError as exc:
                fallback = self.fallback_adapter.analyze(request, context)
                fallback.used_fallback = True
                fallback.warnings.append(str(exc))
                return fallback

        fallback = self.fallback_adapter.analyze(request, context)
        fallback.used_fallback = True
        fallback.warnings.append("Cloud model is not configured; using heuristic fallback.")
        return fallback

    def _build_context(self, *, query: str, snapshot: CampusAssistantSnapshot) -> AnalysisContext:
        tool_trace: list[dict[str, Any]] = []
        chunks = self._related_chunks(query=query, snapshot=snapshot)
        tool_trace.append({"tool": "selected_building_snapshot", "count": len(chunks)})
        ontology = {
            "building": {
                "building_id": snapshot.building_id,
                "building_name": snapshot.building_name,
                "scope": snapshot.scope,
                "year": snapshot.year,
                "meter_name": snapshot.meter_name,
                "source": snapshot.source,
            },
            "metrics": snapshot.metrics,
            "meta": snapshot.meta,
            "campus_metrics": snapshot.campus_metrics,
        }
        memory_entries = [
            entry.to_dict()
            for entry in (
                self.workbench.list_memory(snapshot.building_id)[:5]
                + ([] if snapshot.building_id == "general" else self.workbench.list_memory("general")[:3])
            )
        ]
        tool_trace.append({"tool": "memory_lookup", "count": len(memory_entries)})

        # Harness long-term memory: search the optional local memory MCP and inject matches.
        _harness_count = 0
        try:
            import asyncio
            import json as _json
            from concurrent.futures import ThreadPoolExecutor
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            harness_config = resolve_harness_memory_config()

            async def _harness_search() -> list[dict[str, Any]]:
                if harness_config is None:
                    return []
                params = StdioServerParameters(
                    command=harness_config["python"],
                    args=[harness_config["server"]],
                    cwd=harness_config["cwd"],
                    env={"MCP_CONFIG_PATH": harness_config["config"], "PYTHONUTF8": "1"},
                )
                async with stdio_client(params) as (r, w):
                    async with ClientSession(r, w) as session:
                        await session.initialize()
                        result = await session.call_tool(
                            "search_patterns", arguments={"query": query, "top_k": 3}
                        )
                        raw = "\n".join(
                            str(getattr(p, "text", p)) for p in (result.content or []) if p
                        ).strip()
                        if not raw:
                            return []
                        try:
                            parsed = _json.loads(raw)
                        except _json.JSONDecodeError:
                            return [{"source": "harness_rag", "content": raw}]
                        items = (
                            parsed.get("results") or parsed.get("patterns") or
                            (parsed if isinstance(parsed, list) else [parsed])
                        )
                        return [
                            {"source": "harness_rag", "content": item.get("content", _json.dumps(item))
                             if isinstance(item, dict) else str(item)}
                            for item in items
                        ]

            with ThreadPoolExecutor(max_workers=1) as _pool:
                _harness_entries = _pool.submit(asyncio.run, _harness_search()).result(timeout=15)
            memory_entries.extend(_harness_entries)
            _harness_count = len(_harness_entries)
        except Exception:
            pass

        tool_trace.append({"tool": "harness_rag_search", "count": _harness_count})
        csv_summary = {
            "demo_current_selection": {
                "rows": 1,
                "columns": list(snapshot.metrics.keys()),
                "numeric_columns": [key for key, value in snapshot.metrics.items() if isinstance(value, (int, float))],
                "stats": {
                    key: {"mean": value, "min": value, "max": value}
                    for key, value in snapshot.metrics.items()
                    if isinstance(value, (int, float))
                },
            }
        }
        tool_trace.append({"tool": "query_meter_or_kpi", "count": len(csv_summary["demo_current_selection"]["stats"])})
        return AnalysisContext(
            ontology=ontology,
            memory=memory_entries,
            chunks=chunks,
            csv_summary=csv_summary,
            tool_trace=tool_trace,
        )

    def _related_chunks(self, *, query: str, snapshot: CampusAssistantSnapshot) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = [
            {
                "chunk_id": f"{snapshot.building_id}:snapshot",
                "doc_id": f"{snapshot.building_id}:snapshot",
                "building_id": snapshot.building_id,
                "title": f"{snapshot.building_name} current dashboard snapshot",
                "source_type": "dashboard",
                "score": 1.0,
                "path": "dashboard://current-selection",
                "text": snapshot.summary_markdown(),
                "excerpt": snapshot.summary_markdown(),
            }
        ]
        if snapshot.campus_metrics:
            campus_text = "\n".join(f"- {key}: {value}" for key, value in snapshot.campus_metrics.items())
            chunks.append(
                {
                    "chunk_id": f"{snapshot.building_id}:campus",
                    "doc_id": f"{snapshot.building_id}:campus",
                    "building_id": snapshot.building_id,
                    "title": "Campus comparison snapshot",
                    "source_type": "dashboard",
                    "score": 0.8,
                    "path": "dashboard://campus-summary",
                    "text": campus_text,
                    "excerpt": campus_text,
                }
            )

        search_query = " ".join(part for part in [snapshot.building_name, snapshot.meter_name, query] if part).strip()
        seen_ids = {chunk["chunk_id"] for chunk in chunks}
        for candidate_building in [snapshot.building_id, "general"]:
            hits = self.workbench.search_chunks(query=search_query, building_id=candidate_building, top_k=3)
            for hit in hits:
                if hit["chunk_id"] in seen_ids:
                    continue
                chunks.append(hit)
                seen_ids.add(hit["chunk_id"])
        return chunks[:6]
