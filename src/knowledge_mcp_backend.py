from __future__ import annotations

import json
from typing import Any, Sequence

from src.harness_memory import HarnessMemory
from src.knowledge_analysis import CloudFirstAnalysisService
from src.knowledge_base import KnowledgeWorkbench
from src.knowledge_models import AnalysisRequest, CuratedTraceRecord, new_id


class KnowledgeMCPBackend:
    def __init__(
        self,
        *,
        workbench: KnowledgeWorkbench | None = None,
        analysis_service: CloudFirstAnalysisService | None = None,
        harness_memory: HarnessMemory | None = None,
    ) -> None:
        self.workbench = workbench or KnowledgeWorkbench()
        self.analysis_service = analysis_service or CloudFirstAnalysisService(self.workbench)
        self.harness = harness_memory or HarnessMemory()

    def status(self) -> dict[str, Any]:
        status = self.workbench.status()
        status["cloud_configured"] = self.analysis_service.cloud_adapter.configured()
        return status

    def list_buildings(self) -> list[dict[str, Any]]:
        rows = []
        for building_id in self.workbench.list_buildings():
            docs = self.workbench.list_documents(building_id)
            rows.append(
                {
                    "building_id": building_id,
                    "document_count": len(docs),
                    "csv_count": sum(1 for item in docs if item.source_type == "csv"),
                    "group_paths": {key: str(value) for key, value in self.workbench.get_group_paths(building_id).items()},
                }
            )
        return rows

    def list_documents(self, building_id: str = "") -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.workbench.list_documents(building_id or None)]

    def search_docs(
        self,
        *,
        query: str,
        building_id: str = "",
        selected_docs: Sequence[str] = (),
        selected_csvs: Sequence[str] = (),
        top_k: int = 6,
    ) -> dict[str, Any]:
        hits = self.workbench.search_chunks(
            query=query,
            building_id=building_id,
            selected_docs=selected_docs,
            selected_csvs=selected_csvs,
            top_k=top_k,
        )
        return {
            "query": query,
            "building_id": building_id,
            "count": len(hits),
            "chunks": hits,
        }

    def fetch_chunk(self, *, chunk_id: str) -> dict[str, Any]:
        chunk = self.workbench.get_chunk(chunk_id)
        if chunk is None:
            raise ValueError(f"Chunk not found: {chunk_id}")
        return chunk.to_dict()

    def lookup_building_entity(self, *, building_id: str) -> dict[str, Any]:
        return self.workbench.get_ontology(building_id)

    def query_meter_or_kpi(
        self,
        *,
        building_id: str = "",
        selected_csvs: Sequence[str] = (),
    ) -> dict[str, Any]:
        csv_ids = list(selected_csvs)
        if building_id and not csv_ids:
            csv_ids = [item.doc_id for item in self.workbench.list_documents(building_id, source_type="csv")]
        return self.workbench.describe_csvs(csv_ids)

    def run_analysis(
        self,
        *,
        building_id: str,
        task_type: str,
        user_query: str,
        selected_docs: Sequence[str] = (),
        selected_csvs: Sequence[str] = (),
    ) -> dict[str, Any]:
        request = AnalysisRequest(
            building_id=building_id,
            task_type=task_type,  # type: ignore[arg-type]
            user_query=user_query,
            selected_docs=list(selected_docs),
            selected_csvs=list(selected_csvs),
        )
        result = self.analysis_service.analyze(request)
        return {
            "request": request.to_dict(),
            "result": result.to_dict(),
        }

    def save_curated_trace(
        self,
        *,
        building_id: str,
        task_type: str,
        user_query: str,
        answer_markdown: str,
        extracted_json: dict[str, Any] | None = None,
        cited_chunks: list[dict[str, Any]] | None = None,
        confidence: float = 0.7,
        followups: Sequence[str] = (),
        adapter_name: str = "manual",
        reviewer_notes: str = "",
        save_to_memory: bool = False,
        memory_title: str = "",
    ) -> dict[str, Any]:
        request = AnalysisRequest(
            building_id=building_id,
            task_type=task_type,  # type: ignore[arg-type]
            user_query=user_query,
        )
        result = {
            "answer_markdown": answer_markdown,
            "extracted_json": extracted_json or {},
            "cited_chunks": list(cited_chunks or []),
            "confidence": confidence,
            "followups": list(followups),
            "adapter_name": adapter_name,
        }
        trace = CuratedTraceRecord(
            trace_id=new_id("trace"),
            request=request.to_dict(),
            result=result,
            reviewer_notes=reviewer_notes,
            approved=True,
        )
        path = self.workbench.save_curated_trace(
            trace,
            save_to_memory=save_to_memory,
            memory_title=memory_title or f"{task_type} approved result",
        )
        return {"trace_id": trace.trace_id, "path": str(path)}

    def read_memory(self, building_id: str) -> str:
        path = self.workbench.get_group_paths(building_id)["memory_md"]
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def read_ontology(self, building_id: str) -> str:
        path = self.workbench.get_group_paths(building_id)["ontology_ttl"]
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def read_curated_traces(self, limit: int = 50) -> str:
        rows = self.workbench.list_curated_traces(limit=limit)
        return json.dumps(rows, ensure_ascii=False, indent=2)

    def find_docs(
        self,
        query: str = "",
        building_id: str = "",
        source_type: str = "",
        limit: int = 20,
    ) -> dict[str, Any]:
        return self.workbench.find_documents(
            query=query,
            building_id=building_id,
            source_type=source_type,
            limit=limit,
        )

    def grep_docs(
        self,
        pattern: str,
        building_id: str = "",
        selected_docs: Sequence[str] = (),
        regex: bool = False,
        case_sensitive: bool = False,
        limit: int = 50,
    ) -> dict[str, Any]:
        return self.workbench.grep_documents(
            pattern=pattern,
            building_id=building_id,
            selected_docs=selected_docs,
            regex=regex,
            case_sensitive=case_sensitive,
            limit=limit,
        )

    def read_doc_chunk(
        self,
        doc_id: str = "",
        chunk_id: str = "",
        path: str = "",
        start_line: int = 1,
        max_lines: int = 80,
    ) -> dict[str, Any]:
        return self.workbench.read_document_lines(
            doc_id=doc_id,
            path=path,
            start_line=start_line,
            max_lines=max_lines,
        )

    def inspect_doc_context(
        self,
        pattern: str,
        doc_id: str = "",
        path: str = "",
        before: int = 5,
        after: int = 8,
        regex: bool = False,
        case_sensitive: bool = False,
        limit: int = 10,
    ) -> dict[str, Any]:
        return self.workbench.inspect_document_context(
            pattern=pattern,
            doc_id=doc_id,
            path=path,
            before=before,
            after=after,
            regex=regex,
            case_sensitive=case_sensitive,
            limit=limit,
        )

    def count_doc_matches(
        self,
        pattern: str,
        building_id: str = "",
        regex: bool = False,
        case_sensitive: bool = False,
        limit: int = 50,
    ) -> dict[str, Any]:
        return self.workbench.count_document_matches(
            pattern=pattern,
            building_id=building_id,
            regex=regex,
            case_sensitive=case_sensitive,
            limit=limit,
        )

    def extract_harness_keywords(self, *, query: str) -> dict[str, Any]:
        return self.harness.extract_keywords(query)

    def search_harness_memory(self, *, query: str, top_k: int = 3) -> dict[str, Any]:
        return self.harness.search_memory(query, top_k=top_k)

    def record_harness_event(self, event: dict[str, Any]) -> dict[str, Any]:
        event_id = self.harness.append_event(event)
        promote = event.get("promote_to_procedure", False)
        promotion_result = None
        if promote:
            promotion_result = self.harness.promote_to_procedure(event_id, event.get("intent", ""))
        return {"status": "ok", "event_id": event_id, "promotion": promotion_result}

    def promote_harness_procedure(self, *, event_id: str, procedure_hint: str = "") -> dict[str, Any]:
        return self.harness.promote_to_procedure(event_id, procedure_hint=procedure_hint)

    def get_harness_startup_context(self, *, campus: str = "ntu", limit: int = 8) -> dict[str, Any]:
        return self.harness.get_startup_context(campus=campus, limit=limit)
