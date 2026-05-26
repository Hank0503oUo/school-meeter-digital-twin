"""Tests for DCI document tools (v0.45).

Covers:
1. find_documents — query, building_id, source_type filtering
2. grep_documents — plain text + regex, line numbers, limit + truncated
3. read_document_lines — start_line, max_lines, path traversal rejection
4. inspect_document_context — before/match/after context
5. count_document_matches — per-document occurrence counts
6. Invalid regex returns clean error
7. Path outside corpus is rejected
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from src.knowledge_base import KnowledgeWorkbench


@pytest.fixture
def wb(tmp_path: Path) -> KnowledgeWorkbench:
    wb = KnowledgeWorkbench(root=tmp_path / "kw")
    wb._ensure_dirs()
    return wb


def _ingest_doc(wb: KnowledgeWorkbench, *, title: str, content: str, building_id: str = "general") -> str:
    doc = wb.ingest_upload(
        filename=f"{title}.md",
        content=content.encode("utf-8"),
        building_id=building_id,
        title=title,
    )
    return doc.doc_id


class TestFindDocuments:
    def test_find_by_query(self, wb: KnowledgeWorkbench):
        _ingest_doc(wb, title="OpenBSE calibration guide", content="# OpenBSE\nCalibration steps here.")
        _ingest_doc(wb, title="EUI metrics overview", content="# EUI\nEnergy Use Intensity definition.")
        result = wb.find_documents(query="OpenBSE")
        assert result["status"] == "ok"
        assert result["count"] == 1
        assert "OpenBSE" in result["docs"][0]["title"]

    def test_find_by_building_id(self, wb: KnowledgeWorkbench):
        _ingest_doc(wb, title="HVAC report A", content="hvac", building_id="building-a")
        _ingest_doc(wb, title="HVAC report B", content="hvac", building_id="building-b")
        result = wb.find_documents(building_id="building-a")
        assert result["count"] == 1
        assert result["docs"][0]["building_id"] == "building-a"

    def test_find_by_source_type(self, wb: KnowledgeWorkbench):
        _ingest_doc(wb, title="report", content="data", building_id="general")
        result = wb.find_documents(source_type="markdown")
        assert result["count"] >= 1

    def test_find_empty_query_returns_all(self, wb: KnowledgeWorkbench):
        _ingest_doc(wb, title="doc1", content="a", building_id="general")
        _ingest_doc(wb, title="doc2", content="b", building_id="general")
        result = wb.find_documents()
        assert result["count"] == 2

    def test_find_limit(self, wb: KnowledgeWorkbench):
        for i in range(5):
            _ingest_doc(wb, title=f"doc{i}", content="x", building_id="general")
        result = wb.find_documents(limit=2)
        assert result["count"] == 2


class TestGrepDocuments:
    def test_grep_plain_text(self, wb: KnowledgeWorkbench):
        _ingest_doc(wb, title="metrics", content="Line 1\nCV-RMSE is used to evaluate model accuracy.\nLine 3")
        result = wb.grep_documents(pattern="CV-RMSE")
        assert result["status"] == "ok"
        assert result["count"] >= 1
        match = result["matches"][0]
        assert match["line"] == 2
        assert "CV-RMSE" in match["text"]

    def test_grep_regex(self, wb: KnowledgeWorkbench):
        _ingest_doc(wb, title="data", content="EUI = 123.4 kWh/m2\nOther line\nEUI = 56.7")
        result = wb.grep_documents(pattern=r"EUI = \d+\.\d+", regex=True)
        assert result["count"] == 2

    def test_grep_case_insensitive(self, wb: KnowledgeWorkbench):
        _ingest_doc(wb, title="case", content="OpenBSE is great\nopenbse is also great")
        result = wb.grep_documents(pattern="openbse", case_sensitive=False)
        assert result["count"] == 2

    def test_grep_case_sensitive(self, wb: KnowledgeWorkbench):
        _ingest_doc(wb, title="case", content="OpenBSE is great\nopenbse is also great")
        result = wb.grep_documents(pattern="openbse", case_sensitive=True)
        assert result["count"] == 1

    def test_grep_limit_truncated(self, wb: KnowledgeWorkbench):
        content = "\n".join(f"EUI line {i}" for i in range(100))
        _ingest_doc(wb, title="big", content=content)
        result = wb.grep_documents(pattern="EUI", limit=5)
        assert result["count"] == 5
        assert result["truncated"] is True

    def test_grep_selected_docs(self, wb: KnowledgeWorkbench):
        doc_a = _ingest_doc(wb, title="a", content="target keyword here")
        _ingest_doc(wb, title="b", content="target keyword there")
        result = wb.grep_documents(pattern="target", selected_docs=[doc_a])
        assert result["count"] == 1
        assert result["matches"][0]["doc_id"] == doc_a

    def test_grep_invalid_regex(self, wb: KnowledgeWorkbench):
        result = wb.grep_documents(pattern="[invalid", regex=True)
        assert result["status"] == "error"
        assert "Invalid regex" in result["message"]

    def test_grep_empty_result(self, wb: KnowledgeWorkbench):
        _ingest_doc(wb, title="empty", content="nothing relevant")
        result = wb.grep_documents(pattern="CV-RMSE")
        assert result["count"] == 0
        assert result["matches"] == []


class TestReadDocumentLines:
    def test_read_by_doc_id(self, wb: KnowledgeWorkbench):
        _ingest_doc(wb, title="multi", content="\n".join(f"Line {i}" for i in range(50)))
        docs = wb._load_documents()
        doc_id = docs[0].doc_id
        result = wb.read_document_lines(doc_id=doc_id, start_line=10, max_lines=5)
        assert result["status"] == "ok"
        assert result["start_line"] == 10
        assert result["end_line"] == 14
        assert len(result["lines"]) == 5
        assert result["lines"][0] == "Line 9"

    def test_read_max_lines_cap(self, wb: KnowledgeWorkbench):
        _ingest_doc(wb, title="big", content="\n".join(f"L{i}" for i in range(500)))
        docs = wb._load_documents()
        result = wb.read_document_lines(doc_id=docs[0].doc_id, max_lines=300)
        assert result["status"] == "ok"
        assert len(result["lines"]) <= 200

    def test_read_no_doc_id_or_path(self, wb: KnowledgeWorkbench):
        result = wb.read_document_lines()
        assert result["status"] == "error"

    def test_read_nonexistent_doc(self, wb: KnowledgeWorkbench):
        result = wb.read_document_lines(doc_id="doc_nonexistent")
        assert result["status"] == "error"

    def test_read_path_traversal_rejected(self, wb: KnowledgeWorkbench):
        result = wb.read_document_lines(path="../../.env")
        assert result["status"] == "error"
        assert "outside" in result["message"].lower() or "Path" in result["message"]


class TestInspectDocumentContext:
    def test_inspect_context(self, wb: KnowledgeWorkbench):
        lines = ["header", "before 1", "before 2", "EUI definition here", "after 1", "after 2", "footer"]
        _ingest_doc(wb, title="ctx", content="\n".join(lines))
        docs = wb._load_documents()
        result = wb.inspect_document_context(
            pattern="EUI",
            doc_id=docs[0].doc_id,
            before=2,
            after=2,
        )
        assert result["status"] == "ok"
        assert result["count"] == 1
        ctx = result["contexts"][0]
        assert ctx["line"] == 4
        assert len(ctx["before"]) == 2
        assert "EUI" in ctx["match"]
        assert len(ctx["after"]) == 2

    def test_inspect_regex(self, wb: KnowledgeWorkbench):
        _ingest_doc(wb, title="rx", content="Line 1\nvalue=42.5\nLine 3")
        docs = wb._load_documents()
        result = wb.inspect_document_context(
            pattern=r"value=\d+\.\d+",
            doc_id=docs[0].doc_id,
            regex=True,
            before=1,
            after=1,
        )
        assert result["count"] == 1
        assert "42.5" in result["contexts"][0]["match"]

    def test_inspect_no_args(self, wb: KnowledgeWorkbench):
        result = wb.inspect_document_context(pattern="test")
        assert result["status"] == "error"

    def test_inspect_invalid_regex(self, wb: KnowledgeWorkbench):
        result = wb.inspect_document_context(pattern="[bad", regex=True)
        assert result["status"] == "error"


class TestCountDocumentMatches:
    def test_count_matches(self, wb: KnowledgeWorkbench):
        _ingest_doc(wb, title="a", content="OpenBSE\nOpenBSE\nOpenBSE")
        _ingest_doc(wb, title="b", content="OpenBSE once")
        _ingest_doc(wb, title="c", content="nothing here")
        result = wb.count_document_matches(pattern="OpenBSE")
        assert result["status"] == "ok"
        assert len(result["docs"]) == 2
        counts = {d["title"]: d["count"] for d in result["docs"]}
        assert counts["a"] == 3
        assert counts["b"] == 1

    def test_count_by_building(self, wb: KnowledgeWorkbench):
        _ingest_doc(wb, title="x", content="EUI data", building_id="bldg-a")
        _ingest_doc(wb, title="y", content="EUI data", building_id="bldg-b")
        result = wb.count_document_matches(pattern="EUI", building_id="bldg-a")
        assert result["status"] == "ok"
        assert len(result["docs"]) == 1

    def test_count_empty(self, wb: KnowledgeWorkbench):
        _ingest_doc(wb, title="z", content="nothing")
        result = wb.count_document_matches(pattern="CV-RMSE")
        assert result["docs"] == []

    def test_count_sorted_by_count(self, wb: KnowledgeWorkbench):
        _ingest_doc(wb, title="low", content="target")
        _ingest_doc(wb, title="high", content="target\ntarget\ntarget")
        result = wb.count_document_matches(pattern="target")
        assert result["docs"][0]["count"] >= result["docs"][1]["count"]


class TestBackendWrapper:
    def test_backend_find_docs(self, wb: KnowledgeWorkbench):
        from src.knowledge_mcp_backend import KnowledgeMCPBackend
        backend = KnowledgeMCPBackend(workbench=wb)
        _ingest_doc(wb, title="test", content="hello")
        result = backend.find_docs(query="test")
        assert result["status"] == "ok"

    def test_backend_grep_docs(self, wb: KnowledgeWorkbench):
        from src.knowledge_mcp_backend import KnowledgeMCPBackend
        backend = KnowledgeMCPBackend(workbench=wb)
        _ingest_doc(wb, title="g", content="CV-RMSE line")
        result = backend.grep_docs(pattern="CV-RMSE")
        assert result["count"] >= 1

    def test_backend_read_doc_chunk(self, wb: KnowledgeWorkbench):
        from src.knowledge_mcp_backend import KnowledgeMCPBackend
        backend = KnowledgeMCPBackend(workbench=wb)
        doc = _ingest_doc(wb, title="r", content="\n".join(f"L{i}" for i in range(20)))
        result = backend.read_doc_chunk(doc_id=doc, start_line=1, max_lines=5)
        assert result["status"] == "ok"

    def test_backend_inspect_doc_context(self, wb: KnowledgeWorkbench):
        from src.knowledge_mcp_backend import KnowledgeMCPBackend
        backend = KnowledgeMCPBackend(workbench=wb)
        _ingest_doc(wb, title="i", content="before\nEUI target\nafter")
        docs = wb._load_documents()
        result = backend.inspect_doc_context(pattern="EUI", doc_id=docs[0].doc_id, before=1, after=1)
        assert result["count"] == 1

    def test_backend_count_doc_matches(self, wb: KnowledgeWorkbench):
        from src.knowledge_mcp_backend import KnowledgeMCPBackend
        backend = KnowledgeMCPBackend(workbench=wb)
        _ingest_doc(wb, title="c", content="OpenBSE\nOpenBSE")
        result = backend.count_doc_matches(pattern="OpenBSE")
        assert len(result["docs"]) == 1
        assert result["docs"][0]["count"] == 2
