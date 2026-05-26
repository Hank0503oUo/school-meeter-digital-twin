from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from src.knowledge_models import (
    ChunkRecord,
    CuratedTraceRecord,
    DocumentRecord,
    MemoryEntry,
    new_id,
)

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None


_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]+")
_KPI_CANDIDATES = ("kwh", "kw", "eui", "cvrmse", "r2", "co2", "carbon", "peak")


def _slugify(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", str(value or "").strip())
    text = text.strip("-").lower()
    return text or "general"


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", str(value or "").strip()) or "upload"


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def _split_tags(raw: str | Sequence[str]) -> list[str]:
    if isinstance(raw, str):
        parts = re.split(r"[,;|]+", raw)
    else:
        parts = [str(item) for item in raw]
    return [part.strip() for part in parts if str(part).strip()]


def _frame_to_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    columns = [str(col) for col in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in frame.columns) + " |")
    return "\n".join([header, separator, *rows])


class KnowledgeWorkbench:
    def __init__(self, root: Path | None = None) -> None:
        demo_root = Path(__file__).resolve().parent.parent
        self.root = Path(root) if root is not None else demo_root / "data" / "knowledge_workbench"
        self.groups_dir = self.root / "groups"
        self.state_dir = self.root / "state"
        self.reports_dir = self.root / "reports"
        self.documents_path = self.state_dir / "documents.json"
        self.chunks_path = self.state_dir / "chunks.json"
        self.ontology_index_path = self.state_dir / "ontology_index.json"
        self.memory_index_path = self.state_dir / "memory_index.json"
        self.curated_path = self.state_dir / "curated_traces.jsonl"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.groups_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        for path, default in (
            (self.documents_path, []),
            (self.chunks_path, []),
            (self.ontology_index_path, {"buildings": {}, "documents": {}, "meters": {}, "kpis": {}}),
            (self.memory_index_path, []),
        ):
            if not path.exists():
                path.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8")
        if not self.curated_path.exists():
            self.curated_path.write_text("", encoding="utf-8")

    def _load_json(self, path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    def _save_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_documents(self) -> list[DocumentRecord]:
        payload = self._load_json(self.documents_path, [])
        return [DocumentRecord.from_dict(item) for item in payload]

    def _save_documents(self, documents: Sequence[DocumentRecord]) -> None:
        self._save_json(self.documents_path, [item.to_dict() for item in documents])

    def _load_chunks(self) -> list[ChunkRecord]:
        payload = self._load_json(self.chunks_path, [])
        return [ChunkRecord.from_dict(item) for item in payload]

    def _save_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
        self._save_json(self.chunks_path, [item.to_dict() for item in chunks])

    def _load_memory(self) -> list[MemoryEntry]:
        payload = self._load_json(self.memory_index_path, [])
        return [MemoryEntry.from_dict(item) for item in payload]

    def _save_memory(self, entries: Sequence[MemoryEntry]) -> None:
        self._save_json(self.memory_index_path, [item.to_dict() for item in entries])

    def list_documents(self, building_id: str | None = None, source_type: str | None = None) -> list[DocumentRecord]:
        documents = self._load_documents()
        if building_id:
            documents = [item for item in documents if item.building_id == _slugify(building_id)]
        if source_type:
            documents = [item for item in documents if item.source_type == source_type]
        return sorted(documents, key=lambda item: item.uploaded_at, reverse=True)

    def list_chunks(self, building_id: str | None = None, doc_id: str | None = None) -> list[ChunkRecord]:
        chunks = self._load_chunks()
        if building_id:
            chunks = [item for item in chunks if item.building_id == _slugify(building_id)]
        if doc_id:
            chunks = [item for item in chunks if item.doc_id == doc_id]
        return chunks

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        for chunk in self._load_chunks():
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        for document in self._load_documents():
            if document.doc_id == doc_id:
                return document
        return None

    def list_buildings(self) -> list[str]:
        documents = self._load_documents()
        if not documents:
            return ["general"]
        return sorted({item.building_id for item in documents})

    def list_memory(self, building_id: str | None = None) -> list[MemoryEntry]:
        entries = self._load_memory()
        if building_id:
            entries = [entry for entry in entries if entry.building_id == _slugify(building_id)]
        return sorted(entries, key=lambda item: item.created_at, reverse=True)

    def list_curated_traces(self, limit: int = 50) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            for line in self.curated_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            return []
        return list(reversed(rows[-limit:]))

    def save_curated_trace(
        self,
        trace: CuratedTraceRecord,
        *,
        save_to_memory: bool = False,
        memory_title: str = "",
    ) -> Path:
        with self.curated_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        if save_to_memory:
            request = trace.request
            result = trace.result
            self.append_memory(
                building_id=str(request.get("building_id", "general")),
                title=memory_title or f"{request.get('task_type', 'analysis')} confirmed result",
                summary=str(result.get("answer_markdown", ""))[:1200],
                source_trace_id=trace.trace_id,
                tags=["curated", str(request.get("task_type", ""))],
            )
        return self.curated_path

    def append_memory(
        self,
        *,
        building_id: str,
        title: str,
        summary: str,
        source_trace_id: str = "",
        tags: Sequence[str] = (),
    ) -> MemoryEntry:
        entry = MemoryEntry(
            memory_id=new_id("mem"),
            building_id=_slugify(building_id),
            title=title.strip() or "Confirmed finding",
            summary=summary.strip(),
            source_trace_id=source_trace_id,
            tags=_split_tags(tags),
        )
        entries = self._load_memory()
        entries.append(entry)
        self._save_memory(entries)
        self._write_group_memory(entry.building_id)
        return entry

    def ingest_upload(
        self,
        *,
        filename: str,
        content: bytes,
        building_id: str,
        title: str = "",
        tags: str | Sequence[str] = (),
    ) -> DocumentRecord:
        building_slug = _slugify(building_id)
        suffix = Path(filename).suffix.lower()
        source_type = {
            ".pdf": "pdf",
            ".csv": "csv",
            ".md": "markdown",
            ".markdown": "markdown",
            ".txt": "text",
        }.get(suffix)
        if source_type is None:
            raise ValueError("Unsupported file type. Use PDF, CSV, Markdown, or text.")

        group_dir = self.groups_dir / building_slug
        raw_dir = group_dir / ("csv" if source_type == "csv" else "docs")
        parsed_dir = group_dir / "parsed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        parsed_dir.mkdir(parents=True, exist_ok=True)

        doc_id = new_id("doc")
        saved_name = f"{doc_id}_{_safe_name(filename)}"
        saved_path = raw_dir / saved_name
        saved_path.write_bytes(content)

        markdown = self._parse_file(saved_path, source_type)
        parsed_path = parsed_dir / f"{doc_id}.md"
        parsed_path.write_text(markdown, encoding="utf-8")

        record = DocumentRecord(
            doc_id=doc_id,
            building_id=building_slug,
            source_type=source_type,
            title=title.strip() or Path(filename).stem,
            path=str(saved_path),
            parsed_md_path=str(parsed_path),
            tags=_split_tags(tags),
            metadata=self._build_metadata(saved_path, source_type),
        )
        documents = self._load_documents()
        documents.append(record)
        self._save_documents(documents)
        self.rebuild_index()
        return record

    def rebuild_index(self) -> int:
        chunks: list[ChunkRecord] = []
        documents = self._load_documents()
        for document in documents:
            parsed_path = Path(document.parsed_md_path)
            try:
                text = parsed_path.read_text(encoding="utf-8")
            except OSError:
                continue
            for idx, chunk_text in enumerate(self._chunk_text(text), start=1):
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"{document.doc_id}_chunk_{idx:03d}",
                        doc_id=document.doc_id,
                        building_id=document.building_id,
                        source_type=document.source_type,
                        title=document.title,
                        text=chunk_text,
                        path=document.path,
                        metadata={"chunk_index": idx},
                    )
                )
        self._save_chunks(chunks)
        self.rebuild_ontology()
        return len(chunks)

    def rebuild_ontology(self) -> dict[str, Any]:
        ontology: dict[str, Any] = {
            "buildings": {},
            "documents": {},
            "meters": {},
            "kpis": {},
        }
        documents = self._load_documents()
        for document in documents:
            building = ontology["buildings"].setdefault(
                document.building_id,
                {"building_id": document.building_id, "documents": [], "meters": [], "kpis": []},
            )
            building["documents"].append(document.doc_id)
            ontology["documents"][document.doc_id] = {
                "title": document.title,
                "building_id": document.building_id,
                "source_type": document.source_type,
                "path": document.path,
                "tags": document.tags,
            }
            if document.source_type == "csv":
                columns = [str(col) for col in document.metadata.get("columns", [])]
                for column in columns:
                    meter_id = f"{document.doc_id}:{column}"
                    ontology["meters"][meter_id] = {
                        "meter_id": meter_id,
                        "building_id": document.building_id,
                        "document_id": document.doc_id,
                        "name": column,
                    }
                    building["meters"].append(meter_id)
                    if any(candidate in column.lower() for candidate in _KPI_CANDIDATES):
                        ontology["kpis"][column] = {
                            "name": column,
                            "building_id": document.building_id,
                            "document_id": document.doc_id,
                        }
                        building["kpis"].append(column)

        self._save_json(self.ontology_index_path, ontology)
        for building_id in self.list_buildings():
            self._write_group_ontology(building_id)
            self._write_group_memory(building_id)
        return ontology

    def get_ontology(self, building_id: str | None = None) -> dict[str, Any]:
        ontology = self._load_json(self.ontology_index_path, {"buildings": {}, "documents": {}, "meters": {}, "kpis": {}})
        if not building_id:
            return ontology
        building_slug = _slugify(building_id)
        building = ontology.get("buildings", {}).get(building_slug, {})
        return {
            "building": building,
            "documents": {
                key: value
                for key, value in ontology.get("documents", {}).items()
                if value.get("building_id") == building_slug
            },
            "meters": {
                key: value
                for key, value in ontology.get("meters", {}).items()
                if value.get("building_id") == building_slug
            },
            "kpis": {
                key: value
                for key, value in ontology.get("kpis", {}).items()
                if value.get("building_id") == building_slug
            },
        }

    def get_group_paths(self, building_id: str) -> dict[str, Path]:
        group_dir = self.groups_dir / _slugify(building_id)
        return {
            "group_dir": group_dir,
            "memory_md": group_dir / "MEMORY.md",
            "ontology_ttl": group_dir / "ONTOLOGY.ttl",
        }

    def status(self) -> dict[str, Any]:
        documents = self._load_documents()
        chunks = self._load_chunks()
        memory = self._load_memory()
        ontology = self._load_json(self.ontology_index_path, {})
        return {
            "buildings": len(self.list_buildings()),
            "documents": len(documents),
            "csv_files": sum(1 for item in documents if item.source_type == "csv"),
            "chunks": len(chunks),
            "memory_entries": len(memory),
            "curated_traces": len(self.list_curated_traces(limit=100000)),
            "ontology_docs": len(ontology.get("documents", {})),
            "ontology_meters": len(ontology.get("meters", {})),
            "ontology_kpis": len(ontology.get("kpis", {})),
        }

    def search_chunks(
        self,
        *,
        query: str,
        building_id: str = "",
        selected_docs: Sequence[str] = (),
        selected_csvs: Sequence[str] = (),
        top_k: int = 6,
    ) -> list[dict[str, Any]]:
        chunks = self._load_chunks()
        building_slug = _slugify(building_id) if building_id else ""
        allowed_ids = set(selected_docs) | set(selected_csvs)
        query_tokens = Counter(_tokenize(query))
        if not query_tokens:
            return []

        hits: list[tuple[float, ChunkRecord]] = []
        for chunk in chunks:
            if building_slug and chunk.building_id != building_slug:
                continue
            if allowed_ids and chunk.doc_id not in allowed_ids:
                continue
            chunk_tokens = Counter(_tokenize(chunk.text))
            overlap = sum(min(chunk_tokens[token], count) for token, count in query_tokens.items())
            density = overlap / max(len(chunk_tokens), 1)
            phrase_bonus = 1.0 if query.strip().lower() in chunk.text.lower() else 0.0
            score = overlap + density + phrase_bonus + chunk.score_hint
            if score <= 0:
                continue
            hits.append((score, chunk))

        hits.sort(key=lambda item: item[0], reverse=True)
        results: list[dict[str, Any]] = []
        for score, chunk in hits[:top_k]:
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "building_id": chunk.building_id,
                    "title": chunk.title,
                    "source_type": chunk.source_type,
                    "score": round(float(score), 4),
                    "path": chunk.path,
                    "text": chunk.text,
                    "excerpt": self._excerpt(chunk.text, query),
                }
            )
        return results

    def describe_csvs(self, csv_ids: Sequence[str]) -> dict[str, Any]:
        summaries: dict[str, Any] = {}
        for doc_id in csv_ids:
            document = self.get_document(doc_id)
            if document is None or document.source_type != "csv":
                continue
            csv_path = Path(document.path)
            try:
                frame = pd.read_csv(csv_path)
            except Exception:
                continue
            numeric = frame.select_dtypes(include=["number"])
            summary = {
                "rows": int(len(frame)),
                "columns": [str(col) for col in frame.columns],
                "numeric_columns": [str(col) for col in numeric.columns],
                "stats": {},
            }
            for column in list(numeric.columns)[:8]:
                series = pd.to_numeric(frame[column], errors="coerce").dropna()
                if series.empty:
                    continue
                summary["stats"][str(column)] = {
                    "mean": round(float(series.mean()), 4),
                    "min": round(float(series.min()), 4),
                    "max": round(float(series.max()), 4),
                }
            summaries[doc_id] = summary
        return summaries

    def export_report(self, building_id: str, title: str, content: str) -> Path:
        building_slug = _slugify(building_id)
        safe_title = _safe_name(title)[:80]
        target = self.reports_dir / f"{building_slug}_{safe_title}.md"
        target.write_text(content, encoding="utf-8")
        return target

    def _chunk_text(self, text: str, *, size: int = 900, overlap: int = 150) -> Iterable[str]:
        cleaned = re.sub(r"\n{3,}", "\n\n", text or "").strip()
        if not cleaned:
            return []
        chunks: list[str] = []
        cursor = 0
        while cursor < len(cleaned):
            stop = min(len(cleaned), cursor + size)
            chunk = cleaned[cursor:stop].strip()
            if chunk:
                chunks.append(chunk)
            if stop >= len(cleaned):
                break
            cursor = max(stop - overlap, cursor + 1)
        return chunks

    def _build_metadata(self, path: Path, source_type: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {"filename": path.name, "suffix": path.suffix.lower()}
        if source_type == "csv":
            try:
                frame = pd.read_csv(path, nrows=25)
                metadata["columns"] = [str(col) for col in frame.columns]
                metadata["preview_rows"] = int(len(frame))
            except Exception:
                metadata["columns"] = []
        return metadata

    def _parse_file(self, path: Path, source_type: str) -> str:
        if source_type in {"markdown", "text"}:
            try:
                return path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return path.read_text(encoding="utf-8", errors="ignore")

        if source_type == "csv":
            frame = pd.read_csv(path)
            lines = [
                f"# CSV Summary: {path.name}",
                "",
                f"- Rows: {len(frame)}",
                f"- Columns: {len(frame.columns)}",
                "",
                "## Columns",
                "",
                ", ".join(str(col) for col in frame.columns),
                "",
            ]
            numeric = frame.select_dtypes(include=["number"])
            if not numeric.empty:
                describe = numeric.describe().transpose().round(4)
                lines.extend(["## Numeric Summary", "", _frame_to_markdown_table(describe.reset_index()), ""])
            lines.extend(["## Preview", "", _frame_to_markdown_table(frame.head(10).reset_index(drop=True)), ""])
            return "\n".join(lines)

        if source_type == "pdf":
            if PdfReader is None:
                raise RuntimeError("PDF support requires pypdf. Install demo requirements first.")
            reader = PdfReader(str(path))
            texts = []
            for idx, page in enumerate(reader.pages, start=1):
                extracted = (page.extract_text() or "").strip()
                if extracted:
                    texts.append(f"## Page {idx}\n\n{extracted}")
            if not texts:
                raise ValueError("The PDF could not be parsed into readable text.")
            return f"# PDF Parse: {path.name}\n\n" + "\n\n".join(texts)

        raise ValueError(f"Unsupported source type: {source_type}")

    def _write_group_memory(self, building_id: str) -> None:
        building_slug = _slugify(building_id)
        group_dir = self.groups_dir / building_slug
        group_dir.mkdir(parents=True, exist_ok=True)
        entries = self.list_memory(building_slug)
        lines = [f"# Memory for {building_slug}", ""]
        if not entries:
            lines.append("- No confirmed findings yet.")
        else:
            for entry in entries[:20]:
                lines.extend(
                    [
                        f"## {entry.title}",
                        f"- Saved: {entry.created_at}",
                        f"- Tags: {', '.join(entry.tags) if entry.tags else 'none'}",
                        "",
                        entry.summary.strip(),
                        "",
                    ]
                )
        (group_dir / "MEMORY.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    def _write_group_ontology(self, building_id: str) -> None:
        building_slug = _slugify(building_id)
        ontology = self.get_ontology(building_slug)
        building = ontology.get("building", {})
        documents = ontology.get("documents", {})
        meters = ontology.get("meters", {})
        kpis = ontology.get("kpis", {})
        group_dir = self.groups_dir / building_slug
        group_dir.mkdir(parents=True, exist_ok=True)

        lines = [
            "@prefix ex: <https://example.org/knowledge#> .",
            "@prefix brick: <https://brickschema.org/schema/Brick#> .",
            "",
            f"ex:{building_slug} a brick:Building .",
            "",
        ]
        for doc_id, document in documents.items():
            doc_slug = _safe_name(doc_id)
            lines.append(
                f'ex:{doc_slug} a ex:Document ; ex:title "{document.get("title", "")}" ; ex:belongsTo ex:{building_slug} .'
            )
        if documents:
            lines.append("")
        for meter_id, meter in meters.items():
            meter_slug = _safe_name(meter_id)
            lines.append(
                f'ex:{meter_slug} a brick:Meter ; ex:name "{meter.get("name", "")}" ; ex:belongsTo ex:{building_slug} .'
            )
        if meters:
            lines.append("")
        for kpi_name in kpis:
            lines.append(
                f'ex:{_safe_name(kpi_name)} a ex:KPI ; ex:name "{kpi_name}" ; ex:belongsTo ex:{building_slug} .'
            )
        if building and not (documents or meters or kpis):
            lines.append(f"# {building_slug} is registered but has no extracted entities yet.")

        (group_dir / "ONTOLOGY.ttl").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    def _excerpt(self, text: str, query: str, radius: int = 180) -> str:
        lowered = text.lower()
        query_text = query.strip().lower()
        if not query_text:
            return text[: radius * 2]
        pos = lowered.find(query_text)
        if pos < 0:
            tokens = _tokenize(query_text)
            for token in tokens:
                pos = lowered.find(token)
                if pos >= 0:
                    break
        if pos < 0:
            return text[: radius * 2]
        start = max(pos - radius, 0)
        stop = min(pos + radius, len(text))
        snippet = text[start:stop].strip()
        if start > 0:
            snippet = "..." + snippet
        if stop < len(text):
            snippet = snippet + "..."
        return snippet


    def _resolve_corpus_path(self, path_str: str) -> Path | None:
        p = Path(path_str).resolve()
        allowed_roots = [
            self.root / "groups",
        ]
        if not any(str(p).startswith(str(root.resolve())) for root in allowed_roots):
            return None
        return p

    def find_documents(
        self,
        query: str = "",
        building_id: str = "",
        source_type: str = "",
        limit: int = 20,
    ) -> dict[str, Any]:
        documents = self._load_documents()
        building_slug = _slugify(building_id) if building_id else ""
        results: list[dict[str, Any]] = []
        query_lower = query.strip().lower()
        for doc in documents:
            if building_slug and doc.building_id != building_slug:
                continue
            if source_type and doc.source_type != source_type:
                continue
            if query_lower:
                searchable = f"{doc.doc_id} {doc.title} {doc.path} {doc.building_id} {' '.join(doc.tags)}".lower()
                if query_lower not in searchable:
                    continue
            results.append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "building_id": doc.building_id,
                "source_type": doc.source_type,
                "path": doc.path,
                "parsed_md_path": doc.parsed_md_path,
                "tags": doc.tags,
            })
            if len(results) >= limit:
                break
        return {"status": "ok", "query": query, "count": len(results), "docs": results}

    def grep_documents(
        self,
        pattern: str,
        building_id: str = "",
        selected_docs: Sequence[str] = (),
        regex: bool = False,
        case_sensitive: bool = False,
        limit: int = 50,
    ) -> dict[str, Any]:
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = None
        if regex:
            try:
                compiled = re.compile(pattern, flags)
            except re.error as exc:
                return {"status": "error", "message": f"Invalid regex: {exc}", "matches": [], "count": 0, "truncated": False}

        documents = self._load_documents()
        building_slug = _slugify(building_id) if building_id else ""
        allowed_ids = set(selected_docs) if selected_docs else None
        matches: list[dict[str, Any]] = []
        truncated = False

        for doc in documents:
            if building_slug and doc.building_id != building_slug:
                continue
            if allowed_ids and doc.doc_id not in allowed_ids:
                continue
            parsed_path = self._resolve_corpus_path(doc.parsed_md_path)
            if parsed_path is None or not parsed_path.exists():
                continue
            try:
                lines = parsed_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line_no, line in enumerate(lines, start=1):
                hit = False
                if compiled:
                    hit = bool(compiled.search(line))
                else:
                    if case_sensitive:
                        hit = pattern in line
                    else:
                        hit = pattern.lower() in line.lower()
                if hit:
                    matches.append({
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "path": str(parsed_path),
                        "line": line_no,
                        "text": line.strip()[:500],
                    })
                    if len(matches) >= limit:
                        truncated = True
                        break
            if truncated:
                break

        return {
            "status": "ok",
            "pattern": pattern,
            "count": len(matches),
            "matches": matches,
            "truncated": truncated,
        }

    def read_document_lines(
        self,
        doc_id: str = "",
        path: str = "",
        start_line: int = 1,
        max_lines: int = 80,
    ) -> dict[str, Any]:
        max_lines = min(max(max_lines, 1), 200)
        start_line = max(start_line, 1)
        resolved_path: Path | None = None

        if doc_id:
            doc = self.get_document(doc_id)
            if doc is None:
                return {"status": "error", "message": f"Document not found: {doc_id}", "lines": [], "start_line": start_line, "end_line": start_line}
            resolved_path = self._resolve_corpus_path(doc.parsed_md_path)
        elif path:
            resolved_path = self._resolve_corpus_path(path)
        else:
            return {"status": "error", "message": "Provide doc_id or path", "lines": [], "start_line": start_line, "end_line": start_line}

        if resolved_path is None:
            return {"status": "error", "message": "Path outside allowed corpus", "lines": [], "start_line": start_line, "end_line": start_line}
        if not resolved_path.exists():
            return {"status": "error", "message": f"File not found: {resolved_path}", "lines": [], "start_line": start_line, "end_line": start_line}

        try:
            all_lines = resolved_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            return {"status": "error", "message": str(exc), "lines": [], "start_line": start_line, "end_line": start_line}

        end_line = min(start_line + max_lines - 1, len(all_lines))
        selected = all_lines[start_line - 1 : end_line]
        return {
            "status": "ok",
            "path": str(resolved_path),
            "total_lines": len(all_lines),
            "start_line": start_line,
            "end_line": end_line,
            "lines": selected,
        }

    def inspect_document_context(
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
        before = max(0, min(before, 20))
        after = max(0, min(after, 30))
        limit = max(1, min(limit, 50))
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = None
        if regex:
            try:
                compiled = re.compile(pattern, flags)
            except re.error as exc:
                return {"status": "error", "message": f"Invalid regex: {exc}", "contexts": [], "count": 0}

        resolved_path: Path | None = None
        title = ""
        resolved_doc_id = doc_id

        if doc_id:
            doc = self.get_document(doc_id)
            if doc is None:
                return {"status": "error", "message": f"Document not found: {doc_id}", "contexts": [], "count": 0}
            resolved_path = self._resolve_corpus_path(doc.parsed_md_path)
            title = doc.title
        elif path:
            resolved_path = self._resolve_corpus_path(path)
            resolved_doc_id = Path(path).stem
        else:
            return {"status": "error", "message": "Provide doc_id or path", "contexts": [], "count": 0}

        if resolved_path is None:
            return {"status": "error", "message": "Path outside allowed corpus", "contexts": [], "count": 0}
        if not resolved_path.exists():
            return {"status": "error", "message": f"File not found: {resolved_path}", "contexts": [], "count": 0}

        try:
            lines = resolved_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return {"status": "error", "message": "Cannot read file", "contexts": [], "count": 0}

        contexts: list[dict[str, Any]] = []
        for line_no, line in enumerate(lines, start=1):
            hit = False
            if compiled:
                hit = bool(compiled.search(line))
            else:
                if case_sensitive:
                    hit = pattern in line
                else:
                    hit = pattern.lower() in line.lower()
            if hit:
                ctx_before = lines[max(0, line_no - 1 - before) : line_no - 1]
                ctx_after = lines[line_no : min(len(lines), line_no + after)]
                contexts.append({
                    "doc_id": resolved_doc_id,
                    "title": title,
                    "line": line_no,
                    "before": ctx_before,
                    "match": line.strip()[:500],
                    "after": ctx_after,
                })
                if len(contexts) >= limit:
                    break

        return {"status": "ok", "pattern": pattern, "count": len(contexts), "contexts": contexts}

    def count_document_matches(
        self,
        pattern: str,
        building_id: str = "",
        regex: bool = False,
        case_sensitive: bool = False,
        limit: int = 50,
    ) -> dict[str, Any]:
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = None
        if regex:
            try:
                compiled = re.compile(pattern, flags)
            except re.error as exc:
                return {"status": "error", "message": f"Invalid regex: {exc}", "docs": []}

        documents = self._load_documents()
        building_slug = _slugify(building_id) if building_id else ""
        results: list[dict[str, Any]] = []

        for doc in documents:
            if building_slug and doc.building_id != building_slug:
                continue
            parsed_path = self._resolve_corpus_path(doc.parsed_md_path)
            if parsed_path is None or not parsed_path.exists():
                continue
            try:
                content = parsed_path.read_text(encoding="utf-8")
            except OSError:
                continue
            lines = content.splitlines()
            count = 0
            for line in lines:
                if compiled:
                    if compiled.search(line):
                        count += 1
                else:
                    if case_sensitive:
                        if pattern in line:
                            count += 1
                    else:
                        if pattern.lower() in line.lower():
                            count += 1
            if count > 0:
                results.append({"doc_id": doc.doc_id, "title": doc.title, "count": count})
                if len(results) >= limit:
                    break

        results.sort(key=lambda x: x["count"], reverse=True)
        return {"status": "ok", "pattern": pattern, "docs": results}


def score_chunk_similarity(query: str, text: str) -> float:
    query_tokens = Counter(_tokenize(query))
    text_tokens = Counter(_tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = sum(min(text_tokens[token], count) for token, count in query_tokens.items())
    denom = math.sqrt(sum(value * value for value in query_tokens.values())) * math.sqrt(
        sum(value * value for value in text_tokens.values())
    )
    if denom <= 0:
        return 0.0
    return overlap / denom



