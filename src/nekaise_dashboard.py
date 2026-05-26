from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import re

import pandas as pd
import panel as pn
import plotly.graph_objects as go

from src.dashboard_charts import apply_custom_theme
from src.knowledge_analysis import CloudFirstAnalysisService
from src.knowledge_base import KnowledgeWorkbench
from src.knowledge_models import AnalysisRequest, CuratedTraceRecord, new_id
from src.rtem_codex_bridge import local_mcp_available

pn.extension("plotly", "tabulator", sizing_mode="stretch_width")


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CUSTOM_CSS_PATH = _PROJECT_ROOT / "assets" / "custom.css"


def _slugify_like_workbench(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", str(value or "").strip())
    text = text.strip("-").lower()
    return text or "general"


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(value or "").strip()).lower()


def _extract_name_from_label(label: str) -> str:
    text = str(label or "").strip()
    text = re.sub(r"^\[[^\]]+\]\s*", "", text)
    text = re.sub(r"^[^\w\u4e00-\u9fff]+", "", text)
    match = re.match(r"(.+?)\s*\(([^()]+)\)\s*$", text)
    if match:
        return match.group(1).strip()
    return text


def _ensure_custom_css_loaded(path: Path = _CUSTOM_CSS_PATH) -> None:
    try:
        css_text = path.read_text(encoding="utf-8")
    except OSError:
        return
    if css_text not in pn.config.raw_css:
        pn.config.raw_css.append(css_text)


_NEKAISE_CSS = """
.nekaise-shell {
  max-width: 1520px;
  margin: 0 auto;
  padding: 8px 12px 28px;
}

.nekaise-hero,
.nekaise-card,
.nekaise-chat-shell,
.nekaise-file-shell {
  background: var(--dt-surface);
  color: var(--dt-ink);
  border: 1px solid var(--dt-border);
  border-radius: 18px;
  box-shadow: var(--dt-shadow);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
}

.nekaise-hero {
  padding: 24px 26px;
  margin-bottom: 18px;
}

.nekaise-kicker {
  color: var(--dt-accent);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 11px;
  font-weight: 700;
}

.nekaise-hero h1 {
  margin: 8px 0 10px;
  font-size: clamp(34px, 5vw, 60px);
  line-height: 0.98;
  letter-spacing: -0.03em;
  color: var(--dt-ink);
}

.nekaise-hero p,
.nekaise-card p,
.nekaise-panel-note {
  color: var(--dt-ink-soft);
  line-height: 1.7;
  font-size: 14px;
}

.nekaise-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin-top: 18px;
}

.nekaise-metric {
  background: var(--dt-surface-soft);
  border: 1px solid var(--dt-border);
  border-radius: 12px;
  padding: 14px 16px;
}

.nekaise-metric-label {
  color: var(--dt-ink-soft);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 11px;
  font-weight: 600;
}

.nekaise-metric-value {
  margin-top: 6px;
  font-size: 24px;
  font-weight: 700;
  color: var(--dt-ink);
}

.nekaise-card {
  padding: 20px 22px;
}

.nekaise-card h3 {
  margin: 0 0 10px;
  font-size: 22px;
  color: var(--dt-ink);
}

.nekaise-tabs .bk-tabs-header {
  display: inline-flex;
  background: rgba(255, 255, 255, 0.4);
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.2);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  padding: 4px;
  border-radius: 20px;
  margin-bottom: 16px;
  border-bottom: none !important;
}

.nekaise-tabs .bk-tab {
  font-size: 14px;
  font-weight: 500;
  color: var(--dt-ink-soft) !important;
  padding: 8px 16px !important;
  margin: 0 !important;
  border: none !important;
  border-radius: 16px;
  background: transparent !important;
}

.nekaise-tabs .bk-tab:hover {
  color: var(--dt-ink) !important;
  background: rgba(32, 33, 36, 0.04) !important;
}

.nekaise-tabs .bk-tab.bk-active {
  background: rgba(255, 255, 255, 0.82) !important;
  color: var(--dt-accent) !important;
  font-weight: 600;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.nekaise-chat-shell {
  overflow: hidden;
}

.nekaise-chat-header {
  padding: 20px 22px 14px;
  border-bottom: 1px solid var(--dt-border);
}

.nekaise-chat-header h3 {
  margin: 0 0 6px;
  font-size: 26px;
  color: var(--dt-ink);
}

.nekaise-chat-body {
  max-height: 640px;
  overflow-y: auto;
  padding: 12px 16px 8px;
}

.nekaise-chat-body::-webkit-scrollbar {
  width: 10px;
}

.nekaise-chat-body::-webkit-scrollbar-thumb {
  background: rgba(95, 99, 104, 0.32);
  border-radius: 999px;
}

.nekaise-message {
  margin: 12px 0;
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid var(--dt-border);
  background: var(--dt-surface-soft);
}

.nekaise-message-user {
  border-left: 4px solid var(--dt-accent);
}

.nekaise-message-agent {
  border-left: 4px solid #0ea5e9;
}

.nekaise-message-role,
.nekaise-message-meta,
.nekaise-file-title {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}

.nekaise-message-role {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--dt-ink-soft);
  margin-bottom: 8px;
}

.nekaise-message-meta {
  font-size: 11px;
  color: var(--dt-ink-soft);
  margin-top: 8px;
}

.nekaise-chat-inputbar {
  padding: 16px;
  border-top: 1px solid var(--dt-border);
}

.nekaise-file-shell {
  padding: 20px 22px;
  min-height: 680px;
}

.nekaise-file-title {
  color: var(--dt-accent);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 8px;
}

.nekaise-file-shell pre {
  background: var(--dt-surface-soft);
  border: 1px solid var(--dt-border);
  border-radius: 14px;
  padding: 16px;
  color: var(--dt-ink);
  max-height: 620px;
  overflow: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 12px;
  line-height: 1.65;
  white-space: pre-wrap;
}

@media (max-width: 960px) {
  .nekaise-shell {
    padding: 6px 0 24px;
  }

  .nekaise-hero {
    padding: 20px;
  }
}
"""


_ensure_custom_css_loaded()
if _NEKAISE_CSS not in pn.config.raw_css:
    pn.config.raw_css.append(_NEKAISE_CSS)


class NekaiseWorkbenchDashboard:
    def __init__(self) -> None:
        self.workbench = KnowledgeWorkbench()
        self.analysis = CloudFirstAnalysisService(self.workbench)
        self.last_request: AnalysisRequest | None = None
        self.last_result: dict[str, Any] | None = None
        self.external_context: dict[str, Any] = {
            "campus_id": "",
            "campus_name": "",
            "year": "",
            "selected_uid": "",
            "selected_label": "",
            "mapped_building": "",
            "task_type": "",
        }
        buildings = self.workbench.list_buildings()
        default_building = buildings[0]

        self.chat_entries: list[dict[str, str]] = [
            {
                "role": "agent",
                "body": (
                    "Ask grounded building questions here, or run direct MCP commands such as "
                    "`dataset_statistics`, `list_buildings`, "
                    "`forecast_meter meter_id=01B_P1_01 horizon_hours=24`, or "
                    "`predict_energy t_out=35 humidity=70 hours=24`."
                ),
                "meta": "Workbench chat with local MCP bridge",
            }
        ]

        self.building_select = pn.widgets.Select(
            name="Building",
            options=buildings,
            value=default_building,
        )
        self.task_select = pn.widgets.Select(
            name="Mode",
            options={
                "Agent Q&A": "qa",
                "Ops Summary": "energy_summary",
                "Structured Extraction": "structured_extraction",
                "Report": "report_generation",
            },
            value="qa",
        )
        self.auto_save_memory = pn.widgets.Checkbox(name="Auto Save to MEMORY.md", value=False)
        self.query_input = pn.widgets.TextAreaInput(
            name="Prompt",
            placeholder="Ask about the building or enter a direct MCP command...",
            height=92,
        )
        self.send_button = pn.widgets.Button(name="Ask Nekaise Agent", button_type="primary")
        self.save_button = pn.widgets.Button(name="Save Last Result", button_type="success")
        self.reset_chat_button = pn.widgets.Button(name="Reset Chat", button_type="default")

        self.upload_building = pn.widgets.TextInput(name="Upload to Building", value=default_building)
        self.upload_title = pn.widgets.TextInput(name="Title", placeholder="Optional display title")
        self.upload_tags = pn.widgets.TextInput(name="Tags", placeholder="ahu, baseline, csv")
        self.file_input = pn.widgets.FileInput(name="Upload PDF / Markdown / CSV", accept=".pdf,.md,.markdown,.txt,.csv")
        self.upload_button = pn.widgets.Button(name="Upload File", button_type="primary")
        self.rebuild_button = pn.widgets.Button(name="Rebuild Index", button_type="default")

        self.chat_body = pn.Column(css_classes=["nekaise-chat-body"], sizing_mode="stretch_both")
        self.agent_graph = pn.pane.Plotly(height=640, sizing_mode="stretch_width", config={"displayModeBar": False})
        self.home_chart = pn.pane.Plotly(height=280, sizing_mode="stretch_width", config={"displayModeBar": False})
        self.ontology_graph = pn.pane.Plotly(height=640, sizing_mode="stretch_width", config={"displayModeBar": False})
        self.ontology_file = pn.pane.HTML("", sizing_mode="stretch_both")
        self.memory_file = pn.pane.HTML("", sizing_mode="stretch_both")
        self.home_status = pn.pane.HTML("", sizing_mode="stretch_width")
        self.home_note = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.home_context = pn.pane.HTML("", sizing_mode="stretch_width")
        self.upload_status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.agent_status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.agent_context = pn.pane.HTML("", sizing_mode="stretch_width")
        self.memory_status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.documents_table = pn.widgets.Tabulator(pd.DataFrame(), height=260, disabled=True, sizing_mode="stretch_width")
        self.curated_table = pn.widgets.Tabulator(pd.DataFrame(), height=260, disabled=True, sizing_mode="stretch_width")

        self.send_button.on_click(self._send_query)
        self.save_button.on_click(self._save_last_result)
        self.reset_chat_button.on_click(self._reset_chat)
        self.upload_button.on_click(self._handle_upload)
        self.rebuild_button.on_click(self._handle_rebuild)
        self.building_select.param.watch(lambda event: self.refresh_all(), "value")

        self.refresh_all()

    def refresh_all(self) -> None:
        buildings = self.workbench.list_buildings()
        self.building_select.options = buildings
        if self.building_select.value not in buildings:
            self.building_select.value = buildings[0]
        self.upload_building.value = str(self.building_select.value)
        self._refresh_context_status()
        self._refresh_home()
        self._refresh_agent_graph()
        self._refresh_files()
        self._refresh_tables()
        self._render_chat()

    def set_external_context(
        self,
        *,
        campus_id: str = "",
        campus_name: str = "",
        year: int | str | None = None,
        selected_uid: str = "",
        selected_label: str = "",
        task_type: str = "",
        query: str = "",
    ) -> None:
        mapped = self._map_external_building(selected_uid=selected_uid, selected_label=selected_label)
        self.external_context = {
            "campus_id": str(campus_id or "").strip(),
            "campus_name": str(campus_name or "").strip(),
            "year": "" if year is None else str(year),
            "selected_uid": str(selected_uid or "").strip(),
            "selected_label": str(selected_label or "").strip(),
            "mapped_building": mapped or "",
            "task_type": str(task_type or "").strip(),
        }
        if mapped and mapped in list(self.building_select.options) and self.building_select.value != mapped:
            self.building_select.value = mapped
        if task_type and task_type in set(self.task_select.options.values()):
            self.task_select.value = task_type
        if query and not str(self.query_input.value or "").strip():
            self.query_input.value = str(query)
        self._refresh_context_status()

    def _refresh_context_status(self) -> None:
        selected_demo_label = str(self.external_context.get("selected_label", "") or "Not linked")
        selected_demo_uid = str(self.external_context.get("selected_uid", "") or "")
        selected_campus = str(self.external_context.get("campus_name", "") or "")
        selected_year = str(self.external_context.get("year", "") or "")
        mapped_building = str(self.external_context.get("mapped_building", "") or "")
        task_type = str(self.external_context.get("task_type", "") or str(self.task_select.value))
        current_building = str(self.building_select.value)
        if selected_demo_uid and selected_demo_uid != "ALL":
            demo_line = f"{selected_demo_label} ({selected_demo_uid})" if selected_demo_label and selected_demo_uid not in selected_demo_label else selected_demo_label
        else:
            demo_line = selected_demo_label or "Campus overview"
        link_status = mapped_building or "No exact KB mapping yet"
        html = f"""
        <div class="nekaise-card">
          <div class="nekaise-kicker">Synced Demo Context</div>
          <h3>Current shared state</h3>
          <p>The main campus dashboard and this agent workspace now share the same high-level context. When a matching knowledge-base building exists, the agent switches automatically.</p>
          <div class="nekaise-metrics">
            <div class="nekaise-metric"><div class="nekaise-metric-label">Campus</div><div class="nekaise-metric-value">{selected_campus or 'N/A'}</div></div>
            <div class="nekaise-metric"><div class="nekaise-metric-label">Year</div><div class="nekaise-metric-value">{selected_year or 'N/A'}</div></div>
            <div class="nekaise-metric"><div class="nekaise-metric-label">Demo Building</div><div class="nekaise-metric-value">{demo_line}</div></div>
            <div class="nekaise-metric"><div class="nekaise-metric-label">KB Building</div><div class="nekaise-metric-value">{current_building}</div></div>
            <div class="nekaise-metric"><div class="nekaise-metric-label">Link Status</div><div class="nekaise-metric-value">{link_status}</div></div>
            <div class="nekaise-metric"><div class="nekaise-metric-label">Mode</div><div class="nekaise-metric-value">{task_type or 'qa'}</div></div>
          </div>
        </div>
        """
        self.home_context.object = html
        self.agent_context.object = html

    def _map_external_building(self, *, selected_uid: str, selected_label: str) -> str | None:
        uid = str(selected_uid or "").strip()
        if not uid or uid == "ALL":
            return None
        buildings = [str(item) for item in self.workbench.list_buildings()]
        if not buildings:
            return None

        name = _extract_name_from_label(selected_label)
        raw_candidates = [uid, selected_label, name]
        for value in list(raw_candidates):
            if value:
                raw_candidates.append(_slugify_like_workbench(value))

        exact_set = {item: item for item in buildings}
        for candidate in raw_candidates:
            if candidate in exact_set:
                return exact_set[candidate]

        normalized_map = {_normalize_identifier(item): item for item in buildings}
        for candidate in raw_candidates:
            normalized = _normalize_identifier(candidate)
            if normalized and normalized in normalized_map:
                return normalized_map[normalized]
        return None

    def _refresh_home(self) -> None:
        status = self.workbench.status()
        local_ready = local_mcp_available()
        self.home_status.object = f"""
        <div class="nekaise-card">
          <div class="nekaise-kicker">Knowledge Core</div>
          <h3>Operational status</h3>
          <p>Ontology, memory, uploaded evidence, curated traces, and the RTEM local MCP bridge are surfaced from the same workspace so the agent stays grounded in real building context.</p>
          <div class="nekaise-metrics">
            <div class="nekaise-metric"><div class="nekaise-metric-label">Buildings</div><div class="nekaise-metric-value">{status["buildings"]}</div></div>
            <div class="nekaise-metric"><div class="nekaise-metric-label">Documents</div><div class="nekaise-metric-value">{status["documents"]}</div></div>
            <div class="nekaise-metric"><div class="nekaise-metric-label">Chunks</div><div class="nekaise-metric-value">{status["chunks"]}</div></div>
            <div class="nekaise-metric"><div class="nekaise-metric-label">Memory</div><div class="nekaise-metric-value">{status["memory_entries"]}</div></div>
            <div class="nekaise-metric"><div class="nekaise-metric-label">Curated</div><div class="nekaise-metric-value">{status["curated_traces"]}</div></div>
            <div class="nekaise-metric"><div class="nekaise-metric-label">Local MCP</div><div class="nekaise-metric-value">{'READY' if local_ready else 'OFF'}</div></div>
          </div>
        </div>
        """
        self.home_note.object = (
            "Example commands:\n"
            "- `dataset_statistics`\n"
            "- `list_buildings`\n"
            "- `forecast_meter meter_id=01B_P1_01 horizon_hours=24`\n"
            "- `predict_energy t_out=35 humidity=70 hours=24`\n"
            "- `rank_buildings metric=eui_deviation year=2017`"
        )
        self.home_chart.object = self._build_status_chart(status)

    def _refresh_agent_graph(self) -> None:
        building_id = str(self.building_select.value)
        ontology = self.workbench.get_ontology(building_id)
        self.agent_graph.object = self._build_ontology_figure(ontology, title=building_id)
        self.ontology_graph.object = self._build_ontology_figure(ontology, title=f"{building_id} ontology")

    def _refresh_files(self) -> None:
        paths = self.workbench.get_group_paths(str(self.building_select.value))
        ontology_text = self._safe_read(paths["ontology_ttl"], fallback="# ONTOLOGY.ttl\n\nNo ontology yet.")
        memory_text = self._safe_read(paths["memory_md"], fallback="# MEMORY.md\n\nNo confirmed findings yet.")
        self.ontology_file.object = self._wrap_pre("ONTOLOGY.ttl", ontology_text)
        self.memory_file.object = self._wrap_pre("MEMORY.md", memory_text)

    def _refresh_tables(self) -> None:
        documents = self.workbench.list_documents(str(self.building_select.value))
        curated = self.workbench.list_curated_traces(limit=50)
        self.documents_table.value = pd.DataFrame(
            [
                {
                    "title": item.title,
                    "type": item.source_type,
                    "tags": ", ".join(item.tags),
                    "doc_id": item.doc_id,
                }
                for item in documents
            ]
        )
        self.curated_table.value = pd.DataFrame(
            [
                {
                    "trace_id": row.get("trace_id", ""),
                    "building_id": row.get("request", {}).get("building_id", ""),
                    "task_type": row.get("request", {}).get("task_type", ""),
                    "saved_at": row.get("saved_at", ""),
                }
                for row in curated
            ]
        )

    def _render_chat(self) -> None:
        items = []
        for entry in self.chat_entries:
            role = entry["role"]
            css = ["nekaise-message", "nekaise-message-user" if role == "user" else "nekaise-message-agent"]
            items.append(
                pn.Column(
                    pn.pane.HTML(f"<div class='nekaise-message-role'>{role.upper()}</div>"),
                    pn.pane.Markdown(entry["body"], sizing_mode="stretch_width"),
                    pn.pane.HTML(
                        f"<div class='nekaise-message-meta'>{entry['meta']}</div>" if entry.get("meta") else "",
                        sizing_mode="stretch_width",
                    ),
                    css_classes=css,
                    sizing_mode="stretch_width",
                )
            )
        self.chat_body[:] = items

    def _send_query(self, event=None) -> None:
        query = str(self.query_input.value).strip()
        if not query:
            self.agent_status.object = "Enter a question or an MCP command first."
            return

        building_id = str(self.building_select.value)
        self.chat_entries.append({"role": "user", "body": query, "meta": building_id})
        request = AnalysisRequest(
            building_id=building_id,
            task_type=str(self.task_select.value),
            user_query=query,
            selected_docs=self._all_doc_ids(building_id),
            selected_csvs=self._all_csv_ids(building_id),
        )
        result = self.analysis.analyze(request)
        meta = f"adapter={result.adapter_name} | confidence={result.confidence:.2f} | fallback={result.used_fallback}"
        self.chat_entries.append({"role": "agent", "body": result.answer_markdown, "meta": meta})
        self.last_request = request
        self.last_result = result.to_dict()
        self.agent_status.object = "\n".join(
            [
                f"- adapter: `{result.adapter_name}`",
                f"- fallback: `{result.used_fallback}`",
                f"- confidence: `{result.confidence:.2f}`",
            ]
            + [f"- warning: {item}" for item in result.warnings]
        )
        self.query_input.value = ""
        self._render_chat()

        if self.auto_save_memory.value:
            self._save_last_result(auto_mode=True)

    def _save_last_result(self, event=None, auto_mode: bool = False) -> None:
        if self.last_request is None or self.last_result is None:
            self.memory_status.object = "Run one conversation turn first."
            return
        trace = CuratedTraceRecord(
            trace_id=new_id("trace"),
            request=self.last_request.to_dict(),
            result=self.last_result,
            reviewer_notes="Saved from the Nekaise workbench.",
            approved=True,
        )
        target = self.workbench.save_curated_trace(
            trace,
            save_to_memory=True,
            memory_title=f"{self.last_request.task_type} assistant result",
        )
        self.memory_status.object = (
            f"Saved `{trace.trace_id}` to curated traces and MEMORY.md.\n\n"
            f"- path: `{target}`\n"
            f"- building: `{self.last_request.building_id}`\n"
            f"- mode: `{'auto' if auto_mode else 'manual'}`"
        )
        self.refresh_all()

    def _reset_chat(self, event=None) -> None:
        self.chat_entries = self.chat_entries[:1]
        self.agent_status.object = "Conversation reset."
        self._render_chat()

    def _handle_upload(self, event=None) -> None:
        if not self.file_input.filename or self.file_input.value is None:
            self.upload_status.object = "Choose a file first."
            return
        record = self.workbench.ingest_upload(
            filename=str(self.file_input.filename),
            content=bytes(self.file_input.value),
            building_id=self.upload_building.value,
            title=self.upload_title.value,
            tags=self.upload_tags.value,
        )
        self.upload_status.object = (
            f"Uploaded `{record.title}` into `{record.building_id}`.\n\n"
            f"- doc_id: `{record.doc_id}`\n"
            f"- source_type: `{record.source_type}`"
        )
        self.upload_title.value = ""
        self.upload_tags.value = ""
        self.refresh_all()

    def _handle_rebuild(self, event=None) -> None:
        count = self.workbench.rebuild_index()
        self.upload_status.object = f"Rebuilt index. Total chunks: `{count}`"
        self.refresh_all()

    def _all_doc_ids(self, building_id: str) -> list[str]:
        docs = []
        for source_type in ("markdown", "text", "pdf"):
            docs.extend(self.workbench.list_documents(building_id, source_type=source_type))
        return [item.doc_id for item in docs]

    def _all_csv_ids(self, building_id: str) -> list[str]:
        return [item.doc_id for item in self.workbench.list_documents(building_id, source_type="csv")]

    def build_tabs(self, *, active: int = 1) -> pn.Tabs:
        home_page = pn.Column(
            pn.pane.HTML(
                """
                <div class="nekaise-hero">
                  <div class="nekaise-kicker">Open Building Intelligence</div>
                  <h1>Agent workflow, but inside the current demo.</h1>
                  <p>Use the same knowledge base, ontology, memory, uploaded evidence, and local RTEM MCP tools in one workbench. The visual language follows the existing demo so this page feels like part of the same product, not a separate site.</p>
                </div>
                """,
                sizing_mode="stretch_width",
            ),
            self.home_context,
            self.home_status,
            self.home_chart,
            pn.Row(
                pn.Column(
                    pn.pane.HTML("<div class='nekaise-card'><h3>Document intake</h3><p>Upload Markdown, PDF, text, or CSV evidence and rebuild the workbench index.</p></div>"),
                    pn.Row(self.upload_building, self.upload_title, self.upload_tags, sizing_mode="stretch_width"),
                    pn.Row(self.file_input, self.upload_button, self.rebuild_button, sizing_mode="stretch_width"),
                    self.upload_status,
                    sizing_mode="stretch_both",
                ),
                pn.Column(
                    pn.pane.HTML("<div class='nekaise-card'><h3>Agent notes</h3><p>Use grounded Q&A for document evidence, or explicit commands when you want deterministic MCP execution.</p></div>"),
                    self.home_note,
                    sizing_mode="stretch_both",
                ),
                sizing_mode="stretch_both",
            ),
            pn.Row(
                pn.Column(pn.pane.HTML("<div class='nekaise-card'><h3>Documents</h3></div>"), self.documents_table, sizing_mode="stretch_both"),
                pn.Column(pn.pane.HTML("<div class='nekaise-card'><h3>Curated traces</h3></div>"), self.curated_table, sizing_mode="stretch_both"),
                sizing_mode="stretch_both",
            ),
            css_classes=["nekaise-shell"],
            sizing_mode="stretch_both",
        )

        agent_page = pn.Column(
            pn.Row(
                self.building_select,
                self.task_select,
                self.auto_save_memory,
                sizing_mode="stretch_width",
            ),
            pn.Row(
                pn.Column(
                    self.agent_context,
                    pn.pane.HTML("<div class='nekaise-panel-note'>The graph below is built from the current ONTOLOGY.ttl projection for the selected building.</div>"),
                    self.agent_graph,
                    sizing_mode="stretch_both",
                ),
                pn.Column(
                    pn.pane.HTML("<div class='nekaise-chat-header'><h3>Nekaise Agent</h3><div class='nekaise-panel-note'>Ask about the building or enter direct commands like forecast_meter, dataset_statistics, or predict_energy.</div></div>"),
                    self.chat_body,
                    pn.Column(
                        pn.Row(
                            self.query_input,
                            pn.Column(
                                self.send_button, 
                                self.save_button, 
                                self.reset_chat_button,
                                sizing_mode="stretch_width",
                                margin=(0, 0, 0, 8)
                            ),
                            sizing_mode="stretch_width"
                        ),
                        self.agent_status, 
                        css_classes=["nekaise-chat-inputbar"], 
                        sizing_mode="stretch_width"
                    ),
                    css_classes=["nekaise-chat-shell"],
                    sizing_mode="stretch_both",
                ),
                sizing_mode="stretch_width",
            ),
            css_classes=["nekaise-shell"],
            sizing_mode="stretch_both",
        )

        ontology_page = pn.Column(
            pn.pane.HTML("<div class='nekaise-card'><h3>ONTOLOGY.ttl</h3><p>Review the structured graph projection that anchors every grounded answer.</p></div>"),
            pn.Row(self.ontology_graph, self.ontology_file, sizing_mode="stretch_both"),
            css_classes=["nekaise-shell"],
            sizing_mode="stretch_both",
        )

        memory_page = pn.Column(
            pn.pane.HTML("<div class='nekaise-card'><h3>MEMORY.md</h3><p>Approved results can be promoted into long-term building memory for later retrieval and distillation.</p></div>"),
            self.memory_status,
            pn.Row(
                self.memory_file,
                pn.Column(pn.pane.HTML("<div class='nekaise-card'><h3>Curated trace log</h3></div>"), self.curated_table, sizing_mode="stretch_both"),
                sizing_mode="stretch_both",
            ),
            css_classes=["nekaise-shell"],
            sizing_mode="stretch_both",
        )

        return pn.Tabs(
            ("Home", home_page),
            ("Nekaise Agent", agent_page),
            ("ONTOLOGY.ttl", ontology_page),
            ("MEMORY.md", memory_page),
            dynamic=True,
            active=active,
            sizing_mode="stretch_both",
            css_classes=["dashboard-tabs", "nekaise-tabs"],
        )

    def build_embedded(self, *, active: int = 1) -> pn.Column:
        return pn.Column(
            self.build_tabs(active=active),
            sizing_mode="stretch_both",
        )

    def build(self) -> pn.template.FastListTemplate:
        return pn.template.FastListTemplate(
            title="Building Energy Knowledge Workbench",
            header_background="#ffffff",
            header_color="#202124",
            accent_base_color="#1a73e8",
            sidebar=[],
            main=[self.build_tabs(active=1)],
            main_max_width="1600px",
        )

    def _build_status_chart(self, status: dict[str, Any]) -> go.Figure:
        figure = go.Figure(
            data=[
                go.Bar(
                    x=["Documents", "Chunks", "Memory", "Curated", "Ontology docs", "Ontology meters"],
                    y=[
                        status["documents"],
                        status["chunks"],
                        status["memory_entries"],
                        status["curated_traces"],
                        status["ontology_docs"],
                        status["ontology_meters"],
                    ],
                    marker_color=["#1a73e8", "#0ea5e9", "#10b981", "#f59e0b", "#6366f1", "#ef4444"],
                )
            ]
        )
        figure.update_layout(margin=dict(l=24, r=16, t=24, b=20))
        return apply_custom_theme(figure)

    def _build_ontology_figure(self, ontology: dict[str, Any], title: str) -> go.Figure:
        building = ontology.get("building", {}) or {"building_id": title}
        documents = list((ontology.get("documents") or {}).values())
        meters = list((ontology.get("meters") or {}).values())
        kpis = list((ontology.get("kpis") or {}).values())

        nodes: list[dict[str, Any]] = [
            {
                "id": building.get("building_id", title),
                "label": building.get("building_id", title),
                "group": "building",
                "x": 0.0,
                "y": 0.0,
                "size": 34,
            }
        ]
        edges: list[tuple[int, int]] = []

        def add_ring(items: list[dict[str, Any]], group: str, radius: float, size: float, start_angle: float) -> None:
            base_index = len(nodes)
            count = max(len(items), 1)
            for idx, item in enumerate(items):
                angle = start_angle + (2 * math.pi * idx / count)
                nodes.append(
                    {
                        "id": item.get("name") or item.get("title") or f"{group}-{idx}",
                        "label": item.get("name") or item.get("title") or f"{group}-{idx}",
                        "group": group,
                        "x": radius * math.cos(angle),
                        "y": radius * math.sin(angle),
                        "size": size,
                    }
                )
                edges.append((0, base_index + idx))

        add_ring(documents, "document", 1.45, 20, -1.2)
        add_ring(meters, "meter", 2.25, 18, -0.15)
        add_ring(kpis, "kpi", 3.05, 15, 0.65)

        colors = {
            "building": "#1a73e8",
            "document": "#0ea5e9",
            "meter": "#10b981",
            "kpi": "#f59e0b",
        }
        edge_x: list[float] = []
        edge_y: list[float] = []
        for source_idx, target_idx in edges:
            edge_x.extend([nodes[source_idx]["x"], nodes[target_idx]["x"], None])
            edge_y.extend([nodes[source_idx]["y"], nodes[target_idx]["y"], None])

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="rgba(148, 163, 184, 0.5)", width=1.4),
                hoverinfo="skip",
            )
        )
        for group in ("building", "document", "meter", "kpi"):
            group_nodes = [node for node in nodes if node["group"] == group]
            if not group_nodes:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[node["x"] for node in group_nodes],
                    y=[node["y"] for node in group_nodes],
                    mode="markers+text",
                    text=[node["label"] for node in group_nodes],
                    textposition="middle center",
                    textfont=dict(color="#202124", family="Inter, sans-serif", size=12 if group == "building" else 11),
                    marker=dict(
                        size=[node["size"] for node in group_nodes],
                        color=colors[group],
                        opacity=0.88,
                        line=dict(color="rgba(255,255,255,0.75)", width=1),
                    ),
                    hovertemplate="%{text}<extra></extra>",
                    name=group,
                )
            )

        figure.update_layout(
            title=dict(
                text=f"{len(documents)} docs | {len(meters)} meters | {len(kpis)} kpis | {len(edges)} edges",
                x=0.5,
                font=dict(color="#334155", family="Inter, sans-serif", size=20),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=0, r=0, t=56, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        )
        return figure

    def _wrap_pre(self, title: str, text: str) -> str:
        safe = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<div class='nekaise-file-shell'><div class='nekaise-file-title'>{title}</div><pre>{safe}</pre></div>"

    def _safe_read(self, path: Path, *, fallback: str) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return fallback


def create_dashboard():
    return NekaiseWorkbenchDashboard().build()


def create_embedded_dashboard(*, active: int = 1) -> pn.Column:
    return NekaiseWorkbenchDashboard().build_embedded(active=active)
