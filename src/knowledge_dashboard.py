from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import panel as pn

from src.knowledge_analysis import CloudFirstAnalysisService, CloudModelAdapter, LocalLLMAdapter
from src.knowledge_base import KnowledgeWorkbench
from src.knowledge_models import AnalysisRequest, CuratedTraceRecord, new_id

pn.extension("tabulator", sizing_mode="stretch_width")


def _markdown_status(label: str, ok: bool, detail: str) -> str:
    state = "Ready" if ok else "Fallback"
    return f"**{label}:** {state}\n\n{detail}"


class KnowledgeWorkbenchDashboard:
    def __init__(self) -> None:
        self.workbench = KnowledgeWorkbench()
        self.analysis = CloudFirstAnalysisService(self.workbench)
        self.cloud_adapter = self.analysis.cloud_adapter
        self.local_llm_adapter = self.analysis.local_llm_adapter
        self.last_results: dict[str, tuple[AnalysisRequest, dict]] = {}
        self._context_groups: list[dict[str, pn.widgets.Widget]] = []

        self.upload_building = pn.widgets.TextInput(name="Building / Group", value="general")
        self.upload_title = pn.widgets.TextInput(name="Title", placeholder="Optional display title")
        self.upload_tags = pn.widgets.TextInput(name="Tags", placeholder="energy, csv, baseline")
        self.file_input = pn.widgets.FileInput(name="Upload PDF / Markdown / CSV", accept=".pdf,.md,.markdown,.txt,.csv")
        self.upload_button = pn.widgets.Button(name="Upload File", button_type="primary")
        self.rebuild_button = pn.widgets.Button(name="Rebuild Index", button_type="default")
        self.upload_status = pn.pane.Markdown("", sizing_mode="stretch_width")

        self.status_pane = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.cloud_status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.documents_table = pn.widgets.Tabulator(pd.DataFrame(), height=250, sizing_mode="stretch_width", disabled=True)
        self.memory_table = pn.widgets.Tabulator(pd.DataFrame(), height=220, sizing_mode="stretch_width", disabled=True)
        self.curated_table = pn.widgets.Tabulator(pd.DataFrame(), height=220, sizing_mode="stretch_width", disabled=True)
        self.ontology_preview = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.memory_preview = pn.pane.Markdown("", sizing_mode="stretch_width")

        self.auto_building = pn.widgets.Select(name="Building", options=self.workbench.list_buildings(), value=self.workbench.list_buildings()[0])
        self.auto_focus = pn.widgets.TextAreaInput(
            name="Auto Review Goal",
            value="Summarize this building's uploaded documents and CSVs, extract key KPIs, and produce a concise report for later distillation.",
            height=120,
        )
        self.auto_save_memory = pn.widgets.Checkbox(name="Save approved result to MEMORY.md", value=True)
        self.auto_button = pn.widgets.Button(name="Run Auto Review", button_type="primary")
        self.auto_status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.auto_answer = pn.pane.Markdown("Choose a building and click `Run Auto Review`.", sizing_mode="stretch_width")
        self.auto_json = pn.pane.JSON({}, depth=3, sizing_mode="stretch_width")
        self.auto_citations = pn.widgets.Tabulator(pd.DataFrame(), height=220, sizing_mode="stretch_width", disabled=True)

        self.qa = self._make_task_widgets("qa", "Question")
        self.extract = self._make_task_widgets("structured_extraction", "Extraction Goal")
        self.report = self._make_task_widgets("report_generation", "Report Focus")

        self.upload_button.on_click(self._handle_upload)
        self.rebuild_button.on_click(self._handle_rebuild)
        self.auto_button.on_click(self._run_auto_review)
        self.qa["run"].on_click(lambda event: self._run_task("qa", self.qa))
        self.extract["run"].on_click(lambda event: self._run_task("structured_extraction", self.extract))
        self.report["run"].on_click(lambda event: self._run_task("report_generation", self.report))
        self.qa["save"].on_click(lambda event: self._save_trace("qa", self.qa))
        self.extract["save"].on_click(lambda event: self._save_trace("structured_extraction", self.extract))
        self.report["save"].on_click(lambda event: self._save_trace("report_generation", self.report))
        self.report["export"].on_click(lambda event: self._export_report(self.report))

        self.refresh_all()

    def _make_task_widgets(self, task_type: str, query_label: str) -> dict[str, pn.widgets.Widget | pn.pane.PaneBase]:
        building = pn.widgets.Select(name="Building", options=self.workbench.list_buildings(), value=self.workbench.list_buildings()[0])
        documents = pn.widgets.MultiChoice(name="Documents", options={}, solid=False)
        csvs = pn.widgets.MultiChoice(name="CSV Files", options={}, solid=False)
        query = pn.widgets.TextAreaInput(name=query_label, placeholder="Describe the question or objective", height=120)
        notes = pn.widgets.TextAreaInput(name="Reviewer Notes", placeholder="Why is this output good enough for distillation?", height=90)
        save_memory = pn.widgets.Checkbox(name="Also save to MEMORY.md", value=(task_type == "report_generation"))
        run = pn.widgets.Button(name="Run", button_type="primary")
        save = pn.widgets.Button(name="Save Good Result", button_type="success")
        export = pn.widgets.Button(name="Export Report", button_type="default", visible=(task_type == "report_generation"))
        status = pn.pane.Markdown("", sizing_mode="stretch_width")
        answer = pn.pane.Markdown("Run an analysis to see the answer here.", sizing_mode="stretch_width")
        extracted = pn.pane.JSON({}, depth=3, sizing_mode="stretch_width")
        citations = pn.widgets.Tabulator(pd.DataFrame(), height=220, sizing_mode="stretch_width", disabled=True)
        followups = pn.pane.Markdown("", sizing_mode="stretch_width")

        group = {
            "building": building,
            "documents": documents,
            "csvs": csvs,
            "query": query,
            "notes": notes,
            "save_memory": save_memory,
            "run": run,
            "save": save,
            "export": export,
            "status": status,
            "answer": answer,
            "extracted": extracted,
            "citations": citations,
            "followups": followups,
        }
        self._context_groups.append(group)
        building.param.watch(lambda event, widgets=group: self._refresh_group_options(widgets), "value")
        return group

    def refresh_all(self) -> None:
        self._refresh_status()
        self._refresh_tables()
        for group in self._context_groups:
            self._refresh_group_options(group)

    def _refresh_status(self) -> None:
        status = self.workbench.status()
        self.status_pane.object = "\n".join(
            [
                "### Knowledge Base Status",
                "",
                f"- Buildings: {status['buildings']}",
                f"- Documents: {status['documents']}",
                f"- CSV files: {status['csv_files']}",
                f"- Chunks: {status['chunks']}",
                f"- Memory entries: {status['memory_entries']}",
                f"- Curated traces: {status['curated_traces']}",
                f"- Ontology docs: {status['ontology_docs']}",
                f"- Ontology meters: {status['ontology_meters']}",
                f"- Ontology KPIs: {status['ontology_kpis']}",
            ]
        )
        configured = self.cloud_adapter.configured()
        detail = (
            "Cloud inference is configured via ENERGY_LLM_API_URL / ENERGY_LLM_API_KEY / ENERGY_LLM_MODEL."
            if configured
            else "Cloud inference is not configured yet. The app will keep working with heuristic fallback so you can still curate traces."
        )
        self.cloud_status.object = _markdown_status("Cloud Adapter", configured, detail)

    def _refresh_tables(self) -> None:
        documents = self.workbench.list_documents()
        memory = self.workbench.list_memory()
        curated = self.workbench.list_curated_traces()
        docs_df = pd.DataFrame(
            [
                {
                    "doc_id": item.doc_id,
                    "building_id": item.building_id,
                    "source_type": item.source_type,
                    "title": item.title,
                    "tags": ", ".join(item.tags),
                }
                for item in documents
            ]
        )
        memory_df = pd.DataFrame(
            [
                {
                    "building_id": item.building_id,
                    "title": item.title,
                    "created_at": item.created_at,
                    "tags": ", ".join(item.tags),
                }
                for item in memory
            ]
        )
        curated_df = pd.DataFrame(
            [
                {
                    "trace_id": item.get("trace_id", ""),
                    "building_id": item.get("request", {}).get("building_id", ""),
                    "task_type": item.get("request", {}).get("task_type", ""),
                    "saved_at": item.get("saved_at", ""),
                    "approved": bool(item.get("approved", True)),
                }
                for item in curated
            ]
        )
        self.documents_table.value = docs_df
        self.memory_table.value = memory_df
        self.curated_table.value = curated_df

        default_building = self.workbench.list_buildings()[0]
        paths = self.workbench.get_group_paths(default_building)
        self.ontology_preview.object = self._safe_read(paths["ontology_ttl"], fallback="Ontology file will appear after documents are indexed.")
        self.memory_preview.object = self._safe_read(paths["memory_md"], fallback="Memory file will appear after saving confirmed findings.")

    def _refresh_group_options(self, group: dict[str, pn.widgets.Widget | pn.pane.PaneBase]) -> None:
        buildings = self.workbench.list_buildings()
        building_widget = group["building"]
        if getattr(building_widget, "value", None) not in buildings:
            building_widget.value = buildings[0]
        building_widget.options = buildings
        selected_building = str(building_widget.value)
        docs = self.workbench.list_documents(selected_building, source_type="markdown") + self.workbench.list_documents(selected_building, source_type="text") + self.workbench.list_documents(selected_building, source_type="pdf")
        csvs = self.workbench.list_documents(selected_building, source_type="csv")
        doc_options = {f"{item.title} [{item.source_type}]": item.doc_id for item in docs}
        csv_options = {f"{item.title} [csv]": item.doc_id for item in csvs}
        group["documents"].options = doc_options
        group["csvs"].options = csv_options
        group["documents"].value = [value for value in group["documents"].value if value in doc_options.values()]
        group["csvs"].value = [value for value in group["csvs"].value if value in csv_options.values()]

        paths = self.workbench.get_group_paths(selected_building)
        self.ontology_preview.object = self._safe_read(paths["ontology_ttl"], fallback="Ontology file will appear after documents are indexed.")
        self.memory_preview.object = self._safe_read(paths["memory_md"], fallback="Memory file will appear after saving confirmed findings.")
        if self.auto_building.value not in buildings:
            self.auto_building.value = buildings[0]
        self.auto_building.options = buildings

    def _handle_upload(self, event=None) -> None:
        if not self.file_input.filename or self.file_input.value is None:
            self.upload_status.object = "Please select a file first."
            return
        try:
            record = self.workbench.ingest_upload(
                filename=str(self.file_input.filename),
                content=bytes(self.file_input.value),
                building_id=self.upload_building.value,
                title=self.upload_title.value,
                tags=self.upload_tags.value,
            )
        except Exception as exc:
            self.upload_status.object = f"Upload failed: `{exc}`"
            return

        self.upload_status.object = (
            f"Uploaded `{record.title}` to building `{record.building_id}`.\n\n"
            f"- doc_id: `{record.doc_id}`\n"
            f"- parsed markdown: `{record.parsed_md_path}`"
        )
        self.upload_title.value = ""
        self.upload_tags.value = ""
        self.refresh_all()

    def _handle_rebuild(self, event=None) -> None:
        count = self.workbench.rebuild_index()
        self.upload_status.object = f"Rebuilt index successfully. Total chunks: `{count}`"
        self.refresh_all()

    def _all_doc_ids(self, building_id: str) -> list[str]:
        docs = []
        for source_type in ("markdown", "text", "pdf"):
            docs.extend(self.workbench.list_documents(building_id, source_type=source_type))
        return [item.doc_id for item in docs]

    def _all_csv_ids(self, building_id: str) -> list[str]:
        return [item.doc_id for item in self.workbench.list_documents(building_id, source_type="csv")]

    def _populate_widgets_with_result(
        self,
        widgets: dict[str, pn.widgets.Widget | pn.pane.PaneBase],
        request: AnalysisRequest,
        result: dict,
    ) -> None:
        widgets["building"].value = request.building_id
        self._refresh_group_options(widgets)
        widgets["documents"].value = list(request.selected_docs)
        widgets["csvs"].value = list(request.selected_csvs)
        widgets["query"].value = request.user_query
        widgets["answer"].object = str(result.get("answer_markdown", ""))
        widgets["extracted"].object = dict(result.get("extracted_json", {}))
        widgets["citations"].value = pd.DataFrame(
            [
                {
                    "title": chunk.get("title", ""),
                    "source_type": chunk.get("source_type", ""),
                    "score": chunk.get("score", ""),
                    "excerpt": chunk.get("excerpt", ""),
                }
                for chunk in result.get("cited_chunks", [])
            ]
        )
        followups = result.get("followups", []) or []
        widgets["followups"].object = "### Follow-ups\n\n" + (
            "\n".join(f"- {item}" for item in followups) if followups else "- No follow-up suggestions."
        )
        status_lines = [
            f"- adapter: `{result.get('adapter_name', '')}`",
            f"- fallback: `{result.get('used_fallback', False)}`",
            f"- confidence: `{float(result.get('confidence', 0.0) or 0.0):.2f}`",
        ]
        for warning in result.get("warnings", []) or []:
            status_lines.append(f"- warning: {warning}")
        widgets["status"].object = "### Run Status\n\n" + "\n".join(status_lines)

    def _run_auto_review(self, event=None) -> None:
        building_id = str(self.auto_building.value)
        selected_docs = self._all_doc_ids(building_id)
        selected_csvs = self._all_csv_ids(building_id)
        if not selected_docs and not selected_csvs:
            self.auto_status.object = "### Auto Review\n\nThis building has no uploaded documents or CSVs yet."
            return

        request = AnalysisRequest(
            building_id=building_id,
            task_type="report_generation",
            user_query=str(self.auto_focus.value).strip(),
            selected_docs=selected_docs,
            selected_csvs=selected_csvs,
        )
        result = self.analysis.analyze(request)
        self.last_results["report_generation"] = (request, result.to_dict())
        self._populate_widgets_with_result(self.report, request, result.to_dict())

        trace = CuratedTraceRecord(
            trace_id=new_id("trace"),
            request=request.to_dict(),
            result=result.to_dict(),
            reviewer_notes="Auto review generated from all currently indexed building inputs.",
            approved=True,
        )
        target = self.workbench.save_curated_trace(
            trace,
            save_to_memory=bool(self.auto_save_memory.value),
            memory_title="auto review approved result",
        )
        report_path = self.workbench.export_report(
            request.building_id,
            request.user_query or "auto_review",
            result.answer_markdown,
        )

        self.auto_answer.object = result.answer_markdown
        self.auto_json.object = result.extracted_json
        self.auto_citations.value = pd.DataFrame(
            [
                {
                    "title": chunk.get("title", ""),
                    "source_type": chunk.get("source_type", ""),
                    "score": chunk.get("score", ""),
                    "excerpt": chunk.get("excerpt", ""),
                }
                for chunk in result.cited_chunks
            ]
        )
        self.auto_status.object = "\n".join(
            [
                "### Auto Review",
                "",
                f"- building: `{building_id}`",
                f"- docs: `{len(selected_docs)}`",
                f"- csvs: `{len(selected_csvs)}`",
                f"- adapter: `{result.adapter_name}`",
                f"- fallback: `{result.used_fallback}`",
                f"- curated trace: `{trace.trace_id}`",
                f"- trace file: `{target}`",
                f"- report: `{report_path}`",
                f"- save_to_memory: `{bool(self.auto_save_memory.value)}`",
            ]
        )
        self.refresh_all()

    def _run_task(self, task_type: str, widgets: dict[str, pn.widgets.Widget | pn.pane.PaneBase]) -> None:
        request = AnalysisRequest(
            building_id=str(widgets["building"].value),
            task_type=task_type,  # type: ignore[arg-type]
            user_query=str(widgets["query"].value).strip(),
            selected_docs=list(widgets["documents"].value),
            selected_csvs=list(widgets["csvs"].value),
        )
        result = self.analysis.analyze(request)
        widgets["answer"].object = result.answer_markdown
        widgets["extracted"].object = result.extracted_json
        widgets["citations"].value = pd.DataFrame(
            [
                {
                    "title": chunk.get("title", ""),
                    "source_type": chunk.get("source_type", ""),
                    "score": chunk.get("score", ""),
                    "excerpt": chunk.get("excerpt", ""),
                }
                for chunk in result.cited_chunks
            ]
        )
        followups = "\n".join(f"- {item}" for item in result.followups) if result.followups else "- No follow-up suggestions."
        widgets["followups"].object = f"### Follow-ups\n\n{followups}"
        status_lines = [
            f"- adapter: `{result.adapter_name}`",
            f"- fallback: `{result.used_fallback}`",
            f"- confidence: `{result.confidence:.2f}`",
        ]
        for warning in result.warnings:
            status_lines.append(f"- warning: {warning}")
        widgets["status"].object = "### Run Status\n\n" + "\n".join(status_lines)
        self.last_results[task_type] = (request, result.to_dict())

    def _save_trace(self, task_type: str, widgets: dict[str, pn.widgets.Widget | pn.pane.PaneBase]) -> None:
        saved = self.last_results.get(task_type)
        if saved is None:
            widgets["status"].object = "### Run Status\n\nRun the analysis first."
            return
        request, result = saved
        trace = CuratedTraceRecord(
            trace_id=new_id("trace"),
            request=request.to_dict(),
            result=result,
            reviewer_notes=str(widgets["notes"].value).strip(),
            approved=True,
        )
        target = self.workbench.save_curated_trace(
            trace,
            save_to_memory=bool(widgets["save_memory"].value),
            memory_title=f"{request.task_type} approved result",
        )
        widgets["status"].object = (
            "### Run Status\n\n"
            f"Saved curated trace to `{target}`.\n\n"
            f"- trace_id: `{trace.trace_id}`\n"
            f"- save_to_memory: `{bool(widgets['save_memory'].value)}`"
        )
        widgets["notes"].value = ""
        self.refresh_all()

    def _export_report(self, widgets: dict[str, pn.widgets.Widget | pn.pane.PaneBase]) -> None:
        saved = self.last_results.get("report_generation")
        if saved is None:
            widgets["status"].object = "### Run Status\n\nGenerate a report first."
            return
        request, result = saved
        report_path = self.workbench.export_report(
            request.building_id,
            request.user_query or "building_report",
            str(result.get("answer_markdown", "")),
        )
        widgets["status"].object = f"### Run Status\n\nExported report to `{report_path}`"

    def build(self) -> pn.template.FastListTemplate:
        upload_page = pn.Column(
            "## 1. Document Upload & Knowledge Base Status",
            pn.Row(self.upload_building, self.upload_title, self.upload_tags),
            pn.Row(self.file_input, self.upload_button, self.rebuild_button),
            self.upload_status,
            pn.layout.Divider(),
            self.status_pane,
            pn.Row(
                pn.Column("### Documents", self.documents_table, sizing_mode="stretch_both"),
                pn.Column("### Curated Traces", self.curated_table, sizing_mode="stretch_both"),
                sizing_mode="stretch_both",
            ),
            pn.Row(
                pn.Column("### ONTOLOGY.ttl", self.ontology_preview, sizing_mode="stretch_both"),
                pn.Column("### MEMORY.md", self.memory_preview, sizing_mode="stretch_both"),
                sizing_mode="stretch_both",
            ),
            sizing_mode="stretch_both",
        )

        auto_page = pn.Column(
            "## 2. Auto Review",
            "Choose a building and let the system automatically gather all indexed documents and CSVs, generate a report, save a curated trace, and optionally update MEMORY.md.",
            pn.Row(self.auto_building, self.auto_button, self.auto_save_memory, sizing_mode="stretch_width"),
            self.auto_focus,
            self.auto_status,
            pn.Row(
                pn.Column("### Auto Review Answer", self.auto_answer, sizing_mode="stretch_both"),
                pn.Column("### Auto Review JSON", self.auto_json, sizing_mode="stretch_both"),
                sizing_mode="stretch_both",
            ),
            pn.Column("### Auto Review Citations", self.auto_citations, sizing_mode="stretch_both"),
            sizing_mode="stretch_both",
        )

        qa_page = self._task_page(
            "3. Building / Equipment Q&A",
            self.qa,
            "Ask a grounded question against uploaded documents and CSVs. Good answers can be saved into the distillation set.",
        )
        extraction_page = self._task_page(
            "4. Structured Energy Extraction",
            self.extract,
            "Turn building documents into JSON that can later be used for Colab distillation or downstream automation.",
        )
        report_page = self._task_page(
            "5. Analysis Summary & Report",
            self.report,
            "Generate a concise report, then export or promote the approved result into MEMORY.md and the curated trace dataset.",
        )

        template = pn.template.FastListTemplate(
            title="Building Energy Knowledge Workbench",
            header_background="#ffffff",
            header_color="#202124",
            accent_base_color="#1a73e8",
            sidebar=[
                pn.pane.Markdown(
                    "\n".join(
                        [
                            "## Deployment Path",
                            "",
                            "1. Use a cloud-first MCP-style knowledge agent now.",
                            "2. Save approved outputs into a curated trace dataset.",
                            "3. Train in Colab later.",
                            "4. Distill to a local deployable model afterward.",
                        ]
                    )
                ),
                self.cloud_status,
                pn.layout.Divider(),
                pn.Column("### Latest Memory Entries", self.memory_table, sizing_mode="stretch_both"),
            ],
            main=[
                pn.Tabs(
                    ("Upload & Status", upload_page),
                    ("Auto Review", auto_page),
                    ("Q&A", qa_page),
                    ("Structured Extraction", extraction_page),
                    ("Report", report_page),
                    dynamic=True,
                    active=0,
                    sizing_mode="stretch_both",
                )
            ],
            sidebar_width=360,
        )
        return template

    def _task_page(self, title: str, widgets: dict[str, pn.widgets.Widget | pn.pane.PaneBase], intro: str) -> pn.Column:
        controls = pn.Row(
            widgets["building"],
            widgets["documents"],
            widgets["csvs"],
            sizing_mode="stretch_width",
        )
        actions = pn.Row(
            widgets["run"],
            widgets["save"],
            widgets["export"],
            widgets["save_memory"],
            sizing_mode="stretch_width",
        )
        return pn.Column(
            f"## {title}",
            intro,
            controls,
            widgets["query"],
            widgets["notes"],
            actions,
            widgets["status"],
            pn.Row(
                pn.Column("### Answer", widgets["answer"], sizing_mode="stretch_both"),
                pn.Column("### Extracted JSON", widgets["extracted"], sizing_mode="stretch_both"),
                sizing_mode="stretch_both",
            ),
            pn.Row(
                pn.Column("### Cited Chunks", widgets["citations"], sizing_mode="stretch_both"),
                pn.Column(widgets["followups"], sizing_mode="stretch_both"),
                sizing_mode="stretch_both",
            ),
            sizing_mode="stretch_both",
        )

    def _safe_read(self, path: Path, *, fallback: str) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return fallback


def create_dashboard():
    return KnowledgeWorkbenchDashboard().build()
