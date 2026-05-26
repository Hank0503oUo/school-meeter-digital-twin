# -*- coding: utf-8 -*-
import json
import tempfile
import unittest
from pathlib import Path

from src.knowledge_analysis import CloudFirstAnalysisService
from src.knowledge_base import KnowledgeWorkbench
from src.knowledge_models import AnalysisRequest, CuratedTraceRecord


class TestKnowledgeWorkbench(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "kb"
        self.workbench = KnowledgeWorkbench(root=self.root)

        self.doc = self.workbench.ingest_upload(
            filename="ahu_notes.md",
            content=(
                "# AHU Notes\n\n"
                "The supply temperature setpoint is 22 C.\n\n"
                "Weekly energy review recommends reducing lighting after 18:00."
            ).encode("utf-8"),
            building_id="E1-Building",
            title="AHU Notes",
            tags="ahu, setpoint",
        )
        self.csv = self.workbench.ingest_upload(
            filename="meter.csv",
            content=(
                "timestamp,kw,kwh,eui\n"
                "2024-01-01 00:00,100,2400,180\n"
                "2024-01-01 01:00,120,2450,181\n"
                "2024-01-01 02:00,110,2425,179\n"
            ).encode("utf-8"),
            building_id="E1-Building",
            title="Meter History",
            tags="meter, baseline",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_ingest_builds_index_and_opennekaise_style_files(self):
        hits = self.workbench.search_chunks(query="setpoint 22", building_id="e1-building")
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0]["doc_id"], self.doc.doc_id)

        group_paths = self.workbench.get_group_paths("e1-building")
        self.assertTrue(group_paths["ontology_ttl"].exists())
        self.assertTrue(group_paths["memory_md"].exists())
        ontology_text = group_paths["ontology_ttl"].read_text(encoding="utf-8")
        self.assertIn("brick:Building", ontology_text)
        self.assertIn(self.csv.doc_id, ontology_text)

    def test_cloud_first_service_falls_back_and_uses_algorithm_tools(self):
        class DisabledCloudAdapter:
            def configured(self) -> bool:
                return False

        class DisabledLocalMCPAdapter:
            def maybe_analyze(self, request, context):
                return None

        class DisabledLocalLLMAdapter:
            def available(self) -> bool:
                return False

        service = CloudFirstAnalysisService(
            self.workbench,
            cloud_adapter=DisabledCloudAdapter(),
            local_mcp_adapter=DisabledLocalMCPAdapter(),
            local_llm_adapter=DisabledLocalLLMAdapter(),
        )
        request = AnalysisRequest(
            building_id="e1-building",
            task_type="energy_summary",
            user_query="Summarize the energy situation and suggest what to review next.",
            selected_docs=[self.doc.doc_id],
            selected_csvs=[self.csv.doc_id],
        )
        result = service.analyze(request)
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.adapter_name, "heuristic")
        self.assertIn("counterfactual_preview", result.extracted_json)
        self.assertIsNotNone(result.extracted_json["counterfactual_preview"])
        self.assertGreater(len(result.tool_trace), 0)

    def test_curated_trace_can_also_update_memory(self):
        request = AnalysisRequest(
            building_id="e1-building",
            task_type="qa",
            user_query="What is the supply temperature setpoint?",
            selected_docs=[self.doc.doc_id],
            selected_csvs=[],
        )
        result = {
            "answer_markdown": "The supply temperature setpoint is 22 C based on AHU Notes.",
            "extracted_json": {"setpoint_c": 22},
            "cited_chunks": [],
            "confidence": 0.81,
            "followups": ["Verify against the latest commissioning sheet."],
            "adapter_name": "heuristic",
        }
        trace = CuratedTraceRecord(
            trace_id="trace_001",
            request=request.to_dict(),
            result=result,
            reviewer_notes="Clear citation and directly useful for training.",
            approved=True,
        )
        self.workbench.save_curated_trace(trace, save_to_memory=True, memory_title="AHU setpoint confirmed")

        curated_rows = self.workbench.list_curated_traces()
        self.assertEqual(len(curated_rows), 1)
        self.assertEqual(curated_rows[0]["trace_id"], "trace_001")

        memory_text = self.workbench.get_group_paths("e1-building")["memory_md"].read_text(encoding="utf-8")
        self.assertIn("AHU setpoint confirmed", memory_text)
        self.assertIn("22 C", memory_text)


if __name__ == "__main__":
    unittest.main()
