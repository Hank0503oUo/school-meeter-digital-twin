# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.knowledge_analysis import CloudFirstAnalysisService
from src.knowledge_base import KnowledgeWorkbench
from src.knowledge_mcp_backend import KnowledgeMCPBackend
from src.knowledge_mcp_server import build_server


class TestKnowledgeMCPBackend(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "kb"
        self.workbench = KnowledgeWorkbench(root=self.root)
        self.doc = self.workbench.ingest_upload(
            filename="guide.md",
            content=b"# Building Guide\n\nAHU serves the lab floor.\n\nSetpoint remains 22 C.",
            building_id="lab-a",
            title="Guide",
            tags="ahu",
        )
        self.csv = self.workbench.ingest_upload(
            filename="trend.csv",
            content=b"timestamp,kw,kwh\n2024-01-01 00:00,80,1920\n2024-01-01 01:00,84,1930\n",
            building_id="lab-a",
            title="Trend",
            tags="meter",
        )
        self.backend = KnowledgeMCPBackend(
            workbench=self.workbench,
            analysis_service=CloudFirstAnalysisService(self.workbench),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_backend_can_search_and_fetch_chunk(self):
        result = self.backend.search_docs(query="setpoint", building_id="lab-a")
        self.assertEqual(result["count"], 1)
        chunk_id = result["chunks"][0]["chunk_id"]
        chunk = self.backend.fetch_chunk(chunk_id=chunk_id)
        self.assertEqual(chunk["doc_id"], self.doc.doc_id)
        self.assertIn("22 C", chunk["text"])

    def test_backend_can_run_analysis_and_save_trace(self):
        analysis = self.backend.run_analysis(
            building_id="lab-a",
            task_type="qa",
            user_query="What is the setpoint?",
            selected_docs=[self.doc.doc_id],
            selected_csvs=[self.csv.doc_id],
        )
        self.assertIn("result", analysis)
        save = self.backend.save_curated_trace(
            building_id="lab-a",
            task_type="qa",
            user_query="What is the setpoint?",
            answer_markdown="The setpoint is 22 C.",
            extracted_json={"setpoint_c": 22},
            save_to_memory=True,
        )
        self.assertIn("trace_id", save)
        self.assertEqual(len(self.workbench.list_curated_traces()), 1)


class TestKnowledgeMCPServerRegistration(unittest.TestCase):
    def test_server_registers_expected_tools_and_resources(self):
        server = build_server()
        tool_names = set(server._tool_manager._tools.keys())
        resource_uris = {str(resource.uri) for resource in server._resource_manager._resources.values()}
        template_uris = {str(template.uri_template) for template in server._resource_manager._templates.values()}

        self.assertTrue(
            {
                "search_docs",
                "fetch_chunk",
                "lookup_building_entity",
                "query_meter_or_kpi",
                "run_analysis",
                "save_curated_trace",
                "run_pvid",
                "correlate_algorithms",
            }.issubset(tool_names)
        )
        self.assertIn("knowledge://status", resource_uris)
        self.assertIn("knowledge://buildings", resource_uris)
        self.assertIn("knowledge://curated-traces", resource_uris)
        self.assertIn("knowledge://building/{building_id}/ontology", template_uris)
        self.assertIn("knowledge://building/{building_id}/memory", template_uris)


class _DummyServer:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def run(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


class TestMCPServerEntrypoint(unittest.TestCase):
    def test_main_defaults_to_stdio(self):
        import mcp_server

        server = _DummyServer()
        captured: dict[str, object] = {}

        def fake_build_server(*, host: str = "127.0.0.1", port: int = 8000):
            captured["host"] = host
            captured["port"] = port
            return server

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(mcp_server, "build_server", side_effect=fake_build_server):
                mcp_server.main()

        self.assertEqual(captured, {"host": "127.0.0.1", "port": 8765})
        self.assertEqual(server.calls, [((), {})])

    def test_main_can_switch_to_sse(self):
        import mcp_server

        server = _DummyServer()
        captured: dict[str, object] = {}

        def fake_build_server(*, host: str = "127.0.0.1", port: int = 8000):
            captured["host"] = host
            captured["port"] = port
            return server

        with patch.dict(
            "os.environ",
            {"MCP_TRANSPORT": "sse", "MCP_PORT": "8765", "MCP_HOST": "0.0.0.0"},
            clear=True,
        ):
            with patch.object(mcp_server, "build_server", side_effect=fake_build_server):
                mcp_server.main()

        self.assertEqual(captured, {"host": "0.0.0.0", "port": 8765})
        self.assertEqual(server.calls, [((), {"transport": "sse"})])


if __name__ == "__main__":
    unittest.main()
