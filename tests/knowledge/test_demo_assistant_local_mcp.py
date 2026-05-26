# -*- coding: utf-8 -*-
import unittest

from src.demo_assistant import CampusAssistantService, CampusAssistantSnapshot
from src.rtem_codex_bridge import local_mcp_available


@unittest.skipUnless(local_mcp_available(), "Local RTEM MCP backend is not available in this workspace.")
class TestCampusAssistantLocalMCP(unittest.TestCase):
    def setUp(self):
        self.service = CampusAssistantService()
        self.snapshot = CampusAssistantSnapshot(
            building_id="AT3035",
            building_name="國青大樓",
            year=2020,
            meter_name="",
            source="inferred",
            metrics={"uid": "AT3035", "mean_kw": 121.411, "annual_kwh": 1063558.6, "eui": 81.598},
        )

    def test_force_local_mcp_runs_supported_command(self):
        result = self.service.analyze(
            query="dataset_statistics",
            task_type="energy_summary",
            snapshot=self.snapshot,
            force_local_mcp=True,
        )
        self.assertEqual(result.adapter_name, "local_mcp")
        self.assertFalse(result.used_fallback)
        self.assertIn("dataset_statistics", result.answer_markdown)

    def test_force_local_mcp_rejects_unmapped_question(self):
        result = self.service.analyze(
            query="請根據目前這棟樓的資料，說明可能的節能改善空間與下一步建議。",
            task_type="energy_summary",
            snapshot=self.snapshot,
            force_local_mcp=True,
        )
        self.assertEqual(result.adapter_name, "local_mcp")
        self.assertTrue(result.used_fallback)
        self.assertIn("Could Not Route", result.answer_markdown)


if __name__ == "__main__":
    unittest.main()
