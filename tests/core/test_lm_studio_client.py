import unittest
from unittest.mock import patch

from src.knowledge_analysis import LocalLLMAdapter
from src.lm_studio_client import (
    _extract_text_tool_calls,
    _extract_years,
    _fallback_answer_from_tool_result,
    _infer_energy_metric,
    _infer_energy_tool,
    _should_prefetch_energy_records,
    default_local_llm_base_url,
    normalize_lm_studio_base_url,
)


class TestLMStudioBaseUrl(unittest.TestCase):
    def test_normalize_base_url_adds_v1_suffix(self):
        self.assertEqual(
            normalize_lm_studio_base_url("http://127.0.0.1:1234"),
            "http://127.0.0.1:1234/v1",
        )

    def test_normalize_base_url_accepts_chat_completion_endpoint(self):
        self.assertEqual(
            normalize_lm_studio_base_url("http://localhost:1234/v1/chat/completions"),
            "http://localhost:1234/v1",
        )

    def test_local_llm_adapter_defaults_to_localhost(self):
        with patch.dict("os.environ", {}, clear=True):
            adapter = LocalLLMAdapter()
        self.assertEqual(adapter.base_url, "http://127.0.0.1:8088/v1")

    def test_default_local_llm_prefers_gemma(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(default_local_llm_base_url(), "http://127.0.0.1:8088/v1")

    def test_default_local_llm_allows_explicit_lmstudio_provider(self):
        with patch.dict(
            "os.environ",
            {"ENERGY_LOCAL_LLM_PROVIDER": "lmstudio", "ENERGY_LOCAL_LLM_BASE_URL": "http://localhost:1234"},
            clear=True,
        ):
            self.assertEqual(default_local_llm_base_url(), "http://localhost:1234")

    def test_local_llm_adapter_normalizes_env_origin(self):
        with patch.dict("os.environ", {"ENERGY_LOCAL_LLM_BASE_URL": "http://localhost:1234"}, clear=True):
            adapter = LocalLLMAdapter()
        self.assertEqual(adapter.base_url, "http://localhost:1234/v1")

    def test_loose_text_tool_call_is_parsed(self):
        calls = _extract_text_tool_calls(
            "<tool_call>call'query_energy_records{year:2018,campus:'NTU'}<tool_call>",
            {"query_energy_records"},
        )
        self.assertEqual(calls[0]["name"], "query_energy_records")
        self.assertEqual(calls[0]["arguments"]["years"], [2018])
        self.assertEqual(calls[0]["arguments"]["campus"], "NTU")

    def test_llama_style_text_tool_call_is_parsed(self):
        calls = _extract_text_tool_calls(
            "<|tool_call>call:compare_building_trends{end_year:2017,start_year:2016,year:2016}<tool_call|>",
            {"compare_building_trends"},
        )
        self.assertEqual(calls[0]["name"], "compare_building_trends")
        self.assertEqual(calls[0]["arguments"]["years"], [2016, 2017])

    def test_total_consumption_prefetch_routes_to_records(self):
        prompt = "我想比較2018跟2020得總體耗電量"
        self.assertTrue(_should_prefetch_energy_records(prompt))
        self.assertEqual(_extract_years(prompt), [2018, 2020])
        self.assertEqual(_infer_energy_tool(prompt), "compare_energy_usage")
        self.assertEqual(_infer_energy_metric(prompt), "annual_kwh")

    def test_ntu_year_comparison_routes_to_campus_records(self):
        prompt = "幫我比較2017跟2016的台大電力使用情況"
        self.assertTrue(_should_prefetch_energy_records(prompt))
        self.assertEqual(_extract_years(prompt), [2016, 2017])
        self.assertEqual(_infer_energy_tool(prompt), "compare_energy_usage")
        self.assertEqual(_infer_energy_metric(prompt), "annual_kwh")

    def test_fallback_answer_sums_total_consumption_by_year(self):
        answer = _fallback_answer_from_tool_result(
            "query_energy_records",
            '{"rows":[{"year":2018,"annual_kwh":100},{"year":2018,"annual_kwh":50},{"year":2020,"annual_kwh":120,"mean_kw":1}]}',
            "比較2018跟2020總體耗電量",
        )
        self.assertIn("2018 年約 150", answer)
        self.assertIn("2020 年約 120", answer)

    def test_fallback_answer_uses_mean_kw_when_annual_kwh_missing(self):
        answer = _fallback_answer_from_tool_result(
            "query_energy_records",
            '{"rows":[{"year":2018,"annual_kwh":100},{"year":2020,"annual_kwh":null,"mean_kw":2}]}',
            "比較2018跟2020總體耗電量",
        )
        self.assertIn("mean_kw * 8760", answer)
        self.assertIn("2020 年約 17,520", answer)

    def test_fallback_answer_warns_when_requested_year_is_zero(self):
        answer = _fallback_answer_from_tool_result(
            "query_energy_records",
            '{"rows":[{"year":2018,"annual_kwh":100},{"year":2020,"annual_kwh":0,"mean_kw":0}]}',
            "比較2018跟2020總體耗電量",
        )
        self.assertIn("資料品質不足", answer)
        self.assertIn("不能把 0 解讀成實際耗電歸零", answer)


if __name__ == "__main__":
    unittest.main()
