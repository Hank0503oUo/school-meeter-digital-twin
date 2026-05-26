from __future__ import annotations

import asyncio

import pytest

from src.demo_mcp_server import build_server
from src.meter_screenshot_analysis import analyze_meter_screenshot_impl

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def test_analyze_meter_screenshot_missing_file_returns_error():
    result = analyze_meter_screenshot_impl("nonexistent.png")

    assert result["status"] == "error"
    assert result["warnings"]


@pytest.mark.skipif(not PIL_AVAILABLE, reason="Pillow not installed")
def test_analyze_meter_screenshot_reads_image_metadata_if_pillow_available(tmp_path):
    image_path = tmp_path / "test.png"
    Image.new("RGB", (100, 200), color="red").save(image_path)

    result = analyze_meter_screenshot_impl(str(image_path), prefer_ocr=False)

    assert result["status"] == "ok"
    assert result["width"] == 100
    assert result["height"] == 200


@pytest.mark.skipif(not PIL_AVAILABLE, reason="Pillow not installed")
def test_analyze_meter_screenshot_gracefully_handles_no_ocr(tmp_path):
    image_path = tmp_path / "test.png"
    Image.new("RGB", (100, 200), color="blue").save(image_path)

    result = analyze_meter_screenshot_impl(str(image_path), prefer_ocr=True)

    assert result["status"] in {"ok", "degraded"}
    assert isinstance(result["warnings"], list)


def test_demo_mcp_server_exposes_analyze_meter_screenshot_tool():
    server = build_server()
    tools = asyncio.run(server.list_tools())
    tool_names = {tool.name for tool in tools}

    assert "analyze_meter_screenshot" in tool_names
