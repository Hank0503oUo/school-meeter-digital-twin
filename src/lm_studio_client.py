from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

try:
    from src.wiki_memory import WikiMemory
except ImportError:  # allow direct script execution without the src package on path
    from wiki_memory import WikiMemory  # type: ignore[no-redef]

try:
    from src.local_gemma_runtime import gemma_autostart_enabled, resolve_local_gemma_config
except ImportError:  # allow direct script execution without the src package on path
    from local_gemma_runtime import gemma_autostart_enabled, resolve_local_gemma_config  # type: ignore[no-redef]


class LMStudioMCPError(RuntimeError):
    """Raised when the LM Studio + MCP agent loop cannot complete."""


@dataclass(slots=True)
class LMStudioMCPResponse:
    answer: str
    model: str
    turns: int
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)


def convert_mcp_to_openai_schema(tool: Any) -> dict[str, Any]:
    name = str(getattr(tool, "name", "")).strip()
    description = str(getattr(tool, "description", "") or "").strip()
    raw_schema = getattr(tool, "inputSchema", None)
    if raw_schema is None:
        raw_schema = getattr(tool, "input_schema", None)

    if not isinstance(raw_schema, dict):
        raw_schema = {}

    if raw_schema.get("type") != "object":
        raw_schema = {
            "type": "object",
            "properties": dict(raw_schema.get("properties", {})) if isinstance(raw_schema, dict) else {},
            "required": list(raw_schema.get("required", [])) if isinstance(raw_schema, dict) else [],
            "additionalProperties": True,
        }

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": raw_schema,
        },
    }


def _provider_prefers_gemma() -> bool:
    provider = os.getenv("ENERGY_LOCAL_LLM_PROVIDER", "gemma").strip().lower()
    return provider in {"", "gemma", "local_gemma"} or gemma_autostart_enabled()


def default_local_llm_base_url() -> str:
    if _provider_prefers_gemma():
        return resolve_local_gemma_config().base_url
    return os.getenv(
        "LM_STUDIO_URL",
        os.getenv("ENERGY_LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1"),
    )


DEFAULT_LM_STUDIO_BASE_URL = default_local_llm_base_url()


def _lm_studio_headers() -> dict[str, str]:
    token = (
        os.getenv("ENERGY_LOCAL_LLM_API_KEY", "").strip()
        or os.getenv("LM_API_TOKEN", "").strip()
        or os.getenv("LM_STUDIO_API_KEY", "").strip()
    )
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _command_code_headers() -> dict[str, str]:
    token = (
        os.getenv("COMMAND_CODE_API_KEY", "").strip()
        or os.getenv("OPENCODE_API_KEY", "").strip()
        or os.getenv("ENERGY_COMMAND_CODE_API_KEY", "").strip()
        or os.getenv("ENERGY_ONLINE_LLM_API_KEY", "").strip()
    )
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def normalize_lm_studio_base_url(base_url: str | None) -> str:
    value = str(base_url or "").strip()
    if not value:
        raise LMStudioMCPError("ENERGY_LOCAL_LLM_BASE_URL is empty.")

    value = value.rstrip("/")
    for suffix in ("/chat/completions", "/models"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break

    if not value.endswith("/v1"):
        value = f"{value}/v1"
    return value.rstrip("/")


def _normalize_base_url(base_url: str | None) -> str:
    if not base_url and _provider_prefers_gemma():
        return normalize_lm_studio_base_url(resolve_local_gemma_config().base_url)
    value = (
        base_url
        or os.getenv("ENERGY_LOCAL_LLM_BASE_URL")
        or os.getenv("LM_STUDIO_URL")
    )
    value = (value or "").strip()
    if not value:
        return DEFAULT_LM_STUDIO_BASE_URL
    return normalize_lm_studio_base_url(value)


def _normalize_command_code_base_url(base_url: str | None) -> str:
    value = (
        base_url
        or os.getenv("COMMAND_CODE_BASE_URL")
        or os.getenv("OPENCODE_BASE_URL")
        or os.getenv("ENERGY_COMMAND_CODE_BASE_URL")
        or "https://opencode.ai/zen/go/v1"
    )
    return normalize_lm_studio_base_url(value)


def _resolve_default_server_script(server_script: str | Path | None) -> Path:
    if server_script is None:
        return (Path(__file__).resolve().parent / "demo_mcp_server.py").resolve()
    return Path(server_script).resolve()


def _resolve_server_cwd(server_script: Path) -> Path:
    parent = server_script.parent
    if parent.name == "src":
        return parent.parent
    return parent


def _request_chat_completion(
    base_url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
    provider_label: str = "Local OpenAI-compatible",
) -> dict[str, Any]:
    endpoint = f"{base_url}/chat/completions"
    try:
        response = requests.post(
            endpoint,
            headers=headers if headers is not None else _lm_studio_headers(),
            json=payload,
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise LMStudioMCPError(f"{provider_label} request failed at {endpoint}: {exc}") from exc
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:1200] if response is not None else ""
        raise LMStudioMCPError(
            f"{provider_label} server returned HTTP {response.status_code} at {endpoint}: {body}"
        ) from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise LMStudioMCPError(
            f"{provider_label} server returned a non-JSON payload. Verify the endpoint is reachable."
        ) from exc
    if not isinstance(data, dict):
        raise LMStudioMCPError(f"{provider_label} server returned a non-JSON payload.")
    return data


def _resolve_model(
    base_url: str,
    preferred_model: str,
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
    provider_label: str = "Local",
) -> str:
    preferred_model = str(preferred_model or "").strip()
    if preferred_model:
        return preferred_model

    endpoint = f"{base_url}/models"
    try:
        response = requests.get(
            endpoint,
            headers=headers if headers is not None else _lm_studio_headers(),
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise LMStudioMCPError(f"{provider_label} model probe failed at {endpoint}: {exc}") from exc
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(models, list) or not models:
        raise LMStudioMCPError(
            "No models are available from the local /models endpoint. "
            "Start bundled Gemma or verify the OpenAI-compatible server is running."
        )

    model_id = str(models[0].get("id", "")).strip()
    if not model_id:
        raise LMStudioMCPError("LM Studio /models returned an empty model id.")
    return model_id


def _safe_load_arguments(raw_arguments: Any) -> dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        text = raw_arguments.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LMStudioMCPError(f"Tool arguments are not valid JSON: {text}") from exc
        if isinstance(parsed, dict):
            return parsed
        raise LMStudioMCPError(f"Tool arguments must decode to an object, got: {type(parsed).__name__}")
    if raw_arguments is None:
        return {}
    raise LMStudioMCPError(f"Unsupported tool argument type: {type(raw_arguments).__name__}")


def _extract_mcp_content(result: Any) -> str:
    content = getattr(result, "content", None)
    if not isinstance(content, list):
        return str(result)

    chunks: list[str] = []
    for item in content:
        text: str | None = None
        if isinstance(item, dict):
            if "text" in item:
                text = str(item.get("text", ""))
            elif "json" in item:
                text = json.dumps(item.get("json", {}), ensure_ascii=False)
            else:
                text = json.dumps(item, ensure_ascii=False)
        else:
            raw_text = getattr(item, "text", None)
            if raw_text is not None:
                text = str(raw_text)
            else:
                raw_json = getattr(item, "json", None)
                if raw_json is not None:
                    text = json.dumps(raw_json, ensure_ascii=False)
                else:
                    text = str(item)

        if text is not None:
            stripped = text.strip()
            if stripped:
                chunks.append(stripped)

    if not chunks:
        return ""
    return "\n".join(chunks)


def _should_prefetch_seasonal(prompt: str) -> bool:
    lowered = prompt.lower()
    markers = (
        "季節策略", "夏季節能", "冬季節能", "過渡季",
        "空調季節", "不同季節", "季節性", "分季節",
        "夏天空調", "冬天照明", "夏季調適", "冬季調適",
        "seasonal", "per season",
    )
    return any(marker in lowered for marker in markers)


def _should_prefetch_portfolio(prompt: str) -> bool:
    lowered = prompt.lower()
    markers = (
        "全校", "校園", "最佳化", "組合", "投資",
        "預算", "哪幾棟", "優先順序", "roi",
        "先做哪棟", "最佳組合", "資源配置",
        "portfolio", "budget", "prioritize",
    )
    has_portfolio = any(marker in lowered for marker in markers)
    has_energy = any(m in lowered for m in ("節能", "省電", "能耗", "energy", "saving"))
    return has_portfolio and has_energy
    return "\n".join(chunks)


def _memory_disabled() -> bool:
    for key in (
        "ENERGY_LOCAL_MEMORY_DISABLED",
        "ENERGY_LM_STUDIO_MEMORY_DISABLED",
        "ENERGY_WIKI_DISABLED",
    ):
        raw = os.getenv(key)
        if raw is not None:
            return raw.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _load_wiki_memory() -> WikiMemory | None:
    if _memory_disabled():
        return None
    try:
        root = os.getenv("ENERGY_LOCAL_MEMORY_ROOT") or os.getenv("ENERGY_WIKI_ROOT")
        return WikiMemory(root=root) if root else WikiMemory()
    except Exception:
        return None


def _truthy_env(name: str, default: str = "") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _local_llm_tool_enabled(tool_name: str) -> bool:
    if tool_name in {"search_harness_memory", "store_harness_memory"}:
        return _truthy_env("ENERGY_HARNESS_MEMORY_TOOLS_ENABLED")
    return True


def _should_prefetch_docs(prompt: str) -> bool:
    lowered = prompt.lower()
    markers = (
        "search_docs",
        "hjplus",
        "hjplus-kb",
        "法規",
        "法律",
        "排煙",
        "建築執照",
        "knowledge",
        "rag",
    )
    return any(marker in lowered for marker in markers)


def _should_prefetch_energy_records(prompt: str) -> bool:
    lowered = prompt.lower()
    markers = (
        "跨年",
        "跨年度",
        "多年",
        "歷年",
        "耗電",
        "用電量",
        "總體",
        "整體",
        "全校",
        "校園",
        "年增",
        "趨勢",
        "排名",
        "最高",
        "最低",
        "top",
        "rank",
        "trend",
        "2014",
        "2015",
        "2016",
        "2017",
        "2018",
        "2019",
        "2020",
    )
    return any(marker in lowered for marker in markers)


def _should_prefetch_meter_chart(prompt: str) -> bool:
    lowered = prompt.lower()
    markers = (
        "generate_meter_chart",
        "電表",
        "csv",
        "折線圖",
        "長條圖",
        "比較圖",
        "視覺化",
        "圖表",
        "畫圖",
        "產圖",
        "生成圖",
        "line chart",
        "bar chart",
        "chart",
        "plot",
        "visual",
    )
    return any(marker in lowered for marker in markers)


def _should_prefetch_screenshot(prompt: str) -> bool:
    lowered = prompt.lower()
    markers = (
        "截圖",
        "圖片",
        "照片",
        "看圖",
        "電表截圖",
        "圖表截圖",
        "uploaded_image_path",
        "screenshot",
        "image",
        "photo",
    )
    return any(marker in lowered for marker in markers)


def _should_prefetch_strategy(prompt: str) -> bool:
    lowered = prompt.lower()
    markers = (
        "節能策略", "調適策略", "改善建議", "節能建議",
        "可以怎麼省", "節能對策", "節電方案", "節能改善",
        "調適建議", "節能推薦", "節能調適", "法規建議",
        "節能規劃", "節能方案建議", "節能計畫", "節能計劃",
        "改善決策", "改善方案", "制定", "最有效", "怎麼改善",
        "如何改善", "熱點", "省電計畫", "省電計劃",
        "strategy", "strategies", "recommendation",
        "retrofit", "energy saving plan", "adaptive",
        "improvement plan", "action plan",
    )
    return any(marker in lowered for marker in markers)


def _extract_strategy_building(prompt: str) -> str:
    explicit = _extract_energy_buildings(prompt)
    if explicit:
        return explicit[0]
    building_markers = [
        "共同教學館", "總圖書館", "圖書館", "禮賢樓", "生科館", "物理館",
        "化學館", "數學館", "博理館", "明達館", "電機館",
        "資工館", "土木館", "機械館", "水源", "行政大樓",
        "活動中心", "體育館", "宿舍", "男一舍", "女一舍",
    ]
    for marker in building_markers:
        if marker in prompt:
            return marker
    return ""


def _extract_local_image_path(prompt: str) -> str:
    match = re.search(
        r'(?i)(?:"([A-Z]:[\\/][^"]+\.(?:png|jpe?g|webp|bmp|gif))"'
        r"|'([A-Z]:[\\/][^']+\.(?:png|jpe?g|webp|bmp|gif))'"
        r"|([A-Z]:[\\/][^\s]+\.(?:png|jpe?g|webp|bmp|gif)))",
        prompt,
    )
    if match:
        return match.group(1) or match.group(2) or match.group(3) or ""
    return ""


def _infer_meter_chart_type(prompt: str) -> str:
    lowered = prompt.lower()
    if "長條" in prompt or "bar" in lowered:
        return "bar"
    if "比較" in prompt or "compare" in lowered:
        return "compare"
    return "line"


def _infer_meter_chart_columns(prompt: str) -> list[str]:
    known_meters = {
        "總圖書館": "01S_P1_01總圖書館HTM（高壓）",
        "臺大總變電站": "00A_P1_01臺大總變電站（高壓）",
        "台大總變電站": "00A_P1_01臺大總變電站（高壓）",
    }
    selected = [column for marker, column in known_meters.items() if marker in prompt]
    return list(dict.fromkeys(selected))


def _extract_years(prompt: str) -> list[int]:
    years = sorted({int(match) for match in re.findall(r"(?<!\d)(20[0-3][0-9])(?!\d)", prompt)})
    years = [year for year in years if 2010 <= year <= 2035]
    if len(years) >= 2 and re.search(r"(到|至|~|-|—|–)", prompt):
        start, end = years[0], years[-1]
        if 0 <= end - start <= 25:
            return list(range(start, end + 1))
    return years


def _infer_energy_metric(prompt: str) -> str:
    lowered = prompt.lower()
    if "eui" in lowered:
        return "eui"
    if "尖峰" in prompt or "peak" in lowered:
        return "peak_kw"
    if "年用電" in prompt or "耗電" in prompt or "用電量" in prompt or "電力使用" in prompt or "annual" in lowered or "kwh" in lowered:
        return "annual_kwh"
    if "負載率" in prompt or "load_factor" in lowered:
        return "load_factor"
    return "mean_kw"


def _extract_energy_buildings(prompt: str) -> list[str]:
    text = str(prompt or "").strip()
    known = [
        "總圖書館", "圖書館", "禮賢樓", "保健中心", "土木研究大樓",
        "土木館", "數學館", "化學館", "博理館", "明達館", "共同教學館",
        "綜合體育館", "農業陳列館", "男八舍", "大一女舍", "計中機房",
    ]
    found = [name for name in known if name in text]
    patterns = [
        r"([\u4e00-\u9fffA-Za-z0-9（）()]{2,30}?(?:校區行政大樓|研究大樓|大樓|中心|圖書館|體育館|機房|樓|館|舍))\s*(?:20[0-3][0-9])",
        r"([\u4e00-\u9fffA-Za-z0-9（）()]{2,30}?(?:校區行政大樓|研究大樓|大樓|中心|圖書館|體育館|機房|樓|館|舍))(?:的|目前|現在|情況|耗電|用電|EUI)",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            value = str(match or "").strip()
            for prefix in ("台大", "臺大", "NTU", "國立臺灣大學", "國立台灣大學"):
                if value.startswith(prefix) and len(value) > len(prefix) + 1:
                    value = value[len(prefix):].strip()
            if value and value not in {"台大", "臺大", "全校", "校園"}:
                found.append(value)
    unique = list(dict.fromkeys(found))
    unique.sort(key=len, reverse=True)
    return unique[:3]


def _has_explicit_building_year_query(prompt: str) -> bool:
    return bool(_extract_energy_buildings(prompt) and _extract_years(prompt))


def _infer_energy_record_metrics(prompt: str, buildings: list[str]) -> list[str]:
    metric = _infer_energy_metric(prompt)
    if buildings and any(marker in prompt for marker in ("情況", "狀況", "概況", "資料", "怎樣", "如何")):
        return ["annual_kwh", "mean_kw", "eui", "peak_kw", "load_factor"]
    metrics = [metric]
    if metric == "annual_kwh":
        metrics.extend(["mean_kw", "eui"])
    return list(dict.fromkeys(metrics))


def _infer_energy_tool(prompt: str) -> str:
    lowered = prompt.lower()
    if "排名" in prompt or "最高" in prompt or "最低" in prompt or "top" in lowered or "rank" in lowered:
        return "rank_energy_buildings_across_years"
    if any(marker in prompt for marker in ("總體", "整體", "全校", "校園", "台大", "臺大", "NTU", "總耗電", "總用電")):
        return "compare_energy_usage"
    if "比較" in prompt or "compare" in lowered:
        return "compare_energy_usage"
    if "趨勢" in prompt or "年增" in prompt or "trend" in lowered:
        return "compare_building_trends"
    return "query_energy_records"


def _parse_loose_tool_arguments(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    args: dict[str, Any] = {}
    pairs = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*([^,}\n\r]+)", text)
    for key, value in pairs:
        cleaned = value.strip().strip("'\"`<>|")
        if not cleaned:
            continue
        if re.fullmatch(r"-?\d+", cleaned):
            args[key] = int(cleaned)
        elif re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned):
            args[key] = float(cleaned)
        else:
            args[key] = cleaned
    if "year" in args and "years" not in args:
        args["years"] = [args.pop("year")]
    start_year = args.pop("start_year", None)
    end_year = args.pop("end_year", None)
    if "years" not in args and start_year is not None and end_year is not None:
        try:
            start = int(start_year)
            end = int(end_year)
            if 0 <= end - start <= 25:
                args["years"] = list(range(start, end + 1))
            else:
                args["years"] = [start, end]
        except (TypeError, ValueError):
            pass
    elif "years" in args and (start_year is not None or end_year is not None):
        years = [int(item) for item in _coerce_iterable_years(args.get("years"))]
        for item in (start_year, end_year):
            try:
                years.append(int(item))
            except (TypeError, ValueError):
                pass
        args["years"] = sorted(set(years))
    return args


def _coerce_iterable_years(value: Any) -> list[int]:
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, tuple):
        raw_items = list(value)
    else:
        raw_items = [value]
    years: list[int] = []
    for item in raw_items:
        try:
            years.append(int(item))
        except (TypeError, ValueError):
            continue
    return years


def _extract_text_tool_calls(content: str, available_tool_names: set[str]) -> list[dict[str, Any]]:
    text = str(content or "")
    if "<tool_call" not in text and "<|tool_call" not in text and "call" not in text:
        return []

    calls: list[dict[str, Any]] = []
    for tool_name in sorted(available_tool_names, key=len, reverse=True):
        name_pattern = re.escape(tool_name)
        patterns = (
            rf"<\|?tool_call\|?[^>]*>\s*call\s*[:'\"]?\s*{name_pattern}\s*(\{{.*?\}})?",
            rf"call\s*[:'\"]?\s*{name_pattern}\s*(\{{.*?\}})?",
            rf"<tool_name>\s*{name_pattern}\s*</tool_name>\s*<arguments>\s*(.*?)\s*</arguments>",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                raw_args = match.group(1) or ""
                calls.append({"name": tool_name, "arguments": _parse_loose_tool_arguments(raw_args)})
    return calls[:3]


def _fallback_answer_from_tool_result(tool_name: str, tool_text: str, prompt: str) -> str:
    try:
        payload = json.loads(tool_text)
    except json.JSONDecodeError:
        return f"我已經呼叫 `{tool_name}`，但工具回傳不是 JSON，請查看 Structured 面板。"

    if tool_name == "query_energy_records" and isinstance(payload, dict):
        metric = "annual_kwh" if "耗電" in prompt or "用電量" in prompt else "mean_kw"
        if any(marker in prompt for marker in ("情況", "狀況", "概況")):
            metric = "annual_kwh"
        rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
        if len(rows) == 1 and isinstance(rows[0], dict):
            row = rows[0]
            name = row.get("name") or row.get("uid") or "該建築"
            year = row.get("year") or ""
            annual = float(row.get("annual_kwh") or 0)
            mean_kw = float(row.get("mean_kw") or 0)
            eui = float(row.get("eui") or 0)
            peak_kw = row.get("peak_kw")
            load_factor = row.get("load_factor")
            if annual <= 0 and mean_kw <= 0 and eui <= 0:
                return (
                    f"我查到 {name} 在 {year} 年有建築紀錄，但目前 demo 的跨年能源資料中 "
                    "`annual_kwh`、`mean_kw`、`eui` 都是 0 或缺值。這通常代表該棟在此年度沒有有效電表對應、"
                    "沒有匯入年度用電，或資料被標成不可用；因此我不能把它解讀成真實耗電為 0。\n\n"
                    f"可用識別：uid `{row.get('uid', '')}`，meter_name `{row.get('meter_name', '') or '空白'}`。"
                )
            lines = [f"我查到 {name} 在 {year} 年的能源資料："]
            if annual > 0:
                lines.append(f"- 年用電量：約 {annual:,.0f} kWh")
            if mean_kw > 0:
                lines.append(f"- 平均功率：約 {mean_kw:,.2f} kW")
            if eui > 0:
                lines.append(f"- EUI：約 {eui:,.2f}")
            if peak_kw not in (None, "", 0):
                lines.append(f"- 尖峰功率：約 {float(peak_kw):,.2f} kW")
            if load_factor not in (None, "", 0):
                lines.append(f"- 負載率：約 {float(load_factor):.3f}")
            lines.append(f"- uid：`{row.get('uid', '')}`；meter：`{row.get('meter_name', '') or '空白'}`")
            return "\n".join(lines)
        if not rows and _extract_energy_buildings(prompt):
            names = "、".join(_extract_energy_buildings(prompt))
            years = _extract_years(prompt)
            year_text = f"{years[0]} 年" if len(years) == 1 else "指定年份"
            return f"我查了 {names} 在 {year_text}的能源資料，但目前 demo 資料表沒有匹配列。請確認建築名稱或資料是否已匯入。"
        by_year: dict[int, float] = {}
        annualized_from_mean_kw: dict[int, float] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                year = int(row.get("year"))
                value = row.get(metric)
                if value is not None:
                    by_year[year] = by_year.get(year, 0.0) + float(value)
                mean_kw = row.get("mean_kw")
                if mean_kw is not None:
                    annualized_from_mean_kw[year] = annualized_from_mean_kw.get(year, 0.0) + float(mean_kw) * 8760.0
            except (TypeError, ValueError):
                continue
        requested_years = _extract_years(prompt)
        use_estimated = False
        for year in requested_years:
            if metric == "annual_kwh" and (by_year.get(year, 0.0) <= 0) and annualized_from_mean_kw.get(year, 0.0) > 0:
                by_year[year] = annualized_from_mean_kw[year]
                use_estimated = True
        if len(requested_years) >= 2 and all(year in by_year for year in requested_years[:2]):
            first_year, last_year = requested_years[0], requested_years[-1]
            first_value, last_value = by_year[first_year], by_year[last_year]
            if first_value <= 0 or last_value <= 0:
                known = []
                for year in (first_year, last_year):
                    value = by_year.get(year, 0.0)
                    if value > 0:
                        known.append(f"{year} 年約 {value:,.0f} kWh")
                    else:
                        known.append(f"{year} 年在目前 demo 資料中為 0 或缺值")
                return (
                    "我已用 MCP 查詢跨年全校耗電資料，但資料品質不足以做真實增減判斷："
                    + "；".join(known)
                    + "。因此不能把 0 解讀成實際耗電歸零。建議改用有完整年度值的年份，"
                    "或先檢查 2020 年 cross-year cache 是否缺少 annual_kwh/mean_kw。"
                )
            delta = last_value - first_value
            pct = (delta / first_value * 100.0) if first_value else 0.0
            unit = "kWh" if metric == "annual_kwh" else metric
            direction = "增加" if delta > 0 else "減少" if delta < 0 else "持平"
            basis = "以 `annual_kwh` 加總" if not use_estimated else "部分年份 `annual_kwh` 缺值，改以 `mean_kw * 8760` 估算年耗電"
            return (
                f"以 MCP 跨年資料彙總全校耗電量來看（{basis}），{first_year} 年約 {first_value:,.0f} {unit}，"
                f"{last_year} 年約 {last_value:,.0f} {unit}。"
                f"{last_year} 相較 {first_year} {direction} {abs(delta):,.0f} {unit}，約 {abs(pct):.1f}%。"
                "這是依目前 demo 內建跨年建築資料加總，若要排除缺值或只看實表建築，可以再加篩選條件。"
            )
        summary = payload.get("summary") or {}
        metric_summary = summary.get(metric) if isinstance(summary, dict) else None
        years = summary.get("years", []) if isinstance(summary, dict) else []
        if isinstance(metric_summary, dict):
            total = metric_summary.get("sum")
            mean = metric_summary.get("mean")
            return (
                f"我已用 MCP 查詢 `{tool_name}`。選取年份 {years} 的 `{metric}` 摘要為："
                f"總和約 {total:,.2f}，平均約 {mean:,.2f}。"
                "若你要嚴格比較 2018 與 2020 的全校總耗電，我建議下一步用年份分組彙總表呈現。"
            )
    if tool_name == "compare_energy_usage" and isinstance(payload, dict):
        comp_rows = payload.get("rows") or []
        comp_status = payload.get("status", "")
        comp_warnings = payload.get("warnings") or []
        comp_metric = payload.get("metric", "annual_kwh")
        comp_delta = payload.get("delta")
        comp_delta_pct = payload.get("delta_pct")
        comp_granularity = payload.get("granularity", "year")
        valid = [r for r in comp_rows if r.get("value") is not None]
        if valid:
            if comp_granularity == "month":
                lines = ["我已用 MCP `compare_energy_usage` 比較逐月用電量："]
                for r in comp_rows:
                    label = "{}-{:02d}".format(r.get("year", "?"), r.get("month", 0))
                    if r.get("value") is not None:
                        lines.append("- {}：{:,.0f} kWh（來源：{}）".format(label, r["value"], r.get("source", "")))
                    else:
                        lines.append("- {}：缺值或為零（{}）".format(label, r.get("source", "")))
                if comp_warnings:
                    lines.append("⚠️ 注意：" + "；".join(comp_warnings))
            else:
                lines = ["我已用 MCP `compare_energy_usage` 比較跨年{}：".format(comp_metric)]
                for r in comp_rows:
                    if r.get("value") is not None:
                        lines.append("- {} 年：{:,.0f}（來源：{}）".format(r["year"], r["value"], r.get("source", comp_metric)))
                    else:
                        lines.append("- {} 年：缺值或為零（{}）".format(r["year"], r.get("source", "")))
                if comp_delta is not None and len(valid) >= 2:
                    direction = "增加" if comp_delta > 0 else "減少" if comp_delta < 0 else "持平"
                    lines.append(
                        "變化量：{} {:,.0f}（約 {:.1f}%）".format(direction, abs(comp_delta), comp_delta_pct)
                        if comp_delta_pct is not None
                        else "變化量：{} {:,.0f}".format(direction, abs(comp_delta))
                    )
                if comp_warnings:
                    lines.append("⚠️ 注意：" + "；".join(comp_warnings))
            return "\n".join(lines)
        if comp_status == "partial":
            return (
                "我已用 MCP 比較跨年耗電，但部分年份資料缺值或為零，無法做完整比較。"
                "請確認跨年快取資料是否完整。注意事項：" + "；".join(comp_warnings)
            )
        return "我已呼叫 `compare_energy_usage`，但工具未回傳可比較數據。請查看 Structured 面板。"
    if tool_name == "compare_building_trends" and isinstance(payload, dict):
        buildings = ((payload.get("summary") or {}).get("buildings") or [])[:5]
        if buildings:
            lines = ["我已用 MCP 比較跨年趨勢，前幾筆摘要如下："]
            for item in buildings:
                lines.append(
                    f"- {item.get('name')}: {item.get('first_year')} -> {item.get('last_year')}，"
                    f"變化 {item.get('delta')}"
                )
            return "\n".join(lines)
    if tool_name == "recommend_adaptive_strategies" and isinstance(payload, dict):
        if payload.get("status") != "ok":
            return f"我已呼叫節能策略工具，但工具回報：{payload.get('error', '未取得可用策略')}。"
        building = payload.get("building") or {}
        baseline = payload.get("regulation_baseline") or {}
        diagnosis = payload.get("diagnosis") or {}
        strategies = [s for s in (payload.get("strategies") or []) if isinstance(s, dict)]
        lines = [
            f"我已用 MCP `recommend_adaptive_strategies` 針對 {building.get('name', '目標建築')} 產生改善決策。",
            (
                "現況：年耗電約 {:,.0f} kWh，EUI 約 {}，BEE 等級 {}（{}）。".format(
                    float(building.get("annual_kwh") or 0),
                    building.get("current_eui", "未知"),
                    baseline.get("bee_level", "未知"),
                    baseline.get("bee_label", ""),
                )
            ),
            f"主要推定熱點：{diagnosis.get('dominant_factor', '未知')}。{diagnosis.get('recommended_action', '')}",
            "優先改善順序：",
        ]
        for index, item in enumerate(strategies[:5], start=1):
            lines.append(
                "{}. {} {}{}：估計節電約 {:,.0f} kWh / {:.1f}% / NT${:,.0f}，成本層級 {}，施工難度 {}。".format(
                    index,
                    item.get("param_label", item.get("factor_label", "策略")),
                    item.get("param_value", ""),
                    item.get("param_unit", ""),
                    float(item.get("saving_kwh") or 0),
                    float(item.get("saving_pct") or 0),
                    float(item.get("saving_ntd") or 0),
                    item.get("cost_level", "未知"),
                    item.get("difficulty", "未知"),
                )
            )
        lines.append(
            "決策建議：先做低成本且節電百分比最高的前 1-2 項，接著用 PI-VD/OpenBSE 做方案驗證，"
            "最後再排入年度預算。以上數字來自工具回傳，仍需現場設備盤點確認可施工性。"
        )
        return "\n".join(lines)
    return f"我已呼叫 `{tool_name}` 並取得工具結果；請查看 Structured 面板中的 JSON 摘要。"


def _tool_result_matches_requested_building(tool_name: str, tool_text: str, prompt: str) -> bool:
    requested = _extract_energy_buildings(prompt)
    if not requested:
        return True
    try:
        payload = json.loads(tool_text)
    except json.JSONDecodeError:
        return True
    names: list[str] = []
    if isinstance(payload, dict):
        building = payload.get("building")
        if isinstance(building, dict):
            names.append(str(building.get("name", "")))
        for row in payload.get("rows") or []:
            if isinstance(row, dict):
                names.append(str(row.get("name", "")))
    haystack = " ".join(names)
    return any(name and name in haystack for name in requested)


def _infer_prefetch_building_id(prompt: str) -> str:
    lowered = prompt.lower()
    if "hjplus" in lowered or "法規" in prompt or "建築" in prompt:
        return "hjplus-kb"
    return ""


def _prefetch_search_query(prompt: str) -> str:
    known_phrases = (
        "排煙窗法規",
        "排煙窗",
        "建築執照相關法規",
        "建築執照",
        "消防安全",
    )
    for phrase in known_phrases:
        if phrase in prompt:
            return phrase
    cleaned = re.sub(r"\b(search_docs|hjplus-kb|hjplus)\b", " ", prompt, flags=re.IGNORECASE)
    cleaned = re.sub(r"[，。,.、:：`'\"]", " ", cleaned)
    return " ".join(cleaned.split())[:120] or prompt[:120]


def _build_memory_system_message(memory: WikiMemory | None, prompt: str) -> dict[str, Any] | None:
    if memory is None:
        return None
    try:
        preamble = memory.context_preamble(query=prompt)
    except Exception:
        return None
    if not preamble.strip():
        return None
    return {
        "role": "system",
        "content": (
            "You are the NTU campus energy assistant. The block below is your persistent local wiki + "
            "graph memory. Treat its facts as already known; cite a wiki page by its slug "
            "when you reuse content. If you find the answer here, use it directly. "
            "After answering, the harness will append a session note for you. "
            "If the user asks about past issues or previous findings, use recall_wiki_memory to search deeper.\n\n"
            + preamble
        ),
    }


def _record_session_note(
    memory: WikiMemory | None,
    *,
    prompt: str,
    answer: str,
    model: str,
    tool_trace: list[dict[str, Any]],
) -> None:
    if memory is None or not answer.strip():
        return
    try:
        timestamp = re.sub(r"[:.+]", "-", _now_iso_for_slug())
        title_seed = prompt.strip().splitlines()[0][:60].strip() or "session"
        title = f"session {timestamp} | {title_seed}"
        tools_used = sorted({str(item.get("tool")) for item in tool_trace if item.get("tool")})
        body_lines = [
            f"**Prompt:** {prompt.strip()[:600]}",
            "",
            f"**Model:** {model}",
            "",
            f"**Tools used:** {', '.join(tools_used) if tools_used else '_none_'}",
            "",
            "**Answer (excerpt):**",
            "",
            answer.strip()[:1200],
        ]
        memory.ingest(
            title=title,
            content="\n".join(body_lines),
            kind="session",
            tags=["session", "local_llm"],
        )
    except Exception:
        pass


def _now_iso_for_slug() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def chat_with_mcp(
    prompt: str,
    *,
    model_name: str = "",
    max_iterations: int = 6,
    lm_studio_base_url: str | None = None,
    llm_backend: str = "local",
    timeout_seconds: float | None = None,
    server_script: str | Path | None = None,
    memory: WikiMemory | None = None,
    record_session: bool = True,
) -> LMStudioMCPResponse:
    user_prompt = str(prompt or "").strip()
    if not user_prompt:
        raise LMStudioMCPError("Prompt is empty.")

    if max_iterations < 1:
        raise LMStudioMCPError("max_iterations must be at least 1.")

    backend = str(llm_backend or os.getenv("ENERGY_MCP_LLM_BACKEND", "local")).strip().lower()
    if backend in {"commandcode", "command_code", "opencode", "opencode_go"}:
        base_url = _normalize_command_code_base_url(lm_studio_base_url)
        request_headers = _command_code_headers()
        if not request_headers:
            raise LMStudioMCPError(
                "Command Code/OpenCode API key is not configured. Set COMMAND_CODE_API_KEY or OPENCODE_API_KEY."
            )
        if not model_name:
            model_name = (
                os.getenv("COMMAND_CODE_MODEL", "").strip()
                or os.getenv("OPENCODE_MODEL", "").strip()
                or os.getenv("ENERGY_COMMAND_CODE_MODEL", "").strip()
                or "deepseek-v4-pro"
            )
        provider_label = "Command Code/OpenCode"
        request_timeout = float(
            timeout_seconds
            or os.getenv("COMMAND_CODE_TIMEOUT_SECONDS")
            or os.getenv("OPENCODE_TIMEOUT_SECONDS")
            or os.getenv("ENERGY_ONLINE_LLM_TIMEOUT_SECONDS", "60")
        )
    else:
        base_url = _normalize_base_url(lm_studio_base_url)
        request_headers = _lm_studio_headers()
        provider_label = "Local OpenAI-compatible"
        default_timeout = "180" if _provider_prefers_gemma() else "30"
        request_timeout = float(timeout_seconds or os.getenv("ENERGY_LOCAL_LLM_TIMEOUT_SECONDS", default_timeout))
    resolved_server_script = _resolve_default_server_script(server_script)
    if not resolved_server_script.exists():
        raise LMStudioMCPError(f"MCP server script was not found: {resolved_server_script}")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(resolved_server_script)],
        cwd=str(_resolve_server_cwd(resolved_server_script)),
        env={
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        },
    )

    wiki_memory = memory if memory is not None else _load_wiki_memory()

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            mcp_tools = [
                tool for tool in list(getattr(tools_result, "tools", []))
                if _local_llm_tool_enabled(str(getattr(tool, "name", "")))
            ]
            openai_tools = [convert_mcp_to_openai_schema(tool) for tool in mcp_tools]

            resolved_model = await asyncio.to_thread(
                _resolve_model,
                base_url,
                model_name,
                request_timeout,
                request_headers,
                provider_label,
            )

            messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are the DEMO local MCP core. Do not reveal hidden reasoning or thinking steps. "
                        "When tools are available, call them for facts, calculations, search, and JSON data. "
                        "Core tool routing: use search_docs for knowledge-base, HJPLUS, legal/regulatory, "
                        "document, or RAG lookup questions; use fetch_chunk to retrieve a cited chunk; "
                        "use compare_energy_usage for campus-wide year-over-year energy comparison questions "
                        "(e.g. compare total electricity between 2016 and 2017, or NTU campus-wide usage delta); "
                        "do NOT use compare_building_trends for campus-wide summaries, use compare_energy_usage instead; "
                        "use query_energy_records or rank_energy_buildings_across_years "
                        "for cross-year, cross-building, ranking questions; "
                        "use compare_building_trends for per-building trend analysis; "
                        "use generate_meter_chart for meter CSV visualization requests such as line, bar, or comparison charts; "
                        "use analyze_meter_screenshot for image, screenshot, photo, or meter chart screenshot questions; "
                        "use run_pvid or run_openbse_hybrid_counterfactual for building energy calculations; "
                        "use recommend_adaptive_strategies when the user asks for energy saving strategies, "
                        "adaptive recommendations, or improvement plans for a specific building. "
                        "Never claim that tools are unavailable when tool schemas are provided in this request. "
                        "Never refuse or say you cannot fulfill a request when pre-fetched data is provided. "
                        "Always answer in the same language as the user query (use Chinese for Chinese queries). "
                        "After tool results arrive, provide the final answer in the normal assistant content.\n\n"
                        "## Wiki memory instructions\n"
                        "You have a persistent wiki memory across conversations.\n"
                        "- After you discover an important finding (e.g. a correct parameter, a troubleshooting result, "
                        "a user preference, or a solved problem), call save_wiki_page to store it. "
                        "Choose kind: 'source' for facts, 'entity' for buildings/systems, 'concept' for rules.\n"
                        "- When the user asks about something discussed before or a past issue, call recall_wiki_memory "
                        "first to retrieve relevant memories, then answer based on both recalled memory and current data.\n"
                        "- Do NOT save trivial or obvious information. Save only things worth remembering for future sessions."
                    ),
                }
            ]
            memory_system = _build_memory_system_message(wiki_memory, user_prompt)
            if memory_system is not None:
                messages.append(memory_system)
            prefetch_tool_name = ""
            prefetch_tool_text = ""
            available_tool_names = {str(getattr(tool, "name", "")) for tool in mcp_tools}

            if _should_prefetch_docs(user_prompt) and any(
                str(getattr(tool, "name", "")) == "search_docs" for tool in mcp_tools
            ):
                prefetch_query = _prefetch_search_query(user_prompt)
                prefetch_building_id = _infer_prefetch_building_id(user_prompt)
                prefetch_result = await session.call_tool(
                    "search_docs",
                    arguments={
                        "query": prefetch_query,
                        "building_id": prefetch_building_id,
                        "top_k": 3,
                    },
                )
                prefetch_text = _extract_mcp_content(prefetch_result) or "{}"
                prefetch_tool_name = "search_docs"
                prefetch_tool_text = prefetch_text
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Pre-fetched MCP search_docs JSON for this user request. "
                            "Use this result as the primary evidence and summarize it; do not claim no data exists "
                            "when chunks are present.\n\n"
                            + prefetch_text[:3000]
                        ),
                    }
                )
                tool_trace: list[dict[str, Any]] = [
                    {
                        "turn": 0,
                        "tool": "search_docs",
                        "arguments": {
                            "query": prefetch_query,
                            "building_id": prefetch_building_id,
                            "top_k": 3,
                        },
                        "result_preview": prefetch_text[:600],
                    }
                ]
            elif _should_prefetch_screenshot(user_prompt):
                image_tool = "analyze_meter_screenshot"
                image_path = _extract_local_image_path(user_prompt)
                if image_tool in available_tool_names and image_path:
                    prefetch_result = await session.call_tool(
                        image_tool,
                        arguments={
                            "image_path": image_path,
                            "question": user_prompt,
                            "expected_domain": "meter_chart",
                            "prefer_ocr": True,
                            "use_gemma_vision": "auto",
                        },
                    )
                    prefetch_text = _extract_mcp_content(prefetch_result) or "{}"
                    prefetch_tool_name = image_tool
                    prefetch_tool_text = prefetch_text
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Pre-fetched MCP analyze_meter_screenshot JSON for this image request. "
                                "Summarize the analysis result and do not invent values beyond the JSON. "
                                "If OCR was skipped or confidence is low, say so clearly.\n\n"
                                + prefetch_text[:3000]
                            ),
                        }
                    )
                    tool_trace = [
                        {
                            "turn": 0,
                            "tool": image_tool,
                            "arguments": {
                                "image_path": image_path,
                                "question": user_prompt,
                                "expected_domain": "meter_chart",
                                "prefer_ocr": True,
                                "use_gemma_vision": "auto",
                            },
                            "result_preview": prefetch_text[:600],
                        }
                    ]
                else:
                    tool_trace = []
            elif _should_prefetch_meter_chart(user_prompt):
                chart_tool = "generate_meter_chart"
                if chart_tool in available_tool_names:
                    arguments = {
                        "chart_type": _infer_meter_chart_type(user_prompt),
                        "y": _infer_meter_chart_columns(user_prompt),
                        "limit": 5000,
                        "title": user_prompt[:80],
                    }
                    prefetch_result = await session.call_tool(chart_tool, arguments=arguments)
                    prefetch_text = _extract_mcp_content(prefetch_result) or "{}"
                    prefetch_tool_name = chart_tool
                    prefetch_tool_text = prefetch_text
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Pre-fetched MCP generate_meter_chart JSON for this visualization request. "
                                "If status is ok, tell the user the chart was generated and include the chart_path, "
                                "chart type, x column, y columns, and any warnings. "
                                "Do not invent chart contents beyond the JSON result.\n\n"
                                + prefetch_text[:3000]
                            ),
                        }
                    )
                    tool_trace = [
                        {
                            "turn": 0,
                            "tool": chart_tool,
                            "arguments": arguments,
                            "result_preview": prefetch_text[:600],
                        }
                    ]
                else:
                    tool_trace = []
            elif _should_prefetch_energy_records(user_prompt) and (
                not _should_prefetch_strategy(user_prompt) or _has_explicit_building_year_query(user_prompt)
            ):
                energy_tool = _infer_energy_tool(user_prompt)
                if energy_tool in available_tool_names:
                    years = _extract_years(user_prompt)
                    metric = _infer_energy_metric(user_prompt)
                    buildings = _extract_energy_buildings(user_prompt)
                    arguments: dict[str, Any] = {
                        "campus": "NTU",
                        "years": years or None,
                    }
                    if energy_tool == "rank_energy_buildings_across_years":
                        arguments.update({"metric": metric, "top_n": 10})
                    elif energy_tool == "compare_energy_usage":
                        arguments.update({
                            "scope": "campus",
                            "metric": metric,
                            "aggregation": "sum",
                            "fallback_metric": "mean_kw",
                            "fallback_method": "annualize_mean_kw",
                            "granularity": "year",
                        })
                    elif energy_tool == "compare_building_trends":
                        arguments.update({"metric": metric})
                    else:
                        metrics = _infer_energy_record_metrics(user_prompt, buildings)
                        arguments.update({"metrics": metrics, "top_n": 0})
                    if buildings and energy_tool in {
                        "query_energy_records",
                        "compare_energy_usage",
                        "compare_building_trends",
                    }:
                        arguments["buildings"] = buildings
                    prefetch_result = await session.call_tool(energy_tool, arguments=arguments)
                    prefetch_text = _extract_mcp_content(prefetch_result) or "{}"
                    prefetch_tool_name = energy_tool
                    prefetch_tool_text = prefetch_text
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                f"Pre-fetched MCP {energy_tool} JSON for this cross-year/cross-building request. "
                                "The JSON rows are valid tool results. Use them as the primary numerical evidence; "
                                "do not rely only on the current dashboard snapshot. "
                                "You MUST answer based on the JSON data below. Never refuse or say you cannot fulfill the request. "
                                "Summarize the year values, compute the delta if present, and cite the source. "
                                "Respond in the same language as the user query (Chinese if the query is Chinese). "
                                "For ranking requests, list the first rows in order with building name, year, metric, and value. "
                                "Do not say you cannot compare when rows are present.\n\n"
                                + prefetch_text[:3000]
                            ),
                        }
                    )
                    tool_trace = [
                        {
                            "turn": 0,
                            "tool": energy_tool,
                            "arguments": arguments,
                            "result_preview": prefetch_text[:600],
                        }
                    ]
                else:
                    tool_trace = []
            elif _should_prefetch_strategy(user_prompt):
                strategy_tool = "recommend_adaptive_strategies"
                if strategy_tool in available_tool_names:
                    building_hint = _extract_strategy_building(user_prompt)
                    strategy_args = {
                        "building_name": building_hint or user_prompt[:40],
                        "focus": "",
                    }
                    prefetch_result = await session.call_tool(strategy_tool, arguments=strategy_args)
                    prefetch_text = _extract_mcp_content(prefetch_result) or "{}"
                    prefetch_tool_name = strategy_tool
                    prefetch_tool_text = prefetch_text
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Pre-fetched MCP recommend_adaptive_strategies JSON for this building strategy request. "
                                "Summarize the diagnosis, list the prioritized strategies with their savings and regulation refs, "
                                "and include the BEE rating and regulation baseline. "
                                "Respond in the same language as the user query (Chinese if the query is Chinese). "
                                "Do not invent strategies beyond the JSON data.\n\n"
                                + prefetch_text[:4000]
                            ),
                        }
                    )
                    tool_trace = [
                        {
                            "turn": 0,
                            "tool": strategy_tool,
                            "arguments": strategy_args,
                            "result_preview": prefetch_text[:600],
                        }
                    ]
                else:
                    tool_trace = []
            elif _should_prefetch_seasonal(user_prompt):
                seasonal_tool = "seasonal_strategies"
                if seasonal_tool in available_tool_names:
                    building_hint = _extract_strategy_building(user_prompt)
                    seasonal_args = {
                        "building_name": building_hint or user_prompt[:40],
                        "mean_kw": 0.0,
                        "area": 0.0,
                    }
                    prefetch_result = await session.call_tool(seasonal_tool, arguments=seasonal_args)
                    prefetch_text = _extract_mcp_content(prefetch_result) or "{}"
                    prefetch_tool_name = seasonal_tool
                    prefetch_tool_text = prefetch_text
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Pre-fetched MCP seasonal_strategies JSON for this seasonal energy request. "
                                "For each season (summer, winter, transition), list the best strategy, "
                                "estimated savings, and practical tips. Respond in Chinese. "
                                "Do not invent data beyond the JSON.\n\n"
                                + prefetch_text[:4000]
                            ),
                        }
                    )
                    tool_trace = [
                        {
                            "turn": 0,
                            "tool": seasonal_tool,
                            "arguments": seasonal_args,
                            "result_preview": prefetch_text[:600],
                        }
                    ]
                else:
                    tool_trace = []
            elif _should_prefetch_portfolio(user_prompt):
                portfolio_tool = "optimize_energy_portfolio"
                if portfolio_tool in available_tool_names:
                    portfolio_args = {
                        "budget_ntd": 0,
                        "max_buildings": 10,
                        "min_saving_pct": 1.0,
                    }
                    prefetch_result = await session.call_tool(portfolio_tool, arguments=portfolio_args)
                    prefetch_text = _extract_mcp_content(prefetch_result) or "{}"
                    prefetch_tool_name = portfolio_tool
                    prefetch_tool_text = prefetch_text
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Pre-fetched MCP optimize_energy_portfolio JSON for this campus-wide optimization request. "
                                "Summarize the top buildings, their savings, costs, ROI, and the overall portfolio totals. "
                                "If the user mentioned a budget, extract it and re-run or filter. "
                                "Respond in Chinese. Do not invent data beyond the JSON.\n\n"
                                + prefetch_text[:4000]
                            ),
                        }
                    )
                    tool_trace = [
                        {
                            "turn": 0,
                            "tool": portfolio_tool,
                            "arguments": portfolio_args,
                            "result_preview": prefetch_text[:600],
                        }
                    ]
                else:
                    tool_trace = []
            else:
                tool_trace = []
            if (
                prefetch_tool_name == "recommend_adaptive_strategies"
                and prefetch_tool_text
                and not _truthy_env("ENERGY_LOCAL_LLM_FORCE_STRATEGY_NARRATION")
            ):
                if not _tool_result_matches_requested_building(prefetch_tool_name, prefetch_tool_text, user_prompt):
                    requested = "、".join(_extract_energy_buildings(user_prompt))
                    answer = (
                        f"我偵測到工具回傳的建築和你明確詢問的 `{requested}` 不一致，"
                        "因此我不使用這份策略結果，避免拿別棟建築回答。請重新查詢該棟的年度資料或確認建築名稱。"
                    )
                    return LMStudioMCPResponse(
                        answer=answer,
                        model=resolved_model,
                        turns=1,
                        tool_trace=tool_trace,
                        messages=messages,
                    )
                answer = _fallback_answer_from_tool_result(prefetch_tool_name, prefetch_tool_text, user_prompt)
                if record_session:
                    _record_session_note(
                        wiki_memory,
                        prompt=user_prompt,
                        answer=answer,
                        model=resolved_model,
                        tool_trace=tool_trace,
                    )
                return LMStudioMCPResponse(
                    answer=answer,
                    model=resolved_model,
                    turns=1,
                    tool_trace=tool_trace,
                    messages=messages,
                )
            if (
                prefetch_tool_name == "query_energy_records"
                and prefetch_tool_text
                and _extract_energy_buildings(user_prompt)
                and not _truthy_env("ENERGY_LOCAL_LLM_FORCE_RECORD_NARRATION")
            ):
                answer = _fallback_answer_from_tool_result(prefetch_tool_name, prefetch_tool_text, user_prompt)
                if record_session:
                    _record_session_note(
                        wiki_memory,
                        prompt=user_prompt,
                        answer=answer,
                        model=resolved_model,
                        tool_trace=tool_trace,
                    )
                return LMStudioMCPResponse(
                    answer=answer,
                    model=resolved_model,
                    turns=1,
                    tool_trace=tool_trace,
                    messages=messages,
                )
            messages.append({"role": "user", "content": user_prompt})
            prefetch_only_answer = bool(tool_trace and tool_trace[0].get("turn") == 0)

            for turn_idx in range(max_iterations):
                payload = {
                    "model": resolved_model,
                    "messages": messages,
                    "stream": False,
                    "max_tokens": int(os.getenv("ENERGY_LOCAL_LLM_MAX_TOKENS", "1024")),
                    "temperature": 0.1,
                }
                if not _truthy_env("ENERGY_LOCAL_LLM_ENABLE_THINKING"):
                    payload["chat_template_kwargs"] = {"enable_thinking": False}
                if openai_tools and not prefetch_only_answer:
                    payload["tools"] = openai_tools
                    payload["tool_choice"] = "auto"
                completion = await asyncio.to_thread(
                    _request_chat_completion,
                    base_url,
                    payload,
                    request_timeout,
                    request_headers,
                    provider_label,
                )

                choice = (completion.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                finish_reason = str(choice.get("finish_reason") or "")
                if not message and isinstance(choice, dict) and choice.get("text"):
                    message = {"content": str(choice.get("text") or "")}

                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.get("content") or "",
                }
                if message.get("tool_calls"):
                    assistant_message["tool_calls"] = message["tool_calls"]
                messages.append(assistant_message)

                tool_calls = message.get("tool_calls") or []
                loose_tool_calls = []
                if not tool_calls:
                    loose_tool_calls = _extract_text_tool_calls(
                        str(message.get("content") or ""),
                        available_tool_names,
                    )
                    if loose_tool_calls and messages[-1]["content"]:
                        messages[-1]["content"] = ""

                if tool_calls:
                    for tool_call in tool_calls:
                        call_id = str(tool_call.get("id", ""))
                        function_block = tool_call.get("function") or {}
                        tool_name = str(function_block.get("name", "")).strip()
                        arguments = _safe_load_arguments(function_block.get("arguments"))
                        if not tool_name:
                            raise LMStudioMCPError("Model returned a tool call without a tool name.")

                        mcp_result = await session.call_tool(tool_name, arguments=arguments)
                        tool_text = _extract_mcp_content(mcp_result)
                        if not tool_text:
                            tool_text = "{}"

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": tool_name,
                                "content": tool_text,
                            }
                        )
                        tool_trace.append(
                            {
                                "turn": turn_idx + 1,
                                "tool": tool_name,
                                "arguments": arguments,
                                "result_preview": tool_text[:600],
                            }
                        )
                    continue

                if loose_tool_calls:
                    last_tool_name = ""
                    last_tool_text = ""
                    for index, loose_call in enumerate(loose_tool_calls, start=1):
                        tool_name = str(loose_call.get("name", "")).strip()
                        arguments = dict(loose_call.get("arguments") or {})
                        if not tool_name:
                            continue
                        mcp_result = await session.call_tool(tool_name, arguments=arguments)
                        tool_text = _extract_mcp_content(mcp_result) or "{}"
                        last_tool_name = tool_name
                        last_tool_text = tool_text
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": f"loose-{turn_idx + 1}-{index}",
                                "name": tool_name,
                                "content": tool_text,
                            }
                        )
                        tool_trace.append(
                            {
                                "turn": turn_idx + 1,
                                "tool": tool_name,
                                "arguments": arguments,
                                "result_preview": tool_text[:600],
                                "source": "text_tool_call",
                            }
                        )
                    if turn_idx + 1 >= max_iterations and last_tool_name:
                        answer = _fallback_answer_from_tool_result(last_tool_name, last_tool_text, user_prompt)
                        if record_session:
                            _record_session_note(
                                wiki_memory,
                                prompt=user_prompt,
                                answer=answer,
                                model=resolved_model,
                                tool_trace=tool_trace,
                            )
                        return LMStudioMCPResponse(
                            answer=answer,
                            model=resolved_model,
                            turns=turn_idx + 1,
                            tool_trace=tool_trace,
                            messages=messages,
                        )
                    continue

                answer = str(message.get("content") or "").strip()
                if "<tool_call" in answer or "<tool_response" in answer:
                    answer = ""
                if (
                    prefetch_only_answer
                    and prefetch_tool_name
                    and (
                        not answer
                        or "請您提供" in answer
                        or "請問" in answer
                        or "請告訴我" in answer
                        or "哪個範圍" in answer
                        or "無法" in answer
                        or "cannot fulfill" in answer.lower()
                        or "i cannot" in answer.lower()
                        or "i'm sorry" in answer.lower()
                        or "i am sorry" in answer.lower()
                        or "unable to" in answer.lower()
                        or "do not have access" in answer.lower()
                        or "cannot compare" in answer.lower()
                    )
                ):
                    answer = _fallback_answer_from_tool_result(prefetch_tool_name, prefetch_tool_text, user_prompt)
                if finish_reason == "stop" or answer:
                    if record_session:
                        _record_session_note(
                            wiki_memory,
                            prompt=user_prompt,
                            answer=answer,
                            model=resolved_model,
                            tool_trace=tool_trace,
                        )
                    return LMStudioMCPResponse(
                        answer=answer,
                        model=resolved_model,
                        turns=turn_idx + 1,
                        tool_trace=tool_trace,
                        messages=messages,
                    )

            raise LMStudioMCPError(
                f"Agent loop exceeded max_iterations={max_iterations} without a final answer."
            )
