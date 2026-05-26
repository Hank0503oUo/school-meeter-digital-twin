from __future__ import annotations

import os
import re
from typing import Any


def _try_import_pillow() -> Any:
    try:
        from PIL import Image

        return Image
    except Exception:
        return None


def _try_import_pytesseract() -> Any:
    try:
        import pytesseract

        return pytesseract
    except Exception:
        return None


def _detect_unit_from_text(text: str) -> str:
    patterns = [
        (r"\bkWh\b", "kWh"),
        (r"\bkW\b", "kW"),
        (r"\bkVA\b", "kVA"),
        (r"\bkVar\b", "kVar"),
        (r"\bMWh\b", "MWh"),
        (r"\bMW\b", "MW"),
        (r"\bGWh\b", "GWh"),
        (r"度電", "kWh"),
        (r"用電量|電量|用電|度", "kWh"),
        (r"需量|尖峰|半尖峰|離峰", "kW"),
    ]
    for pattern, unit in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return unit
    return ""


def _detect_title(text_lines: list[str]) -> str:
    for line in text_lines:
        stripped = line.strip()
        if stripped and len(stripped) > 2:
            return stripped
    return ""


def _detect_axis(text_lines: list[str], full_text: str, keywords: list[str]) -> str:
    for keyword in keywords:
        if re.search(keyword, full_text, re.IGNORECASE):
            for line in text_lines:
                if re.search(keyword, line, re.IGNORECASE):
                    return line.strip()
            return keyword
    return ""


def _detect_series(text_lines: list[str], full_text: str) -> list[str]:
    series_keywords = [
        r"總圖書館",
        r"圖書館",
        r"變電站",
        r"電表",
        r"meter",
        r"transformer",
        r"substation",
        r"照明",
        r"lighting",
        r"HVAC",
        r"chiller",
        r"總用電",
        r"總計",
        r"合計",
        r"主變",
        r"饋線",
    ]
    found: list[str] = []
    seen: set[str] = set()
    for line in text_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(re.search(keyword, stripped, re.IGNORECASE) for keyword in series_keywords):
            if stripped not in seen:
                found.append(stripped)
                seen.add(stripped)
    if not found and full_text:
        for keyword in series_keywords:
            match = re.search(r"(.{0,20}" + keyword + r".{0,20})", full_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value not in seen:
                    found.append(value)
                    seen.add(value)
    return found


def _extract_visible_values(text: str) -> list[dict[str, Any]]:
    pattern = r"(\d[\d,]*\.?\d*)\s*(kWh|kW|kVA|kVar|MWh|MW|GWh|度|kwh|kw|KWH|KW)?"
    values: list[dict[str, Any]] = []
    for number, unit in re.findall(pattern, text, re.IGNORECASE):
        try:
            value = float(number.replace(",", ""))
        except ValueError:
            continue
        values.append({"value": value, "unit": unit or "", "raw": f"{number} {unit}".strip()})
    return values


def _detect_trend(text: str) -> str:
    has_up = bool(re.search(r"上升|增加|成長|increase|rising|higher|升高", text, re.IGNORECASE))
    has_down = bool(re.search(r"下降|減少|decrease|falling|lower|降低", text, re.IGNORECASE))
    has_stable = bool(re.search(r"穩定|平穩|stable|steady|flat|持平", text, re.IGNORECASE))
    parts = []
    if has_up:
        parts.append("upward")
    if has_down:
        parts.append("downward")
    if has_stable:
        parts.append("stable")
    return "/".join(parts) + " trend detected" if parts else ""


def _detect_issues(text: str) -> list[str]:
    patterns = [
        (r"異常|anomal|outlier", "Anomalous data points may be present"),
        (r"缺漏|missing|gap|空白|nan|null", "Missing data gaps detected"),
        (r"超載|overload|過載", "Possible overload condition detected"),
        (r"峰值|peak|最高值|max", "Peak demand values detected; verify against contract demand"),
        (r"功率因數|power.?factor|pf低", "Power factor issue may be present"),
        (r"諧波|harmonic|THD", "Harmonic distortion indicators detected"),
    ]
    return [message for pattern, message in patterns if re.search(pattern, text, re.IGNORECASE)]


def _suggest_followup_tool(question: str, detected_text: str) -> str:
    combined = (question + " " + detected_text).lower()
    if any(keyword in combined for keyword in ("csv", "視覺化", "折線", "長條", "圖表", "chart", "plot", "bar", "line")):
        return "generate_meter_chart"
    if any(keyword in combined for keyword in ("跨年", "排名", "最高", "最低", "trend", "rank", "compare", "比較", "top", "max", "min")):
        return "query_energy_records"
    return ""


def _compute_confidence(
    *,
    has_pillow: bool,
    has_ocr: bool,
    detected_text: str,
    title: str,
    unit: str,
    series: list[str],
    visible_values: list[dict[str, Any]],
) -> float:
    score = 0.1
    if has_pillow:
        score += 0.1
    if has_ocr:
        score += 0.15
    if detected_text.strip():
        score += 0.15
    if title:
        score += 0.1
    if unit:
        score += 0.1
    if series:
        score += 0.1
    if visible_values:
        score += 0.1
    if len(detected_text.strip()) > 100:
        score += 0.1
    elif len(detected_text.strip()) > 30:
        score += 0.05
    return min(max(score, 0.0), 1.0)


def analyze_meter_screenshot_impl(
    image_path: str,
    question: str = "",
    expected_domain: str = "meter_chart",
    prefer_ocr: bool = True,
    use_gemma_vision: str = "auto",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "ok",
        "image_path": image_path,
        "width": 0,
        "height": 0,
        "detected_text": "",
        "text_blocks": [],
        "detected_title": "",
        "x_axis": "",
        "y_axis": "",
        "unit": "",
        "series": [],
        "visible_values": [],
        "trend_summary": "",
        "possible_issues": [],
        "confidence": 0.0,
        "warnings": [],
        "suggested_followup_tool": "",
    }
    warnings: list[str] = []

    if not image_path or not isinstance(image_path, str):
        result.update({"status": "error", "warnings": ["image_path is empty or not a string"]})
        return result
    if not os.path.exists(image_path):
        result.update({"status": "error", "warnings": [f"File does not exist: {image_path}"]})
        return result
    if not os.path.isfile(image_path):
        result.update({"status": "error", "warnings": [f"Path is not a regular file: {image_path}"]})
        return result

    Image = _try_import_pillow()
    pytesseract = _try_import_pytesseract()
    loaded_image = None

    if Image is None:
        result["status"] = "degraded"
        warnings.append("Pillow (PIL) is not installed; image dimensions and OCR are unavailable.")
    else:
        try:
            loaded_image = Image.open(image_path)
            loaded_image.load()
            result["width"], result["height"] = loaded_image.size
        except Exception as exc:
            result["status"] = "degraded"
            warnings.append(f"Failed to open image with Pillow: {exc}")

    detected_text = ""
    if prefer_ocr and pytesseract is not None and loaded_image is not None:
        try:
            detected_text = pytesseract.image_to_string(loaded_image, lang="chi_tra+chi_sim+eng")
        except Exception as exc:
            warnings.append(f"OCR via pytesseract failed: {exc}")
    elif prefer_ocr and pytesseract is None:
        warnings.append("pytesseract is not installed; OCR skipped.")
    elif prefer_ocr and loaded_image is None:
        warnings.append("Pillow unavailable; OCR skipped.")

    text_lines = detected_text.splitlines()
    result["detected_text"] = detected_text
    result["text_blocks"] = text_lines
    result["detected_title"] = _detect_title(text_lines)
    result["x_axis"] = _detect_axis(text_lines, detected_text, ["日期", "時間", "time", "date", "year", "month", "day", "年", "月", "日"])
    result["y_axis"] = _detect_axis(text_lines, detected_text, ["kW", "kWh", "用電", "需量", "電量", "power", "energy", "demand"])
    result["unit"] = _detect_unit_from_text(detected_text)
    result["series"] = _detect_series(text_lines, detected_text)
    result["visible_values"] = _extract_visible_values(detected_text)
    result["trend_summary"] = _detect_trend(detected_text)
    result["possible_issues"] = _detect_issues(detected_text)
    result["suggested_followup_tool"] = _suggest_followup_tool(question, detected_text)
    result["confidence"] = _compute_confidence(
        has_pillow=Image is not None,
        has_ocr=pytesseract is not None and bool(detected_text.strip()),
        detected_text=detected_text,
        title=str(result["detected_title"]),
        unit=str(result["unit"]),
        series=list(result["series"]),
        visible_values=list(result["visible_values"]),
    )
    result["warnings"] = warnings
    return result
