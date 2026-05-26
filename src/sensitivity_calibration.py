from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.counterfactual import SensitivityCoefficients
from src.constants import HOURS_PER_YEAR
from src.wiki_memory import WikiMemory


_CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "config" / "sensitivity_calibration.json"

_DEFAULT_CALIBRATION = {
    "version": 1,
    "updated_at": "",
    "calibration_count": 0,
    "coefficients": {
        "cooling_pct_per_degC": -0.030,
        "lighting_fraction": 0.15,
        "occupancy_fraction": 0.08,
        "equipment_fraction": 0.35,
    },
    "history": [],
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_calibrated_coefficients() -> SensitivityCoefficients:
    cal = _load_calibration()
    c = cal["coefficients"]
    return SensitivityCoefficients(
        cooling_pct_per_degC=c["cooling_pct_per_degC"],
        lighting_fraction=c["lighting_fraction"],
        occupancy_fraction=c["occupancy_fraction"],
        equipment_fraction=c["equipment_fraction"],
    )


def _load_calibration() -> dict[str, Any]:
    if _CALIBRATION_FILE.exists():
        try:
            data = json.loads(_CALIBRATION_FILE.read_text(encoding="utf-8"))
            if "coefficients" in data:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return dict(_DEFAULT_CALIBRATION)


def _save_calibration(cal: dict[str, Any]) -> None:
    _CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CALIBRATION_FILE.write_text(
        json.dumps(cal, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def calibrate_from_feedback(
    *,
    building_name: str,
    predicted_delta_kwh: float,
    actual_delta_kwh: float,
    dominant_factor: str,
    notes: str = "",
) -> dict[str, Any]:
    if predicted_delta_kwh == 0:
        return {"status": "error", "error": "predicted_delta_kwh cannot be zero"}

    error_ratio = actual_delta_kwh / predicted_delta_kwh

    cal = _load_calibration()
    old_coefficients = dict(cal["coefficients"])

    factor_to_key = {
        "cooling_load": "cooling_pct_per_degC",
        "lighting_load": "lighting_fraction",
        "equipment_load": "equipment_fraction",
        "occupancy_load": "occupancy_fraction",
        "operational_variability": "cooling_pct_per_degC",
        "weather_driven_load": "cooling_pct_per_degC",
    }

    target_key = factor_to_key.get(dominant_factor)
    if not target_key:
        return {
            "status": "skipped",
            "reason": f"unknown dominant_factor '{dominant_factor}'",
            "error_ratio": round(error_ratio, 4),
        }

    old_value = cal["coefficients"][target_key]
    correction = 1.0 + (error_ratio - 1.0) * 0.3
    new_value = old_value * correction

    if target_key == "cooling_pct_per_degC":
        sign = -1 if old_value < 0 else 1
        new_value = sign * max(0.005, min(0.10, abs(new_value)))
    else:
        new_value = max(0.01, min(0.60, new_value))

    cal["coefficients"][target_key] = round(new_value, 6)
    cal["updated_at"] = _now_iso()
    cal["calibration_count"] = cal.get("calibration_count", 0) + 1

    entry = {
        "timestamp": _now_iso(),
        "building": building_name,
        "dominant_factor": dominant_factor,
        "coefficient_key": target_key,
        "old_value": old_value,
        "new_value": round(new_value, 6),
        "predicted_kwh": round(predicted_delta_kwh, 0),
        "actual_kwh": round(actual_delta_kwh, 0),
        "error_ratio": round(error_ratio, 4),
        "correction_factor": round(correction, 4),
        "notes": notes,
    }
    history = cal.get("history", [])
    history.append(entry)
    cal["history"] = history[-50:]

    _save_calibration(cal)
    _write_calibration_to_wiki(entry, old_coefficients, cal["coefficients"])

    return {
        "status": "ok",
        "building": building_name,
        "dominant_factor": dominant_factor,
        "coefficient_key": target_key,
        "old_value": old_value,
        "new_value": round(new_value, 6),
        "predicted_kwh": round(predicted_delta_kwh, 0),
        "actual_kwh": round(actual_delta_kwh, 0),
        "error_ratio": round(error_ratio, 4),
        "correction_factor": round(correction, 4),
        "all_coefficients": cal["coefficients"],
        "calibration_count": cal["calibration_count"],
    }


def get_calibration_status() -> dict[str, Any]:
    cal = _load_calibration()
    return {
        "status": "ok",
        "version": cal.get("version", 0),
        "updated_at": cal.get("updated_at", "never"),
        "calibration_count": cal.get("calibration_count", 0),
        "current_coefficients": cal["coefficients"],
        "recent_history": cal.get("history", [])[-5:],
    }


def _write_calibration_to_wiki(
    entry: dict[str, Any],
    old_coefficients: dict[str, float],
    new_coefficients: dict[str, float],
) -> None:
    try:
        mem = WikiMemory()
        title = f"校準記錄 | {entry['building'][:15]} | {entry['coefficient_key']}"

        body_lines = [
            "## 敏感度係數校準",
            "",
            f"- **建築**: {entry['building']}",
            f"- **主因子**: {entry['dominant_factor']}",
            f"- **校準係數**: {entry['coefficient_key']}",
            f"- **舊值**: {entry['old_value']}",
            f"- **新值**: {entry['new_value']}",
            f"- **預測省電**: {entry['predicted_kwh']:,.0f} kWh",
            f"- **實際省電**: {entry['actual_kwh']:,.0f} kWh",
            f"- **誤差比**: {entry['error_ratio']:.4f}",
            "",
            "### 校準前係數",
        ]
        for k, v in old_coefficients.items():
            body_lines.append(f"- `{k}`: {v}")
        body_lines.append("")
        body_lines.append("### 校準後係數")
        for k, v in new_coefficients.items():
            body_lines.append(f"- `{k}`: {v}")

        mem.ingest(
            title=title,
            content="\n".join(body_lines),
            kind="concept",
            tags=["calibration", "sensitivity", entry["building"][:15].replace(" ", "-")],
        )
        mem.build_graph()
    except Exception:
        pass
