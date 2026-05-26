from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from src.openbse_building_scaler import scale_yaml_for_building
from src.wiki_memory import WikiMemory


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_strategy_with_openbse(
    *,
    building_uid: str,
    building_name: str,
    floor_area_m2: float,
    mean_kw: float,
    b_floors: int = 1,
    strategy_params: dict[str, float],
    strategy_label: str = "",
    write_to_wiki: bool = True,
) -> dict[str, Any]:
    started_at = time.perf_counter()

    try:
        root = Path(__file__).resolve().parent.parent
        outputs_dir = root / "outputs" / "strategy_validation"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            scaled_yaml = tmp / "building_scaled.yaml"
            scale_result = scale_yaml_for_building(
                building_uid=building_uid,
                floor_area_m2=floor_area_m2,
                mean_kw=mean_kw,
                b_floors=b_floors,
                output_path=str(scaled_yaml),
            )

            from src.openbse_counterfactual import OpenBSEDeltaEngine

            try:
                engine = OpenBSEDeltaEngine()
            except Exception as exc:
                fallback_yaml = outputs_dir / f"{building_uid}_scaled.yaml"
                return {
                    "status": "unavailable",
                    "error": f"OpenBSE engine init failed: {exc}",
                    "building_uid": building_uid,
                    "scale_result": scale_result,
                    "saved_yaml": str(fallback_yaml) if scaled_yaml.exists() else None,
                    "message": "OpenBSE binary not found; scaled YAML generated but not simulated.",
                }

            cooling_delta = strategy_params.get("cooling_delta_degC", 0.0)
            lighting_ratio = strategy_params.get("lighting_ratio", 1.0)
            equipment_ratio = strategy_params.get("equipment_ratio", 1.0)
            occupancy_ratio = strategy_params.get("occupancy_ratio", 1.0)
            cop_ratio = strategy_params.get("cop_ratio", 1.0)

            try:
                breakdown = engine.compute_hvac_breakdown(
                    cooling_delta_degC=cooling_delta,
                    lighting_ratio=lighting_ratio,
                    occupancy_ratio=occupancy_ratio,
                    equipment_ratio=equipment_ratio,
                    cop_ratio=cop_ratio,
                )
            except Exception as exc:
                return {
                    "status": "simulation_failed",
                    "error": str(exc),
                    "building_uid": building_uid,
                }

        runtime_ms = int((time.perf_counter() - started_at) * 1000)

        validation = {
            "status": "ok",
            "building_uid": building_uid,
            "building_name": building_name,
            "strategy_label": strategy_label or "custom",
            "strategy_params": strategy_params,
            "scale_result": scale_result,
            "openbse_result": breakdown,
            "summary": {
                "baseline_total_kwh": breakdown.get("baseline_total_annual_kwh", 0),
                "scenario_total_kwh": breakdown.get("scenario_total_annual_kwh", 0),
                "delta_kwh": breakdown.get("total_delta_kwh", 0),
                "delta_pct": round(
                    breakdown.get("total_delta_kwh", 0)
                    / max(breakdown.get("baseline_total_annual_kwh", 1), 1)
                    * 100,
                    2,
                ),
                "cooling_delta_kwh": (breakdown.get("delta", {}).get("cooling_load_annual_kwh")),
                "hvac_cooling_delta_kwh": (breakdown.get("delta", {}).get("hvac_cooling_annual_kwh")),
                "fan_delta_kwh": (breakdown.get("delta", {}).get("fan_annual_kwh")),
                "dx_cooling_delta_kwh": (breakdown.get("delta", {}).get("dx_cooling_annual_kwh")),
            },
            "baseline_values": engine.baseline_values,
            "provenance": {
                "engine": "openbse_strategy_runner_v1",
                "runtime_ms": runtime_ms,
            },
        }

        if write_to_wiki:
            _write_validation_to_wiki(validation)

        return validation

    except Exception as exc:
        return {"status": "error", "error": str(exc), "building_uid": building_uid}


def _write_validation_to_wiki(validation: dict[str, Any]) -> None:
    try:
        mem = WikiMemory()
        bname = validation.get("building_name", validation.get("building_uid", "unknown"))
        label = validation.get("strategy_label", "custom")
        summary = validation.get("summary", {})

        title = f"OpenBSE驗證 | {bname[:20]} | {label}"

        body_lines = [
            "## OpenBSE 物理模擬驗證結果",
            "",
            f"- **建築**: {bname} ({validation.get('building_uid', '')})",
            f"- **策略**: {label}",
            f"- **參數**: {json.dumps(validation.get('strategy_params', {}), ensure_ascii=False)}",
            f"- **基準年耗電**: {summary.get('baseline_total_kwh', 0):,.0f} kWh",
            f"- **情境年耗電**: {summary.get('scenario_total_kwh', 0):,.0f} kWh",
            f"- **節電量**: {summary.get('delta_kwh', 0):,.0f} kWh ({summary.get('delta_pct', 0):.1f}%)",
            "",
            "### HVAC 逐項拆解",
            f"- 冷房負載 Δ: {summary.get('cooling_delta_kwh', 'N/A')} kWh",
            f"- HVAC 冷房 Δ: {summary.get('hvac_cooling_delta_kwh', 'N/A')} kWh",
            f"- 風機 Δ: {summary.get('fan_delta_kwh', 'N/A')} kWh",
            f"- DX 冷房 Δ: {summary.get('dx_cooling_delta_kwh', 'N/A')} kWh",
            "",
            f"**驗證時間**: {_now_iso()}",
            f"**引擎**: openbse_strategy_runner_v1",
        ]

        mem.ingest(
            title=title,
            content="\n".join(body_lines),
            kind="concept",
            tags=["strategy", "openbse-validation", bname[:20].replace(" ", "-")],
            links=[],
        )
        mem.build_graph()
    except Exception:
        pass
