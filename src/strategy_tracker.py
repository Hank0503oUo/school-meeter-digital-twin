from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.wiki_memory import WikiMemory
from src.constants import HOURS_PER_YEAR


STRATEGY_WIKI_KIND = "concept"
STRATEGY_TAG = "strategy"
STRATEGY_TRACKED_TAG = "strategy-tracked"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_strategy(
    *,
    building_name: str,
    strategy_label: str,
    params: dict[str, float],
    predicted_saving_kwh: float,
    predicted_saving_pct: float,
    regulation_refs: list[str] | None = None,
    dominant_factor: str = "",
    notes: str = "",
) -> dict[str, Any]:
    try:
        mem = WikiMemory()
        timestamp = re.sub(r"[:.+]", "-", _now_iso())
        title = f"策略 | {building_name[:20]} | {strategy_label} | {timestamp}"

        body_lines = [
            f"## 策略記錄",
            "",
            f"- **建築**: {building_name}",
            f"- **策略**: {strategy_label}",
            f"- **主要因子**: {dominant_factor}",
            f"- **參數**: {json.dumps(params, ensure_ascii=False)}",
            f"- **預測省電量**: {predicted_saving_kwh:,.0f} kWh ({predicted_saving_pct:.1f}%)",
            f"- **狀態**: 已推薦（待確認採納）",
            f"- **建立時間**: {_now_iso()}",
        ]

        if regulation_refs:
            body_lines.append("")
            body_lines.append("## 法規依據")
            for ref in regulation_refs:
                body_lines.append(f"- {ref}")

        if notes:
            body_lines.append("")
            body_lines.append(f"## 備註")
            body_lines.append(notes)

        body_lines.append("")
        body_lines.append("## 追蹤記錄")
        body_lines.append(f"- {_now_iso()} | 推薦")

        result = mem.ingest(
            title=title,
            content="\n".join(body_lines),
            kind=STRATEGY_WIKI_KIND,
            tags=[STRATEGY_TAG, building_name[:20].replace(" ", "-"), dominant_factor],
        )
        mem.build_graph()
        return {"status": "ok", "action": "recorded", "slug": result["slug"], "title": title}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def confirm_strategy(
    *,
    building_name: str,
    strategy_label: str,
) -> dict[str, Any]:
    try:
        mem = WikiMemory()
        hits = mem.query(f"策略 {building_name} {strategy_label}", kind=STRATEGY_WIKI_KIND, limit=5)
        if not hits:
            return {"status": "not_found", "message": f"找不到 {building_name} 的策略「{strategy_label}」記錄"}

        slug = hits[0]["slug"]
        kind_dir = mem._kind_dir(STRATEGY_WIKI_KIND)
        page_path = kind_dir / f"{slug}.md"

        if not page_path.exists():
            return {"status": "not_found", "message": f"Wiki 頁面不存在: {slug}"}

        body = page_path.read_text(encoding="utf-8")
        tracking_line = f"- {_now_iso()} | ✅ 已確認採納"
        body = body.replace("已推薦（待確認採納）", "✅ 已採納")
        body = body.rstrip() + "\n" + tracking_line + "\n"

        page_path.write_text(body, encoding="utf-8")
        mem.rebuild_index()
        return {"status": "ok", "action": "confirmed", "slug": slug, "building": building_name}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def check_strategy_adoption(
    *,
    building_name: str,
) -> dict[str, Any]:
    try:
        mem = WikiMemory()
        hits = mem.query(f"策略 {building_name}", kind=STRATEGY_WIKI_KIND, limit=10)

        strategies: list[dict[str, Any]] = []
        for hit in hits:
            excerpt = hit.get("excerpt", "")
            slug = hit["slug"]
            adopted = "✅ 已採納" in excerpt or "已確認採納" in excerpt
            strategies.append({
                "slug": slug,
                "summary": hit["summary"],
                "adopted": adopted,
                "status": "已採納" if adopted else "待確認",
            })

        adopted_count = sum(1 for s in strategies if s["adopted"])
        pending_count = len(strategies) - adopted_count

        return {
            "status": "ok",
            "building": building_name,
            "total_strategies": len(strategies),
            "adopted": adopted_count,
            "pending": pending_count,
            "strategies": strategies,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def compare_actual_vs_predicted(
    *,
    building_name: str,
    actual_mean_kw: float | None = None,
) -> dict[str, Any]:
    try:
        mem = WikiMemory()
        hits = mem.query(f"策略 {building_name}", kind=STRATEGY_WIKI_KIND, limit=10)

        comparisons: list[dict[str, Any]] = []
        for hit in hits:
            excerpt = hit.get("excerpt", "")
            if "✅ 已採納" not in excerpt:
                continue

            predicted_kwh = 0.0
            kwh_match = re.search(r"預測省電量\*?:\s*([\d,]+(?:\.\d+)?)\s*kWh", excerpt)
            if kwh_match:
                predicted_kwh = float(kwh_match.group(1).replace(",", ""))

            pct_match = re.search(r"\(([\d.]+)%\)", excerpt)
            predicted_pct = float(pct_match.group(1)) if pct_match else 0.0

            param_match = re.search(r"參數:\s*(\{[^}]+\})", excerpt)
            params = {}
            if param_match:
                try:
                    params = json.loads(param_match.group(1))
                except json.JSONDecodeError:
                    pass

            comparisons.append({
                "slug": hit["slug"],
                "summary": hit["summary"],
                "predicted_saving_kwh": predicted_kwh,
                "predicted_saving_pct": predicted_pct,
                "params": params,
            })

        if actual_mean_kw is not None and actual_mean_kw > 0:
            v12_df = _load_v12_for_building(building_name)
            baseline_kw = v12_df if v12_df > 0 else actual_mean_kw * 1.1
            baseline_kwh = baseline_kw * HOURS_PER_YEAR
            actual_kwh = actual_mean_kw * HOURS_PER_YEAR
            actual_delta_kwh = baseline_kwh - actual_kwh
            actual_delta_pct = (actual_delta_kwh / baseline_kwh * 100) if baseline_kwh > 0 else 0.0

            return {
                "status": "ok",
                "building": building_name,
                "baseline_mean_kw": round(baseline_kw, 2),
                "actual_mean_kw": round(actual_mean_kw, 2),
                "actual_delta_kwh": round(actual_delta_kwh, 0),
                "actual_delta_pct": round(actual_delta_pct, 1),
                "adopted_strategies": comparisons,
                "comparison": {
                    "total_predicted_kwh": sum(c["predicted_saving_kwh"] for c in comparisons),
                    "actual_observed_kwh": round(actual_delta_kwh, 0),
                    "accuracy_pct": round(
                        min(actual_delta_kwh, sum(c["predicted_saving_kwh"] for c in comparisons))
                        / max(sum(c["predicted_saving_kwh"] for c in comparisons), 1) * 100,
                        1,
                    ) if comparisons else 0.0,
                },
            }

        return {
            "status": "ok",
            "building": building_name,
            "adopted_strategies": comparisons,
            "note": "Provide actual_mean_kw to compare actual vs predicted savings.",
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _load_v12_for_building(building_name: str) -> float:
    import pandas as pd

    root = Path(__file__).resolve().parent.parent
    for candidate in (
        root / "campuses" / "ntu" / "models" / "v12_per_building_summary.csv",
        root / "models" / "v12_per_building_summary.csv",
    ):
        if candidate.exists():
            df = pd.read_csv(candidate, encoding="utf-8")
            name_lower = building_name.lower()
            for _, row in df.iterrows():
                mn = str(row.get("meter_name", "")).lower()
                if name_lower in mn or mn in name_lower:
                    return float(row.get("mean_kw", 0) or 0)
    return 0.0
