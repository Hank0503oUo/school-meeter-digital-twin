from __future__ import annotations

from typing import Any, Sequence

from src.counterfactual import run_counterfactual
from src.knowledge_base import KnowledgeWorkbench


def search_docs(
    workbench: KnowledgeWorkbench,
    *,
    query: str,
    building_id: str,
    selected_docs: Sequence[str],
    selected_csvs: Sequence[str],
    top_k: int = 6,
) -> list[dict[str, Any]]:
    return workbench.search_chunks(
        query=query,
        building_id=building_id,
        selected_docs=selected_docs,
        selected_csvs=selected_csvs,
        top_k=top_k,
    )


def fetch_chunk(chunks: Sequence[dict[str, Any]], chunk_id: str) -> dict[str, Any] | None:
    for chunk in chunks:
        if str(chunk.get("chunk_id", "")) == chunk_id:
            return dict(chunk)
    return None


def lookup_building_entity(workbench: KnowledgeWorkbench, building_id: str) -> dict[str, Any]:
    return workbench.get_ontology(building_id)


def query_meter_or_kpi(workbench: KnowledgeWorkbench, selected_csvs: Sequence[str]) -> dict[str, Any]:
    return workbench.describe_csvs(selected_csvs)


def estimate_counterfactual_savings(csv_summary: dict[str, Any]) -> dict[str, Any] | None:
    for summary in csv_summary.values():
        stats = summary.get("stats", {})
        for metric_name, values in stats.items():
            mean_value = values.get("mean")
            if mean_value is None:
                continue
            try:
                result = run_counterfactual(
                    float(mean_value),
                    cooling_delta_degC=1.0,
                    lighting_ratio=0.95,
                    occupancy_ratio=0.98,
                    equipment_ratio=0.98,
                )
            except Exception:
                continue
            return {
                "metric": metric_name,
                "baseline_mean": float(mean_value),
                "delta_kwh": round(float(result.delta_kwh), 4),
                "delta_pct": round(float(result.delta_pct), 4),
                "delta_ntd": round(float(result.delta_ntd), 4),
                "delta_carbon_kg": round(float(result.delta_carbon_kg), 4),
            }
    return None
