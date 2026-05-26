from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_DIR = _ROOT / "outputs" / "energy_manager"
_RTEM_BMS_DIR = Path(
    r"C:\Users\User\Downloads\build llm\drive-download-20260320T153715Z-3-001"
    r"\RTEM dataset\BMS data"
)
_RTEM_META_PATH = Path(
    r"C:\Users\User\Downloads\build llm\drive-download-20260320T153715Z-3-001"
    r"\RTEM dataset\meta data\all_points_metadata.csv"
)


_ANOMALY_PATTERNS = ("spike", "drift", "zero", "oscillation", "step", "stuck", "noise")


_PATTERN_DESCRIPTIONS: dict[str, str] = {
    "spike": "Sudden large deviation from local baseline, followed by quick recovery.",
    "drift": "Gradual monotonic trend away from baseline over a sustained window.",
    "zero": "Value drops to or near zero unexpectedly (sensor fault or equipment off).",
    "oscillation": "Rapid alternating high-low values around the baseline (control hunting).",
    "step": "Abrupt sustained shift to a new level without recovery (setpoint change or fault).",
    "stuck": "Value remains constant for an abnormally long period (sensor stuck or comm loss).",
    "noise": "Excessive high-frequency variance beyond normal operating band.",
}

_SEVERITY_RULES: dict[str, tuple[float, str]] = {
    "spike": (3.0, "high"),
    "drift": (5.0, "medium"),
    "zero": (0.5, "critical"),
    "oscillation": (4.0, "medium"),
    "step": (4.0, "high"),
    "stuck": (0.0, "high"),
    "noise": (3.0, "low"),
}


def _load_rtem_metadata() -> pd.DataFrame:
    if _RTEM_META_PATH.is_file():
        return pd.read_csv(_RTEM_META_PATH, encoding="utf-8")
    return pd.DataFrame()


def _list_building_subsystems(building_id: int | str) -> list[dict[str, Any]]:
    bid = str(building_id)
    prefix = f"rtem_API_data_{bid}_"
    results: list[dict[str, Any]] = []
    if not _RTEM_BMS_DIR.is_dir():
        return results
    for f in sorted(_RTEM_BMS_DIR.iterdir()):
        if f.name.startswith(prefix) and f.name.endswith(".csv.gzip"):
            tag = f.name[len(prefix):].replace(".csv.gzip", "")
            results.append({"building_id": int(bid), "equip_tag": tag, "file": str(f)})
    return results


def _load_rtem_series(file_path: str, point_id: str | int | None = None) -> pd.DataFrame:
    path = Path(file_path)
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path, compression="gzip")
    if "timestamp" not in df.columns:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if point_id is not None:
        pid = str(point_id)
        if pid in df.columns:
            return df[["timestamp", pid]].copy()
    return df


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _infer_timestamp_column(df: pd.DataFrame, requested: str = "") -> str:
    if requested and requested in df.columns:
        return requested
    lower = {str(col).lower(): str(col) for col in df.columns}
    for name in ("timestamp", "datetime", "date", "time", "ts"):
        if name in lower:
            return lower[name]
    for col in df.columns:
        if "time" in str(col).lower() or "date" in str(col).lower():
            return str(col)
    return ""


def _infer_value_column(df: pd.DataFrame, requested: str = "") -> str:
    if requested and requested in df.columns:
        return requested
    preferred = ("kw", "mean_kw", "value", "demand_kw", "power", "reading")
    lookup = {str(col).strip().lower(): str(col) for col in df.columns}
    for name in preferred:
        if name in lookup and pd.to_numeric(df[lookup[name]], errors="coerce").notna().any():
            return lookup[name]
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() >= max(3, min(10, len(df) // 4)):
            return str(col)
    return ""


def _filter_optional(df: pd.DataFrame, column_names: tuple[str, ...], value: str) -> pd.DataFrame:
    if not value:
        return df
    value_norm = str(value).strip().lower()
    for column in column_names:
        if column in df.columns:
            mask = df[column].fillna("").astype(str).str.strip().str.lower()
            return df.loc[mask.eq(value_norm) | mask.str.contains(value_norm, regex=False)].copy()
    return df


def classify_anomaly_pattern(
    values: list[float] | np.ndarray | pd.Series,
    timestamps: list[str] | None = None,
    baseline_window: int = 12,
) -> dict[str, Any]:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    n = len(series)
    if n < 5:
        return {"pattern": "insufficient_data", "confidence": 0.0, "detail": f"Only {n} valid points, need >= 5."}
    arr = series.values.astype(float)
    global_median = float(np.median(arr))
    global_std = float(np.std(arr, ddof=0))
    if global_std <= 0:
        return {"pattern": "constant", "confidence": 1.0, "detail": "Series is effectively constant."}
    candidates: dict[str, float] = {}
    details: dict[str, str] = {}
    diff = np.diff(arr)
    abs_diff = np.abs(diff)
    rel_diff = abs_diff / (global_std + 1e-9)
    spike_score = float(np.sum(rel_diff > 3.0)) / max(1, n - 1)
    if spike_score > 0:
        candidates["spike"] = min(1.0, spike_score * 5)
        idx = int(np.argmax(rel_diff))
        details["spike"] = f"Largest jump: {arr[idx]:.4f} -> {arr[idx+1]:.4f} ({rel_diff[idx]:.1f}σ)"
    if n >= baseline_window * 2:
        first_half = np.median(arr[: n // 2])
        second_half = np.median(arr[n // 2 :])
        drift_sigma = abs(second_half - first_half) / (global_std + 1e-9)
        if drift_sigma > 1.5:
            candidates["drift"] = min(1.0, drift_sigma / 5.0)
            details["drift"] = f"Median shift: {first_half:.4f} -> {second_half:.4f} ({drift_sigma:.1f}σ)"
    near_zero_count = int(np.sum(np.abs(arr) < global_std * 0.01))
    zero_ratio = near_zero_count / n
    if zero_ratio > 0.3:
        candidates["zero"] = min(1.0, zero_ratio)
        details["zero"] = f"{near_zero_count}/{n} points near zero ({zero_ratio:.0%})"
    if n >= 8:
        signs = np.sign(diff[diff != 0])
        if len(signs) >= 4:
            sign_changes = int(np.sum(np.diff(signs) != 0))
            oscillation_ratio = sign_changes / max(1, len(signs) - 1)
            if oscillation_ratio > 0.6:
                candidates["oscillation"] = min(1.0, oscillation_ratio)
                details["oscillation"] = f"Sign changes: {sign_changes}/{len(signs)-1} ({oscillation_ratio:.0%})"
    if n >= baseline_window:
        rolling_med = pd.Series(arr).rolling(baseline_window, center=True, min_periods=3).median().values
        valid_mask = ~np.isnan(rolling_med)
        if valid_mask.sum() > baseline_window:
            residuals = arr[valid_mask] - rolling_med[valid_mask]
            step_jumps = np.abs(np.diff(rolling_med[valid_mask]))
            max_step = float(np.max(step_jumps)) if len(step_jumps) > 0 else 0.0
            step_sigma = max_step / (global_std + 1e-9)
            if step_sigma > 2.5:
                candidates["step"] = min(1.0, step_sigma / 5.0)
                details["step"] = f"Max rolling-median jump: {max_step:.4f} ({step_sigma:.1f}σ)"
    unique_count = len(np.unique(arr))
    unique_ratio = unique_count / n
    if unique_ratio < 0.02 and n > 20:
        candidates["stuck"] = min(1.0, (0.02 - unique_ratio) * 50)
        details["stuck"] = f"Only {unique_count} unique values in {n} samples ({unique_ratio:.3%})"
    local_std = float(np.median(pd.Series(arr).rolling(min(5, n), min_periods=3).std().dropna()))
    noise_ratio = local_std / (global_std + 1e-9)
    if noise_ratio > 1.5:
        candidates["noise"] = min(1.0, noise_ratio / 3.0)
        details["noise"] = f"Local/Global std ratio: {noise_ratio:.2f}"
    if not candidates:
        return {"pattern": "normal", "confidence": 0.9, "detail": "No anomaly pattern detected."}
    best = max(candidates, key=lambda k: candidates[k])
    confidence = candidates[best]
    threshold, severity = _SEVERITY_RULES.get(best, (3.0, "medium"))
    return {
        "pattern": best,
        "confidence": round(confidence, 3),
        "severity": severity,
        "description": _PATTERN_DESCRIPTIONS.get(best, ""),
        "detail": details.get(best, ""),
        "all_candidates": {k: round(v, 3) for k, v in sorted(candidates.items(), key=lambda x: -x[1])},
        "stats": {
            "n_points": n,
            "median": round(global_median, 4),
            "std": round(global_std, 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        },
    }


def detect_energy_anomalies_impl(
    csv_path: str = "",
    building_uid: str = "",
    meter_name: str = "",
    value_column: str = "",
    timestamp_column: str = "",
    window: int = 24,
    z_threshold: float = 3.0,
    max_points: int = 20,
) -> dict[str, Any]:
    path = Path(str(csv_path or "")).expanduser()
    if not csv_path or not path.is_file():
        return {"status": "failed", "error": f"CSV file not found: {path}", "warnings": []}

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return {"status": "failed", "error": f"CSV file could not be read: {exc}", "warnings": []}
    if df.empty:
        return {"status": "failed", "error": "CSV has no rows.", "warnings": []}

    df = _filter_optional(df, ("building_uid", "uid", "building_id", "building"), building_uid)
    df = _filter_optional(df, ("meter_name", "meter", "meter_id"), meter_name)
    if df.empty:
        return {"status": "failed", "error": "No rows matched the requested building_uid/meter_name.", "warnings": []}

    ts_col = _infer_timestamp_column(df, timestamp_column)
    val_col = _infer_value_column(df, value_column)
    if not val_col:
        return {"status": "failed", "error": "Could not infer a numeric value column.", "warnings": []}

    values = pd.to_numeric(df[val_col], errors="coerce")
    valid = df.loc[values.notna()].copy()
    valid["_energy_value"] = values.loc[values.notna()].astype(float)
    if ts_col:
        valid["_energy_timestamp"] = valid[ts_col].astype(str)
    else:
        valid["_energy_timestamp"] = valid.index.astype(str)

    if len(valid) < 3:
        return {"status": "failed", "error": "At least 3 numeric readings are required.", "warnings": []}

    window_size = max(3, min(int(window or 24), len(valid)))
    center = valid["_energy_value"].rolling(window=window_size, min_periods=3, center=True).median()
    residual = (valid["_energy_value"] - center).abs()
    mad = residual.rolling(window=window_size, min_periods=3, center=True).median()
    global_median = float(valid["_energy_value"].median())
    global_mad = float((valid["_energy_value"] - global_median).abs().median())
    fallback_scale = global_mad if global_mad > 0 else float(valid["_energy_value"].std(ddof=0) or 0.0)
    if fallback_scale <= 0:
        return {
            "status": "ok",
            "input": {"csv_path": str(path), "value_column": val_col, "timestamp_column": ts_col},
            "summary": {"row_count": int(len(valid)), "anomaly_count": 0},
            "anomalies": [],
            "warnings": ["All numeric readings are effectively constant; no anomaly score can be computed."],
            "evidence_level": "low",
            "suggested_actions": ["Verify the meter export period and whether values were pre-aggregated."],
        }

    scale = mad.mask(mad <= 0).fillna(fallback_scale)
    score = 0.6745 * (valid["_energy_value"] - center.fillna(global_median)).abs() / scale
    valid["_anomaly_score"] = score.fillna(0.0)
    threshold = max(0.1, float(z_threshold or 3.0))
    anomalies = valid.loc[valid["_anomaly_score"] >= threshold].copy()
    anomalies = anomalies.sort_values("_anomaly_score", ascending=False).head(max(1, int(max_points or 20)))

    anomaly_records: list[dict[str, Any]] = []
    for _, row in anomalies.iterrows():
        anomaly_records.append(
            {
                "timestamp": _json_safe(row.get("_energy_timestamp")),
                "value": _json_safe(row.get("_energy_value")),
                "score": round(float(row.get("_anomaly_score", 0.0)), 3),
                "direction": "high" if float(row.get("_energy_value", 0.0)) >= global_median else "low",
            }
        )

    return {
        "status": "ok",
        "input": {
            "csv_path": str(path),
            "building_uid": building_uid,
            "meter_name": meter_name,
            "value_column": val_col,
            "timestamp_column": ts_col,
            "window": window_size,
            "z_threshold": threshold,
        },
        "summary": {
            "row_count": int(len(valid)),
            "anomaly_count": int(len(anomaly_records)),
            "median": float(global_median),
            "max": float(valid["_energy_value"].max()),
            "min": float(valid["_energy_value"].min()),
        },
        "anomalies": anomaly_records,
        "warnings": [] if anomaly_records else ["No anomalies exceeded the configured threshold."],
        "evidence_level": "medium" if anomaly_records else "low",
        "suggested_actions": [
            "Check whether the anomaly aligns with schedule, weather, or occupancy changes.",
            "Compare the same meter against adjacent days or similar buildings.",
            "After human review, store reusable patterns with store_energy_memory_pattern and keep LOG.md as an audit trail.",
        ],
    }


def cross_sensor_diagnosis(
    building_id: int | str,
    subsystems: list[str] | str | None = None,
    window_hours: int = 24,
    correlation_threshold: float = 0.3,
) -> dict[str, Any]:
    bid = int(building_id)
    subs = [subsystems] if isinstance(subsystems, str) else list(subsystems or [])
    available = _list_building_subsystems(bid)
    if not available:
        return {"status": "error", "error": f"No BMS data found for building {bid}.", "building_id": bid}
    if subs:
        available = [s for s in available if s["equip_tag"] in subs]
    if not available:
        return {"status": "error", "error": f"No matching subsystems for building {bid}.", "building_id": bid}
    loaded: dict[str, pd.DataFrame] = {}
    for entry in available:
        tag = entry["equip_tag"]
        try:
            df = _load_rtem_series(entry["file"])
            if df.empty or len(df.columns) < 2:
                continue
            loaded[tag] = df
        except Exception:
            continue
    if not loaded:
        return {"status": "error", "error": "All subsystem files failed to load.", "building_id": bid}
    findings: list[dict[str, Any]] = []
    per_subsystem: dict[str, dict[str, Any]] = {}
    for tag, df in loaded.items():
        numeric_cols = [c for c in df.columns if c != "timestamp"]
        point_anomalies: list[dict[str, Any]] = []
        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) < 10:
                continue
            recent = series.tail(min(len(series), window_hours * 4))
            result = classify_anomaly_pattern(recent.values)
            if result["pattern"] not in ("normal", "constant", "insufficient_data"):
                point_anomalies.append({
                    "point_id": col,
                    "pattern": result["pattern"],
                    "confidence": result["confidence"],
                    "severity": result.get("severity", "medium"),
                    "detail": result.get("detail", ""),
                })
        per_subsystem[tag] = {
            "point_count": len(numeric_cols),
            "anomaly_count": len(point_anomalies),
            "top_anomalies": point_anomalies[:5],
        }
        for pa in point_anomalies[:3]:
            findings.append({
                "subsystem": tag,
                "point_id": pa["point_id"],
                "pattern": pa["pattern"],
                "severity": pa["severity"],
                "confidence": pa["confidence"],
                "detail": pa["detail"],
            })
    cross_findings: list[dict[str, Any]] = []
    tags_list = list(loaded.keys())
    if len(tags_list) >= 2:
        for i in range(len(tags_list)):
            for j in range(i + 1, len(tags_list)):
                t1, t2 = tags_list[i], tags_list[j]
                df1, df2 = loaded[t1], loaded[t2]
                merged = pd.merge(df1, df2, on="timestamp", how="inner", suffixes=(f"_{t1}", f"_{t2}"))
                if len(merged) < 20:
                    continue
                cols1 = [c for c in merged.columns if c.endswith(f"_{t1}")]
                cols2 = [c for c in merged.columns if c.endswith(f"_{t2}")]
                for c1 in cols1[:3]:
                    s1 = pd.to_numeric(merged[c1], errors="coerce").dropna()
                    if s1.std() == 0:
                        continue
                    for c2 in cols2[:3]:
                        s2 = pd.to_numeric(merged[c2], errors="coerce").dropna()
                        if s2.std() == 0:
                            continue
                        common = merged[[c1, c2]].dropna()
                        if len(common) < 20:
                            continue
                        corr = float(common[c1].corr(common[c2]))
                        if abs(corr) > correlation_threshold:
                            cross_findings.append({
                                "subsystem_pair": f"{t1} <-> {t2}",
                                "point_a": c1,
                                "point_b": c2,
                                "correlation": round(corr, 3),
                                "direction": "positive" if corr > 0 else "negative",
                            })
    return {
        "status": "ok",
        "building_id": bid,
        "subsystems_scanned": list(loaded.keys()),
        "per_subsystem_summary": per_subsystem,
        "top_anomalies": findings[:15],
        "cross_correlations": cross_findings[:10],
        "total_anomaly_points": sum(v["anomaly_count"] for v in per_subsystem.values()),
        "total_cross_findings": len(cross_findings),
    }


def diagnose_energy_anomaly(
    building_id: int | str = 0,
    csv_path: str = "",
    point_id: str | int = "",
    subsystem: str = "",
    window_hours: int = 168,
) -> dict[str, Any]:
    bid = int(building_id) if building_id else 0
    results: dict[str, Any] = {"status": "ok", "building_id": bid}
    if csv_path:
        path = Path(csv_path).expanduser()
        if path.is_file():
            try:
                df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_csv(path, compression="gzip")
                results["single_point"] = _diagnose_single_csv(df, point_id=str(point_id), window_hours=window_hours)
            except Exception as exc:
                results["single_point"] = {"status": "error", "error": str(exc)}
    if bid:
        cross = cross_sensor_diagnosis(bid, subsystems=[subsystem] if subsystem else None)
        results["cross_sensor"] = cross
    if not csv_path and not bid:
        return {"status": "error", "error": "Provide either csv_path or building_id.", "building_id": bid}
    all_findings: list[dict[str, Any]] = []
    sp = results.get("single_point", {})
    if sp.get("pattern") and sp["pattern"] not in ("normal", "constant", "insufficient_data"):
        all_findings.append({"source": "single_point", **sp})
    for a in (results.get("cross_sensor", {}).get("top_anomalies") or []):
        all_findings.append({"source": "cross_sensor", **a})
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    all_findings.sort(key=lambda f: severity_order.get(f.get("severity", "medium"), 2))
    results["diagnosis"] = {
        "total_findings": len(all_findings),
        "findings": all_findings[:10],
        "summary": _build_diagnosis_summary(all_findings),
    }
    return results


def _diagnose_single_csv(
    df: pd.DataFrame, point_id: str = "", window_hours: int = 168,
) -> dict[str, Any]:
    ts_col = _infer_timestamp_column(df)
    val_col = str(point_id).strip() if point_id else ""
    if val_col and val_col not in df.columns:
        val_col = ""
    if not val_col:
        numeric_cols = [c for c in df.columns if c != ts_col and pd.to_numeric(df[c], errors="coerce").notna().sum() > 5]
        if not numeric_cols:
            return {"status": "error", "error": "No numeric columns found."}
        val_col = numeric_cols[0]
    series = pd.to_numeric(df[val_col], errors="coerce").dropna()
    if len(series) == 0:
        return {"status": "error", "error": f"No valid data in column {val_col}."}
    tail_n = min(len(series), window_hours * 4)
    recent = series.tail(tail_n)
    classification = classify_anomaly_pattern(recent.values)
    ts_series = df[ts_col].astype(str) if ts_col else pd.Series(range(len(df))).astype(str)
    ts_recent = ts_series.iloc[recent.index].tolist() if len(ts_series) == len(df) else []
    return {
        "status": "ok",
        "point_id": val_col,
        "n_points": int(len(recent)),
        "pattern": classification["pattern"],
        "confidence": classification.get("confidence", 0),
        "severity": classification.get("severity", "unknown"),
        "description": classification.get("description", ""),
        "detail": classification.get("detail", ""),
        "stats": classification.get("stats", {}),
        "all_candidates": classification.get("all_candidates", {}),
        "sample_timestamps": ts_recent[:3] if ts_recent else [],
    }


def _build_diagnosis_summary(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "No anomalies detected across scanned data sources."
    patterns = [f.get("pattern", "?") for f in findings]
    from collections import Counter
    counts = Counter(patterns)
    parts = [f"{k}: {v}" for k, v in counts.most_common(4)]
    top = findings[0].get("severity", "medium")
    return f"Top severity: {top}. Patterns found: {', '.join(parts)}."


def append_energy_log_impl(
    event_type: str,
    title: str,
    summary: str,
    building_uid: str = "",
    meter_name: str = "",
    severity: str = "info",
    evidence: dict[str, Any] | None = None,
    decisions: list[str] | str | None = None,
    log_path: str = "",
) -> dict[str, Any]:
    """Append a human-readable audit entry.

    Harness memory remains the long-term RAG store. This LOG.md file is for
    delivery review, operator audit trails, and report provenance.
    """
    output = Path(log_path).expanduser() if log_path else _DEFAULT_OUTPUT_DIR / "LOG.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry_id = "energy-" + timestamp.replace(":", "").replace("+", "z")
    decision_items = [decisions] if isinstance(decisions, str) else list(decisions or [])
    evidence_payload = evidence or {}

    block = [
        f"\n## {timestamp} [{str(severity or 'info').upper()}] {title or event_type or 'Energy event'}",
        f"- entry_id: `{entry_id}`",
        "- memory_role: `audit_log_only`",
        "- long_term_memory: `harness_memory_mcp`",
        f"- event_type: `{event_type or 'unspecified'}`",
        f"- building_uid: `{building_uid or 'unspecified'}`",
        f"- meter_name: `{meter_name or 'unspecified'}`",
        "",
        str(summary or "").strip() or "No summary provided.",
        "",
        "```json",
        json.dumps(evidence_payload, ensure_ascii=False, separators=(",", ":"), default=str),
        "```",
    ]
    if decision_items:
        block.extend(["", "Decisions:"])
        block.extend(f"- {item}" for item in decision_items)
    block.append("")
    with output.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(block))

    return {"status": "ok", "path": str(output), "entry_id": entry_id, "memory_role": "audit_log_only"}


def generate_energy_saving_report_impl(
    anomaly_result: dict[str, Any] | None = None,
    building_context: dict[str, Any] | None = None,
    report_title: str = "",
    output_path: str = "",
) -> dict[str, Any]:
    output = Path(output_path).expanduser() if output_path else (
        _DEFAULT_OUTPUT_DIR / f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    anomaly = anomaly_result or {}
    context = building_context or {}
    title = report_title or "Energy Saving Report"
    anomalies = list(anomaly.get("anomalies") or [])

    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Findings",
        f"- Status: {anomaly.get('status', 'not_provided')}",
        f"- Summary: {anomaly.get('summary', 'No anomaly result provided.')}",
        f"- Anomaly count: {len(anomalies)}",
        "",
        "## Likely Causes",
        "- Schedule, occupancy, weather, meter rollover, or equipment operation should be checked before action.",
        "- Treat this report as evidence packaging, not an automatic control command.",
        "",
        "## Suggested Next Actions",
    ]
    for action in anomaly.get("suggested_actions") or ["Review meter trend and compare against peer buildings."]:
        lines.append(f"- {action}")
    if context:
        lines.extend(["", "## Building Context", "```json", json.dumps(context, ensure_ascii=False, indent=2, default=str), "```"])
    if anomalies:
        lines.extend(["", "## Top Anomalies", "```json", json.dumps(anomalies[:20], ensure_ascii=False, indent=2, default=str), "```"])
    lines.extend(
        [
            "",
        "## Data Limitations",
        "- Results depend on CSV column quality, timestamp coverage, and meter metadata.",
        "- Weather, class schedules, and maintenance logs are not automatically joined unless provided by other MCP tools.",
        "- Long-term reusable memory should be stored in harness memory MCP; this report is not the memory store.",
        "",
    ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return {"status": "ok", "path": str(output), "anomaly_count": len(anomalies)}


_ANOMALY_PATTERN_RULES: list[dict[str, Any]] = [
    {
        "pattern": "spike",
        "label_en": "Spike",
        "label_zh": "突波",
        "description": "Single-point extreme deviation from local baseline, rapid return.",
        "test": lambda ctx: ctx.get("max_single_jump_ratio", 0) >= 3.0,
    },
    {
        "pattern": "drift",
        "label_en": "Drift",
        "label_zh": "漂移",
        "description": "Gradual monotonic trend away from historical baseline over a sustained window.",
        "test": lambda ctx: ctx.get("slope_per_hour", 0.0) != 0.0
        and abs(ctx.get("slope_per_hour", 0.0)) > ctx.get("drift_threshold", 1.0),
    },
    {
        "pattern": "zero_flatline",
        "label_en": "Zero Flatline",
        "label_zh": "歸零",
        "description": "Readings drop to zero or near-zero and remain flat, likely sensor offline or meter failure.",
        "test": lambda ctx: ctx.get("zero_ratio", 0.0) >= 0.5,
    },
    {
        "pattern": "oscillation",
        "label_en": "Oscillation",
        "label_zh": "震盪",
        "description": "Rapid alternating high/low values around baseline, possible control hunting or sensor noise.",
        "test": lambda ctx: ctx.get("sign_change_rate", 0.0) >= 0.4,
    },
    {
        "pattern": "step_change",
        "label_en": "Step Change",
        "label_zh": "階梯",
        "description": "Abrupt sustained shift to a new level, possible schedule change, equipment switch, or setpoint change.",
        "test": lambda ctx: ctx.get("step_ratio", 0.0) >= 2.0 and ctx.get("sustained_after_step", False),
    },
]


def _compute_pattern_context(values: pd.Series) -> dict[str, Any]:
    n = len(values)
    if n < 3:
        return {"max_single_jump_ratio": 0, "slope_per_hour": 0, "zero_ratio": 0,
                "sign_change_rate": 0, "step_ratio": 0, "sustained_after_step": False,
                "drift_threshold": 0, "global_std": 0}

    global_std = float(values.std(ddof=0))
    global_median = float(values.median())
    global_mad = float((values - global_median).abs().median()) or 1.0

    diffs = values.diff().dropna()
    max_jump = float(diffs.abs().max()) if len(diffs) > 0 else 0.0
    max_single_jump_ratio = max_jump / global_mad if global_mad > 0 else 0.0

    x = np.arange(n, dtype=float)
    if n >= 6:
        slope = float(np.polyfit(x, values.values, 1)[0])
    else:
        slope = float(diffs.mean()) if len(diffs) > 0 else 0.0
    slope_per_hour = slope

    zero_threshold = global_median * 0.05 if global_median > 0 else 1.0
    zero_count = int((values.abs() <= zero_threshold).sum())
    zero_ratio = zero_count / n

    if len(diffs) >= 2:
        signs = np.sign(diffs.values)
        sign_changes = int(np.sum(signs[1:] != signs[:-1]))
        sign_change_rate = sign_changes / max(1, len(diffs) - 1)
    else:
        sign_change_rate = 0.0

    half = max(1, n // 2)
    first_half_mean = float(values.iloc[:half].mean())
    second_half_mean = float(values.iloc[half:].mean())
    half_mad = float((values.iloc[:half] - first_half_mean).abs().median()) or 1.0
    step_diff = abs(second_half_mean - first_half_mean)
    step_ratio = step_diff / half_mad if half_mad > 0 else 0.0
    sustained_after_step = False
    if step_ratio >= 2.0 and half >= 3:
        tail_std = float(values.iloc[half:].std(ddof=0))
        sustained_after_step = tail_std < step_diff * 1.5 if step_diff > 0 else False

    return {
        "max_single_jump_ratio": round(max_single_jump_ratio, 3),
        "slope_per_hour": round(slope_per_hour, 4),
        "drift_threshold": round(global_mad, 4),
        "zero_ratio": round(zero_ratio, 3),
        "sign_change_rate": round(sign_change_rate, 3),
        "step_ratio": round(step_ratio, 3),
        "sustained_after_step": sustained_after_step,
        "global_std": round(global_std, 4),
        "global_median": round(global_median, 4),
        "global_mad": round(global_mad, 4),
    }


def classify_anomaly_pattern(
    values: list[float] | pd.Series | None = None,
    csv_path: str = "",
    value_column: str = "",
    timestamp_column: str = "",
) -> dict[str, Any]:
    series: pd.Series | None = None

    if values is not None:
        series = pd.Series(values, dtype=float).dropna()
    elif csv_path:
        path = Path(str(csv_path)).expanduser()
        if not path.is_file():
            return {"status": "error", "error": f"CSV not found: {path}", "patterns": []}
        df = pd.read_csv(path)
        if df.empty:
            return {"status": "error", "error": "CSV is empty", "patterns": []}
        val_col = _infer_value_column(df, value_column)
        if not val_col:
            return {"status": "error", "error": "Cannot infer numeric column", "patterns": []}
        series = pd.to_numeric(df[val_col], errors="coerce").dropna()

    if series is None or len(series) < 3:
        return {"status": "error", "error": "Need >= 3 numeric values", "patterns": []}

    ctx = _compute_pattern_context(series)
    matched: list[dict[str, Any]] = []
    for rule in _ANOMALY_PATTERN_RULES:
        try:
            if rule["test"](ctx):
                matched.append({
                    "pattern": rule["pattern"],
                    "label_en": rule["label_en"],
                    "label_zh": rule["label_zh"],
                    "description": rule["description"],
                })
        except Exception:
            continue

    if not matched:
        matched.append({
            "pattern": "none",
            "label_en": "No anomaly pattern",
            "label_zh": "無明顯異常模式",
            "description": "Values are within normal statistical bounds.",
        })

    return {
        "status": "ok",
        "input_length": len(series),
        "context": ctx,
        "patterns": matched,
        "primary_pattern": matched[0]["pattern"],
    }


_SENSOR_CORRELATION_RULES: list[dict[str, Any]] = [
    {
        "rule_id": "temp_up_power_down",
        "description_zh": "溫度上升但功率下降，可能冷房設備故障或停機",
        "description_en": "Temperature rising while power dropping: possible cooling equipment failure or shutdown",
        "severity": "high",
        "conditions": {"temp_trend": "up", "power_trend": "down"},
        "possible_causes": ["chiller_failure", "ahu_offline", "cooling_valve_stuck_closed"],
        "suggested_tools": ["run_pvid", "openbse_hvac_breakdown"],
    },
    {
        "rule_id": "temp_down_power_up",
        "description_zh": "溫度下降但功率上升，可能過度製冷或控制失靈",
        "description_en": "Temperature dropping while power rising: possible over-cooling or control malfunction",
        "severity": "medium",
        "conditions": {"temp_trend": "down", "power_trend": "up"},
        "possible_causes": ["thermostat_fault", "valve_stuck_open", "setpoint_error"],
        "suggested_tools": ["run_pvid", "detect_energy_anomalies"],
    },
    {
        "rule_id": "humidity_up_temp_normal",
        "description_zh": "濕度異常上升但溫度正常，可能除濕系統故障",
        "description_en": "Humidity rising abnormally while temperature normal: possible dehumidification failure",
        "severity": "medium",
        "conditions": {"humidity_trend": "up", "temp_trend": "normal"},
        "possible_causes": ["dehumidifier_off", "fresh_air_damper_leak", "condensate_drain_block"],
        "suggested_tools": ["openbse_hvac_breakdown"],
    },
    {
        "rule_id": "all_up",
        "description_zh": "溫度、濕度、功率同時上升，可能熱負荷異常增加",
        "description_en": "Temp, humidity, and power all rising: abnormal heat load increase",
        "severity": "high",
        "conditions": {"temp_trend": "up", "humidity_trend": "up", "power_trend": "up"},
        "possible_causes": ["occupancy_surge", "equipment_malfunction", "external_heat_source"],
        "suggested_tools": ["compare_energy_usage", "detect_energy_anomalies"],
    },
    {
        "rule_id": "power_flat_temp_swing",
        "description_zh": "功率平穩但溫度劇烈震盪，可能感測器故障或控制迴路震盪",
        "description_en": "Power stable but temperature oscillating wildly: possible sensor fault or control loop hunting",
        "severity": "medium",
        "conditions": {"power_trend": "flat", "temp_trend": "oscillating"},
        "possible_causes": ["sensor_noise", "pid_hunting", "thermostat_degradation"],
        "suggested_tools": ["classify_anomaly_pattern", "detect_energy_anomalies"],
    },
]


def _infer_trend(values: list[float] | pd.Series) -> str:
    if not values or len(values) < 3:
        return "unknown"
    s = pd.Series(values, dtype=float).dropna()
    if len(s) < 3:
        return "unknown"

    x = np.arange(len(s), dtype=float)
    slope = float(np.polyfit(x, s.values, 1)[0])
    total_change = abs(slope * (len(s) - 1))
    value_range = float(s.max() - s.min()) or 1.0
    mean_val = float(s.mean()) or 1.0
    slope_norm = slope * (len(s) - 1) / mean_val

    diffs = s.diff().dropna()
    if len(diffs) >= 2:
        signs = np.sign(diffs.values)
        sign_changes = int(np.sum(signs[1:] != signs[:-1]))
        if sign_changes / max(1, len(diffs) - 1) >= 0.6:
            return "oscillating"

    if slope_norm > 0.3:
        return "up"
    if slope_norm < -0.3:
        return "down"
    return "normal"


def cross_sensor_diagnosis(
    power_values: list[float] | None = None,
    temp_values: list[float] | None = None,
    humidity_values: list[float] | None = None,
    sensor_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    power = list(power_values or [])
    temp = list(temp_values or [])
    humidity = list(humidity_values or [])

    trends: dict[str, str] = {}
    if power:
        trends["power_trend"] = _infer_trend(power)
    if temp:
        trends["temp_trend"] = _infer_trend(temp)
    if humidity:
        trends["humidity_trend"] = _infer_trend(humidity)

    if not trends:
        return {"status": "error", "error": "Provide at least one sensor series", "diagnoses": []}

    matched: list[dict[str, Any]] = []
    for rule in _SENSOR_CORRELATION_RULES:
        conditions = rule["conditions"]
        all_match = all(trends.get(k) == v for k, v in conditions.items())
        if all_match:
            matched.append({
                "rule_id": rule["rule_id"],
                "description_zh": rule["description_zh"],
                "description_en": rule["description_en"],
                "severity": rule["severity"],
                "possible_causes": rule["possible_causes"],
                "suggested_tools": rule["suggested_tools"],
            })

    if not matched:
        matched.append({
            "rule_id": "no_cross_anomaly",
            "description_zh": "感測器交叉比對未發現異常關聯",
            "description_en": "No anomalous cross-sensor correlation detected",
            "severity": "low",
            "possible_causes": [],
            "suggested_tools": [],
        })

    labels = sensor_labels or {}
    return {
        "status": "ok",
        "sensor_labels": labels,
        "detected_trends": trends,
        "diagnoses": matched,
        "primary_diagnosis": matched[0],
    }


def diagnose_energy_anomaly_impl(
    csv_path: str = "",
    building_uid: str = "",
    meter_name: str = "",
    value_column: str = "",
    timestamp_column: str = "",
    window: int = 24,
    z_threshold: float = 3.0,
    max_points: int = 20,
    power_values: list[float] | None = None,
    temp_values: list[float] | None = None,
    humidity_values: list[float] | None = None,
    sensor_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    anomaly_result = detect_energy_anomalies_impl(
        csv_path=csv_path,
        building_uid=building_uid,
        meter_name=meter_name,
        value_column=value_column,
        timestamp_column=timestamp_column,
        window=window,
        z_threshold=z_threshold,
        max_points=max_points,
    )

    values_for_pattern: list[float] | None = None
    if csv_path:
        pattern_result = classify_anomaly_pattern(
            csv_path=csv_path,
            value_column=value_column,
            timestamp_column=timestamp_column,
        )
    elif power_values:
        pattern_result = classify_anomaly_pattern(values=power_values)
    else:
        pattern_result = {"status": "skipped", "patterns": [], "primary_pattern": "unknown",
                          "reason": "No CSV or direct values provided"}

    cross_result: dict[str, Any] | None = None
    if power_values or temp_values or humidity_values:
        cross_result = cross_sensor_diagnosis(
            power_values=power_values,
            temp_values=temp_values,
            humidity_values=humidity_values,
            sensor_labels=sensor_labels,
        )

    anomaly_count = anomaly_result.get("summary", {}).get("anomaly_count", 0)
    pattern_primary = pattern_result.get("primary_pattern", "unknown")
    cross_primary = cross_result.get("primary_diagnosis", {}) if cross_result else {}

    if anomaly_count > 0:
        if pattern_primary in ("spike", "oscillation"):
            severity = "high"
            short_description = f"偵測到 {pattern_primary} 模式，共 {anomaly_count} 個異常點"
        elif pattern_primary in ("drift", "step_change"):
            severity = "medium"
            short_description = f"偵測到 {pattern_primary} 模式，共 {anomaly_count} 個異常點"
        elif pattern_primary == "zero_flatline":
            severity = "critical"
            short_description = f"偵測到歸零/斷訊，共 {anomaly_count} 個異常點"
        else:
            severity = "low"
            short_description = f"統計異常 {anomaly_count} 點，模式不明確"
    else:
        severity = "none"
        short_description = "未偵測到統計異常"

    if cross_primary and cross_primary.get("severity") in ("high", "medium"):
        severity = max(severity, cross_primary["severity"], key=lambda s: {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}.get(s, 0))
        short_description += f"；交叉診斷：{cross_primary.get('description_zh', '')}"

    suggested_actions: list[str] = []
    if pattern_primary != "unknown" and pattern_primary != "none":
        suggested_actions.append(f"異常模式：{pattern_primary}，建議檢查對應設備狀態")
    if cross_primary and cross_primary.get("possible_causes"):
        suggested_actions.append(f"可能原因：{', '.join(cross_primary['possible_causes'])}")
    if cross_primary and cross_primary.get("suggested_tools"):
        suggested_actions.append(f"建議使用的診斷工具：{', '.join(cross_primary['suggested_tools'])}")
    if anomaly_count > 0:
        suggested_actions.append("比較同時段相鄰建築或相鄰日趨勢")
    suggested_actions.append("確認後可使用 append_energy_decision_log 記錄審計軌跡")

    return {
        "status": "ok",
        "severity": severity,
        "short_description": short_description,
        "anomaly_detection": anomaly_result,
        "pattern_classification": pattern_result,
        "cross_sensor_diagnosis": cross_result,
        "suggested_actions": suggested_actions,
        "building_uid": building_uid,
        "meter_name": meter_name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
