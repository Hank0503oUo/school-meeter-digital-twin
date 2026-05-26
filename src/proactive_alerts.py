from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.energy_manager_skills import classify_anomaly_pattern


_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_DIR = _ROOT / "outputs" / "energy_manager"
_ALERTS_FILE = _DEFAULT_OUTPUT_DIR / "alerts.jsonl"
_OUTBOX_FILE = _DEFAULT_OUTPUT_DIR / "notification_outbox.jsonl"

_SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
_PATTERN_SEVERITY = {
    "zero_flatline": "critical",
    "zero": "critical",
    "stuck": "high",
    "step_change": "high",
    "step": "high",
    "spike": "high",
    "oscillation": "medium",
    "drift": "medium",
    "noise": "low",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def _make_alert_id(payload: dict[str, Any]) -> str:
    key = "|".join(
        str(payload.get(name, ""))
        for name in ("event_type", "building_uid", "meter_name", "anomaly_type", "detected_at", "summary")
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"alert_{digest}"


def _normalize_severity(severity: str) -> str:
    value = str(severity or "medium").strip().lower()
    return value if value in _SEVERITY_ORDER else "medium"


def _normalize_classification(result: dict[str, Any]) -> dict[str, Any]:
    if "pattern" in result:
        pattern = str(result.get("pattern") or "normal")
    else:
        pattern = str(result.get("primary_pattern") or "normal")
    if pattern in {"none", "normal"}:
        severity = "low"
    else:
        severity = _normalize_severity(str(result.get("severity") or _PATTERN_SEVERITY.get(pattern, "medium")))
    detail = str(result.get("detail") or result.get("description") or "")
    if not detail and result.get("patterns"):
        detail = str((result.get("patterns") or [{}])[0].get("description") or "")
    normalized = dict(result)
    normalized["pattern"] = pattern
    normalized["severity"] = severity
    normalized["detail"] = detail
    return normalized


def create_energy_alert_impl(
    *,
    title: str,
    summary: str,
    severity: str = "medium",
    event_type: str = "anomaly",
    building_uid: str = "",
    meter_name: str = "",
    anomaly_type: str = "",
    evidence: dict[str, Any] | None = None,
    recommended_actions: list[str] | str | None = None,
    source: str = "agent",
) -> dict[str, Any]:
    actions = [recommended_actions] if isinstance(recommended_actions, str) else list(recommended_actions or [])
    payload = {
        "event_type": event_type or "anomaly",
        "building_uid": building_uid,
        "meter_name": meter_name,
        "anomaly_type": anomaly_type,
        "detected_at": _utc_now(),
        "summary": summary,
    }
    alert = {
        "alert_id": _make_alert_id(payload),
        "status": "open",
        "created_at": payload["detected_at"],
        "updated_at": payload["detected_at"],
        "source": source or "agent",
        "event_type": payload["event_type"],
        "title": title or "Energy anomaly alert",
        "summary": summary or "No summary provided.",
        "severity": _normalize_severity(severity),
        "building_uid": building_uid,
        "meter_name": meter_name,
        "anomaly_type": anomaly_type,
        "evidence": evidence or {},
        "recommended_actions": actions,
        "audit": [],
    }

    rows = _read_jsonl(_ALERTS_FILE)
    if any(row.get("alert_id") == alert["alert_id"] for row in rows):
        return {"status": "duplicate", "alert_id": alert["alert_id"], "path": str(_ALERTS_FILE)}
    _append_jsonl(_ALERTS_FILE, alert)
    return {"status": "ok", "alert": alert, "path": str(_ALERTS_FILE)}


def list_active_energy_alerts_impl(
    *,
    severity_min: str = "low",
    building_uid: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    threshold = _SEVERITY_ORDER.get(_normalize_severity(severity_min), 1)
    rows = []
    for row in _read_jsonl(_ALERTS_FILE):
        if row.get("status") not in {"open", "acknowledged"}:
            continue
        if building_uid and str(row.get("building_uid", "")).lower() != str(building_uid).lower():
            continue
        if _SEVERITY_ORDER.get(str(row.get("severity", "low")).lower(), 1) < threshold:
            continue
        rows.append(row)
    rows.sort(key=lambda row: (_SEVERITY_ORDER.get(str(row.get("severity", "low")).lower(), 1), row.get("created_at", "")), reverse=True)
    return {"status": "ok", "alerts": rows[: max(1, int(limit))], "total": len(rows), "path": str(_ALERTS_FILE)}


def update_energy_alert_status_impl(
    *,
    alert_id: str,
    status: str,
    operator: str = "",
    note: str = "",
) -> dict[str, Any]:
    allowed = {"open", "acknowledged", "closed", "false_positive"}
    new_status = str(status or "").strip().lower()
    if new_status not in allowed:
        return {"status": "error", "error": f"status must be one of {sorted(allowed)}"}
    rows = _read_jsonl(_ALERTS_FILE)
    now = _utc_now()
    for row in rows:
        if row.get("alert_id") == alert_id:
            row["status"] = new_status
            row["updated_at"] = now
            audit = list(row.get("audit") or [])
            audit.append({"at": now, "operator": operator or "unknown", "status": new_status, "note": note})
            row["audit"] = audit
            _write_jsonl(_ALERTS_FILE, rows)
            return {"status": "ok", "alert": row, "path": str(_ALERTS_FILE)}
    return {"status": "not_found", "alert_id": alert_id, "path": str(_ALERTS_FILE)}


def notify_energy_manager_impl(
    *,
    alert_id: str = "",
    channel: str = "outbox",
    recipients: list[str] | str | None = None,
    message: str = "",
    dry_run: bool = True,
) -> dict[str, Any]:
    recipient_list = [recipients] if isinstance(recipients, str) else list(recipients or [])
    alert = None
    if alert_id:
        alert = next((row for row in _read_jsonl(_ALERTS_FILE) if row.get("alert_id") == alert_id), None)
    if not message and alert:
        message = f"[{alert.get('severity', 'medium').upper()}] {alert.get('title')}: {alert.get('summary')}"
    notification = {
        "notification_id": "notif_" + hashlib.sha1(f"{alert_id}|{message}|{_utc_now()}".encode("utf-8")).hexdigest()[:10],
        "created_at": _utc_now(),
        "alert_id": alert_id,
        "channel": channel or "outbox",
        "recipients": recipient_list,
        "message": message or "Energy alert notification requested.",
        "dry_run": bool(dry_run),
        "delivery_status": "queued" if not dry_run else "dry_run",
    }
    _append_jsonl(_OUTBOX_FILE, notification)
    return {"status": "ok", "notification": notification, "path": str(_OUTBOX_FILE)}


def recommend_anomaly_decision_impl(alert: dict[str, Any] | None = None, alert_id: str = "") -> dict[str, Any]:
    event = alert or {}
    if alert_id and not event:
        event = next((row for row in _read_jsonl(_ALERTS_FILE) if row.get("alert_id") == alert_id), {})
    if not event:
        return {"status": "not_found", "alert_id": alert_id}
    severity = _normalize_severity(str(event.get("severity", "medium")))
    anomaly_type = str(event.get("anomaly_type", "unknown")) or "unknown"
    immediate = severity in {"critical", "high"}
    actions = list(event.get("recommended_actions") or [])
    if not actions:
        actions = [
            "Confirm whether the signal is real by checking neighboring BMS/RTEM points.",
            "Review schedules, setpoints, and recent maintenance logs for the same time window.",
            "If the condition persists for another scan interval, create a maintenance ticket.",
        ]
    return {
        "status": "ok",
        "severity": severity,
        "anomaly_type": anomaly_type,
        "should_notify": immediate,
        "should_create_ticket": severity in {"critical", "high"},
        "decision_summary": (
            f"Treat this as {severity} severity {anomaly_type}. "
            f"{'Notify the energy manager now.' if immediate else 'Keep it on the dashboard and monitor the next scan.'}"
        ),
        "recommended_actions": actions,
        "next_tools": ["diagnose_energy_anomaly", "notify_energy_manager", "append_energy_decision_log"],
    }


def scan_iot_snapshot_for_alerts_impl(
    *,
    snapshot: dict[str, Any],
    source: str = "scheduled_scan",
    create_alerts: bool = True,
) -> dict[str, Any]:
    readings = snapshot.get("readings") if isinstance(snapshot, dict) else []
    if isinstance(readings, dict):
        readings = [readings]
    if not isinstance(readings, list):
        return {"status": "error", "error": "snapshot.readings must be a list or object"}

    findings: list[dict[str, Any]] = []
    created: list[dict[str, Any]] = []
    for item in readings:
        if not isinstance(item, dict):
            continue
        values = item.get("values") or item.get("series") or []
        if isinstance(values, (int, float)):
            values = [values]
        if len(values) < 5:
            continue
        try:
            classification = classify_anomaly_pattern(values=values, timestamps=item.get("timestamps"))
        except TypeError:
            classification = classify_anomaly_pattern(values=values)
        classification = _normalize_classification(classification)
        pattern = str(classification.get("pattern", "normal"))
        severity = _normalize_severity(str(classification.get("severity", "low")))
        if pattern in {"normal", "insufficient_data"} or _SEVERITY_ORDER[severity] < _SEVERITY_ORDER["medium"]:
            continue
        finding = {
            "building_uid": item.get("building_uid") or item.get("building_id") or "",
            "meter_name": item.get("meter_name") or item.get("point_id") or item.get("sensor_id") or "",
            "anomaly_type": pattern,
            "severity": severity,
            "classification": classification,
        }
        findings.append(finding)
        if create_alerts:
            result = create_energy_alert_impl(
                title=f"{pattern} detected on {finding['meter_name'] or 'RTEM point'}",
                summary=classification.get("detail") or classification.get("description") or f"{pattern} anomaly detected.",
                severity=severity,
                event_type="scheduled_iot_scan",
                building_uid=str(finding["building_uid"]),
                meter_name=str(finding["meter_name"]),
                anomaly_type=pattern,
                evidence={"source_snapshot": item, "classification": classification},
                recommended_actions=[
                    "Check adjacent RTEM/BMS points for the same time window.",
                    "Verify schedule, setpoint, and recent maintenance changes.",
                    "Escalate to maintenance if the next scan confirms persistence.",
                ],
                source=source,
            )
            created.append(result)
    return {"status": "ok", "findings": findings, "created_alerts": created, "finding_count": len(findings)}
