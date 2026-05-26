from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from src.utils import to_float as _safe_float

_DEMO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_METADATA_LOOP = _DEMO_ROOT / "data" / "NTU" / "BUILD DATA" / "metadata_loop.csv"
_DEFAULT_METADATA_UID = _DEMO_ROOT / "data" / "NTU" / "BUILD DATA" / "metadata_uid.csv"
_DEFAULT_METER_BUILDING_MAP = _DEMO_ROOT / "data" / "NTU" / "meter_building_map.csv"
_DEFAULT_METER_UID_OVERRIDES = _DEMO_ROOT / "config" / "meter_uid_overrides.csv"

_METER_ID_RE = re.compile(r"^\s*([0-9A-Za-z]+_[0-9A-Za-z]+_[0-9A-Za-z]+)")
_PAREN_PATTERN = re.compile(r"\([^)]*\)|（[^）]*）")
_PAREN_CONTENT_PATTERN = re.compile(r"\(([^)]*)\)|（([^）]*)）")

_METER_NAME_HINT_COLS = [
    "osm_name",
    "building_name_extracted",
    "building_name",
]

_METADATA_NAME_COLS = [
    "name",
    "nameC",
    "nameE",
    "nameE.1",
    "nameE.2",
    "nameE.3",
]

_NAME_STRIP_TOKENS = [
    "總表", "總錶", "總用電", "總變電站", "配電站", "變電站",
    "饋線", "高壓", "低壓", "空調盤", "總站",
    "MCB", "GCB", "MVCB", "HTM", "MAIN", "VCB", "ACB", "SUBMETER",
]

_NAME_SUFFIXES = [
    "館", "大樓", "教學館", "研究所", "學系", "系館",
]

_METER_CANDIDATE_STRIP_TOKENS = [
    "站", "饋線", "總用電", "總電源", "總錶", "總表", "高壓", "低壓",
]


def _normalize_uid(uid: str) -> str:
    return str(uid or "").strip()


def _meter_name_to_meter_id(meter_name: str) -> str:
    match = _METER_ID_RE.match(str(meter_name or ""))
    return match.group(1) if match else ""


def _normalize_text(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    s = _PAREN_PATTERN.sub("", s)
    s = s.replace("－", "-")
    for token in _NAME_STRIP_TOKENS:
        s = s.replace(token, "")
    s = re.sub(r"[\s\-_、，,;；:/\\]+", "", s)
    return s.lower().strip()


def _name_variants(name: str) -> list[str]:
    base = str(name or "").strip()
    if not base:
        return []
    variants = [base]
    for suffix in _NAME_SUFFIXES:
        if base.endswith(suffix) and len(base) > len(suffix):
            variants.append(base[: -len(suffix)])
    # Remove all suffix tokens once for robust matching
    stripped = base
    for suffix in _NAME_SUFFIXES:
        stripped = stripped.replace(suffix, "")
    stripped = stripped.strip()
    if stripped:
        variants.append(stripped)
    # Keep unique order
    seen = set()
    out = []
    for v in variants:
        k = _normalize_text(v)
        if k and k not in seen:
            seen.add(k)
            out.append(v)
    return out


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "cp950", "big5"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (OSError, pd.errors.EmptyDataError):
            continue
    return pd.DataFrame()


def _build_uid_name_lookup(metadata_uid_path: Path | str = _DEFAULT_METADATA_UID) -> dict[str, set[str]]:
    df = _read_csv_if_exists(Path(metadata_uid_path))
    if df.empty:
        return {}

    out: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        uid = _normalize_uid(row.get("uid", ""))
        if not uid:
            continue
        for col in _METADATA_NAME_COLS:
            if col not in df.columns:
                continue
            raw_name = str(row.get(col, "")).strip().replace("\n", "")
            if not raw_name or raw_name.lower() == "nan":
                continue
            for v in _name_variants(raw_name):
                key = _normalize_text(v)
                if not key:
                    continue
                if key not in out:
                    out[key] = set()
                out[key].add(uid)
    return out


def _build_valid_uid_set(metadata_uid_path: Path | str = _DEFAULT_METADATA_UID) -> set[str]:
    df = _read_csv_if_exists(Path(metadata_uid_path))
    if df.empty or "uid" not in df.columns:
        return set()
    out = set()
    for uid in df["uid"].tolist():
        key = _normalize_uid(uid)
        if key:
            out.add(key)
    return out


def _build_meter_name_hints(meter_building_map_path: Path | str = _DEFAULT_METER_BUILDING_MAP) -> dict[str, list[str]]:
    df = _read_csv_if_exists(Path(meter_building_map_path))
    if df.empty or "meter_name" not in df.columns:
        return {}

    hints: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        meter_name = str(row.get("meter_name", "")).strip()
        if not meter_name:
            continue
        candidate_names: list[str] = []
        for col in _METER_NAME_HINT_COLS:
            if col not in df.columns:
                continue
            v = str(row.get(col, "")).strip()
            if not v or v.lower() == "nan":
                continue
            for part in v.split("|"):
                p = part.strip()
                if p and p.lower() != "nan":
                    candidate_names.append(p)

        # Keep unique order
        seen = set()
        uniq = []
        for c in candidate_names:
            k = _normalize_text(c)
            if k and k not in seen:
                seen.add(k)
                uniq.append(c)

        if uniq:
            hints[meter_name] = uniq
    return hints


def _build_meter_uid_overrides(
    overrides_path: Path | str = _DEFAULT_METER_UID_OVERRIDES,
    valid_uid_set: set[str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Load manual overrides from CSV.

    CSV columns (flexible):
      - enabled (optional, default true)
      - meter_id (optional)
      - meter_name (optional)
      - uid (required for active mapping)
    """
    df = _read_csv_if_exists(Path(overrides_path))
    if df.empty:
        return {}, {}

    by_meter_id: dict[str, str] = {}
    by_meter_name: dict[str, str] = {}

    def _is_enabled(value) -> bool:
        if value is None:
            return True
        s = str(value).strip().lower()
        if s in {"", "1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
        return True

    for _, row in df.iterrows():
        if ("enabled" in df.columns) and (not _is_enabled(row.get("enabled"))):
            continue
        uid = _normalize_uid(row.get("uid", ""))
        if not uid:
            continue
        if valid_uid_set and uid not in valid_uid_set:
            continue

        meter_id = str(row.get("meter_id", "")).strip()
        meter_name = str(row.get("meter_name", "")).strip()

        if meter_id:
            by_meter_id[meter_id] = uid
        if meter_name:
            by_meter_name[meter_name] = uid
            by_meter_name[_normalize_text(meter_name)] = uid

    return by_meter_id, by_meter_name


def _extract_meter_name_candidates(meter_name: str) -> list[str]:
    s = str(meter_name or "").strip()
    if not s:
        return []
    meter_id = _meter_name_to_meter_id(s)
    tail = (s[len(meter_id) :] if meter_id else s).strip()

    raw_candidates: list[str] = []
    if tail:
        raw_candidates.append(tail)

        # Keep parenthetical content like "森林館站" because it often contains key hints.
        for match in _PAREN_CONTENT_PATTERN.finditer(tail):
            content = (match.group(1) or match.group(2) or "").strip()
            if content:
                raw_candidates.append(content)

        tail_wo_paren = _PAREN_PATTERN.sub("", tail).strip()
        if tail_wo_paren:
            raw_candidates.append(tail_wo_paren)
    else:
        raw_candidates.append(s)

    variants: list[str] = []
    for cand in raw_candidates:
        variants.append(cand)
        variants.extend(_name_variants(cand))

        stripped = cand
        for token in _METER_CANDIDATE_STRIP_TOKENS:
            stripped = stripped.replace(token, "")
        stripped = stripped.strip()
        if stripped and stripped != cand:
            variants.append(stripped)
            variants.extend(_name_variants(stripped))

    seen = set()
    out = []
    for v in variants:
        k = _normalize_text(v)
        if k and k not in seen:
            seen.add(k)
            out.append(v)
    return out


def _resolve_uid_from_meter_name(
    meter_name: str,
    meter_to_uid: dict[str, str],
    uid_name_lookup: dict[str, set[str]],
    meter_name_hints: dict[str, list[str]],
    valid_uid_set: set[str] | None = None,
    override_by_meter_id: dict[str, str] | None = None,
    override_by_meter_name: dict[str, str] | None = None,
) -> str:
    meter = str(meter_name or "").strip()
    if not meter:
        return ""

    meter_id = _meter_name_to_meter_id(meter)
    override_by_meter_id = override_by_meter_id or {}
    override_by_meter_name = override_by_meter_name or {}

    uid = _normalize_uid(override_by_meter_name.get(meter, ""))
    if not uid:
        uid = _normalize_uid(override_by_meter_name.get(_normalize_text(meter), ""))
    if uid and (not valid_uid_set or uid in valid_uid_set):
        return uid

    uid = _normalize_uid(override_by_meter_id.get(meter_id, ""))
    if uid and (not valid_uid_set or uid in valid_uid_set):
        return uid

    uid = _normalize_uid(meter_to_uid.get(meter_id, ""))
    if uid and (not valid_uid_set or uid in valid_uid_set):
        return uid

    candidates: list[str] = []
    candidates.extend(_extract_meter_name_candidates(meter))
    candidates.extend(meter_name_hints.get(meter, []))

    # Exact normalized name match first.
    seen_keys = set()
    for cand in candidates:
        key = _normalize_text(cand)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        matched = uid_name_lookup.get(key, set())
        if matched:
            return sorted(matched)[0]

    # Fuzzy fallback among unique lookup keys.
    lookup_keys = list(uid_name_lookup.keys())
    best_key = ""
    best_score = 0.0
    for cand in candidates:
        q = _normalize_text(cand)
        if not q:
            continue
        for k in lookup_keys:
            score = SequenceMatcher(None, q, k).ratio()
            if score > best_score:
                best_score = score
                best_key = k
    if best_key and best_score >= 0.50:
        matched = uid_name_lookup.get(best_key, set())
        if matched:
            return sorted(matched)[0]
    return ""


def build_meter_uid_lookup(
    metadata_loop_path: Path | str = _DEFAULT_METADATA_LOOP,
) -> Dict[str, str]:
    """Build mapping from meter id (e.g. 01A_P1_01) to building UID."""
    path = Path(metadata_loop_path)
    if not path.exists():
        return {}

    loop_df = pd.read_csv(path, encoding="utf-8")
    if loop_df.empty:
        return {}

    meter_col = loop_df.columns[0]
    uid_col = "uid" if "uid" in loop_df.columns else (loop_df.columns[1] if len(loop_df.columns) > 1 else None)
    if uid_col is None:
        return {}

    meter_to_uid: Dict[str, str] = {}
    for _, row in loop_df.iterrows():
        meter_id = str(row.get(meter_col, "")).strip()
        uid = _normalize_uid(row.get(uid_col, ""))
        if meter_id and uid:
            meter_to_uid[meter_id] = uid
    return meter_to_uid


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    valid = vals.notna() & w.notna() & (w >= 0)
    if not bool(valid.any()):
        return np.nan
    vals = vals[valid]
    w = w[valid]
    w_sum = float(w.sum())
    if w_sum <= 0:
        return float(vals.mean()) if len(vals) else np.nan
    return float(np.average(vals, weights=w))


def aggregate_meter_summary_by_uid(
    meter_summary: pd.DataFrame,
    metadata_loop_path: Path | str = _DEFAULT_METADATA_LOOP,
    metadata_uid_path: Path | str = _DEFAULT_METADATA_UID,
    meter_building_map_path: Path | str = _DEFAULT_METER_BUILDING_MAP,
    meter_uid_overrides_path: Path | str = _DEFAULT_METER_UID_OVERRIDES,
) -> pd.DataFrame:
    """
    Aggregate meter summary rows to UID-level metrics.

    Expected inputs:
    - `uid` and `mean_kw`, or
    - `meter_name` and `mean_kw` (meter id prefix mapped via metadata_loop.csv).
    """
    if meter_summary is None or meter_summary.empty:
        return pd.DataFrame(columns=["uid", "mean_kw"])

    df = meter_summary.copy()
    mapping_df = build_meter_id_uid_mapping(
        meter_summary=df,
        metadata_loop_path=metadata_loop_path,
        metadata_uid_path=metadata_uid_path,
        meter_building_map_path=meter_building_map_path,
        meter_uid_overrides_path=meter_uid_overrides_path,
    )
    if "uid" in df.columns:
        df["uid"] = [
            _normalize_uid(u)
            for u in df["uid"].tolist()
        ]
    else:
        df["uid"] = ""
    if "meter_name" in df.columns:
        resolved_by_name = dict(zip(mapping_df["meter_name"], mapping_df["uid"]))
        df["uid"] = [
            _normalize_uid(u if u else resolved_by_name.get(str(m).strip(), ""))
            for u, m in zip(df["uid"], df["meter_name"])
        ]

    df["mean_kw"] = pd.to_numeric(df.get("mean_kw", np.nan), errors="coerce")
    df = df[df["uid"] != ""].copy()
    df = df[df["mean_kw"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["uid", "mean_kw"])

    agg_rows = []
    for uid, group in df.groupby("uid", sort=False):
        row = {
            "uid": uid,
            "mean_kw": float(pd.to_numeric(group["mean_kw"], errors="coerce").fillna(0.0).sum()),
        }

        for col in ["best_r2_oof", "best_r_oof", "best_cvrmse_oof", "coverage_ratio", "n_valid_hours"]:
            if col not in group.columns:
                continue
            if col in {"coverage_ratio", "n_valid_hours"}:
                row[col] = float(pd.to_numeric(group[col], errors="coerce").max())
            else:
                row[col] = _weighted_mean(group[col], group["mean_kw"])

        agg_rows.append(row)

    out = pd.DataFrame(agg_rows)
    for col in ["mean_kw", "best_r2_oof", "best_r_oof", "best_cvrmse_oof", "coverage_ratio", "n_valid_hours"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_meter_id_uid_mapping(
    meter_summary: pd.DataFrame,
    metadata_loop_path: Path | str = _DEFAULT_METADATA_LOOP,
    metadata_uid_path: Path | str = _DEFAULT_METADATA_UID,
    meter_building_map_path: Path | str = _DEFAULT_METER_BUILDING_MAP,
    meter_uid_overrides_path: Path | str = _DEFAULT_METER_UID_OVERRIDES,
) -> pd.DataFrame:
    """
    Resolve each meter row to UID and return a detailed mapping table.
    """
    if meter_summary is None or meter_summary.empty:
        return pd.DataFrame(
            columns=["meter_name", "meter_id", "uid", "mapping_method", "mapped"]
        )

    df = meter_summary.copy()
    uid_col = "uid" if "uid" in df.columns else None
    meter_col = "meter_name" if "meter_name" in df.columns else None

    meter_to_uid = build_meter_uid_lookup(metadata_loop_path)
    uid_name_lookup = _build_uid_name_lookup(metadata_uid_path)
    valid_uid_set = _build_valid_uid_set(metadata_uid_path)
    meter_name_hints = _build_meter_name_hints(meter_building_map_path)
    override_by_meter_id, override_by_meter_name = _build_meter_uid_overrides(
        meter_uid_overrides_path,
        valid_uid_set=valid_uid_set,
    )

    rows = []
    for _, row in df.iterrows():
        meter_name = str(row.get(meter_col, "")).strip() if meter_col else ""
        meter_id = _meter_name_to_meter_id(meter_name)
        uid = _normalize_uid(row.get(uid_col, "")) if uid_col else ""
        mapping_method = "input_uid" if uid else "unmapped"

        if (not uid) and meter_col:
            uid = _resolve_uid_from_meter_name(
                meter_name=meter_name,
                meter_to_uid=meter_to_uid,
                uid_name_lookup=uid_name_lookup,
                meter_name_hints=meter_name_hints,
                valid_uid_set=valid_uid_set,
                override_by_meter_id=override_by_meter_id,
                override_by_meter_name=override_by_meter_name,
            )
            if uid:
                if meter_name in override_by_meter_name or _normalize_text(meter_name) in override_by_meter_name or meter_id in override_by_meter_id:
                    mapping_method = "manual_override"
                elif meter_id in meter_to_uid and _normalize_uid(meter_to_uid.get(meter_id, "")) in valid_uid_set:
                    mapping_method = "metadata_loop"
                else:
                    mapping_method = "name_hint_or_fuzzy"

        rows.append(
            {
                "meter_name": meter_name,
                "meter_id": meter_id,
                "uid": uid,
                "mapping_method": mapping_method,
                "mapped": bool(uid),
            }
        )

    return pd.DataFrame(rows)


def infer_all_buildings(
    engine,
    weather_df: pd.DataFrame,
    meter_summary: pd.DataFrame,
    metadata_loop_path: Path | str = _DEFAULT_METADATA_LOOP,
    metadata_uid_path: Path | str = _DEFAULT_METADATA_UID,
    meter_building_map_path: Path | str = _DEFAULT_METER_BUILDING_MAP,
    meter_uid_overrides_path: Path | str = _DEFAULT_METER_UID_OVERRIDES,
) -> pd.DataFrame:
    """
    Infer all building-level timeseries from campus-level PI-VD predictions.

    Returns DataFrame columns:
      uid, name, area, floors, buildType, scaler,
      mean_kw, annual_kwh, annual_mwh,
      eui_kw_per_m2, energy_tier, tier_color,
      data_source, timeseries,
      best_r2_oof, best_r_oof, best_cvrmse_oof, coverage_ratio, n_valid_hours
    """
    if weather_df is None or weather_df.empty:
        raise ValueError("weather_df must not be empty")
    if not hasattr(engine, "metadata_scaler"):
        raise ValueError("engine must provide metadata_scaler")

    uid_list = [
        _normalize_uid(uid)
        for uid in getattr(engine.metadata_scaler, "list_uids")()
        if _normalize_uid(uid)
    ]
    if not uid_list:
        raise ValueError("No building metadata loaded in engine.metadata_scaler")

    pred_df = engine.predict(weather_df)
    if "total_pred" not in pred_df.columns:
        raise ValueError("engine.predict(weather_df) must include column 'total_pred'")

    campus_pred = pd.to_numeric(pred_df["total_pred"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if campus_pred.size == 0:
        raise ValueError("Campus prediction is empty")

    scalers: Dict[str, float] = {}
    meta_map: Dict[str, dict] = {}
    for uid in uid_list:
        scaler = _safe_float(engine.metadata_scaler.get_scaler(uid), 1.0)
        if not np.isfinite(scaler) or scaler <= 0:
            scaler = 1.0
        scalers[uid] = float(max(scaler, 1e-9))
        meta_map[uid] = engine.metadata_scaler.get_metadata(uid) or {}

    scaler_sum = float(sum(scalers.values()))
    if scaler_sum <= 0:
        raise ValueError("Invalid scaler sum <= 0")

    metered_df = aggregate_meter_summary_by_uid(
        meter_summary=meter_summary,
        metadata_loop_path=metadata_loop_path,
        metadata_uid_path=metadata_uid_path,
        meter_building_map_path=meter_building_map_path,
        meter_uid_overrides_path=meter_uid_overrides_path,
    )
    metered_by_uid = {
        _normalize_uid(row["uid"]): row
        for _, row in metered_df.iterrows()
        if _normalize_uid(row.get("uid", ""))
    }

    rows = []
    for uid in uid_list:
        meta = meta_map.get(uid, {})
        scaler = scalers[uid]
        base_ts = campus_pred * (scaler / scaler_sum)
        base_mean = float(np.mean(base_ts)) if base_ts.size else 0.0

        data_source = "inferred"
        target_mean = base_mean
        if uid in metered_by_uid:
            measured_mean = _safe_float(metered_by_uid[uid].get("mean_kw", np.nan), np.nan)
            if np.isfinite(measured_mean) and measured_mean >= 0:
                data_source = "metered"
                target_mean = float(measured_mean)

        if base_mean > 1e-9:
            ts = base_ts * (target_mean / base_mean)
        else:
            ts = np.full_like(base_ts, target_mean, dtype=float)

        area = _safe_float(meta.get("area", np.nan), np.nan)
        floors = _safe_float(meta.get("floors", np.nan), np.nan)
        basement = _safe_float(meta.get("basement", 0.0), 0.0)
        total_floors = floors + basement if np.isfinite(floors) else np.nan
        mean_kw = float(np.mean(ts)) if ts.size else 0.0
        eui_kw_per_m2 = mean_kw / area if (np.isfinite(area) and area > 0) else np.nan

        meter_row = metered_by_uid.get(uid, {})
        rows.append(
            {
                "uid": uid,
                "name": str(meta.get("name", "")),
                "area": area,
                "floors": total_floors,
                "buildType": str(meta.get("buildType", "")),
                "scaler": scaler,
                "mean_kw": mean_kw,
                "annual_kwh": mean_kw * 8760.0,
                "annual_mwh": mean_kw * 8.76,
                "eui_kw_per_m2": eui_kw_per_m2,
                "energy_tier": "NORMAL",
                "tier_color": "#f0c419",
                "data_source": data_source,
                "timeseries": ts.astype(float),
                "best_r2_oof": _safe_float(meter_row.get("best_r2_oof", np.nan), np.nan),
                "best_r_oof": _safe_float(meter_row.get("best_r_oof", np.nan), np.nan),
                "best_cvrmse_oof": _safe_float(meter_row.get("best_cvrmse_oof", np.nan), np.nan),
                "coverage_ratio": _safe_float(meter_row.get("coverage_ratio", np.nan), np.nan),
                "n_valid_hours": _safe_float(meter_row.get("n_valid_hours", np.nan), np.nan),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    eui_vals = pd.to_numeric(result["eui_kw_per_m2"], errors="coerce")
    finite_mask = eui_vals.notna() & np.isfinite(eui_vals)
    if finite_mask.any():
        mu = float(eui_vals[finite_mask].mean())
        sigma = float(eui_vals[finite_mask].std(ddof=0))
    else:
        mu = 0.0
        sigma = 0.0

    high_th = mu + sigma
    low_th = mu - sigma

    def _tier(eui: float) -> str:
        if not np.isfinite(eui):
            return "NORMAL"
        if eui > high_th:
            return "HIGH"
        if eui < low_th:
            return "LOW"
        return "NORMAL"

    tier_map = {"HIGH": "#d73027", "NORMAL": "#f0c419", "LOW": "#1a9850"}
    result["energy_tier"] = [ _tier(float(v) if np.isfinite(v) else np.nan) for v in eui_vals.fillna(np.nan) ]
    result["tier_color"] = result["energy_tier"].map(tier_map).fillna("#f0c419")

    # Keep a deterministic ordering for dropdowns and tests.
    result = result.sort_values(["name", "uid"]).reset_index(drop=True)
    return result
