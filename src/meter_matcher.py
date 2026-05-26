# -*- coding: utf-8 -*-
"""
Step 2: 電表名稱 -> 建物名稱匹配

用途:
1) 從電表名稱抽取建物名稱
2) 先做硬編碼映射，再做官方名與 OSM 模糊匹配
3) 產出 meter_building_map.csv 給 map_builder 合併使用
4) 額外標註 meter_role，避免後續總表/分表重複加總
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.project_paths import campus_data_dir, models_dir, resolve_project_path

_ROOT_DIR = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT_DIR / "data"
_MODELS_DIR = _ROOT_DIR / "models"

try:
    from thefuzz import fuzz
    from thefuzz import process as fuzz_process
except ImportError:  # pragma: no cover
    from fuzzywuzzy import fuzz  # type: ignore
    from fuzzywuzzy import process as fuzz_process  # type: ignore


# 已知高信心映射（關鍵字 -> OSM name）
# 值可使用 "|" 代表一個電表覆蓋多棟建物（共享總表）
HARDCODED_MAP = {
    "水源行政大樓及育成AB棟": "行政大樓 (水源校區)|育成中心A棟|育成中心B棟",
    "理化大樓、飲水樓、機械工廠": "理化大樓|飲水樓",
    "育成大樓A/B棟": "育成中心A棟|育成中心B棟",
    "育成大樓": "育成中心A棟|育成中心B棟|育成中心C棟",
    "水源行政大樓": "行政大樓 (水源校區)",
    "生科館": "生命科學館",
    "總圖書館": "總圖書館",
    "總圖": "總圖書館",
    "管理學院教研館": "管理學院",
    "管理學院1號館": "管理學院一號館",
    "男六八舍": "男六舍",
    "男一舍": "男一舍",
    "生機系": "生機系配電站 (總站)",
    "理化大樓": "理化大樓",
    "推廣中心": "進修推廣學院",
    "海洋所": "海洋研究所",
    "資訊工程館": "資訊工程館",
    "計資中心": "計算機及資訊網路中心",
    "工科海洋系": "工程科學及海洋工程學系館",
    "應用力學研究大樓": "應用力學研究所",
    "工學院綜合大樓": "工學院綜合大樓",
    "霖澤館": "霖澤館",
    "化學館": "化學館",
    "地質系": "地質科學館",
    "食科": "食品科技館",
    "生技中心": "生物技術中心",
    "凝態科學館": "凝態科學暨物理學館",
    "凝態": "凝態科學暨物理學館",
    "原子及分子研究所": "原子與分子科學研究所",
    "原分所": "原子與分子科學研究所",
    "語言大樓": "語文大樓",
    "體育館": "綜合體育館",
    "環研大樓": "環境研究大樓",
    "卓越研究大樓": "卓越研究大樓",
    "博理館": "博理館",
    "電機新館": "電機二館",
    "普通教室": "普通教學館",
    "農綜館": "農業綜合館",
    "森林館": "森林館",
    "望樂樓": "望樂樓",
    "社科院": "社會科學院",
    "獸醫館": "獸醫館",
    "實驗動物資源中心": "實驗動物中心",
}


ALIASES = {
    "新生大樓": "新生教學館",
    "綜合教室": "綜合教學館",
    "共同教室": "共同教學館",
    "水工所": "水工試驗所",
    "教研館": "管理學院",
    "語文大樓": "語言中心",
    "工綜大樓": "工學院綜合大樓",
    "生技中心": "生物技術研究中心",
    "社科院": "社科院大樓",
    "生科館": "生命科學館",
    "海洋所": "海洋研究所",
    "工科海洋系": "工程科學及海洋工程學系館",
    "計資中心": "計算機及網路資訊中心",
    "男六八舍": "男六舍",
    "環研大樓": "環境大樓",
    "學生第二活動中心": "第二學生活動中心",
    "男一舍": "男一舍",
    "女一舍": "大一女舍",
}


def extract_building_name(meter_name: str) -> str:
    """
    從電表名稱提取建物中文名。

    例：
      '07F_P1_01生科館(總)' -> '生科館'
      '01B_P1_01化學館（MVCB）(高壓)' -> '化學館'
      '04B_P1_01管理學院教研館(管理學院教學館)（高壓）' -> '管理學院教研館'
    """
    name = str(meter_name or "")
    name = re.sub(r"^[0-9A-Z_]+_[PB]\d+_\d+", "", name).strip()
    name = re.sub(r"[（(][^）)]*[）)]", "", name).strip()
    name = re.sub(r"[A-Z0-9#]+$", "", name).strip()

    # 常見後綴（多次剝除直到穩定）
    suffix_patterns = [
        r"(饋線|總錶|總表|總用電|總電表|總電源|總電|電源)$",
        r"(MCB|HTM|GCB|GCBM|MVCB|MAIN|VCB|ACB\d*)$",
        r"(空調盤|照明盤|動力盤|主盤|分盤)$",
        r"(HV\d*|LV\d*|PANEL\d*)$",
    ]
    changed = True
    while changed:
        changed = False
        for pat in suffix_patterns:
            new_name = re.sub(pat, "", name, flags=re.IGNORECASE).strip()
            if new_name != name:
                name = new_name
                changed = True

    return name.strip()


def compute_centroid(geometry: dict) -> tuple[float, float]:
    """計算 GeoJSON geometry 的中心點 (lon, lat)。"""
    g_type = geometry.get("type")
    if g_type == "Polygon":
        coords = geometry["coordinates"][0]
    elif g_type == "MultiPolygon":
        coords = geometry["coordinates"][0][0]
    else:
        return 0.0, 0.0

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return float(np.mean(lons)), float(np.mean(lats))


def classify_meter_role(meter_name: str, osm_name: str = "") -> str:
    """分類電表角色，供後續避免重複加總使用。"""
    n = str(meter_name or "").upper()
    shared = "|" in str(osm_name or "")

    # 校區層級總表（需同時具備站/區域型特徵，避免把單棟校總區建物誤判）
    has_station_phrase = any(
        t in n for t in ["站總用電", "總站", "主站", "總配電", "全校", "總電源"]
    )
    zone_station = bool(re.search(r"(?:^|[^A-Z0-9])([一二三四五六七八九]|\\d+)區", n)) and (
        "站" in n or "總用電" in n or "總電源" in n
    )
    if has_station_phrase or zone_station:
        return "campus_total"

    if any(t in n for t in ["分表", "子表", "SUBMETER", "空調盤", "照明盤", "動力盤"]):
        return "submeter"
    if any(t in n for t in ["饋線", "FEEDER"]):
        return "feeder"
    if any(t in n for t in ["備援", "備用", "TEST", "試驗"]):
        return "backup"

    if any(t in n for t in ["總表", "總錶", "總用電", "總電源", "MAIN", "MCB", "GCB", "GCBM", "HTM", "VCB", "ACB"]):
        return "shared_total" if shared else "building_total"

    return "shared_total" if shared else "unknown"


def _load_osm_df(geojson_path: str | Path) -> pd.DataFrame:
    with open(geojson_path, encoding="utf-8") as f:
        geojson = json.load(f)

    rows = []
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        name = str(props.get("name", "") or "").strip()
        if not name:
            continue
        lon, lat = compute_centroid(feat.get("geometry", {}))
        rows.append(
            {
                "osm_id": props.get("osm_id"),
                "osm_name": name,
                "lon": round(float(lon), 6),
                "lat": round(float(lat), 6),
                "height": props.get("height", 10.5),
            }
        )
    return pd.DataFrame(rows, columns=["osm_id", "osm_name", "lon", "lat", "height"])


def _read_catalog_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "cp950", "big5"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (ValueError, TypeError) as exc:
            last_error = exc
            continue
    raise OSError(f"Failed to read catalog CSV: {path}") from last_error


def _load_official_names() -> list[str]:
    """
    Load official names from base + user patch catalogs.
    """
    paths = [
        _DATA_DIR / "NTU" / "ntu_official_buildings_patch.csv",
        _DATA_DIR / "NTU" / "ntu_official_buildings.csv",
        _DATA_DIR / "ntu_official_buildings_patch.csv",
        _DATA_DIR / "ntu_official_buildings.csv",
    ]
    names: set[str] = set()
    found_any = False

    for path in paths:
        if not path.exists():
            continue
        found_any = True
        try:
            df = _read_catalog_csv(path)
        except (OSError, pd.errors.EmptyDataError) as e:
            print(f"[Matcher] Ignore unreadable catalog: {path} ({e})")
            continue

        if "Name_ZH" in df.columns:
            for x in df["Name_ZH"].dropna().tolist():
                s = str(x).strip()
                if s:
                    names.add(s)

        if "Alias_ZH" in df.columns:
            for x in df["Alias_ZH"].dropna().tolist():
                for alias in str(x).split("|"):
                    s = alias.strip()
                    if s:
                        names.add(s)

    if not found_any:
        print("[Matcher] No official catalogs found, fallback to OSM-only matching")
        return []
    return sorted(names)


def _pick_first_osm_row(osm_df: pd.DataFrame, osm_name: str) -> Optional[pd.Series]:
    first = str(osm_name or "").split("|")[0].strip()
    if not first:
        return None
    rows = osm_df[osm_df["osm_name"] == first]
    if rows.empty:
        return None
    return rows.iloc[0]


def _fuzzy_extract_one(query: str, choices: list[str]) -> tuple[Optional[str], int]:
    if not query or not choices:
        return None, 0
    res = fuzz_process.extractOne(query, choices, scorer=fuzz.token_set_ratio)
    if not res:
        return None, 0
    return str(res[0]), int(res[1])


def _augment_summary_with_quality(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Attach corrected quality metrics from raw meter CSV.
    Valid hour rule: non-null and >= 0 (zero is valid).
    """
    out = summary.copy()
    meter_csv = _MODELS_DIR / "NTU_powerMeter_kW_hourly.csv"
    if (not meter_csv.exists()) or ("meter_name" not in out.columns):
        return out

    meters = [str(m) for m in out["meter_name"].dropna().tolist() if str(m).strip()]
    if not meters:
        return out

    try:
        header = pd.read_csv(meter_csv, nrows=0)
        dt_col = header.columns[0]
        available = [c for c in meters if c in header.columns]
        if not available:
            return out

        raw = pd.read_csv(meter_csv, usecols=[dt_col] + available, encoding="utf-8")
    except (ValueError, TypeError, KeyError):
        return out

    dt = pd.to_datetime(raw[dt_col], errors="coerce")
    raw = raw.loc[dt.notna()].copy()
    dt = dt.loc[dt.notna()]

    # Align with demo target year
    ymask = dt.dt.year == 2017
    if int(ymask.sum()) > 100:
        raw = raw.loc[ymask].copy()
        dt = dt.loc[ymask]
    if len(raw) == 0:
        return out

    n_total = int(len(raw))
    hours = dt.dt.hour.values
    num = raw[available].apply(pd.to_numeric, errors="coerce")
    valid = num.notna() & num.ge(0.0)
    zero = valid & num.eq(0.0)

    n_valid = valid.sum(axis=0).astype(float)
    mean_inc_zero = num.where(valid).mean(axis=0)
    coverage_ratio = (n_valid / max(n_total, 1)).astype(float)
    zero_ratio = (zero.sum(axis=0) / n_valid.replace(0, np.nan)).fillna(0.0)

    night_mask = (hours >= 0) & (hours <= 5)
    day_mask = (hours >= 9) & (hours <= 17)
    night_mean = num.where(valid).loc[night_mask].mean(axis=0)
    day_mean = num.where(valid).loc[day_mask].mean(axis=0)
    night_to_day = (night_mean / day_mean.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    p95 = num.where(valid).quantile(0.95, axis=0)
    peak_to_mean = (p95 / mean_inc_zero.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    quality = pd.DataFrame({
        "meter_name": available,
        "n_valid_hours_corrected": [float(n_valid.get(c, np.nan)) for c in available],
        "coverage_ratio_corrected": [float(coverage_ratio.get(c, np.nan)) for c in available],
        "mean_kw_including_zero": [float(mean_inc_zero.get(c, np.nan)) for c in available],
        "zero_ratio_valid": [float(zero_ratio.get(c, np.nan)) for c in available],
        "night_to_day_ratio": [float(night_to_day.get(c, np.nan)) for c in available],
        "peak_to_mean_ratio_p95": [float(peak_to_mean.get(c, np.nan)) for c in available],
    })

    overlap_cols = [c for c in quality.columns if c != "meter_name" and c in out.columns]
    if overlap_cols:
        out = out.drop(columns=overlap_cols)
    out = out.merge(quality, on="meter_name", how="left")
    if "n_valid_hours" in out.columns:
        out["n_valid_hours_raw"] = out["n_valid_hours"]
        out["n_valid_hours"] = out["n_valid_hours_corrected"].fillna(out["n_valid_hours"])
    else:
        out["n_valid_hours"] = out["n_valid_hours_corrected"]
    return out


def build_summary_from_meter_csv(
    meter_csv_path: str | Path,
    target_year: int = 2017,
    exclude_total_meter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a summary table directly from the raw hourly meter CSV so we can map
    all available channels instead of only the 54 modeled buildings.
    """
    meter_csv_path = Path(meter_csv_path)
    raw = pd.read_csv(meter_csv_path, encoding="utf-8", low_memory=False)
    if raw.empty:
        raise ValueError(f"Raw meter CSV is empty: {meter_csv_path}")

    dt_col = raw.columns[0]
    dt = pd.to_datetime(raw[dt_col], errors="coerce")
    raw = raw.loc[dt.notna()].copy()
    dt = dt.loc[dt.notna()]

    if int((dt.dt.year == target_year).sum()) > 100:
        mask = dt.dt.year == target_year
        raw = raw.loc[mask].copy()
        dt = dt.loc[mask]

    meter_cols = [c for c in raw.columns if c != dt_col]
    if exclude_total_meter:
        meter_cols = [c for c in meter_cols if str(c) != str(exclude_total_meter)]
    if not meter_cols:
        raise ValueError("No meter columns found after exclusions.")

    num = raw[meter_cols].apply(pd.to_numeric, errors="coerce")
    valid = num.notna() & num.ge(0.0)
    zero = valid & num.eq(0.0)

    n_total = int(len(num))
    hours = dt.dt.hour.values
    night_mask = (hours >= 0) & (hours <= 5)
    day_mask = (hours >= 9) & (hours <= 17)

    n_valid = valid.sum(axis=0).astype(float)
    mean_kw = num.where(valid).mean(axis=0)
    coverage_ratio = (n_valid / max(n_total, 1)).astype(float)
    zero_ratio = (zero.sum(axis=0) / n_valid.replace(0, np.nan)).fillna(0.0)
    night_mean = num.where(valid).loc[night_mask].mean(axis=0)
    day_mean = num.where(valid).loc[day_mask].mean(axis=0)
    night_to_day = (night_mean / day_mean.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    p95 = num.where(valid).quantile(0.95, axis=0)
    peak_to_mean = (p95 / mean_kw.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    summary = pd.DataFrame(
        {
            "meter_name": meter_cols,
            "n_valid_hours": [float(n_valid.get(c, np.nan)) for c in meter_cols],
            "mean_kw": [float(mean_kw.get(c, np.nan)) for c in meter_cols],
            "mean_kw_including_zero": [float(mean_kw.get(c, np.nan)) for c in meter_cols],
            "coverage_ratio_corrected": [float(coverage_ratio.get(c, np.nan)) for c in meter_cols],
            "zero_ratio_valid": [float(zero_ratio.get(c, np.nan)) for c in meter_cols],
            "night_to_day_ratio": [float(night_to_day.get(c, np.nan)) for c in meter_cols],
            "peak_to_mean_ratio_p95": [float(peak_to_mean.get(c, np.nan)) for c in meter_cols],
            "best_r2_oof": np.nan,
            "best_r_oof": np.nan,
            "best_cvrmse_oof": np.nan,
        }
    )
    return summary


def match_meters_to_buildings(
    geojson_path: str | Path,
    summary_csv_path: str | Path,
    threshold_auto: int = 80,
    threshold_review: int = 55,
) -> pd.DataFrame:
    """
    模糊比對電表名稱 -> 官方名稱 -> OSM 建物。
    """
    osm_df = _load_osm_df(geojson_path)
    osm_names = osm_df["osm_name"].astype(str).tolist()
    print(f"[Matcher] OSM: {len(osm_names)} 棟有名稱建物")

    official_names = _load_official_names()
    summary = pd.read_csv(summary_csv_path)
    summary = _augment_summary_with_quality(summary)
    print(f"[Matcher] 電表: {len(summary)} 棟")

    results = []
    for _, row in summary.iterrows():
        meter_name = str(row.get("meter_name", "") or "")
        raw_name = extract_building_name(meter_name)

        # alias normalization
        bldg_name = raw_name
        for old, new in ALIASES.items():
            if old in bldg_name:
                bldg_name = bldg_name.replace(old, new)

        matched_official: Optional[str] = None
        match_score = 0
        matched_osm: Optional[str] = None
        osm_score = 0
        search_target = bldg_name

        # 1) hardcoded mapping
        for k, v in HARDCODED_MAP.items():
            if k in bldg_name:
                matched_osm = v
                osm_score = 100
                matched_official = bldg_name
                match_score = 100
                break

        # 2) official names + OSM matching
        if not matched_osm:
            if official_names:
                if bldg_name in official_names:
                    matched_official = bldg_name
                    match_score = 100
                else:
                    for off_name in official_names:
                        if off_name in bldg_name or bldg_name in off_name:
                            matched_official = off_name
                            match_score = 90
                            break
                    if not matched_official and bldg_name:
                        m_name, m_score = _fuzzy_extract_one(bldg_name, official_names)
                        if m_name and m_score >= 80:
                            matched_official = m_name
                            match_score = m_score

            search_target = matched_official if matched_official else bldg_name
            if search_target and osm_names:
                if search_target in osm_names:
                    matched_osm = search_target
                    osm_score = 100
                else:
                    for o_name in osm_names:
                        if search_target in o_name or o_name in search_target:
                            matched_osm = o_name
                            osm_score = 90
                            break
                    if not matched_osm:
                        o_name, o_score = _fuzzy_extract_one(search_target, osm_names)
                        if o_name and o_score >= 85:
                            matched_osm = o_name
                            osm_score = o_score

        final_score = match_score if (matched_osm and matched_official) else osm_score
        if final_score >= threshold_auto:
            status = "auto_matched"
        elif final_score >= threshold_review:
            status = "needs_review"
        else:
            status = "unmatched"

        lon, lat = 0.0, 0.0
        osm_id = None
        height = 10.5
        if matched_osm is not None:
            osm_row = _pick_first_osm_row(osm_df, matched_osm)
            if osm_row is not None:
                lon = float(osm_row["lon"])
                lat = float(osm_row["lat"])
                osm_id = osm_row["osm_id"]
                height = float(osm_row["height"])

        best_r2 = pd.to_numeric(pd.Series([row.get("best_r2_oof", np.nan)]), errors="coerce").iloc[0]
        best_r = pd.to_numeric(pd.Series([row.get("best_r_oof", np.nan)]), errors="coerce").iloc[0]
        best_r2_corr_sq = (best_r * best_r) if pd.notna(best_r) else np.nan

        results.append(
            {
                "meter_name": meter_name,
                "building_name_extracted": search_target,
                "osm_name": matched_osm or "",
                "osm_id": osm_id,
                "match_score": final_score,
                "match_status": status,
                "lon": lon,
                "lat": lat,
                "height": height,
                "mean_kw": row.get("mean_kw", 0),
                "mean_kw_including_zero": row.get("mean_kw_including_zero", np.nan),
                "n_valid_hours": row.get("n_valid_hours", np.nan),
                "coverage_ratio": row.get("coverage_ratio_corrected", np.nan),
                "zero_ratio_valid": row.get("zero_ratio_valid", np.nan),
                "night_to_day_ratio": row.get("night_to_day_ratio", np.nan),
                "peak_to_mean_ratio_p95": row.get("peak_to_mean_ratio_p95", np.nan),
                "best_r2_oof": best_r2,
                "best_r_oof": best_r,
                "best_r2_from_corr_sq": best_r2_corr_sq,
                "best_cvrmse_oof": row.get("best_cvrmse_oof", 0),
                "meter_role": classify_meter_role(meter_name, matched_osm or ""),
            }
        )

    result_df = pd.DataFrame(results)
    n_auto = int((result_df["match_status"] == "auto_matched").sum())
    n_review = int((result_df["match_status"] == "needs_review").sum())
    n_none = int((result_df["match_status"] == "unmatched").sum())
    print(f"[Matcher] 匹配統計: auto={n_auto}, needs_review={n_review}, unmatched={n_none}")
    return result_df


def save_match_result(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Matcher] 已儲存: {path}")


def main():
    parser = argparse.ArgumentParser(description="Match meter names to OSM buildings")
    parser.add_argument("--geojson", default=None)
    parser.add_argument(
        "--summary",
        default=None,
        help="per_building_summary.csv 路徑",
    )
    parser.add_argument(
        "--meter-csv",
        default=None,
        help="Optional raw hourly meter CSV. If provided, build a full all-channel summary from raw data.",
    )
    parser.add_argument("--target-year", type=int, default=2017)
    parser.add_argument("--exclude-total-meter", default=None)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--threshold-auto", type=int, default=80)
    parser.add_argument("--threshold-review", type=int, default=55)
    args = parser.parse_args()

    geojson_path = resolve_project_path(args.geojson) if args.geojson else campus_data_dir("ntu", "ntu_buildings.geojson")
    output_path = resolve_project_path(args.output) if args.output else campus_data_dir("NTU", "meter_building_map.csv")
    summary_arg_path = resolve_project_path(args.summary) if args.summary else models_dir("v12_per_building_summary.csv")
    meter_csv_path = resolve_project_path(args.meter_csv) if args.meter_csv else None

    if meter_csv_path is not None:
        tmp_summary = build_summary_from_meter_csv(
            meter_csv_path=meter_csv_path,
            target_year=args.target_year,
            exclude_total_meter=args.exclude_total_meter,
        )
        tmp_summary_path = Path(output_path).with_suffix(".tmp_summary.csv")
        tmp_summary.to_csv(tmp_summary_path, index=False, encoding="utf-8-sig")
        summary_path = tmp_summary_path
    else:
        summary_path = Path(summary_arg_path)
        if not summary_path.exists():
            legacy = resolve_project_path("../idf_r2_optimizer/results_v12_building/per_building_summary.csv")
            if legacy.exists():
                summary_path = legacy

    df = match_meters_to_buildings(
        geojson_path=geojson_path,
        summary_csv_path=summary_path,
        threshold_auto=args.threshold_auto,
        threshold_review=args.threshold_review,
    )
    save_match_result(df, output_path)

    review = df[df["match_status"] == "needs_review"]
    if len(review) > 0:
        print("\n[Matcher] 以下匹配需人工審核:")
        for _, r in review.iterrows():
            print(f"  [{r['match_score']}] {r['meter_name']}")
            print(f"       提取: {r['building_name_extracted']}")
            print(f"       匹配: {r['osm_name']}")


if __name__ == "__main__":
    main()
