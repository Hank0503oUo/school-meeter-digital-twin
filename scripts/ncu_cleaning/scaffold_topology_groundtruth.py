"""
Scaffold a topology ground-truth CSV from 114 Q1 source data.

Produces:  outputs/ncu_114/topology_ground_truth_Q1.scaffold.csv

User edits the `role` column to one of:
  PRIMARY       — 該棟建物加總時要算這顆
  PRIMARY_ALT   — 替代主表,僅在 PRIMARY 全缺時 fallback
  SUB           — 子表,不要算(parent 已含)
  SKIP          — 不歸這棟(誤抓 / 共用 / 設施)

Plus optional columns user can fill:
  parent_meter_id  — 上游主表的 meter_id(可選,給 mass-balance 用)
  note             — 自由欄

The script suggests a default `role_suggested` using meter_id prefix + panel
naming heuristics. User can keep suggestions or overwrite per row.

Rows are sorted by (canonical_building, role_suggested, panel) so each building's
meters are grouped and you can scan top-down quickly in Excel.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean_ncu_114 import normalize_building_name, classify_meter, parse_multiplier
from canonicalize_building_names import NCU_RENAMES

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# Source: 114 Q1 MinerU output (most populated quarter for 114)
SRC = Path(r"G:/我的雲端硬碟/mineru/mineru_output_mineru_vlm/csv_by_quarter"
            r"/中大電表資料_114Q1_mineru_vlm.csv")
OUT_CSV = ROOT / "outputs" / "ncu_114" / "topology_ground_truth_Q1.scaffold.csv"


def suggest_role(meter_id: str, panel: str, multiplier: float | None) -> tuple[str, str]:
    """Return (role_suggested, reason)."""
    kind = classify_meter(meter_id)
    if kind == "placeholder":
        return "SKIP", "VL_* virtual placeholder (not read)"
    if kind == "aggregate":
        return "PRIMARY_ALT", "A1_* virtual total (fallback only)"

    p = (panel or "").strip().upper()
    # Top-level building MAIN / total
    if any(k in p for k in ("MAIN", "總表", "主表", "本表", "總幹線", "TOTAL")):
        return "PRIMARY", f"panel contains MAIN-like marker: {panel!r}"
    if "電力盤" in p or "電燈盤" in p:
        return "PRIMARY", f"電力盤/電燈盤 (parallel mains)"
    # Feeders / sub-panels
    if any(k in p for k in ("MVCB", "VCB", "高壓")):
        return "SUB", f"MVCB/VCB/高壓 feeder under MAIN"
    if any(k in p for k in ("FDR", "饋線", "MP", "ML")):
        return "SUB", f"FDR/MP/ML feeder or sub-panel"
    if any(k in p for k in ("AC", "ACR", "P3T", "PAC", "BF", "319-")):
        return "SUB", f"AC/P3T/BF detail sub-meter"
    if "分錶" in p or "分表" in p:
        return "SUB", f"分錶/分表 (sub-meter)"
    # Unknown — let user decide; default to SUB (safer to under-count than double-count)
    return "SUB", "unrecognized panel — please verify"


def main():
    if not SRC.exists():
        print(f"[ERROR] source not found: {SRC}")
        sys.exit(1)
    df = pd.read_csv(SRC, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    df = df.rename(columns={
        "PDF來源": "pdf_src",
        "建築物": "building_raw",
        "次": "seq",
        "表號": "meter_id",
        "開關箱": "panel",
        "安裝位置": "location",
        "倍數": "multiplier_raw",
        "使用單位": "usage_unit",
    })

    # Drop header-like rows
    df = df[~df["seq"].astype(str).str.strip().isin({"项次", "次數", "次数", "次",
                                                       "序號", "序号"})]
    # ffill building name within page
    df["building_raw"] = df.groupby("pdf_src")["building_raw"].ffill()
    df["building_cleaned"] = df["building_raw"].map(normalize_building_name)
    df["multiplier"] = df["multiplier_raw"].map(parse_multiplier)
    df = df.dropna(subset=["meter_id"]).copy()
    df["meter_id"] = df["meter_id"].astype(str).str.strip()
    df = df[df["meter_id"] != ""]

    # Apply canonical rename to building_cleaned where possible
    df["building"] = df["building_cleaned"].map(lambda n: NCU_RENAMES.get(n, n))

    # Dedupe to unique (meter_id, building)
    grouped = df.groupby(["meter_id", "building"], dropna=False).agg(
        panel=("panel", lambda s: ", ".join(sorted({str(x).strip() for x in s if pd.notna(x)}))),
        location=("location", lambda s: ", ".join(sorted({str(x).strip()[:30] for x in s if pd.notna(x)}))),
        multiplier=("multiplier", lambda s: max([x for x in s if pd.notna(x)] or [None])),
        usage_unit=("usage_unit", lambda s: ", ".join(sorted({str(x).strip()[:30] for x in s if pd.notna(x)}))),
        n_rows=("meter_id", "count"),
    ).reset_index()

    # Suggest role + reason
    suggestions = grouped.apply(
        lambda r: pd.Series(
            suggest_role(r["meter_id"], r["panel"], r["multiplier"]),
            index=["role_suggested", "suggestion_reason"],
        ),
        axis=1,
    )
    grouped = pd.concat([grouped, suggestions], axis=1)

    # User-editable columns (blank by default)
    grouped["role"] = ""               # USER FILL: PRIMARY / PRIMARY_ALT / SUB / SKIP
    grouped["parent_meter_id"] = ""    # USER OPTIONAL: which meter is parent
    grouped["note"] = ""               # USER OPTIONAL

    # Final column order
    cols = [
        "building",                # canonical building name
        "meter_id",                # 表號
        "panel",                   # 開關箱(可能多值合併)
        "multiplier",              # 倍率
        "location",                # 安裝位置
        "usage_unit",              # 使用單位
        "n_rows",                  # 在 Q1 出現幾次(通常 1)
        "role_suggested",          # 我猜的(供參考,可信時複製到 role)
        "suggestion_reason",       # 為何這樣猜
        "role",                    # USER FILL ← 主要要填
        "parent_meter_id",         # USER OPTIONAL
        "note",                    # USER OPTIONAL
    ]
    grouped = grouped[cols]

    # Sort: building, then role_suggested order (PRIMARY first), then panel
    role_order = {"PRIMARY": 0, "PRIMARY_ALT": 1, "SUB": 2, "SKIP": 3}
    grouped["_sort_role"] = grouped["role_suggested"].map(role_order).fillna(9)
    grouped = grouped.sort_values(["building", "_sort_role", "panel"]).drop(columns=["_sort_role"])

    grouped.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Wrote scaffold: {OUT_CSV}")
    print()
    print(f"  unique buildings:  {grouped['building'].nunique()}")
    print(f"  unique meters:     {len(grouped)}")
    print(f"  suggested PRIMARY:     {(grouped['role_suggested']=='PRIMARY').sum()}")
    print(f"  suggested PRIMARY_ALT: {(grouped['role_suggested']=='PRIMARY_ALT').sum()}")
    print(f"  suggested SUB:         {(grouped['role_suggested']=='SUB').sum()}")
    print(f"  suggested SKIP:        {(grouped['role_suggested']=='SKIP').sum()}")
    print()
    print("=== Top 10 buildings by meter count (these are your topology hotspots) ===")
    top = grouped.groupby("building").agg(
        n_meters=("meter_id", "count"),
        suggested_primary=("role_suggested", lambda s: (s=="PRIMARY").sum()),
        suggested_sub=("role_suggested", lambda s: (s=="SUB").sum()),
    ).sort_values("n_meters", ascending=False).head(10)
    print(top.to_string())


if __name__ == "__main__":
    main()
