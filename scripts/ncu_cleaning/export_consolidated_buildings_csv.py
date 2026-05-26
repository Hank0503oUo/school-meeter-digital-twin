"""
Export a single consolidated CSV of all NCU buildings, merging:
  1. DEMO buildings_enhanced.geojson (OSM-confirmed buildings with levels/footprint)
  2. v4 + meter audit (all years) — which buildings actually have meters
  3. NCU 建築物 infro.xlsx — user's manual building info table

Output:
  C:\\Users\\User\\Downloads\\中大建築物_合併清單.csv

Sorted by: (has_meter desc, canonical_name asc) so the most actionable
rows (buildings with meters) sit at the top.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd


GEOJSON_PATH = Path(r"D:\idf優化\demo\data\NCU\buildings_enhanced.geojson")
V4_PATH = Path(r"C:\Users\User\Downloads\中大電表資料\per_quarter\中大電表資料_109Q1_v4.csv")
AUDIT_PATHS = [
    Path(r"C:\Users\User\demo\outputs\ncu_109\meter_audit.csv"),
    Path(r"C:\Users\User\demo\outputs\ncu_110\meter_audit.csv"),
    Path(r"C:\Users\User\demo\outputs\ncu_111\meter_audit.csv"),
    Path(r"C:\Users\User\demo\outputs\ncu_114\meter_audit.csv"),
]
XLSX_PATH = Path(r"C:\Users\User\Downloads\NCU 建築物 infro.xlsx")
OUT_CSV = Path(r"C:\Users\User\Downloads\中大建築物_合併清單.csv")


NCU_RENAMES = {
    "中正圖書館": "總圖書館",
    "研究生宿舍": "曦望居",
    "中大十舍": "中大會館",
    "館(理學院)": "科學二館(理學院大樓)",
    "科二館(理學院)": "科學二館(理學院大樓)",
    "管二館": "管理學院二館",
    "科四館(健雄館)": "科學四館",
    "科四館": "科學四館",
    "工五館B棟增建": "工五館(A、B棟)",
    "工五館B棟増建": "工五館(A、B棟)",
    "教職員單身一舍": "單一舍",
    "教職員單身二舍": "單二舍",
    "教職員單身四舍": "單四舍",
    "游藝館": "遊藝館",
    "學生女一宿舍": "女一舍",
    "學生女二宿舍": "女二舍",
    "學生女三宿舍": "女三舍",
    "學生女四宿舍": "女四舍",
    "學生女五舍": "女五舍",
    "學生女五舍(原男四舍)": "女五舍",
    "學生女六宿舍": "女六舍",
    "學生女六宿舍(原男五舍)": "女六舍",
    "學生男三宿舍": "男三舍",
    "學生男五舍": "男五舍",
    "學生男六宿舍": "男六舍",
    "學生男七宿舍": "男七舍",
    "學生男九宿舍": "男九舍",
    "學生男十一宿舍": "男十一舍",
    "學生男十二宿舍": "男十二舍",
    "學生男十三宿舍": "男13舍",
    "學生女十四舍": "女十四舍",
    "學生自主學習空間": "iHouse享想空間",
    "享想空間": "iHouse享想空間",
    "科一館": "科學一館",
    "科三館": "科學三館",
    "科五館": "科學五館",
    "文一館": "文學一館",
    "文二館": "文學二館",
    "電機館": "電機系",
    "機械館": "工程三館(機械館)",
    "工一館": "工程一館",
    "工程四館二期(機電實驗室)": "機電實驗室",
    "工程四館三期(大型力學實驗室)": "大型力學實驗室",
    "綜教館": "綜教館(語言中心)",
    "客家學院大樓": "客家學院",
    "理學院教學館": "理學院教學館(普化實驗大樓)",
    "太空遙測中心": "太空及遙測研究中心",
    "程五館C棟": "工程五館",
    "實習三廠(機械鑄造廠)": "實習三廠",
    "國際學舍(原二招)": "國際學生宿舍",
    "國際學舍": "國際學生宿舍",
    "學生活動中心(志道樓)": "志道樓",
    "體育館(依仁堂)": "依仁堂",
    "雷達車車庫": "車庫",
    "倉庫": "車庫",
    "垃圾集中處理場": "資源回收及垃圾處理場",
    "前瞻科技中心": "前瞻科技研究中心",
}


_PAREN = re.compile(r"[（()）]")
_SP = re.compile(r"\s+")


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def canonical(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    return NCU_RENAMES.get(s, s)


def name_key(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return _SP.sub("", _PAREN.sub("", s.strip())).lower()


def load_demo() -> pd.DataFrame:
    with GEOJSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for feat in data.get("features", []):
        props = feat.get("properties", {}) or {}
        name = (props.get("name") or "").strip()
        if not name:
            continue
        rows.append({
            "demo_name": name,
            "osm_id": props.get("osm_id"),
            "demo_building_type": props.get("building_type"),
            "demo_levels": props.get("levels"),
            "demo_height_m": props.get("height"),
            "demo_footprint_m2": props.get("footprint_area_m2"),
        })
    return pd.DataFrame(rows)


def load_meter_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    v4 = pd.read_csv(V4_PATH, dtype=str, encoding="utf-8-sig")
    v4.columns = [c.strip().lstrip("﻿") for c in v4.columns]
    bld = "建築物" if "建築物" in v4.columns else v4.columns[1]
    for b, n in v4.groupby(bld).size().items():
        if isinstance(b, str) and b.strip():
            cn = canonical(b)
            counts[cn] = counts.get(cn, 0) + int(n)
    for ap in AUDIT_PATHS:
        if not ap.exists():
            continue
        a = pd.read_csv(ap, dtype=str, encoding="utf-8-sig")
        a.columns = [c.strip().lstrip("﻿") for c in a.columns]
        if "building" not in a.columns or "meter_id" not in a.columns:
            continue
        a = a[a["building"].notna() & a["meter_id"].notna()]
        ud = a.drop_duplicates(["building", "meter_id"])
        for b, n in ud.groupby("building").size().items():
            cn = canonical(str(b).strip())
            counts[cn] = max(counts.get(cn, 0), int(n))
    return counts


def main() -> None:
    configure_stdout()

    demo = load_demo()
    print(f"DEMO geojson buildings: {len(demo)}")

    meter_counts = load_meter_counts()
    print(f"Meter unique canonical buildings: {len(meter_counts)}")

    xlsx = pd.read_excel(XLSX_PATH)
    xlsx.columns = [str(c).strip() for c in xlsx.columns]
    print(f"xlsx rows: {len(xlsx)}")

    # Build unified set of canonical names
    all_names: set[str] = set()
    all_names.update(demo["demo_name"].tolist())
    all_names.update(meter_counts.keys())

    # xlsx names mapped via NCU_RENAMES
    name_col = "建築物正式名稱"
    xlsx_canonical: dict[str, dict] = {}
    for _, r in xlsx.iterrows():
        n = str(r[name_col]).strip()
        if not n or n.lower() == "nan":
            continue
        # Try DEMO direct, then NCU_RENAMES, then keep original
        if n in demo["demo_name"].values:
            cn = n
        elif n in NCU_RENAMES and NCU_RENAMES[n] in demo["demo_name"].values:
            cn = NCU_RENAMES[n]
        else:
            # Try fuzzy via name_key against demo
            k = name_key(n)
            match = None
            for dn in demo["demo_name"]:
                if name_key(dn) == k:
                    match = dn
                    break
            cn = match if match else n
        xlsx_canonical.setdefault(cn, {
            "xlsx_name": n,
            "xlsx_竣工年份": r.get("竣工年份 (屋齡)"),
            "xlsx_總樓地板面積_m2": r.get("總樓地板面積 (平方公尺)"),
            "xlsx_地上樓層數": r.get("地上樓層數"),
            "xlsx_地下樓層數": r.get("地下樓層數"),
            "xlsx_建築代碼": r.get("建築代碼"),
        })
        all_names.add(cn)

    # Index DEMO data by name (dedupe: keep first occurrence per name)
    demo_dedup = demo.drop_duplicates(subset=["demo_name"], keep="first")
    demo_idx = demo_dedup.set_index("demo_name").to_dict("index")

    # Build final rows
    rows = []
    for cn in sorted(all_names):
        demo_data = demo_idx.get(cn, {})
        xlsx_data = xlsx_canonical.get(cn, {})
        meter_n = meter_counts.get(cn, 0)

        has_demo = bool(demo_data)
        has_meter = meter_n > 0
        has_xlsx = bool(xlsx_data)

        sources = []
        if has_demo: sources.append("DEMO")
        if has_meter: sources.append("METER")
        if has_xlsx: sources.append("XLSX")

        if has_demo and has_meter and has_xlsx:
            status = "all_three"
        elif has_demo and has_meter:
            status = "demo+meter (no xlsx)"
        elif has_demo and has_xlsx:
            status = "demo+xlsx (no meter)"
        elif has_meter and has_xlsx:
            status = "meter+xlsx (not in demo geojson)"
        elif has_demo:
            status = "demo_only (sub-facility?)"
        elif has_meter:
            status = "meter_only (no demo geojson, no xlsx)"
        elif has_xlsx:
            status = "xlsx_only (no meter, no demo)"
        else:
            status = "??"

        rows.append({
            "canonical_name": cn,
            "sources": "+".join(sources),
            "status": status,
            "meter_count": meter_n,
            "demo_levels": demo_data.get("demo_levels"),
            "demo_height_m": demo_data.get("demo_height_m"),
            "demo_footprint_m2": demo_data.get("demo_footprint_m2"),
            "demo_building_type": demo_data.get("demo_building_type"),
            "demo_osm_id": demo_data.get("osm_id"),
            "xlsx_name": xlsx_data.get("xlsx_name", ""),
            "xlsx_建築代碼": xlsx_data.get("xlsx_建築代碼", ""),
            "xlsx_竣工年份": xlsx_data.get("xlsx_竣工年份"),
            "xlsx_總樓地板面積_m2": xlsx_data.get("xlsx_總樓地板面積_m2"),
            "xlsx_地上樓層數": xlsx_data.get("xlsx_地上樓層數"),
            "xlsx_地下樓層數": xlsx_data.get("xlsx_地下樓層數"),
        })

    df = pd.DataFrame(rows)

    # Sort: buildings with meters first, then by name
    df["_sort"] = (df["meter_count"] == 0).astype(int)
    df = df.sort_values(["_sort", "canonical_name"], kind="mergesort").drop(columns="_sort")

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # Summary
    print()
    print("=" * 60)
    print(f"Wrote: {OUT_CSV}")
    print(f"Total rows: {len(df)}")
    print()
    print("By status:")
    for s, n in df["status"].value_counts().items():
        print(f"  {s:45s} {n}")

    print()
    print("Top 10 with most meters:")
    top = df.nlargest(10, "meter_count")[["canonical_name", "sources", "meter_count"]]
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
