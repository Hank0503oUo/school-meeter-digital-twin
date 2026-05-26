"""
Three-way alignment using DEMO's canonical building list as the spine.

Source of truth (DEMO):
  D:\\idf優化\\demo\\data\\NCU\\buildings_enhanced.geojson  (88 buildings, name/levels/height/footprint)
  D:\\idf優化\\demo\\scripts\\ncu_cleaning\\canonicalize_building_names.py  (NCU_RENAMES dict)

Cross-references:
  v4 meter list                                  → which buildings have metered data
  ncu_109/110/111/114 meter_audit.csv (audits)   → which buildings have any audit trace
  NCU 建築物 infro.xlsx                            → user-provided manual building info

Output:
  C:\\Users\\User\\Downloads\\NCU 建築物 infro_demo_aligned.xlsx
  C:\\Users\\User\\Downloads\\NCU 建築物 infro_demo_alignment_report.csv
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
OUT_XLSX = Path(r"C:\Users\User\Downloads\NCU 建築物 infro_demo_aligned.xlsx")
REPORT_CSV = Path(r"C:\Users\User\Downloads\NCU 建築物 infro_demo_alignment_report.csv")


# NCU_RENAMES is the mapping: long-form meter name → DEMO canonical name.
# Copied from D:\idf優化\demo\scripts\ncu_cleaning\canonicalize_building_names.py
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
    "中大幼稚園": "中大幼稚園",
    "科一館": "科學一館",
    "科三館": "科學三館",
    "科五館": "科學五館",
    "文一館": "文學一館",
    "文二館": "文學二館",
    "電機館": "電機系",
    "機械館": "工程三館(機械館)",
    "工一館": "工程一館",
    "工五館(A、B棟)": "工五館(A、B棟)",
    "工四館一期(環化館)": "工四館一期(環化館)",
    "工程四館二期(機電實驗室)": "機電實驗室",
    "工程四館三期(大型力學實驗室)": "大型力學實驗室",
    "志希館": "志希館",
    "據德樓": "據德樓",
    "綜教館": "綜教館(語言中心)",
    "客家學院大樓": "客家學院",
    "國鼎光電大樓": "國鼎光電大樓",
    "國鼎圖書資料館": "國鼎圖書資料館",
    "理學院教學館": "理學院教學館(普化實驗大樓)",
    "太空遙測中心": "太空及遙測研究中心",
    "鴻經館": "鴻經館",
    "管二館": "管理學院二館",
    "程五館C棟": "工程五館",
    "風洞實驗室及品保中心": "風洞實驗室及品保中心",
    "實習三廠(機械鑄造廠)": "實習三廠",
    "國際學舍(原二招)": "國際學生宿舍",
    "學生活動中心(志道樓)": "志道樓",
    "體育館(依仁堂)": "依仁堂",
    "11號深水井": "11號深水井",
    "6號水井": "6號水井",
    "第1水塔": "第1水塔",
    "自來水3號水塔": "自來水3號水塔",
    "地下水1號水塔": "地下水1號水塔",
    "地下水4號水塔": "地下水4號水塔",
    "雷達站": "雷達站",
    "雷達車車庫": "車庫",
    "氣象觀測站": "氣象觀測站",
    "垃圾集中處理場": "資源回收及垃圾處理場",
    "大氣環境實驗室": "大氣環境實驗室",
    "倉庫": "車庫",
    "中大會館": "中大會館",
    "人文社會科學大樓": "人文社會科學大樓",
    "行政大樓": "行政大樓",
    "總圖書館": "總圖書館",
    "大講堂": "大講堂",
    "松苑餐廳": "松苑餐廳",
    "產學營運中心": "產學營運中心",
    "研究中心大樓二期": "研究中心大樓二期",
    "科思創研究中心": "科思創研究中心",
    "前瞻科技中心": "前瞻科技研究中心",
    "教學研究綜合大樓": "教學研究綜合大樓",
}


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


_PAREN = re.compile(r"[（()）]")
_SP = re.compile(r"\s+")


def name_key(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return _SP.sub("", _PAREN.sub("", s.strip())).lower()


def canonical(name: str) -> str:
    """Convert any meter-side name to DEMO canonical via NCU_RENAMES."""
    if not isinstance(name, str):
        return ""
    s = name.strip()
    return NCU_RENAMES.get(s, s)


def load_demo_buildings(path: Path) -> pd.DataFrame:
    """Load DEMO's 88-building geojson and return a clean DataFrame."""
    with path.open("r", encoding="utf-8") as f:
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
            "building_type": props.get("building_type"),
            "operator": props.get("operator"),
            "levels": props.get("levels"),
            "height_m": props.get("height"),
            "footprint_area_m2": props.get("footprint_area_m2"),
            "data_source": props.get("data_source"),
        })
    df = pd.DataFrame(rows)
    df["demo_name_key"] = df["demo_name"].map(name_key)
    return df


def collect_meter_buildings(v4: pd.DataFrame, audit_paths: list[Path]) -> dict[str, int]:
    """Return {canonical_demo_name: meter_count} aggregated from all sources."""
    counts: dict[str, int] = {}

    # v4 buildings
    bld_col = "建築物" if "建築物" in v4.columns else v4.columns[1]
    for b, n in v4.groupby(bld_col).size().items():
        if not isinstance(b, str) or not b.strip():
            continue
        canon = canonical(b)
        counts[canon] = counts.get(canon, 0) + int(n)

    # Audit buildings (count unique meter_ids per building)
    for ap in audit_paths:
        if not ap.exists():
            continue
        a = pd.read_csv(ap, dtype=str, encoding="utf-8-sig")
        a.columns = [c.strip().lstrip("﻿") for c in a.columns]
        if "building" not in a.columns or "meter_id" not in a.columns:
            continue
        a = a[a["building"].notna() & a["meter_id"].notna()]
        ud = a.drop_duplicates(["building", "meter_id"])
        for b, n in ud.groupby("building").size().items():
            canon = canonical(str(b).strip())
            existing = counts.get(canon, 0)
            counts[canon] = max(existing, int(n))
    return counts


def main() -> None:
    configure_stdout()

    # 1. DEMO canonical buildings (88-row spine)
    demo = load_demo_buildings(GEOJSON_PATH)
    print(f"DEMO buildings_enhanced.geojson: {len(demo)} buildings")

    # 2. Meter side: v4 + audits → {canonical_name: meter_count}
    v4 = pd.read_csv(V4_PATH, dtype=str, encoding="utf-8-sig")
    v4.columns = [c.strip().lstrip("﻿") for c in v4.columns]
    meter_counts = collect_meter_buildings(v4, AUDIT_PATHS)
    print(f"Unique canonical buildings with meters: {len(meter_counts)}")

    # 3. xlsx side
    xlsx = pd.read_excel(XLSX_PATH)
    xlsx.columns = [str(c).strip() for c in xlsx.columns]
    name_col = "建築物正式名稱"
    if name_col not in xlsx.columns:
        for c in xlsx.columns:
            if "名稱" in c or "建築" in c:
                name_col = c
                break
    print(f"xlsx rows: {len(xlsx)}, name col: {name_col!r}")

    # Map xlsx names → DEMO canonical via NCU_RENAMES + fuzzy on demo set
    demo_key_to_name = dict(zip(demo["demo_name_key"], demo["demo_name"]))
    demo_name_set = set(demo["demo_name"])

    def match_to_demo(xlsx_name: str) -> str | None:
        if not isinstance(xlsx_name, str) or not xlsx_name.strip():
            return None
        s = xlsx_name.strip()
        # Direct via rename dict
        if s in NCU_RENAMES and NCU_RENAMES[s] in demo_name_set:
            return NCU_RENAMES[s]
        if s in demo_name_set:
            return s
        # Strip parens and try
        bare = _PAREN.sub("", s).strip()
        if bare in demo_name_set:
            return bare
        # Key match (normalized)
        k = name_key(s)
        if k and k in demo_key_to_name:
            return demo_key_to_name[k]
        # Substring fallback
        for dname in demo_name_set:
            dkey = name_key(dname)
            if not dkey:
                continue
            if k and len(k) >= 3 and (k in dkey or dkey in k):
                return dname
        return None

    xlsx["_matched_demo_name"] = xlsx[name_col].apply(match_to_demo)

    # 4. Build aligned output: spine = DEMO 88 buildings, then append extras
    rows: list[dict] = []
    matched_demo: set[str] = set()

    # 4a. For each DEMO building, attach matching xlsx row + meter count
    xlsx_matches_by_demo: dict[str, list[int]] = {}
    for idx, dname in zip(xlsx.index, xlsx["_matched_demo_name"]):
        if isinstance(dname, str) and dname:
            xlsx_matches_by_demo.setdefault(dname, []).append(idx)

    for _, drow in demo.iterrows():
        dname = drow["demo_name"]
        meter_n = meter_counts.get(dname, 0)
        xlsx_idxs = xlsx_matches_by_demo.get(dname, [])
        if xlsx_idxs:
            x = xlsx.loc[xlsx_idxs[0]]
            xlsx_name = str(x[name_col]).strip()
            row = {
                "demo_canonical_name": dname,
                "xlsx_name": xlsx_name,
                "meter_count": meter_n,
                "demo_levels": drow.get("levels"),
                "demo_height_m": drow.get("height_m"),
                "demo_footprint_area_m2": drow.get("footprint_area_m2"),
                "demo_building_type": drow.get("building_type"),
                "xlsx_竣工年份": x.get("竣工年份 (屋齡)"),
                "xlsx_總樓地板面積": x.get("總樓地板面積 (平方公尺)"),
                "xlsx_地上樓層數": x.get("地上樓層數"),
                "xlsx_地下樓層數": x.get("地下樓層數"),
                "status": "matched" if meter_n > 0 else "in_demo_xlsx_no_meter",
            }
        else:
            row = {
                "demo_canonical_name": dname,
                "xlsx_name": "",
                "meter_count": meter_n,
                "demo_levels": drow.get("levels"),
                "demo_height_m": drow.get("height_m"),
                "demo_footprint_area_m2": drow.get("footprint_area_m2"),
                "demo_building_type": drow.get("building_type"),
                "xlsx_竣工年份": None,
                "xlsx_總樓地板面積": None,
                "xlsx_地上樓層數": None,
                "xlsx_地下樓層數": None,
                "status": "in_demo_meter_no_xlsx" if meter_n > 0 else "in_demo_only",
            }
        rows.append(row)
        matched_demo.add(dname)

    # 4b. Meter-canonical names that didn't land on any DEMO building
    for canon_name, n in meter_counts.items():
        if canon_name in matched_demo:
            continue
        rows.append({
            "demo_canonical_name": "",
            "xlsx_name": "",
            "meter_count": n,
            "demo_levels": None,
            "demo_height_m": None,
            "demo_footprint_area_m2": None,
            "demo_building_type": None,
            "xlsx_竣工年份": None,
            "xlsx_總樓地板面積": None,
            "xlsx_地上樓層數": None,
            "xlsx_地下樓層數": None,
            "status": f"meter_only (raw_name={canon_name})",
        })

    # 4c. xlsx rows that didn't match any DEMO building
    for idx, dname in zip(xlsx.index, xlsx["_matched_demo_name"]):
        if isinstance(dname, str) and dname:
            continue
        x = xlsx.loc[idx]
        xlsx_name = str(x[name_col]).strip()
        if not xlsx_name or xlsx_name == "nan":
            continue
        rows.append({
            "demo_canonical_name": "",
            "xlsx_name": xlsx_name,
            "meter_count": 0,
            "demo_levels": None,
            "demo_height_m": None,
            "demo_footprint_area_m2": None,
            "demo_building_type": None,
            "xlsx_竣工年份": x.get("竣工年份 (屋齡)"),
            "xlsx_總樓地板面積": x.get("總樓地板面積 (平方公尺)"),
            "xlsx_地上樓層數": x.get("地上樓層數"),
            "xlsx_地下樓層數": x.get("地下樓層數"),
            "status": "xlsx_only_no_demo_no_meter",
        })

    final = pd.DataFrame(rows)
    final.to_excel(OUT_XLSX, index=False)

    # 5. Report
    counts = final["status"].value_counts()
    print()
    print("=" * 60)
    print("Alignment report")
    print("=" * 60)
    print(counts.to_string())
    print()
    print(f"Wrote: {OUT_XLSX}")

    counts.to_csv(REPORT_CSV, encoding="utf-8-sig", header=["count"])
    print(f"Wrote: {REPORT_CSV}")

    print()
    print("--- Sample: in_demo_only (DEMO 有，但無電表 & 無 xlsx) ---")
    for _, r in final[final["status"] == "in_demo_only"].head(20).iterrows():
        print(f"  {r['demo_canonical_name']}")

    print()
    print("--- Sample: meter_only (有電表但 DEMO 沒這棟) ---")
    for _, r in final[final["status"].str.startswith("meter_only")].head(20).iterrows():
        n = r["status"].split("raw_name=")[1].rstrip(")")
        print(f"  {n}  (meters={r['meter_count']})")


if __name__ == "__main__":
    main()
