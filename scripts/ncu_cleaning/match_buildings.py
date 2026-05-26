"""
Match cleaned NCU 114 building names → buildings.geojson osm_id (UID).

Strategy:
  1. Load monthly_kwh.csv (cleaned names).
  2. Load buildings.geojson (osm_id + name fields).
  3. For each cleaned name, try in order:
     a) Manual alias (hand-curated synonyms)
     b) Exact match against geojson name
     c) Exact match after stripping parens / whitespace
     d) Fuzzy match (rapidfuzz partial_ratio ≥ 80) against geojson names

Output:
  outputs/ncu_114/name_to_uid.csv         building, osm_id, geojson_name, match_type, score
  outputs/ncu_114/monthly_kwh_with_uid.csv   monthly_kwh.csv joined with osm_id
  outputs/ncu_114/match_report.md
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "ncu_114"
GEOJSON = ROOT / "campuses" / "ncu" / "data" / "buildings.geojson"
MONTHLY = OUT_DIR / "monthly_kwh.csv"

# Manual aliases — meter CSV name → geojson name.
# Keys are CLEANED names from monthly_kwh.csv.
MANUAL_ALIASES: dict[str, str] = {
    "科一館": "科學一館" if False else "科一館",  # geojson has no 科一館 entry
    "科二館(理學院)": "科學二館(理學院大樓)",
    "科二館": "科學二館(理學院大樓)",
    "館(理學院)": "科學二館(理學院大樓)",
    "科三館": "科三館",
    "科四館(健雄館)": "科學四館",
    "科四館": "科學四館",
    "科五館(S5)": "科五館(S5)",
    "科五館": "科五館(S5)",
    "工一館": "工程一館",
    "工程四館一期(環化館)": "工四館一期(環化館)",
    "工四館一期(環化館)": "工四館一期(環化館)",
    "工程四館二期(機電實驗室)": "機電實驗室",
    "工程四館三期(大型力學實驗室)": "大型力學實驗室",
    "工五館(A、B棟)": "工五館(A、B棟)",
    "工程五館": "工五館(A、B棟)",
    "工程五館大樓": "工五館(A、B棟)",
    "工五館B棟増建": "工五館(A、B棟)",
    "工程五館C棟": "工五館(A、B棟)",
    "太空及遙測研究中心": "太空遙測中心",
    "機械館": "工程三館(機械館)",
    "實習三廠(機械鑄造廠)": "實習一廠",
    "中正圖書館": "總圖書館",
    "總圖書館": "總圖書館",
    "管二館": "管理學院二館",
    "文一館": "文學一館",
    "文一餡": "文學一館",         # OCR 餡 = 館
    "文二館": "文學二館",
    "教學研究綜合大樓": "教學研究綜合大樓",
    "人文社會科學大樓": "人文社會科學大樓",
    "行政大樓": "行政大樓",
    "志希館": "志希館",
    "綜教館": "綜教館(語言中心)",
    "中大十舍(原中大會館)": "中大會館",
    "享想空間(原女五舍)": "iHouse享想空間",
    "國際學舍(原二招)": "國際學生宿舍",
    "曦望居(原研究生宿舍)": "曦望居",
    "中大國民運動中心": "中大國民運動中心",
    "大講堂": "大講堂",
    "大禮堂": "大禮堂",
    "松苑餐廳": "松苑餐廳",
    "遊藝館": "遊藝館",
    "游藝館": "遊藝館",
    "體育館(依仁堂)": "依仁堂",
    "依仁堂": "依仁堂",
    "羽球館": "羽球館",
    "鴻經館": "鴻經館",
    "教職員單身一舍": "單一舍",
    "教職員單身二舍": "單二舍",
    "教職員單身四舍": "單四舍",
    "學生女一宿舍": "女一舍",
    "學生女二宿舍": "女二舍",
    "學生女三宿舍": "女三舍",
    "學生女四宿舍": "女四舍",
    "學生女十四舍": "女十四舍",
    "學生男三宿舍": "男三舍",
    "學生男五舍": "男五舍",
    "學生男六舍": "男六舍",
    "男六宿舍": "男六舍",
    "學生男七舍": "男七舍",
    "學生男九宿舍": "男九舍",
    "學生男十一宿舍": "男十一舍",
    "生男十一宿舍": "男十一舍",
    "學生男十二宿舍": "男十二舍",
    "學生男十三宿舍": "男13舍",
    "客家學院": "客家學院",
    "地球科學院": "地球科學院",
    "通訊系": "通訊系",
    "電機系": "電機系",
    "電機館": "電機館",
    "資策會": "資策會",
    "依仁堂": "依仁堂",
    "據德樓": "據德樓",
    "志道樓": "志道樓",
    "土木品保中心": "土木品保中心",
    "校長宿舍": "校長宿舍",
    "車庫": "車庫",
    "中大幼稚園": "中大幼稚園",
    "資源回收及垃圾處理場": "資源回收及垃圾處理場",
    "垃圾集中處理場": "垃圾集中處理場",
    "產學營運中心": "產學營運中心",
    "國鼎光電大樓": "國鼎光電大樓",
    "國鼎圖書資料館": "國鼎圖書資料館",
    "理學院教學館(普化實驗大樓)": "理學院教學館(普化實驗大樓)",
    "第二行政中心": "第二行政中心",
    "風洞實驗室及品保中心": "風洞實驗室及品保中心",
    "科思創研究中心": "科思創研究中心",
    "研究中心大樓二期": "研究中心大樓二期",
    "高壓變電站": "高壓變電站",
    "體育器材室": "體育器材室",
}

_PAREN = re.compile(r"[()（）]")
_SPACES = re.compile(r"\s+")


def strip_punct(s: str) -> str:
    s = _PAREN.sub("", s or "")
    s = _SPACES.sub("", s)
    return s


def load_geojson_names() -> pd.DataFrame:
    with open(GEOJSON, encoding="utf-8") as f:
        gj = json.load(f)
    rows = []
    for ft in gj["features"]:
        p = ft.get("properties", {})
        nm = (p.get("name") or "").strip()
        if not nm:
            continue
        rows.append({
            "osm_id": p.get("osm_id"),
            "geojson_name": nm,
            "geojson_name_norm": strip_punct(nm),
            "footprint_area_m2": p.get("footprint_area_m2"),
            "building_type": p.get("building_type"),
        })
    return pd.DataFrame(rows)


def match_one(name: str, gj_df: pd.DataFrame) -> dict:
    if not isinstance(name, str) or not name:
        return {"osm_id": None, "geojson_name": None, "match_type": "no_name", "score": 0.0}
    # 1) manual alias
    aliased = MANUAL_ALIASES.get(name)
    target = aliased or name
    # 2) exact match
    exact = gj_df[gj_df["geojson_name"] == target]
    if len(exact):
        r = exact.iloc[0]
        return {
            "osm_id": int(r["osm_id"]),
            "geojson_name": r["geojson_name"],
            "match_type": "alias_exact" if aliased else "exact",
            "score": 100.0,
        }
    # 3) normalized (strip parens/spaces)
    target_norm = strip_punct(target)
    norm = gj_df[gj_df["geojson_name_norm"] == target_norm]
    if len(norm):
        r = norm.iloc[0]
        return {
            "osm_id": int(r["osm_id"]),
            "geojson_name": r["geojson_name"],
            "match_type": "norm",
            "score": 95.0,
        }
    # 4) fuzzy match
    candidates = gj_df["geojson_name"].tolist()
    best = process.extractOne(target, candidates, scorer=fuzz.WRatio, score_cutoff=80)
    if best:
        match_str, score, idx = best
        r = gj_df.iloc[idx]
        return {
            "osm_id": int(r["osm_id"]),
            "geojson_name": match_str,
            "match_type": "fuzzy",
            "score": float(score),
        }
    return {"osm_id": None, "geojson_name": None, "match_type": "unmatched", "score": 0.0}


def main():
    monthly = pd.read_csv(MONTHLY, encoding="utf-8-sig")
    gj_df = load_geojson_names()
    print(f"geojson named buildings: {len(gj_df)}")
    print(f"unique cleaned NCU names: {monthly['building'].nunique()}")

    unique_names = sorted(monthly["building"].unique())
    rows = []
    for name in unique_names:
        m = match_one(name, gj_df)
        m["building"] = name
        rows.append(m)
    name_map = pd.DataFrame(rows)[
        ["building", "osm_id", "geojson_name", "match_type", "score"]
    ]
    name_map.to_csv(OUT_DIR / "name_to_uid.csv", index=False, encoding="utf-8-sig")

    # Join
    monthly_uid = monthly.merge(
        name_map[["building", "osm_id", "geojson_name", "match_type"]],
        on="building", how="left",
    )
    monthly_uid.to_csv(OUT_DIR / "monthly_kwh_with_uid.csv",
                       index=False, encoding="utf-8-sig")

    # Report
    counts = name_map["match_type"].value_counts()
    matched_kwh = monthly_uid.dropna(subset=["osm_id"])["kwh"].sum()
    total_kwh = monthly_uid["kwh"].sum()
    coverage = matched_kwh / total_kwh * 100 if total_kwh else 0

    unmatched = name_map[name_map["match_type"] == "unmatched"]
    fuzzy_rows = name_map[name_map["match_type"] == "fuzzy"].sort_values("score")

    lines = [
        "# 建物名 → osm_id (UID) 對應報告",
        "",
        f"- 來源:`{MONTHLY.name}` 共 {monthly['building'].nunique()} 個唯一名稱",
        f"- buildings.geojson 中具名建物:{len(gj_df)} 棟",
        "",
        "## 對應結果分布",
        "",
        "| match_type | 數量 |",
        "|---|---:|",
    ]
    for k, v in counts.items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        f"- **kWh 覆蓋率(已對到 UID 的 kWh / 總 kWh)**:{coverage:.1f}%",
        f"  ({matched_kwh:,.0f} / {total_kwh:,.0f})",
        "",
        "## 模糊匹配明細(score < 100,需人工複核)",
        "",
        "| 電表名 | 對到 geojson | score |",
        "|---|---|---:|",
    ]
    for _, r in fuzzy_rows.iterrows():
        lines.append(f"| {r['building']} | {r['geojson_name']} | {r['score']:.1f} |")
    if len(unmatched):
        lines += [
            "",
            "## 完全對不到的名稱(可能 geojson 沒有這棟,或別名待補)",
            "",
        ]
        for _, r in unmatched.iterrows():
            lines.append(f"- {r['building']}")
    (OUT_DIR / "match_report.md").write_text("\n".join(lines), encoding="utf-8")

    print()
    print("Match type counts:")
    print(counts.to_string())
    print()
    print(f"kWh coverage: {coverage:.1f}%")
    print()
    print(f"Wrote:")
    print(f"  {OUT_DIR / 'name_to_uid.csv'}")
    print(f"  {OUT_DIR / 'monthly_kwh_with_uid.csv'}")
    print(f"  {OUT_DIR / 'match_report.md'}")


if __name__ == "__main__":
    main()
