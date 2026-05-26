"""
Align NCU 建築物 infro.xlsx with the building list from meter v4.

Source of truth for "which buildings need to be in the table":
  C:\\Users\\User\\Downloads\\中大電表資料\\per_quarter\\中大電表資料_109Q1_v4.csv

The xlsx is corrected so that:
  - Every metered building appears once (no duplicates)
  - Buildings in the xlsx that are NOT in the meter list are kept but
    marked "no_meter" so the user can decide whether to drop them.
  - Buildings in the meter list that are NOT in the xlsx are added
    with blank metadata and marked "added_from_meter_list".
  - A status column is added for clarity.

Output:
  C:\\Users\\User\\Downloads\\NCU 建築物 infro_aligned.xlsx
  C:\\Users\\User\\Downloads\\NCU 建築物 infro_alignment_report.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


V4_PATH = Path(r"C:\Users\User\Downloads\中大電表資料\per_quarter\中大電表資料_109Q1_v4.csv")
AUDIT_PATHS = [
    Path(r"C:\Users\User\demo\outputs\ncu_109\meter_audit.csv"),
    Path(r"C:\Users\User\demo\outputs\ncu_110\meter_audit.csv"),
    Path(r"C:\Users\User\demo\outputs\ncu_111\meter_audit.csv"),
    Path(r"C:\Users\User\demo\outputs\ncu_114\meter_audit.csv"),
]
XLSX_PATH = Path(r"C:\Users\User\Downloads\NCU 建築物 infro.xlsx")
OUT_XLSX = Path(r"C:\Users\User\Downloads\NCU 建築物 infro_aligned.xlsx")
REPORT_CSV = Path(r"C:\Users\User\Downloads\NCU 建築物 infro_alignment_report.csv")


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def normalize_name(name: str) -> str:
    """Aggressive normalization for fuzzy matching."""
    if not isinstance(name, str):
        return ""
    s = name.strip()
    # Strip spaces
    s = re.sub(r"\s+", "", s)
    # Normalize parens
    s = s.replace("（", "(").replace("）", ")")
    # Remove trailing punctuation
    s = re.sub(r"[：:。、，,]$", "", s)
    return s


def core_token(name: str) -> str:
    """Extract main building name without parenthesised aliases."""
    s = normalize_name(name)
    # Strip everything in parens
    s = re.sub(r"\([^)]*\)", "", s)
    s = s.strip()
    return s


def all_tokens(name: str) -> set[str]:
    """Return both the core name and any names inside parens for matching."""
    if not isinstance(name, str):
        return set()
    s = normalize_name(name)
    out: set[str] = set()
    out.add(s)
    out.add(core_token(s))
    for m in re.findall(r"\(([^)]+)\)", s):
        if m:
            out.add(m.strip())
            # Some are like "原男四舍" — keep both with and without 原
            if m.startswith("原"):
                out.add(m[1:])
    out.discard("")
    return out


# Manual aliases for names that can't be matched by simple normalization.
# Key = a canonical fragment, value = list of equivalent fragments.
MANUAL_ALIASES = {
    "科一館": {"科學一館"},
    "科二館": {"科學二館", "理學院"},
    "科三館": {"科學三館"},
    "科四館": {"健雄館", "科學四館"},
    "科五館": {"科學五館"},
    "鴻經館": {"科學五館"},
    "文一館": {"文學一館"},
    "文二館": {"文學二館"},
    "管二館": {"管理二館"},
    "電機館": {"電機系館", "工程二館"},
    "機械館": {"工程二館A棟"},
    "工一館": {"工程一館"},
    "工五館(A、B棟)": {"工程五館", "工五館", "工程五館大樓", "工五館A、B棟", "工五館A棟", "工五館B棟"},
    "工四館一期(環化館)": {"環工化工館", "工程四館一期", "環化館"},
    "工程四館二期(機電實驗室)": {"機電實驗室"},
    "工程四館三期(大型力學實驗室)": {"大型力學實驗室", "大力館"},
    "志希館": {"管理一館"},
    "綜教館": {"綜合教學館"},
    "據德樓": {"大講堂"},
    "客家學院大樓": {"客家學院"},
    "國鼎光電大樓": {"光電大樓"},
    "國鼎圖書資料館": {"圖書資料館"},
    "中大會館": {"國際會議廳"},
    "理學院教學館": {"普通化學實驗大樓", "普化實驗大樓"},
    "太空遙測中心": {"太空及遙測研究中心"},
    "教職員單身一舍": {"教職員宿舍一舍", "單身一舍"},
    "教職員單身二舍": {"教職員宿舍二舍", "單身二舍"},
    "教職員單身四舍": {"教職員宿舍四舍", "單身四舍"},
    "學生男三宿舍": {"男三舍"},
    "學生男六宿舍": {"男六舍"},
    "學生男七宿舍": {"男七舍"},
    "學生男九宿舍": {"男九舍", "男9A舍", "男9B舍"},
    "學生男十一宿舍": {"男十一舍", "男11舍"},
    "學生男十二宿舍": {"男十二舍", "男12舍"},
    "學生男十三宿舍": {"男十三舍", "男13舍"},
    "學生女一宿舍": {"女一舍", "女1舍"},
    "學生女二宿舍": {"女二舍", "女2舍"},
    "學生女三宿舍": {"女三舍", "女3舍"},
    "學生女四宿舍": {"女四舍", "女4舍"},
    "學生女五舍(原男四舍)": {"女五舍", "原男四舍", "男四舍"},
    "學生女六宿舍(原男五舍)": {"女六舍", "原男五舍", "男五舍"},
    "學生女十四舍": {"女十四舍", "女14舍"},
    "體育館(依仁堂)": {"依仁堂", "體育館"},
    "游藝館": set(),
    "學生活動中心(志道樓)": {"志道樓", "學生活動中心"},
    "11號深水井": {"11號井"},
    "6號水井": {"6號井"},
    "第1水塔": {"水塔1", "1號水塔"},
    "自來水3號水塔": {"3號水塔"},
    "地下水1號水塔": {"水塔1號"},
    "地下水4號水塔": {"水塔4號"},
    "風洞實驗室及品保中心": {"風洞實驗室", "品保中心"},
    "氣象觀測站": set(),
    "雷達站": set(),
    "雷達車車庫": set(),
    "大氣環境實驗室": set(),
    "實習三廠(機械鑄造廠)": {"實習三廠", "機械鑄造廠"},
    "工五館B棟増建": {"工五館B棟增建"},
    "大氣環境實驗室": {"大氣微星物質分析實驗室"},
    "程五館C棟": {"工五館C棟", "工程五館C棟"},
    "研究中心大樓二期": {"研究中心二期"},
    "產學營運中心": set(),
    "松苑餐廳": set(),
    "國際學舍(原二招)": {"國際學舍", "二招", "原二招"},
    "倉庫": set(),
    "垃圾集中處理場": {"垃圾場"},
    "研究生宿舍": {"研舍", "新研舍", "男研舍", "研究生宿舍A棟", "研究生宿舍B棟"},
}


def build_alias_index() -> dict[str, str]:
    """Map every alias token to its canonical meter-list name."""
    idx: dict[str, str] = {}
    for canon, aliases in MANUAL_ALIASES.items():
        canon_norm = normalize_name(canon)
        for tok in {canon_norm, *(normalize_name(a) for a in aliases)}:
            if tok and tok not in idx:
                idx[tok] = canon
    return idx


def match_xlsx_to_meter(
    xlsx_name: str, meter_names: list[str], alias_idx: dict[str, str]
) -> str | None:
    """Return the canonical meter name if matched, else None."""
    tokens = all_tokens(xlsx_name)
    if not tokens:
        return None

    # Both full-normalized form AND core (parens-stripped) form
    meter_norm_to_orig = {normalize_name(m): m for m in meter_names}
    meter_core_to_orig: dict[str, str] = {}
    for m in meter_names:
        c = core_token(m)
        if c and c not in meter_core_to_orig:
            meter_core_to_orig[c] = m
    # Inner alias tokens of meter names (e.g. "科二館(理學院)" → "理學院")
    meter_inner_to_orig: dict[str, str] = {}
    for m in meter_names:
        for inner in all_tokens(m):
            if inner != normalize_name(m) and inner != core_token(m):
                meter_inner_to_orig.setdefault(inner, m)

    # 1. Full-normalized direct hit
    for tok in tokens:
        if tok in meter_norm_to_orig:
            return meter_norm_to_orig[tok]

    # 2. Hit on meter's CORE name (科二館(理學院) → 科二館)
    for tok in tokens:
        if tok in meter_core_to_orig:
            return meter_core_to_orig[tok]

    # 3. Hit on meter's INNER alias (健雄館 ↔ 科四館(健雄館))
    for tok in tokens:
        if tok in meter_inner_to_orig:
            return meter_inner_to_orig[tok]

    # 4. Alias canonical → meter (科學二館 → canonical 科二館 → 科二館(理學院))
    for tok in tokens:
        if tok in alias_idx:
            canon = alias_idx[tok]
            canon_norm = normalize_name(canon)
            canon_core = core_token(canon)
            if canon_norm in meter_norm_to_orig:
                return meter_norm_to_orig[canon_norm]
            if canon_core in meter_core_to_orig:
                return meter_core_to_orig[canon_core]

    # 5. Substring fallback — be conservative, require length >= 4
    for tok in tokens:
        if len(tok) < 4:
            continue
        for m_norm, m_orig in meter_norm_to_orig.items():
            if tok in m_norm or m_norm in tok:
                return m_orig

    return None


def main() -> None:
    configure_stdout()

    if not V4_PATH.exists():
        raise FileNotFoundError(V4_PATH)
    if not XLSX_PATH.exists():
        raise FileNotFoundError(XLSX_PATH)

    # 1. Meter-side: unique building names from v4 + all audit files.
    # v4 is the canonical baseline; later-quarter audits add buildings
    # that were constructed after 109Q1 (e.g. iHouse享想空間 first
    # appearing in 114).
    v4 = pd.read_csv(V4_PATH, dtype=str, encoding="utf-8-sig")
    v4.columns = [c.strip().lstrip("﻿") for c in v4.columns]
    bld_col = "建築物" if "建築物" in v4.columns else v4.columns[1]
    v4_set = {
        str(b).strip()
        for b in v4[bld_col]
        if isinstance(b, str) and str(b).strip()
    }
    print(f"v4 unique buildings: {len(v4_set)}")

    audit_set: set[str] = set()
    for ap in AUDIT_PATHS:
        if not ap.exists():
            print(f"  [skip] missing audit: {ap}")
            continue
        a = pd.read_csv(ap, dtype=str, encoding="utf-8-sig")
        a.columns = [c.strip().lstrip("﻿") for c in a.columns]
        if "building" in a.columns:
            for b in a["building"]:
                if isinstance(b, str) and b.strip():
                    audit_set.add(b.strip())
    print(f"Audit-only unique buildings (added): {len(audit_set - v4_set)}")

    meter_set = v4_set | audit_set
    meter_buildings = sorted(meter_set)
    print(f"Total canonical meter buildings: {len(meter_buildings)}")

    # Meter counts: prefer v4 row counts; fallback to audit unique meter_ids
    meter_counts: dict[str, int] = (
        v4[v4[bld_col].notna()]
        .groupby(bld_col)
        .size()
        .to_dict()
    )
    # Fill missing buildings using audits (unique meter_ids per building)
    for ap in AUDIT_PATHS:
        if not ap.exists():
            continue
        a = pd.read_csv(ap, dtype=str, encoding="utf-8-sig")
        a.columns = [c.strip().lstrip("﻿") for c in a.columns]
        if "building" not in a.columns or "meter_id" not in a.columns:
            continue
        a = a[a["building"].notna() & a["meter_id"].notna()]
        for b, n in a.drop_duplicates(["building", "meter_id"]).groupby("building").size().items():
            if b not in meter_counts:
                meter_counts[b] = int(n)

    # 2. XLSX side
    xlsx = pd.read_excel(XLSX_PATH)
    xlsx.columns = [str(c).strip() for c in xlsx.columns]
    name_col = "建築物正式名稱"
    if name_col not in xlsx.columns:
        # Try second column
        for c in xlsx.columns:
            if "名稱" in c or "建築" in c:
                name_col = c
                break
    print(f"XLSX rows: {len(xlsx)}  name col: {name_col!r}")

    alias_idx = build_alias_index()

    # 3. Match each xlsx row to a meter-side canonical name
    xlsx["_matched_meter_building"] = xlsx[name_col].apply(
        lambda n: match_xlsx_to_meter(n, meter_buildings, alias_idx)
    )

    matched_set = {n for n in xlsx["_matched_meter_building"] if isinstance(n, str)}

    # 4. Buildings that are in meters but not yet in xlsx
    missing_from_xlsx = [m for m in meter_buildings if m not in matched_set]

    # 5. Build aligned output
    out_rows: list[dict] = []
    seen_canon: set[str] = set()

    # 5a. xlsx rows in their original order, with status
    for _, row in xlsx.iterrows():
        match = row["_matched_meter_building"]
        status = "matched" if isinstance(match, str) else "in_xlsx_no_meter"
        canon = match if isinstance(match, str) else str(row[name_col]).strip()
        n_meters = meter_counts.get(canon, 0) if isinstance(match, str) else 0
        out = {col: row[col] for col in xlsx.columns if not col.startswith("_")}
        out["canonical_building_name"] = canon
        out["meter_v4_count"] = n_meters
        out["alignment_status"] = status
        out_rows.append(out)
        if isinstance(match, str):
            seen_canon.add(match)

    # 5b. Append meter-only buildings (not in xlsx)
    for canon in missing_from_xlsx:
        if canon in seen_canon:
            continue
        blank = {col: None for col in xlsx.columns if not col.startswith("_")}
        blank[name_col] = canon
        blank["canonical_building_name"] = canon
        blank["meter_v4_count"] = meter_counts.get(canon, 0)
        blank["alignment_status"] = "added_from_meter_list"
        out_rows.append(blank)
        seen_canon.add(canon)

    final = pd.DataFrame(out_rows)

    # Drop helper col if present
    if "_matched_meter_building" in final.columns:
        final = final.drop(columns=["_matched_meter_building"])

    final.to_excel(OUT_XLSX, index=False)

    # 6. Report
    report = pd.DataFrame({
        "category": [
            "Meter v4 unique buildings",
            "Original xlsx rows",
            "Matched (xlsx ↔ meter)",
            "In xlsx but no meter (probably non-electrical structures)",
            "In meter but missing from xlsx (added)",
            "Final aligned rows",
        ],
        "count": [
            len(meter_buildings),
            len(xlsx),
            len(matched_set),
            (final["alignment_status"] == "in_xlsx_no_meter").sum(),
            (final["alignment_status"] == "added_from_meter_list").sum(),
            len(final),
        ],
    })
    report.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")

    print()
    print("=" * 60)
    print("Alignment report")
    print("=" * 60)
    print(report.to_string(index=False))
    print()
    print(f"Wrote aligned xlsx: {OUT_XLSX}")
    print(f"Wrote report:       {REPORT_CSV}")
    print()

    print("--- in_xlsx_no_meter (sample) ---")
    samp = final[final["alignment_status"] == "in_xlsx_no_meter"]
    if not samp.empty:
        for _, r in samp.iterrows():
            print(f"  {r[name_col]}")
    else:
        print("  (none)")
    print()
    print("--- added_from_meter_list (sample) ---")
    samp2 = final[final["alignment_status"] == "added_from_meter_list"]
    if not samp2.empty:
        for _, r in samp2.iterrows():
            print(f"  {r[name_col]}  (meter count = {r['meter_v4_count']})")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
