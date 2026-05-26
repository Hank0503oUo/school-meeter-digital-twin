"""
Canonicalize NCU building names in cleaned monthly_kwh.csv files using
the OSM/Google authoritative name list built by build_canonical_name_dict.py.

For each (year, building) row:
  - Try exact match against canonical names
  - Else try normalized match (strip parens/whitespace)
  - Else fuzzy match (rapidfuzz WRatio ≥ 80) against canonical
  - Else keep original (flagged as "uncanonical")

Aggregates kWh by (canonical_name, year, month) — multiple input variants merge.

Inputs:
  outputs/_cleaning_diagnosis/ncu_canonical_names.json
  outputs/ncu_<roc>/monthly_kwh.csv     (for each year in YEARS)

Outputs (in-place rename + backup):
  outputs/ncu_<roc>/monthly_kwh.csv               ← canonicalized
  outputs/ncu_<roc>/monthly_kwh.precanon.csv      ← original backup
  outputs/_cleaning_diagnosis/canonicalization_audit_<roc>.csv
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[2]
CANON_JSON = ROOT / "outputs" / "_cleaning_diagnosis" / "ncu_canonical_names.json"
DIAG_DIR = ROOT / "outputs" / "_cleaning_diagnosis"

YEARS = [109, 110, 111, 114]
PARTIAL_THRESHOLD = 75   # rapidfuzz.partial_ratio cutoff (more lenient for CJK)

# NCU rename / common-vs-formal mapping. Built from observed uncanonical list:
#   long form (in our cleaned data) → canonical short (in OSM/geojson)
NCU_RENAMES = {
    "中正圖書館": "總圖書館",
    "研究生宿舍": "曦望居",
    "中大十舍": "中大會館",
    "館(理學院)": "科學二館(理學院大樓)",
    "管二館": "管理學院二館",
    "科四館(健雄館)": "科學四館",
    "工五館B棟增建": "工五館(A、B棟)",
    "工五館B棟増建": "工五館(A、B棟)",
    "教職員單身一舍": "單一舍",
    "教職員單身二舍": "單二舍",
    "教職員單身四舍": "單四舍",
    "游藝館": "遊藝館",
    "學生男一宿舍": "男三舍",   # adjust as needed; keep cautious
    "學生女一宿舍": "女一舍",
    "學生女二宿舍": "女二舍",
    "學生女三宿舍": "女三舍",
    "學生女四宿舍": "女四舍",
    "學生女五舍": "女五舍",       # may not exist anymore (renovated)
    "學生男五舍": "男五舍",
    "學生男六宿舍": "男六舍",
    "學生男七宿舍": "男七舍",
    "學生男九宿舍": "男九舍",
    "學生男十一宿舍": "男十一舍",
    "學生男十二宿舍": "男十二舍",
    "學生男十三宿舍": "男13舍",   # OSM has '男13舍'
    "學生女十四舍": "女十四舍",
    "學生自主學習空間": "iHouse享想空間",
    "享想空間": "iHouse享想空間",
    "中大幼稚園": "中大幼稚園",
}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

_PAREN = re.compile(r"[（()）]")
_SP = re.compile(r"\s+")


def name_key(s: str) -> str:
    return _SP.sub("", _PAREN.sub("", (s or "").strip())).lower()


# ── Number/category conflict guards ──────────────────────────────────────
# Block fuzzy matches that confuse different building series:
#   工五館 vs 工三館,  學生男三 vs 學生男七,  學生女六 vs 學生男六,  文一 vs 工一,
#   實習三廠 vs 實習一廠
NUMBER_QUANT_RE = re.compile(r"([一二三四五六七八九十百千]+|\d+)\s*(舍|館|宿舍|號|期|棟|廠)")
GENDER_RE = re.compile(r"([男女])([一二三四五六七八九十百千]+|\d+)?\s*(舍|宿舍)")
# Strip common university-prefix words before checking the series character.
_UNIV_PREFIXES = ("國立中央大學", "中央大學", "國立", "中大")
_COLLEGE_PREFIX_CHARS = set("工文管理科理農法社")


def _strip_univ_prefix(s: str) -> str:
    s = (s or "").strip().lstrip(" 　")
    for pfx in _UNIV_PREFIXES:
        if s.startswith(pfx):
            return s[len(pfx):].lstrip(" 　")
    return s


def _normalized_quant(s: str) -> set[str]:
    return {f"{n}{q}" for n, q in NUMBER_QUANT_RE.findall(s or "")}


def _normalized_gender(s: str) -> set[str]:
    out = set()
    for m in GENDER_RE.finditer(s or ""):
        gender, num, _ = m.group(1), m.group(2), m.group(3)
        out.add(f"{gender or ''}{num or 'X'}".strip() or "X")
    return out


def _series_prefix(s: str) -> str | None:
    """If the name starts with a college series character (工/文/管/理/...) AND
    references 館 / 學院 within the first ~6 chars, return that character."""
    stripped = _strip_univ_prefix(s or "")
    if not stripped:
        return None
    if stripped[0] in _COLLEGE_PREFIX_CHARS and (
        "館" in stripped[:6] or "學院" in stripped[:6]
    ):
        return stripped[0]
    return None


def conflicting_match(a: str, b: str) -> bool:
    """True if a and b reference different specific items in same series."""
    qa, qb = _normalized_quant(a), _normalized_quant(b)
    if qa and qb and qa.isdisjoint(qb):
        return True
    # Gender check is asymmetric: if A has a gender token (男X/女X) but B doesn't
    # mention that gender at all, refuse the merge — 學生男三 should NOT match
    # 國際學生宿舍 just because they share "學生宿舍".
    ga, gb = _normalized_gender(a), _normalized_gender(b)
    if ga and not gb:
        # If B mentions neither '男' nor '女', it's a generic dorm — not the same
        if "男" not in (b or "") and "女" not in (b or ""):
            return True
    if gb and not ga:
        if "男" not in (a or "") and "女" not in (a or ""):
            return True
    if ga and gb and ga.isdisjoint(gb):
        return True
    # Series prefix comparison (after stripping 國立中央大學 etc)
    pa, pb = _series_prefix(a), _series_prefix(b)
    if pa and pb and pa != pb:
        return True
    return False


def main():
    if not CANON_JSON.exists():
        print(f"[ERROR] {CANON_JSON} not found — run build_canonical_name_dict.py first")
        sys.exit(1)
    canon = json.loads(CANON_JSON.read_text(encoding="utf-8"))
    canonical_names = canon["names"]
    canon_keys = {name_key(n): n for n in canonical_names}
    print(f"loaded {len(canonical_names)} canonical NCU building names")
    print()

    summary = []
    for roc in YEARS:
        monthly_csv = ROOT / "outputs" / f"ncu_{roc}" / "monthly_kwh.csv"
        if not monthly_csv.exists():
            print(f"[skip] {monthly_csv} not found")
            continue
        monthly = pd.read_csv(monthly_csv, encoding="utf-8-sig")
        n_in = monthly["building"].nunique()

        # Per-name resolution
        unique_names = sorted(monthly["building"].dropna().unique())
        mapping = {}
        audit_rows = []
        for name in unique_names:
            kept = name
            method = "no_change"
            score = 100.0

            # 0) Manual NCU rename / canonicalization map (highest priority)
            if name in NCU_RENAMES:
                kept = NCU_RENAMES[name]
                method = "manual_rename"
            # 1) exact match in canonical
            elif name in canonical_names:
                kept = name
                method = "exact"
            # 2) normalized key match
            elif name_key(name) in canon_keys:
                kept = canon_keys[name_key(name)]
                method = "normalized"
                score = 95.0
            else:
                # 3) fuzzy with partial_ratio, scanning all canonical names but
                #    rejecting any candidate that conflicts on a numeric series
                #    (學生男三 vs 學生男七), gender (女六 vs 男六), or college
                #    prefix (文一 vs 工一).
                scored = process.extract(name, canonical_names,
                                          scorer=fuzz.partial_ratio,
                                          limit=10,
                                          score_cutoff=PARTIAL_THRESHOLD)
                kept_candidate = None
                for cand, sc, _ in scored:
                    if len(cand) < 2:
                        continue
                    if conflicting_match(name, cand):
                        continue
                    kept_candidate = (cand, sc)
                    break
                if kept_candidate:
                    kept, score = kept_candidate
                    method = "fuzzy_partial"
                else:
                    # Non-building meter (water well, warehouse, radar, factory etc)
                    # — keep the name as-is, no canonical match. NOT an error.
                    method = "facility_meter"
                    score = 0.0

            mapping[name] = kept
            audit_rows.append({
                "input_name": name,
                "canonical_name": kept,
                "method": method,
                "score": round(score, 1),
                "n_records": int((monthly["building"] == name).sum()),
            })

        # Apply mapping
        monthly["building_canonical"] = monthly["building"].map(mapping)
        # Re-aggregate per (canonical, year, month)
        agg = (monthly.dropna(subset=["building_canonical"])
                      .groupby(["building_canonical", "year", "month"], as_index=False)
                      .agg(kwh=("kwh", "sum"),
                           n_meters=("n_meters", "sum"),
                           n_rollover=("n_rollover", "sum"),
                           sources=("source", lambda s: ",".join(sorted(set(s))))))
        agg = agg.rename(columns={"building_canonical": "building",
                                   "sources": "source"})
        agg = agg[["building", "year", "month", "kwh", "n_meters",
                    "n_rollover", "source"]]

        n_out = agg["building"].nunique()
        before_total = float(monthly["kwh"].sum())
        after_total = float(agg["kwh"].sum())
        print(f"民國 {roc}: {n_in} → {n_out} canonical buildings  "
              f"(kWh {before_total/1e6:.2f} → {after_total/1e6:.2f} GWh)")

        # Write backup + canonical version
        backup = monthly_csv.with_suffix(".precanon.csv")
        if not backup.exists():
            monthly_csv.rename(backup)
        else:
            backup.write_bytes(monthly_csv.read_bytes())
        agg.to_csv(monthly_csv, index=False, encoding="utf-8-sig")

        # Audit
        audit_df = pd.DataFrame(audit_rows).sort_values(["method", "input_name"])
        audit_df.to_csv(DIAG_DIR / f"canonicalization_audit_{roc}.csv",
                          index=False, encoding="utf-8-sig")

        summary.append({
            "roc_year": roc,
            "n_input": n_in,
            "n_canonical": n_out,
            "kwh_before_GWh": round(before_total / 1e6, 2),
            "kwh_after_GWh": round(after_total / 1e6, 2),
            "n_uncanonical": int((audit_df["method"] == "uncanonical_kept").sum()),
        })

    print()
    print("=== Summary ===")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
