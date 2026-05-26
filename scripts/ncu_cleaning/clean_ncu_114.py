"""
Clean NCU 114 (民國年) electricity meter readings.

Source CSVs (from MinerU PDF extraction):
  G:/我的雲端硬碟/mineru/mineru_output_mineru_vlm/csv_by_quarter/
    中大電表資料_114Q1_mineru_vlm.csv   (cols 12月,1月,2月,3月  → 2024-12, 2025-01..03)
    中大電表資料_114Q2_mineru_vlm.csv   (cols same names    → 2025-03..06)
    中大電表資料_114Q3_mineru_vlm.csv                      → 2025-06..09
    中大電表資料_114Q4_mineru_vlm.csv                      → 2025-09..12

Pipeline:
  1. Read each quarter, rename the 4 reading columns to (year, month).
  2. Normalize 建築物 (strip OCR contamination like "又=6328", " 1499" suffixes).
  3. Filter rows: drop 虛擬電表 (placeholder, 不抄表). Keep 電力盤 / 電燈盤 / 虛擬總表.
  4. For each meter (表號), order by (year, month) globally, compute delta with rollover fix
     for 4-digit meters (NCU spec: rolls at 9999 → 0).
  5. kwh_meter = delta * 倍數.
  6. Aggregate per (building, year, month): sum across 電力盤+電燈盤. Use 虛擬總表 only when
     no physical meters exist for that building (avoid double counting).
  7. Emit 3 audit files into outputs/ncu_114/.

Outputs:
  monthly_kwh.csv          building × month × kwh   (the main artifact)
  meter_audit.csv          per-meter, per-month with delta + rollover/outlier flags
  cleaning_report.md       summary stats + warnings

Run:
  cd D:/idf優化/demo
  python scripts/ncu_cleaning/clean_ncu_114.py
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(r"G:/我的雲端硬碟/mineru/mineru_output_mineru_vlm/csv_by_quarter")
OUT_DIR = ROOT / "outputs" / "ncu_114"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Quarter → list of (year, month) for the four reading columns in CSV order.
# 民國 114 = 西元 2025
QUARTER_MONTHS: dict[str, list[tuple[int, int]]] = {
    "Q1": [(2024, 12), (2025, 1), (2025, 2), (2025, 3)],
    "Q2": [(2025, 3),  (2025, 4), (2025, 5), (2025, 6)],
    "Q3": [(2025, 6),  (2025, 7), (2025, 8), (2025, 9)],
    "Q4": [(2025, 9),  (2025, 10),(2025, 11),(2025, 12)],
}

QUARTER_FILES = {q: SRC_DIR / f"中大電表資料_114{q}_mineru_vlm.csv" for q in QUARTER_MONTHS}

READING_COLS = ["12月", "1月", "2月", "3月"]   # raw CSV column names, same in every quarter

# Meter topology — classify by meter_id prefix (the panel name 開關箱 is too noisy:
# it includes MP/ML/MVCB/VCB1/BF/AC3 sub-distribution codes plus 電力盤/電燈盤/總表 etc.).
#   VL_*   → 虛擬電表 placeholder, "不抄表"     → drop entirely
#   A1_*   → 虛擬總表 building aggregate read   → fallback when no physical meter
#   anything else (numeric ID like 3651464, or S1_*, E1_*) → physical sub-meter
VIRTUAL_PLACEHOLDER_PREFIX = "VL_"
AGGREGATE_PREFIX = "A1_"

ROLLOVER_MAX = 10000   # 4-digit meter wraps at 9999 → 0
ROLLOVER_DROP_THRESHOLD = 5000  # only treat as rollover if drop is large

# Sanity bounds — anything beyond is treated as OCR digit-shift garbage.
# A single sub-meter rarely exceeds 100 MWh/month even for industrial loads;
# 200 MWh/month is a generous ceiling.
MAX_KWH_PER_METER_MONTH = 200_000.0
# If the digit length of a meter reading differs from the previous reading by ≥ 2,
# treat as OCR error rather than a real rollover.
MAX_DIGIT_LEN_DIFF = 2


def classify_meter(meter_id: str) -> str:
    """Return 'placeholder', 'aggregate', or 'physical'."""
    s = str(meter_id).strip().upper()
    if s.startswith(VIRTUAL_PLACEHOLDER_PREFIX):
        return "placeholder"
    if s.startswith(AGGREGATE_PREFIX):
        return "aggregate"
    return "physical"


# ── Building name normalization ───────────────────────────────────────────
_BUILDING_NORMALIZE_PATTERNS = [
    (re.compile(r"^又=\d+$"), None),          # OCR garbage like "又=6328" → drop
    (re.compile(r"^\d+/\d+\s*\d*$"), None),   # "1/9 800" → drop
    (re.compile(r"^[\d\s,]+$"), None),         # all digits → drop
]

# Strip trailing OCR contamination: a separator (space/colon/comma/slash/period)
# followed by digits and/or parenthesised digit groups, glued after the real name.
# Examples handled:
#   "宿舍 8:404", "曦望居(原研究生宿舍) 5836 (487)", "宿舍99187"
#   "學生女一宿舍 3/66.1844", "工四館一期(環化館) 39/09"
_TRAILING_DIGIT_BLOCK = re.compile(
    r"(?:[\s,:：]+|(?<=[一-鿿)〉）]))"     # separator OR right after CJK / closing paren
    r"\d[\d\s,./\\]*"                                  # digits with /,. separators
    r"(?:[（(]\s*\d[\d\s,./\\]*\s*[)）])?"           # optional trailing (digits)
    r"\s*$"
)

# Canonicalize "<NewName>(原<OldName>)" → "<NewName>" so the building's
# data across years (where some years use the new name only, others the
# annotated form) all aggregate to the same canonical name.
_PARENS_FORMER_NAME = re.compile(
    r"\s*[（(]\s*原[^()（）]{1,15}\s*[)）]\s*$"
)

# Strip prefix/suffix decorations the OCR added:
#   "工一館√"  / "工五館(A、B棟) √" / "體育館(依仁堂)√"  → drop trailing tick
#   "人文社會科學大樓·14 $^{n}$" → strip latex remnants
#   "工四館一期(環化館) → 109、12、13 結号可表" → strip after arrow
_TRAILING_TICK_OR_NOISE = re.compile(
    r"\s*(?:[√✓✅·]+\s*)+"   # √ ✓ ✅ · sequences
    r"|\s*\$\^?\{[^}]*\}\$?"                       # $^{n}$ latex remnants
    r"|\s*[→⇒]+.*$"                                # arrow + everything after
    r"|\s*[，,].{0,30}$"                          # trailing ", 七9h20" etc
    r"$"
)

# Common OCR character confusions specific to the NCU PDF set.
_OCR_CHAR_FIXES = {
    "丁四館": "工四館",   # 丁/工
    "丁五館": "工五館",
    "丁一館": "工一館",
    "丁二館": "工二館",
    "丁三館": "工三館",
    "工一工": "工一館",   # 109Q1 OCR 把「館」誤抄成「工」
    "目來水": "自來水",   # 目/自
    "餡": "館",           # 餡/館 (already used by match_buildings alias too)
    "棟増": "棟增",       # 増/增 alt form
    "煉男": "原男",       # 煉/原
}

# Suffix-completion: building names that are OCR-truncated forms of canonical names
_SUFFIX_COMPLETIONS = [
    ("氣象觀測", "氣象觀測站"),    # 111 有些季度漏「站」字
    ("號水井", None),              # 缺前置數字, 直接丟
]


# Header-like rows that MinerU sometimes captured as data — when the "次" cell
# looks like a column header rather than a row number, we drop the whole row.
_NON_NUMERIC_SEQ_TOKENS = {
    "项次", "次數", "次数", "次", "序號", "序号", "編號", "编号",
    "Item", "item", "No.", "no.",
}

# Dorms missing the "學" prefix (生男X舍 → 學生男X舍, 生女X舍 → 學生女X舍)
_MISSING_XUE_PREFIX = re.compile(r"^生(男|女)([一二三四五六七八九十百千]+|\d+)?")

# Dorm "男X舍" / "男X宿舍" without 學生 prefix → add it
_BARE_DORM = re.compile(r"^([男女])([一二三四五六七八九十百千]+|\d+)\s*(舍|宿舍)$")
# OCR sometimes drops a closing paren — close it if we have an unmatched "("
_OPEN_PAREN_COUNT = re.compile(r"[(（]")
_CLOSE_PAREN_COUNT = re.compile(r"[)）]")


def _balance_parens(s: str) -> str:
    n_open = len(_OPEN_PAREN_COUNT.findall(s))
    n_close = len(_CLOSE_PAREN_COUNT.findall(s))
    if n_open > n_close:
        s = s + ")" * (n_open - n_close)
    elif n_close > n_open:
        s = s.rstrip("()")
    return s


def normalize_building_name(raw: str) -> str | None:
    if not isinstance(raw, str):
        return None
    s = raw.strip().strip('"').strip()
    if not s:
        return None
    for pat, _ in _BUILDING_NORMALIZE_PATTERNS:
        if pat.match(s):
            return None
    if s.startswith(("114年", "抄表卡", "1140", "提交", "微笑單車")):
        return None
    # Strip OCR decorations (tick marks, latex remnants, "→ 注解" tails)
    for _ in range(3):
        new_s = _TRAILING_TICK_OR_NOISE.sub("", s).strip()
        if new_s == s:
            break
        s = new_s
    # Iteratively strip trailing digit garbage
    for _ in range(3):
        new_s = _TRAILING_DIGIT_BLOCK.sub("", s).strip()
        if new_s == s:
            break
        s = new_s
    # Apply common OCR character fixes
    for bad, good in _OCR_CHAR_FIXES.items():
        if bad in s:
            s = s.replace(bad, good)
    # Add missing "學" prefix to dorm names
    if _MISSING_XUE_PREFIX.match(s):
        s = "學" + s
    # Bare "男X舍" / "女X舍" without prefix → assume student dorm
    if _BARE_DORM.match(s):
        s = "學生" + s
    # Strip "(原XXX)" annotation (former name) so the building canonicalizes
    # to its current name across years.
    s = _PARENS_FORMER_NAME.sub("", s).strip()
    # Suffix completion: known truncations from inconsistent OCR across quarters
    for trunc, full in _SUFFIX_COMPLETIONS:
        if s == trunc:
            if full is None:
                return None
            s = full
            break
    # Drop trailing punctuation that is clearly noise (colons, commas)
    s = s.rstrip(" ,:：")
    # Balance unmatched parens (OCR drops one side)
    s = _balance_parens(s)
    # Reject names that start with digits or punctuation (pure garbage)
    if not s or len(s) < 2 or s[0].isdigit():
        return None
    return s


# ── Reading parsing (handle √, blank, OCR noise) ──────────────────────────
def parse_reading(v) -> float | None:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if not s or s in {"√", "_", "-", "—"}:
        return None
    # Strip stray non-digit chars except leading minus
    cleaned = re.sub(r"[^\d.]", "", s)
    if not cleaned:
        return None
    try:
        val = float(cleaned)
    except ValueError:
        return None
    if val < 0:
        return None
    return val


def parse_multiplier(v) -> float | None:
    if pd.isna(v):
        return None
    try:
        return float(str(v).strip())
    except (ValueError, TypeError):
        return None


# ── Load quarters and melt to long format ─────────────────────────────────
def load_quarter(q: str) -> pd.DataFrame:
    """Read one quarter CSV, normalize, return long-format DataFrame.

    Columns: pdf_src, building, seq, meter_id, panel, location, multiplier,
             year, month, reading_raw, reading, usage_unit, source_q
    """
    path = QUARTER_FILES[q]
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    # Strip BOM if present
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    rename = {
        "PDF來源": "pdf_src",
        "建築物": "building_raw",
        "次": "seq",
        "表號": "meter_id",
        "開關箱": "panel",
        "安裝位置": "location",
        "倍數": "multiplier_raw",
        "使用單位": "usage_unit",
    }
    df = df.rename(columns=rename)

    # Drop rows where 次 / seq cell is a header-like token (means MinerU captured
    # a PDF table-header row as data — common in 109Q1 v4)
    seq_str = df["seq"].astype(str).str.strip()
    header_row_mask = seq_str.isin(_NON_NUMERIC_SEQ_TOKENS)
    if header_row_mask.any():
        df = df[~header_row_mask].copy()

    # Forward-fill building name within a PDF source page (some rows have empty 建築物)
    # Note: only ffill within the same pdf_src so we don't bleed across PDFs
    df["building_raw"] = df.groupby("pdf_src")["building_raw"].ffill()

    df["building"] = df["building_raw"].map(normalize_building_name)
    df["multiplier"] = df["multiplier_raw"].map(parse_multiplier)

    months = QUARTER_MONTHS[q]
    melt_records = []
    for col_name, (yr, mon) in zip(READING_COLS, months):
        sub = df[[
            "pdf_src", "building", "seq", "meter_id", "panel", "location",
            "multiplier", "usage_unit", col_name,
        ]].copy()
        sub = sub.rename(columns={col_name: "reading_raw"})
        sub["year"] = yr
        sub["month"] = mon
        sub["source_q"] = q
        sub["reading"] = sub["reading_raw"].map(parse_reading)
        melt_records.append(sub)
    long = pd.concat(melt_records, ignore_index=True)
    return long


def load_all_quarters() -> pd.DataFrame:
    parts = [load_quarter(q) for q in QUARTER_MONTHS]
    return pd.concat(parts, ignore_index=True)


# ── Compute monthly deltas with rollover handling ─────────────────────────
def compute_meter_deltas(long: pd.DataFrame) -> pd.DataFrame:
    """For each meter_id, sort by (year, month), compute monthly kWh delta."""
    long = long.copy()

    # Drop rows we can't process at all
    long = long.dropna(subset=["meter_id"]).copy()
    long["meter_id"] = long["meter_id"].astype(str).str.strip()
    long = long[long["meter_id"] != ""]

    # Classify by meter_id prefix and drop placeholders entirely
    long["panel"] = long["panel"].fillna("").astype(str).str.strip()
    long["meter_kind"] = long["meter_id"].map(classify_meter)
    long = long[long["meter_kind"] != "placeholder"].copy()

    # Sort and dedupe (same meter, same month → keep first non-null reading)
    long["_period"] = long["year"] * 100 + long["month"]
    long = long.sort_values(["meter_id", "_period", "source_q"])

    # When a meter appears multiple times in the same period (boundary months Q1/Q2),
    # keep the first non-null reading. Sort so non-null come first within each group,
    # then drop duplicates on (meter_id, _period).
    long["_has_reading"] = long["reading"].notna().astype(int)
    long = long.sort_values(["meter_id", "_period", "_has_reading", "source_q"],
                            ascending=[True, True, False, True])
    deduped = long.drop_duplicates(subset=["meter_id", "_period"], keep="first").copy()
    deduped = deduped.drop(columns=["_has_reading"])

    # Compute deltas per meter
    out_rows = []
    for meter_id, grp in deduped.groupby("meter_id", sort=False):
        grp = grp.sort_values("_period").reset_index(drop=True)
        # Estimate this meter's natural digit cap from observed readings.
        # Cap = next power of 10 above the largest valid reading we've seen.
        valid_readings = grp["reading"].dropna()
        if len(valid_readings):
            max_obs = valid_readings.max()
            digit_cap = 10 ** max(4, len(str(int(max_obs))))
        else:
            digit_cap = ROLLOVER_MAX
        prev_reading = None
        for _, row in grp.iterrows():
            curr = row["reading"]
            mult = row["multiplier"]
            rec = {
                "meter_id": meter_id,
                "building": row["building"],
                "panel": row["panel"],
                "meter_kind": row["meter_kind"],
                "location": row["location"],
                "multiplier": mult,
                "year": int(row["year"]),
                "month": int(row["month"]),
                "reading": curr,
                "prev_reading": prev_reading,
                "delta": None,
                "kwh": None,
                "rollover_fix": False,
                "outlier_flag": "",
                "source_q": row["source_q"],
            }
            if curr is not None and prev_reading is not None:
                # Detect OCR digit-shift first (length changes by ≥ 2 digits)
                len_curr = len(str(int(curr))) if curr >= 1 else 1
                len_prev = len(str(int(prev_reading))) if prev_reading >= 1 else 1
                if abs(len_curr - len_prev) >= MAX_DIGIT_LEN_DIFF:
                    rec["outlier_flag"] = "ocr_digit_shift"
                    delta = None
                else:
                    delta = curr - prev_reading
                    rollover = False
                    if delta < 0:
                        drop = prev_reading - curr
                        # Try wrap with this meter's natural cap; only accept
                        # if it produces a sensible positive delta.
                        candidate = (digit_cap - prev_reading) + curr
                        if drop >= ROLLOVER_DROP_THRESHOLD and 0 < candidate < digit_cap:
                            delta = candidate
                            rollover = True
                        else:
                            # Small negative drop (< ROLLOVER_DROP_THRESHOLD) is
                            # almost always OCR noise on the last digit. Treat
                            # the month as zero consumption rather than dropping.
                            rec["outlier_flag"] = "negative_drop_zeroed"
                            delta = 0.0
                    if delta is not None and delta >= 0:
                        rec["delta"] = delta
                        if mult is not None:
                            kwh_val = delta * mult
                            if kwh_val > MAX_KWH_PER_METER_MONTH:
                                rec["outlier_flag"] = "kwh_too_large"
                                rec["delta"] = None  # exclude from aggregation
                            else:
                                rec["kwh"] = kwh_val
                                rec["rollover_fix"] = rollover
            elif curr is not None and prev_reading is None:
                rec["outlier_flag"] = "first_reading"
            elif curr is None:
                rec["outlier_flag"] = "missing"
            out_rows.append(rec)
            # Only advance prev_reading on physically plausible readings, so that
            # one bad OCR record doesn't poison the next month's delta.
            if curr is not None and rec["outlier_flag"] not in {"ocr_digit_shift"}:
                prev_reading = curr
    return pd.DataFrame(out_rows)


# ── Aggregate per (building, year, month) with topology dedupe ────────────
def aggregate_building_monthly(meter_kwh: pd.DataFrame) -> pd.DataFrame:
    """Sum physical sub-meters per building × month.
    Use aggregate (A1_*) virtual total only when no physical meters report that month."""
    df = meter_kwh.copy()
    df = df.dropna(subset=["building", "kwh"])

    physical = df[df["meter_kind"] == "physical"]
    aggregate = df[df["meter_kind"] == "aggregate"]

    phys_agg = (physical
                .groupby(["building", "year", "month"], as_index=False)
                .agg(kwh=("kwh", "sum"),
                     n_meters=("meter_id", "nunique"),
                     n_rollover=("rollover_fix", "sum")))
    phys_agg["source"] = "physical"

    # Fallback: per (building, month), if physical didn't cover, use aggregate.
    phys_keys = set(zip(phys_agg["building"], phys_agg["year"], phys_agg["month"]))
    agg_mask = ~aggregate.apply(
        lambda r: (r["building"], r["year"], r["month"]) in phys_keys, axis=1
    )
    agg_only = aggregate[agg_mask] if len(aggregate) else aggregate
    agg_agg = (agg_only
               .groupby(["building", "year", "month"], as_index=False)
               .agg(kwh=("kwh", "sum"),
                    n_meters=("meter_id", "nunique"),
                    n_rollover=("rollover_fix", "sum")))
    agg_agg["source"] = "aggregate_fallback"

    return pd.concat([phys_agg, agg_agg], ignore_index=True).sort_values(
        ["building", "year", "month"]
    ).reset_index(drop=True)


# ── Reporting ─────────────────────────────────────────────────────────────
def write_report(long: pd.DataFrame, meter_kwh: pd.DataFrame,
                 monthly: pd.DataFrame) -> str:
    n_raw_rows = len(long)
    n_buildings_raw = long["building"].dropna().nunique()
    n_buildings_clean = monthly["building"].nunique()
    n_meters = meter_kwh["meter_id"].nunique()
    n_rollover = int(meter_kwh["rollover_fix"].sum())
    n_outliers = (meter_kwh["outlier_flag"] != "").sum()

    months_present = sorted(monthly[["year", "month"]].drop_duplicates().itertuples(index=False))
    months_str = ", ".join(f"{y}-{m:02d}" for y, m in months_present)

    total_kwh = monthly["kwh"].sum()
    top10 = (monthly.groupby("building", as_index=False)["kwh"].sum()
             .sort_values("kwh", ascending=False).head(10))

    lines = [
        "# NCU 114 年電表資料清洗報告",
        "",
        f"- 來源:`{SRC_DIR}` 共 4 個季度檔",
        f"- 原始(melt 後)記錄數:{n_raw_rows:,}",
        f"- 含名建物(melt 後唯一):{n_buildings_raw}",
        f"- 清洗後產出建物數:{n_buildings_clean}",
        f"- 唯一電表 ID 數:{n_meters}",
        f"- 觸發倒轉修正(rollover fix):{n_rollover} 次",
        f"- 標記異常 / 缺值:{n_outliers} 個月電表記錄",
        f"- 涵蓋月份:{months_str}",
        f"- 總用電量(kWh):{total_kwh:,.0f}",
        "",
        "## 拓樸處理規則",
        "- 以 `meter_id` 前綴判別,因 `開關箱` 文字過於雜亂(MP/ML/MVCB/VCB1/BF 等子盤名混雜)。",
        "- `VL_*` 虛擬電表:全數忽略(原始 PDF 標註「不抄表」)。",
        "- `A1_*` 虛擬總表:僅在某 (建物 × 月) 完全沒有實體電表記錄時才採用,避免重複計算。",
        "- 其他(數字 ID 如 3651464、S1_* 等):視為實體子表,同建物 × 同月加總。",
        "",
        "## 倒轉(rollover)處理",
        "- 各電表 digit cap 由「該表全期最大讀數的下個 10 次方」動態推估(因 NCU 同時存在 4/5/6 位數電表)。",
        f"- 若兩月讀數下降 ≥ {ROLLOVER_DROP_THRESHOLD} 且 `(cap − prev) + curr` 為合理正值,視為倒轉。",
        "- 否則(包含負值或修正後仍負)標記為 `negative_drop` 並丟棄該月。",
        "",
        "## OCR 數字錯位(digit-shift)處理",
        f"- 若連續兩月讀數位數差 ≥ {MAX_DIGIT_LEN_DIFF},視為 OCR 錯位(例如 22319507 → 2244903),",
        "  該月標記 `ocr_digit_shift` 並跳過(下個月仍以前一個有效讀數為基準)。",
        f"- 任何單表單月 kWh > {MAX_KWH_PER_METER_MONTH:,.0f} 視為仍未檢出之 OCR 噪聲,標記 `kwh_too_large` 丟棄。",
        "",
        "## 用電 Top 10 建物(114 全年)",
        "",
        "| 建物 | 全年 kWh |",
        "|---|---:|",
    ]
    for _, r in top10.iterrows():
        lines.append(f"| {r['building']} | {r['kwh']:,.0f} |")
    lines += [
        "",
        "## 後續注意",
        "- `monthly_kwh.csv` 的 `source` 欄位:`physical` = 由實體電表加總;"
        "`virtual_total_only` = 退而求其次用虛擬總表(資料較粗)。",
        "- `meter_audit.csv` 提供逐月電表審計記錄,可追溯任何單月數值來源。",
        "- 名稱對應(→ buildings.geojson UID)在下一步腳本 `match_buildings.py` 處理。",
    ]
    text = "\n".join(lines)
    return text


def main():
    print("[1/4] Loading 4 quarters...")
    long = load_all_quarters()
    print(f"      melt rows: {len(long):,}")

    print("[2/4] Computing per-meter deltas with rollover handling...")
    meter_kwh = compute_meter_deltas(long)
    n_rollover = int(meter_kwh["rollover_fix"].sum())
    print(f"      rollover fixes applied: {n_rollover}")
    print(f"      meter rows: {len(meter_kwh):,}")

    print("[3/4] Aggregating per building × month...")
    monthly = aggregate_building_monthly(meter_kwh)
    print(f"      buildings with monthly kWh: {monthly['building'].nunique()}")

    # Write outputs
    monthly.to_csv(OUT_DIR / "monthly_kwh.csv", index=False, encoding="utf-8-sig")
    meter_kwh.to_csv(OUT_DIR / "meter_audit.csv", index=False, encoding="utf-8-sig")

    print("[4/4] Writing report...")
    report = write_report(long, meter_kwh, monthly)
    (OUT_DIR / "cleaning_report.md").write_text(report, encoding="utf-8")

    print()
    print("Wrote:")
    print(f"  {OUT_DIR / 'monthly_kwh.csv'}")
    print(f"  {OUT_DIR / 'meter_audit.csv'}")
    print(f"  {OUT_DIR / 'cleaning_report.md'}")


if __name__ == "__main__":
    main()
