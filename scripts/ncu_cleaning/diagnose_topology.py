"""
Diagnose meter topology — for each building per year, classify meters into
hierarchy levels (MAIN / FEEDER / SUB-PANEL / SUB-METER) and report which
buildings risk double-counting under the current "sum all physical" rule.

Levels (heuristic from 開關箱 column):
  L1 MAIN       : "MAIN", "總表", "本表", "主表", "高壓"
  L2 FEEDER     : "MVCB", "VCB", "VCB1", "VCB2", "VCB3", "FDR", "BF"
  L3 SUB-PANEL  : "MP" (main power), "ML" (main lighting),
                  "電力盤", "電燈盤", "AC", "ACR", "P3T"
  L4 SUB-METER  : "分錶", "分表", "319-...", "iL-...", numeric panel codes

A building exhibits "topology risk" if it has BOTH L1 (MAIN) AND any of L2/L3/L4
in the same year — summing would double-count.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "outputs" / "_cleaning_diagnosis" / "topology_risk.csv"

YEARS = [109, 110, 111, 114]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def classify_meter_level(panel) -> str:
    if panel is None or (isinstance(panel, float) and pd.isna(panel)):
        return "L?_UNKNOWN"
    s = str(panel).strip().upper()
    if not s:
        return "L?_UNKNOWN"
    if any(k in s for k in ("MAIN", "總表", "主表", "本表", "高壓", "總幹線")):
        return "L1_MAIN"
    if any(k in s for k in ("MVCB", "VCB", "FDR", "饋線", "BF")):
        return "L2_FEEDER"
    if any(k in s for k in ("電力盤", "電燈盤", "MP", "ML",
                              "AC", "ACR", "P3T", "PAC")):
        return "L3_SUBPANEL"
    if any(k in s for k in ("分錶", "分表", "Sub", "SUB")) or re.match(r"\d", s):
        return "L4_SUBMETER"
    return "L5_OTHER"


def main():
    rows = []
    for roc in YEARS:
        audit = pd.read_csv(ROOT / f"outputs/ncu_{roc}/meter_audit.csv",
                            encoding="utf-8-sig")
        # Skip placeholder/aggregate meters
        audit = audit[audit["meter_kind"] == "physical"].copy()
        audit = audit.dropna(subset=["building", "kwh"])
        audit["level"] = audit["panel"].map(classify_meter_level)

        for bld, sub in audit.groupby("building"):
            level_counts = sub["level"].value_counts().to_dict()
            level_kwh = sub.groupby("level")["kwh"].sum().to_dict()
            has_main = "L1_MAIN" in level_counts
            has_sub = any(k in level_counts for k in
                          ("L2_FEEDER", "L3_SUBPANEL", "L4_SUBMETER"))
            risk = has_main and has_sub
            total_kwh_summed = sub["kwh"].sum()
            main_only_kwh = level_kwh.get("L1_MAIN", 0.0)
            rows.append({
                "roc_year": roc,
                "building": bld,
                "topology_risk": risk,
                "total_kwh_current_method": round(total_kwh_summed, 0),
                "main_only_kwh": round(main_only_kwh, 0),
                "ratio_main_to_total": round(main_only_kwh / total_kwh_summed, 3)
                                            if total_kwh_summed > 0 else 0,
                "n_L1_MAIN": level_counts.get("L1_MAIN", 0),
                "n_L2_FEEDER": level_counts.get("L2_FEEDER", 0),
                "n_L3_SUBPANEL": level_counts.get("L3_SUBPANEL", 0),
                "n_L4_SUBMETER": level_counts.get("L4_SUBMETER", 0),
                "n_L5_OTHER": level_counts.get("L5_OTHER", 0),
                "n_unknown": level_counts.get("L?_UNKNOWN", 0),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("=== Topology risk per year ===")
    risk_summary = df.groupby("roc_year").agg(
        n_buildings=("building", "nunique"),
        n_at_risk=("topology_risk", "sum"),
        total_kwh_GWh=("total_kwh_current_method", lambda s: round(s.sum() / 1e6, 2)),
        risky_total_kwh_GWh=("topology_risk", lambda s: round(
            df.loc[s.index].loc[s, "total_kwh_current_method"].sum() / 1e6, 2
        )),
        risky_main_only_GWh=("topology_risk", lambda s: round(
            df.loc[s.index].loc[s, "main_only_kwh"].sum() / 1e6, 2
        )),
    )
    print(risk_summary.to_string())

    print()
    print("=== Top 15 buildings at greatest double-count risk (by 114) ===")
    risky_114 = df[(df["roc_year"] == 114) & df["topology_risk"]].copy()
    risky_114["over_count_kwh"] = (risky_114["total_kwh_current_method"]
                                    - risky_114["main_only_kwh"])
    risky_114 = risky_114.sort_values("over_count_kwh", ascending=False).head(15)
    cols = ["building", "n_L1_MAIN", "n_L2_FEEDER", "n_L3_SUBPANEL",
            "n_L4_SUBMETER", "total_kwh_current_method", "main_only_kwh",
            "ratio_main_to_total"]
    print(risky_114[cols].to_string(index=False))

    print()
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
