"""
報告「載入的建築」在同一邏輯下各欄位缺值與 inferred / metered 分佈。

對齊 `MapViewController.building_quick_view`：以 inference 年度快取為主列，
並用 PI-VD metadata 補 area / floors。

「電表已連結」：_inference 列的 `meter_name` 非空，或 `data_source == metered`
（快取常未有 meter 字串但已標記實測，兩者擇一即視為連結）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard_modules.runtime import DashboardRuntime  # noqa: E402
from src.utils import to_float as _to_float  # noqa: E402


def _finite(v: object) -> bool:
    x = _to_float(v, np.nan)
    return bool(np.isfinite(x))


def _iter_buildings(
    runtime: DashboardRuntime, year: int
) -> tuple[list[dict[str, object]], str]:
    """
    回傳與下拉選單相同的建築清單（不含 ALL）。
    notes: 說明資料來源（inference parquet 或僅 metadata）。
    """
    inference_df = runtime.get_yearly_inference(int(year))
    out: list[dict[str, object]] = []
    notes = ""

    if inference_df is not None and not inference_df.empty:
        notes = "building list from inference cache (parquet)"
        tmp = inference_df.copy()
        tmp["name"] = tmp["name"].fillna("").astype(str)
        for _, row in tmp.sort_values(["name", "uid"]).iterrows():
            uid = str(row.get("uid", "")).strip()
            if not uid:
                continue
            out.append(row.to_dict())
    elif runtime.pivd_engine and runtime.pivd_engine.metadata_scaler.is_loaded:
        notes = "no inference parquet; building list from PI-VD metadata only"
        for uid in runtime.pivd_engine.metadata_scaler.list_uids():
            uid_text = str(uid).strip()
            if not uid_text:
                continue
            meta = runtime.pivd_engine.metadata_scaler.get_metadata(uid_text) or {}
            out.append(
                {
                    "uid": uid_text,
                    "name": meta.get("name", ""),
                    "data_source": "inferred",
                    "mean_kw": np.nan,
                    "annual_kwh": np.nan,
                    "eui_kw_per_m2": np.nan,
                    "meter_name": "",
                    "area": meta.get("area", np.nan),
                    "floors": meta.get("floors", np.nan),
                }
            )
    else:
        notes = "no inference cache and no PI-VD metadata"

    return out, notes


def main() -> int:
    parser = argparse.ArgumentParser(description="Building field coverage vs dashboard quick view")
    parser.add_argument("--campus", default="ntu", help="campus id, e.g. ntu")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--csv", default="", help="optional path to write per-building CSV")
    args = parser.parse_args()

    rt = DashboardRuntime()
    cid = str(args.campus).strip().lower()
    rt.prepare_campus_shell(cid)
    rt.load_campus(cid)

    if not rt.campus_loaded:
        print("Campus failed to load.", file=sys.stderr)
        return 2

    rows_raw, list_notes = _iter_buildings(rt, args.year)
    if not rows_raw:
        print(f"No buildings to scan. ({list_notes})")
        print(f"Inference rows for {args.year}: {len(rt.get_yearly_inference(args.year))}")
        return 0

    pivd = rt.pivd_engine
    records: list[dict[str, object]] = []

    for row in rows_raw:
        uid = str(row.get("uid", "")).strip()
        meta: dict = {}
        if pivd and pivd.metadata_scaler.is_loaded:
            meta = pivd.metadata_scaler.get_metadata(uid) or {}

        record = dict(row)
        name = str(record.get("name") or meta.get("name") or uid).strip()
        source = str(record.get("data_source", "inferred")).strip().lower()

        mean_ok = _finite(record.get("mean_kw", np.nan))
        annual_ok = _finite(record.get("annual_kwh", np.nan))
        eui_ok = _finite(record.get("eui_kw_per_m2", np.nan))
        area_ok = _finite(record.get("area", meta.get("area", np.nan)))
        floors_ok = _finite(record.get("floors", meta.get("floors", np.nan)))
        meter_name_present = bool(str(record.get("meter_name", "")).strip())
        meter_linked = meter_name_present or source == "metered"

        energy_all_ok = mean_ok and annual_ok and eui_ok

        records.append(
            {
                "uid": uid,
                "name": name,
                "data_source": source,
                "mean_kw_ok": mean_ok,
                "annual_kwh_ok": annual_ok,
                "eui_kw_per_m2_ok": eui_ok,
                "area_ok": area_ok,
                "floors_ok": floors_ok,
                "meter_name_present": meter_name_present,
                "meter_linked": meter_linked,
                "energy_all_ok": energy_all_ok,
                "missing_energy_count": int(3 - (mean_ok + annual_ok + eui_ok)),
                "missing_ui_field_count": int(
                    (not mean_ok)
                    + (not annual_ok)
                    + (not eui_ok)
                    + (not area_ok)
                    + (not floors_ok)
                    + (not meter_linked)
                ),
            }
        )

    df = pd.DataFrame(records)
    n = len(df)

    def pct(x: int) -> str:
        return f"{100.0 * x / n:.1f}%" if n else "n/a"

    inferred_n = int((df["data_source"] == "inferred").sum())
    metered_n = int((df["data_source"] == "metered").sum())
    other_src_n = int(n - inferred_n - metered_n)

    summary = {
        "campus": cid,
        "year": int(args.year),
        "building_list_source": list_notes,
        "buildings_total": n,
        "data_source_inferred": inferred_n,
        "data_source_metered": metered_n,
        "data_source_other": other_src_n,
        "missing_mean_kw": int((~df["mean_kw_ok"]).sum()),
        "missing_annual_kwh": int((~df["annual_kwh_ok"]).sum()),
        "missing_eui_kw_per_m2": int((~df["eui_kw_per_m2_ok"]).sum()),
        "missing_area": int((~df["area_ok"]).sum()),
        "missing_floors": int((~df["floors_ok"]).sum()),
        "meter_name_present_count": int(df["meter_name_present"].sum()),
        "meter_linked_count": int(df["meter_linked"].sum()),
        "meter_not_linked": int((~df["meter_linked"]).sum()),
        "buildings_with_all_energy_ok": int(df["energy_all_ok"].sum()),
        "buildings_missing_any_energy": int((~df["energy_all_ok"]).sum()),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("Percentages (of all listed buildings):")
    print(f"  inferred:      {inferred_n} ({pct(inferred_n)})")
    print(f"  metered:       {metered_n} ({pct(metered_n)})")
    print(f"  other source:  {other_src_n} ({pct(other_src_n)})")
    print(f"  missing mean_kw:        {summary['missing_mean_kw']} ({pct(summary['missing_mean_kw'])})")
    print(f"  missing annual_kwh:     {summary['missing_annual_kwh']} ({pct(summary['missing_annual_kwh'])})")
    print(f"  missing eui_kw_per_m2:  {summary['missing_eui_kw_per_m2']} ({pct(summary['missing_eui_kw_per_m2'])})")
    print(f"  missing area:           {summary['missing_area']} ({pct(summary['missing_area'])})")
    print(f"  missing floors:         {summary['missing_floors']} ({pct(summary['missing_floors'])})")
    print(
        f"  meter name in cache:    {summary['meter_name_present_count']} ({pct(summary['meter_name_present_count'])})"
    )
    print(f"  meter linked (rule):    {summary['meter_linked_count']} ({pct(summary['meter_linked_count'])})")
    print(f"  meter not linked:       {summary['meter_not_linked']} ({pct(summary['meter_not_linked'])})")

    worst = df.sort_values(["missing_ui_field_count", "uid"], ascending=[False, True]).head(15)
    print()
    print("Top 15 by missing UI fields (uid, name, source, missing count):")
    for _, r in worst.iterrows():
        print(
            f"  {r['uid']}\t{r['data_source']}\tmissing={int(r['missing_ui_field_count'])}\t{r['name'][:40]}"
        )

    if args.csv:
        out_path = Path(args.csv)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nWrote {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
