"""
End-to-end PIVD + Counterfactual demo for NCU 114 (2025) — using NTU model
skeleton + 2024 Taipei weather as proxy.

Pipeline:
  1. Load NTU PIVDEngine (Layers 1-3 trained on NTU 2017 data).
  2. Feed 2024 Taipei hourly weather → hourly physics_pred (Layer 1+2 only;
     Layer 3 residual deliberately ignored for NCU because residual is
     NTU-building-specific and would mislead).
  3. Pick top-N NCU buildings by 114 actual kWh from monthly_kwh_with_uid.csv.
  4. For each, fit a multiplicative scale so monthly aggregated physics_pred
     matches each building's actual monthly kWh shape.
  5. Run counterfactual scenarios on the scaled physics_pred:
       a. Cooling setpoint −2 °C (more cooling)
       b. Cooling setpoint +2 °C (less cooling)
       c. Lighting ratio 0.80 (LED retrofit)
       d. Equipment ratio 0.90
  6. Output:
       outputs/ncu_114/pivd_hourly_2024.csv     8784-hour PIVD output
       outputs/ncu_114/building_scaling.csv     per-building fit scale
       outputs/ncu_114/counterfactual_results.csv  per-(building, scenario) KPIs
       outputs/ncu_114/demo_chart.png           visualization
       outputs/ncu_114/demo_report.md           narrative summary
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.real_inference_engine import PIVDEngine
from src.counterfactual import run_counterfactual
from src.epw_reader import read_weather

OUT_DIR = ROOT / "outputs" / "ncu_114"
WEATHER_CSV = ROOT / "models" / "weather" / "CWBTP_2024.csv"
MONTHLY_UID_CSV = OUT_DIR / "monthly_kwh_with_uid.csv"

TOP_N = 6
SCENARIOS = [
    {"label": "冷卻 −2°C(調冷)",       "cooling_delta_degC": -2.0},
    {"label": "冷卻 +2°C(調暖)",       "cooling_delta_degC":  2.0},
    {"label": "燈光 −20%(LED retrofit)","lighting_ratio": 0.80},
    {"label": "設備 −10%",              "equipment_ratio": 0.90},
]

# Some Chinese fonts available on Windows; matplotlib defaults can't render CJK.
plt.rcParams["font.family"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_weather(path: Path) -> pd.DataFrame:
    print(f"[1/6] Reading weather: {path.name}")
    wx = read_weather(path)
    wx = wx.sort_index()
    print(f"      hours: {len(wx)}, t_out [{wx['t_out'].min():.1f}, "
          f"{wx['t_out'].max():.1f}] °C, mean RH {wx['humidity'].mean():.0f}%")
    return wx


def run_engine(weather_df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Initializing PIVDEngine (NTU skeleton)…")
    engine = PIVDEngine.from_defaults()
    print("[3/6] Running PIVD over 2024 hourly weather…")
    pred = engine.predict(weather_df)
    print(f"      physics_pred mean={pred['physics_pred'].mean():.2f}, "
          f"residual_pred mean={pred['residual_pred'].mean():.2f}, "
          f"residual_std mean={pred['residual_std'].mean():.2f}")
    return pred


def fit_building_scale(physics_monthly: pd.Series,
                       building_actual: pd.Series) -> tuple[float, dict]:
    """Find scalar k such that k * physics_monthly best matches actual.

    physics_monthly and building_actual are both Series indexed by month (1-12).
    """
    # Align: only months where both have data
    aligned = pd.concat({"phys": physics_monthly, "act": building_actual},
                       axis=1).dropna()
    if aligned.empty or aligned["phys"].sum() == 0:
        return 0.0, {"r2": float("nan"), "n_months": 0}
    # Closed-form least-squares scale (no intercept): k = sum(act*phys) / sum(phys^2)
    k = float((aligned["act"] * aligned["phys"]).sum() / (aligned["phys"] ** 2).sum())
    pred = k * aligned["phys"]
    ss_res = ((aligned["act"] - pred) ** 2).sum()
    ss_tot = ((aligned["act"] - aligned["act"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return k, {"r2": float(r2), "n_months": int(len(aligned))}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    weather_df = load_weather(WEATHER_CSV)
    pivd = run_engine(weather_df)

    # Save raw PIVD output
    pivd.to_csv(OUT_DIR / "pivd_hourly_2024.csv", encoding="utf-8")
    print(f"      wrote {OUT_DIR / 'pivd_hourly_2024.csv'}")

    # Aggregate physics_pred to monthly (use physics_pred only — Layer 1+2)
    physics_hourly = pivd["physics_pred"].clip(lower=0)
    monthly_phys = physics_hourly.groupby(physics_hourly.index.month).sum()
    monthly_phys.name = "physics_kwh"

    # Load NCU actual monthly per building
    print("[4/6] Loading NCU 114 actual monthly kWh + UID mapping…")
    actual = pd.read_csv(MONTHLY_UID_CSV, encoding="utf-8-sig")
    actual = actual.dropna(subset=["osm_id"]).copy()
    actual["osm_id"] = actual["osm_id"].astype("int64")

    # Pick top N by total annual kWh
    annual = (actual.groupby(["building", "osm_id", "geojson_name"], as_index=False)
                    ["kwh"].sum()
                    .sort_values("kwh", ascending=False)
                    .head(TOP_N))
    print(f"      top {TOP_N} buildings:")
    for _, r in annual.iterrows():
        print(f"        {r['building']:24s}  {r['kwh']:>12,.0f} kWh")

    # Fit per-building scale and run counterfactual
    print("[5/6] Fitting per-building scale + running counterfactual scenarios…")
    fit_rows = []
    cf_rows = []
    chart_data = {}  # building → dict(month, actual, baseline_phys, scenarios)

    for _, b in annual.iterrows():
        bname = b["building"]
        uid = b["osm_id"]
        bld_actual = (actual[actual["osm_id"] == uid]
                      .set_index("month")["kwh"])
        k, fit_info = fit_building_scale(monthly_phys, bld_actual)
        fit_rows.append({
            "building": bname,
            "osm_id": uid,
            "geojson_name": b["geojson_name"],
            "annual_actual_kwh": b["kwh"],
            "scale_k": k,
            "fit_r2": fit_info["r2"],
            "fit_n_months": fit_info["n_months"],
        })

        # Hourly baseline kWh for THIS building = physics_hourly * k
        baseline_hourly = physics_hourly * k
        baseline_total = baseline_hourly.sum()

        bld_charts = {
            "actual_monthly": bld_actual,
            "baseline_monthly": (baseline_hourly
                                 .groupby(baseline_hourly.index.month).sum()),
            "scenarios": {},
        }

        for scen in SCENARIOS:
            label = scen["label"]
            cf = run_counterfactual(
                baseline_kwh=baseline_hourly.values,
                cooling_delta_degC=scen.get("cooling_delta_degC", 0.0),
                lighting_ratio=scen.get("lighting_ratio", 1.0),
                equipment_ratio=scen.get("equipment_ratio", 1.0),
                label=label,
            )
            kpis = cf.summary_dict()
            cf_rows.append({
                "building": bname,
                "osm_id": uid,
                "scenario": label,
                "baseline_kwh": float(baseline_total),
                "new_kwh": float(baseline_total + kpis["delta_kwh"]),
                **kpis,
            })
            bld_charts["scenarios"][label] = pd.Series(
                cf.timeseries_new, index=baseline_hourly.index
            ).groupby(baseline_hourly.index.month).sum()

        chart_data[bname] = bld_charts

    fit_df = pd.DataFrame(fit_rows)
    cf_df = pd.DataFrame(cf_rows)
    fit_df.to_csv(OUT_DIR / "building_scaling.csv", index=False, encoding="utf-8-sig")
    cf_df.to_csv(OUT_DIR / "counterfactual_results.csv", index=False, encoding="utf-8-sig")

    # Save monthly physics aggregate as reference
    monthly_phys.to_frame().to_csv(OUT_DIR / "physics_monthly_aggregate.csv",
                                    encoding="utf-8-sig")

    # ── Visualization: 2x3 grid of (top 6 buildings × actual vs scenarios) ──
    print("[6/6] Rendering chart…")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    months = list(range(1, 13))
    for ax, (bname, data) in zip(axes, chart_data.items()):
        actual_m = data["actual_monthly"].reindex(months)
        baseline_m = data["baseline_monthly"].reindex(months)
        ax.plot(months, actual_m / 1000, "ko-", label="實際量測", linewidth=2)
        ax.plot(months, baseline_m / 1000, "b--", label="PIVD baseline", alpha=0.7)
        for label, series in data["scenarios"].items():
            ax.plot(months, series.reindex(months) / 1000, "--", alpha=0.5,
                    label=label)
        ax.set_title(f"{bname}", fontsize=11)
        ax.set_xlabel("月")
        ax.set_ylabel("MWh")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle(
        "NCU 114(2025)反事實情境分析 — NTU PIVD 骨架 + 2024 台北氣象代理",
        fontsize=13,
    )
    fig.tight_layout()
    chart_path = OUT_DIR / "demo_chart.png"
    fig.savefig(chart_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"      wrote {chart_path}")

    # ── Markdown report ──
    lines = [
        "# NCU 114 PIVD + Counterfactual Demo 結果",
        "",
        "## 設定",
        "- **物理引擎**:`PIVDEngine.from_defaults()` (NTU 2017 訓練,Layer 1+2 物理層輸出)",
        "- **氣象**:2024 台北 Songshan 站(meteostat / NOAA ISD)做為 NCU 中壢的代理",
        "- **真實量測**:NCU 114 (2025) 月度電表清洗結果(`monthly_kwh_with_uid.csv`)",
        "- **層 3 殘差**:**故意不採用**(NTU-building-specific,套用到 NCU 會誤導)",
        "",
        "## Top 6 建物 baseline 擬合",
        "",
        "| 建物 | osm_id | 全年實際 kWh | scale k | R² | 月數 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in fit_df.iterrows():
        lines.append(
            f"| {r['building']} | {int(r['osm_id'])} | "
            f"{r['annual_actual_kwh']:,.0f} | {r['scale_k']:.4f} | "
            f"{r['fit_r2']:.2f} | {int(r['fit_n_months'])} |"
        )

    lines += [
        "",
        "## 反事實情境 KPI(平均 across Top 6 建物)",
        "",
        "| 情境 | Δ kWh / 棟 / 年 | Δ % | Δ 噸 CO₂ | Δ NTD |",
        "|---|---:|---:|---:|---:|",
    ]
    for scen in SCENARIOS:
        sub = cf_df[cf_df["scenario"] == scen["label"]]
        lines.append(
            f"| {scen['label']} | {sub['delta_kwh'].mean():,.0f} | "
            f"{sub['delta_pct'].mean():.2f}% | "
            f"{sub['delta_carbon_kg'].mean()/1000:.1f} | "
            f"{sub['delta_ntd'].mean():,.0f} |"
        )

    lines += [
        "",
        "## 解讀(demo 用話術)",
        "- **PIVD 物理層成功跨校移植**:同一套 NTU 訓練的 Physics Surrogate + V9 重建",
        "  權重,改餵 2024 台北氣象,即輸出可解讀的小時級物理預測。",
        "- **真實 vs 物理對照**:每棟用 closed-form least-squares 擬合一個 scale k,",
        "  R² 顯示物理形狀與該建物實際月度的相關度;k 則代表該棟相對 NTU 校區尺度的因子。",
        "- **反事實推理示範**:在物理 baseline 上施加單一參數變化(冷卻溫度、燈光、設備),",
        "  立即得到節能、減碳、省錢、等同種樹數的結構化 KPI。",
        "- **未來 Phase 2 待補**:Chungli 在地氣象、NCU 自有 Layer 3 殘差再訓練、",
        "  Solar API 補建物高度/樓層 → 真實 EUI 排名。",
        "",
        "圖表:`demo_chart.png`(2×3,六棟 Top 建物 × 實際/baseline/4 情境)",
    ]
    (OUT_DIR / "demo_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"      wrote {OUT_DIR / 'demo_report.md'}")
    print()
    print("=== Done ===")


if __name__ == "__main__":
    main()
