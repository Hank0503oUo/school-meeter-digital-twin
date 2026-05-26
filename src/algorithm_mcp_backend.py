from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.real_inference_engine import PIVDEngine
from src.counterfactual import run_openbse_hybrid_counterfactual


def _iter_pivd_weather_dirs() -> list[Path]:
    """EPW/CSV 搜尋目錄：環境變數優先，其次 demo 內建 models/weather。"""
    root = Path(__file__).resolve().parent.parent
    dirs: list[Path] = []
    env = os.environ.get("PIVD_WEATHER_DIR", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            dirs.append(p)
    for rel in (
        root / "models" / "weather",
        root / "campuses" / "ntu" / "models" / "weather",
        root / "campuses" / "ncu" / "models" / "weather",
    ):
        if rel.is_dir() and rel.resolve() not in dirs:
            dirs.append(rel.resolve())
    return dirs


_ENGINE_LAYERS = [
    "PhysicsSurrogate",
    "V9WeightReconstructor",
    "BuildingMetadataScaler",
    "V10BootEnsemble",
]


class AlgorithmMCPBackend:
    """MCP-facing orchestration layer for algorithmic building-energy tools."""

    def __init__(self, *, engine_factory: Callable[[], PIVDEngine] | None = None) -> None:
        self._engine_factory = engine_factory or PIVDEngine.from_defaults
        self._engine: PIVDEngine | None = None

    def _get_engine(self) -> PIVDEngine:
        if self._engine is None:
            self._engine = self._engine_factory()
        return self._engine

    def run_pvid(
        self,
        building_uid: str,
        hours: int,
        t_out_series: Sequence[float],
        humidity_series: Sequence[float],
        start_time: str,
    ) -> dict[str, Any]:
        building_uid = building_uid.strip()
        started_at = time.perf_counter()
        try:
            weather_df, weather_meta = self._build_weather_frame(
                hours=hours,
                t_out_series=t_out_series,
                humidity_series=humidity_series,
                start_time=start_time,
            )
            engine = self._get_engine()
            result_df = (
                engine.predict_building(weather_df, building_uid)
                if building_uid
                else engine.predict(weather_df)
            )
            payload = self._format_pvid_payload(
                building_uid=building_uid,
                hours=len(weather_df),
                result_df=result_df,
                start_time=start_time,
                t_out_series=weather_df["t_out"].tolist(),
                humidity_series=weather_df["humidity"].tolist(),
                runtime_ms=int((time.perf_counter() - started_at) * 1000),
                weather_meta=weather_meta,
            )
            return payload
        except Exception as exc:
            return {
                "algo": "pvid",
                "status": "error",
                "error": str(exc),
                "building_uid": building_uid,
            }

    def correlate_algorithms(
        self,
        results: Sequence[dict[str, Any]],
        question: str,
        building_uid: str,
    ) -> dict[str, Any]:
        try:
            normalized_results = [item for item in results if isinstance(item, dict)]
            if not normalized_results:
                raise ValueError("results must contain at least one algorithm result dict.")

            algos_used = [
                str(item.get("algo", "")).strip()
                for item in normalized_results
                if str(item.get("algo", "")).strip()
            ]
            pvid_ctx = next(
                (
                    self._extract_pvid_context(item)
                    for item in normalized_results
                    if str(item.get("algo", "")).strip().lower() == "pvid"
                ),
                None,
            )
            counterfactual_contexts = [
                ctx
                for ctx in (
                    self._extract_counterfactual_context(item)
                    for item in normalized_results
                    if str(item.get("algo", "")).strip().lower() == "counterfactual"
                )
                if ctx is not None
            ]

            relationships: list[dict[str, Any]] = []
            factor_scores: dict[str, float] = {}

            if pvid_ctx and counterfactual_contexts:
                best_cf = min(counterfactual_contexts, key=lambda item: item["delta_pct_fraction"])
                factor = best_cf["factor"]
                factor_scores[factor] = factor_scores.get(factor, 0.0) + abs(best_cf["delta_pct_fraction"])

                if factor == "cooling_load" and pvid_ctx["is_daytime_peak"]:
                    factor_scores[factor] += 0.05

                confidence = 0.68
                if pvid_ctx["uncertainty_pct"] <= 5.0:
                    confidence += 0.08
                if factor == "cooling_load" and pvid_ctx["is_daytime_peak"]:
                    confidence += 0.08
                if abs(best_cf["delta_pct_fraction"]) >= 0.05:
                    confidence += 0.06

                relationships.append(
                    {
                        "finding": (
                            f"PVID 預測峰值 {pvid_ctx['peak_total_pred_kw']:.1f} kW 出現在 {pvid_ctx['peak_time_label']}，"
                            f"counterfactual 顯示 {best_cf['label']} 可讓負載變化 {best_cf['delta_pct_display']:+.2f}%"
                        ),
                        "confidence": round(min(confidence, 0.92), 2),
                        "type": f"{best_cf['type_prefix']}_peak_correlation",
                    }
                )

            if pvid_ctx:
                if pvid_ctx["uncertainty_pct"] >= 8.0:
                    factor_scores["operational_variability"] = factor_scores.get("operational_variability", 0.0) + 0.08
                    relationships.append(
                        {
                            "finding": (
                                f"PVID 平均不確定性約 {pvid_ctx['uncertainty_pct']:.2f}%，"
                                "代表仍有未被物理層或天氣條件解釋的操作差異。"
                            ),
                            "confidence": 0.67,
                            "type": "high_uncertainty_signal",
                        }
                    )
                elif not counterfactual_contexts:
                    factor_scores["weather_driven_load"] = factor_scores.get("weather_driven_load", 0.0) + 0.04
                    relationships.append(
                        {
                            "finding": (
                                f"PVID 峰值 {pvid_ctx['peak_total_pred_kw']:.1f} kW 出現在 {pvid_ctx['peak_time_label']}，"
                                "目前只看到單一演算法結果，建議再搭配情境分析驗證主要驅動因子。"
                            ),
                            "confidence": 0.58,
                            "type": "single_algo_signal",
                        }
                    )

            question_hint = self._infer_factor_from_text(question)
            if question_hint:
                factor_scores[question_hint] = factor_scores.get(question_hint, 0.0) + 0.03

            dominant_factor = max(factor_scores.items(), key=lambda item: item[1])[0] if factor_scores else "unknown"
            confidence = round(
                float(np.mean([item["confidence"] for item in relationships])) if relationships else 0.45,
                2,
            )
            resolved_building_uid = building_uid.strip() or self._infer_building_uid(normalized_results)

            return {
                "status": "ok",
                "building_uid": resolved_building_uid,
                "algos_used": list(dict.fromkeys(algos_used)),
                "relationships": relationships,
                "dominant_factor": dominant_factor,
                "recommended_action": self._recommended_action(dominant_factor, pvid_ctx),
                "confidence": confidence,
                "reasoning_method": "rule_based_v1",
            }
        except Exception as exc:
            return {
                "status": "error",
                "building_uid": building_uid.strip(),
                "error": str(exc),
                "reasoning_method": "rule_based_v1",
            }

    @staticmethod
    def _coerce_float_series(values: Sequence[float] | None) -> list[float]:
        if values is None:
            return []
        out: list[float] = []
        for x in values:
            try:
                if x is None:
                    continue
                out.append(float(x))
            except (TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _should_try_weather_files(
        t_out_series: Sequence[float] | None,
        humidity_series: Sequence[float] | None,
        hours: int,
    ) -> bool:
        t = AlgorithmMCPBackend._coerce_float_series(t_out_series)
        h = AlgorithmMCPBackend._coerce_float_series(humidity_series)
        return len(t) != hours or len(h) != hours

    @staticmethod
    def _normalize_weather_series(
        name: str,
        values: Sequence[float] | None,
        hours: int,
        *,
        default_c: float,
        clamp: tuple[float, float],
    ) -> tuple[list[float], str]:
        lo, hi = clamp

        def clamp_list(vals: list[float]) -> list[float]:
            return [float(min(hi, max(lo, v))) for v in vals]

        seq = AlgorithmMCPBackend._coerce_float_series(values)
        if len(seq) == 0:
            return clamp_list([default_c] * hours), "default_constant"
        if len(seq) == 1:
            return clamp_list([seq[0]] * hours), "broadcast_single_value"
        if len(seq) == hours:
            arr = np.asarray(seq, dtype=float)
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain only finite numeric values.")
            return clamp_list(arr.tolist()), "user_provided"
        if len(seq) > hours:
            arr = np.asarray(seq[:hours], dtype=float)
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain only finite numeric values.")
            return clamp_list(arr.tolist()), "user_truncated"
        out = list(seq)
        last = float(out[-1])
        while len(out) < hours:
            out.append(last)
        arr = np.asarray(out, dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain only finite numeric values.")
        return clamp_list(arr.tolist()), "user_padded"

    @staticmethod
    def _slice_weather_to_hours(wx: pd.DataFrame, start: pd.Timestamp, hours: int) -> pd.DataFrame | None:
        if wx.empty or "t_out" not in wx.columns:
            return None
        if "humidity" not in wx.columns:
            wx = wx.copy()
            wx["humidity"] = 70.0

        target_index = pd.date_range(start=start, periods=hours, freq="h")
        sub = wx.reindex(target_index)
        if len(sub) == hours and sub["t_out"].notna().all():
            out = sub.copy()
            out["humidity"] = out["humidity"].fillna(70.0)
            return out

        mask_md = (wx.index.month == start.month) & (wx.index.day == start.day)
        day_rows = wx.loc[mask_md].sort_index()
        if day_rows.empty:
            return None
        hmatch = day_rows[day_rows.index.hour == start.hour]
        t0 = hmatch.index[0] if not hmatch.empty else day_rows.index[0]
        idx_src = pd.date_range(start=t0, periods=hours, freq="h")
        sub2 = wx.reindex(idx_src)
        if len(sub2) < hours or sub2["t_out"].isna().any():
            return None
        sub2 = sub2.iloc[:hours].copy()
        sub2["humidity"] = sub2["humidity"].fillna(70.0)
        sub2.index = target_index
        return sub2

    def _load_hourly_weather_slice(self, start: pd.Timestamp, hours: int) -> tuple[pd.DataFrame, str] | None:
        from src.epw_reader import read_weather

        year = start.year
        seen_paths: set[Path] = set()
        for wdir in _iter_pivd_weather_dirs():
            candidates = (
                sorted(wdir.glob(f"*{year}*.epw"))
                + sorted(wdir.glob(f"*{year}*.csv"))
                + sorted(wdir.glob("*.epw"))
                + sorted(wdir.glob("*.csv"))
            )
            for path in candidates:
                rp = path.resolve()
                if rp in seen_paths:
                    continue
                seen_paths.add(rp)
                try:
                    wx = read_weather(path)
                except (OSError, ValueError, TypeError, KeyError):
                    continue
                if wx is None or wx.empty:
                    continue
                chunk = self._slice_weather_to_hours(wx, start, hours)
                if chunk is not None:
                    return chunk, path.name
        return None

    def _build_weather_frame(
        self,
        *,
        hours: int,
        t_out_series: Sequence[float] | None,
        humidity_series: Sequence[float] | None,
        start_time: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        try:
            hours = int(hours)
        except (TypeError, ValueError) as exc:
            raise ValueError("hours must be an integer between 1 and 168.") from exc

        if not 1 <= hours <= 168:
            raise ValueError(f"hours must be between 1 and 168; got {hours}.")

        if start_time.strip():
            try:
                start = pd.Timestamp(start_time).floor("h")
            except Exception as exc:
                raise ValueError(f"start_time must be a valid ISO 8601 string; got {start_time!r}.") from exc
        else:
            start = pd.Timestamp.now().floor("h")

        if self._should_try_weather_files(t_out_series, humidity_series, hours):
            loaded = self._load_hourly_weather_slice(start, hours)
            if loaded is not None:
                wx_df, fname = loaded
                meta = {
                    "t_out_series_source": f"weather_file:{fname}",
                    "humidity_series_source": f"weather_file:{fname}",
                    "weather_file": fname,
                    "requested_start": start.isoformat(),
                    "note": (
                        "Hourly t_out/humidity loaded from project weather files for the requested start_time. "
                        "Values are typical-year/station data (not building-specific microclimate)."
                    ),
                }
                return wx_df, meta

        t_out_values, t_src = self._normalize_weather_series(
            "t_out_series", t_out_series, hours, default_c=28.0, clamp=(-10.0, 45.0)
        )
        humidity_values, h_src = self._normalize_weather_series(
            "humidity_series", humidity_series, hours, default_c=70.0, clamp=(5.0, 100.0)
        )

        index = pd.date_range(start=start, periods=hours, freq="h")
        meta = {
            "t_out_series_source": t_src,
            "humidity_series_source": h_src,
            "note": (
                "No matching hourly row in models/weather for this start_time, or series were padded with defaults. "
                "Set PIVD_WEATHER_DIR or add *{year}*.epw / .csv; or pass full t_out_series and humidity_series (length=hours)."
            ),
        }
        return (
            pd.DataFrame(
                {
                    "t_out": t_out_values,
                    "humidity": humidity_values,
                },
                index=index,
            ),
            meta,
        )

    def _format_pvid_payload(
        self,
        *,
        building_uid: str,
        hours: int,
        result_df: pd.DataFrame,
        start_time: str,
        t_out_series: Sequence[float],
        humidity_series: Sequence[float],
        runtime_ms: int,
        weather_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "timestamps": [ts.isoformat() for ts in result_df.index.to_pydatetime()],
            "physics_pred": self._series_to_float_list(result_df["physics_pred"]),
            "residual_pred": self._series_to_float_list(result_df["residual_pred"]),
            "residual_std": self._series_to_float_list(result_df["residual_std"]),
            "total_pred": self._series_to_float_list(result_df["total_pred"]),
        }

        if building_uid and "building_rank_index" in result_df.columns:
            result["building_rank_index"] = self._series_to_float_list(result_df["building_rank_index"])

        if "building_eui_index" in result_df.columns:
            eui_values = pd.to_numeric(result_df["building_eui_index"], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(eui_values).all():
                result["building_eui_index"] = [float(value) for value in eui_values.tolist()]

        mean_total = float(np.mean(result_df["total_pred"]))
        peak_total = float(np.max(result_df["total_pred"]))
        mean_std = float(np.mean(result_df["residual_std"]))
        uncertainty_pct = (abs(mean_std) / abs(mean_total) * 100.0) if mean_total else 0.0

        input_payload = {
            "building_uid": building_uid,
            "hours": hours,
            "t_out_series": [float(value) for value in t_out_series],
            "humidity_series": [float(value) for value in humidity_series],
            "start_time": start_time,
        }

        provenance: dict[str, Any] = {
            "model_version": "pivd-v12",
            "engine_layers": list(_ENGINE_LAYERS),
            "input_hash": hashlib.sha256(
                json.dumps(input_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "runtime_ms": runtime_ms,
        }
        if weather_meta:
            provenance["weather"] = weather_meta

        return {
            "algo": "pvid",
            "status": "ok",
            "building_uid": building_uid,
            "hours": hours,
            "result": result,
            "summary": {
                "mean_total_pred_kw": mean_total,
                "peak_total_pred_kw": peak_total,
                "mean_residual_std": mean_std,
                "uncertainty_pct": uncertainty_pct,
            },
            "provenance": provenance,
        }

    @staticmethod
    def _validate_numeric_series(name: str, values: Sequence[float], hours: int) -> list[float]:
        if values is None:
            raise ValueError(f"{name} is required.")

        sequence = list(values)
        if len(sequence) != hours:
            raise ValueError(f"{name} length {len(sequence)} must equal hours={hours}.")

        try:
            array = np.asarray(sequence, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric values only.") from exc

        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite numeric values.")

        return array.astype(float).tolist()

    @staticmethod
    def _series_to_float_list(series: pd.Series | np.ndarray) -> list[float]:
        array = np.asarray(series, dtype=float)
        return [float(value) for value in array.tolist()]

    @staticmethod
    def _infer_building_uid(results: Sequence[dict[str, Any]]) -> str:
        for item in results:
            value = str(item.get("building_uid", "")).strip()
            if value:
                return value
        return ""

    def _extract_pvid_context(self, result: dict[str, Any]) -> dict[str, Any] | None:
        result_block = result.get("result")
        summary_block = result.get("summary")
        if not isinstance(result_block, dict) or not isinstance(summary_block, dict):
            return None

        timestamps = result_block.get("timestamps")
        total_pred = result_block.get("total_pred")
        if not isinstance(timestamps, list) or not isinstance(total_pred, list) or not timestamps or not total_pred:
            return None

        peak_index = int(np.argmax(np.asarray(total_pred, dtype=float)))
        peak_time = pd.Timestamp(timestamps[peak_index])
        uncertainty_pct = float(summary_block.get("uncertainty_pct", 0.0) or 0.0)

        return {
            "peak_total_pred_kw": float(summary_block.get("peak_total_pred_kw", total_pred[peak_index])),
            "peak_time": peak_time,
            "peak_time_label": peak_time.isoformat(),
            "uncertainty_pct": uncertainty_pct,
            "is_daytime_peak": 10 <= peak_time.hour <= 17,
        }

    def _extract_counterfactual_context(self, result: dict[str, Any]) -> dict[str, Any] | None:
        summary_like = result.get("summary") if isinstance(result.get("summary"), dict) else result
        try:
            delta_pct_raw = float(summary_like.get("delta_pct"))
        except (TypeError, ValueError, AttributeError):
            return None

        delta_pct_fraction = delta_pct_raw / 100.0 if abs(delta_pct_raw) > 1.0 else delta_pct_raw
        label = str(summary_like.get("label") or result.get("scenario") or "該情境").strip() or "該情境"
        factor = self._infer_factor_from_text(label, default="weather_driven_load")
        type_prefix = {
            "cooling_load": "cooling",
            "lighting_load": "lighting",
            "equipment_load": "equipment",
            "occupancy_load": "occupancy",
        }.get(factor, "scenario")
        return {
            "label": label,
            "delta_pct_fraction": float(delta_pct_fraction),
            "delta_pct_display": float(delta_pct_fraction * 100.0),
            "factor": factor,
            "type_prefix": type_prefix,
        }

    @staticmethod
    def _infer_factor_from_text(text: str, default: str | None = None) -> str | None:
        lowered = text.lower()
        if any(token in lowered for token in ("cool", "冷卻", "空調", "chiller", "hvac", "setpoint", "設定點")):
            return "cooling_load"
        if any(token in lowered for token in ("light", "照明")):
            return "lighting_load"
        if any(token in lowered for token in ("equip", "設備", "plug")):
            return "equipment_load"
        if any(token in lowered for token in ("occup", "人員", "使用率")):
            return "occupancy_load"
        return default

    def run_openbse_counterfactual(
        self,
        building_uid: str,
        cooling_delta_degC: float = 0.0,
        lighting_ratio: float = 1.0,
        occupancy_ratio: float = 1.0,
        equipment_ratio: float = 1.0,
        cop_ratio: float = 1.0,
        mean_kw_override: float | None = None,
    ) -> dict[str, Any]:
        """
        Run PI-VD + OpenBSE hybrid counterfactual for a building.

        The annual baseline is constructed as ``mean_kw * 8760`` (uniform load profile).
        OpenBSE then provides physics-accurate per-hour deltas for the requested
        parameter changes; the hybrid result is their sum.

        Parameters
        ----------
        building_uid : str
            Building UID used to look up mean_kw from the PI-VD engine metadata.
        mean_kw_override : float or None
            If provided, overrides the engine's mean_kw lookup for the building.
        """
        started_at = time.perf_counter()
        building_uid = building_uid.strip()
        try:
            # 1. Resolve mean_kw
            if mean_kw_override is not None:
                mean_kw = float(mean_kw_override)
                if mean_kw <= 0:
                    raise ValueError("mean_kw_override must be positive")
            else:
                if not building_uid:
                    raise ValueError("building_uid is required")
                engine = self._get_engine()
                building_meta = getattr(engine, "building_meta", None)
                if not isinstance(building_meta, dict):
                    raise ValueError("building_meta is unavailable")
                if building_uid not in building_meta:
                    raise ValueError("building_uid not found in building_meta")
                meta = building_meta.get(building_uid, {})
                mean_kw = float(meta.get("mean_kw", 0) or 0)
                if mean_kw <= 0:
                    raise ValueError(f"mean_kw missing or non-positive for building_uid={building_uid}")

            # 2. Build 8760-hr PI-VD baseline
            pivd_baseline = np.full(8760, mean_kw, dtype=float)

            # 3. Run OpenBSE delta
            from src.openbse_counterfactual import OpenBSEDeltaEngine  # noqa: PLC0415
            obe_engine = OpenBSEDeltaEngine()
            result = run_openbse_hybrid_counterfactual(
                pivd_baseline=pivd_baseline,
                cooling_delta_degC=float(cooling_delta_degC),
                lighting_ratio=float(lighting_ratio),
                occupancy_ratio=float(occupancy_ratio),
                equipment_ratio=float(equipment_ratio),
                cop_ratio=float(cop_ratio),
                engine=obe_engine,
            )

            runtime_ms = int((time.perf_counter() - started_at) * 1000)
            return {
                "algo": "openbse_hybrid_counterfactual",
                "status": "ok",
                "building_uid": building_uid,
                "mean_kw_used": mean_kw,
                "scenario": {
                    "cooling_delta_degC": float(cooling_delta_degC),
                    "lighting_ratio": float(lighting_ratio),
                    "occupancy_ratio": float(occupancy_ratio),
                    "equipment_ratio": float(equipment_ratio),
                    "cop_ratio": float(cop_ratio),
                },
                "summary": result.summary_dict(),
                "openbse_baseline_values": obe_engine.baseline_values,
                "provenance": {
                    "model_version": "pivd+openbse-hybrid-v1",
                    "formula": "E_new = E_PIVD_baseline + (E_OpenBSE_scenario - E_OpenBSE_baseline)",
                    "runtime_ms": runtime_ms,
                },
            }
        except Exception as exc:
            return {
                "algo": "openbse_hybrid_counterfactual",
                "status": "error",
                "building_uid": building_uid,
                "error": str(exc),
            }

    @staticmethod
    def _recommended_action(dominant_factor: str, pvid_ctx: dict[str, Any] | None) -> str:
        peak_hint = ""
        if pvid_ctx is not None:
            peak_hint = f"（峰值時段約 {pvid_ctx['peak_time_label']}）"

        actions = {
            "cooling_load": f"優先檢查冷卻控制序列、供回水溫與設定點{peak_hint}",
            "lighting_load": f"優先檢查照明排程與感測控制{peak_hint}",
            "equipment_load": f"優先盤點高耗能設備排程與待機負載{peak_hint}",
            "occupancy_load": f"優先比對實際使用率與空調/新風排程{peak_hint}",
            "operational_variability": f"先查控制紀錄、感測器漂移與異常事件{peak_hint}",
            "weather_driven_load": f"建議再搭配 counterfactual 驗證主要負載來源{peak_hint}",
        }
        return actions.get(dominant_factor, "建議補充更多演算法結果後再進一步推理")
