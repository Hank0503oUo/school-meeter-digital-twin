# -*- coding: utf-8 -*-
"""
完整 PI-VD 推論引擎 — 三層架構，任意年份即時模擬。

Layer 1: PhysicsSurrogate   — 5 個 HistGBR 取代 EnergyPlus (sim_A~E)
Layer 2: V9WeightReconstructor — 用 YAML 權重重建 predicted_physics
Layer 3: V10BootEnsemble    — 載入 pkl 殘差 ensemble → mean + σ

Usage:
    engine = PIVDEngine.from_defaults()
    result = engine.predict(weather_df)
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from src.campus_config import CampusConfig, CampusConfigError

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
except ImportError:
    HistGradientBoostingRegressor = None
    Ridge = None

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

log = logging.getLogger(__name__)

# ── 預設路徑（向後相容 + 校區設定）─────────────────────
_DEMO_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _DEMO_ROOT / "models"
_LEGACY_DEFAULT_DATASET = _MODELS_DIR / "v10_boot_dataset_2017.csv"
_LEGACY_DEFAULT_V9_YAML = _MODELS_DIR / "best_tow_adaptive_v9.yaml"
_LEGACY_DEFAULT_BOOT_PKL = _MODELS_DIR / "v10_boot_ensemble.pkl"
_LEGACY_DEFAULT_V12_SUMMARY = _MODELS_DIR / "v12_per_building_summary.csv"
_LEGACY_DEFAULT_METER_CSV = _MODELS_DIR / "NTU_powerMeter_kW_hourly.csv"
_LEGACY_DEFAULT_EPW_DIR = _MODELS_DIR / "weather"
_LEGACY_DEFAULT_METADATA_UID = _DEMO_ROOT / "data" / "BUILD DATA" / "metadata_uid.csv"
_DEFAULT_CAMPUS_ID = "ntu"


def _load_default_campus_config() -> CampusConfig | None:
    try:
        return CampusConfig.load(_DEFAULT_CAMPUS_ID)
    except CampusConfigError:
        return None
    except (ImportError, ModuleNotFoundError):
        return None


_DEFAULT_CAMPUS_CONFIG = _load_default_campus_config()


def _campus_or_legacy_path(path_key: str, legacy_path: Path) -> Path:
    if _DEFAULT_CAMPUS_CONFIG is None:
        return legacy_path
    from_cfg = _DEFAULT_CAMPUS_CONFIG.get_path(path_key)
    return from_cfg if from_cfg is not None else legacy_path


_DEFAULT_DATASET = _campus_or_legacy_path("v10_dataset", _LEGACY_DEFAULT_DATASET)
_DEFAULT_V9_YAML = _campus_or_legacy_path("v9_yaml", _LEGACY_DEFAULT_V9_YAML)
_DEFAULT_BOOT_PKL = _campus_or_legacy_path("v10_ensemble", _LEGACY_DEFAULT_BOOT_PKL)
_DEFAULT_V12_SUMMARY = _campus_or_legacy_path("v12_summary", _LEGACY_DEFAULT_V12_SUMMARY)
_DEFAULT_METER_CSV = _campus_or_legacy_path("meter_csv", _LEGACY_DEFAULT_METER_CSV)
_DEFAULT_EPW_DIR = _campus_or_legacy_path("weather_dir", _LEGACY_DEFAULT_EPW_DIR)
_DEFAULT_METADATA_UID = _campus_or_legacy_path("metadata_uid", _LEGACY_DEFAULT_METADATA_UID)

# 16 個 V10 BOOT 輸入特徵
FEATURE_COLS = [
    "hour", "dow", "month",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "t_out", "humidity",
    "sim_A", "sim_B", "sim_C", "sim_D", "sim_E",
]

ARCHETYPE_KEYS = ["A", "B", "C", "D", "E"]

# 用於訓練 surrogate 的輸入特徵 (不含 sim 本身)
_SURROGATE_INPUT_COLS = [
    "hour", "dow", "month",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "t_out", "humidity",
]


# ═══════════════════════════════════════════════════════════
# Layer 1: Physics Surrogate
# ═══════════════════════════════════════════════════════════

class PhysicsSurrogate:
    """
    用 HistGBR 擬合 5 個 archetype 的 EnergyPlus 輸出。

    訓練一次（~1秒），之後 predict 只需 numpy 運算。
    """

    def __init__(self):
        self.models: Dict[str, HistGradientBoostingRegressor] = {}
        self._trained = False

    def train(self, dataset_path: Path | str = _DEFAULT_DATASET) -> "PhysicsSurrogate":
        """從快取 dataset CSV 訓練 5 個 surrogate 模型 (多核並行)。"""
        if HistGradientBoostingRegressor is None:
            raise ImportError("scikit-learn is required for PhysicsSurrogate")

        dataset_path = Path(dataset_path)
        log.info(f"Training physics surrogate from: {dataset_path}")
        df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)

        X = df[_SURROGATE_INPUT_COLS].values

        def train_one(key):
            col = f"sim_{key}"
            y = df[col].values
            model = HistGradientBoostingRegressor(
                max_iter=200, max_depth=5, learning_rate=0.1,
                min_samples_leaf=20, random_state=42,
            )
            model.fit(X, y)
            r2 = model.score(X, y)
            return key, model, r2, col

        if _HAS_JOBLIB:
            results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(train_one)(key) for key in ARCHETYPE_KEYS
            )
        else:
            results = [train_one(key) for key in ARCHETYPE_KEYS]

        for key, model, r2, col in results:
            self.models[key] = model
            log.info(f"  Surrogate {col}: R² = {r2:.4f}")

        self._trained = True
        return self

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        預測 sim_A ~ sim_E。

        Parameters
        ----------
        features_df : DataFrame with columns matching _SURROGATE_INPUT_COLS

        Returns
        -------
        DataFrame with columns ['sim_A', 'sim_B', 'sim_C', 'sim_D', 'sim_E']
        """
        if not self._trained:
            raise RuntimeError("PhysicsSurrogate has not been trained yet. Call .train() first.")

        X = features_df[_SURROGATE_INPUT_COLS].values
        result = {}
        for key in ARCHETYPE_KEYS:
            result[f"sim_{key}"] = self.models[key].predict(X)
        return pd.DataFrame(result, index=features_df.index)


# ═══════════════════════════════════════════════════════════
# Layer 2: V9 Weight Reconstruction
# ═══════════════════════════════════════════════════════════

class V9WeightReconstructor:
    """
    使用 best_tow_adaptive YAML 的 context-interaction 權重
    從 sim_A~E 重建 predicted_physics。

    公式: predicted_physics = intercept + Σ_k [w_base_k + Σ_j w_int_k_j * ctx_j] * sim_k
    """

    # context interaction 特徵名稱 (與 YAML key 對應)
    _CTX_FEATURES = [
        "month_sin", "month_cos", "hour_sin", "hour_cos",
        "dow_sin", "dow_cos", "t_out", "humidity",
        "t_out_sq", "t_out_x_hour_sin", "t_out_x_month_cos",
    ]

    def __init__(self):
        self.intercept: float = 0.0
        self.w_base: Dict[str, float] = {}
        self.w_int: Dict[str, Dict[str, float]] = {}  # w_int[arch_key][ctx_feat]
        self._loaded = False

    def load(self, yaml_path: Path | str = _DEFAULT_V9_YAML) -> "V9WeightReconstructor":
        """從 YAML 載入權重。"""
        yaml_path = Path(yaml_path)
        log.info(f"Loading V9 weights from: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        attrs = cfg.get("user_attrs", {})
        self.intercept = float(attrs.get("w_intercept", 0.0))

        for key in ARCHETYPE_KEYS:
            self.w_base[key] = float(attrs.get(f"w_base_{key}", 0.0))
            self.w_int[key] = {}
            for feat in self._CTX_FEATURES:
                w = attrs.get(f"w_int_{key}_{feat}", 0.0)
                self.w_int[key][feat] = float(w)

        self._loaded = True
        log.info(f"  V9 weights loaded: intercept={self.intercept:.2f}")
        return self

    def predict_physics(self, features_df: pd.DataFrame, sim_df: pd.DataFrame) -> np.ndarray:
        """
        重建 predicted_physics。

        Parameters
        ----------
        features_df : DataFrame with cyclical time + weather columns
        sim_df : DataFrame with sim_A ~ sim_E columns

        Returns
        -------
        ndarray of predicted_physics values
        """
        if not self._loaded:
            raise RuntimeError("V9 weights not loaded. Call .load() first.")

        n = len(features_df)

        # 建立 context 特徵矩陣（含 interaction terms）
        ctx = pd.DataFrame(index=features_df.index)
        for col in self._CTX_FEATURES:
            if col == "t_out_sq":
                ctx[col] = features_df["t_out"] ** 2
            elif col == "t_out_x_hour_sin":
                ctx[col] = features_df["t_out"] * features_df["hour_sin"]
            elif col == "t_out_x_month_cos":
                ctx[col] = features_df["t_out"] * features_df["month_cos"]
            else:
                ctx[col] = features_df[col]

        # 計算 predicted_physics
        result = np.full(n, self.intercept)
        for key in ARCHETYPE_KEYS:
            sim_col = f"sim_{key}"
            sim_vals = sim_df[sim_col].values

            # w_effective(t) = w_base + Σ w_int_j * ctx_j(t)
            w_eff = np.full(n, self.w_base[key])
            for feat, w_val in self.w_int[key].items():
                w_eff += w_val * ctx[feat].values

            result += w_eff * sim_vals

        return result

    def train_from_dataset(
        self,
        dataset_path: Path | str = _DEFAULT_DATASET,
    ) -> "V9WeightReconstructor":
        """
        V9 線上訓練：從 V10 快取資料集直接擬合 context-interaction 權重。

        這是歷史 ToW 優化流程中 _solve_context_interaction_weights 的輕量版，
        用 Ridge 回歸在全部資料上擬合：
            measured = Σ_k alpha_k · sim_k + Σ_{k,f} beta_{k,f} · sim_k · ctx_f + intercept

        不需要跑 EnergyPlus 或 Optuna。
        """
        if Ridge is None:
            raise ImportError("scikit-learn is required for V9 training")

        log.info(f"V9 on-the-fly training from: {dataset_path}")
        df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)

        # 準備目標值
        measured = df["measured"].values

        # 準備 sim 矩陣
        sim_cols = [f"sim_{k}" for k in ARCHETYPE_KEYS]
        S = df[sim_cols].values  # [n, 5]

        # 準備 context 特徵
        cal = build_calendar_features(df.index)
        for col in ["t_out", "humidity"]:
            cal[col] = df[col].values
        cal["t_out_sq"] = cal["t_out"] ** 2
        cal["t_out_x_hour_sin"] = cal["t_out"] * cal["hour_sin"]
        cal["t_out_x_month_cos"] = cal["t_out"] * cal["month_cos"]

        interaction_feats = [f for f in self._CTX_FEATURES if f in cal.columns]

        # 建立 X = [S | S*ctx1 | S*ctx2 | ...]
        X_parts = [S]
        feature_names = [f"base:{k}" for k in ARCHETYPE_KEYS]
        for feat in interaction_feats:
            c = cal[feat].values.reshape(-1, 1)
            X_parts.append(S * c)
            feature_names.extend([f"{k}*{feat}" for k in ARCHETYPE_KEYS])

        X = np.concatenate(X_parts, axis=1)

        # Ridge 回歸
        model = Ridge(alpha=1.0)
        model.fit(X, measured)
        coef = model.coef_
        intercept = model.intercept_

        # 解析權重
        self.intercept = float(intercept)
        K = len(ARCHETYPE_KEYS)
        for i, key in enumerate(ARCHETYPE_KEYS):
            self.w_base[key] = float(coef[i])
            self.w_int[key] = {}

        offset = K
        for feat in interaction_feats:
            for i, key in enumerate(ARCHETYPE_KEYS):
                self.w_int[key][feat] = float(coef[offset + i])
            offset += K

        # 驗證
        y_pred = model.predict(X)
        r = np.corrcoef(measured, y_pred)[0, 1]
        log.info(f"  V9 on-the-fly training done: R={r:.4f}, intercept={self.intercept:.2f}")

        self._loaded = True
        return self


# ═══════════════════════════════════════════════════════════
# Layer 3: V10 BOOT Residual Ensemble
# ═══════════════════════════════════════════════════════════

class V10BootEnsemble:
    """
    載入 V10 BOOT ensemble pkl，回傳殘差預測 + 不確定性。
    """

    def __init__(self):
        self.models = []
        self._loaded = False

    def load(self, pkl_path: Path | str = _DEFAULT_BOOT_PKL) -> "V10BootEnsemble":
        """載入 pickle 模型。"""
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            raise FileNotFoundError(f"BOOT ensemble not found: {pkl_path}")

        log.info(f"Loading V10 BOOT ensemble from: {pkl_path}")
        data = pkl_path.read_bytes()
        file_hash = hashlib.sha256(data).hexdigest()[:16]
        log.info(f"Model file hash: {file_hash} ({pkl_path.name})")
        payload = pickle.loads(data)

        if isinstance(payload, dict):
            self.models = payload.get("models", [])
        elif isinstance(payload, list):
            self.models = payload
        else:
            raise ValueError(f"Unexpected pkl format: {type(payload)}")

        if not self.models:
            raise ValueError(f"BOOT ensemble contains no models: {pkl_path}")

        self._loaded = True
        log.info(f"Loaded {len(self.models)} BOOT models")
        return self

    @property
    def is_loaded(self) -> bool:
        return self._loaded and bool(self.models)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        預測殘差 (residual) 與不確定性 (std)。

        Parameters
        ----------
        X : ndarray of shape (n_samples, 16) — V10 BOOT 特徵

        Returns
        -------
        (mean_residual, std_residual) each of shape (n_samples,)
        """
        if not self.is_loaded:
            raise RuntimeError("V10 BOOT models not loaded. Call .load() first.")

        if _HAS_JOBLIB and len(self.models) > 4:
            # Parallel prediction across bootstrap models
            preds_list = Parallel(n_jobs=-1, prefer="threads")(
                delayed(m.predict)(X) for m in self.models
            )
            preds = np.column_stack(preds_list)
        else:
            preds = np.column_stack([m.predict(X) for m in self.models])
        return preds.mean(axis=1), preds.std(axis=1)


# ═══════════════════════════════════════════════════════════
# Layer 2.5: Building Metadata Scaler
# ═══════════════════════════════════════════════════════════

# 建物用途 EUI 修正因子（來自台灣能效基準統計）
_BUILDING_TYPE_FACTORS: Dict[str, float] = {
    "Academic Units": 1.10,         # 學術研究用電密度高
    "Instructional Building": 0.85, # 教學大樓相對低
    "Library": 1.20,                # 圖書館 24hr 空調
    "Administration": 0.80,         # 行政辦公
    "Dormitories": 0.55,            # 宿舍最低
    "Athletics": 0.60,              # 運動設施
    "Student AC": 0.75,             # 活動中心
    "Others": 0.90,                 # 其他
}


class BuildingMetadataScaler:
    """
    Layer 2.5: 根據建物物理 metadata 計算逐棟修正因子。

    使用 metadata_uid.csv 的面積、樓層、地下室、建物類型產生 scaler，
    讓 PI-VD 從「全校統一預測」升級為「逐棟自適應預測」。

    scaler 公式:
        area_factor   = area / median_area
        floor_factor  = log2(floors + basement + 1) / log2(median_floors + 1)
        type_factor   = 用途修正係數
        scaler        = area_factor × floor_factor × type_factor，clamp 至 [0.1, 10.0]
    """

    def __init__(self):
        self._metadata: Dict[str, dict] = {}  # uid -> {area, floors, basement, type, ...}
        self._scalers: Dict[str, float] = {}   # uid -> scaler
        self._loaded = False

    def load(
        self,
        csv_path: Path | str = _DEFAULT_METADATA_UID,
    ) -> "BuildingMetadataScaler":
        """從 metadata_uid.csv 載入建物 metadata 並計算 scalers。"""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            log.warning(f"Metadata CSV not found: {csv_path}")
            self._loaded = True
            return self

        log.info(f"Loading building metadata from: {csv_path}")
        df = pd.read_csv(csv_path, encoding="utf-8")

        # 解析每棟建物的 metadata
        areas = []
        floors_list = []
        for _, row in df.iterrows():
            uid = str(row.get("uid", "")).strip()
            if not uid:
                continue

            # 面積: 處理逗號千分位和非數值
            raw_area = str(row.get("area", "")).replace(",", "")
            try:
                area = float(raw_area)
            except (ValueError, TypeError):
                area = np.nan

            # 樓層
            try:
                fl = int(float(str(row.get("floor", row.get("floors", 1)))))
            except (ValueError, TypeError):
                fl = 1

            # 地下室
            try:
                bsmt = int(float(str(row.get("basement", 0))))
            except (ValueError, TypeError):
                bsmt = 0

            # 建物類型
            btype = str(row.get("buildType1E", "")).strip()

            # 建造年份
            raw_year = str(row.get("year", "")).strip()
            try:
                byear = int(float(raw_year.split("/")[0]))
            except (ValueError, TypeError):
                byear = 0

            # 建物名稱
            bname = str(row.get("name", "")).strip()
            bnameE = str(row.get("nameE", "")).strip()

            self._metadata[uid] = {
                "uid": uid,
                "name": bname,
                "nameE": bnameE,
                "area": area,
                "floors": fl,
                "basement": bsmt,
                "buildType": btype,
                "year": byear,
            }

            if np.isfinite(area) and area > 0:
                areas.append(area)
            if fl > 0:
                floors_list.append(fl)

        # 中位數作為正規化基準
        median_area = float(np.median(areas)) if areas else 3000.0
        median_floors = float(np.median(floors_list)) if floors_list else 4.0

        # 計算每棟的 scaler
        for uid, meta in self._metadata.items():
            area = meta["area"]
            fl = meta["floors"]
            bsmt = meta["basement"]
            btype = meta["buildType"]

            # area_factor: 正規化面積
            if np.isfinite(area) and area > 0:
                area_factor = area / median_area
            else:
                area_factor = 1.0

            # floor_factor: 對數縮放
            total_floors = max(fl + bsmt, 1)
            floor_factor = np.log2(total_floors + 1) / np.log2(median_floors + 1)

            # type_factor: 用途修正
            type_factor = _BUILDING_TYPE_FACTORS.get(btype, 1.0)

            scaler = float(np.clip(
                area_factor * floor_factor * type_factor,
                0.1, 10.0,
            ))
            self._scalers[uid] = scaler

        self._loaded = True
        log.info(
            f"  BuildingMetadataScaler: {len(self._scalers)} buildings, "
            f"median area={median_area:.0f}m², median floors={median_floors:.0f}"
        )
        return self

    def get_scaler(self, uid: str) -> float:
        """取得指定建物的修正因子。未知 uid 回傳 1.0。"""
        return self._scalers.get(uid, 1.0)

    def get_metadata(self, uid: str) -> Optional[dict]:
        """取得指定建物的 metadata dict。未知 uid 回傳 None。"""
        return self._metadata.get(uid)

    def list_uids(self) -> list:
        """列出所有已載入的 UID。"""
        return list(self._metadata.keys())

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ═══════════════════════════════════════════════════════════
# 整合引擎
# ═══════════════════════════════════════════════════════════

def build_calendar_features(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
    """從 DatetimeIndex 產生所有日曆特徵 (cyclical encoding)。"""
    hour = dt_index.hour.astype(float)
    dow = dt_index.dayofweek.astype(float)
    month = dt_index.month.astype(float)

    return pd.DataFrame({
        "hour": hour,
        "dow": dow,
        "month": month,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
    }, index=dt_index)


class PIVDEngine:
    """
    完整 PI-VD 推論引擎。

    四層架構：
        1)   PhysicsSurrogate        → sim_A~E
        2)   V9WeightReconstructor   → predicted_physics
        2.5) BuildingMetadataScaler  → 逐棟修正因子
        3)   V10BootEnsemble         → residual + uncertainty
    """

    def __init__(self):
        self.surrogate = PhysicsSurrogate()
        self.v9_weights = V9WeightReconstructor()
        self.v10_boot = V10BootEnsemble()
        self.metadata_scaler = BuildingMetadataScaler()
        self._ready = False

    @classmethod
    def from_defaults(cls) -> "PIVDEngine":
        """向後相容入口，預設優先載入 NTU 校區設定。"""
        try:
            campus_cfg = CampusConfig.load(_DEFAULT_CAMPUS_ID)
            return cls.from_campus(campus_cfg)
        except CampusConfigError as e:
            log.warning(f"Campus config unavailable, fallback to legacy defaults: {e}")
            engine = cls()
            engine.initialize()
            return engine
        except (ValueError, KeyError, OSError) as e:
            log.warning(f"Campus config unavailable, fallback to legacy defaults: {e}")
            engine = cls()
            engine.initialize()
            return engine

    @classmethod
    def from_campus(cls, config: CampusConfig) -> "PIVDEngine":
        """依校區設定檔初始化引擎。"""
        dataset = config.get_path("v10_dataset", _DEFAULT_DATASET)
        v9_yaml = config.get_path("v9_yaml", _DEFAULT_V9_YAML)
        boot = config.get_path("v10_ensemble", _DEFAULT_BOOT_PKL)
        metadata = config.get_path("metadata_uid", _DEFAULT_METADATA_UID)

        engine = cls()
        engine.initialize(
            dataset_path=dataset or _DEFAULT_DATASET,
            v9_yaml_path=v9_yaml or _DEFAULT_V9_YAML,
            boot_pkl_path=boot or _DEFAULT_BOOT_PKL,
            metadata_csv=metadata or _DEFAULT_METADATA_UID,
        )
        return engine

    def initialize(
        self,
        dataset_path: Path | str = _DEFAULT_DATASET,
        v9_yaml_path: Path | str = _DEFAULT_V9_YAML,
        boot_pkl_path: Path | str = _DEFAULT_BOOT_PKL,
        metadata_csv: Path | str = _DEFAULT_METADATA_UID,
    ) -> None:
        """載入 / 訓練所有模型 + building metadata。"""
        log.info("Initializing PI-VD Engine...")
        self.surrogate.train(dataset_path)

        # V9: 先試 YAML，沒有則 on-the-fly 訓練
        v9_yaml_path = Path(v9_yaml_path)
        if v9_yaml_path.exists():
            self.v9_weights.load(v9_yaml_path)
        else:
            log.warning(f"V9 YAML not found ({v9_yaml_path}), training from dataset...")
            self.v9_weights.train_from_dataset(dataset_path)

        self.v10_boot.load(boot_pkl_path)
        if not self.v10_boot.is_loaded:
            raise RuntimeError("V10 BOOT models failed to load.")

        # Layer 2.5: Building Metadata Scaler
        self.metadata_scaler.load(metadata_csv)

        self._ready = True
        log.info("PI-VD Engine ready ✓")

    @property
    def is_ready(self) -> bool:
        return self._ready

    def predict(
        self,
        weather_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        全流程推論。

        Parameters
        ----------
        weather_df : DataFrame with DatetimeIndex + ['t_out', 'humidity']

        Returns
        -------
        DataFrame with columns:
            - physics_pred : 物理層預測
            - residual_pred : 殘差層預測 (mean)
            - residual_std : 殘差不確定性 (σ)
            - total_pred : 最終預測 (physics + residual)
            - sim_A ~ sim_E : 各 archetype 模擬值
        """
        if not self._ready:
            raise RuntimeError("Engine not initialized. Call .initialize() first.")

        # Step 0: build calendar features
        cal = build_calendar_features(weather_df.index)
        features = cal.copy()
        features["t_out"] = weather_df["t_out"].values
        features["humidity"] = weather_df["humidity"].values

        # Step 1: Physics Surrogate → sim_A~E
        sim_df = self.surrogate.predict(features)

        # Step 2: V9 Weights → predicted_physics
        physics_pred = self.v9_weights.predict_physics(features, sim_df)

        # Step 3: V10 BOOT → residual
        # 組建 16-feature 輸入
        boot_features = features[_SURROGATE_INPUT_COLS].copy()
        for key in ARCHETYPE_KEYS:
            boot_features[f"sim_{key}"] = sim_df[f"sim_{key}"].values
        X_boot = boot_features[FEATURE_COLS].values
        residual_mean, residual_std = self.v10_boot.predict(X_boot)

        # 組合結果
        result = pd.DataFrame({
            "physics_pred": physics_pred,
            "residual_pred": residual_mean,
            "residual_std": residual_std,
            "total_pred": physics_pred + residual_mean,
        }, index=weather_df.index)

        for key in ARCHETYPE_KEYS:
            result[f"sim_{key}"] = sim_df[f"sim_{key}"].values

        return result

    def predict_building(
        self,
        weather_df: pd.DataFrame,
        uid: str,
    ) -> pd.DataFrame:
        """
        逐棟排序指標：campus-level 預測 × 建物修正因子。

        ⚠️ 注意：回傳值為**相對排序指標 (Ranking Index)**，
        用於「哪棟建物相對用電密度較高」的篩選與排序，
        並非該棟建物的絕對用電量 (kW) 預測。

        Parameters
        ----------
        weather_df : DataFrame with DatetimeIndex + ['t_out', 'humidity']
        uid : 建物 UID（如 'AT1040'）

        Returns
        -------
        DataFrame with additional columns:
            - building_scaler : 建物修正因子
            - building_rank_index : 排序指標 (total_pred × scaler)
            - building_physics_index : 物理層排序指標 (physics_pred × scaler)
            - building_eui_index : 單位面積排序指標 (rank_index / area)，無面積資料時為 NaN
        """
        result = self.predict(weather_df)
        scaler = self.metadata_scaler.get_scaler(uid)
        result["building_scaler"] = scaler
        result["building_rank_index"] = result["total_pred"] * scaler
        result["building_physics_index"] = result["physics_pred"] * scaler

        # EUI index: normalized by area for cross-building comparison
        meta = self.metadata_scaler.get_metadata(uid)
        area = meta.get("area", np.nan) if meta else np.nan
        if np.isfinite(area) and area > 0:
            result["building_eui_index"] = result["building_rank_index"] / area
        else:
            result["building_eui_index"] = np.nan

        return result

    def get_building_metadata(self, uid: str) -> Optional[dict]:
        """取得建物 metadata dict（面積、樓層、用途等）。"""
        return self.metadata_scaler.get_metadata(uid)

    def predict_from_epw(self, epw_path: Path | str) -> pd.DataFrame:
        """
        從 EPW 天氣檔直接執行全流程推論。

        Parameters
        ----------
        epw_path : path to .epw file

        Returns
        -------
        Same as predict()
        """
        from src.epw_reader import read_epw
        weather = read_epw(epw_path)
        return self.predict(weather)


# ═══════════════════════════════════════════════════════════
# 電表資料載入器 (附加功能)
# ═══════════════════════════════════════════════════════════

def load_meter_hourly(
    path: Path | str = _DEFAULT_METER_CSV,
    year: Optional[int] = None,
    campus_config: CampusConfig | None = None,
) -> pd.DataFrame:
    """
    載入 NTU 全校電表 hourly 資料。

    Parameters
    ----------
    path : CSV 路徑
    year : 篩選年份，None = 全部

    Returns
    -------
    DataFrame，index=DatetimeIndex，columns=各電表名稱
    """
    if campus_config is not None:
        path = campus_config.get_path("meter_csv", Path(path)) or path
    path = Path(path)
    log.info(f"Loading meter data from: {path}")
    df = pd.read_csv(path, encoding="utf-8")

    # 找到日期時間欄位 (第一欄)
    dt_col = df.columns[0]
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col])
    df = df.set_index(dt_col).sort_index()

    # 轉數值
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if year is not None:
        df = df[df.index.year == year]

    log.info(f"  Loaded: {df.shape[0]} hours × {df.shape[1]} meters")
    return df


def load_v12_building_summary(
    path: Path | str = _DEFAULT_V12_SUMMARY,
    campus_config: CampusConfig | None = None,
) -> pd.DataFrame:
    """載入 V12 逐棟建物回歸摘要 (含 R², mean_kw 等)。"""
    if campus_config is not None:
        path = campus_config.get_path("v12_summary", Path(path)) or path
    path = Path(path)
    if not path.exists():
        log.warning(f"V12 summary not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== PI-VD Engine Self-Test ===")
    engine = PIVDEngine.from_defaults()

    # Test with 2017 EPW
    epw_path = _DEFAULT_EPW_DIR / "CWBTP_2017.epw"
    if epw_path.exists():
        result = engine.predict_from_epw(epw_path)
        print(f"\n2017 prediction shape: {result.shape}")
        print(f"Physics mean: {result['physics_pred'].mean():.1f} kW")
        print(f"Residual mean: {result['residual_pred'].mean():.1f} kW")
        print(f"Total mean: {result['total_pred'].mean():.1f} kW")
        print(f"Uncertainty (σ) mean: {result['residual_std'].mean():.1f} kW")

        # Compare with cached dataset
        cached = pd.read_csv(_DEFAULT_DATASET, index_col=0, parse_dates=True)
        common_idx = result.index.intersection(cached.index)
        if len(common_idx) > 100:
            cached_phy = cached.loc[common_idx, "predicted_physics"]
            engine_phy = result.loc[common_idx, "physics_pred"]
            corr = np.corrcoef(cached_phy, engine_phy)[0, 1]
            print(f"\nSurrogate validation (physics vs cache):")
            print(f"  Pearson R = {corr:.4f}")
            print(f"  N matching timestamps = {len(common_idx)}")
    else:
        print(f"EPW not found at {epw_path}, skipping prediction test.")

    print("\n=== Done ===")
