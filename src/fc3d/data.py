from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset
import torch

logger = logging.getLogger(__name__)

REQUIRED_COLS = ["期号", "百位", "十位", "个位", "和值", "跨度", "重复数字"]
FEATURE_COLS_EXCLUDE = ["期号", "百位", "十位", "个位", "重复数字"]


class DataValidationError(RuntimeError):
    """Raised when input data is malformed."""


def validate_dataframe(df: pd.DataFrame) -> None:
    """Strict validation."""
    if df.empty:
        raise DataValidationError("Empty dataframe")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    for col in ["百位", "十位", "个位"]:
        if not pd.api.types.is_integer_dtype(df[col]):
            raise DataValidationError(f"{col} must be integer")
        if not df[col].between(0, 9).all():
            raise DataValidationError(f"{col} out of range 0-9")


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], RobustScaler]:
    """Engineer features and return (features_df, feature_names, scaler)."""
    df = df.sort_values("期号").reset_index(drop=True)
    out = df.copy()
    out["period"] = out.index

    # ---- technicals ---- #
    for w in [3, 5, 10, 15, 20]:
        for col in ["百位", "十位", "个位"]:
            out[f"{col}_ma_{w}"] = out[col].rolling(w, min_periods=1).mean()
            out[f"{col}_std_{w}"] = out[col].rolling(w, min_periods=1).std()
        out[f"和值_ma_{w}"] = out["和值"].rolling(w, min_periods=1).mean()
        out[f"跨度_ma_{w}"] = out["跨度"].rolling(w, min_periods=1).mean()

    # ---- lags ---- #
    for lag in [1, 2, 3, 5, 7, 10, 15]:
        for col in ["百位", "十位", "个位"]:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)

    # ---- calendar ---- #
    out["day_of_week"] = out.index % 7
    out["period_in_month"] = out.index % 30

    # ---- combos ---- #
    out["百十组合"] = out["百位"] * 10 + out["十位"]
    out["十个组合"] = out["十位"] * 10 + out["个位"]
    out["百个组合"] = out["百位"] * 10 + out["个位"]

    # ---- one-hot repeat ---- #
    for val in [0, 1, 2]:
        out[f"重复数_{val}"] = (out["重复数字"] == val).astype(int)

    # ---- fillna ---- #
    out = out.ffill().bfill().fillna(0)

    # ---- feature list ---- #
    feature_cols = [c for c in out.columns if c not in FEATURE_COLS_EXCLUDE]
    scaler = RobustScaler()
    scaled = scaler.fit_transform(out[feature_cols])
    out[feature_cols] = scaled
    return out, feature_cols, scaler


class FC3DDataset(Dataset):
    """Torch dataset."""
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 60,
        feature_cols: list[str] | None = None,
        scaler: RobustScaler | None = None,
        is_pred: bool = False,
    ):
        validate_dataframe(df)
        self.df, self.feature_cols, self.scaler = build_features(df)
        if feature_cols:
            self.feature_cols = feature_cols
        if scaler:
            self.scaler = scaler
        self.seq_len = seq_len
        self.is_pred = is_pred

    def __len__(self) -> int:
        if self.is_pred:
            return 1 if len(self.df) >= self.seq_len else 0
        return max(0, len(self.df) - self.seq_len)

    def __getitem__(self, idx: int):
        if self.is_pred:
            start = len(self.df) - self.seq_len
            seq = self.df.iloc[start : start + self.seq_len]
        else:
            seq = self.df.iloc[idx : idx + self.seq_len]
        x = seq[self.feature_cols].to_numpy(dtype=np.float32)
        if self.is_pred:
            return torch.tensor(x), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])
        target_row = self.df.iloc[idx + self.seq_len]
        return (
            torch.tensor(x),
            torch.tensor(target_row["百位"], dtype=torch.long),
            torch.tensor(target_row["十位"], dtype=torch.long),
            torch.tensor(target_row["个位"], dtype=torch.long),
        )


def load_csv(path: Path | str) -> pd.DataFrame:
    """Safe csv loader."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    validate_dataframe(df)
    logger.info("Data loaded and validated: %s rows from %s", len(df), path)
    return df