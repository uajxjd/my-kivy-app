import numpy as np
import lightgbm as lgb
from sklearn.multiclass import OneVsRestClassifier
from pathlib import Path

class LGBModel:
    """LightGBM 三位置多分类"""
    def __init__(self, model_path: Path):
        self.models = {
            "bai": lgb.Booster(model_file=str(model_path / "lgb_bai.txt")),
            "shi": lgb.Booster(model_file=str(model_path / "lgb_shi.txt")),
            "ge":  lgb.Booster(model_file=str(model_path / "lgb_ge.txt")),
        }

    def predict_proba(self, feat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.models["bai"].predict(feat),
            self.models["shi"].predict(feat),
            self.models["ge"].predict(feat),
        )