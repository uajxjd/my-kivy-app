# src/fc3d/models_cat.py
import catboost as cbt
from pathlib import Path

class CatModel:
    def __init__(self, model_dir: Path):
        self.models = {
            "bai": cbt.CatBoostClassifier().load_model(model_dir / "cat_bai.cbm"),
            "shi": cbt.CatBoostClassifier().load_model(model_dir / "cat_shi.cbm"),
            "ge":  cbt.CatBoostClassifier().load_model(model_dir / "cat_ge.cbm"),
        }

    def predict_proba(self, feat) -> tuple:
        return (
            self.models["bai"].predict_proba(feat)[:, 1],
            self.models["shi"].predict_proba(feat)[:, 1],
            self.models["ge"].predict_proba(feat)[:, 1],
        )