# app.py  (根目录)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import numpy as np

from fc3d.data import load_csv, FC3DDataset
from fc3d.ensemble.voter import SuperVoter

app = FastAPI(title="FC3D SuperVoter API", version="3.0.0")

# 全局加载一次
DATA_PATH = Path("UAFC3D.csv")
df = load_csv(DATA_PATH)
dataset = FC3DDataset(df.tail(60), 60, is_pred=True)
voter = SuperVoter(len(dataset.feature_cols), Path("models"))


class PredictOut(BaseModel):
    bai: list[int]
    shi: list[int]
    ge: list[int]
    best: str


@app.get("/predict", response_model=PredictOut)
def predict():
    b, s, g = voter.predict(dataset[0][0].unsqueeze(0))
    best = f"{b[0]}{s[0]}{g[0]}"
    return PredictOut(bai=b.tolist(), shi=s.tolist(), ge=g.tolist(), best=best)


@app.get("/health")
def health():
    return {"status": "ok", "model": "6-model-super-voter"}