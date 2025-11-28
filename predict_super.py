#!/usr/bin/env python
# predict_super.py  (根目录)
from pathlib import Path
from fc3d.data import load_csv, FC3DDataset
from fc3d.ensemble.voter import SuperVoter  # 待会创建

df = load_csv("UAFC3D.csv")
latest = df.tail(60)
dataset = FC3DDataset(latest, 60, is_pred=True)
input_dim = len(dataset.feature_cols)

voter = SuperVoter(input_dim, Path("models"))
b, s, g = voter.predict(dataset[0][0].unsqueeze(0))

print("🎯 超级集成 TOP-6")
print("百位：", b)
print("十位：", s)
print("个位：", g)
print("🌟 最优组合：", f"{b[0]}{s[0]}{g[0]}")