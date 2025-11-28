#!/usr/bin/env python
# train_lgb.py  (根目录)
import lightgbm as lgb
from pathlib import Path
from fc3d.data import load_csv, build_features
import numpy as np

# 中文列名映射
COL_MAP = {"bai": "百位", "shi": "十位", "ge": "个位"}

df = load_csv("UAFC3D.csv")
split = len(df) - 60
train_df, val_df = df.iloc[:split], df.iloc[split:]

# 1. 训练集：fit scaler
train_feat, cols, scaler = build_features(train_df)
X_train = train_feat[cols].values

# 2. 验证集：复用 scaler
val_feat, _, _ = build_features(val_df)
X_val = scaler.transform(val_feat[cols])

for name in ("bai", "shi", "ge"):
    y_train = train_df[COL_MAP[name]].values
    y_val   = val_df[COL_MAP[name]].values
    dtrain = lgb.Dataset(X_train, y_train)
    dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)
    params = {"objective": "multiclass", "num_class": 10, "metric": "multi_logloss", "verbose": -1}
    model = lgb.train(params, dtrain, valid_sets=[dvalid], num_boost_round=800)
    Path("models").mkdir(exist_ok=True)
    model.save_model(f"models/lgb_{name}.txt")
    print(f"✅ LightGBM {name} 保存完成")