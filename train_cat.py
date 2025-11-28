#!/usr/bin/env python
# train_cat.py  (根目录)
import catboost as cb
from pathlib import Path
from fc3d.data import load_csv, build_features
import numpy as np

# 1. 数据划分
df = load_csv("UAFC3D.csv")
split = len(df) - 60
train_df, val_df = df.iloc[:split], df.iloc[split:]

# 2. 训练集：fit scaler
train_feat, cols, scaler = build_features(train_df)
X_train = train_feat[cols].values

# 3. 验证集：复用 scaler
val_feat = build_features(val_df)[0]   # 只拿 DataFrame
X_val = scaler.transform(val_feat[cols])

# 4. 列名映射
COL_MAP = {"bai": "百位", "shi": "十位", "ge": "个位"}

for name in ("bai", "shi", "ge"):
    y_train = train_df[COL_MAP[name]].values
    y_val   = val_df[COL_MAP[name]].values

    model = cb.CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        loss_function='MultiClass',
        classes_count=10,
        verbose=0,
        random_seed=42,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

    Path("models").mkdir(exist_ok=True)
    model.save_model(f"models/cat_{name}.cbm")
    print(f"✅ CatBoost {name} 保存完成")