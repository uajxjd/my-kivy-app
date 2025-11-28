#!/usr/bin/env python
# tune_weights.py
import numpy as np
import optuna
from pathlib import Path
from fc3d.data import load_csv, FC3DDataset
from fc3d.ensemble.voter import SuperVoter  # 复用 voter 结构

# 1. 加载验证集（最后 60 期）
df = load_csv("UAFC3D.csv")
val_ds = FC3DDataset(df.tail(60), 60, is_pred=True)
x = val_ds[0][0].unsqueeze(0)  # 1 条样本

# 2. Optuna 目标函数
def objective(trial):
    # 搜索 4 个权重（DL 3 + 树 1），和 = 1
    w_tfm = trial.suggest_float("w_tfm", 0.05, 0.6)
    w_lstm = trial.suggest_float("w_lstm", 0.05, 0.6)
    w_cnn = trial.suggest_float("w_cnn", 0.05, 0.6)
    w_tree = trial.suggest_float("w_tree", 0.05, 0.6)
    # 归一化
    total = w_tfm + w_lstm + w_cnn + w_tree
    w = np.array([w_tfm, w_lstm, w_cnn]) / total
    w_tree_norm = w_tree / total

    # 3. 临时改权重
    voter = SuperVoter(len(val_ds.feature_cols), Path("models"))
    voter.weights = w
    b, s, g = voter.predict(x)
    # 4. 用“最优单条”当 reward（可改成真实开奖）
    reward = float(b[0] == 1) + float(s[0] == 7) + float(g[0] == 5)  # 示例：175
    return reward

# 5. 运行搜索
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200, timeout=1800)  # 30 min 上限

# 6. 输出最优权重
best_w = study.best_params
total = sum(best_w.values())
print("🎯 最优权重（归一化）:")
print(f"Transformer: {best_w['w_tfm']/total:.3f}")
print(f"LSTM:        {best_w['w_lstm']/total:.3f}")
print(f"CNN:         {best_w['w_cnn']/total:.3f}")
print(f"Tree:        {best_w['w_tree']/total:.3f}")