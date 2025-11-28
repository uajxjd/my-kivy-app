#!/usr/bin/env python
# train_cnn.py  (与 train_lgb.py 同级)
from pathlib import Path
from fc3d.data import load_csv, FC3DDataset
from fc3d.models_cnn import CNN1D
from fc3d.train import train_model  # 直接复用通用训练器

df = load_csv("UAFC3D.csv")
model_type = "cnn"
best = train_model(
    csv_path="UAFC3D.csv",
    model_type=model_type,
    seq_len=60,
    batch_size=64,
    epochs=200,
    lr=2e-4,
)
print(f"CNN 训练完成，最佳验证准确率：{best['best_val_acc']:.4f}")