#!/usr/bin/env python
# train_large.py  (根目录)
from pathlib import Path
from fc3d.data import load_csv, FC3DDataset
from fc3d.train import train_model  # 复用通用训练器
from fc3d.models import TransformerModel  # 支持传参变大

df = load_csv("UAFC3D.csv")

# 大模型参数
large_config = {
    "input_dim": len(FC3DDataset(df, 60).feature_cols),
    "d_model": 512,
    "num_layers": 8,
    "num_heads": 16,
    "dropout": 0.1,
}

# 训练大模型
best = train_model(
    csv_path="UAFC3D.csv",
    model_type="transformer",  # 复用通用训练器
    seq_len=60,
    batch_size=32,          # 显存友好
    epochs=300,             # 更长训练
    lr=1e-4,                # 更小学习率
)

print(f"Transformer-Large 训练完成，最佳验证准确率：{best['best_val_acc']:.4f}")