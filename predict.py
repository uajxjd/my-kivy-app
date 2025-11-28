#!/usr/bin/env python
"""
高准确率预测脚本
支持：
  - EMA 权重
  - 多模型投票（Transformer + LSTM）
  - 温度缩放校准概率
用法：
    python predict.py --csv UAFC3D.csv --vote
"""
from __future__ import annotations

import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

from fc3d.data import FC3DDataset, load_csv
from fc3d.models import TransformerModel, AttentionLSTM

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 60
TOP_K = 6
TEMP = 0.8  # 温度缩放，越小越尖锐


def load_model(model_type: str, input_dim: int, use_ema: bool):
    ckpt = Path(f"{model_type}_ema.pt" if use_ema else f"{model_type}.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"缺失 {ckpt}，请先训练并生成 EMA 权重")
    if model_type == "transformer":
        model = TransformerModel(input_dim).to(DEVICE)
    else:
        model = AttentionLSTM(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def predict_once(model, dataset: FC3DDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, _, _, _ = dataset[0]
    x = x.unsqueeze(0).to(DEVICE)
    b_prob, s_prob, g_prob = model(x)
    # 温度缩放
    b_prob = F.softmax(b_prob[0] / TEMP, dim=-1).cpu().numpy()
    s_prob = F.softmax(s_prob[0] / TEMP, dim=-1).cpu().numpy()
    g_prob = F.softmax(g_prob[0] / TEMP, dim=-1).cpu().numpy()
    b_top = np.argsort(b_prob)[-TOP_K:][::-1]
    s_top = np.argsort(s_prob)[-TOP_K:][::-1]
    g_top = np.argsort(g_prob)[-TOP_K:][::-1]
    return b_top, s_top, g_top


def vote(models: list, dataset: FC3DDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """概率平均投票"""
    b_agg = np.zeros(10)
    s_agg = np.zeros(10)
    g_agg = np.zeros(10)
    for model in models:
        b, s, g = model(dataset)
        b_agg += b
        s_agg += s
        g_agg += g
    b_agg /= len(models)
    s_agg /= len(models)
    g_agg /= len(models)
    return (
        np.argsort(b_agg)[-TOP_K:][::-1],
        np.argsort(s_agg)[-TOP_K:][::-1],
        np.argsort(g_agg)[-TOP_K:][::-1],
    )


def main():
    parser = argparse.ArgumentParser(description="高准确率预测下一期")
    parser.add_argument("--csv", type=Path, default="UAFC3D.csv")
    parser.add_argument("--model", choices=["transformer", "attention_lstm"], help="单模型")
    parser.add_argument("--vote", action="store_true", help="启用多模型投票")
    parser.add_argument("--temp", type=float, default=TEMP, help="温度缩放")
    args = parser.parse_args()

    df = load_csv(args.csv)
    latest_df = df.tail(SEQ_LEN)
    dataset = FC3DDataset(latest_df, SEQ_LEN, is_pred=True)
    input_dim = len(dataset.feature_cols)

    if args.vote:
        models = [
            load_model("transformer", input_dim, use_ema=True),
            load_model("attention_lstm", input_dim, use_ema=True),
        ]
        b_top, s_top, g_top = vote(models, dataset)
    else:
        if not args.model:
            parser.error("单模型时必须指定 --model")
        model = load_model(args.model, input_dim, use_ema=True)
        b_top, s_top, g_top = predict_once(model, dataset)

    print("\n🎯 下一期推荐号码（TOP-6）")
    print("百位：", b_top)
    print("十位：", s_top)
    print("个位：", g_top)
    print("🌟 最优组合：", f"{b_top[0]}{s_top[0]}{g_top[0]}")


if __name__ == "__main__":
    main()