from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Literal, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fc3d.data import FC3DDataset, load_csv
from fc3d.models import AttentionLSTM, TransformerModel
from fc3d.models_cnn import CNN1D
from fc3d.ema import EMA  # EMA 工具类

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 默认超参
BATCH = 64
SEQ = 60
EPOCHS = 200
PATIENCE = 20
LR = 2e-4
LABEL_SMOOTH = 0.1


class EarlyStopper:
    def __init__(self, patience: int = PATIENCE):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for x, bai, shi, ge in tqdm(loader, leave=False):
        x, bai, shi, ge = x.to(DEVICE), bai.to(DEVICE), shi.to(DEVICE), ge.to(DEVICE)
        optimizer.zero_grad()
        pb, ps, pg = model(x)
        loss = (criterion(pb, bai) + criterion(ps, shi) + criterion(pg, ge)) / 3
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        running_acc += (
            ((pb.argmax(1) == bai) & (ps.argmax(1) == shi) & (pg.argmax(1) == ge))
            .float()
            .sum()
            .item()
        )
        n += x.size(0)
    return running_loss / n, running_acc / n


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for x, bai, shi, ge in loader:
        x, bai, shi, ge = x.to(DEVICE), bai.to(DEVICE), shi.to(DEVICE), ge.to(DEVICE)
        pb, ps, pg = model(x)
        loss = (criterion(pb, bai) + criterion(ps, shi) + criterion(pg, ge)) / 3
        running_loss += loss.item() * x.size(0)
        running_acc += (
            ((pb.argmax(1) == bai) & (ps.argmax(1) == shi) & (pg.argmax(1) == ge))
            .float()
            .sum()
            .item()
        )
        n += x.size(0)
    return running_loss / n, running_acc / n


def train_model(
    csv_path: str | Path,
    model_type: Literal["attention_lstm", "transformer", "cnn"],
    seq_len: int = SEQ,
    batch_size: int = BATCH,
    epochs: int = EPOCHS,
    lr: float = LR,
) -> Dict[str, Any]:
    csv_path = Path(csv_path).expanduser().resolve()
    df = load_csv(csv_path)
    split = int(len(df) * 0.85)
    train_ds = FC3DDataset(df.iloc[:split], seq_len)
    val_ds = FC3DDataset(
        df.iloc[split:], seq_len, feature_cols=train_ds.feature_cols, scaler=train_ds.scaler
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = len(train_ds.feature_cols)

    # ===== 模型工厂 =====
    if model_type == "attention_lstm":
        model = AttentionLSTM(input_dim).to(DEVICE)
    elif model_type == "transformer":
        model = TransformerModel(input_dim).to(DEVICE)
    elif model_type == "cnn":
        model = CNN1D(input_dim).to(DEVICE)
    elif model_type == "transformer_large":   # ← 新增大模型分支
        model = TransformerModel(
            input_dim=input_dim,
            d_model=512,
            num_layers=8,
            num_heads=16,
            dropout=0.1,
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    stopper = EarlyStopper()
    ema = EMA(model)  # ← EMA 初始化

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)
        ema.update(model)  # ← 每 epoch 更新 EMA

        logger.info(
            "Epoch %02d | train loss %.4f acc %.4f | val loss %.4f acc %.4f",
            epoch, train_loss, train_acc, val_loss, val_acc,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            # 保存普通权重
            torch.save(model.state_dict(), f"{model_type}.pt")
            # 保存 EMA 权重
            ema.apply_shadow(model)
            torch.save(model.state_dict(), f"{model_type}_ema.pt")

        if stopper(val_loss):
            logger.info("Early stop at epoch %d", epoch)
            break

    logger.info("Training finished. Best val acc=%.4f", best_acc)
    return {"best_val_acc": best_acc}