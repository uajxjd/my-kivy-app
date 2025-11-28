from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

from fc3d.data import load_csv
from fc3d.train import train_model

# 可选模型列表（与 train_model 保持一致）
MODEL_CHOICES = ["attention_lstm", "transformer", "transformer_large", "cnn"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fc3d", description="FC3D industrial predictor")
    parser.add_argument("csv", type=Path, help="Historical csv (e.g. UAFC3D.csv)")
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="transformer",
        help="Model architecture",
    )
    parser.add_argument("--seq-len", type=int, default=60, help="Sequence length")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Max epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")

    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(f"❌ CSV not found: {args.csv}")
        raise SystemExit(1)

    # 一键训练（含 EMA、早停、日志）
    best = train_model(
        csv_path=args.csv,
        model_type=args.model,
        seq_len=args.seq_len,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
    )

    print(f"\n✅ 训练完成！最佳验证准确率：{best['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()