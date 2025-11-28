from __future__ import annotations
import numpy as np
import torch
from pathlib import Path
from fc3d.models import TransformerModel, AttentionLSTM
from fc3d.models_cnn import CNN1D
from fc3d.models_lgb import LGBModel
from fc3d.models_cat import CatModel  # 如有则导入

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""5 模型超级集成投票器"""


class SuperVoter:
    def __init__(self, input_dim: int, model_dir: Path):
        root = Path(".")
        self.tfm = TransformerModel(input_dim).to(DEVICE)
        self.tfm.load_state_dict(torch.load(root / "transformer_ema.pt", map_location=DEVICE, weights_only=True))
        self.lstm = AttentionLSTM(input_dim).to(DEVICE)
        self.lstm.load_state_dict(torch.load(root / "attention_lstm_ema.pt", map_location=DEVICE, weights_only=True))
        self.cnn = CNN1D(input_dim).to(DEVICE)
        self.cnn.load_state_dict(torch.load(root / "cnn_ema.pt", map_location=DEVICE, weights_only=True))
        self.large = TransformerModel(input_dim=input_dim, d_model=512, num_layers=8, num_heads=16, dropout=0.1).to(
            DEVICE)
        self.large.load_state_dict(
            torch.load(root / "transformer_large_ema.pt", map_location=DEVICE, weights_only=True))

        self.lgb = LGBModel(model_dir)
        self.cat = CatModel(model_dir)

        self.models = [self.tfm, self.lstm, self.cnn, self.large]
        self.weights = np.array([0.25, 0.23, 0.12, 0.30])
        self.tree_weight = 0.35
        self.cat_ratio = 0.5

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        b_agg = np.zeros(10)
        s_agg = np.zeros(10)
        g_agg = np.zeros(10)

        # DL 投票
        for model, w in zip(self.models, self.weights):
            model.eval()
            b, s, g = model(x)
            b_agg += w * b[0].cpu().numpy()
            s_agg += w * s[0].cpu().numpy()
            g_agg += w * g[0].cpu().numpy()

        # 树模型补充
        feat_np = x[:, -1, :].cpu().numpy()
        b_tree, s_tree, g_tree = self.lgb.predict_proba(feat_np)
        b_cat, s_cat, g_cat = self.cat.predict_proba(feat_np)
        b_agg += self.tree_weight * (0.5 * np.squeeze(b_tree) + 0.5 * np.squeeze(b_cat))
        s_agg += self.tree_weight * (0.5 * np.squeeze(s_tree) + 0.5 * np.squeeze(s_cat))
        g_agg += self.tree_weight * (0.5 * np.squeeze(g_tree) + 0.5 * np.squeeze(g_cat))

        # 归一化
        b_agg /= b_agg.sum()
        s_agg /= s_agg.sum()
        g_agg /= g_agg.sum()

        top = lambda p: np.argsort(p)[-6:][::-1]
        return top(b_agg), top(s_agg), top(g_agg)
