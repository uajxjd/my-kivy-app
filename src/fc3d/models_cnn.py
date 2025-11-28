import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    """1D-CNN 提取局部模式"""
    def __init__(self, input_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 10),
        )

    def forward(self, x):
        # x: [B, L, D] -> [B, D, L]
        x = x.transpose(1, 2)
        out = self.conv(x).squeeze(-1)  # [B, hidden]
        b = self.head(out)
        s = self.head(out)
        g = self.head(out)
        return F.softmax(b, dim=-1), F.softmax(s, dim=-1), F.softmax(g, dim=-1)