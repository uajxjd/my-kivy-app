from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AttentionLSTM(nn.Module):
    """LSTM + Multi-head attention."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        d = hidden_dim * 2
        self.attn = nn.MultiheadAttention(d, num_heads=8, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, 10),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        attn_out, _ = self.attn(out, out, out)
        out = self.norm(out + attn_out)
        pooled = out.mean(dim=1)
        b = self.head(pooled)
        s = self.head(pooled)
        g = self.head(pooled)
        return F.softmax(b, dim=-1), F.softmax(s, dim=-1), F.softmax(g, dim=-1)


class TransformerModel(nn.Module):
    """Pure transformer."""
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 10),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.proj(x)
        x = self.pos_enc(x)
        out = self.encoder(x)
        pooled = out.mean(dim=1)
        b = self.head(pooled)
        s = self.head(pooled)
        g = self.head(pooled)
        return F.softmax(b, dim=-1), F.softmax(s, dim=-1), F.softmax(g, dim=-1)