# src/fc3d/ema.py
import torch
from copy import deepcopy

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(model.state_dict())
        for k, v in self.shadow.items():
            v.requires_grad = False

    def update(self, model: torch.nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].mul_(self.decay).add_(param, alpha=1 - self.decay)

    def apply_shadow(self, model: torch.nn.Module):
        model.load_state_dict(self.shadow, strict=False)