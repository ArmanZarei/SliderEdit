import math
import torch.nn as nn
import torch
import os
import torch.nn.functional as F


class SelectiveLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int, alpha: int, dropout: float = 0.0, apply_zero_padding=False):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)

        self.base = base_linear

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.alpha = alpha
        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # non-persistent buffer to hold the mask set via a context manager
        self.register_buffer("_token_mask", None, persistent=False)
    
        self.apply_zero_padding = apply_zero_padding

    def set_scaling(self, scaling):
        self.scaling = scaling
    
    def reset_scaling(self):
        self.scaling = self.alpha / self.r

    def set_token_mask(self, mask): # mask: (B, S) bool
        self._token_mask = mask
    
    def pad_tensor(self, x, target_tens):
        diff = target_tens.size(1) - x.size(1)
        if diff > 0:
            pad = [0, 0] * (x.dim() - 2) + [0, diff]
            x = F.pad(x, pad, value=0)
        return x

    def forward(self, x):
        if self._token_mask is None or x.shape[1] == 209 or x.shape[1] == 1377: # TODO: Hack for Qwen -> [make it clean]
            return self.base(x)

        y = self.base(x)

        y_prev_dtype = y.dtype
        if y.dtype != self.lora_A.weight.dtype:
            x = x.to(self.lora_A.weight.dtype)
            y = y.to(self.lora_A.weight.dtype)
        
        scaling = self.scaling
        if torch.is_tensor(self.scaling) and self.apply_zero_padding:
            scaling = self.pad_tensor(self.scaling, x)

        lora = self.lora_B(self.lora_A(self.dropout(x))) * scaling

        # x is (B, S, D); mask is (B, S)
        m = self._token_mask.to(lora.dtype).unsqueeze(-1)
        if self.apply_zero_padding:
            m = self.pad_tensor(m, lora)
        lora = lora * m

        return (y + lora).to(y_prev_dtype)

