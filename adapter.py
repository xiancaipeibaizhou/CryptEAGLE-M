import math
import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial
from typing import Any, Callable

# 注意这里：已经修改为直接从同级目录的 approx_module 导入
from approx_module import LinAtten
from torchvision.ops.misc import MLP

class MLPBlock(MLP):
    """Transformer MLP block."""
    _version = 2
    def __init__(self, in_dim: int, mlp_dim: int, dropout: float, act_layer: nn.Module = nn.GELU):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=act_layer, inplace=None, dropout=dropout)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

class EncoderBlock(nn.Module):
    """Transformer encoder block with LinAtten."""
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.ln_1 = norm_layer(hidden_dim)
        # 序列长度 seq_len 默认给一个小值，因为我们的输入维度已经被池化了
        self.self_attention = LinAtten(dim=hidden_dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout, seq_len=10)
        self.ln_2 = norm_layer(hidden_dim)
        # MPC-friendly MLP
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout, act_layer=nn.ReLU)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = x + input
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

class CryptPEFT_adapter(nn.Module):
    def __init__(self,
                 num_heads: int,
                 attention_dropout: float,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 adapter_scaler="1.0",
                 mlp_dim=None,
                 num_blk=1,):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.dropout = nn.Dropout(dropout)
        self.scale = float(adapter_scaler)

        self.lora_A = nn.Linear(in_features=self.n_embd, out_features=self.down_size)
        self.lora_B = nn.Linear(in_features=self.down_size, out_features=self.n_embd)
        
        blks: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_blk):
            blks[f"blk_{i}"] = EncoderBlock(num_heads=num_heads, hidden_dim=self.down_size, mlp_dim=mlp_dim, dropout=dropout, attention_dropout=attention_dropout, norm_layer=norm_layer)
        
        self.blks = nn.Sequential(blks) 

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            nn.init.zeros_(self.lora_A.bias)
            nn.init.zeros_(self.lora_B.bias)

    def forward(self, x):
        x = self.lora_A(x)
        x = self.dropout(x)
        x = self.blks(x)
        x = self.lora_B(x)
        x = x * self.scale
        return x