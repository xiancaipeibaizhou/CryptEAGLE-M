import math
import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial
from typing import Any, Callable

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
        seq_len: int = 10, # 【修复点 1】：接收动态传入的 seq_len
    ):
        super().__init__()
        self.num_heads = num_heads
        self.ln_1 = norm_layer(hidden_dim)
        # 【修复点 2】：把写死的 10 改为动态的 seq_len
        self.self_attention = LinAtten(dim=hidden_dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout, seq_len=seq_len)
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout, act_layer=nn.ReLU)

    def forward(self, input: torch.Tensor):
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
                 num_blk=1,
                 seq_len=10): # 【修复点 3】：让 Adapter 接收 seq_len
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.dropout = nn.Dropout(dropout)
        self.scale = float(adapter_scaler)

        self.lora_A = nn.Linear(in_features=self.n_embd, out_features=self.down_size)
        self.lora_B = nn.Linear(in_features=self.down_size, out_features=self.n_embd)
        
        blks: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_blk):
            blks[f"blk_{i}"] = EncoderBlock(
                num_heads=num_heads, 
                hidden_dim=self.down_size, 
                mlp_dim=mlp_dim, 
                dropout=dropout, 
                attention_dropout=attention_dropout, 
                norm_layer=norm_layer,
                seq_len=seq_len # 【修复点 4】：传递给 EncoderBlock
            )
        
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