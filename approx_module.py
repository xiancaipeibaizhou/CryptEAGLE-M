import torch
import torch.nn as nn
import torch.nn.functional as F

class LinAtten(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0., proj_drop: float = 0., norm_layer: nn.Module = nn.LayerNorm, seq_len: int = 10) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # 🌟【修复】：真正的 Linear Attention 实现 (ELU + 1 防止负数，改变矩阵乘法顺序 Q * (K^T * V))
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        
        # 计算 K^T * V (复杂度 O(N * d^2))
        kv = torch.einsum('b h n d, b h n e -> b h d e', k, v)
        # 归一化项
        z = 1.0 / (torch.einsum('b h n d, b h d -> b h n', q, k.sum(dim=2)) + 1e-6)
        
        # 计算 Q * (K^T * V)
        out = torch.einsum('b h n d, b h d e -> b h n e', q, kv)
        out = out * z.unsqueeze(-1)
        
        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x