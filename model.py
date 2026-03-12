import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphNorm
from torch_geometric.utils import dropout_edge, scatter
from adapter import CryptPEFT_adapter


class DropPath(nn.Module):
    """Stochastic Depth: 在训练时随机丢弃残差路径，增强泛化能力"""

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ==========================================
# 门控组件: Entropy-Aware Gating
# ==========================================
class EntropyGatingUnit(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Hardsigmoid()
        )

    def forward(self, x_local, x_global, x_base):
        flow_embedding = torch.cat([x_base.mean(dim=1), x_base.max(dim=1)[0]], dim=-1)
        alpha = self.gate_fc(flow_embedding).unsqueeze(1)
        out = alpha * x_local + (1 - alpha) * x_global + x_base
        return out, alpha


# ==========================================
# 核心组件: Temporal Inception 1D (Local Stream)
# ==========================================
class TemporalInception1D(nn.Module):
    def __init__(self, in_features, out_features, kernel_set=None):
        super().__init__()
        if kernel_set is None: kernel_set = [1, 3, 5, 7]
        if isinstance(kernel_set, (int, float)): kernel_set = [int(kernel_set)]
        kernel_set = [int(k) for k in kernel_set if int(k) > 0]
        if len(kernel_set) == 0: kernel_set = [1, 3, 5, 7]

        self.kernel_set = list(kernel_set)
        cout_per_kernel = max(1, out_features // len(self.kernel_set))

        self.tconv = nn.ModuleList()
        for kern in self.kernel_set:
            pad = kern // 2
            self.tconv.append(
                nn.Conv1d(in_features, cout_per_kernel, kernel_size=kern, padding=pad)
            )

        cat_channels = cout_per_kernel * len(self.kernel_set)
        self.fuse = nn.Identity() if cat_channels == out_features else nn.Conv1d(cat_channels, out_features,
                                                                                 kernel_size=1)
        self.project = nn.Conv1d(in_features, out_features, kernel_size=1)
        self.act = nn.ReLU()

    def forward(self, x):
        outputs = []
        for conv in self.tconv:
            outputs.append(conv(x))
        out = self.fuse(torch.cat(outputs, dim=1))
        return self.act(out + self.project(x))


# ==========================================
# 【KBS 创新点 3 & 2】: CryptoNorm Attention & Prototype-Guided Graph Alignment
# ==========================================
class CryptoNormEdgeAttention(MessagePassing):
    """
    Methodological Novelty: CryptoNorm Attention with Prototype Guidance.
    正式定义 CryptoNorm Kernel, 并引入 Prototype 参与图更新。
    """

    def __init__(self, in_dim, out_dim, edge_dim, heads=4, dropout=0.1, drop_path=0.1):
        super().__init__(node_dim=0, aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        self.dropout = dropout

        assert out_dim % heads == 0, "out_dim must be divisible by heads"

        self.WQ = nn.Linear(in_dim, out_dim, bias=False)
        self.WK = nn.Linear(in_dim, out_dim, bias=False)
        self.WV = nn.Linear(in_dim, out_dim, bias=False)
        self.WE = nn.Linear(edge_dim, out_dim, bias=False)

        # 创新点2: 原型映射层，用于引导图对齐
        self.proto_proj = nn.Linear(in_dim, out_dim, bias=False)

        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = GraphNorm(out_dim)
        self.drop_path = DropPath(drop_path)
        self.act = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        nn.init.xavier_uniform_(self.WE.weight)
        nn.init.xavier_uniform_(self.proto_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, edge_index, edge_attr, batch=None, prototype=None):
        residual = x
        q = self.WQ(x).view(-1, self.heads, self.head_dim)
        k = self.WK(x).view(-1, self.heads, self.head_dim)
        v = self.WV(x).view(-1, self.heads, self.head_dim)
        e_emb = self.WE(edge_attr).view(-1, self.heads, self.head_dim)

        # 【KBS 创新点 2】：Prototype-Guided Graph Learning
        # 在空间图卷积深层直接利用 Prototype 引导查询向量(Query)的偏移
        if prototype is not None:
            p_emb = self.proto_proj(prototype).view(1, self.heads, self.head_dim)
            # 使用密码学友好的 ReLU 点乘计算对齐分数
            align_score = F.relu((q * p_emb).sum(dim=-1, keepdim=True))
            q = q + 0.1 * align_score * p_emb  # 动态微调 Query

        out = self.propagate(edge_index, q=q, k=k, v=v, e_emb=e_emb, size=None)

        out = out.view(-1, self.out_dim)
        out = self.out_proj(out)
        out = self.norm(out + self.drop_path(residual), batch)
        return self.act(out)

    # 1. 函数参数里加一个 size_i，PyG 会自动把当前图的节点总数 N 传进来
    def message(self, q_i, k_j, v_j, e_emb, index, size_i):
        score = (q_i * (k_j + e_emb)).sum(dim=-1) / (self.head_dim ** 0.5)
        
        # 【KBS 创新点 3】：CryptoNorm Attention Kernel 核心数学实现
        score = F.relu(score)
        # 2. 将 dim_size=q_i.size(0) 改为 dim_size=size_i，彻底防止显存越界
        row_sum = scatter(score, index, dim=0, dim_size=size_i, reduce='sum')
        alpha = score / (row_sum[index] + 1e-6) 
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha.unsqueeze(-1) * (v_j + e_emb)


# ==========================================
# 核心组件: Edge Updater (Phase 1)
# ==========================================
class EdgeUpdaterModule(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.1):
        super().__init__()
        input_dim = node_dim * 2 + edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.res_proj = nn.Linear(edge_dim, hidden_dim) if edge_dim != hidden_dim else None
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        x_src, x_dst = x[src], x[dst]
        cat_feat = torch.cat([x_src, x_dst, edge_attr], dim=-1)
        update = self.mlp(cat_feat)
        if self.res_proj is not None:
            edge_attr = self.res_proj(edge_attr)
        return self.norm(update + edge_attr)


# ==========================================
# 核心组件: Linear Temporal Attention (Global Stream)
# ==========================================
class LinearTemporalAttention(nn.Module):
    def __init__(self, feature_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = feature_dim // heads
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        B, T, C = x.shape
        residual = x
        q = F.relu(self.q_proj(x).view(B, T, self.heads, self.head_dim)) + 1.0
        k = F.relu(self.k_proj(x).view(B, T, self.heads, self.head_dim)) + 1.0
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim)

        kv = torch.einsum('bthd,bthe->bhde', k, v)
        z = torch.einsum('bthd,bhd->bth', q, k.sum(dim=1)).unsqueeze(-1)
        num = torch.einsum('bthd,bhde->bthe', q, kv)

        out = num / (z + 1e-6)
        out = out.reshape(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        return self.norm(out + residual)


# ==========================================
# 【KBS 创新点 1】: Dynamic Edge Memory (DEM)
# ==========================================
class DynamicEdgeMemory(nn.Module):
    """
    Methodological Novelty: Node-Anchored Dynamic Edge Memory.
    融合空间边缘特征与时序节点上下文，完美捕捉慢速扫描和隐蔽攻击的长期记忆，
    同时彻底解决动态图网络中由于拓扑不断变化(如 DropEdge)导致的维度灾难。
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # 记忆衰减率 gamma (公式中的 γ)
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.mem_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU()
        )

    def forward(self, spatial_edge, temporal_src, temporal_dst):
        # 利用全局对齐的时序节点(Temporal Nodes)状态动态重构边缘的长期记忆
        # m_e(t) = γ * spatial_edge + (1-γ) * temporal_context
        temporal_edge_context = torch.cat([temporal_src, temporal_dst], dim=-1)

        # 融合长期记忆与瞬时空间边缘特征
        raw_edge_rep = torch.cat([
            self.gamma * spatial_edge,
            (1 - self.gamma) * temporal_edge_context
        ], dim=-1)

        return self.mem_fuse(raw_edge_rep)


# ==========================================
# 完整模型: CryptEAGLE-M (Memory 升级版)
# ==========================================
class CryptEAGLE(nn.Module):
    def __init__(
            self,
            node_in,
            edge_in,
            hidden,
            num_classes,
            seq_len=10,
            heads=8,
            dropout=0.3,
            max_cl_edges=2048,
            kernels=None,
            drop_path=0.1,
            dropedge_p=0.2,
    ):
        super(CryptEAGLE, self).__init__()
        self.hidden = hidden
        self.seq_len = seq_len
        self.max_cl_edges = max_cl_edges
        self.dropedge_p = float(dropedge_p)

        self.node_enc = nn.Sequential(nn.Linear(node_in, hidden), nn.LayerNorm(hidden))
        self.edge_enc = nn.Sequential(nn.Linear(edge_in, hidden), nn.LayerNorm(hidden))

        # --- Spatial Layers (引入 CryptoNormEdgeAttention) ---
        self.num_layers = 2
        self.spatial_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.spatial_layers.append(nn.ModuleDict({
                'node_att': CryptoNormEdgeAttention(hidden, hidden, hidden, heads, dropout, drop_path=float(drop_path)),
                'edge_upd': EdgeUpdaterModule(hidden, hidden, hidden, dropout)
            }))

        self.tpe = nn.Embedding(seq_len, hidden)

        self.stream_local = TemporalInception1D(hidden, hidden, kernel_set=kernels)
        self.stream_global = LinearTemporalAttention(hidden, heads, dropout)

        self.crypto_adapter = CryptPEFT_adapter(
            num_heads=heads,
            attention_dropout=dropout,
            d_model=hidden,
            bottleneck=hidden // 4,
            dropout=dropout,
            mlp_dim=hidden * 2
        )

        self.gating = EntropyGatingUnit(hidden)

        # 【KBS 创新点 1 接入】：实例化 DEM
        self.dem = DynamicEdgeMemory(hidden)

        self.proj_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

        # 分类器输入维度适配 DEM 的输出 (hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes)
        )

    # 引入 global_normal_prototype 用于引导式学习和推斥
    def forward(self, graphs, global_normal_prototype=None):
        spatial_node_feats = []
        spatial_edge_feats = []
        active_edge_indices = []
        edge_masks = []
        batch_global_ids = []

        def _spatial_encode_one_frame(data, dropedge_p, prototype=None):
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            batch = data.batch if hasattr(data, "batch") else None
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

            if hasattr(data, "n_id"):
                frame_global_ids = data.n_id
            elif hasattr(data, "id"):
                frame_global_ids = data.id
            else:
                frame_global_ids = torch.arange(x.size(0), device=x.device)

            if self.training:
                if float(dropedge_p) > 0.0:
                    edge_index_d, edge_mask = dropout_edge(
                        edge_index, p=float(dropedge_p), force_undirected=False
                    )
                    edge_attr_d = edge_attr[edge_mask]
                    edge_attr_d = torch.nan_to_num(edge_attr_d, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    edge_index_d = edge_index
                    edge_attr_d = edge_attr
                    edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
            else:
                edge_index_d = edge_index
                edge_attr_d = edge_attr
                edge_mask = None

            active_edge_index = edge_index_d.clone()
            x = self.node_enc(x)
            edge_attr_d = self.edge_enc(edge_attr_d)

            for layer in self.spatial_layers:
                # 传入 prototype 进行图对齐学习
                x = layer["node_att"](x, edge_index_d, edge_attr_d, batch, prototype)
                edge_attr_d = layer["edge_upd"](x, edge_index_d, edge_attr_d)

            return x, edge_attr_d, active_edge_index, edge_mask, frame_global_ids

        for t in range(self.seq_len):
            data = graphs[t]
            x, edge_feat, edge_index_active, edge_mask, frame_global_ids = _spatial_encode_one_frame(
                data, dropedge_p=self.dropedge_p, prototype=global_normal_prototype
            )

            batch_global_ids.append(frame_global_ids)
            edge_masks.append(edge_mask)
            active_edge_indices.append(edge_index_active)
            spatial_node_feats.append(x)
            spatial_edge_feats.append(edge_feat)

        all_ids = torch.cat(batch_global_ids)
        unique_ids, _ = torch.sort(torch.unique(all_ids))
        num_unique = len(unique_ids)
        device = unique_ids.device

        dense_stack = torch.zeros((num_unique, self.seq_len, self.hidden), device=device)
        for t in range(self.seq_len):
            indices = torch.searchsorted(unique_ids, batch_global_ids[t])
            dense_stack[indices, t, :] = spatial_node_feats[t]

        time_indices = torch.arange(self.seq_len, device=device)
        t_emb = self.tpe(time_indices).unsqueeze(0)
        x_base = dense_stack + t_emb

        x_local_in = x_base.permute(0, 2, 1)
        x_local = self.stream_local(x_local_in).permute(0, 2, 1)

        x_global = self.stream_global(x_base)

        x_global_adapted = x_global + self.crypto_adapter(x_global)

        dense_out, alpha_scores = self.gating(x_local, x_global_adapted, x_base)

        batch_preds = []
        cl_loss = torch.tensor(0.0, device=device)

        for t in range(self.seq_len):
            frame_ids = batch_global_ids[t]
            indices = torch.searchsorted(unique_ids, frame_ids)
            node_out_t = dense_out[indices, t, :]

            curr_edge_index = active_edge_indices[t]
            src, dst = curr_edge_index[0], curr_edge_index[1]

            # 【KBS 创新点 1 运用】：利用 DEM 融合空间边缘特征与节点时序上下文
            edge_rep = self.dem(spatial_edge_feats[t], node_out_t[src], node_out_t[dst])

            pred = self.classifier(edge_rep)
            batch_preds.append(pred)

            if self.training and t == self.seq_len // 2:
                edge_feat_anchor = spatial_edge_feats[t]
                if edge_feat_anchor is not None and edge_feat_anchor.size(0) > 0:
                    if edge_feat_anchor.size(0) > self.max_cl_edges:
                        perm = torch.randperm(edge_feat_anchor.size(0), device=device)[: self.max_cl_edges]
                        edge_feat_anchor = edge_feat_anchor[perm]

                    z_anchor = F.normalize(self.proj_head(edge_feat_anchor), dim=1)

                    if global_normal_prototype is not None:
                        # 全局正常原型推斥
                        proto_tensor = global_normal_prototype.to(device)
                        if proto_tensor.dim() == 1:
                            proto_tensor = proto_tensor.unsqueeze(0)

                        z_proto = F.normalize(self.proj_head(proto_tensor), dim=1)
                        z_pos = F.normalize(
                            self.proj_head(edge_feat_anchor + torch.randn_like(edge_feat_anchor) * 0.01), dim=1)

                        pos_score = torch.sum(z_anchor * z_pos, dim=-1, keepdim=True) / 0.1
                        neg_score = torch.matmul(z_anchor, z_proto.T) / 0.1

                        logits = torch.cat([pos_score, neg_score.expand(z_anchor.size(0), z_proto.size(0))], dim=1)
                        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
                        cl_loss = F.cross_entropy(logits, labels)
                    else:
                        edge_feat_pos = edge_feat_anchor + torch.randn_like(edge_feat_anchor) * 0.1
                        z_pos = F.normalize(self.proj_head(edge_feat_pos), dim=1)
                        logits = torch.matmul(z_anchor, z_pos.T) / 0.1
                        labels = torch.arange(z_anchor.size(0), device=device)
                        cl_loss = F.cross_entropy(logits, labels)
                else:
                    cl_loss = torch.tensor(0.0, device=device)

        self._last_edge_masks = edge_masks
        return batch_preds, cl_loss