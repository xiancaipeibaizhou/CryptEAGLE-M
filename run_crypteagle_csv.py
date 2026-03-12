import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm

# 导入我们刚刚写好的完全体模型
from model import CryptEAGLE

# ==========================================
# 1. 配置与超参数
# ==========================================
DATA_PATH = "your_dataset_path/cic_ids2017_seq.pt"  # 替换为你之前 MILAN 用到的 .pt 数据路径
HIDDEN_DIM = 128
SEQ_LEN = 10  # 需与你生成的图序列长度一致
NUM_CLASSES = 2  # 比如二分类：正常 vs 攻击
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 1  # 序列图通常以 batch_size=1 (即一个完整的时序窗口) 喂入

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 加载数据并自动推断维度
# ==========================================
print(f"[*] 加载 CSV 图数据: {DATA_PATH}")
# 假设你的数据是一个包含多个时序窗口的列表：[ window1, window2, ... ]
# 其中 window = [data_t1, data_t2, ..., data_t10]
try:
    dataset = torch.load(DATA_PATH)
except Exception as e:
    print(f"[!] 数据加载失败，请确保路径正确且为 PyG 数据格式。报错: {e}")
    exit()

# 从第一帧图中自动推断 CSV 的统计特征维度
sample_window = dataset[0] if isinstance(dataset[0], list) else [dataset[0]]
node_in_dim = sample_window[0].x.shape[1]
edge_in_dim = sample_window[0].edge_attr.shape[1]

print(f"[*] 自动推断维度成功 -> Node Feature Dim: {node_in_dim}, Edge Feature Dim: {edge_in_dim}")

# ==========================================
# 3. 初始化 CryptEAGLE-M 引擎
# ==========================================
print("[*] 初始化 CryptEAGLE-M 引擎...")
model = CryptEAGLE(
    node_in=node_in_dim,
    edge_in=edge_in_dim,
    hidden=HIDDEN_DIM,
    num_classes=NUM_CLASSES,
    seq_len=SEQ_LEN,
    heads=4,
    dropout=0.3,
    dropedge_p=0.2  # 适度随机丢弃边，增强对网络波动的鲁棒性
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
# 引入 Cosine Annealing (如论文所述)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

criterion = nn.CrossEntropyLoss()

# ==========================================
# 4. 训练循环 (Training Loop)
# ==========================================
print("\n" + "=" * 50)
print("🚀 开始单机版 CryptEAGLE-M 训练 (CSV 统计特征模式)")
print("=" * 50)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_cl_loss = 0
    total_cls_loss = 0
    start_time = time.time()

    # tqdm 进度条
    pbar = tqdm(dataset, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for step, graph_sequence in enumerate(pbar):
        # 将序列转移到 GPU
        graphs = [g.to(device) for g in graph_sequence]

        optimizer.zero_grad()

        # 前向传播：注意单机版我们不传 global_normal_prototype
        batch_preds, cl_loss = model(graphs, global_normal_prototype=None)

        # 计算分类损失 (这里为了简化，取序列最后一帧的预测和标签)
        # 实际情况请根据你之前 MILAN 的标签对齐逻辑进行调整
        pred_last_frame = batch_preds[-1]

        # 假设 graph_sequence[-1].y 存储了边的真实标签
        # 你可能需要根据实际 y 的形状做 squeeze 或 view 操作
        labels = graphs[-1].y.long()

        cls_loss = criterion(pred_last_frame, labels)

        # 联合优化 (分类损失 + 0.1 * 对比推斥损失)
        loss = cls_loss + 0.1 * cl_loss

        loss.backward()

        # 梯度裁剪，防止初期震荡
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_cl_loss += cl_loss.item()

        pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'CL_Loss': f"{cl_loss.item():.4f}"})

    scheduler.step()
    epoch_time = time.time() - start_time

    print(f"\n[Epoch {epoch + 1} Summary] Time: {epoch_time:.2f}s | Avg Loss: {total_loss / len(dataset):.4f} "
          f"(Cls: {total_cls_loss / len(dataset):.4f}, CL: {total_cl_loss / len(dataset):.4f}) | LR: {scheduler.get_last_lr()[0]:.6f}")

print("[*] 训练完成！这是你走向顶刊的第一批实验数据。")