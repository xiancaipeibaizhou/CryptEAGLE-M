import os
import torch
import pickle
from tqdm import tqdm
from transformers import AutoModel
from torch_geometric.data import Data

# ==========================================
# 配置参数
# ==========================================
INPUT_FILE = "llm_tokens.pkl"
OUTPUT_FILE = "train_graphs.pt"
LLM_MODEL_NAME = "bert-base-uncased"  # 需与 generate_llm_data.py 保持一致
SEQ_LEN = 10                          # CryptEAGLE 期待的时间步长

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_flow_id(flow_id):
    """解析五元组 ID 获取源和目的 IP，作为图的节点"""
    # 假设 flow_id 格式为: "192.168.1.1:1234-10.0.0.1:80-TCP"
    parts = flow_id.split('-')
    src_ip = parts[0].split(':')[0]
    dst_ip = parts[1].split(':')[0]
    return src_ip, dst_ip

def build_graph():
    if not os.path.exists(INPUT_FILE):
        print(f"[!] Error: {INPUT_FILE} not found. Run generate_llm_data.py first.")
        return

    # 1. 加载 Token 数据
    print(f"[*] Loading tokens from {INPUT_FILE}...")
    with open(INPUT_FILE, 'rb') as f:
        data = pickle.load(f)
    flow_tokens = data['flow_tokens']
    
    # 2. 加载冻结的 LLM 模型 (特征透镜)
    print(f"[*] Loading frozen LLM ({LLM_MODEL_NAME}) as feature lens...")
    llm = AutoModel.from_pretrained(LLM_MODEL_NAME).to(device)
    llm.eval() # 开启评估模式
    for param in llm.parameters():
        param.requires_grad = False # 彻底冻结底座，防止显存爆炸

    # 3. 提取 Level-1 数据包动态知识 (Packet Dynamics Knowledge)
    print("[*] Extracting semantic embeddings via LLM...")
    edge_features = []
    src_nodes, dst_nodes = [], []
    node_mapping = {}
    node_counter = 0

    # 批量处理防止 OOM (为了简化代码，这里采用逐个流处理，实战中可增加 DataLoader)
    for flow_id, tokens in tqdm(flow_tokens.items(), desc="LLM Inference"):
        input_ids = tokens['input_ids'].unsqueeze(0).to(device)
        attention_mask = tokens['attention_mask'].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = llm(input_ids=input_ids, attention_mask=attention_mask)
            # 提取 [CLS] 向量作为整个流的语义表征 (768 维)
            flow_emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
            edge_features.append(flow_emb)
        
        # 映射 IP 为图的节点索引
        src_ip, dst_ip = parse_flow_id(flow_id)
        for ip in [src_ip, dst_ip]:
            if ip not in node_mapping:
                node_mapping[ip] = node_counter
                node_counter += 1
        src_nodes.append(node_mapping[src_ip])
        dst_nodes.append(node_mapping[dst_ip])

    # 4. 构建图拓扑 (Interaction Topology)
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_attr = torch.stack(edge_features)  # Shape: [num_edges, 768]
    
    # 节点特征暂用度数/或者简单的零向量替代，因为核心信息在大模型提取的边特征(流载荷)里
    x = torch.zeros((node_counter, 768), dtype=torch.float) 

    print(f"[*] Built global graph: {node_counter} nodes, {edge_attr.size(0)} edges.")

    # 5. 切分序列图 (适配 CryptEAGLE 的时空双流)
    # 实际场景中应根据时间戳划分，这里为演示，将大图随机拆分为 SEQ_LEN 个子图
    graphs = []
    chunk_size = max(1, edge_attr.size(0) // SEQ_LEN)
    
    for t in range(SEQ_LEN):
        start_idx = t * chunk_size
        end_idx = start_idx + chunk_size if t < SEQ_LEN - 1 else edge_attr.size(0)
        
        sub_edge_index = edge_index[:, start_idx:end_idx]
        sub_edge_attr = edge_attr[start_idx:end_idx]
        
        data_t = Data(x=x.clone(), edge_index=sub_edge_index, edge_attr=sub_edge_attr)
        graphs.append(data_t)

    # 保存图序列
    torch.save(graphs, OUTPUT_FILE)
    print(f"[*] Success! Saved {SEQ_LEN} sequential graphs to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_graph()