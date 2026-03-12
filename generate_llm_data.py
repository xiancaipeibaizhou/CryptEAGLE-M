import os
import torch
import pickle
from scapy.all import PcapReader, IP, TCP, UDP
from transformers import AutoTokenizer
from tqdm import tqdm

# ==========================================
# 配置参数
# ==========================================
PCAP_FILE = "sample.pcap"          # 替换为你下载的 USTC-TFC2016 或 ISCX PCAP 路径
OUTPUT_FILE = "llm_tokens.pkl"     # 输出文件路径
MAX_PACKETS_PER_FLOW = 5           # 每个流最多截取前 5 个数据包 (聚焦握手阶段)
MAX_BYTES_PER_PACKET = 256         # 每个包最多读取 256 字节 (防止单包过长)
MAX_TOKEN_LENGTH = 512             # LLM 模型的最大 Token 输入长度

# 假设使用基础的 BERT Tokenizer (后期可直接替换为 ET-BERT 或 NetBERT 的本地路径)
TOKENIZER_NAME = "bert-base-uncased" 

# ==========================================
# 初始化 Tokenizer 并添加自定义拓扑标识符
# ==========================================
print(f"[*] Loading Tokenizer: {TOKENIZER_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# 【核心创新】：引入结构化层级标识符
special_tokens_dict = {'additional_special_tokens': ['<pkt>', '<head>', '<payload>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"[*] Added {num_added_toks} special tokens: <pkt>, <head>, <payload>")

# ==========================================
# 辅助函数：提取五元组 (五元组作为 Flow 的唯一 ID)
# ==========================================
def get_flow_id(pkt):
    """提取双向流的五元组 ID (修复端点对齐 Bug)"""
    if IP in pkt:
        src_ip, dst_ip, proto = pkt[IP].src, pkt[IP].dst, pkt[IP].proto
        if TCP in pkt:
            src_port, dst_port = pkt[TCP].sport, pkt[TCP].dport
        elif UDP in pkt:
            src_port, dst_port = pkt[UDP].sport, pkt[UDP].dport
        else:
            src_port, dst_port = 0, 0
            
        # 🌟【修复】：将 IP 和 端口 绑定为元组后再排序，确保真正的双向流唯一标识
        endpoints = sorted([(src_ip, src_port), (dst_ip, dst_port)])
        return f"{endpoints[0][0]}:{endpoints[0][1]}-{endpoints[1][0]}:{endpoints[1][1]}-{proto}"
    return None

# ==========================================
# 主流程：解析 PCAP 并截取字节
# ==========================================
def process_pcap(pcap_path):
    print(f"[*] Reading PCAP file: {pcap_path} ... (This may take a while)")
    
    flows_raw_bytes = {}
    
    # 使用 PcapReader 进行流式读取，防止几十GB的 PCAP 撑爆内存
    with PcapReader(pcap_path) as pcap_reader:
        for i, pkt in enumerate(tqdm(pcap_reader, desc="Processing Packets")):
            flow_id = get_flow_id(pkt)
            if not flow_id:
                continue
                
            if flow_id not in flows_raw_bytes:
                flows_raw_bytes[flow_id] = []
                
            # 如果该流的数据包还没达到上限，则继续截取
            if len(flows_raw_bytes[flow_id]) < MAX_PACKETS_PER_FLOW:
                # 提取原始字节流，并截断过长的包
                raw_bytes = bytes(pkt)[:MAX_BYTES_PER_PACKET]
                # 转化为十六进制字符串，两个字符加一个空格 (符合 WordPiece 分词习惯)
                hex_str = " ".join([f"{b:02x}" for b in raw_bytes])
                flows_raw_bytes[flow_id].append(hex_str)

    print(f"[*] Extracted {len(flows_raw_bytes)} unique flows.")
    return flows_raw_bytes

# ==========================================
# 转化为大模型 Token 向量
# ==========================================
def tokenize_flows(flows_raw_bytes, tokenizer):
    print("[*] Tokenizing hex streams for LLM...")
    flow_tokens = {}
    
    for flow_id, hex_pkts in tqdm(flows_raw_bytes.items(), desc="Tokenizing"):
        # 【核心操作】：用 <pkt> 将多个数据包连接起来，赋予时序边界知识
        sequence = " <pkt> ".join(hex_pkts)
        
        # 调用 Tokenizer 进行编码、截断和 Padding
        encoded = tokenizer(
            sequence,
            padding='max_length',
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors='pt'
        )
        
        # 存储 input_ids 和 attention_mask
        flow_tokens[flow_id] = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
        
    return flow_tokens

# ==========================================
# 执行入口
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(PCAP_FILE):
        print(f"[!] Error: PCAP file '{PCAP_FILE}' not found. Please provide a valid PCAP.")
        # 创建一个极其微小的伪造 PCAP 用于测试逻辑 (可选)
        # pass
    else:
        # 1. 解析 PCAP 提取前 N 个包的 Hex 序列
        flows_raw_bytes = process_pcap(PCAP_FILE)
        
        # 2. 将 Hex 序列 Token 化
        flow_tokens = tokenize_flows(flows_raw_bytes, tokenizer)
        
        # 3. 保存至本地
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump({
                'flow_tokens': flow_tokens,
                'max_len': MAX_TOKEN_LENGTH,
                'tokenizer_name': TOKENIZER_NAME
            }, f)
        
        print(f"[*] Success! Saved tokenized flows to {OUTPUT_FILE}")
        
        # 打印一个样本看看长什么样
        sample_id = list(flow_tokens.keys())[0]
        print("\n--- Sample Output ---")
        print(f"Flow ID: {sample_id}")
        print(f"Input IDs Shape: {flow_tokens[sample_id]['input_ids'].shape}")
        print(f"First 20 Tokens: {flow_tokens[sample_id]['input_ids'][:20]}")