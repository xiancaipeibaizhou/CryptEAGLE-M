import torch
from model import CryptEAGLE

def run_test():
    print("[*] 1. 加载构建好的图序列 (train_graphs.pt)...")
    try:
        graphs = torch.load("train_graphs.pt")
        print(f"    成功加载，图序列长度: {len(graphs)}")
        print(f"    单帧图节点特征维度: {graphs[0].x.shape}, 边特征维度: {graphs[0].edge_attr.shape}")
    except FileNotFoundError:
        print("[!] 找不到 train_graphs.pt，请先运行 build_knowledge_graph.py")
        return

    print("\n[*] 2. 初始化 CryptEAGLE 模型...")
    # 注意：这里的 node_in 和 edge_in 必须与 LLM 提取的特征维度一致 (BERT 默认 768)
    model = CryptEAGLE(
        node_in=768,
        edge_in=768,
        hidden=128,      # 我们的降维目标和核心计算维度
        num_classes=2,   # 假设为二分类 (正常/异常)
        seq_len=len(graphs),
        heads=4          # 适度调小 heads 以防测试时显存不足
    )
    
    # 将模型设为训练模式以触发对比学习 (Phase 5) 和 Dropout
    model.train()

    print("\n[*] 3. 模拟联邦服务器下发『全局正常原型』(Global Normal Prototype)...")
    # 原型维度与 hidden 维度对齐
    dummy_global_prototype = torch.randn(128) 

    print("\n[*] 4. 执行前向传播 (含 CryptPEFT 与 联邦推斥对比学习)...")
    try:
        batch_preds, cl_loss = model(graphs, global_normal_prototype=dummy_global_prototype)
        print("\n=========================================")
        print("🚀 测试圆满成功！全链路贯通！")
        print("=========================================")
        print(f"-> 最终分类器输出形状 (以最后一帧为例): {batch_preds[-1].shape}")
        print(f"-> 联邦对比学习损失 (CL Loss): {cl_loss.item():.4f}")
    except Exception as e:
        print(f"\n[!] 运行中出现错误: {str(e)}")
        raise e

if __name__ == "__main__":
    run_test()