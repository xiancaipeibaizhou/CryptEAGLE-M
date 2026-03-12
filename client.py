import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from datetime import datetime
from collections import OrderedDict
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score

# 导入你的模块
from hparams_crypteagle import resolve_hparams
from model import CryptEAGLE
from run_crypteagle import (
    TemporalGraphDataset, temporal_collate_fn, get_eval_predictions, 
    get_normal_indices, find_best_macro_f1_threshold_and_predict,
    compute_all_metrics, plot_and_save_confusion_matrix
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cic_ids2017')
    parser.add_argument('--data_dir', type=str, default='../processed_data')
    parser.add_argument('--cid', type=int, required=True, help="Client ID (e.g., 0, 1, 2...)")
    parser.add_argument('--num_clients', type=int, default=2, help="Total number of clients in FL")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    group_str = "FL_PEFT"
    h = resolve_hparams(env=os.environ, dataset=args.dataset)
    
    # 1. 建立极其严密的实验日志保存目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"FL_Client{args.cid}_dim{h['HIDDEN']}_seq{h['SEQ_LEN']}"
    save_dir = os.path.join("results", args.dataset, exp_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 [Client {args.cid}] 日志与模型将保存至: {save_dir}")

    # 2. 加载数据
    print(f"[*] [Client {args.cid}] 正在加载并切分 {args.dataset} 数据...")
    dataset_path = os.path.join(args.data_dir, args.dataset)
    train_graphs = torch.load(os.path.join(dataset_path, "train_graphs.pt"), weights_only=False)
    val_graphs = torch.load(os.path.join(dataset_path, "val_graphs.pt"), weights_only=False)
    test_graphs = torch.load(os.path.join(dataset_path, "test_graphs.pt"), weights_only=False)

    node_dim, edge_dim = train_graphs[0].x.shape[1], train_graphs[0].edge_attr.shape[1]
    
    counts = np.zeros(100) 
    for g in train_graphs: counts += np.bincount(g.edge_labels.numpy(), minlength=100)
    num_classes = int(np.max(np.nonzero(counts))) + 1
    
    import joblib
    label_enc_path = os.path.join(dataset_path, "label_encoder.pkl")
    class_names = joblib.load(label_enc_path).classes_ if os.path.exists(label_enc_path) else [f"Class_{i}" for i in range(num_classes)]

    # 切分数据构造联邦孤岛
    total_len = len(train_graphs)
    # 【真实联邦数据模拟】：将训练集动态切分为 N 个孤岛
    total_len = len(train_graphs)
    chunk_size = total_len // args.num_clients
    
    start_idx = args.cid * chunk_size
    # 确保最后一个节点能拿完剩下的所有余数
    end_idx = total_len if args.cid == args.num_clients - 1 else (args.cid + 1) * chunk_size
    
    train_graphs = train_graphs[start_idx:end_idx]
    print(f"[*] [Client {args.cid}] 认领了数据分片: {start_idx} 到 {end_idx} (共 {len(train_graphs)} 个图序列)")
    
    train_loader = DataLoader(TemporalGraphDataset(train_graphs, h["SEQ_LEN"]), batch_size=32, shuffle=True, collate_fn=temporal_collate_fn)
    val_loader = DataLoader(TemporalGraphDataset(val_graphs, h["SEQ_LEN"]), batch_size=32, shuffle=False, collate_fn=temporal_collate_fn)
    test_loader = DataLoader(TemporalGraphDataset(test_graphs, h["SEQ_LEN"]), batch_size=32, shuffle=False, collate_fn=temporal_collate_fn)

    # 3. 初始化模型与 PEFT 冻结
    model = CryptEAGLE(
        node_in=node_dim, edge_in=edge_dim, hidden=h["HIDDEN"], num_classes=num_classes,
        seq_len=h["SEQ_LEN"], heads=h["HEADS"], dropout=0.3, max_cl_edges=h.get("MAX_CL_EDGES", 4096),
        kernels=h["KERNELS"], drop_path=h.get("DROP_PATH", 0.1), dropedge_p=h.get("DROPEDGE_P", 0.2)
    ).to(device)

    for name, param in model.named_parameters():
        if "crypto_adapter" not in name and "classifier" not in name:
            param.requires_grad = False
            
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=h["LR"])
    criterion = nn.CrossEntropyLoss()

    # 用于保存日志的全局变量
    training_log = []
    best_val_score = -1.0

    # 4. 联邦客户端定义
    class CryptEAGLEClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            state_dict = model.state_dict()
            peft_keys = [k for k in state_dict.keys() if "crypto_adapter" in k or "classifier" in k]
            return [state_dict[k].cpu().numpy() for k in peft_keys]

        def set_parameters(self, parameters):
            state_dict = model.state_dict()
            peft_keys = [k for k in state_dict.keys() if "crypto_adapter" in k or "classifier" in k]
            params_dict = zip(peft_keys, parameters)
            state_dict.update({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            model.train()
            total_loss = 0
            
            from tqdm import tqdm
            pbar = tqdm(train_loader, desc=f"Client {args.cid} Fitting", leave=False)
            for batched_seq in pbar:
                batched_seq = [g.to(device) for g in batched_seq]
                optimizer.zero_grad()
                out = model(batched_seq, global_normal_prototype=None)
                all_preds, cl_loss = out if isinstance(out, tuple) else (out, None)
                
                edge_masks = getattr(model, "_last_edge_masks", None)
                if edge_masks is not None and len(edge_masks) > 0 and edge_masks[-1] is not None:
                    labels = batched_seq[-1].edge_labels[edge_masks[-1]]
                else:
                    labels = batched_seq[-1].edge_labels
                    
                loss = criterion(all_preds[-1], labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                
            return self.get_parameters(config={}), len(train_loader.dataset), {"loss": total_loss / len(train_loader)}

        def evaluate(self, parameters, config):
            nonlocal best_val_score, training_log
            self.set_parameters(parameters)
            val_true, val_probs = get_eval_predictions(model, val_loader, device)
            
            normal_indices = get_normal_indices(class_names)
            val_normal_probs = np.sum(val_probs[:, normal_indices], axis=1)
            val_attack_probs = 1.0 - val_normal_probs
            val_true_bin = (~np.isin(val_true, normal_indices)).astype(int)
            
            try: auprc = average_precision_score(val_true_bin, val_attack_probs)
            except: auprc = 0.0
                
            log_line = f"Round Eval | Val AUPRC: {auprc:.4f}"
            print(f"[Client {args.cid}] {log_line}")
            training_log.append(log_line)
            
            # 🌟【核心：保存最好的模型】🌟
            if auprc > best_val_score:
                best_val_score = auprc
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                print(f"  --> [Saved] Best model updated (AUPRC: {auprc:.4f})")

            return float(1.0 - auprc), len(val_loader.dataset), {"auprc": auprc}

    # 5. 启动联邦学习，阻塞直到 10 轮结束
    print(f"[*] [Client {args.cid}] 连接服务器，开始联邦学习...")
    try:
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CryptEAGLEClient())
    except Exception as e:
        print(f"\n[*] [Client {args.cid}] 收到 Server 终止信号 (联邦早停触发)，断开连接。")

    # ========================================================
    # 🌟 6. 联邦学习结束！加载最好的模型进行测试并彻底保存指标 🌟
    # ========================================================
    print(f"\n[Client {args.cid}] 联邦学习结束！开始最终的 Test 评估...")
    
    # 加载刚才保存在本地的最佳权重
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
    
    val_true, val_probs = get_eval_predictions(model, val_loader, device)
    test_true, test_probs = get_eval_predictions(model, test_loader, device)
    normal_indices = get_normal_indices(class_names)
        
    test_pred, best_th, val_best_macro_f1, val_best_far = find_best_macro_f1_threshold_and_predict(
        val_true, val_probs, test_probs, normal_indices
    )
        
    metrics, cm = compute_all_metrics(test_true, test_pred, test_probs, class_names, normal_indices)
    
    test_normal_probs = np.sum(test_probs[:, normal_indices], axis=1)
    test_attack_probs = 1.0 - test_normal_probs
    test_true_bin = (~np.isin(test_true, normal_indices)).astype(int)
    test_auprc = average_precision_score(test_true_bin, test_attack_probs)
        
    print(f"  ✅ [Client {args.cid}] 评估完成! Test AUPRC: {test_auprc:.4f}, Test Macro F1: {metrics['F1 (Macro)']:.4f}, ASA: {metrics['ASA']:.4f}")
    plot_and_save_confusion_matrix(cm, class_names, os.path.join(save_dir, f"cm_thresh_{best_th:.2f}.png"))
    
    # 将日志写入硬盘
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"=== {exp_name} (Thresh: {best_th:.2f}) ===\n")
        f.write(f"Test AUPRC: {test_auprc:.4f}\n")
        for k, v in metrics.items(): 
            f.write(f"{k}: {v:.4f}\n")
            
    with open(os.path.join(save_dir, "training_history.log"), "w") as f:
        for log_line in training_log:
            f.write(log_line + "\n")
            
    csv_file = "fl_results.csv" 
    if not os.path.isfile(csv_file):
        with open(csv_file, "w") as f:
            f.write("Dataset,Client_ID,Threshold,ACC,APR,RE,F1_Macro,AUC,ASA,FAR\n")
            
    with open(csv_file, "a") as f:
        f.write(f"{args.dataset},Client_{args.cid},{best_th:.4f},"
                f"{metrics['ACC']:.4f},{metrics['APR']:.4f},{metrics['RE']:.4f},"
                f"{metrics['F1 (Macro)']:.4f},{metrics['AUC']:.4f},"
                f"{metrics['ASA']:.4f},{metrics['FAR']:.4f}\n")
    print(f"🎉 结果已完美存入 {save_dir} 与 fl_results.csv 中！")

if __name__ == "__main__":
    main()