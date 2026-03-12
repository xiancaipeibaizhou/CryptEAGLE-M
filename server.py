# server.py
import flwr as fl
import os
import csv
import sys
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy

# 🌟 全局指标 CSV 存储路径
GLOBAL_CSV_FILE = "server_global_metrics.csv"
# 🌟 全局早停耐心值 (连续 5 轮不提升则终止全网训练)
PATIENCE = 5 

class EarlyStoppingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, patience: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.best_auprc = -1.0
        self.patience_counter = 0

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        # 1. 调用父类方法进行底层的加权聚合
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if not results:
            return aggregated_loss, aggregated_metrics

        # 2. 计算全局加权平均 AUPRC (根据孤岛数据量加权)
        total_examples = sum([res.num_examples for _, res in results])
        weighted_auprc = sum([res.num_examples * res.metrics.get("auprc", 0.0) for _, res in results]) / total_examples
        
        print(f"\n{'='*50}")
        print(f"🌍 [Global Brain] Round {server_round} 聚合完成 | 全局加权 AUPRC: {weighted_auprc:.4f}")
        print(f"{'='*50}\n")
        
        # 3. 将全局指标严格写入 CSV
        file_exists = os.path.isfile(GLOBAL_CSV_FILE)
        with open(GLOBAL_CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Round", "Total_Samples", "Global_AUPRC"])
            writer.writerow([server_round, total_examples, f"{weighted_auprc:.4f}"])
        
        # 4. 🌟【核心突破】：全局早停机制 (Global Early Stopping)
        if weighted_auprc > self.best_auprc + 0.0001:
            self.best_auprc = weighted_auprc
            self.patience_counter = 0
            print(f"  --> 🌟 [Server] 发现新的最佳全局 AUPRC: {self.best_auprc:.4f}！早停计数器清零。")
        else:
            self.patience_counter += 1
            print(f"  --> ⚠️ [Server] 全局 AUPRC 未提升. 早停计数器: {self.patience_counter}/{self.patience}")
            if self.patience_counter >= self.patience:
                print(f"\n🛑 [Server] 触发全局早停机制！连续 {self.patience} 轮未提升。")
                print(f"🛑 [Server] 联邦网络达到纳什均衡，正在强行终止所有孤岛训练...")
                # 强行终止 Server 进程，这会导致底下的 Clients 自动断开连接并进入保存测试流程
                sys.exit(0)
        
        if aggregated_metrics is None:
            aggregated_metrics = {}
        aggregated_metrics["global_auprc"] = weighted_auprc
        
        return aggregated_loss, aggregated_metrics

def main():
    print("🚀 启动 CryptEAGLE 联邦中心服务器 (早停机制 + CSV 保存版)...")
    
    if os.path.exists(GLOBAL_CSV_FILE):
        os.remove(GLOBAL_CSV_FILE)
        print(f"[*] 已清理旧的 {GLOBAL_CSV_FILE}，准备记录新实验。")
        
    # 使用我们刚刚手写的早停联邦策略
    strategy = EarlyStoppingFedAvg(
        patience=PATIENCE,
        fraction_fit=1.0,               
        fraction_evaluate=1.0,          
        min_fit_clients=2,              # 严格锁定为 2 个客户端
        min_evaluate_clients=2,         
        min_available_clients=2,        
    )

    # 这里的 50 轮是理论上限，因为加了早停，极大概率在 15~20 轮左右就会自动结束
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=50), 
        strategy=strategy,
    )

if __name__ == "__main__":
    main()