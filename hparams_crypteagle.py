import os

def get_default_hparams(dataset="cic_ids2017"):
    """
    CryptEAGLE-M 顶刊级黄金超参数字典
    """
    dataset = dataset.lower()
    
    # ---------------------------------------------------------
    # 1. CIC-IDS2017 (高频大流量 DDoS/Web 攻击)
    # 策略：短时间窗口(T=3)，中等丢边率防止过拟合(p=0.2)，关闭对比学习(CL=0.0)
    # ---------------------------------------------------------
    if "2017" in dataset:
        return {
            "SEQ_LEN": 3,               # 短窗口捕捉微观爆发
            "BATCH_SIZE": 1,
            "NUM_EPOCHS": 50,
            "LR": 1e-3,
            "HIDDEN": 128,              # 核心特征维度
            "HEADS": 4,                 # CryptoNorm Attention 头数
            "KERNELS": [1, 3],          # Local Stream 偏向极短期的 Inception 核
            "DROPEDGE_P": 0.2,          # Moderate Jitter 适度拓扑抖动
            "CL_LOSS_WEIGHT": 0.0,      # 大流量攻击特征极其明显，开启强推斥反而增加方差
            "MAX_CL_EDGES": 4096,
            "PATIENCE": 10,             # 早停耐心值
            "ACCUM_STEPS": 4,           # 梯度累加，稳定 batch 训练
            "DROP_PATH": 0.1,
            "WARMUP_EPOCHS": 5,
            "COSINE_T0": 10,
            "COSINE_TMULT": 2,
            "MIN_DELTA": 0.0001
        }
        
    # ---------------------------------------------------------
    # 2. CIC-UNSW-NB15 (隐蔽、多步长 APT 攻击)
    # 策略：长时序感受野(T=10, 核长1..11)，保持静态拓扑(p=0.0)防止攻击链断裂
    # ---------------------------------------------------------
    elif "nb15" in dataset or "unsw" in dataset:
        return {
            "SEQ_LEN": 10,
            "BATCH_SIZE": 1,
            "NUM_EPOCHS": 60,
            "LR": 5e-4,                 # 学习率略低，防止多步攻击特征被冲刷
            "HIDDEN": 128,
            "HEADS": 4,
            "KERNELS": [1, 3, 5, 7, 9, 11], # 极宽的感受野捕捉长期渗透
            "DROPEDGE_P": 0.0,          # 必须为0！不能丢边，否则多步攻击拓扑变成孤岛
            "CL_LOSS_WEIGHT": 0.0,      
            "MAX_CL_EDGES": 8192,
            "PATIENCE": 12,
            "ACCUM_STEPS": 2,
            "DROP_PATH": 0.1,
            "WARMUP_EPOCHS": 5,
            "COSINE_T0": 15,
            "COSINE_TMULT": 1,
            "MIN_DELTA": 0.0001
        }
        
    # ---------------------------------------------------------
    # 3. CIC-Darknet2020 (高熵加密隧道流量)
    # 策略：长窗口(T=10)，基础多尺度(1,3,5,7)，强力对比推斥(CL=0.5)强制分离加密特征
    # ---------------------------------------------------------
    elif "2020" in dataset or "darknet" in dataset:
        return {
            "SEQ_LEN": 10,              # 长窗口分析持续的加密通信
            "BATCH_SIZE": 1,
            "NUM_EPOCHS": 50,
            "LR": 8e-4,
            "HIDDEN": 128,
            "HEADS": 8,                 # 增加注意力头数，解析复杂的高熵空间
            "KERNELS": [1, 3, 5, 7],
            "DROPEDGE_P": 0.1,          # Mild Jitter 清除随机加密的结构噪声
            "CL_LOSS_WEIGHT": 0.5,      # 🌟 杀手锏：强力开启对比学习，强制分离加密特征
            "MAX_CL_EDGES": 8192,
            "PATIENCE": 10,
            "ACCUM_STEPS": 4,
            "DROP_PATH": 0.2,
            "WARMUP_EPOCHS": 3,
            "COSINE_T0": 10,
            "COSINE_TMULT": 2,
            "MIN_DELTA": 0.0001
        }
        
    # 默认兜底配置
    else:
        return {
            "SEQ_LEN": 5, "BATCH_SIZE": 1, "NUM_EPOCHS": 50, "LR": 1e-3,
            "HIDDEN": 128, "HEADS": 4, "KERNELS": [1, 3, 5], "DROPEDGE_P": 0.1,
            "CL_LOSS_WEIGHT": 0.1, "MAX_CL_EDGES": 4096, "PATIENCE": 10,
            "ACCUM_STEPS": 2, "DROP_PATH": 0.1, "WARMUP_EPOCHS": 5,
            "COSINE_T0": 10, "COSINE_TMULT": 1, "MIN_DELTA": 0.0001
        }

def resolve_hparams(group_str=None, env=None, dataset="cic_ids2017"):
    """
    提供和之前 MILAN 兼容的超参解析接口。
    你可以通过环境变量覆盖这些默认设置，比如：
    export LR=0.002
    """
    hparams = get_default_hparams(dataset)
    
    if env is not None:
        for k in hparams.keys():
            if k in env:
                # 自动类型转换
                if isinstance(hparams[k], int):
                    hparams[k] = int(env[k])
                elif isinstance(hparams[k], float):
                    hparams[k] = float(env[k])
                elif isinstance(hparams[k], list):
                    hparams[k] = [int(x) for x in env[k].split(",") if x.strip()]
                else:
                    hparams[k] = env[k]
                    
    return hparams