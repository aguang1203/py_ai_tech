"""
=============================================================================
LSTM 时序异常检测任务模板 (Time Series Anomaly Detection)
=============================================================================

【原理】
时序异常检测的目标是找出时间序列中"不正常"的数据点。
基于LSTM的方法属于"重构式"异常检测——用LSTM学习正常数据的模式，
重构误差大的数据点就是异常。

【重构式异常检测流程】
  训练阶段: 只用正常数据训练LSTM自编码器
    输入 → LSTM编码器 → 瓶颈层 → LSTM解码器 → 重构输出
    目标: 让重构输出尽可能接近输入(学习正常模式)

  检测阶段: 对新数据计算重构误差
    正常数据: 重构误差小(LSTM见过的模式，能较好重构)
    异常数据: 重构误差大(LSTM没见过的模式，无法重构)

  判定: 重构误差 > 阈值 → 异常

【阈值选择】
  常用方法:
    1. 统计法: 阈值 = μ + kσ (正常数据重构误差的均值+k倍标准差)
    2. 百分位法: 阈值 = 第99百分位的重构误差
    3. 动态阈值: 用滑动窗口计算局部统计量
  本模板使用统计法(k=3，即3σ原则)

【LSTM自编码器架构】
  编码器: LSTM将输入序列压缩为低维瓶颈表示
  解码器: LSTM从瓶颈表示重建原始序列

  两种解码方式:
    1. 自回归解码: 每步用上一步的输出作为输入(序列生成)
    2. 教师强制解码: 每步用真实值作为输入(本模板使用，更稳定)

【应用场景】
- 工业设备故障检测(传感器异常)
- 网络入侵检测(流量异常)
- 金融欺诈检测(交易异常)
- 服务器监控(CPU/内存异常)
- 医疗监护(生命体征异常)

【本数据集: 合成传感器数据】
- 正常数据: 多频正弦波+噪声(模拟传感器周期信号)
- 异常数据: 突发尖峰/电平偏移/频率变化(模拟设备故障)
- 异常比例: 约5%(符合真实场景中异常稀少的特征)
- 即时生成，无需下载

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python lstm/anomaly_detection.py
3. 数据集自动生成
=============================================================================
"""

# ============================================================
# Step 1: 导入必要的库
# ============================================================
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# 设置中文字体
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

now = datetime.datetime.now


# ============================================================
# Step 2: 配置超参数
# ============================================================
class CONFIG:
    """超参数配置中心 —— 时序异常检测任务的所有可调参数。"""

    # --- 数据相关 ---
    # seq_len=30: 输入序列长度(用30个时间步作为检测窗口)
    #   为什么30？窗口需要足够长以包含正常模式
    #   太短(如5): 看不出周期性，正常模式都像异常
    #   太长(如200): 计算量大，且异常可能被稀释
    seq_len = 30

    # num_normal=3000: 正常样本数
    num_normal = 3000

    # num_test=1000: 测试样本数(含正常+异常)
    num_test = 1000

    # anomaly_ratio=0.05: 测试集中异常比例
    #   为什么5%？真实场景中异常通常很稀少(1%-10%)
    anomaly_ratio = 0.05

    # anomaly_types: 异常类型
    #   "spike": 突发尖峰(模拟设备突然过载)
    #   "level_shift": 电平偏移(模拟传感器漂移)
    #   "frequency_change": 频率变化(模拟设备运转异常)
    anomaly_types = ["spike", "level_shift", "frequency_change"]

    # --- 模型相关(编码器) ---
    # input_dim=1: 输入特征维度(单变量传感器数据)
    input_dim = 1

    # hidden_dim=64: LSTM隐藏层维度
    hidden_dim = 64

    # num_layers=2: LSTM层数
    num_layers = 2

    # latent_dim=16: 瓶颈层维度
    #   为什么16？将30维的序列压缩到16维，迫使LSTM学习最关键的模式
    #   太大(如64): 没有压缩效果，模型可以"记住"每个样本，无法检测异常
    #   太小(如2): 信息瓶颈太窄，正常数据也重构不好
    latent_dim = 16

    # dropout=0.2: Dropout比例
    dropout = 0.2

    # --- 训练相关 ---
    # batch_size=32: 批次大小
    batch_size = 32

    # learning_rate=1e-3: 初始学习率
    learning_rate = 1e-3

    # epochs=80: 最大训练轮数
    epochs = 80

    # weight_decay=1e-5: L2正则化
    weight_decay = 1e-5

    # --- 早停策略 ---
    early_stop_patience = 15

    # --- 学习率调度器 ---
    scheduler_type = "cosine"

    # --- 梯度裁剪 ---
    max_grad_norm = 1.0

    # --- 混合精度训练(AMP) ---
    use_amp = True

    # --- 数据加载优化 ---
    num_workers = min(4, os.cpu_count() or 1)

    # --- 异常检测相关 ---
    # threshold_k=3: 阈值倍数(3σ原则)
    #   阈值 = μ + k × σ (正常数据重构误差的均值 + k倍标准差)
    #   k=3: 覆盖99.7%的正常数据(正态分布假设)
    #   k=2: 覆盖95%(更敏感，检测更多异常，但误报也多)
    #   k=4: 覆盖99.99%(更保守，漏检多但误报少)
    threshold_k = 3.0

    # --- 保存相关 ---
    save_dir = "lstm/output/anomaly_detection"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 数据生成与加载
# ============================================================
def generate_normal_signal(length, seed=None):
    """
    生成正常传感器信号。

    正常信号 = 主周期(日周期) + 次周期(小时周期) + 微弱噪声
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    t = np.arange(length, dtype=np.float32)
    # 主周期: 周期=24, 振幅=1.0
    signal = 1.0 * np.sin(2 * np.pi * t / 24.0)
    # 次周期: 周期=6, 振幅=0.3
    signal += 0.3 * np.sin(2 * np.pi * t / 6.0)
    # 噪声
    signal += rng.randn(length) * 0.05

    return signal.astype(np.float32)


def inject_anomaly(signal, anomaly_type="spike", rng=None):
    """
    在信号中注入异常。

    【三种异常类型】
    1. spike(尖峰): 在随机位置添加大幅值脉冲
       模拟: 传感器故障/电磁干扰
    2. level_shift(电平偏移): 信号整体向上/下偏移
       模拟: 传感器漂移/基线变化
    3. frequency_change(频率变化): 改变信号的频率
       模拟: 设备转速异常/振动频率变化
    """
    if rng is None:
        rng = np.random.RandomState()

    signal = signal.copy()
    length = len(signal)

    if anomaly_type == "spike":
        # 在1-3个位置注入尖峰
        n_spikes = rng.randint(1, 4)
        for _ in range(n_spikes):
            pos = rng.randint(0, length)
            # 尖峰振幅: 3-6倍正常振幅
            amplitude = rng.choice([-1, 1]) * rng.uniform(3, 6)
            signal[pos] += amplitude

    elif anomaly_type == "level_shift":
        # 在某个时间点开始偏移
        shift_start = rng.randint(length // 4, 3 * length // 4)
        # 偏移量: 2-4倍正常振幅
        shift_value = rng.choice([-1, 1]) * rng.uniform(2, 4)
        signal[shift_start:] += shift_value

    elif anomaly_type == "frequency_change":
        # 改变频率: 原频率的1.5-3倍
        freq_factor = rng.uniform(1.5, 3.0)
        t = np.arange(length, dtype=np.float32)
        # 重新生成但保持偏移后的相位
        signal = 1.0 * np.sin(2 * np.pi * t * freq_factor / 24.0)
        signal += 0.3 * np.sin(2 * np.pi * t * freq_factor / 6.0)
        signal += rng.randn(length) * 0.05
        signal = signal.astype(np.float32)

    return signal


class AnomalyDataset(Dataset):
    """
    异常检测数据集。

    【数据格式】
    每个样本: (序列, 标签)
    - 序列: (seq_len, input_dim) 传感器数据
    - 标签: 0=正常, 1=异常

    【训练集只有正常数据】
    自编码器只学习正常模式，异常数据的重构误差自然大。
    """

    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        if seq.ndim == 1:
            seq = seq.unsqueeze(-1)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return seq, label
        return seq, torch.tensor(0, dtype=torch.long)


def generate_data(cfg):
    """生成训练和测试数据"""
    np.random.seed(42)

    # --- 生成正常训练数据 ---
    train_sequences = []
    for i in range(cfg.num_normal):
        # 随机起始点，从更长的信号中截取
        start = np.random.randint(0, 500)
        signal = generate_normal_signal(start + cfg.seq_len + 10, seed=i)
        segment = signal[start:start + cfg.seq_len]
        train_sequences.append(segment)

    # --- 生成测试数据(含正常+异常) ---
    test_sequences = []
    test_labels = []

    n_anomaly = int(cfg.num_test * cfg.anomaly_ratio)
    n_normal_test = cfg.num_test - n_anomaly

    # 正常测试数据
    for i in range(n_normal_test):
        start = np.random.randint(500, 1000)
        signal = generate_normal_signal(start + cfg.seq_len + 10, seed=10000 + i)
        segment = signal[start:start + cfg.seq_len]
        test_sequences.append(segment)
        test_labels.append(0)

    # 异常测试数据
    for i in range(n_anomaly):
        start = np.random.randint(500, 1000)
        signal = generate_normal_signal(start + cfg.seq_len + 10, seed=20000 + i)
        segment = signal[start:start + cfg.seq_len]
        # 注入随机类型的异常
        anomaly_type = cfg.anomaly_types[i % len(cfg.anomaly_types)]
        segment = inject_anomaly(segment, anomaly_type, rng=np.random.RandomState(30000 + i))
        test_sequences.append(segment)
        test_labels.append(1)

    # 打乱测试数据
    perm = np.random.permutation(len(test_sequences))
    test_sequences = [test_sequences[i] for i in perm]
    test_labels = [test_labels[i] for i in perm]

    print(f"训练集: {len(train_sequences)}样本(全部正常)")
    print(f"测试集: {len(test_sequences)}样本(正常{n_normal_test} + 异常{n_anomaly})")

    return train_sequences, test_sequences, test_labels


def get_dataloaders(cfg):
    """创建DataLoader"""
    train_sequences, test_sequences, test_labels = generate_data(cfg)

    # 从训练集划出10%作为验证集
    n_val = int(len(train_sequences) * 0.1)
    val_sequences = train_sequences[:n_val]
    train_sequences = train_sequences[n_val:]

    train_dataset = AnomalyDataset(train_sequences)
    val_dataset = AnomalyDataset(val_sequences)
    test_dataset = AnomalyDataset(test_sequences, test_labels)

    pin_mem = cfg.device.type == "cuda"
    pw = cfg.num_workers > 0

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=pw,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=pw,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=pw,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# Step 4: 模型定义
# ============================================================
class LSTMAutoencoder(nn.Module):
    """
    LSTM自编码器用于异常检测。

    【架构设计】
    编码器: 输入序列 → LSTM → 瓶颈表示
    解码器: 瓶颈表示 → LSTM → 重构序列

    【为什么用自编码器而不是预测模型？】
    - 预测模型: 预测下一步，只能检测"突变型"异常
    - 自编码器: 重构整个序列，能检测各种类型的异常
    - 自编码器学习正常数据的"压缩表示"，异常数据无法被有效压缩
    - 重构误差就是异常分数

    【瓶颈层的作用】
    信息瓶颈迫使LSTM学习最本质的特征，而非"记住"每个样本。
    如果没有瓶颈(编码维度=输入维度)，模型可以逐元素复制，
    任何输入都能完美重构，失去检测能力。
    """

    def __init__(self, cfg):
        super().__init__()

        # 编码器: 将输入序列压缩为瓶颈表示
        self.encoder = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
        )

        # 瓶颈层: 将编码器的输出压缩到低维
        self.bottleneck = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 解码器: 从瓶颈表示重构序列
        # 输入: 重复瓶颈表示seq_len次 → 解码
        self.decoder = nn.LSTM(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
        )

        # 输出层: 映射到输入维度
        self.output_layer = nn.Linear(cfg.hidden_dim, cfg.input_dim)

        self.seq_len = cfg.seq_len

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                if "lstm" in name:
                    hidden_dim = param.shape[0] // 4
                    param.data.fill_(0)
                    param.data[hidden_dim:2 * hidden_dim].fill_(1.0)
                else:
                    nn.init.constant_(param, 0)

    def encode(self, x):
        """编码: 输入序列 → 瓶颈表示"""
        _, (h_n, _) = self.encoder(x)
        # 取最后一层的隐藏状态
        encoded = h_n[-1]  # (batch, hidden_dim)
        # 通过瓶颈层
        latent = self.bottleneck(encoded)  # (batch, hidden_dim)
        return latent

    def decode(self, latent):
        """解码: 瓶颈表示 → 重构序列"""
        # 将瓶颈表示重复seq_len次，作为解码器的输入
        # (batch, hidden_dim) → (batch, seq_len, hidden_dim)
        repeated = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded, _ = self.decoder(repeated)
        # 映射到输入维度
        output = self.output_layer(decoded)  # (batch, seq_len, input_dim)
        return output

    def forward(self, x):
        """
        前向传播: 编码 → 瓶颈 → 解码

        参数:
            x: (batch, seq_len, input_dim) 输入序列
        返回:
            reconstructed: (batch, seq_len, input_dim) 重构序列
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed

    def get_reconstruction_error(self, x):
        """
        计算重构误差(逐点MSE)。

        返回:
            errors: (batch, seq_len) 每个时间步的重构误差
        """
        reconstructed = self.forward(x)
        errors = ((x - reconstructed) ** 2).squeeze(-1)  # (batch, seq_len)
        return errors


# ============================================================
# Step 5: 训练函数
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, cfg, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for batch_x, _ in loader:
        batch_x = batch_x.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, cfg):
    """评估模型(重构损失)"""
    model.eval()
    total_loss = 0
    n_batches = 0
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for batch_x, _ in loader:
        batch_x = batch_x.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def compute_threshold(model, loader, cfg):
    """
    计算异常检测阈值。

    【3σ原则】
    假设正常数据的重构误差服从正态分布:
    - 68.3% 的数据在 μ±1σ 内
    - 95.4% 的数据在 μ±2σ 内
    - 99.7% 的数据在 μ±3σ 内

    阈值 = μ + kσ
    - k=2: 覆盖95%，5%正常数据被判为异常(误报率高)
    - k=3: 覆盖99.7%，0.3%正常数据被判为异常(误报率低)
    - k=4: 极保守，几乎不误报，但可能漏检
    """
    model.eval()
    all_errors = []
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for batch_x, _ in loader:
        batch_x = batch_x.to(cfg.device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            errors = model.get_reconstruction_error(batch_x)
        all_errors.append(errors.detach().cpu().numpy())

    all_errors = np.concatenate(all_errors, axis=0)
    # 每个样本的平均重构误差
    sample_errors = all_errors.mean(axis=1)

    mean = sample_errors.mean()
    std = sample_errors.std()
    threshold = mean + cfg.threshold_k * std

    print(f"正常数据重构误差: μ={mean:.6f}, σ={std:.6f}")
    print(f"异常阈值(k={cfg.threshold_k}): {threshold:.6f}")

    return threshold, mean, std


def detect_anomalies(model, loader, threshold, cfg):
    """
    检测异常。

    返回:
        results: 字典，包含各种评估指标
    """
    model.eval()
    all_errors = []
    all_labels = []
    all_sequences = []
    all_reconstructed = []
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for batch_x, batch_labels in loader:
        batch_x = batch_x.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            reconstructed = model(batch_x)
            errors = model.get_reconstruction_error(batch_x)

        all_errors.append(errors.detach().cpu().numpy())
        all_labels.extend(batch_labels.numpy())
        all_sequences.append(batch_x.cpu().numpy())
        all_reconstructed.append(reconstructed.detach().cpu().numpy())

    all_errors = np.concatenate(all_errors, axis=0)
    all_labels = np.array(all_labels)

    # 每个样本的平均重构误差
    sample_errors = all_errors.mean(axis=1)

    # 根据阈值判定异常
    predictions = (sample_errors > threshold).astype(int)

    # 计算指标
    precision = precision_score(all_labels, predictions, zero_division=0)
    recall = recall_score(all_labels, predictions, zero_division=0)
    f1 = f1_score(all_labels, predictions, zero_division=0)
    accuracy = (predictions == all_labels).mean()

    results = {
        "errors": sample_errors,
        "labels": all_labels,
        "predictions": predictions,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

    return results


def train(model, train_loader, val_loader, cfg):
    """完整训练流程"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    if cfg.scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    history = {"train_loss": [], "val_loss": []}

    use_amp = cfg.use_amp and cfg.device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n{'='*60}")
    print("开始训练(仅使用正常数据)...")
    print(f"{'='*60}")
    print(f"设备: {cfg.device} | 优化器: Adam(lr={cfg.learning_rate}) | AMP: {use_amp}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg, amp_scaler)
        val_loss = evaluate(model, val_loader, criterion, cfg)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if cfg.scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg.epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"\n⚠ 早停触发")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(cfg.device)
        print(f"\n✓ 已恢复最优模型 (Val Loss: {best_val_loss:.6f})")

    return model, history


# ============================================================
# Step 6: 可视化函数
# ============================================================
def plot_training_curves(history, cfg):
    """绘制训练曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("自编码器重构损失")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()


def plot_reconstruction_examples(model, test_loader, cfg, num_normal=3, num_anomaly=3):
    """可视化重构结果: 正常 vs 异常"""
    model.eval()
    normal_samples = []
    anomaly_samples = []
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for batch_x, batch_labels in test_loader:
        batch_x_dev = batch_x.to(cfg.device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            reconstructed = model(batch_x_dev).cpu()

        for i in range(len(batch_labels)):
            label = batch_labels[i].item()
            sample = {
                "original": batch_x[i].squeeze().numpy(),
                "reconstructed": reconstructed[i].squeeze().detach().numpy(),
            }
            if label == 0 and len(normal_samples) < num_normal:
                normal_samples.append(sample)
            elif label == 1 and len(anomaly_samples) < num_anomaly:
                anomaly_samples.append(sample)

            if len(normal_samples) >= num_normal and len(anomaly_samples) >= num_anomaly:
                break
        if len(normal_samples) >= num_normal and len(anomaly_samples) >= num_anomaly:
            break

    fig, axes = plt.subplots(2, max(num_normal, num_anomaly), figsize=(4 * max(num_normal, num_anomaly), 8))

    for i, sample in enumerate(normal_samples):
        axes[0, i].plot(sample["original"], "b-", label="原始", linewidth=1.5)
        axes[0, i].plot(sample["reconstructed"], "r--", label="重构", linewidth=1.5)
        axes[0, i].set_title(f"正常样本 {i+1}", color="green")
        axes[0, i].legend(fontsize=8)
        axes[0, i].grid(True, alpha=0.3)

    for i, sample in enumerate(anomaly_samples):
        axes[1, i].plot(sample["original"], "b-", label="原始", linewidth=1.5)
        axes[1, i].plot(sample["reconstructed"], "r--", label="重构", linewidth=1.5)
        axes[1, i].set_title(f"异常样本 {i+1}", color="red")
        axes[1, i].legend(fontsize=8)
        axes[1, i].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("正常数据", fontsize=12, color="green")
    axes[1, 0].set_ylabel("异常数据", fontsize=12, color="red")

    plt.suptitle("LSTM自编码器重构结果对比", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "reconstruction_examples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 重构对比已保存: {save_path}")
    plt.close()


def plot_anomaly_scores(results, threshold, cfg):
    """可视化异常分数分布"""
    errors = results["errors"]
    labels = results["labels"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 异常分数散点图
    normal_mask = labels == 0
    anomaly_mask = labels == 1

    ax1.scatter(np.where(normal_mask)[0], errors[normal_mask],
                c="blue", s=10, alpha=0.5, label="正常")
    ax1.scatter(np.where(anomaly_mask)[0], errors[anomaly_mask],
                c="red", s=20, alpha=0.8, label="异常")
    ax1.axhline(y=threshold, color="orange", linestyle="--", linewidth=2,
                label=f"阈值={threshold:.4f}")
    ax1.set_xlabel("样本索引")
    ax1.set_ylabel("重构误差")
    ax1.set_title("异常分数分布")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 误差分布直方图
    ax2.hist(errors[normal_mask], bins=50, color="blue", alpha=0.5, label="正常", density=True)
    ax2.hist(errors[anomaly_mask], bins=50, color="red", alpha=0.5, label="异常", density=True)
    ax2.axvline(x=threshold, color="orange", linestyle="--", linewidth=2,
                label=f"阈值={threshold:.4f}")
    ax2.set_xlabel("重构误差")
    ax2.set_ylabel("密度")
    ax2.set_title("误差分布对比")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "anomaly_scores.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 异常分数已保存: {save_path}")
    plt.close()


# ============================================================
# Step 7: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("LSTM 时序异常检测")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # 加载数据
    print("\n生成合成传感器数据...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # 创建模型
    model = LSTMAutoencoder(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: LSTMAutoencoder")
    print(f"总参数量: {total_params:,}")
    print(f"瓶颈维度: {cfg.latent_dim}")
    print(f"\n模型结构:\n{model}")

    # 训练(仅使用正常数据)
    model, history = train(model, train_loader, val_loader, cfg)

    # 计算异常阈值
    print(f"\n{'='*60}")
    print("计算异常阈值...")
    threshold, mean_err, std_err = compute_threshold(model, train_loader, cfg)

    # 在测试集上检测异常
    print(f"\n{'='*60}")
    print("异常检测评估...")
    results = detect_anomalies(model, test_loader, threshold, cfg)

    print(f"\n检测指标:")
    print(f"  准确率:   {results['accuracy']:.4f}")
    print(f"  精确率:   {results['precision']:.4f}")
    print(f"  召回率:   {results['recall']:.4f}")
    print(f"  F1分数:   {results['f1']:.4f}")

    print("\n分类报告:")
    print(classification_report(
        results["labels"], results["predictions"],
        target_names=["正常", "异常"], digits=4,
    ))

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "lstm_autoencoder.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
        "threshold": threshold,
        "mean_error": mean_err,
        "std_error": std_err,
    }, model_path)
    print(f"✓ 模型已保存: {model_path}")

    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, cfg)
    plot_reconstruction_examples(model, test_loader, cfg)
    plot_anomaly_scores(results, threshold, cfg)

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
