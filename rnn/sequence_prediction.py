"""
=============================================================================
RNN 时间序列预测任务模板 (LSTM for Time Series Forecasting)
=============================================================================

【原理】
时间序列预测是RNN最自然的应用——数据本身就是有序的序列，预测未来的值需要
理解过去的趋势和模式。LSTM通过记忆机制，能够捕捉长期依赖关系。

核心思想：
  给定过去的序列 [x_1, x_2, ..., x_t]，预测未来的值 [x_{t+1}, ..., x_{t+k}]

两种预测模式：
1. 单步预测 (One-step): 只预测下一个时间步的值
   输入: [x_1, ..., x_t] → 输出: x_{t+1}

2. 多步预测 (Multi-step): 预测未来多个时间步的值
   输入: [x_1, ..., x_t] → 输出: [x_{t+1}, ..., x_{t+k}]
   本模板使用多步预测

【为什么LSTM适合时间序列？】
- 时间序列有长期依赖: 今天的温度受过去几天影响
- LSTM的遗忘门决定"忘记"不相关的历史，保留有用的趋势
- LSTM的输入门决定"记住"新的观测，更新对趋势的判断
- 细胞状态作为"信息高速公路"，长期信息可以几乎无损传递

【本数据集: 合成正弦波时间序列】
- 合成带有不同频率和噪声的正弦波数据
- 模拟真实时间序列的周期性和随机性
- 使用过去30个时间步预测未来10个时间步

【应用场景】
- 股票/汇率预测
- 天气/温度预测
- 电力负荷预测
- 交通流量预测
- 设备故障预测

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python rnn/sequence_prediction.py
3. 合成数据自动生成，无需下载
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
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    """超参数配置中心 —— 所有可调参数集中在此。"""

    # --- 数据相关 ---
    # input_size=1: 每个时间步的输入特征数
    #   单变量时间序列: 只有1个值(如温度)
    #   多变量时间序列: 有多个值(如温度+湿度+风速)
    input_size = 1

    # seq_length=30: 输入序列长度(回看窗口)
    #   用过去30个时间步来预测未来
    #   为什么30？正弦波周期约60步，30步覆盖半个周期
    #   实际应用: 根据数据特性选择，太短看不到趋势，太长引入噪音
    seq_length = 30

    # pred_length=10: 预测序列长度(前瞻窗口)
    #   预测未来10个时间步
    #   预测越远越不准，10步是合理范围
    pred_length = 10

    # num_samples=2000: 合成数据总样本数
    num_samples = 2000

    # test_size=0.2: 测试集比例
    test_size = 0.2

    # random_state=42: 随机种子
    random_state = 42

    # --- 合成数据参数 ---
    # sine_freq=0.1: 正弦波频率
    #   周期 = 2π/0.1 ≈ 63个时间步
    sine_freq = 0.1

    # noise_level=0.1: 噪声水平
    #   添加高斯噪声模拟真实数据的随机性
    noise_level = 0.1

    # --- 模型相关 ---
    # rnn_type="lstm": RNN类型
    rnn_type = "lstm"

    # hidden_size=64: 隐藏状态维度
    #   为什么64？正弦波模式简单，64足够
    #   真实金融/气象数据: 建议128-256
    hidden_size = 64

    # num_layers=2: RNN层数
    num_layers = 2

    # dropout_rate=0.2: Dropout比例
    #   时间序列预测对Dropout很敏感，0.1-0.2通常就够
    dropout_rate = 0.2

    # --- 训练相关 ---
    batch_size = 64

    learning_rate = 1e-3

    epochs = 50

    weight_decay = 1e-5

    # --- 早停策略 ---
    early_stop_patience = 10

    # --- 学习率调度器 ---
    scheduler_type = "cosine"

    lr_step_size = 15

    lr_gamma = 0.5

    # --- 梯度裁剪 ---
    max_grad_norm = 1.0

    # --- 混合精度训练(AMP) ---
    use_amp = True

    # --- 数据加载优化 ---
    num_workers = min(2, os.cpu_count() or 1)

    # --- 保存相关 ---
    save_dir = "rnn/output/sequence_prediction"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 合成数据和数据加载
# ============================================================
def generate_sine_data(cfg):
    """
    生成合成正弦波时间序列数据。

    【数据生成原理】
    基础信号: y = sin(freq * t)
    加上噪声: y = sin(freq * t) + noise_level * N(0,1)

    这样生成的数据有:
    - 周期性: 正弦波的基本特征
    - 随机性: 噪声模拟真实数据
    - 长期依赖: 某个时刻的值受之前多个时刻影响

    【滑动窗口切分】
    将长序列按滑动窗口切分为 (输入序列, 目标序列) 对:
    [x_1, x_2, ..., x_30] → [x_31, x_32, ..., x_40]
    [x_2, x_3, ..., x_31] → [x_32, x_33, ..., x_41]
    ...

    这是最常用的时间序列数据准备方法
    """
    np.random.seed(cfg.random_state)

    # 生成足够长的正弦波序列
    total_length = cfg.seq_length + cfg.pred_length + cfg.num_samples + 100
    t = np.arange(total_length)
    # 正弦波 + 随机噪声
    data = np.sin(cfg.sine_freq * t) + cfg.noise_level * np.random.randn(total_length)

    # 滑动窗口切分
    X, y = [], []
    for i in range(len(data) - cfg.seq_length - cfg.pred_length + 1):
        X.append(data[i:i + cfg.seq_length])
        y.append(data[i + cfg.seq_length:i + cfg.seq_length + cfg.pred_length])

    X = np.array(X, dtype=np.float32)  # (num_windows, seq_length)
    y = np.array(y, dtype=np.float32)  # (num_windows, pred_length)

    # 取前num_samples个样本
    X = X[:cfg.num_samples]
    y = y[:cfg.num_samples]

    # 标准化 (使用训练集的统计量)
    # 【为什么要标准化？】
    # LSTM对输入范围敏感，标准化后训练更稳定
    # 关键: 必须用训练集的统计量，否则数据泄露
    n_train = int(len(X) * (1 - cfg.test_size))
    train_mean = X[:n_train].mean()
    train_std = X[:n_train].std()
    train_std = max(train_std, 1e-8)  # 防止除零

    X = (X - train_mean) / train_std
    y = (y - train_mean) / train_std

    return X, y, train_mean, train_std


class TimeSeriesDataset(Dataset):
    """时间序列预测数据集。"""

    def __init__(self, X, y):
        """
        参数:
            X: 输入序列 (num_samples, seq_length)
            y: 目标序列 (num_samples, pred_length)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # RNN输入需要3D: (seq_length, input_size)
        x = self.X[idx].unsqueeze(-1)  # (seq_length, 1)
        y = self.y[idx]
        return x, y


def get_dataloaders(cfg):
    """生成合成数据并创建DataLoader。"""
    X, y, mean, std = generate_sine_data(cfg)
    cfg.train_mean = mean
    cfg.train_std = std

    # 划分训练/验证/测试集
    n_total = len(X)
    n_test = int(n_total * cfg.test_size)
    n_val = int(n_total * cfg.test_size)
    n_train = n_total - n_val - n_test

    train_dataset = TimeSeriesDataset(X[:n_train], y[:n_train])
    val_dataset = TimeSeriesDataset(X[n_train:n_train + n_val], y[n_train:n_train + n_val])
    test_dataset = TimeSeriesDataset(X[n_train + n_val:], y[n_train + n_val:])

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

    print(f"训练集: {n_train}个窗口 | 验证集: {n_val}个窗口 | 测试集: {n_test}个窗口")
    print(f"输入形状: ({cfg.seq_length}, {cfg.input_size}) | 预测长度: {cfg.pred_length}")

    return train_loader, val_loader, test_loader


# ============================================================
# Step 4: 模型定义
# ============================================================
class SequencePredictor(nn.Module):
    """
    RNN时间序列预测模型 (LSTM)

    【架构设计思路】
    输入: (batch, seq_length, 1)  ← 过去的序列值
      → LSTM Layer1 (1→64)
      → LSTM Layer2 (64→64)
      → 取最后时间步: (batch, 64)
      → FC(64→pred_length): (batch, pred_length)

    【为什么全连接层直接输出pred_length个值？】
    - 这是最简单的多步预测方法: 一次输出所有预测值
    - 优点: 训练简单，推理快
    - 缺点: 预测步数固定，不能动态调整
    - 替代方案: Seq2Seq(编码器-解码器)，可变长预测

    【维度变化详解】
    输入: (batch, 30, 1)  ← 30个历史值
      → LSTM: (batch, 30, 64)  ← 每个时间步64维隐藏状态
      → 取最后时间步: (batch, 64)  ← 聚合了整个序列信息
      → FC: (batch, 10)  ← 未来10步的预测值

    【参数量计算 (LSTM)】
    第1层: 4 × [(1 + 64 + 1) × 64] = 16,896
    第2层: 4 × [(64 + 64 + 1) × 64] = 33,024
    FC层: 64×10 + 10 = 650
    总计 ≈ 50K (非常轻量)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_layers

        # ---- RNN层 ----
        if cfg.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=cfg.input_size,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=cfg.dropout_rate if cfg.num_layers > 1 else 0,
            )
        elif cfg.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=cfg.input_size,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=cfg.dropout_rate if cfg.num_layers > 1 else 0,
            )

        # ---- 全连接预测头 ----
        # 直接输出pred_length个预测值
        self.fc = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(cfg.hidden_size, cfg.pred_length),
        )

        self._init_weights()

    def _init_weights(self):
        """权重初始化。"""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                if isinstance(self.rnn, nn.LSTM):
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1.0)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入序列 (batch, seq_length, input_size)

        数据流动:
        x: (batch, 30, 1)            ← 30个历史值
          → rnn: (batch, 30, 64)
          → 取最后时间步: (batch, 64)
          → fc: (batch, 10)           ← 未来10步预测
        """
        batch_size = x.size(0)

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros_like(h0)
            rnn_out, _ = self.rnn(x, (h0, c0))
        else:
            rnn_out, _ = self.rnn(x, h0)

        # 取最后时间步
        out = rnn_out[:, -1, :]  # (batch, hidden_size)

        # 预测未来值
        out = self.fc(out)  # (batch, pred_length)

        return out


# ============================================================
# Step 5: 训练函数
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, cfg, scaler=None):
    """训练一个epoch。"""
    model.train()
    total_loss = 0
    total = 0

    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for inputs, targets in loader:
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

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

        total_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)

    avg_loss = total_loss / total
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, criterion, cfg):
    """评估模型性能。"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for inputs, targets in loader:
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / sum(t.shape[0] for t in all_targets)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 计算指标
    mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
    rmse = np.sqrt(mean_squared_error(all_targets.flatten(), all_preds.flatten()))

    return avg_loss, mae, rmse, all_preds, all_targets


def train(model, train_loader, val_loader, cfg):
    """完整训练流程。"""
    # 损失函数: MSE(均方误差)
    # 【为什么时间序列用MSE而非CrossEntropy？】
    # 时间序列预测是回归任务(输出连续值)，不是分类任务(输出类别)
    # MSE: 对大误差惩罚更重，适合预测任务
    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    if cfg.scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}

    use_amp = cfg.use_amp and cfg.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")
    print(f"设备: {cfg.device} | 优化器: Adam(lr={cfg.learning_rate}) | AMP: {use_amp}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg, scaler)
        val_loss, val_mae, val_rmse, _, _ = evaluate(model, val_loader, criterion, cfg)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{cfg.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | "
              f"LR: {current_lr:.6f}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ✓ 最优模型已更新 (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"\n⚠ 早停触发: 验证损失连续{cfg.early_stop_patience}轮未改善")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(cfg.device)
        print(f"\n✓ 已恢复最优模型 (Val Loss: {best_val_loss:.4f})")

    return model, history


# ============================================================
# Step 6: 可视化函数
# ============================================================
def plot_training_curves(history, cfg):
    """绘制训练曲线。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("训练/验证损失曲线")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_mae"], "g-", label="Val MAE", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("验证集MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["val_rmse"], "m-", label="Val RMSE", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("RMSE")
    axes[2].set_title("验证集RMSE")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()


def plot_predictions(model, test_loader, cfg, num_samples=5):
    """
    可视化预测结果。

    【如何解读预测图？】
    - 蓝色实线: 输入序列(已知的历史值)
    - 绿色实线: 真实的未来值
    - 红色虚线: 模型的预测值
    - 预测线与真实线越接近越好
    - 通常预测越远的步(线末端)误差越大
    """
    model.eval()
    inputs, targets = next(iter(test_loader))
    inputs, targets = inputs[:num_samples].to(cfg.device), targets[:num_samples]

    with torch.no_grad():
        preds = model(inputs).cpu().numpy()

    inputs = inputs.cpu().numpy()
    targets = targets.numpy()

    # 反标准化
    mean, std = cfg.train_mean, cfg.train_std
    inputs_real = inputs.squeeze(-1) * std + mean
    targets_real = targets * std + mean
    preds_real = preds * std + mean

    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        ax = axes[i]
        # 输入序列
        input_x = np.arange(cfg.seq_length)
        ax.plot(input_x, inputs_real[i], "b-o", markersize=3, label="输入(历史)", linewidth=2)

        # 真实未来
        future_x = np.arange(cfg.seq_length, cfg.seq_length + cfg.pred_length)
        ax.plot(future_x, targets_real[i], "g-o", markersize=3, label="真实值", linewidth=2)

        # 预测未来
        ax.plot(future_x, preds_real[i], "r--o", markersize=3, label="预测值", linewidth=2)

        # 分割线
        ax.axvline(x=cfg.seq_length - 0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"样本 {i+1}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_xlabel("时间步")

    plt.suptitle("RNN时间序列预测结果", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 预测结果已保存: {save_path}")
    plt.close()


def plot_error_by_horizon(all_targets, all_preds, cfg):
    """
    绘制不同预测步的误差分布。

    【为什么要看各步误差？】
    - 近期预测通常更准确，远期预测误差增大
    - 这帮助确定模型的有效预测范围
    - 如果第5步后误差剧增，说明模型只适合短期预测
    """
    # 计算每个预测步的MAE
    maes = []
    for step in range(cfg.pred_length):
        mae = np.mean(np.abs(all_targets[:, step] - all_preds[:, step]))
        maes.append(mae)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, cfg.pred_length + 1), maes, color="steelblue", alpha=0.8)
    ax.set_xlabel("预测步数")
    ax.set_ylabel("MAE (标准化)")
    ax.set_title("各预测步的误差分布")
    ax.grid(True, alpha=0.3, axis="y")

    for i, mae in enumerate(maes):
        ax.text(i + 1, mae + 0.001, f"{mae:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "error_by_horizon.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 各步误差图已保存: {save_path}")
    plt.close()


# ============================================================
# Step 7: 预测函数
# ============================================================
@torch.no_grad()
def predict_future(model, history_sequence, cfg):
    """
    给定历史序列，预测未来值。

    参数:
        history_sequence: 历史值数组 (seq_length,) 或 (seq_length, 1)
    返回:
        predictions: 预测的未来值 (pred_length,)
    """
    model.eval()

    if isinstance(history_sequence, np.ndarray):
        # 标准化
        history_sequence = (history_sequence - cfg.train_mean) / cfg.train_std
        x = torch.tensor(history_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    else:
        x = history_sequence

    x = x.to(cfg.device)
    output = model(x).cpu().numpy().flatten()

    # 反标准化
    predictions = output * cfg.train_std + cfg.train_mean
    return predictions


# ============================================================
# Step 8: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("RNN 时间序列预测 - 合成正弦波数据")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # 加载数据
    print("\n生成合成数据...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # 创建模型
    model = SequencePredictor(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型: SequencePredictor ({cfg.rnn_type.upper()})")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"\n模型结构:\n{model}")

    # 训练
    model, history = train(model, train_loader, val_loader, cfg)

    # 在测试集上评估
    print(f"\n{'='*60}")
    print("测试集评估...")
    criterion = nn.MSELoss()
    test_loss, test_mae, test_rmse, all_preds, all_targets = evaluate(
        model, test_loader, criterion, cfg,
    )
    print(f"测试集 MSE: {test_loss:.4f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f}")

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "sequence_predictor.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
        "history": history,
    }, model_path)
    print(f"✓ 模型已保存: {model_path}")

    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, cfg)
    plot_predictions(model, test_loader, cfg)
    plot_error_by_horizon(all_targets, all_preds, cfg)

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
