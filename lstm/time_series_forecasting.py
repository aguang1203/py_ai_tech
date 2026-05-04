"""
=============================================================================
LSTM 时间序列预测任务模板 (Time Series Forecasting)
=============================================================================

【原理】
LSTM(Long Short-Term Memory)是一种特殊的循环神经网络(RNN)，通过"门控机制"
解决标准RNN的"长期依赖"问题——即无法记住长距离的上下文信息。

LSTM的核心是"细胞状态"(cell state)和三个"门"：
  - 遗忘门(Forget Gate): 决定丢弃哪些旧信息 → σ(W_f·[h_{t-1}, x_t] + b_f)
  - 输入门(Input Gate):  决定存入哪些新信息 → σ(W_i·[h_{t-1}, x_t] + b_i)
  - 输出门(Output Gate):  决定输出哪些信息 → σ(W_o·[h_{t-1}, x_t] + b_o)

信息流动：
  遗忘门 → 旧细胞状态筛选 → 输入门×候选状态 → 新细胞状态 → 输出门 → 隐藏状态

【为什么LSTM比标准RNN好？】
标准RNN的问题：
  - 梯度消失：反向传播时梯度逐层指数衰减，无法学习长距离依赖
  - 梯度爆炸：梯度逐层指数增长，训练不稳定
LSTM的解决方案：
  - 细胞状态是一条"信息高速公路"，梯度可以直接流过(加法而非乘法)
  - 门控机制让网络自己学会"什么该记、什么该忘"
  - 遗忘门接近1时，信息可以无损传递几十甚至上百步

【时间序列预测原理】
给定过去seq_len个时间步的观测值 x_{t-seq_len+1}, ..., x_t
预测未来pred_len个时间步的值 x_{t+1}, ..., x_{t+pred_len}

两种预测模式：
  1. 单步预测：只预测下一个时间步 x_{t+1}
  2. 多步预测：预测未来多个时间步 x_{t+1}, ..., x_{t+pred_len}
     - 直接多步：模型一次输出多个值(本模板使用)
     - 递归多步：用预测值作为输入，逐步向前(误差会累积)

【应用场景】
- 股票/期货价格预测
- 天气/气温预报
- 电力/能源负荷预测
- 交通流量预测
- 销售额预测
- 传感器数据预测(工业IoT)

【本数据集: 合成正弦波数据】
- 混合多个不同频率/振幅/相位的正弦波 + 高斯噪声
- 模拟真实世界中周期性+随机性的时间序列
- 即时生成，无需下载
- 特点: 保留时间序列预测任务的核心特性(趋势、周期、噪声)

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python lstm/time_series_forecasting.py
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

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置中文字体
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

now = datetime.datetime.now


# ============================================================
# Step 2: 配置超参数 (修改这里即可适配你的数据)
# ============================================================
class CONFIG:
    """超参数配置中心 —— 所有可调参数集中在此，方便统一管理和实验对比。"""

    # --- 数据相关 ---
    # seq_len=24: 输入序列长度(用过去24个时间步预测未来)
    #   为什么24？模拟每小时数据，用过去24小时(1天)预测未来
    #   如果是分钟级数据，可设为60(1小时)
    #   太短(如4): 信息不够，模型看不到完整周期
    #   太长(如200): 包含过多无关信息，训练慢且可能引入噪声
    seq_len = 24

    # pred_len=6: 预测序列长度(预测未来6个时间步)
    #   为什么6？模拟预测未来6小时
    #   预测越远，误差越大(不确定性随时间增长)
    #   一般 pred_len < seq_len/2 比较合理
    pred_len = 6

    # num_samples=2000: 生成的样本总数
    #   时间序列按滑动窗口切分，2000个时间步约产生2000-seq_len-pred_len个样本
    num_samples = 2000

    # train_ratio=0.7: 训练集比例
    #   时间序列不能随机划分！必须按时间顺序划分
    #   训练70% | 验证15% | 测试15%
    train_ratio = 0.7

    # val_ratio=0.15: 验证集比例
    val_ratio = 0.15

    # --- 模型相关 ---
    # input_dim=1: 输入特征维度
    #   单变量时间序列=1(如只有温度)
    #   多变量时间序列>1(如温度+湿度+风速)
    input_dim = 1

    # hidden_dim=64: LSTM隐藏层维度
    #   为什么64？中等复杂度的时间序列，64维足以捕获模式
    #   太小(如8): 容量不够，欠拟合
    #   太大(如256): 参数多，小数据集容易过拟合
    #   经验: hidden_dim ≈ seq_len × 2~4
    hidden_dim = 64

    # num_layers=2: LSTM堆叠层数
    #   为什么2？单层LSTM学习能力有限，2层可以提取更抽象的时序特征
    #   为什么不用3+层？LSTM深层容易过拟合，且训练困难(梯度问题)
    #   一般1-2层足够，3层以上需要大量数据
    num_layers = 2

    # dropout=0.2: LSTM层间Dropout比例
    #   为什么0.2？时序模型Dropout不宜太大(会破坏时序依赖)
    #   比CNN(0.5)小很多，因为LSTM的信息需要跨时间步传递
    #   只在多层LSTM的层间生效(num_layers>1时)
    dropout = 0.2

    # output_dim=1: 输出维度(单变量预测=1)
    output_dim = 1

    # --- 训练相关 ---
    # batch_size=32: 每次梯度更新使用的样本数
    #   为什么32？时间序列样本间有相关性，batch不宜太大
    #   太大(如256): 梯度估计可能被近期样本主导
    #   太小(如4): 训练不稳定，梯度噪声大
    batch_size = 32

    # learning_rate=1e-3: 初始学习率
    #   LSTM+Adam的标配学习率
    learning_rate = 1e-3

    # epochs=100: 最大训练轮数
    #   早停会自动控制，100是上限
    epochs = 100

    # weight_decay=1e-5: L2正则化
    #   为什么比CNN(5e-4)小？LSTM参数少，不需要强正则化
    weight_decay = 1e-5

    # --- 早停策略 ---
    # early_stop_patience=15: 验证损失连续15轮不下降就停止
    #   为什么15？LSTM收敛比CNN慢，需要更多耐心
    early_stop_patience = 15

    # --- 学习率调度器 ---
    # scheduler_type="cosine": 余弦退火调度
    #   LSTM训练波动大，余弦退火比ReduceLROnPlateau更稳定
    scheduler_type = "cosine"

    # --- 梯度裁剪 ---
    # max_grad_norm=1.0: 梯度L2范数上限
    #   为什么1.0？LSTM对梯度裁剪非常敏感
    #   比CNN(5.0)小很多，因为RNN/LSTM的梯度容易爆炸
    #   1.0是LSTM的标准值，几乎所有LSTM论文都用1.0
    max_grad_norm = 1.0

    # --- 混合精度训练(AMP) ---
    # use_amp=True: 启用自动混合精度
    #   仅CUDA(GPU)有效，CPU自动降级
    use_amp = True

    # --- 数据加载优化 ---
    # num_workers: DataLoader的子进程数
    num_workers = min(4, os.cpu_count() or 1)

    # --- 数据标准化 ---
    # scaler_type="minmax": 标准化方式
    #   "minmax": 缩放到[0,1]，适合有界数据
    #   "standard": 减均值除标准差，适合无界数据
    #   为什么用标准化？LSTM对输入范围敏感，标准化加速收敛
    scaler_type = "minmax"

    # --- 保存相关 ---
    save_dir = "lstm/output/forecasting"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 数据生成与加载
# ============================================================
def generate_time_series(cfg):
    """
    生成合成时间序列数据。

    【合成策略】
    真实时间序列通常包含三部分：
      1. 趋势(Trend): 长期上升/下降趋势
      2. 周期(Seasonality): 周期性波动
      3. 噪声(Noise): 随机扰动

    本合成数据模拟：
      y(t) = 趋势 + 周期1 + 周期2 + 噪声
      = 0.01t + 2sin(2πt/24) + 0.5sin(2πt/6) + ε

    - 0.01t: 缓慢上升趋势(模拟经济增长/通胀)
    - 2sin(2πt/24): 主周期，周期=24(模拟日周期)
    - 0.5sin(2πt/6): 次周期，周期=6(模拟4小时一次的子周期)
    - ε ~ N(0, 0.1): 高斯噪声(模拟随机波动)
    """
    np.random.seed(42)
    t = np.arange(cfg.num_samples, dtype=np.float32)

    # 趋势项: 缓慢线性上升
    trend = 0.01 * t

    # 周期项: 多个不同频率的正弦波叠加
    # 周期1: 周期=24(日周期)，振幅=2(主要波动)
    seasonality1 = 2.0 * np.sin(2 * np.pi * t / 24.0)
    # 周期2: 周期=6(子周期)，振幅=0.5(次要波动)
    seasonality2 = 0.5 * np.sin(2 * np.pi * t / 6.0)

    # 噪声项: 高斯白噪声
    noise = np.random.normal(0, 0.1, cfg.num_samples).astype(np.float32)

    # 合成
    data = trend + seasonality1 + seasonality2 + noise

    return data


class TimeSeriesDataset(Dataset):
    """
    时间序列数据集：用滑动窗口将时间序列切分为(输入, 输出)对。

    【滑动窗口原理】
    时间序列: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]
    seq_len=3, pred_len=1:
      样本1: 输入[0,1,2] → 输出[3]
      样本2: 输入[1,2,3] → 输出[4]
      样本3: 输入[2,3,4] → 输出[5]
      ...

    【为什么用滑动窗口？】
    - 将无限长的时序数据变为有限的监督学习样本
    - 每个样本: 过去seq_len步 → 未来pred_len步
    - 滑动步长=1(最大化利用数据)
    """

    def __init__(self, data, seq_len, pred_len):
        """
        参数:
            data: numpy数组，标准化后的时间序列
            seq_len: 输入序列长度
            pred_len: 预测序列长度
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        # 可生成的样本数 = 总长度 - 输入长度 - 预测长度 + 1
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # 输入: [idx, idx+seq_len)
        x = self.data[idx:idx + self.seq_len]
        # 输出: [idx+seq_len, idx+seq_len+pred_len)
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]

        # 转为Tensor: (seq_len,) → (seq_len, input_dim)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        # (pred_len,) → (pred_len, output_dim)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        return x, y


class DataScaler:
    """
    数据标准化器。

    【为什么要标准化？】
    - LSTM对输入范围敏感，大数值导致梯度爆炸
    - 标准化后数据在0附近，激活函数(门控sigmoid/tanh)工作在线性区
    - 加速收敛：标准化前可能需要1000轮，标准化后50轮就够
    """

    def __init__(self, method="minmax"):
        """
        参数:
            method: "minmax"或"standard"
        """
        self.method = method
        self.min_ = None
        self.max_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, data):
        """在训练数据上计算统计量"""
        if self.method == "minmax":
            self.min_ = data.min()
            self.max_ = data.max()
        else:
            self.mean_ = data.mean()
            self.std_ = data.std()
        return self

    def transform(self, data):
        """应用标准化"""
        if self.method == "minmax":
            # 缩放到[0, 1]
            # 避免除零：如果max==min(常数序列)，直接返回0
            denom = self.max_ - self.min_
            if denom == 0:
                return data - self.min_
            return (data - self.min_) / denom
        else:
            # 标准化: (x - μ) / σ
            if self.std_ == 0:
                return data - self.mean_
            return (data - self.mean_) / self.std_

    def inverse_transform(self, data):
        """反标准化(将预测结果还原为原始尺度)"""
        if self.method == "minmax":
            return data * (self.max_ - self.min_) + self.min_
        else:
            return data * self.std_ + self.mean_


def get_dataloaders(cfg):
    """
    生成数据并创建DataLoader。

    【时序数据的划分原则】
    ⚠️ 时间序列绝不能随机划分！必须按时间顺序划分：
      训练集: 最早的70%数据(学习历史模式)
      验证集: 中间的15%数据(监控过拟合)
      测试集: 最新的15%数据(评估未来预测能力)

    为什么？
    - 随机划分会导致"未来信息泄露"：用未来数据训练，预测过去
    - 这在现实中不可能(你不可能用明天的数据预测今天)
    - 随机划分的精度会虚高，但实际部署效果差
    """
    # 生成时间序列
    data = generate_time_series(cfg)
    print(f"生成时间序列: {len(data)}个时间步")

    # 按时间顺序划分
    n = len(data)
    train_end = int(n * cfg.train_ratio)
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # 标准化(只在训练集上fit！)
    # 【为什么只在训练集上fit？】
    # 验证集和测试集代表"未来数据"，训练时不能看到
    # 如果用全部数据fit，相当于偷看了未来的分布信息
    scaler = DataScaler(method=cfg.scaler_type)
    scaler.fit(train_data)

    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # 创建Dataset
    train_dataset = TimeSeriesDataset(train_scaled, cfg.seq_len, cfg.pred_len)
    val_dataset = TimeSeriesDataset(val_scaled, cfg.seq_len, cfg.pred_len)
    test_dataset = TimeSeriesDataset(test_scaled, cfg.seq_len, cfg.pred_len)

    # 创建DataLoader
    pin_mem = cfg.device.type == "cuda"
    pw = cfg.num_workers > 0

    # 【时序数据要不要shuffle？】
    # 训练集: shuffle=True(样本间已无时序依赖，打乱防梯度相关)
    # 验证/测试集: shuffle=False(保持原始顺序，方便可视化)
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

    print(f"训练集: {len(train_dataset)}样本 | 验证集: {len(val_dataset)}样本 | 测试集: {len(test_dataset)}样本")

    return train_loader, val_loader, test_loader, scaler, test_data


# ============================================================
# Step 4: 模型定义
# ============================================================
class LSTMForecaster(nn.Module):
    """
    LSTM时间序列预测模型。

    【架构设计】
    输入 (batch, seq_len, input_dim)  ← 过去seq_len步的观测值
      → LSTM(多层) → 最后时间步的隐藏状态 (batch, hidden_dim)
      → FC(hidden_dim → pred_len * output_dim) → reshape
      → 输出 (batch, pred_len, output_dim)  ← 未来pred_len步的预测值

    【为什么取最后时间步的隐藏状态？】
    - LSTM的隐藏状态h_t编码了从t_0到t_t的所有历史信息
    - h_{seq_len-1}编码了完整的输入序列信息
    - 用它作为整个序列的"摘要"来预测未来

    【多步预测的两种策略】
    1. 直接多步(本模板): FC一次输出pred_len个值
       - 优点: 训练简单，推理快，无误差累积
       - 缺点: 预测越远精度越低(模型一次要学很多)
    2. 递归多步: 预测1步 → 将预测值拼入输入 → 再预测1步 → ...
       - 优点: 每次只预测1步，模型更简单
       - 缺点: 误差累积(一步错，步步错)，推理慢(pred_len次前向传播)
    """

    def __init__(self, cfg):
        super().__init__()

        # LSTM层
        # batch_first=True: 输入形状为(batch, seq_len, input_dim)
        #   为什么用batch_first？更直观，与DataLoader输出一致
        self.lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
        )

        # 全连接层: 将隐藏状态映射到预测值
        # 输出维度 = pred_len * output_dim，之后reshape为(pred_len, output_dim)
        self.fc = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim * 2, cfg.pred_len * cfg.output_dim),
        )

        # 保存配置
        self.pred_len = cfg.pred_len
        self.output_dim = cfg.output_dim

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        权重初始化。

        【LSTM权重初始化的特殊考虑】
        - 遗忘门偏置初始化为1(而非0)：让遗忘门初始接近1，保留更多信息
        - 为什么？初始时细胞状态为空，如果遗忘门=0，什么信息都不保留
        - 设为1让遗忘门≈0.73(σ(1))，偏向于"先保留，再学习该忘什么"
        - 其他权重用Xavier初始化
        """
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                # 输入-隐藏权重: Xavier初始化
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                # 隐藏-隐藏权重: 正交初始化(有助于梯度流动)
                nn.init.orthogonal_(param)
            elif "bias" in name:
                # 偏置初始化
                if "lstm" in name:
                    # LSTM偏置: 遗忘门偏置=1，其他=0
                    # LSTM偏置顺序: [input, forget, cell, output]
                    # 每个门有hidden_dim个偏置
                    hidden_dim = param.shape[0] // 4
                    param.data.fill_(0)
                    param.data[hidden_dim:2 * hidden_dim].fill_(1.0)  # 遗忘门偏置=1
                else:
                    nn.init.constant_(param, 0)

    def forward(self, x):
        """
        前向传播

        参数:
            x: (batch, seq_len, input_dim) 输入序列
        返回:
            pred: (batch, pred_len, output_dim) 预测序列
        """
        # LSTM前向传播
        # out: (batch, seq_len, hidden_dim) - 每个时间步的输出
        # (h_n, c_n): 最后时间步的隐藏状态和细胞状态
        out, (h_n, c_n) = self.lstm(x)

        # 取最后一个时间步的输出
        # 【为什么用out[:, -1, :]而不是h_n？】
        # 当batch_first=True时，out[:, -1, :]和h_n[-1]是等价的
        # 这里用out[:, -1, :]更直观
        last_out = out[:, -1, :]  # (batch, hidden_dim)

        # 全连接层 → 预测值
        pred = self.fc(last_out)  # (batch, pred_len * output_dim)

        # reshape为(batch, pred_len, output_dim)
        pred = pred.view(-1, self.pred_len, self.output_dim)

        return pred


# ============================================================
# Step 5: 训练函数
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, cfg, scaler=None):
    """
    训练一个epoch。

    【LSTM训练的注意事项】
    1. 隐藏状态初始化：每个batch重新初始化(不跨batch传递)
       为什么？不同样本间没有时序关系，传递隐藏状态没意义
    2. 梯度裁剪：LSTM必须裁剪，1.0是标配
    3. batch_first=True：输入形状(batch, seq, features)
    """
    model.train()
    total_loss = 0
    n_batches = 0
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(cfg.device)
        batch_y = batch_y.to(cfg.device)

        # 前向传播(混合精度)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(batch_x)
            loss = criterion(pred, batch_y)

        # 反向传播
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            # 梯度裁剪: 先unscale再裁剪
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
    """
    评估模型性能。

    返回:
        avg_loss: 平均损失
        all_preds: 所有预测值(numpy)
        all_targets: 所有真实值(numpy)
    """
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_targets = []
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(cfg.device)
        batch_y = batch_y.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(batch_x)
            loss = criterion(pred, batch_y)

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(pred.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return total_loss / n_batches, all_preds, all_targets


def train(model, train_loader, val_loader, cfg):
    """完整训练流程: 训练 + 验证 + 早停 + 学习率调度"""
    # 损失函数: MSELoss(均方误差)
    # 【为什么用MSE而不是MAE？】
    # MSE对大误差惩罚更重，模型会更努力避免大的预测偏差
    # MAE对所有误差一视同仁，对异常值更鲁棒
    # 时间序列预测通常用MSE作为训练损失，MAE作为辅助评估
    criterion = nn.MSELoss()

    # 优化器
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    # 学习率调度器
    if cfg.scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )

    # 早停相关
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # 训练记录
    history = {"train_loss": [], "val_loss": []}

    # 混合精度GradScaler
    use_amp = cfg.use_amp and cfg.device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")
    print(f"设备: {cfg.device} | 优化器: Adam(lr={cfg.learning_rate}) | AMP: {use_amp}")

    for epoch in range(1, cfg.epochs + 1):
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg, amp_scaler)

        # 验证
        val_loss, _, _ = evaluate(model, val_loader, criterion, cfg)

        # 记录
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # 更新学习率
        if cfg.scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # 打印
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg.epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.6f}")

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"\n⚠ 早停触发: 验证损失连续{cfg.early_stop_patience}轮未改善")
                break

    # 恢复最优模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(cfg.device)
        print(f"\n✓ 已恢复最优模型 (Val Loss: {best_val_loss:.6f})")

    return model, history


# ============================================================
# Step 6: 可视化函数
# ============================================================
def plot_time_series(data, cfg, title="合成时间序列数据"):
    """可视化原始时间序列数据"""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(data, linewidth=0.8, alpha=0.8)
    ax.set_xlabel("时间步")
    ax.set_ylabel("数值")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # 标注训练/验证/测试划分
    n = len(data)
    train_end = int(n * cfg.train_ratio)
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))
    ax.axvline(x=train_end, color="r", linestyle="--", alpha=0.5, label="训练/验证分界")
    ax.axvline(x=val_end, color="g", linestyle="--", alpha=0.5, label="验证/测试分界")
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "time_series_data.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 时间序列数据已保存: {save_path}")
    plt.close()


def plot_training_curves(history, cfg):
    """绘制训练曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("训练/验证损失曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")  # 对数刻度，更清楚看到收敛过程

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()


def plot_predictions(model, test_loader, scaler, cfg, num_samples=5):
    """
    可视化预测结果：展示输入序列 + 真实值 + 预测值。

    【如何解读预测图？】
    - 蓝色实线: 输入序列(模型"看到"的历史数据)
    - 绿色实线: 真实值(实际发生的)
    - 红色虚线: 预测值(模型预测的)
    - 预测值和真实值越接近，模型越好
    - 通常近处预测比远处准确(不确定性随时间增长)
    """
    model.eval()
    # 取一批数据
    batch_x, batch_y = next(iter(test_loader))
    batch_x = batch_x[:num_samples].to(cfg.device)
    batch_y = batch_y[:num_samples]

    with torch.no_grad():
        pred = model(batch_x).cpu().numpy()

    batch_x = batch_x.cpu().numpy()
    batch_y = batch_y.numpy()

    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        ax = axes[i]

        # 反标准化
        input_seq = scaler.inverse_transform(batch_x[i].flatten())
        true_seq = scaler.inverse_transform(batch_y[i].flatten())
        pred_seq = scaler.inverse_transform(pred[i].flatten())

        # 绘制输入序列
        input_x = np.arange(len(input_seq))
        ax.plot(input_x, input_seq, "b-", linewidth=1.5, label="输入序列")

        # 绘制真实值
        true_x = np.arange(len(input_seq), len(input_seq) + len(true_seq))
        ax.plot(true_x, true_seq, "g-", linewidth=2, label="真实值")

        # 绘制预测值
        ax.plot(true_x, pred_seq, "r--", linewidth=2, label="预测值")

        # 连接输入和真实/预测
        ax.plot([len(input_seq) - 1, len(input_seq)],
                [input_seq[-1], true_seq[0]], "g-", linewidth=1, alpha=0.5)
        ax.plot([len(input_seq) - 1, len(input_seq)],
                [input_seq[-1], pred_seq[0]], "r--", linewidth=1, alpha=0.5)

        ax.set_title(f"样本 {i+1}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("时间步")

    plt.suptitle("LSTM时间序列预测结果", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 预测结果已保存: {save_path}")
    plt.close()


def plot_error_analysis(all_preds, all_targets, scaler, cfg):
    """
    预测误差分析图。

    【分析内容】
    1. 各预测步的误差: 第1步、第2步...第pred_len步的误差
       通常误差随预测步增加而增大(越远越不准)
    2. 误差分布直方图: 误差是否正态分布(模型是否系统偏差)
    """
    # 反标准化
    preds_original = scaler.inverse_transform(all_preds.flatten())
    targets_original = scaler.inverse_transform(all_targets.flatten())

    # 计算各步误差
    pred_len = all_preds.shape[1]
    step_errors = []
    for step in range(pred_len):
        pred_step = scaler.inverse_transform(all_preds[:, step, 0])
        true_step = scaler.inverse_transform(all_targets[:, step, 0])
        mse = mean_squared_error(true_step, pred_step)
        mae = mean_absolute_error(true_step, pred_step)
        step_errors.append((mse, mae))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 各步误差
    steps = range(1, pred_len + 1)
    mses = [e[0] for e in step_errors]
    maes = [e[1] for e in step_errors]
    ax1.bar(steps, mses, color="steelblue", alpha=0.7, label="MSE")
    ax1.bar(steps, maes, color="coral", alpha=0.7, label="MAE")
    ax1.set_xlabel("预测步数")
    ax1.set_ylabel("误差")
    ax1.set_title("各预测步的误差(反标准化后)")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # 误差分布
    errors = preds_original - targets_original
    ax2.hist(errors, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax2.axvline(x=0, color="r", linestyle="--", alpha=0.5)
    ax2.set_xlabel("预测误差")
    ax2.set_ylabel("频次")
    ax2.set_title("预测误差分布")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "error_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 误差分析已保存: {save_path}")
    plt.close()

    # 打印各步误差
    print("\n各预测步误差(反标准化后):")
    for step, (mse, mae) in enumerate(step_errors, 1):
        print(f"  第{step}步: MSE={mse:.4f}, MAE={mae:.4f}")


# ============================================================
# Step 7: 预测函数
# ============================================================
@torch.no_grad()
def predict(model, sequence, cfg, scaler=None):
    """
    对单个序列进行预测。

    参数:
        sequence: numpy数组 (seq_len,) 或 (seq_len, input_dim)
    返回:
        pred: 预测值(numpy数组, 反标准化后)
    """
    model.eval()

    # 转为Tensor
    if sequence.ndim == 1:
        sequence = sequence.reshape(-1, 1)
    x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(cfg.device)

    # 预测
    pred = model(x).cpu().numpy().flatten()

    # 反标准化
    if scaler is not None:
        pred = scaler.inverse_transform(pred)

    return pred


# ============================================================
# Step 8: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("LSTM 时间序列预测")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 创建输出目录
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 加载数据
    print("\n生成合成时间序列数据...")
    train_loader, val_loader, test_loader, scaler, test_data = get_dataloaders(cfg)

    # 可视化原始数据
    full_data = generate_time_series(cfg)
    plot_time_series(full_data, cfg)

    # 创建模型
    model = LSTMForecaster(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型: LSTMForecaster")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"\n模型结构:\n{model}")

    # 训练
    model, history = train(model, train_loader, val_loader, cfg)

    # 在测试集上评估
    print(f"\n{'='*60}")
    print("测试集评估...")
    criterion = nn.MSELoss()
    test_loss, all_preds, all_targets = evaluate(model, test_loader, criterion, cfg)

    # 反标准化后计算指标
    preds_orig = scaler.inverse_transform(all_preds.flatten())
    targets_orig = scaler.inverse_transform(all_targets.flatten())

    mse = mean_squared_error(targets_orig, preds_orig)
    mae = mean_absolute_error(targets_orig, preds_orig)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_orig, preds_orig)

    print(f"测试集 MSE:  {mse:.4f}")
    print(f"测试集 RMSE: {rmse:.4f}")
    print(f"测试集 MAE:  {mae:.4f}")
    print(f"测试集 R²:   {r2:.4f}")

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "lstm_forecaster.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
        "scaler_min": scaler.min_,
        "scaler_max": scaler.max_,
    }, model_path)
    print(f"✓ 模型已保存: {model_path}")

    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, cfg)
    plot_predictions(model, test_loader, scaler, cfg)
    plot_error_analysis(all_preds, all_targets, scaler, cfg)

    # 演示单次预测
    print("\n单次预测演示:")
    test_seq = test_data[cfg.seq_len:cfg.seq_len * 2]
    test_seq_scaled = scaler.transform(test_seq)
    pred = predict(model, test_seq_scaled, cfg, scaler)
    true_val = test_data[cfg.seq_len * 2:cfg.seq_len * 2 + cfg.pred_len]
    print(f"  输入: 最近{cfg.seq_len}个时间步")
    print(f"  真实值: {true_val[:cfg.pred_len]}")
    print(f"  预测值: {pred[:cfg.pred_len]}")

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
