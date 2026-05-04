"""
=============================================================================
RNN 序列分类任务模板 (Recurrent Neural Network for Sequence Classification)
=============================================================================

【原理】
循环神经网络(RNN)是专门处理**序列数据**的深度学习模型。与CNN处理空间结构不同，
RNN处理的是时间结构——数据有先后顺序，当前输出依赖于之前的输入。

核心思想：网络在时间步之间共享参数，通过"隐藏状态"(hidden state)传递历史信息，
相当于网络拥有了"记忆"。

RNN的典型架构：输入序列 → [RNN/LSTM/GRU] → 隐藏状态 → 全连接层 → 类别概率

为什么用LSTM而非原始RNN？
  原始RNN: h_t = tanh(W·[h_{t-1}, x_t] + b)
  问题：梯度在时间步上连乘，导致梯度消失(长期信息丢失)或梯度爆炸
  LSTM解决方案：引入3个门控机制 + 细胞状态，让梯度可以"无损"传递

LSTM的3个门：
  1. 遗忘门(Forget Gate): 决定丢弃什么旧信息
  2. 输入门(Input Gate):  决定存储什么新信息
  3. 输出门(Output Gate): 决定输出什么信息
  细胞状态(Cell State): "信息高速公路"，信息可以几乎不变地流过

【为什么用RNN处理图像？】
虽然CNN更适合图像，但将MNIST视为序列可以：
  1. 直观理解RNN处理序列数据的思想
  2. 深入理解LSTM的门控机制
  3. 为NLP等真正的序列任务打基础
  4. 展示不同架构的灵活性

将28×28的MNIST图像视为序列：
  每一行(28像素) = 一个时间步的输入(28维)
  共28个时间步，对应图像的28行
  LSTM逐步"读"完28行后，用最后的隐藏状态做分类

【应用场景】
- 手写数字识别 (MNIST，本模板使用)
- 语音识别 (音频序列 → 文字)
- 情感分析 (评论文本 → 正面/负面)
- 动作识别 (视频帧序列 → 动作类别)
- 心电/脑电信号分类 (生理信号序列 → 疾病类别)

【本数据集: MNIST】
- 10个类别: 数字0-9
- 70,000张 28×28 灰度图像 (训练60,000 + 测试10,000)
- RNN视角: 28个时间步，每步28维输入

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python rnn/classification.py
3. 数据集自动下载到 data/ 目录
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

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
    # data_dir: 数据集存放目录
    #   torchvision会自动下载MNIST到此目录
    data_dir = "data"

    # num_classes=10: MNIST有10个数字类别(0-9)
    num_classes = 10

    # class_names: 类别名称(用于可视化)
    class_names = [str(i) for i in range(10)]

    # --- RNN序列参数 ---
    # sequence_length=28: 序列长度 = MNIST图像的行数
    #   每一行是一个时间步，共28个时间步
    sequence_length = 28

    # input_size=28: 每个时间步的输入维度 = MNIST图像的列数
    #   每行28个像素，所以输入维度是28
    input_size = 28

    # test_size=0.2: 验证集比例(从训练集中划出)
    #   MNIST训练集60000张，划出20%=12000张作为验证集
    test_size = 0.2

    # random_state=42: 固定随机种子，确保每次运行结果可复现
    random_state = 42

    # --- 模型相关 ---
    # rnn_type="lstm": RNN类型
    #   "lstm": 长短期记忆网络，解决梯度消失，推荐绝大多数场景
    #   "gru": 门控循环单元，比LSTM简单(只有2个门)，速度快约20%
    #   "rnn": 原始RNN，最简单但容易梯度消失，仅用于学习理解
    rnn_type = "lstm"

    # hidden_size=128: 隐藏状态维度
    #   为什么128？MNIST较简单，128足够
    #   为什么不是64？64也能用但准确率略低
    #   为什么不是256？MNIST用256容易过拟合，且训练慢
    #   经验: hidden_size ≈ 输入维度的4-8倍通常效果好
    hidden_size = 128

    # num_layers=2: RNN堆叠层数
    #   为什么2层？第1层学低级时序模式，第2层学高级时序模式
    #   为什么不是1层？1层表达能力有限
    #   为什么不是3层+？MNIST序列短(28步)，3层以上收益递减
    #   经验: 短序列2层，长序列2-3层，超长序列3-4层
    num_layers = 2

    # bidirectional=False: 是否使用双向RNN
    #   True: 同时从前向后和从后向前读取序列，捕获双向上下文
    #   False: 只从前向后读取
    #   MNIST场景: False即可，因为图像从上到下扫描就够用
    #   NLP场景: True更好，因为句子的后文也影响前文理解
    bidirectional = False

    # dropout_rate=0.3: Dropout比例
    #   为什么0.3而不是0.5(CNN)？RNN对Dropout更敏感
    #   RNN的Dropout分两种:
    #     层间Dropout: LSTM的dropout参数，在RNN层之间添加
    #     FC层Dropout: 全连接层的Dropout
    #   0.3是RNN的常用值，太大容易破坏时序信息
    dropout_rate = 0.3

    # fc_dims: 全连接层维度
    #   [64]: LSTM输出128维隐藏状态，压缩到64再输出10类
    #   为什么只有1层FC？LSTM已经做了特征提取，1层FC映射足够
    fc_dims = [64]

    # --- 训练相关 ---
    # batch_size=128: 每次梯度更新使用128个序列
    #   为什么128？MNIST序列短(28步×28维)，内存占用小
    #   注意: RNN的batch_size受序列长度影响，长序列需减小
    batch_size = 128

    # learning_rate=1e-3: 初始学习率
    #   为什么1e-3？LSTM+Adam的标准学习率
    #   注意: 原始RNN可能需要更小的LR(5e-4)，因为更容易梯度爆炸
    learning_rate = 1e-3

    # epochs=30: 最大训练轮数
    #   早停会自动控制，30是上限
    #   MNIST+LSTM通常15-20轮收敛
    epochs = 30

    # weight_decay=1e-5: L2正则化强度
    #   为什么比CNN(5e-4)小很多？LSTM参数量少(~170K vs ~100万)
    #   而且LSTM本身有正则化效果(门控机制)，不需要太强L2
    weight_decay = 1e-5

    # --- 早停策略 ---
    # early_stop_patience=7: 验证损失连续7轮不下降就停止
    #   为什么比CNN(10)小？LSTM收敛快，7轮足以判断
    early_stop_patience = 7

    # --- 学习率调度器 ---
    # scheduler_type="step": 阶梯下降调度
    #   为什么不用Cosine？RNN训练波动大，阶梯式更可控
    #   StepLR: 每step_size个epoch，LR乘以gamma
    scheduler_type = "step"  # "step" 或 "cosine"

    # lr_step_size=10: StepLR的步长
    lr_step_size = 10

    # lr_gamma=0.5: StepLR的衰减因子
    #   每10轮学习率减半，逐步精调
    lr_gamma = 0.5

    # --- 梯度裁剪 ---
    # max_grad_norm=1.0: 梯度L2范数上限
    #   为什么比CNN(5.0)小？RNN/LSTM的梯度在时间步上连乘
    #   虽然LSTM缓解了梯度消失，但梯度爆炸仍可能发生
    #   1.0是RNN的标准值，既防止爆炸又不过度限制学习
    max_grad_norm = 1.0

    # --- 混合精度训练(AMP) ---
    # use_amp=True: 启用自动混合精度
    #   与CNN相同，加速训练，减少显存
    #   仅在CUDA(GPU)上有效，CPU会自动降级
    use_amp = True

    # --- 数据加载优化 ---
    # num_workers: DataLoader的子进程数
    num_workers = min(4, os.cpu_count() or 1)

    # --- 保存相关 ---
    # save_dir: 模型和图表保存目录
    save_dir = "rnn/output/classification"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 数据加载和预处理
# ============================================================
def get_transforms():
    """
    创建MNIST的数据变换管道。

    【为什么RNN不需要数据增强？】
    - 我们将图像按行视为序列，旋转/翻转会破坏行的时序关系
    - CNN的增强(随机裁剪/翻转)在RNN中不适用
    - 如果用CNN做MNIST分类，可以考虑添加轻微增强
    - RNN的优势在于序列建模，不需要空间增强
    """
    # MNIST的均值和标准差(对训练集统计得出)
    normalize = transforms.Normalize(
        mean=[0.1307],
        std=[0.3081],
    )

    transform = transforms.Compose([
        transforms.ToTensor(),  # 0~255 → 0~1
        normalize,              # 标准化
    ])

    return transform


def get_dataloaders(cfg):
    """
    加载MNIST数据集并创建DataLoader。

    【RNN处理图像的数据转换】
    原始MNIST图像: (batch, 1, 28, 28)  — (batch, channel, height, width)
    RNN需要的格式: (batch, 28, 28)     — (batch, seq_len, input_size)

    转换方法: squeeze去掉channel维度
    - 图像的每一行(28像素) → 一个时间步的输入(28维)
    - 共28行 → 28个时间步
    - LSTM逐步"读"完28行，每步读28个像素

    【为什么不在DataLoader中做转换？】
    - transforms处理的是单个样本，不便于做维度调整
    - 在训练/推理时squeeze更灵活
    """
    transform = get_transforms()

    # 下载并加载训练集(60,000张)
    train_dataset = datasets.MNIST(
        root=cfg.data_dir, train=True, download=True, transform=transform,
    )

    # 下载并加载测试集(10,000张)
    test_dataset = datasets.MNIST(
        root=cfg.data_dir, train=False, download=True, transform=transform,
    )

    # 从训练集中划出验证集
    # 【为什么要验证集？】
    # 训练集: 训练模型参数
    # 验证集: 监控过拟合，调超参数，决定何时早停
    # 测试集: 最终评估，训练过程中绝对不能用
    n_total = len(train_dataset)
    n_val = int(n_total * cfg.test_size)
    n_train = n_total - n_val

    # 固定随机种子确保划分一致
    generator = torch.Generator().manual_seed(cfg.random_state)
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [n_train, n_val], generator=generator,
    )

    # 创建DataLoader
    pin_mem = cfg.device.type == "cuda"
    pw = cfg.num_workers > 0

    train_loader = DataLoader(
        train_subset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=pw,
    )
    val_loader = DataLoader(
        val_subset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=pw,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=pw,
    )

    print(f"训练集: {n_train}张 | 验证集: {n_val}张 | 测试集: {len(test_dataset)}张")

    return train_loader, val_loader, test_loader


# ============================================================
# Step 4: 模型定义
# ============================================================
class RNNClassifier(nn.Module):
    """
    RNN序列分类模型

    【架构设计思路】
    输入序列 (batch, 28, 28)   ← MNIST: 28个时间步，每步28维
      → LSTM Layer1 (28→128)   ← 第1层LSTM，提取低级时序模式
      → LSTM Layer2 (128→128)  ← 第2层LSTM，提取高级时序模式
      → 取最后时间步输出        ← 整个序列的信息聚合到最后
      → FC(128→64) → ReLU → Dropout → FC(64→10)

    【维度变化详解】
    输入: (batch, 28, 28)
      → LSTM: (batch, 28, 128)  — 每个时间步输出128维隐藏状态
      → 取最后时间步: (batch, 128)  — 只取第28步的输出
      → FC: (batch, 10)  — 10个类别的logits

    【为什么取最后时间步？】
    LSTM的设计使信息在时间步之间传递，最后一个时间步的隐藏状态
    已经"看过"整个序列，包含了所有历史信息的总结。
    这就像读完一本书后，你对整个故事的理解最完整。

    【双向RNN时为什么取最后时间步不同？】
    单向: 取最后一个时间步 → (batch, hidden_size)
    双向: 取最后和最先时间步拼接 → (batch, hidden_size*2)
          正向最后一步 + 反向第一步(反向的最后一步) 包含完整上下文

    【参数量计算 (LSTM)】
    LSTM参数量 = 4 × [(input_size + hidden_size + 1) × hidden_size]
    第1层: 4 × [(28 + 128 + 1) × 128] = 4 × 20096 = 80,384
    第2层: 4 × [(128 + 128 + 1) × 128] = 4 × 32896 = 131,584
    FC层: 128×64 + 64 + 64×10 + 10 = 8,834
    总计 ≈ 220K
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_layers
        self.bidirectional = cfg.bidirectional

        # 计算LSTM输出维度(双向时翻倍)
        self.rnn_output_size = cfg.hidden_size * (2 if cfg.bidirectional else 1)

        # ---- RNN层 ----
        # 【LSTM vs GRU vs RNN 对比】
        # LSTM: 3个门(遗忘/输入/输出) + 细胞状态，最强大
        # GRU:  2个门(重置/更新)，比LSTM简单，速度快约20%
        # RNN:  无门控，最简单，容易梯度消失
        if cfg.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=cfg.input_size,       # 输入维度: 28
                hidden_size=cfg.hidden_size,     # 隐藏维度: 128
                num_layers=cfg.num_layers,       # 层数: 2
                batch_first=True,                # 输入格式: (batch, seq, feature)
                dropout=cfg.dropout_rate if cfg.num_layers > 1 else 0,
                bidirectional=cfg.bidirectional,
            )
        elif cfg.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=cfg.input_size,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=cfg.dropout_rate if cfg.num_layers > 1 else 0,
                bidirectional=cfg.bidirectional,
            )
        else:  # 原始RNN
            self.rnn = nn.RNN(
                input_size=cfg.input_size,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=cfg.dropout_rate if cfg.num_layers > 1 else 0,
                bidirectional=cfg.bidirectional,
                nonlinearity="tanh",  # "tanh" 或 "relu"
            )

        # ---- 全连接分类头 ----
        # 将RNN最后的隐藏状态映射到类别
        fc_layers = []
        in_dim = self.rnn_output_size
        for dim in cfg.fc_dims:
            fc_layers.extend([
                nn.Linear(in_dim, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout_rate),
            ])
            in_dim = dim
        fc_layers.append(nn.Linear(in_dim, cfg.num_classes))
        self.classifier = nn.Sequential(*fc_layers)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        权重初始化。

        【为什么RNN的初始化和CNN不同？】
        - LSTM/GRU有特殊的门控参数，不适合用He初始化
        - 遗忘门偏置初始化为1，让遗忘门初始倾向于"记住"信息
        - 这是LSTM训练的重要技巧
        - 如果遗忘门初始偏置为0，训练初期可能遗忘太多信息
        """
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                # 输入到隐藏的权重: Xavier初始化
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                # 隐藏到隐藏的权重: 正交初始化
                # 【为什么用正交初始化？】
                # 正交矩阵的连乘不会放大或缩小，有利于梯度稳定传播
                nn.init.orthogonal_(param)
            elif "bias" in name:
                # 偏置初始化
                nn.init.zeros_(param)
                # LSTM遗忘门偏置设为1
                # 遗忘门偏置在参数中的位置: 第2个门(block)
                # LSTM有4个门，偏置排列: [input, forget, cell, output]
                if isinstance(self.rnn, nn.LSTM):
                    # 遗忘门偏置 = hidden_size到2*hidden_size
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1.0)

    def forward(self, x):
        """
        前向传播

        数据流动:
        x: (batch, 28, 28)       ← MNIST序列: 28步×28维
          → rnn: (batch, 28, hidden_size)
          → 取最后时间步: (batch, hidden_size)
          → classifier: (batch, 10)  ← logits
        """
        batch_size = x.size(0)

        # 初始化隐藏状态(全零)
        # 【为什么每次前向传播都要初始化？】
        # 每个样本独立处理，不需要跨batch的隐藏状态
        # 在NLP生成任务中，可能需要保持隐藏状态(推理时逐步生成)
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size, self.hidden_size,
        ).to(x.device)

        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros_like(h0)
            # LSTM前向传播
            # rnn_out: 所有时间步的隐藏状态 (batch, seq_len, hidden_size)
            # _: 最后的(h_n, c_n)
            rnn_out, _ = self.rnn(x, (h0, c0))
        else:
            # GRU/RNN前向传播
            rnn_out, _ = self.rnn(x, h0)

        # 取最后时间步的输出
        # 【为什么取最后时间步？】
        # 最后一个时间步的隐藏状态包含了整个序列的信息
        # 因为LSTM的信息传递机制，每一步都会将之前的信息传递下来
        out = rnn_out[:, -1, :]  # (batch, hidden_size) 或 (batch, hidden_size*2)

        # 分类
        out = self.classifier(out)
        return out


# ============================================================
# Step 5: 训练函数
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, cfg, scaler=None):
    """
    训练一个epoch。

    【RNN训练的特殊之处】
    1. 输入是3D张量: (batch, seq_len, input_size) — 需要从4D图像squeeze
    2. BPTT(Backpropagation Through Time): 梯度在时间步上反向传播
    3. 梯度裁剪对RNN尤其重要: 防止梯度在时间步连乘中爆炸
    4. LSTM的隐藏状态在每个batch开始时重置为0
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for inputs, targets in loader:
        # 【数据维度转换】
        # 原始: (batch, 1, 28, 28) — 4D图像格式
        # 转换: (batch, 28, 28)    — 3D序列格式 (seq_len=28, input_size=28)
        inputs = inputs.squeeze(1)
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)

        # 前向传播(混合精度)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

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

        # 统计
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, cfg):
    """
    评估模型性能。

    @torch.no_grad(): 不计算梯度，节省GPU显存
    model.eval(): 切换到评估模式
      - LSTM的Dropout不生效
      - 注意: LSTM的batch_norm不常见，所以eval()影响较小
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for inputs, targets in loader:
        inputs = inputs.squeeze(1)  # 4D → 3D
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(all_targets)
    acc = accuracy_score(all_targets, all_preds)

    return avg_loss, acc, all_preds, all_targets


def train(model, train_loader, val_loader, cfg):
    """
    完整训练流程: 训练 + 验证 + 早停 + 学习率调度

    【训练流程】
    每个epoch:
      1. 训练一个epoch (前向+反向+优化+梯度裁剪)
      2. 在验证集上评估
      3. 更新学习率
      4. 检查是否需要早停
      5. 保存最优模型
    """
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    # 学习率调度器
    if cfg.scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # 早停相关变量
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # 记录训练曲线
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # 混合精度训练的GradScaler
    use_amp = cfg.use_amp and cfg.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")
    print(f"设备: {cfg.device} | 优化器: Adam(lr={cfg.learning_rate}) | "
          f"调度器: {cfg.scheduler_type} | AMP: {use_amp}")

    for epoch in range(1, cfg.epochs + 1):
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, cfg, scaler)
        # 验证
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, cfg)

        # 记录
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]["lr"]

        # 打印
        print(f"Epoch {epoch:3d}/{cfg.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")

        # 更新学习率
        scheduler.step()

        # 早停检查
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

    # 恢复最优模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(cfg.device)
        print(f"\n✓ 已恢复最优模型 (Val Loss: {best_val_loss:.4f})")

    return model, history


# ============================================================
# Step 6: 可视化函数
# ============================================================
def plot_training_curves(history, cfg):
    """绘制训练曲线(损失+准确率)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("RNN训练/验证损失曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("RNN训练/验证准确率曲线")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, cfg):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=cfg.class_names, yticklabels=cfg.class_names,
           ylabel="真实类别", xlabel="预测类别",
           title="RNN分类混淆矩阵")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 混淆矩阵已保存: {save_path}")
    plt.close()


def plot_sample_predictions(model, test_loader, cfg, num_samples=16):
    """可视化预测结果：展示部分测试图像及模型预测"""
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_samples], labels[:num_samples]

    with torch.no_grad():
        inputs = images.squeeze(1).to(cfg.device)  # 4D → 3D
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(1).cpu()

    # 反标准化用于显示
    mean = torch.tensor([0.1307]).view(1, 1, 1)
    std = torch.tensor([0.3081]).view(1, 1, 1)

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    for i, ax in enumerate(axes.flat):
        if i >= num_samples:
            break
        img = images[i] * std + mean
        img = img.squeeze().numpy().clip(0, 1)

        ax.imshow(img, cmap="gray")
        true_name = cfg.class_names[labels[i]]
        pred_name = cfg.class_names[preds[i]]
        confidence = probs[i, preds[i]].item()

        color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(f"真实: {true_name}\n预测: {pred_name} ({confidence:.1%})",
                     color=color, fontsize=9)
        ax.axis("off")

    plt.suptitle("RNN序列分类预测结果", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "sample_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 预测结果已保存: {save_path}")
    plt.close()


def plot_hidden_states(model, test_loader, cfg, sample_idx=0):
    """
    可视化LSTM隐藏状态随时间步的变化。

    【为什么要看隐藏状态？】
    - 理解RNN的"记忆"是如何演化的
    - 观察不同时间步的激活模式
    - 验证LSTM是否学到了有意义的时序特征
    """
    model.eval()
    images, labels = next(iter(test_loader))
    img = images[sample_idx:sample_idx+1].squeeze(1).to(cfg.device)  # (1, 28, 28)

    # 手动提取LSTM的隐藏状态
    with torch.no_grad():
        h0 = torch.zeros(model.num_layers, 1, model.hidden_size).to(cfg.device)
        c0 = torch.zeros_like(h0)
        rnn_out, (h_n, c_n) = model.rnn(img, (h0, c0))

    # 可视化前16个隐藏维度
    hidden = rnn_out[0].cpu().numpy()  # (28, hidden_size)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(hidden[:, :16].T, aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1)
    ax.set_xlabel("时间步 (图像行)")
    ax.set_ylabel("隐藏维度")
    ax.set_title(f"LSTM隐藏状态变化 (真实标签: {labels[sample_idx].item()})")
    plt.colorbar(im, ax=ax, label="激活值")

    save_path = os.path.join(cfg.save_dir, "hidden_states.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 隐藏状态可视化已保存: {save_path}")
    plt.close()


# ============================================================
# Step 7: 预测函数
# ============================================================
@torch.no_grad()
def predict(model, image_tensor, cfg):
    """
    对单张图像进行预测。

    参数:
        image_tensor: 预处理后的图像张量 (1, 1, 28, 28)
    返回:
        pred_class: 预测类别索引
        pred_name: 预测类别名称
        confidence: 预测置信度
        probabilities: 各类别概率
    """
    model.eval()
    # 转换为序列格式: (1, 1, 28, 28) → (1, 28, 28)
    inputs = image_tensor.squeeze(1).to(cfg.device)
    output = model(inputs)
    probabilities = torch.softmax(output, dim=1)
    confidence, pred_class = probabilities.max(1)
    pred_name = cfg.class_names[pred_class.item()]

    return pred_class.item(), pred_name, confidence.item(), probabilities.cpu().numpy()


# ============================================================
# Step 8: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("RNN 序列分类 - MNIST手写数字识别")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 创建输出目录
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 加载数据
    print("\n加载数据集...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # 创建模型
    model = RNNClassifier(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型: RNNClassifier ({cfg.rnn_type.upper()})")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"\n模型结构:\n{model}")

    # 训练
    model, history = train(model, train_loader, val_loader, cfg)

    # 在测试集上评估
    print(f"\n{'='*60}")
    print("测试集评估...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, cfg)
    print(f"测试集 Loss: {test_loss:.4f} | 准确率: {test_acc:.4f}")

    # 详细分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=cfg.class_names, digits=4))

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "rnn_classifier.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
        "history": history,
    }, model_path)
    print(f"✓ 模型已保存: {model_path}")

    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, cfg)
    plot_confusion_matrix(y_true, y_pred, cfg)
    plot_sample_predictions(model, test_loader, cfg)
    plot_hidden_states(model, test_loader, cfg)

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
