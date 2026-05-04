"""
=============================================================================
RNN 文本分类/情感分析任务模板 (LSTM for Text Classification / Sentiment Analysis)
=============================================================================

【原理】
文本分类是RNN最经典的应用场景之一。与图像分类不同，文本是天然的序列数据——
每个词的出现都依赖于上下文，RNN正好擅长捕捉这种时序依赖关系。

核心流程：
  原始文本 → 分词 → 词嵌入(Lookup Table) → LSTM编码 → 分类

关键概念：
1. 词嵌入(Word Embedding): 将离散的词索引映射为连续的稠密向量
   - "我" → [0.2, 0.8, 0.1, ...]  (embedding_dim维)
   - 相似的词在嵌入空间中距离近(如"好"和"棒")
   - nn.Embedding: 可学习的查找表(vocab_size × embedding_dim)

2. 变长序列处理:
   - 不同句子长度不同，需要padding对齐
   - pack_padded_sequence: 压缩padding，LSTM不计算padding部分
   - pad_packed_sequence: 解压回原始格式

3. 双向LSTM(BiLSTM):
   - 正向: 从第1个词读到最后1个词
   - 反向: 从最后1个词读到第1个词
   - 拼接双向输出，同时捕获前后文信息
   - 例: "这个手机不好用" → 正向读到"不好"时还不知道"用"
          → 反向从"用"开始读，知道是"不好用"而非"不好"

【应用场景】
- 情感分析 (评论/微博 → 正面/负面)  ← 本模板使用
- 垃圾邮件检测 (邮件 → 正常/垃圾)
- 新闻分类 (新闻 → 体育/科技/娱乐等)
- 意图识别 (用户输入 → 意图类别)
- 关系抽取 (实体对 → 关系类型)

【本数据集: 合成中文评论数据】
- 合成1000条模拟中文评论(正面/负面各500条)
- 词汇表大小: ~200个词
- 正面评论示例: "这个 产品 质量 很好 非常 满意"
- 负面评论示例: "这个 产品 质量 很差 非常 失望"

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python rnn/text_classification.py
3. 合成数据自动生成，无需下载
=============================================================================
"""

# ============================================================
# Step 1: 导入必要的库
# ============================================================
import os
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
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
    """超参数配置中心 —— 所有可调参数集中在此。"""

    # --- 数据相关 ---
    # num_classes=2: 情感分析二分类(正面/负面)
    num_classes = 2

    # class_names: 类别名称
    class_names = ["负面", "正面"]

    # vocab_size: 词汇表大小(合成数据的词汇量)
    vocab_size = 200

    # max_seq_length=20: 最大序列长度
    #   超过此长度的文本会被截断，不足的会被padding
    max_seq_length = 20

    # num_samples=1000: 合成数据总样本数
    num_samples = 1000

    # test_size=0.2: 验证集比例
    test_size = 0.2

    # random_state=42: 随机种子
    random_state = 42

    # --- 词嵌入相关 ---
    # embedding_dim=64: 词嵌入维度
    #   为什么64？合成数据词汇量小(~200)，64维足够
    #   实际NLP任务: 100-300维常用，预训练词向量(GloVe)通常300维
    #   维度越高表达能力越强，但需要更多数据，否则过拟合
    embedding_dim = 64

    # --- 模型相关 ---
    # rnn_type="lstm": RNN类型
    rnn_type = "lstm"

    # hidden_size=128: 隐藏状态维度
    #   文本分类通常128-256维
    hidden_size = 128

    # num_layers=2: RNN层数
    num_layers = 2

    # bidirectional=True: 使用双向LSTM
    #   【为什么文本分类要用双向？】
    #   句子的理解依赖前后文: "不好用"要知道后面的"用"
    #   双向LSTM同时捕获前向和后向信息
    bidirectional = True

    # dropout_rate=0.3: Dropout比例
    dropout_rate = 0.3

    # fc_dims: 全连接层维度
    fc_dims = [64]

    # --- 训练相关 ---
    batch_size = 32

    learning_rate = 1e-3

    epochs = 30

    weight_decay = 1e-4

    # --- 早停策略 ---
    early_stop_patience = 7

    # --- 学习率调度器 ---
    scheduler_type = "step"

    lr_step_size = 10

    lr_gamma = 0.5

    # --- 梯度裁剪 ---
    max_grad_norm = 1.0

    # --- 混合精度训练(AMP) ---
    use_amp = True

    # --- 数据加载优化 ---
    num_workers = min(2, os.cpu_count() or 1)

    # --- 保存相关 ---
    save_dir = "rnn/output/text_classification"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 合成数据和数据加载
# ============================================================
def generate_synthetic_data(cfg):
    """
    生成合成的中文评论情感分析数据。

    【为什么要用合成数据？】
    - 真实NLP数据需要分词工具(jieba等)，增加依赖
    - 合成数据可以控制难度和规模，适合学习
    - 无需下载，即开即用
    - 学会原理后，替换为真实数据只需修改数据加载函数

    【合成数据结构】
    正面评论 = 正面形容词 + 正面副词 + 名词 + ...
    负面评论 = 负面形容词 + 负面副词 + 名词 + ...

    词汇索引分配:
    0: <PAD>   填充符
    1: <UNK>   未知词
    2-50: 正面词
    51-100: 负面词
    101-199: 中性词(名词/代词等)
    """
    random.seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    # 定义词汇和索引
    # 正面词 (索引 2-50)
    positive_words = {
        "好": 2, "棒": 3, "优秀": 4, "满意": 5, "喜欢": 6,
        "赞": 7, "推荐": 8, "不错": 9, "完美": 10, "超赞": 11,
        "给力": 12, "舒适": 13, "漂亮": 14, "实惠": 15, "好用": 16,
        "方便": 17, "快速": 18, "靠谱": 19, "精致": 20, "值得": 21,
    }

    # 负面词 (索引 51-70)
    negative_words = {
        "差": 51, "烂": 52, "失望": 53, "糟糕": 54, "难用": 55,
        "垃圾": 56, "退货": 57, "不满": 58, "恶心": 59, "浪费": 60,
        "难看": 61, "粗糙": 62, "慢": 63, "坑": 64, "不推荐": 65,
    }

    # 中性词 (索引 101-150)
    neutral_words = {
        "产品": 101, "质量": 102, "价格": 103, "服务": 104, "物流": 105,
        "包装": 106, "这个": 107, "那个": 108, "非常": 109, "真的": 110,
        "太": 111, "很": 112, "特别": 113, "比较": 114, "一般": 115,
        "感觉": 116, "觉得": 117, "这次": 118, "第一次": 119, "用了": 120,
    }

    # 生成评论模板
    positive_templates = [
        [107, 101, 102, 112, 2],      # 这个 产品 质量 很 好
        [101, 109, 3],                  # 产品 非常 棒
        [102, 113, 4],                  # 质量 特别 优秀
        [110, 5],                       # 真的 满意
        [107, 101, 112, 9],            # 这个 产品 很 不错
        [101, 111, 10],                # 产品 太 完美
        [103, 112, 15],                # 价格 很 实惠
        [104, 109, 7],                  # 服务 非常 赞
        [101, 112, 16],                # 产品 很 好用
        [107, 101, 109, 8],            # 这个 产品 非常 推荐
    ]

    negative_templates = [
        [107, 101, 102, 112, 51],      # 这个 产品 质量 很 差
        [101, 109, 52],                 # 产品 非常 烂
        [102, 113, 54],                 # 质量 特别 糟糕
        [110, 53],                       # 真的 失望
        [107, 101, 112, 55],            # 这个 产品 很 难用
        [101, 111, 56],                 # 产品 太 垃圾
        [103, 112, 60],                 # 价格 很 浪费
        [104, 109, 58],                 # 服务 非常 不满
        [101, 112, 61],                 # 产品 很 难看
        [107, 101, 109, 65],            # 这个 产品 非常 不推荐
    ]

    # 构建词汇表(索引到词的映射)
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    idx2word.update({v: k for k, v in positive_words.items()})
    idx2word.update({v: k for k, v in negative_words.items()})
    idx2word.update({v: k for k, v in neutral_words.items()})

    # 生成数据
    texts = []
    labels = []

    n_per_class = cfg.num_samples // 2

    for _ in range(n_per_class):
        # 正面评论
        template = random.choice(positive_templates)
        # 随机添加1-3个中性词
        length = random.randint(3, min(cfg.max_seq_length, len(template) + 3))
        text = template[:length]
        # padding
        text = text + [0] * (cfg.max_seq_length - len(text))
        text = text[:cfg.max_seq_length]
        texts.append(text)
        labels.append(1)  # 正面

        # 负面评论
        template = random.choice(negative_templates)
        length = random.randint(3, min(cfg.max_seq_length, len(template) + 3))
        text = template[:length]
        text = text + [0] * (cfg.max_seq_length - len(text))
        text = text[:cfg.max_seq_length]
        texts.append(text)
        labels.append(0)  # 负面

    # 打乱数据
    indices = list(range(len(texts)))
    random.shuffle(indices)
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    return texts, labels, idx2word


class TextDataset(Dataset):
    """文本分类数据集。"""

    def __init__(self, texts, labels):
        """
        参数:
            texts: 词索引列表，每个元素是 [word_idx1, word_idx2, ...]
            labels: 情感标签列表 (0=负面, 1=正面)
        """
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 返回 (词索引序列, 标签, 序列实际长度)
        text = self.texts[idx]
        label = self.labels[idx]
        # 计算实际长度(不含padding)
        length = sum(1 for w in text if w != 0)
        length = max(length, 1)  # 至少为1
        return (
            torch.tensor(text, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )


def collate_fn(batch):
    """
    自定义batch整理函数。

    【为什么需要自定义collate_fn？】
    - 需要按序列长度从长到短排序(pack_padded_sequence的要求)
    - 返回 (序列, 标签, 长度) 三元组
    """
    # 按长度从长到短排序
    batch.sort(key=lambda x: x[2], reverse=True)
    texts, labels, lengths = zip(*batch)
    texts = torch.stack(texts)
    labels = torch.stack(labels)
    lengths = torch.stack(lengths)
    return texts, labels, lengths


def get_dataloaders(cfg):
    """生成合成数据并创建DataLoader。"""
    texts, labels, idx2word = generate_synthetic_data(cfg)

    # 保存词汇表供后续使用
    cfg.idx2word = idx2word

    # 划分训练/验证/测试集
    n_total = len(texts)
    n_val = int(n_total * cfg.test_size)
    n_test = int(n_total * cfg.test_size)
    n_train = n_total - n_val - n_test

    train_texts = texts[:n_train]
    train_labels = labels[:n_train]
    val_texts = texts[n_train:n_train + n_val]
    val_labels = labels[n_train:n_train + n_val]
    test_texts = texts[n_train + n_val:]
    test_labels = labels[n_train + n_val:]

    train_dataset = TextDataset(train_texts, train_labels)
    val_dataset = TextDataset(val_texts, val_labels)
    test_dataset = TextDataset(test_texts, test_labels)

    pin_mem = cfg.device.type == "cuda"
    pw = cfg.num_workers > 0

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=pw, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=pw, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=pw, collate_fn=collate_fn,
    )

    print(f"训练集: {n_train}条 | 验证集: {n_val}条 | 测试集: {n_test}条")
    print(f"词汇表大小: {len(idx2word)}")

    return train_loader, val_loader, test_loader


# ============================================================
# Step 4: 模型定义
# ============================================================
class TextClassifier(nn.Module):
    """
    RNN文本分类模型 (BiLSTM)

    【架构设计思路】
    输入: (batch, seq_len)  ← 词索引序列
      → Embedding: (batch, seq_len, embedding_dim)  ← 词嵌入
      → BiLSTM: (batch, seq_len, hidden_size*2)     ← 双向编码
      → 取最后时间步 或 最大池化: (batch, hidden_size*2)
      → FC: (batch, num_classes)

    【为什么文本分类需要词嵌入？】
    - 原始输入是词的整数索引(如"好"=2)，没有语义信息
    - 词嵌入将离散索引映射为连续向量，让相似的词在空间中接近
    - nn.Embedding是一个可学习的查找表:
      输入: 词索引(如2) → 输出: 对应的embedding_dim维向量
    - 训练过程中，词嵌入会自动学习到语义关系

    【为什么用BiLSTM？】
    - 正向LSTM: "这个 手机 不好 ___" → 还不知道后面是什么
    - 反向LSTM: "___ 不好 用 手机 这个" → 从后往前读，"用"在"不好"前面
    - 拼接双向: 同时拥有前后文信息，分类更准确

    【pack_padded_sequence是什么？】
    - padding的词(LSTM不应该处理)用0填充
    - pack_padded_sequence: 告诉LSTM哪些位置是padding，跳过它们
    - 好处: (1)加速计算 (2)避免padding词干扰隐藏状态
    - 使用前提: batch内的序列要按长度从长到短排序
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_layers
        self.bidirectional = cfg.bidirectional
        self.rnn_output_size = cfg.hidden_size * (2 if cfg.bidirectional else 1)

        # ---- 词嵌入层 ----
        # vocab_size: 词汇表大小
        # embedding_dim: 每个词的向量维度
        # padding_idx=0: 索引0(<PAD>)的嵌入始终为0，不参与训练
        self.embedding = nn.Embedding(
            cfg.vocab_size, cfg.embedding_dim, padding_idx=0,
        )

        # ---- RNN层 ----
        if cfg.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=cfg.embedding_dim,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=cfg.dropout_rate if cfg.num_layers > 1 else 0,
                bidirectional=cfg.bidirectional,
            )
        elif cfg.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=cfg.embedding_dim,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=cfg.dropout_rate if cfg.num_layers > 1 else 0,
                bidirectional=cfg.bidirectional,
            )

        # ---- 全连接分类头 ----
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

        self._init_weights()

    def _init_weights(self):
        """权重初始化。"""
        # 词嵌入初始化(除了padding_idx=0)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[0].zero_()  # padding位置为0

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

    def forward(self, x, lengths):
        """
        前向传播

        参数:
            x: 词索引序列 (batch, seq_len)
            lengths: 实际序列长度 (batch,)

        数据流动:
        x: (batch, 20)              ← 词索引
          → embedding: (batch, 20, 64)  ← 词嵌入
          → pack: 压缩padding
          → BiLSTM: (batch, 20, 256)    ← 双向隐藏状态
          → unpack: 解压
          → 取最后时间步/最大池化: (batch, 256)
          → classifier: (batch, 2)      ← 分类logits
        """
        # 词嵌入
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # pack_padded_sequence: 压缩padding部分
        # 【为什么要pack？】
        # LSTM不需要计算padding位置，pack告诉LSTM每个序列的真实长度
        # 这样LSTM的隐藏状态不会被padding词污染
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=True,
        )

        # RNN前向传播
        if isinstance(self.rnn, nn.LSTM):
            packed_out, (h_n, c_n) = self.rnn(packed)
        else:
            packed_out, h_n = self.rnn(packed)

        # 解压回原始格式
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        # rnn_out: (batch, seq_len, hidden_size*2)

        # 取序列表示
        # 方法1: 取最后时间步(简单)
        # 方法2: 最大池化(更鲁棒，推荐)
        # 【为什么最大池化比最后时间步更好？】
        # 最后时间步可能包含padding的影响
        # 最大池化取每个维度在所有时间步上的最大值，不受padding影响
        # 而且最大池化能捕获整个序列中最显著的特征
        out = rnn_out.max(dim=1)[0]  # (batch, hidden_size*2)

        # 分类
        out = self.classifier(out)
        return out


# ============================================================
# Step 5: 训练函数
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, cfg, scaler=None):
    """训练一个epoch。"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for texts, labels, lengths in loader:
        texts, labels = texts.to(cfg.device), labels.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)

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

        total_loss += loss.item() * texts.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += texts.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, cfg):
    """评估模型性能。"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for texts, labels, lengths in loader:
        texts, labels = texts.to(cfg.device), labels.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * texts.size(0)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_targets)
    acc = accuracy_score(all_targets, all_preds)

    return avg_loss, acc, all_preds, all_targets


def train(model, train_loader, val_loader, cfg):
    """完整训练流程。"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    if cfg.scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    use_amp = cfg.use_amp and cfg.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")
    print(f"设备: {cfg.device} | 优化器: Adam(lr={cfg.learning_rate}) | AMP: {use_amp}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, cfg, scaler)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, cfg)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{cfg.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("文本分类训练/验证损失曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("文本分类训练/验证准确率曲线")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, cfg):
    """绘制混淆矩阵。"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=cfg.class_names, yticklabels=cfg.class_names,
           ylabel="真实类别", xlabel="预测类别",
           title="文本分类混淆矩阵")

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


def plot_embeddings(model, cfg):
    """
    可视化词嵌入空间(t-SNE降维)。

    【为什么要看词嵌入？】
    - 验证模型是否学到了有意义的语义关系
    - 正面词应该聚在一起，负面词应该聚在一起
    - 功能相似的词应该接近
    """
    from sklearn.manifold import TSNE

    # 获取词嵌入权重(排除padding)
    weights = model.embedding.weight.data[2:, :].cpu().numpy()  # 排除PAD和UNK
    n_words = weights.shape[0]

    # t-SNE降维到2D
    if n_words > 5:
        tsne = TSNE(n_components=2, random_state=cfg.random_state, perplexity=min(30, n_words - 1))
        embedded_2d = tsne.fit_transform(weights)

        # 绘制
        fig, ax = plt.subplots(figsize=(12, 8))

        # 正面词(索引2-50中的有效词)
        pos_indices = [i for i in range(min(49, n_words)) if i < n_words]
        ax.scatter(embedded_2d[pos_indices, 0], embedded_2d[pos_indices, 1],
                   c="green", alpha=0.6, label="正面词", s=50)

        # 负面词(索引49-68中的有效词)
        neg_start = 49
        neg_end = min(68, n_words)
        if neg_end > neg_start:
            neg_indices = list(range(neg_start, neg_end))
            ax.scatter(embedded_2d[neg_indices, 0], embedded_2d[neg_indices, 1],
                       c="red", alpha=0.6, label="负面词", s=50)

        ax.set_title("词嵌入空间可视化 (t-SNE)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path = os.path.join(cfg.save_dir, "word_embeddings.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ 词嵌入可视化已保存: {save_path}")
        plt.close()


# ============================================================
# Step 7: 预测函数
# ============================================================
@torch.no_grad()
def predict(model, text_indices, cfg):
    """
    对单条文本进行预测。

    参数:
        text_indices: 词索引列表 [word_idx1, word_idx2, ...]
    返回:
        pred_class: 预测类别(0=负面, 1=正面)
        pred_name: 预测类别名称
        confidence: 置信度
        probabilities: 各类别概率
    """
    model.eval()

    # 转为tensor
    text = torch.tensor([text_indices], dtype=torch.long)
    length = torch.tensor([sum(1 for w in text_indices if w != 0)], dtype=torch.long)

    text = text.to(cfg.device)

    output = model(text, length)
    probabilities = torch.softmax(output, dim=1)
    confidence, pred_class = probabilities.max(1)
    pred_name = cfg.class_names[pred_class.item()]

    return pred_class.item(), pred_name, confidence.item(), probabilities.cpu().numpy()


# ============================================================
# Step 8: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("RNN 文本分类/情感分析 - 合成中文评论数据")
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
    model = TextClassifier(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型: TextClassifier (Bi{cfg.rnn_type.upper()})")
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

    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=cfg.class_names, digits=4))

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "text_classifier.pth")
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
    try:
        plot_embeddings(model, cfg)
    except Exception as e:
        print(f"⚠ 词嵌入可视化跳过: {e}")

    # 演示预测
    print("\n预测演示:")
    idx2word = cfg.idx2word
    demo_texts = [
        [107, 101, 102, 112, 2],   # 这个 产品 质量 很 好
        [107, 101, 102, 112, 51],  # 这个 产品 质量 很 差
        [101, 109, 3],              # 产品 非常 棒
        [101, 109, 52],             # 产品 非常 烂
    ]
    for text_idx in demo_texts:
        words = [idx2word.get(w, "?") for w in text_idx if w > 1]
        pred_class, pred_name, conf, _ = predict(model, text_idx, cfg)
        print(f"  文本: {' '.join(words)} → 预测: {pred_name} (置信度: {conf:.2%})")

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
