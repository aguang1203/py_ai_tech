"""
=============================================================================
LSTM 文本分类任务模板 (Text Classification with LSTM)
=============================================================================

【原理】
文本分类是NLP的基础任务：给定一段文本，判断它属于哪个类别。
LSTM通过逐词处理文本，积累对文本语义的理解，最终做出分类判断。

【LSTM处理文本的流程】
  "这部电影太精彩了" → 分词 → [这, 部, 电影, 太, 精彩, 了]
                                  ↓  ↓    ↓    ↓    ↓    ↓
                                LSTM逐步处理，更新隐藏状态
                                  ↓
                                最后一个隐藏状态 → 全连接 → 类别概率

【为什么LSTM适合文本分类？】
- 文本有明显的序列结构(词序很重要："狗咬人"≠"人咬狗")
- LSTM能记住长距离依赖(句首的"不"可能否定句尾的内容)
- 比词袋模型(Bag of Words)强：考虑了词序和上下文
- 比CNN文本分类强：能捕获任意距离的依赖关系

【文本分类的关键步骤】
  1. 分词(Tokenization): 将文本切分为词/字
  2. 词嵌入(Embedding): 将词映射为稠密向量
  3. 序列编码(LSTM): 逐词处理，积累语义
  4. 分类(FC): 基于语义表示预测类别

【词嵌入(Embedding)原理】
将离散的词映射为连续的稠密向量：
  "猫" → [0.2, -0.5, 0.8, ...]  (embedding_dim维向量)

为什么不用One-Hot？
  - One-Hot: 10000维向量，99.99%是0，浪费空间
  - Embedding: 128维向量，每个维度都有意义
  - One-Hot: 任意两个词距离相等(正交)
  - Embedding: 语义相近的词距离近("猫"和"狗"比"猫"和"汽车"近)

【应用场景】
- 情感分析(正面/负面)
- 垃圾邮件检测(正常/垃圾)
- 新闻分类(体育/科技/财经/娱乐)
- 意图识别(客服机器人)
- 话题分类(社交媒体)

【本数据集: 合成情感分析数据】
- 2个类别: 正面/负面
- 词汇表大小: 200个词(100正面词+100负面词)
- 每条评论: 5-15个词
- 即时生成，无需下载
- 特点: 保留文本分类任务的核心特性(词序、语义、类别)

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python lstm/text_classification.py
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
# Step 2: 配置超参数
# ============================================================
class CONFIG:
    """超参数配置中心 —— 文本分类任务的所有可调参数。"""

    # --- 数据相关 ---
    # vocab_size=200: 词汇表大小
    #   100个正面词 + 100个负面词
    #   真实任务通常10000-100000，这里用小词表便于快速训练
    vocab_size = 200

    # num_classes=2: 类别数(正面/负面)
    num_classes = 2

    # class_names: 类别名称
    class_names = ["负面", "正面"]

    # max_seq_len=15: 最大序列长度
    #   超过此长度的文本截断，不足的用0填充(padding)
    #   为什么15？合成数据每条5-15词，15覆盖所有样本
    #   真实任务: 短文本20-50，长文本200-512
    max_seq_len = 15

    # num_samples=3000: 样本总数
    num_samples = 3000

    # train_ratio=0.7: 训练集比例
    train_ratio = 0.7

    # val_ratio=0.15: 验证集比例
    val_ratio = 0.15

    # pad_idx=0: 填充词的索引
    #   为什么0？0通常作为<PAD>占位符
    pad_idx = 0

    # --- 模型相关 ---
    # embedding_dim=64: 词嵌入维度
    #   为什么64？小词表(200)用64维已足够编码语义
    #   真实任务: 128-300维，Word2Vec用300维，GloVe用100-300维
    embedding_dim = 64

    # hidden_dim=128: LSTM隐藏层维度
    #   为什么128？文本分类需要更大的隐藏维度捕获语义
    #   比时间序列(64)大，因为文本语义更复杂
    hidden_dim = 128

    # num_layers=2: LSTM堆叠层数
    num_layers = 2

    # dropout=0.3: Dropout比例
    #   比时间序列(0.2)大，因为文本分类更容易过拟合
    dropout = 0.3

    # bidirectional=True: 是否使用双向LSTM
    #   【双向LSTM原理】
    #   正向LSTM: 从左到右处理(看到前面的词)
    #   反向LSTM: 从右到左处理(看到后面的词)
    #   拼接两个方向的隐藏状态 → 同时考虑前后文
    #   为什么文本分类用双向？分类需要理解整句话的语义
    #   注意: 预测/生成任务通常用单向(不能看到未来)
    bidirectional = True

    # --- 训练相关 ---
    # batch_size=64: 批次大小
    batch_size = 64

    # learning_rate=1e-3: 初始学习率
    learning_rate = 1e-3

    # epochs=50: 最大训练轮数
    epochs = 50

    # weight_decay=1e-4: L2正则化
    weight_decay = 1e-4

    # --- 早停策略 ---
    early_stop_patience = 10

    # --- 学习率调度器 ---
    scheduler_type = "cosine"

    # --- 梯度裁剪 ---
    max_grad_norm = 1.0

    # --- 混合精度训练(AMP) ---
    use_amp = True

    # --- 数据加载优化 ---
    num_workers = min(4, os.cpu_count() or 1)

    # --- 保存相关 ---
    save_dir = "lstm/output/text_classification"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 数据生成与加载
# ============================================================
def generate_text_data(cfg):
    """
    生成合成情感分析数据。

    【合成策略】
    - 词汇表: 索引1-100为正面词(如"好"、"棒"、"喜欢")
    -         索引101-200为负面词(如"差"、"烂"、"讨厌")
    - 正面评论: 60%-90%的词来自正面词表，10%-40%来自负面词表
    - 负面评论: 60%-90%的词来自负面词表，10%-40%来自正面词表
    - 这种模拟保留了文本分类的核心特性：
      词频与类别的关联、序列长度变化、噪声词的干扰
    """
    np.random.seed(42)

    # 正面词索引: 1-100 (0保留给<PAD>)
    positive_words = np.arange(1, 101)
    # 负面词索引: 101-200
    negative_words = np.arange(101, 201)

    texts = []
    labels = []

    for _ in range(cfg.num_samples):
        # 随机选择标签
        label = np.random.randint(0, 2)

        # 随机文本长度(5-15)
        text_len = np.random.randint(5, cfg.max_seq_len + 1)

        if label == 1:  # 正面
            # 60%-90%正面词
            pos_ratio = np.random.uniform(0.6, 0.9)
            n_pos = max(1, int(text_len * pos_ratio))
            n_neg = text_len - n_pos
            pos_words = np.random.choice(positive_words, n_pos, replace=True)
            neg_words = np.random.choice(negative_words, n_neg, replace=True)
            words = np.concatenate([pos_words, neg_words])
        else:  # 负面
            pos_ratio = np.random.uniform(0.6, 0.9)
            n_neg = max(1, int(text_len * pos_ratio))
            n_pos = text_len - n_neg
            pos_words = np.random.choice(positive_words, n_pos, replace=True)
            neg_words = np.random.choice(negative_words, n_neg, replace=True)
            words = np.concatenate([pos_words, neg_words])

        # 打乱词序(模拟自然语言)
        np.random.shuffle(words)
        texts.append(words)
        labels.append(label)

    labels = np.array(labels)
    return texts, labels


class TextDataset(Dataset):
    """
    文本分类数据集。

    【文本数据的预处理】
    1. 截断: 超过max_seq_len的部分截掉
    2. 填充(Padding): 不足max_seq_len的部分用pad_idx填充
    3. 为什么要统一长度？LSTM需要batch处理，batch内序列必须等长
    """

    def __init__(self, texts, labels, max_seq_len, pad_idx=0):
        """
        参数:
            texts: 词索引列表，每个元素是numpy数组
            labels: 标签数组
            max_seq_len: 最大序列长度
            pad_idx: 填充值
        """
        self.texts = texts
        self.labels = labels
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 截断或填充
        if len(text) > self.max_seq_len:
            text = text[:self.max_seq_len]
        else:
            # 右侧填充(后补0)
            text = np.pad(text, (0, self.max_seq_len - len(text)),
                          constant_values=self.pad_idx)

        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def get_dataloaders(cfg):
    """生成数据并创建DataLoader。"""
    texts, labels = generate_text_data(cfg)
    print(f"生成文本数据: {len(texts)}条, 词汇表大小: {cfg.vocab_size}")

    # 按顺序划分(虽然文本数据可以随机划分，但保持一致性)
    n = len(texts)
    train_end = int(n * cfg.train_ratio)
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))

    train_dataset = TextDataset(texts[:train_end], labels[:train_end], cfg.max_seq_len, cfg.pad_idx)
    val_dataset = TextDataset(texts[train_end:val_end], labels[train_end:val_end], cfg.max_seq_len, cfg.pad_idx)
    test_dataset = TextDataset(texts[val_end:], labels[val_end:], cfg.max_seq_len, cfg.pad_idx)

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

    print(f"训练集: {len(train_dataset)}条 | 验证集: {len(val_dataset)}条 | 测试集: {len(test_dataset)}条")

    return train_loader, val_loader, test_loader


# ============================================================
# Step 4: 模型定义
# ============================================================
class LSTMTextClassifier(nn.Module):
    """
    LSTM文本分类模型。

    【架构设计】
    输入 (batch, seq_len)  ← 词索引序列
      → Embedding → (batch, seq_len, embedding_dim)
      → 双向LSTM → (batch, seq_len, hidden_dim*2)
      → 取最后时间步/最大池化 → (batch, hidden_dim*2)
      → FC → (batch, num_classes)

    【双向LSTM的隐藏状态】
    正向: h_1→, h_2→, ..., h_T→  (从左到右)
    反向: h_1←, h_2←, ..., h_T←  (从右到左)
    拼接: h_t = [h_t→; h_t←]  (2*hidden_dim维)

    【聚合策略: 取最后时间步 vs 最大池化】
    - 最后时间步: 只用h_T，简单但可能丢失中间信息
    - 最大池化: 对所有时间步取最大值，保留最显著的特征
    - 本模板使用最大池化：对文本分类更有效
    """

    def __init__(self, cfg):
        super().__init__()

        # 词嵌入层
        # padding_idx=pad_idx: <PAD>的嵌入始终为0，不参与梯度更新
        # 好处: (1)减少参数量 (2)<PAD>不会影响语义
        self.embedding = nn.Embedding(
            num_embeddings=cfg.vocab_size + 1,  # +1因为索引从0开始
            embedding_dim=cfg.embedding_dim,
            padding_idx=cfg.pad_idx,
        )

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
            bidirectional=cfg.bidirectional,
        )

        # 双向LSTM的输出维度是 hidden_dim * 2
        lstm_output_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

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
            elif "embedding" in name and "weight" in name:
                nn.init.normal_(param, mean=0, std=0.01)

    def forward(self, x):
        """
        前向传播

        参数:
            x: (batch, seq_len) 词索引序列
        返回:
            logits: (batch, num_classes) 分类得分
        """
        # 词嵌入
        emb = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # LSTM编码
        lstm_out, _ = self.lstm(emb)  # (batch, seq_len, hidden_dim*2)

        # 聚合: 最大池化(对所有时间步取最大值)
        # 【为什么用最大池化而不是取最后时间步？】
        # 最后时间步: 只看最后一个词的隐藏状态，可能丢失中间关键信息
        # 最大池化: 每个维度取所有时间步的最大值，保留最显著特征
        # 例: "这部电影很无聊但演员不错" → "不错"在中间，最后时间步看不到
        pooled = lstm_out.max(dim=1)[0]  # (batch, hidden_dim*2)

        # 分类
        logits = self.classifier(pooled)  # (batch, num_classes)

        return logits


# ============================================================
# Step 5: 训练函数
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, cfg, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for texts, labels in loader:
        texts = texts.to(cfg.device)
        labels = labels.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(texts)
            loss = criterion(logits, labels)

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
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += texts.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, cfg):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for texts, labels in loader:
        texts = texts.to(cfg.device)
        labels = labels.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(texts)
            loss = criterion(logits, labels)

        total_loss += loss.item() * texts.size(0)
        _, preds = logits.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc, all_preds, all_labels


def train(model, train_loader, val_loader, cfg):
    """完整训练流程"""
    criterion = nn.CrossEntropyLoss()
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
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    use_amp = cfg.use_amp and cfg.device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")
    print(f"设备: {cfg.device} | 优化器: Adam(lr={cfg.learning_rate}) | 双向LSTM: {cfg.bidirectional}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg, amp_scaler,
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, cfg)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if cfg.scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
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
        print(f"\n✓ 已恢复最优模型 (Val Loss: {best_val_loss:.4f})")

    return model, history


# ============================================================
# Step 6: 可视化函数
# ============================================================
def plot_training_curves(history, cfg):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("训练/验证损失曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("训练/验证准确率曲线")
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
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=cfg.class_names, yticklabels=cfg.class_names,
           ylabel="真实类别", xlabel="预测类别",
           title="混淆矩阵")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 混淆矩阵已保存: {save_path}")
    plt.close()


def plot_embedding_visualization(model, cfg, num_words=50):
    """
    可视化词嵌入空间。

    【原理】
    训练后，语义相近的词嵌入向量应该距离较近。
    用t-SNE将高维嵌入降到2维，观察词的聚类情况。
    """
    from sklearn.manifold import TSNE

    model.eval()
    # 获取嵌入权重
    embed_weights = model.embedding.weight.detach().cpu().numpy()
    # 排除padding
    embed_weights = embed_weights[1:num_words + 1]

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, num_words - 1))
    emb_2d = tsne.fit_transform(embed_weights)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ["red" if i < 50 else "blue" for i in range(num_words)]
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=50, alpha=0.7)
    for i in range(num_words):
        label = f"正{i+1}" if i < 50 else f"负{i-49}"
        ax.annotate(label, (emb_2d[i, 0], emb_2d[i, 1]), fontsize=7)

    ax.set_title("词嵌入空间可视化(红色=正面词, 蓝色=负面词)", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "embedding_tsne.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 词嵌入可视化已保存: {save_path}")
    plt.close()


# ============================================================
# Step 7: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("LSTM 文本分类 - 情感分析")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # 加载数据
    print("\n生成合成情感分析数据...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # 创建模型
    model = LSTMTextClassifier(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: LSTMTextClassifier")
    print(f"总参数量: {total_params:,}")
    print(f"双向LSTM: {cfg.bidirectional}")
    print(f"\n模型结构:\n{model}")

    # 训练
    model, history = train(model, train_loader, val_loader, cfg)

    # 测试集评估
    print(f"\n{'='*60}")
    print("测试集评估...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, cfg)

    print(f"测试集 Loss: {test_loss:.4f} | 准确率: {test_acc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=cfg.class_names, digits=4))

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "lstm_text_classifier.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
    }, model_path)
    print(f"✓ 模型已保存: {model_path}")

    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, cfg)
    plot_confusion_matrix(y_true, y_pred, cfg)
    plot_embedding_visualization(model, cfg)

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
