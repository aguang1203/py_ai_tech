"""
=============================================================================
RNN 文本生成任务模板 (Character-Level RNN for Text Generation)
=============================================================================

【原理】
文本生成是RNN最具创意的应用——模型学习文本的统计规律，然后逐字符/逐词生成
新的文本。这就是ChatGPT等大语言模型的雏形(虽然它们用的是Transformer)。

核心思想：给定之前的字符，预测下一个字符的概率分布
  输入: "你好世" → 模型 → 输出: "界"的概率最高 → 生成: "界"

生成过程：
1. 给定一个起始文本(种子)
2. 模型预测下一个字符的概率分布
3. 从概率分布中采样一个字符(有随机性，所以每次生成不同)
4. 将采样的字符添加到序列末尾
5. 重复2-4，直到达到最大长度或遇到结束符

【字符级 vs 词级生成】
- 字符级: 每次生成1个字符，词汇表小(通常<1000)，但需要学更多步
- 词级: 每次生成1个词，词汇表大(通常>10000)，但语义更明确
- 本模板使用字符级，因为:
  1. 词汇表小，训练快
  2. 可以生成训练集中没有的新词
  3. 更直观地理解RNN的序列生成过程

【三种采样策略】
1. 贪心(Greedy): 每次选概率最高的字符 → 确定性输出，缺乏多样性
2. 随机(Random): 按概率随机采样 → 多样性好，但可能不连贯
3. 温度(Temperature): 控制概率分布的"尖锐度" → 平衡多样性和连贯性
   - 温度低(0.5): 更确定，输出保守
   - 温度高(1.5): 更随机，输出创意

【应用场景】
- 姓名生成 (中文姓名，本模板使用)
- 诗歌/歌词创作
- 代码补全
- 对话生成
- 故事续写

【本数据集: 合成中文姓名】
- 合成500个常见中文姓名
- 姓: 王、李、张、刘、陈、杨、赵、黄、周、吴
- 名: 伟、芳、秀英、敏、静、丽、强、磊、洋、勇...
- 目标: 学习中文姓名的字符组合模式，生成新的姓名

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python rnn/text_generation.py
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

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
    # num_names=500: 合成姓名数量
    num_names = 500

    # max_name_length=5: 最大姓名长度(含起始/结束符)
    #   中文姓名通常2-4个字，加上起始和结束符，5足够
    max_name_length = 5

    # random_state=42: 随机种子
    random_state = 42

    # --- 模型相关 ---
    # rnn_type="lstm": RNN类型
    #   文本生成推荐LSTM或GRU，原始RNN效果差
    rnn_type = "lstm"

    # hidden_size=128: 隐藏状态维度
    #   字符级生成任务，128足够
    hidden_size = 128

    # num_layers=2: RNN层数
    num_layers = 2

    # embedding_dim=32: 字符嵌入维度
    #   字符级词汇表小(~50)，32维足够表示字符关系
    embedding_dim = 32

    # dropout_rate=0.1: Dropout比例
    #   生成任务Dropout不宜太大，0.1-0.2
    dropout_rate = 0.1

    # --- 训练相关 ---
    batch_size = 32

    learning_rate = 5e-3

    # 为什么5e-3？字符级生成任务通常收敛快，可以用稍大的LR
    # 而且词汇表小，模型不需要太精细的调优

    epochs = 50

    weight_decay = 1e-5

    # --- 学习率调度器 ---
    scheduler_type = "step"

    lr_step_size = 15

    lr_gamma = 0.5

    # --- 梯度裁剪 ---
    max_grad_norm = 5.0

    # --- 生成相关 ---
    # temperature=0.8: 采样温度
    #   <1: 更保守，倾向于高概率字符
    #   =1: 按原始概率采样
    #   >1: 更随机，低概率字符也有机会
    temperature = 0.8

    # num_generate=20: 生成姓名数量
    num_generate = 20

    # seed_text="<S>": 生成起始符
    seed_text = "<S>"

    # --- 保存相关 ---
    save_dir = "rnn/output/text_generation"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 合成数据和数据加载
# ============================================================
def generate_name_data(cfg):
    """
    生成合成的中文姓名数据。

    【数据结构】
    每个姓名前后添加特殊符号:
    - <S>: 起始符(Start) — 告诉模型"姓名开始了"
    - <E>: 结束符(End)   — 告诉模型"姓名结束了"

    例: "王伟" → "<S>王伟<E>"

    【为什么要起始/结束符？】
    - <S>: 生成时以它开头，模型知道要开始写姓名
    - <E>: 训练时模型学到遇到<E>就停止，不会无限生成
    - 这是序列生成任务的标准做法

    【训练数据的输入-输出对】
    对于 "<S>王伟<E>"，生成训练对:
    输入: "<S>"      → 目标: "王"
    输入: "<S>王"    → 目标: "伟"
    输入: "<S>王伟"  → 目标: "<E>"

    即: 给定前面的字符，预测下一个字符
    """
    random.seed(cfg.random_state)

    # 常见姓氏
    surnames = ["王", "李", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴",
                "徐", "孙", "胡", "朱", "高", "林", "何", "郭", "马", "罗"]

    # 常见名字用字
    name_chars = ["伟", "芳", "秀", "英", "敏", "静", "丽", "强", "磊", "洋",
                  "勇", "艳", "杰", "涛", "明", "超", "秀英", "华", "慧", "建",
                  "文", "平", "志", "国", "军", "辉", "鹏", "飞", "雪", "婷",
                  "玉", "兰", "鑫", "阳", "博", "宇", "浩", "然", "思", "怡"]

    names = []
    for _ in range(cfg.num_names):
        surname = random.choice(surnames)
        # 1字名或2字名
        if random.random() < 0.4:
            # 1字名: 如"王伟"
            name = random.choice(name_chars[:20])
        else:
            # 2字名: 如"张秀英"
            name = random.choice(name_chars[:20]) + random.choice(name_chars[20:])
        full_name = surname + name
        names.append(full_name)

    # 构建字符词汇表
    # 特殊符号
    all_chars = ["<PAD>", "<S>", "<E>"]
    char_set = set()
    for name in names:
        for ch in name:
            char_set.add(ch)
    all_chars.extend(sorted(char_set))

    char2idx = {ch: i for i, ch in enumerate(all_chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}

    # 编码姓名: 添加<S>和<E>
    encoded_names = []
    for name in names:
        encoded = [char2idx["<S>"]] + [char2idx[ch] for ch in name] + [char2idx["<E>"]]
        # padding到max_name_length
        if len(encoded) < cfg.max_name_length:
            encoded += [char2idx["<PAD>"]] * (cfg.max_name_length - len(encoded))
        encoded = encoded[:cfg.max_name_length]
        encoded_names.append(encoded)

    return encoded_names, char2idx, idx2char, names


class NameDataset(Dataset):
    """姓名生成数据集。"""

    def __init__(self, encoded_names):
        """
        参数:
            encoded_names: 编码后的姓名列表
        """
        self.data = encoded_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回 (输入序列, 目标序列)

        【输入-目标对构造】
        对于序列 [S, 王, 伟, E, PAD]:
        输入:  [S, 王, 伟, E, PAD]
        目标:  [王, 伟, E, PAD, PAD]

        即目标序列是输入序列左移一位
        模型学到: 给定位置t及之前的字符，预测位置t+1的字符
        """
        seq = self.data[idx]
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq[1:], dtype=torch.long)
        return input_seq, target_seq


def get_dataloaders(cfg):
    """生成合成数据并创建DataLoader。"""
    encoded_names, char2idx, idx2char, raw_names = generate_name_data(cfg)

    # 保存词汇表到cfg
    cfg.char2idx = char2idx
    cfg.idx2char = idx2char
    cfg.vocab_size = len(char2idx)

    # 划分训练/测试集
    n_total = len(encoded_names)
    n_test = int(n_total * 0.2)
    n_train = n_total - n_test

    train_dataset = NameDataset(encoded_names[:n_train])
    test_dataset = NameDataset(encoded_names[n_train:])

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
    )

    print(f"训练集: {n_train}个姓名 | 测试集: {n_test}个姓名")
    print(f"词汇表大小: {cfg.vocab_size} (含特殊符号)")
    print(f"示例姓名: {raw_names[:5]}")

    return train_loader, test_loader


# ============================================================
# Step 4: 模型定义
# ============================================================
class CharRNN(nn.Module):
    """
    字符级RNN文本生成模型

    【架构设计思路】
    输入: (batch, seq_len-1)  ← 字符索引序列(不含最后一个)
      → Embedding: (batch, seq_len-1, embedding_dim)  ← 字符嵌入
      → LSTM: (batch, seq_len-1, hidden_size)         ← 逐字符编码
      → FC: (batch, seq_len-1, vocab_size)             ← 每个位置预测下一个字符

    【为什么每个时间步都输出预测？】
    - 分类任务: 只需要最后的隐藏状态做分类
    - 生成任务: 每个时间步都要预测下一个字符
    - 输入"S王伟E" → 输出 [王,伟,E,...] 的概率分布
    - 损失: 对每个时间步的预测都计算交叉熵

    【参数量计算 (LSTM)】
    Embedding: vocab_size × embedding_dim ≈ 50×32 = 1,600
    LSTM第1层: 4 × [(32 + 128 + 1) × 128] = 8,256
    LSTM第2层: 4 × [(128 + 128 + 1) × 128] = 131,584
    FC: 128 × vocab_size ≈ 128×50 = 6,400
    总计 ≈ 148K
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_layers

        # ---- 字符嵌入层 ----
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim, padding_idx=0)

        # ---- RNN层 ----
        if cfg.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=cfg.embedding_dim,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=cfg.dropout_rate if cfg.num_layers > 1 else 0,
            )
        elif cfg.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=cfg.embedding_dim,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=cfg.dropout_rate if cfg.num_layers > 1 else 0,
            )

        # ---- 输出层 ----
        # 将隐藏状态映射到词汇表大小的logits
        self.fc = nn.Linear(cfg.hidden_size, cfg.vocab_size)

        self._init_weights()

    def _init_weights(self):
        """权重初始化。"""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[0].zero_()  # PAD位置为0

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

    def forward(self, x, hidden=None):
        """
        前向传播

        参数:
            x: 输入字符索引 (batch, seq_len)
            hidden: 上一时间步的隐藏状态(生成时需要保持)

        数据流动:
        x: (batch, seq_len)           ← 字符索引序列
          → embedding: (batch, seq_len, embedding_dim)
          → rnn: (batch, seq_len, hidden_size)
          → fc: (batch, seq_len, vocab_size)  ← 每个位置的字符概率
        """
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        if hidden is None:
            if isinstance(self.rnn, nn.LSTM):
                hidden = (torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device),
                          torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))
            else:
                hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        rnn_out, hidden = self.rnn(embedded, hidden)
        # rnn_out: (batch, seq_len, hidden_size)

        # 对每个时间步的输出做预测
        output = self.fc(rnn_out)  # (batch, seq_len, vocab_size)

        return output, hidden

    def generate(self, seed_idx, max_length, temperature=1.0, device="cpu"):
        """
        生成文本(自回归)。

        【自回归生成过程】
        1. 输入种子字符<S>
        2. 模型预测下一个字符的概率分布
        3. 按温度缩放后采样
        4. 将采样的字符作为下一步的输入
        5. 重复直到遇到<E>或达到最大长度

        【温度(Temperature)的作用】
        原始logits → 除以temperature → softmax → 概率

        temperature < 1: 概率分布更尖锐 → 倾向选高概率字符 → 保守
        temperature = 1: 原始概率分布
        temperature > 1: 概率分布更平坦 → 低概率字符也有机会 → 创意

        参数:
            seed_idx: 起始字符索引(通常是<S>)
            max_length: 最大生成长度
            temperature: 采样温度
            device: 计算设备
        """
        self.eval()
        generated = [seed_idx]
        hidden = None

        current_idx = seed_idx

        with torch.no_grad():
            for _ in range(max_length):
                # 当前字符作为输入
                x = torch.tensor([[current_idx]], dtype=torch.long).to(device)

                # 前向传播
                output, hidden = self(x, hidden)
                # output: (1, 1, vocab_size)

                # 取最后一个时间步的输出
                logits = output[0, -1, :] / temperature

                # softmax转为概率
                probs = torch.softmax(logits, dim=0)

                # 按概率采样
                next_idx = torch.multinomial(probs, num_samples=1).item()

                # 如果遇到结束符，停止生成
                if next_idx == self.cfg.char2idx.get("<E>", -1):
                    break

                # 如果遇到PAD，跳过
                if next_idx == 0:
                    break

                generated.append(next_idx)
                current_idx = next_idx

        return generated


# ============================================================
# Step 5: 训练函数
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, cfg):
    """训练一个epoch。"""
    model.train()
    total_loss = 0
    total_chars = 0

    for input_seq, target_seq in loader:
        input_seq, target_seq = input_seq.to(cfg.device), target_seq.to(cfg.device)

        # 前向传播
        output, _ = model(input_seq)
        # output: (batch, seq_len, vocab_size)

        # 计算损失
        # 重塑为 (batch*seq_len, vocab_size) 和 (batch*seq_len,)
        loss = criterion(output.reshape(-1, cfg.vocab_size), target_seq.reshape(-1))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        # 统计(只计算非PAD位置)
        mask = (target_seq != 0).float()
        total_loss += loss.item() * mask.sum().item()
        total_chars += mask.sum().item()

    avg_loss = total_loss / max(total_chars, 1)
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, criterion, cfg):
    """评估模型性能。"""
    model.eval()
    total_loss = 0
    total_chars = 0

    for input_seq, target_seq in loader:
        input_seq, target_seq = input_seq.to(cfg.device), target_seq.to(cfg.device)

        output, _ = model(input_seq)
        loss = criterion(output.reshape(-1, cfg.vocab_size), target_seq.reshape(-1))

        mask = (target_seq != 0).float()
        total_loss += loss.item() * mask.sum().item()
        total_chars += mask.sum().item()

    avg_loss = total_loss / max(total_chars, 1)
    # 困惑度(Perplexity): 衡量模型对数据的"惊讶程度"
    # PPL = exp(loss)，越低越好
    # PPL=1: 模型完美预测每个字符
    # PPL=vocab_size: 模型等于随机猜
    perplexity = np.exp(avg_loss)

    return avg_loss, perplexity


def train(model, train_loader, test_loader, cfg):
    """完整训练流程。"""
    # 损失函数: 交叉熵(ignore_index=0表示忽略PAD)
    # 【为什么ignore_index=0？】
    # PAD位置的预测没有意义，不应该参与损失计算
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    if cfg.scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history = {"train_loss": [], "test_loss": [], "perplexity": []}

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")
    print(f"设备: {cfg.device} | 优化器: Adam(lr={cfg.learning_rate})")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg)
        test_loss, perplexity = evaluate(model, test_loader, criterion, cfg)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["perplexity"].append(perplexity)

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | PPL: {perplexity:.2f} | "
                  f"LR: {current_lr:.6f}")

            # 每隔5轮生成示例
            print("  生成示例:")
            for _ in range(3):
                generated_idx = model.generate(
                    cfg.char2idx["<S>"],
                    cfg.max_name_length,
                    temperature=cfg.temperature,
                    device=cfg.device,
                )
                name = "".join(cfg.idx2char.get(i, "") for i in generated_idx
                              if i not in [0, cfg.char2idx["<S>"], cfg.char2idx.get("<E>", -1)])
                print(f"    → {name}")

        scheduler.step()

    return model, history


# ============================================================
# Step 6: 可视化函数
# ============================================================
def plot_training_curves(history, cfg):
    """绘制训练曲线。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["test_loss"], "r-", label="Test Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("文本生成训练/测试损失曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["perplexity"], "g-", label="Perplexity", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PPL")
    ax2.set_title("困惑度(Perplexity)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()


# ============================================================
# Step 7: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("RNN 文本生成 - 中文姓名生成")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # 加载数据
    print("\n生成合成数据...")
    train_loader, test_loader = get_dataloaders(cfg)

    # 创建模型
    model = CharRNN(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型: CharRNN ({cfg.rnn_type.upper()})")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"\n模型结构:\n{model}")

    # 训练
    model, history = train(model, train_loader, test_loader, cfg)

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "char_rnn.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "char2idx": cfg.char2idx,
        "idx2char": cfg.idx2char,
    }, model_path)
    print(f"✓ 模型已保存: {model_path}")

    # 最终生成
    print(f"\n{'='*60}")
    print(f"生成 {cfg.num_generate} 个新姓名:")
    print(f"{'='*60}")

    temperatures = [0.5, 0.8, 1.2]
    for temp in temperatures:
        print(f"\n--- 温度 = {temp} ({'保守' if temp < 1 else '平衡' if temp == 1 else '创意'}) ---")
        for i in range(cfg.num_generate // len(temperatures)):
            generated_idx = model.generate(
                cfg.char2idx["<S>"],
                cfg.max_name_length,
                temperature=temp,
                device=cfg.device,
            )
            name = "".join(cfg.idx2char.get(i, "") for i in generated_idx
                          if i not in [0, cfg.char2idx["<S>"], cfg.char2idx.get("<E>", -1)])
            print(f"  {i+1}. {name}")

    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, cfg)

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
