"""
=============================================================================
LSTM 序列生成任务模板 (Sequence Generation with LSTM)
=============================================================================

【原理】
序列生成是LSTM最经典的应用之一：给定一段文本的前缀，自动生成后续内容。
核心思想是"自回归"(Autoregressive)——每一步的预测结果成为下一步的输入。

【自回归生成流程】
  输入: "今天天气"
  Step 1: 输入[今,天,天,气] → LSTM → 预测"很"
  Step 2: 输入[天,天,气,很] → LSTM → 预测"好"
  Step 3: 输入[天,气,很,好] → LSTM → 预测"<END>"
  输出: "今天天气很好"

【训练 vs 推理的区别】
  训练时: Teacher Forcing — 每步输入真实答案(而非模型预测)
    为什么？如果某步预测错了，后续都基于错误预测，训练效率极低
    Teacher Forcing让模型总是基于正确的上文学习，收敛更快

  推理时: Free Running — 每步输入上一步的预测结果
    为什么？推理时没有真实答案可用，只能用自己的预测

【温度采样(Temperature Sampling)】
  控制生成文本的随机性：
    温度低(如0.5): 更确定性的输出(保守，重复性高)
    温度高(如1.5): 更随机的输出(创意性强，但可能不连贯)
    温度=1.0: 标准采样

  原理: softmax(z/T)，T越大分布越平坦，T越小分布越尖锐

【应用场景】
- 文本生成(故事/诗歌/代码)
- 音乐生成
- 对话系统(聊天机器人)
- 代码补全
- 音乐/旋律生成

【本数据集: 合成模式序列】
- 词汇表: 20个"词"(数字0-19)
- 模式: 重复的数列模式(如0,1,2,3,4,0,1,2,3,4,...)
- 嵌入噪声使任务有挑战性
- 即时生成，无需下载
- 特点: 保留序列生成的核心特性(模式学习、自回归、采样策略)

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python lstm/sequence_generation.py
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
    """超参数配置中心 —— 序列生成任务的所有可调参数。"""

    # --- 数据相关 ---
    # vocab_size=20: 词汇表大小(数字0-19)
    #   小词表便于快速训练和可视化
    #   真实文本生成: 10000-50000
    vocab_size = 20

    # seq_len=16: 训练序列长度
    #   每个训练样本是长度为seq_len+1的序列
    #   输入: 前seq_len个词，目标: 后seq_len个词(错位1步)
    seq_len = 16

    # num_sequences=5000: 生成的训练序列数
    num_sequences = 5000

    # train_ratio=0.8: 训练集比例
    train_ratio = 0.8

    # --- 模型相关 ---
    # embedding_dim=32: 词嵌入维度
    #   为什么比文本分类(64)小？词表只有20个词，32维足够
    embedding_dim = 32

    # hidden_dim=128: LSTM隐藏层维度
    hidden_dim = 128

    # num_layers=2: LSTM堆叠层数
    num_layers = 2

    # dropout=0.2: Dropout比例
    dropout = 0.2

    # --- 生成相关 ---
    # gen_length=50: 生成序列长度
    gen_length = 50

    # temperature=0.8: 采样温度
    #   <1: 更确定性(保守)
    #   =1: 标准采样
    #   >1: 更随机(创意)
    temperature = 0.8

    # num_samples_gen=5: 生成演示的样本数
    num_samples_gen = 5

    # --- 训练相关 ---
    # batch_size=64: 批次大小
    batch_size = 64

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

    # --- 保存相关 ---
    save_dir = "lstm/output/sequence_generation"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 数据生成与加载
# ============================================================
def generate_sequence_data(cfg):
    """
    生成合成模式序列数据。

    【合成策略】
    生成带有可学习模式的数字序列：
    模式1: 递增序列 0,1,2,3,4,0,1,2,3,4,... (周期=5)
    模式2: 递增序列 0,1,2,3,4,5,6,7,...,0,1,... (周期=8)
    模式3: 交替序列 0,5,1,6,2,7,3,8,... (双序列交织)

    加入轻微噪声(10%概率随机替换)使任务有挑战性。
    模型需要学习这些模式才能正确预测下一个词。
    """
    np.random.seed(42)

    sequences = []
    pattern_types = []

    for i in range(cfg.num_sequences):
        # 随机选择模式类型
        pattern = np.random.randint(0, 3)

        if pattern == 0:
            # 模式1: 周期5递增 (0,1,2,3,4,0,1,2,3,4,...)
            base_seq = [j % 5 for j in range(cfg.seq_len + 1)]
        elif pattern == 1:
            # 模式2: 周期8递增 (0,1,...,7,0,1,...,7,...)
            start = np.random.randint(0, 8)
            base_seq = [(start + j) % 8 for j in range(cfg.seq_len + 1)]
        else:
            # 模式3: 双序列交织 (0,10,1,11,2,12,...)
            start = np.random.randint(0, 5)
            base_seq = []
            for j in range(cfg.seq_len + 1):
                if j % 2 == 0:
                    base_seq.append((start + j // 2) % 10)
                else:
                    base_seq.append(10 + (start + j // 2) % 10)

        # 添加噪声: 10%概率替换为随机词
        noisy_seq = []
        for token in base_seq:
            if np.random.random() < 0.1:
                noisy_seq.append(np.random.randint(0, cfg.vocab_size))
            else:
                noisy_seq.append(min(token, cfg.vocab_size - 1))

        sequences.append(noisy_seq)
        pattern_types.append(pattern)

    return sequences, pattern_types


class SequenceDataset(Dataset):
    """
    序列生成数据集。

    【序列生成的数据格式】
    输入序列: [x_0, x_1, ..., x_{T-1}]
    目标序列: [x_1, x_2, ..., x_T]

    目标是输入序列错位1步——模型学习"给定前文，预测下一个词"。
    """

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # 输入: 前 seq_len 个词
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        # 目标: 后 seq_len 个词(错位1步)
        target_seq = torch.tensor(seq[1:], dtype=torch.long)
        return input_seq, target_seq


def get_dataloaders(cfg):
    """生成数据并创建DataLoader。"""
    sequences, pattern_types = generate_sequence_data(cfg)
    print(f"生成序列数据: {len(sequences)}条, 词汇表大小: {cfg.vocab_size}")

    n = len(sequences)
    train_end = int(n * cfg.train_ratio)

    train_dataset = SequenceDataset(sequences[:train_end])
    val_dataset = SequenceDataset(sequences[train_end:])

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

    print(f"训练集: {len(train_dataset)}条 | 验证集: {len(val_dataset)}条")

    return train_loader, val_loader


# ============================================================
# Step 4: 模型定义
# ============================================================
class LSTMGenerator(nn.Module):
    """
    LSTM序列生成模型。

    【架构设计】
    输入 (batch, seq_len)  ← 词索引序列
      → Embedding → (batch, seq_len, embedding_dim)
      → LSTM → (batch, seq_len, hidden_dim)
      → FC → (batch, seq_len, vocab_size)
      → softmax → 每个位置的词概率分布

    【训练 vs 推理的输出】
    训练: 输出所有位置的logits，计算交叉熵
    推理: 逐步生成，每步只取最后一个位置的预测
    """

    def __init__(self, cfg):
        super().__init__()

        self.vocab_size = cfg.vocab_size

        # 词嵌入层
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
        )

        # 输出层: 映射到词表大小
        self.fc = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

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

    def forward(self, x, hidden=None):
        """
        前向传播(训练模式)

        参数:
            x: (batch, seq_len) 词索引序列
            hidden: 初始隐藏状态(可选)
        返回:
            logits: (batch, seq_len, vocab_size) 每个位置的词概率
            hidden: 最后的隐藏状态
        """
        emb = self.embedding(x)
        lstm_out, hidden = self.lstm(emb, hidden)
        logits = self.fc(lstm_out)
        return logits, hidden

    @torch.no_grad()
    def generate(self, start_tokens, length, temperature=1.0, device=None):
        """
        自回归生成序列。

        【生成过程】
        1. 用start_tokens初始化LSTM隐藏状态
        2. 取最后一个位置的输出作为下一步的输入
        3. 用温度采样从概率分布中采样新词
        4. 重复length次

        参数:
            start_tokens: 起始词序列 (1, prefix_len)
            length: 生成长度
            temperature: 采样温度
            device: 设备
        返回:
            generated: 生成的词索引列表(含起始词)
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        generated = start_tokens[0].cpu().numpy().tolist()
        x = start_tokens.to(device)
        hidden = None

        for _ in range(length):
            # 前向传播
            logits, hidden = self(x, hidden)
            # 取最后一个时间步的输出
            next_logits = logits[:, -1, :] / temperature
            # softmax → 概率分布
            probs = torch.softmax(next_logits, dim=-1)
            # 从概率分布中采样
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token[0].cpu().item())
            # 下一步的输入是刚预测的词
            x = next_token

        return generated


# ============================================================
# Step 5: 训练函数
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, cfg, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for inputs, targets in loader:
        inputs = inputs.to(cfg.device)
        targets = targets.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _ = model(inputs)
            # 重塑为(batch*seq_len, vocab_size)和(batch*seq_len,)
            loss = criterion(logits.view(-1, cfg.vocab_size), targets.view(-1))

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
        _, preds = logits.max(-1)
        # 计算准确率(排除padding)
        correct = (preds == targets).sum().item()
        total_correct += correct
        total_tokens += targets.numel()

    return total_loss / len(loader.dataset), total_correct / total_tokens


@torch.no_grad()
def evaluate(model, loader, criterion, cfg):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for inputs, targets in loader:
        inputs = inputs.to(cfg.device)
        targets = targets.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, cfg.vocab_size), targets.view(-1))

        total_loss += loss.item() * inputs.size(0)
        _, preds = logits.max(-1)
        correct = (preds == targets).sum().item()
        total_correct += correct
        total_tokens += targets.numel()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_tokens

    return avg_loss, accuracy


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
    print(f"设备: {cfg.device} | 优化器: Adam(lr={cfg.learning_rate}) | AMP: {use_amp}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg, amp_scaler,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, cfg)

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
    ax2.set_ylabel("Token Accuracy")
    ax2.set_title("训练/验证准确率曲线")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()


def plot_generation_demo(model, cfg):
    """可视化生成演示"""
    fig, axes = plt.subplots(cfg.num_samples_gen, 1, figsize=(14, 3 * cfg.num_samples_gen))
    if cfg.num_samples_gen == 1:
        axes = [axes]

    # 不同的起始序列
    start_options = [
        [0, 1, 2, 3],
        [0, 5, 1, 6],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [3, 4, 0, 1],
        [2, 7, 3, 8],
    ]

    for i, ax in enumerate(axes):
        start = start_options[i % len(start_options)]
        start_tensor = torch.tensor([start], dtype=torch.long)

        # 生成序列
        generated = model.generate(
            start_tensor, cfg.gen_length, temperature=cfg.temperature, device=cfg.device,
        )

        # 可视化: 用颜色区分不同的词
        colors = plt.cm.tab20(np.array(generated) / cfg.vocab_size)
        ax.imshow(colors.reshape(1, -1, 4), aspect="auto")
        ax.set_yticks([])

        # 标注数值
        step_labels = [str(g) for g in generated]
        for j, label in enumerate(step_labels):
            ax.text(j, 0, label, ha="center", va="center", fontsize=6,
                    color="white" if generated[j] > cfg.vocab_size // 2 else "black")

        ax.set_xlabel("时间步")
        prefix_str = ",".join(str(s) for s in start)
        ax.set_title(f"生成 {i+1}: 起始[{prefix_str},...] → 温度={cfg.temperature}")

    plt.suptitle("LSTM序列生成演示", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "generation_demo.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 生成演示已保存: {save_path}")
    plt.close()


def plot_temperature_comparison(model, cfg, start_tokens=[0, 1, 2, 3]):
    """对比不同温度下的生成结果"""
    temperatures = [0.3, 0.5, 0.8, 1.0, 1.5]
    start_tensor = torch.tensor([start_tokens], dtype=torch.long)

    fig, axes = plt.subplots(len(temperatures), 1, figsize=(14, 2.5 * len(temperatures)))

    for i, temp in enumerate(temperatures):
        generated = model.generate(start_tensor, cfg.gen_length, temperature=temp, device=cfg.device)

        colors = plt.cm.tab20(np.array(generated) / cfg.vocab_size)
        axes[i].imshow(colors.reshape(1, -1, 4), aspect="auto")
        axes[i].set_yticks([])
        axes[i].set_title(f"温度={temp}: {generated[:20]}...")

    plt.suptitle("不同温度下的生成对比", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "temperature_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 温度对比已保存: {save_path}")
    plt.close()


# ============================================================
# Step 7: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("LSTM 序列生成")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # 加载数据
    print("\n生成合成模式序列数据...")
    train_loader, val_loader = get_dataloaders(cfg)

    # 创建模型
    model = LSTMGenerator(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: LSTMGenerator")
    print(f"总参数量: {total_params:,}")
    print(f"\n模型结构:\n{model}")

    # 训练
    model, history = train(model, train_loader, val_loader, cfg)

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "lstm_generator.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
    }, model_path)
    print(f"✓ 模型已保存: {model_path}")

    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, cfg)
    plot_generation_demo(model, cfg)
    plot_temperature_comparison(model, cfg)

    # 打印生成示例
    print("\n生成示例:")
    for start in [[0, 1, 2, 3], [0, 5, 1, 6], [5, 6, 7, 0]]:
        start_tensor = torch.tensor([start], dtype=torch.long)
        generated = model.generate(start_tensor, 30, temperature=0.8, device=cfg.device)
        print(f"  起始{start} → 生成{generated}")

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
