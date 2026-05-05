"""
=============================================================================
DNN 图像分类任务模板 (Deep Neural Network for Image Classification)
=============================================================================

【原理】
深度神经网络(DNN)由多层全连接(Linear)层堆叠而成，是最基础的神经网络形式。
与CNN不同，DNN不利用图像的空间结构，而是将图像展平为一维向量输入网络。

核心思想：通过多层非线性变换，将高维输入映射到低维类别空间。
每一层提取不同抽象级别的特征：
  第1层：学习像素级别的简单组合（如边缘、角落）
  中间层：学习局部图案的组合（如笔画、弧形）
  深层：学习完整的数字形状

【DNN vs CNN 在图像分类上的对比】
┌─────────────┬─────────────────────────┬─────────────────────────┐
│    特性     │          DNN            │          CNN            │
├─────────────┼─────────────────────────┼─────────────────────────┤
│ 连接方式    │ 全连接(每个输入连到每个  │ 局部连接(卷积核滑动)     │
│             │ 神经元)                 │                         │
│ 参数量      │ 巨大(784→512→256→10     │ 小(卷积核共享参数)       │
│             │ ≈ 53万参数)             │                         │
│ 空间感知    │ 无(展平后丢失空间结构)   │ 有(保留2D空间关系)       │
│ 平移不变性  │ 无(数字在左上角≠右下角)  │ 有(卷积核全图滑动)       │
│ 训练速度    │ 快(结构简单)             │ 较慢(卷积运算复杂)       │
│ 图像准确率  │ MNIST约97-98%           │ MNIST约99%+             │
└─────────────┴─────────────────────────┴─────────────────────────┘

【为什么图像任务通常不用DNN？】
- 28×28=784维输入，第一层到512神经元 = 784×512≈40万参数
- 而CNN第一层3×3卷积核 = 只有9个参数，少了4万倍
- 更关键的是：DNN展平图像后丢失了"哪些像素相邻"的空间信息

【那为什么还要学DNN图像分类？】
1. 理解基础：DNN是CNN/RNN/Transformer的共同基础
2. 简单快速：结构简单，训练快，适合快速验证想法
3. 特征向量：如果输入已经是提取好的特征向量（非原始图像），DNN很合适

【应用场景】
- 教学演示：理解神经网络基础原理
- 特征分类：输入是已提取的特征向量（如PCA降维后的特征）
- 快速原型：验证数据是否可被神经网络学习
- 低分辨率图像：MNIST等简单图像

【本数据集: MNIST】
- 70,000张 28×28 灰度手写数字图像（训练60,000 + 测试10,000）
- 10个类别（0-9数字）
- 特点：图像简单、数据干净、无需下载外部数据

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python dnn/classification.py
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
# Step 2: 配置超参数
# ============================================================
class CONFIG:
    """超参数配置中心 —— 所有可调参数集中在此，方便统一管理和实验对比。"""

    # --- 数据相关 ---
    # data_dir: 数据集存放目录
    data_dir = "data"

    # num_classes=10: MNIST有10个数字类别
    num_classes = 10

    # class_names: 类别名称（用于可视化）
    class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # image_size=28: MNIST图像尺寸
    image_size = 28

    # flatten_dim=784: 展平后的维度（28×28=784）
    #   DNN需要将2D图像展平为1D向量输入
    flatten_dim = 784

    # test_size=0.1667: 验证集比例（从训练集60,000张中划出10,000张）
    test_size = 0.1667

    # random_state=42: 固定随机种子
    random_state = 42

    # --- 模型相关 ---
    # hidden_dims=[512, 256, 128]: 3层隐藏层，逐层压缩"漏斗"结构
    #   为什么是[512, 256, 128]？
    #     输入784维，第一层512足够宽（约0.65倍输入），能学习丰富的特征组合
    #     逐层减半是经典设计：先宽后窄，提取特征后压缩到决策空间
    #   为什么3层？
    #     太少（1-2层）表达能力不足，太多（5层+）小数据集容易过拟合
    #   注意：DNN的参数量比CNN大得多！
    #     784×512 + 512×256 + 256×128 + 128×10 ≈ 55万参数
    #     而CNN分类器只有约10万参数
    hidden_dims = [512, 256, 128]

    # dropout_rate=0.3: Dropout比例
    #   为什么0.3？DNN全连接层参数多（55万），需要正则化防止过拟合
    #   比CNN的0.5小，因为DNN配合BN使用，正则化需求降低
    dropout_rate = 0.3

    # use_batch_norm=True: 是否使用批归一化
    #   【什么是批归一化？】
    #   对每一层的输入做归一化：减均值、除标准差，再缩放平移
    #   效果：加速训练收敛、允许更大学习率、减轻初始化敏感
    #   为什么DNN特别需要BN？
    #     深层全连接网络容易出现"内部协变量偏移"：每层的输入分布随训练变化
    #     BN固定每层的输入分布，让训练更稳定
    use_batch_norm = True

    # --- 训练相关 ---
    # batch_size=128: 每次梯度更新使用128张图
    #   MNIST图小（28×28×1≈0.8KB），128张≈100KB，显存轻松处理
    batch_size = 128

    # learning_rate=1e-3: Adam的初始学习率
    #   配合BN使用，1e-3是标准值
    learning_rate = 1e-3

    # epochs=30: 最大训练轮数
    #   MNIST简单，DNN通常20-30轮收敛
    epochs = 30

    # weight_decay=1e-4: L2正则化
    #   惩罚大权重，防止过拟合
    weight_decay = 1e-4

    # --- 早停策略 ---
    # early_stop_patience=5: 验证损失连续5轮不下降就停止
    #   MNIST收敛快，5轮足够判断
    early_stop_patience = 5

    # --- 学习率调度器 ---
    # scheduler_type="cosine": 余弦退火
    scheduler_type = "cosine"

    # --- 梯度裁剪 ---
    # max_grad_norm=1.0: 梯度L2范数上限
    max_grad_norm = 1.0

    # --- 混合精度训练(AMP) ---
    # use_amp=True: 启用混合精度
    use_amp = True

    # --- 数据加载优化 ---
    num_workers = min(4, os.cpu_count() or 1)

    # --- 保存相关 ---
    save_dir = "dnn/output/classification"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 数据加载和预处理
# ============================================================
def get_transforms(cfg):
    """
    创建数据变换管道。

    【DNN的预处理与CNN的区别】
    - CNN: 图像保持2D结构，输入shape=(batch, 1, 28, 28)
    - DNN: 图像需要展平为1D向量，输入shape=(batch, 784)
    - torchvision的MNIST数据集默认返回图像张量，我们在DataLoader后手动展平
    """
    # MNIST标准化参数（对训练集统计得出）
    normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


class FlattenMNIST(datasets.MNIST):
    """
    自定义MNIST数据集，在获取数据时自动展平图像。

    【为什么要展平？】
    DNN的输入必须是1D向量，而MNIST原始数据是2D图像(1, 28, 28)。
    我们在__getitem__中将图像展平为(784,)，方便直接输入DNN。
    """
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # 展平: (1, 28, 28) → (784,)
        img = img.view(-1)
        return img, target


def get_dataloaders(cfg):
    """
    加载MNIST数据集并创建DataLoader。

    【DNN的数据处理流程】
    原始图像 (28, 28) → ToTensor → Normalize → 展平 (784,) → 输入DNN
    """
    train_transform, val_transform = get_transforms(cfg)

    # 加载训练集（60,000张）
    train_dataset = FlattenMNIST(
        root=cfg.data_dir, train=True, download=True, transform=train_transform,
    )

    # 加载测试集（10,000张）
    test_dataset = FlattenMNIST(
        root=cfg.data_dir, train=False, download=True, transform=val_transform,
    )

    # 从训练集划出验证集
    n_total = len(train_dataset)
    n_val = int(n_total * cfg.test_size)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(cfg.random_state)
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [n_train, n_val], generator=generator,
    )

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
class DNNClassifier(nn.Module):
    """
    DNN图像分类模型（纯全连接网络）

    【架构设计】
    输入 (784) → Linear(784→512) → [BN] → ReLU → Dropout
              → Linear(512→256) → [BN] → ReLU → Dropout
              → Linear(256→128) → [BN] → ReLU → Dropout
              → Linear(128→10)

    【为什么是"漏斗"形状？】
    输入784维（高维像素空间）→ 中间层逐渐压缩 → 输出10维（低维类别空间）
    这种逐层压缩的结构让网络先学习丰富的特征组合，再聚焦到类别决策。

    【与CNN的关键区别】
    1. DNN: 784个输入神经元，每个连接所有像素（包括远离的像素）
       CNN: 每个神经元只连接局部3×3区域
    2. DNN: 没有参数共享，每个连接独立权重
       CNN: 同一个卷积核在全图共享
    3. DNN: 输入必须固定784维（不能处理不同尺寸图像）
       CNN: 可以用AdaptivePool处理不同尺寸
    """

    def __init__(self, cfg):
        super().__init__()
        dims = [cfg.flatten_dim] + cfg.hidden_dims + [cfg.num_classes]

        layers = []
        for i in range(len(dims) - 1):
            # 全连接层
            layers.append(nn.Linear(dims[i], dims[i+1]))

            # 最后一层不加BN、ReLU、Dropout
            if i < len(dims) - 2:
                if cfg.use_batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(cfg.dropout_rate))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """
        权重初始化: He初始化(Kaiming Normal)

        【为什么用He初始化？】
        ReLU会截断负值，如果权重太小，信号逐层衰减（梯度消失）。
        He初始化让每层输出的方差≈输入的方差，保持信号强度。
        公式: W ~ N(0, sqrt(2/fan_in))
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        前向传播

        输入: (batch, 784) — 展平后的MNIST图像
        输出: (batch, 10) — 每个数字的logits（未归一化分数）
        """
        return self.network(x)


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
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, cfg):
    """评估模型"""
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
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(all_targets)
    acc = accuracy_score(all_targets, all_preds)

    return avg_loss, acc, all_preds, all_targets


def train(model, train_loader, val_loader, cfg):
    """完整训练流程"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    if cfg.scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    use_amp = cfg.use_amp and cfg.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")
    print(f"设备: {cfg.device} | 优化器: Adam | 调度器: {cfg.scheduler_type} | AMP: {use_amp}")

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
                print(f"\n⚠ 早停触发")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(cfg.device)
        print(f"\n✓ 已恢复最优模型")

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
    fig, ax = plt.subplots(figsize=(10, 8))
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
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 混淆矩阵已保存: {save_path}")
    plt.close()


def plot_sample_predictions(model, test_loader, cfg, num_samples=16):
    """可视化预测结果"""
    model.eval()

    # 获取原始MNIST图像用于可视化
    raw_dataset = datasets.MNIST(root=cfg.data_dir, train=False, download=False,
                                  transform=transforms.ToTensor())
    raw_loader = DataLoader(raw_dataset, batch_size=num_samples, shuffle=False)

    images, labels = next(iter(raw_loader))
    # 标准化
    images_norm = (images - 0.1307) / 0.3081
    # 展平用于模型输入
    flat_images = images_norm.view(images_norm.size(0), -1).to(cfg.device)

    with torch.no_grad():
        outputs = model(flat_images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(1).cpu()

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    for i, ax in enumerate(axes.flat):
        if i >= num_samples:
            break
        img = images[i].squeeze().numpy()

        ax.imshow(img, cmap="gray")
        true_name = cfg.class_names[labels[i]]
        pred_name = cfg.class_names[preds[i]]
        confidence = probs[i, preds[i]].item()

        color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(f"真实: {true_name}\n预测: {pred_name} ({confidence:.1%})",
                     color=color, fontsize=9)
        ax.axis("off")

    plt.suptitle("DNN图像分类预测结果", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "sample_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 预测结果已保存: {save_path}")
    plt.close()


def plot_weight_distribution(model, cfg):
    """
    可视化网络权重分布。

    【为什么要看权重分布？】
    - 权重接近0：该神经元几乎不起作用
    - 权重分布均匀：初始化良好
    - 权重都很大或很小：可能存在梯度问题
    """
    n_layers = len(cfg.hidden_dims) + 1
    fig, axes = plt.subplots(1, n_layers, figsize=(3 * n_layers, 3))
    if n_layers == 1:
        axes = [axes]

    layer_idx = 0
    for name, param in model.named_parameters():
        if "weight" in name and "network" in name and len(param.shape) == 2:
            ax = axes[layer_idx]
            weights = param.detach().cpu().numpy().flatten()
            ax.hist(weights, bins=50, alpha=0.7, color="steelblue")
            ax.set_title(f"Layer {layer_idx + 1}\n({param.shape[0]}×{param.shape[1]})")
            ax.set_xlabel("Weight Value")
            ax.set_ylabel("Count")
            layer_idx += 1
            if layer_idx >= n_layers:
                break

    plt.suptitle("DNN权重分布", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "weight_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 权重分布已保存: {save_path}")
    plt.close()


# ============================================================
# Step 7: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("DNN 图像分类 - MNIST（纯全连接网络）")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # 加载数据
    print("\n加载数据集...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # 创建模型
    model = DNNClassifier(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型: DNNClassifier")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
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
    model_path = os.path.join(cfg.save_dir, "dnn_classifier.pth")
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
    plot_weight_distribution(model, cfg)

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
