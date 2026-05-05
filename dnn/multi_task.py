"""
=============================================================================
DNN 多任务学习模板 (Multi-Task Learning with Deep Neural Networks)
=============================================================================

【原理】
多任务学习(MTL)让一个神经网络同时学习多个相关任务，
底层网络学习通用特征表示，顶层分叉为多个任务头。

优势：
1. 数据效率：多个任务共享数据，互相提供额外监督信号
2. 泛化提升：通用表示更难过拟合到单一任务
3. 计算效率：一个模型同时输出多个预测，推理更快

【结构示意图】
       输入特征
          │
    ┌─────┴─────┐
    ▼           ▼
 共享层1      共享层2
    │           │
    └─────┬─────┘
          ▼
      共享表示
     ┌────┴────┐
     ▼         ▼
  分类头     回归头
     │         │
     ▼         ▼
   类别预测   数值预测

【为什么共享层有帮助？】
想象同时学习"识别动物"和"估计动物体重"：
- 两个任务都需要先识别"这是猫还是狗"（共享特征）
- 分类头：猫=0, 狗=1
- 回归头：猫≈4kg, 狗≈20kg
- 共享层学会了"猫/狗"的区分，两个任务都受益

【损失函数如何设计？】
总损失 = w₁ × L_classification + w₂ × L_regression
- w₁, w₂是任务权重，平衡不同任务的贡献
- 分类损失：CrossEntropyLoss
- 回归损失：MSELoss
- 关键：两个任务的损失量级可能不同，需要权重平衡

【应用场景】
- 自动驾驶：同时检测车道(分割)+识别交通标志(分类)+测距(回归)
- 医疗诊断：同时预测疾病类型(分类)+病情严重程度(回归)
- 推荐系统：同时预测点击率(分类)+停留时长(回归)
- 金融风控：同时预测是否违约(分类)+违约金额(回归)

【本数据集: 合成多任务数据】
- 输入：10维特征
- 任务1（分类）：3个类别，由特征的非线性组合决定
- 任务2（回归）：连续数值，由另一组特征组合决定
- 两个任务共享底层特征，但有独立的输出头

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python dnn/multi_task.py
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
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
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
    # n_samples=2000: 合成数据样本数
    n_samples = 2000

    # n_features=10: 输入特征维度
    #   为什么10维？足够复杂以产生非线性分类和回归任务
    n_features = 10

    # num_classes=3: 分类任务的类别数
    num_classes = 3

    # class_names: 类别名称
    class_names = ["A类", "B类", "C类"]

    # test_size=0.2: 测试集比例
    test_size = 0.2

    # random_state=42: 固定随机种子
    random_state = 42

    # --- 模型相关 ---
    # shared_dims=[128, 64]: 共享层维度
    #   为什么2层？足够学习通用特征表示，又不会太深导致梯度问题
    shared_dims = [128, 64]

    # task_head_dim=32: 任务专用头的隐藏层维度
    task_head_dim = 32

    # use_batch_norm=True: 使用批归一化
    use_batch_norm = True

    # dropout_rate=0.2: Dropout比例
    dropout_rate = 0.2

    # --- 多任务损失权重 ---
    # class_weight=0.5: 分类任务在总损失中的权重
    # reg_weight=0.5: 回归任务在总损失中的权重
    #   【为什么用0.5+0.5？】
    #   两个任务同等重要时，均分权重。
    #   如果某个任务更重要，可以增大其权重（如0.7 vs 0.3）
    #   注意：如果两个任务的损失量级差异很大，需要调整权重平衡
    class_weight = 0.5
    reg_weight = 0.5

    # --- 训练相关 ---
    batch_size = 64
    learning_rate = 1e-3
    epochs = 200
    weight_decay = 1e-4

    # --- 早停策略 ---
    early_stop_patience = 20

    # --- 学习率调度器 ---
    scheduler_type = "reduce"
    lr_factor = 0.5
    lr_patience = 10
    lr_min = 1e-6

    # --- 梯度裁剪 ---
    max_grad_norm = 1.0

    # --- 混合精度训练 ---
    use_amp = True

    # --- 保存相关 ---
    save_dir = "dnn/output/multi_task"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 数据生成
# ============================================================
def generate_data(cfg):
    """
    合成多任务数据集。

    【数据生成原理】
    输入X是10维随机特征。

    分类任务：
      y_class = argmax([X·w₁ + b₁, X·w₂ + b₂, X·w₃ + b₃]) + noise
      即：用3组不同的线性组合+非线性变换决定类别

    回归任务：
      y_reg = sin(X·w₄) + 0.5·(X[:,0]²) + noise
      即：特征的非线性组合决定连续值

    两个任务共享相同的输入特征X，但目标不同。
    这迫使共享层学习"对分类和回归都有用"的通用特征。
    """
    np.random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)

    # 生成输入特征
    X = np.random.randn(cfg.n_samples, cfg.n_features).astype(np.float32)

    # ---- 生成分类标签 ----
    # 3组权重，每组决定一个类别的"分数"
    W_class = np.random.randn(cfg.n_features, cfg.num_classes).astype(np.float32)
    b_class = np.random.randn(cfg.num_classes).astype(np.float32)
    scores = X @ W_class + b_class
    # 加入非线性：让每个类别的决策边界更复杂
    scores = np.tanh(scores) + 0.3 * np.sin(X[:, :1] * 3) @ np.ones((1, cfg.num_classes))
    y_class = np.argmax(scores, axis=1)

    # ---- 生成回归目标 ----
    # 与分类使用不同的权重组合，但有部分重叠特征
    w_reg = np.random.randn(cfg.n_features).astype(np.float32)
    y_reg = np.sin(X @ w_reg) + 0.5 * (X[:, 0] ** 2)
    y_reg = y_reg.astype(np.float32)
    # 加入噪声
    y_reg += np.random.normal(0, 0.3, size=y_reg.shape).astype(np.float32)

    # 划分训练集和测试集
    n_test = int(cfg.n_samples * cfg.test_size)
    n_train = cfg.n_samples - n_test

    indices = np.random.permutation(cfg.n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    yc_train, yc_test = y_class[train_idx], y_class[test_idx]
    yr_train, yr_test = y_reg[train_idx], y_reg[test_idx]

    # 转为PyTorch张量
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    yc_train_t = torch.tensor(yc_train, dtype=torch.long)
    yr_train_t = torch.tensor(yr_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    yc_test_t = torch.tensor(yc_test, dtype=torch.long)
    yr_test_t = torch.tensor(yr_test, dtype=torch.float32).unsqueeze(1)

    print(f"数据集: {cfg.n_samples}条 | 训练: {n_train} | 测试: {n_test}")
    print(f"输入维度: {cfg.n_features}")
    print(f"分类任务: {cfg.num_classes}类")
    print(f"  类别分布: A类={(y_class==0).sum()}, B类={(y_class==1).sum()}, C类={(y_class==2).sum()}")
    print(f"回归任务: y范围 [{y_reg.min():.2f}, {y_reg.max():.2f}]")

    return (
        X_train_t, yc_train_t, yr_train_t,
        X_test_t, yc_test_t, yr_test_t,
        X_test, yc_test, yr_test,
    )


def get_dataloaders(X_train, yc_train, yr_train, X_test, yc_test, yr_test, cfg):
    """创建DataLoader"""
    train_dataset = TensorDataset(X_train, yc_train, yr_train)
    test_dataset = TensorDataset(X_test, yc_test, yr_test)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, test_loader


# ============================================================
# Step 4: 模型定义
# ============================================================
class MultiTaskDNN(nn.Module):
    """
    多任务DNN模型

    【架构设计】
    输入 (10) → Shared(10→128) → [BN] → ReLU → Dropout
              → Shared(128→64) → [BN] → ReLU → Dropout
              → 共享表示 (64)
                 ├─→ ClassHead(64→32→ReLU→3) → 分类输出
                 └─→ RegHead(64→32→ReLU→1) → 回归输出

    【共享层的作用】
    共享层学习对所有任务有用的通用特征。
    例如：如果输入是房产数据，共享层可能学习：
      - 地段好坏（对房价预测和房产类型分类都有用）
      - 房屋大小（对两个任务都有用）

    【任务头的作用】
    每个任务头从通用表示中提取任务特定的信息。
    继续房产例子：
      - 分类头：从"地段+大小"判断"公寓/别墅/ townhouse"
      - 回归头：从"地段+大小"估算价格
    """

    def __init__(self, cfg):
        super().__init__()

        # ---- 共享层 ----
        shared_dims = [cfg.n_features] + cfg.shared_dims
        shared_layers = []
        for i in range(len(shared_dims) - 1):
            shared_layers.append(nn.Linear(shared_dims[i], shared_dims[i+1]))
            if cfg.use_batch_norm:
                shared_layers.append(nn.BatchNorm1d(shared_dims[i+1]))
            shared_layers.append(nn.ReLU(inplace=True))
            if cfg.dropout_rate > 0:
                shared_layers.append(nn.Dropout(cfg.dropout_rate))
        self.shared = nn.Sequential(*shared_layers)

        # ---- 分类头 ----
        self.class_head = nn.Sequential(
            nn.Linear(cfg.shared_dims[-1], cfg.task_head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.task_head_dim, cfg.num_classes),
        )

        # ---- 回归头 ----
        self.reg_head = nn.Sequential(
            nn.Linear(cfg.shared_dims[-1], cfg.task_head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.task_head_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """He初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        前向传播

        输入: (batch, 10)
        输出: (class_logits, reg_value)
          class_logits: (batch, 3) — 分类任务的logits
          reg_value: (batch, 1) — 回归任务的预测值
        """
        shared_features = self.shared(x)
        class_out = self.class_head(shared_features)
        reg_out = self.reg_head(shared_features)
        return class_out, reg_out

    def get_shared_features(self, x):
        """获取共享层的特征表示（用于分析）"""
        return self.shared(x)


# ============================================================
# Step 5: 训练函数
# ============================================================
def train_one_epoch(model, loader, optimizer, class_criterion, reg_criterion, cfg, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_class_loss = 0
    total_reg_loss = 0
    total_samples = 0

    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for inputs, y_class, y_reg in loader:
        inputs = inputs.to(cfg.device)
        y_class = y_class.to(cfg.device)
        y_reg = y_reg.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            class_out, reg_out = model(inputs)
            loss_class = class_criterion(class_out, y_class)
            loss_reg = reg_criterion(reg_out, y_reg)
            # 多任务损失 = 加权组合
            loss = cfg.class_weight * loss_class + cfg.reg_weight * loss_reg

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
        total_class_loss += loss_class.item() * inputs.size(0)
        total_reg_loss += loss_reg.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return (
        total_loss / total_samples,
        total_class_loss / total_samples,
        total_reg_loss / total_samples,
    )


@torch.no_grad()
def evaluate(model, loader, class_criterion, reg_criterion, cfg):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_class_loss = 0
    total_reg_loss = 0
    total_samples = 0

    all_class_preds = []
    all_class_targets = []
    all_reg_preds = []
    all_reg_targets = []

    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for inputs, y_class, y_reg in loader:
        inputs = inputs.to(cfg.device)
        y_class = y_class.to(cfg.device)
        y_reg = y_reg.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            class_out, reg_out = model(inputs)
            loss_class = class_criterion(class_out, y_class)
            loss_reg = reg_criterion(reg_out, y_reg)
            loss = cfg.class_weight * loss_class + cfg.reg_weight * loss_reg

        total_loss += loss.item() * inputs.size(0)
        total_class_loss += loss_class.item() * inputs.size(0)
        total_reg_loss += loss_reg.item() * inputs.size(0)
        total_samples += inputs.size(0)

        _, class_preds = class_out.max(1)
        all_class_preds.extend(class_preds.cpu().numpy())
        all_class_targets.extend(y_class.cpu().numpy())
        all_reg_preds.extend(reg_out.cpu().numpy())
        all_reg_targets.extend(y_reg.cpu().numpy())

    avg_loss = total_loss / total_samples
    avg_class_loss = total_class_loss / total_samples
    avg_reg_loss = total_reg_loss / total_samples

    class_acc = accuracy_score(all_class_targets, all_class_preds)
    reg_mse = mean_squared_error(all_reg_targets, all_reg_preds)
    reg_mae = mean_absolute_error(all_reg_targets, all_reg_preds)
    reg_r2 = r2_score(all_reg_targets, all_reg_preds)

    return (
        avg_loss, avg_class_loss, avg_reg_loss,
        class_acc, reg_mse, reg_mae, reg_r2,
        all_class_preds, all_class_targets,
        all_reg_preds, all_reg_targets,
    )


def train(model, train_loader, test_loader, cfg):
    """完整训练流程"""
    class_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    if cfg.scheduler_type == "reduce":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.lr_factor,
            patience=cfg.lr_patience, min_lr=cfg.lr_min,
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    history = {
        "train_loss": [], "val_loss": [],
        "train_class_loss": [], "train_reg_loss": [],
        "val_class_loss": [], "val_reg_loss": [],
        "val_class_acc": [], "val_reg_mse": [], "val_reg_r2": [],
    }

    use_amp = cfg.use_amp and cfg.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")
    print(f"损失权重: 分类={cfg.class_weight}, 回归={cfg.reg_weight}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, tcl, trl = train_one_epoch(
            model, train_loader, optimizer, class_criterion, reg_criterion, cfg, scaler,
        )
        (
            val_loss, vcl, vrl,
            val_acc, val_mse, val_mae, val_r2,
            _, _, _, _,
        ) = evaluate(model, test_loader, class_criterion, reg_criterion, cfg)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_class_loss"].append(tcl)
        history["train_reg_loss"].append(trl)
        history["val_class_loss"].append(vcl)
        history["val_reg_loss"].append(vrl)
        history["val_class_acc"].append(val_acc)
        history["val_reg_mse"].append(val_mse)
        history["val_reg_r2"].append(val_r2)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg.epochs} | "
                  f"Train: 总Loss={train_loss:.4f} 分类={tcl:.4f} 回归={trl:.4f} | "
                  f"Val: 分类Acc={val_acc:.4f} 回归MSE={val_mse:.4f} R²={val_r2:.4f}")

        if cfg.scheduler_type == "reduce":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"\n⚠ 早停触发 (Epoch {epoch})")
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history["train_loss"]) + 1)

    # 总损失
    axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Val", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("总损失")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 分类损失
    axes[0, 1].plot(epochs, history["train_class_loss"], "b-", label="Train", linewidth=2)
    axes[0, 1].plot(epochs, history["val_class_loss"], "r-", label="Val", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("CrossEntropy Loss")
    axes[0, 1].set_title("分类损失")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 回归损失
    axes[1, 0].plot(epochs, history["train_reg_loss"], "b-", label="Train", linewidth=2)
    axes[1, 0].plot(epochs, history["val_reg_loss"], "r-", label="Val", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MSE Loss")
    axes[1, 0].set_title("回归损失")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 分类准确率 + 回归R²
    ax_twin = axes[1, 1]
    ax_twin.plot(epochs, history["val_class_acc"], "g-", label="Class Acc", linewidth=2)
    ax_twin.set_xlabel("Epoch")
    ax_twin.set_ylabel("Classification Accuracy", color="green")
    ax_twin.tick_params(axis="y", labelcolor="green")

    ax2 = ax_twin.twinx()
    ax2.plot(epochs, history["val_reg_r2"], "purple", label="Reg R²", linewidth=2)
    ax2.set_ylabel("Regression R²", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")

    ax_twin.set_title("分类准确率 vs 回归R²")
    ax_twin.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()


def plot_classification_results(y_true, y_pred, cfg):
    """可视化分类结果"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=cfg.class_names, yticklabels=cfg.class_names,
           ylabel="真实类别", xlabel="预测类别",
           title="分类任务 - 混淆矩阵")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "classification_confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 分类混淆矩阵已保存: {save_path}")
    plt.close()


def plot_regression_results(y_true, y_pred, cfg):
    """可视化回归结果"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 预测 vs 真实
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="完美预测")
    axes[0].set_xlabel("真实值")
    axes[0].set_ylabel("预测值")
    axes[0].set_title("回归预测 vs 真实值")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 残差
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=20, color="steelblue")
    axes[1].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("预测值")
    axes[1].set_ylabel("残差 (真实 - 预测)")
    axes[1].set_title("回归残差图")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "regression_results.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 回归结果已保存: {save_path}")
    plt.close()


# ============================================================
# Step 7: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("DNN 多任务学习 - 分类 + 回归")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # 生成数据
    print("\n生成合成数据...")
    (
        X_train, yc_train, yr_train,
        X_test, yc_test, yr_test,
        X_test_np, yc_test_np, yr_test_np,
    ) = generate_data(cfg)

    # 创建DataLoader
    train_loader, test_loader = get_dataloaders(
        X_train, yc_train, yr_train, X_test, yc_test, yr_test, cfg,
    )

    # 创建模型
    model = MultiTaskDNN(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: MultiTaskDNN")
    print(f"总参数量: {total_params:,}")
    print(f"共享层: {cfg.n_features} → {' → '.join(map(str, cfg.shared_dims))}")
    print(f"分类头: {cfg.shared_dims[-1]} → {cfg.task_head_dim} → {cfg.num_classes}")
    print(f"回归头: {cfg.shared_dims[-1]} → {cfg.task_head_dim} → 1")

    # 训练
    model, history = train(model, train_loader, test_loader, cfg)

    # 最终评估
    print(f"\n{'='*60}")
    print("最终测试评估...")
    class_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    (
        _, _, _,
        class_acc, reg_mse, reg_mae, reg_r2,
        yc_pred, yc_true, yr_pred, yr_true,
    ) = evaluate(model, test_loader, class_criterion, reg_criterion, cfg)

    print(f"\n【分类任务】")
    print(f"  准确率: {class_acc:.4f}")
    print(classification_report(yc_true, yc_pred, target_names=cfg.class_names, digits=4))

    print(f"\n【回归任务】")
    print(f"  MSE: {reg_mse:.6f}")
    print(f"  MAE: {reg_mae:.4f}")
    print(f"  R²:  {reg_r2:.4f}")
    print(f"{'='*60}")

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "multi_task_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
        "history": history,
    }, model_path)
    print(f"✓ 模型已保存: {model_path}")

    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, cfg)
    plot_classification_results(yc_true, yc_pred, cfg)
    plot_regression_results(yr_true, yr_pred, cfg)

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
