"""
=============================================================================
DNN 非线性回归任务模板 (Deep Neural Network for Nonlinear Regression)
=============================================================================

【原理】
根据万能逼近定理(Universal Approximation Theorem)：
"一个具有足够多隐藏层神经元的单隐藏层前馈神经网络，
可以以任意精度逼近任意连续函数。"

DNN通过多层非线性变换，学习从输入到输出的复杂映射关系。
每一层提取不同抽象级别的特征，深层网络能表示更复杂的函数。

【为什么DNN能拟合任意函数？】
想象用很多"小折线"拼接成一条复杂曲线：
- ReLU激活函数本身就是一条折线：f(x)=max(0, x)
- 一个神经元 = 一条折线
- 多个神经元叠加 = 多条折线组合 = 任意形状曲线
- 隐藏层越多、神经元越多，能表示的曲线越复杂

【应用场景】
- 物理建模：拟合实验数据中的未知关系
- 金融预测：预测股价、汇率等连续变量
- 工程优化：预测材料性能、设备寿命
- 气象预测：温度、降雨量预测
- 药物发现：预测分子活性（定量构效关系QSAR）

【本数据集: 合成非线性函数】
y = x³ · sin(5πx) + ε (高斯噪声)
这是一个高度非线性的函数，线性模型完全无法拟合。
DNN通过多个隐藏层的ReLU激活，可以分段逼近这种复杂曲线。

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python dnn/regression.py
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

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    # n_samples=1000: 合成数据样本数
    #   为什么1000？足够展示DNN的拟合能力，又不会训练太久
    n_samples = 1000

    # noise_std=0.15: 高斯噪声标准差
    #   为什么0.15？噪声太小（0.05）则拟合太简单，太大（0.5）则无法学习
    #   0.15让曲线有可见噪声，但DNN仍能捕捉主要趋势
    noise_std = 0.15

    # test_size=0.2: 测试集比例
    test_size = 0.2

    # random_state=42: 固定随机种子
    random_state = 42

    # --- 模型相关 ---
    # hidden_dims=[128, 64, 32]: 3层隐藏层
    #   为什么是[128, 64, 32]？
    #     输入1维（x坐标），输出1维（y值），中间需要足够神经元来拟合复杂曲线
    #     128个神经元 = 128条"折线"，足以组合出sin(5πx)的5个周期波动
    #   为什么3层？
    #     万能逼近定理说"单隐藏层足够"，但实际中多层网络：
    #     - 用更少参数达到相同表达能力
    #     - 训练更稳定（梯度流更好）
    #     - 能学习层次化特征（低层学简单模式，高层学复杂组合）
    hidden_dims = [128, 64, 32]

    # dropout_rate=0.1: Dropout比例
    #   为什么只有0.1？回归任务通常比分类更不容易过拟合
    #   原因：回归的输出是连续值，"记住"具体数值比记住类别标签更难
    #   而且本数据有噪声，模型被迫学习趋势而非记忆点
    dropout_rate = 0.1

    # use_batch_norm=True: 使用批归一化
    use_batch_norm = True

    # --- 训练相关 ---
    # batch_size=32: 批次大小
    #   为什么32？1000条数据，batch=32每轮约31次更新，更新频率适中
    batch_size = 32

    # learning_rate=1e-3: Adam初始学习率
    learning_rate = 1e-3

    # epochs=500: 最大训练轮数
    #   为什么500？非线性函数拟合需要较多轮数，因为：
    #   - 损失曲面可能有多个局部最小值
    #   - 早停会自动控制，500只是上限
    epochs = 500

    # weight_decay=1e-5: L2正则化
    #   为什么比分类(1e-4)小？回归任务过拟合风险更低
    weight_decay = 1e-5

    # --- 早停策略 ---
    # early_stop_patience=30: 验证损失连续30轮不下降就停止
    #   为什么30？回归收敛较慢，需要更多耐心
    early_stop_patience = 30

    # --- 学习率调度器 ---
    # scheduler_type="reduce": ReduceLROnPlateau
    #   为什么用ReduceLROnPlateau？回归任务损失波动大，余弦退火可能在不合适的时候降LR
    #   ReduceLROnPlateau根据实际损失表现调整，更自适应
    scheduler_type = "reduce"

    # lr_factor=0.5: LR衰减因子
    lr_factor = 0.5

    # lr_patience=15: 连续15轮无改善才降LR
    lr_patience = 15

    # lr_min=1e-6: LR下限
    lr_min = 1e-6

    # --- 梯度裁剪 ---
    max_grad_norm = 1.0

    # --- 混合精度训练 ---
    use_amp = True

    # --- 保存相关 ---
    save_dir = "dnn/output/regression"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 数据生成
# ============================================================
def generate_data(cfg):
    """
    生成合成非线性回归数据。

    【目标函数】
    y = x³ · sin(5πx) + ε

    【为什么选择这个函数？】
    1. 非线性：x³和sin的组合，线性回归完全无法拟合
    2. 多尺度：sin(5πx)产生5个周期波动，x³提供整体趋势
    3. 复杂局部结构：波动幅度随x³增大，不同区域难度不同
    4. 有明确解析式：可以精确评估拟合质量

    【数据特点】
    - x ∈ [-1, 1]，均匀分布
    - 训练集800点，测试集200点
    - 加入高斯噪声模拟真实测量误差
    """
    np.random.seed(cfg.random_state)

    # 生成输入
    x = np.linspace(-1, 1, cfg.n_samples).astype(np.float32)
    # 加入少量随机扰动，避免完全均匀（更真实）
    x += np.random.normal(0, 0.02, size=x.shape).astype(np.float32)
    x = np.clip(x, -1, 1)

    # 生成目标：y = x³ · sin(5πx)
    y_true = (x ** 3) * np.sin(5 * np.pi * x)

    # 加入高斯噪声
    noise = np.random.normal(0, cfg.noise_std, size=y_true.shape).astype(np.float32)
    y = y_true + noise

    # 划分为训练集和测试集
    # 注意：回归任务通常不需要stratify（因为目标不是离散类别）
    n_test = int(cfg.n_samples * cfg.test_size)
    n_train = cfg.n_samples - n_test

    # 随机打乱后划分
    indices = np.random.permutation(cfg.n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    y_true_test = y_true[test_idx]  # 无噪声的真实值，用于评估

    # 转为PyTorch张量，增加特征维度（输入维度=1）
    X_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    Y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    Y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    print(f"数据集: {cfg.n_samples}点 | 训练: {n_train} | 测试: {n_test}")
    print(f"目标函数: y = x³ · sin(5πx) + N(0, {cfg.noise_std}²)")
    print(f"y范围: [{y.min():.3f}, {y.max():.3f}]")

    return X_train, Y_train, X_test, Y_test, y_true_test, x_test


def get_dataloaders(X_train, Y_train, X_test, Y_test, cfg):
    """创建DataLoader"""
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, test_loader


# ============================================================
# Step 4: 模型定义
# ============================================================
class DNNRegressor(nn.Module):
    """
    DNN回归模型

    【架构设计】
    输入 (1) → Linear(1→128) → [BN] → ReLU → Dropout
            → Linear(128→64) → [BN] → ReLU → Dropout
            → Linear(64→32) → [BN] → ReLU
            → Linear(32→1)

    【为什么输入/输出都是1维？】
    这是单变量回归：输入一个x，预测一个y。
    如果是多变量回归（如房价预测有10个特征），输入维度=10。

    【输出层为什么不加激活函数？】
    回归任务的输出可以是任意实数（正数、负数、小数）。
    如果加ReLU，负数输出被截断为0，会丢失信息。
    如果加Sigmoid，输出被压缩到[0,1]，范围受限。
    所以回归的输出层不加任何激活函数，直接输出线性变换结果。

    【万能逼近定理的实践】
    这个网络有128+64+32=224个隐藏神经元，
    相当于224条ReLU折线，可以组合出相当复杂的曲线形状。
    """

    def __init__(self, cfg):
        super().__init__()
        dims = [1] + cfg.hidden_dims + [1]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            # 最后一层不加BN、ReLU、Dropout
            if i < len(dims) - 2:
                if cfg.use_batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU(inplace=True))
                if cfg.dropout_rate > 0:
                    layers.append(nn.Dropout(cfg.dropout_rate))

        self.network = nn.Sequential(*layers)
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

        输入: (batch, 1) — x坐标
        输出: (batch, 1) — 预测的y值
        """
        return self.network(x)


# ============================================================
# Step 5: 训练函数
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, cfg, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_samples = 0

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
        total_samples += inputs.size(0)

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, cfg):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for inputs, targets in loader:
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        all_preds.extend(outputs.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / total_samples
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    return avg_loss, mse, mae, r2, all_preds, all_targets


def train(model, train_loader, test_loader, cfg):
    """完整训练流程"""
    # MSELoss: 均方误差，回归任务的标准损失函数
    # 公式: MSE = (1/N) Σ(y_pred - y_true)²
    # 特点：对大误差惩罚更重（平方项），让模型特别关注离群点
    criterion = nn.MSELoss()

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

    history = {"train_loss": [], "val_loss": [], "val_mse": [], "val_mae": [], "val_r2": []}

    use_amp = cfg.use_amp and cfg.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg, scaler)
        val_loss, val_mse, val_mae, val_r2, _, _ = evaluate(model, test_loader, criterion, cfg)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)
        history["val_r2"].append(val_r2)

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg.epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val MSE: {val_mse:.6f} MAE: {val_mae:.4f} R²: {val_r2:.4f} | "
                  f"LR: {current_lr:.2e}")

        # 学习率调度
        if cfg.scheduler_type == "reduce":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # 早停检查
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
        print(f"\n✓ 已恢复最优模型 (Val MSE: {best_val_loss:.6f})")

    return model, history


# ============================================================
# Step 6: 可视化函数
# ============================================================
def plot_training_curves(history, cfg):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # 损失曲线
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("训练/验证损失曲线")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE曲线
    axes[1].plot(epochs, history["val_mae"], "g-", label="Val MAE", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("验证MAE曲线")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # R²曲线
    axes[2].plot(epochs, history["val_r2"], "purple", label="Val R²", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("R² Score")
    axes[2].set_title("验证R²曲线")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()


def plot_fitting_result(model, X_test, Y_test, y_true_test, x_test, cfg):
    """
    可视化拟合结果。

    【如何解读这张图？】
    - 黑色虚线：真实函数（无噪声）
    - 蓝色点：测试数据（含噪声）
    - 红色线：DNN拟合的曲线
    - 绿色区域：拟合曲线与真实函数的差距

    如果红色线紧密跟随黑色虚线，说明DNN成功学会了函数关系。
    """
    model.eval()

    # 生成平滑的曲线用于绘制拟合结果
    x_smooth = np.linspace(-1, 1, 500).astype(np.float32)
    X_smooth = torch.tensor(x_smooth, dtype=torch.float32).unsqueeze(1).to(cfg.device)

    with torch.no_grad():
        y_pred_smooth = model(X_smooth).cpu().numpy().flatten()

    # 测试集预测
    with torch.no_grad():
        y_pred_test = model(X_test.to(cfg.device)).cpu().numpy().flatten()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：拟合曲线 vs 真实函数
    ax = axes[0]
    ax.plot(x_smooth, (x_smooth ** 3) * np.sin(5 * np.pi * x_smooth),
            "k--", linewidth=2, label="真实函数", alpha=0.7)
    ax.scatter(x_test, Y_test.numpy().flatten(), c="blue", s=20, alpha=0.5, label="测试数据(含噪声)")
    ax.plot(x_smooth, y_pred_smooth, "r-", linewidth=2, label="DNN拟合")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("DNN非线性函数拟合")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)

    # 右图：残差分析
    ax = axes[1]
    residuals = Y_test.numpy().flatten() - y_pred_test
    ax.scatter(y_pred_test, residuals, c="steelblue", s=20, alpha=0.6)
    ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("预测值")
    ax.set_ylabel("残差 (真实 - 预测)")
    ax.set_title("残差图")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "fitting_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 拟合结果已保存: {save_path}")
    plt.close()


def plot_neuron_activations(model, cfg):
    """
    可视化第一层神经元的激活函数。

    【原理】
    每个隐藏神经元学习一个ReLU激活的线性函数：f(x) = max(0, wx + b)
    这是"折线"的基本单元。多个折线叠加就能逼近任意曲线。
    """
    model.eval()

    # 获取第一层的权重和偏置
    first_linear = None
    for m in model.network.modules():
        if isinstance(m, nn.Linear):
            first_linear = m
            break

    if first_linear is None:
        return

    weights = first_linear.weight.detach().cpu().numpy()
    biases = first_linear.bias.detach().cpu().numpy()

    x = np.linspace(-1, 1, 500)
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制前16个神经元的激活
    n_show = min(16, weights.shape[0])
    for i in range(n_show):
        w, b = weights[i, 0], biases[i]
        y = np.maximum(0, w * x + b)
        ax.plot(x, y, alpha=0.6, linewidth=1, label=f"Neuron {i+1}" if i < 6 else "")

    ax.set_xlabel("x")
    ax.set_ylabel("Activation")
    ax.set_title(f'第一层前{n_show}个神经元的ReLU激活（每条线 = 一条"折线"）')
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "neuron_activations.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 神经元激活已保存: {save_path}")
    plt.close()


# ============================================================
# Step 7: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("DNN 非线性回归 - 函数逼近")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # 生成数据
    print("\n生成合成数据...")
    X_train, Y_train, X_test, Y_test, y_true_test, x_test = generate_data(cfg)

    # 创建DataLoader
    train_loader, test_loader = get_dataloaders(X_train, Y_train, X_test, Y_test, cfg)

    # 创建模型
    model = DNNRegressor(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: DNNRegressor")
    print(f"总参数量: {total_params:,}")
    print(f"模型结构:\n{model}")

    # 训练
    model, history = train(model, train_loader, test_loader, cfg)

    # 最终评估
    print(f"\n{'='*60}")
    print("最终测试评估...")
    criterion = nn.MSELoss()
    _, test_mse, test_mae, test_r2, _, _ = evaluate(model, test_loader, criterion, cfg)
    print(f"测试集 MSE: {test_mse:.6f}")
    print(f"测试集 MAE: {test_mae:.4f}")
    print(f"测试集 R²:  {test_r2:.4f}")
    print(f"{'='*60}")

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "dnn_regressor.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
        "history": history,
    }, model_path)
    print(f"✓ 模型已保存: {model_path}")

    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, cfg)
    plot_fitting_result(model, X_test, Y_test, y_true_test, x_test, cfg)
    plot_neuron_activations(model, cfg)

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
