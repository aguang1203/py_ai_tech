"""
=============================================================================
DNN 自编码器任务模板 (Autoencoder for Dimensionality Reduction)
=============================================================================

【原理】
自编码器(Autoencoder, AE)是一种无监督神经网络，学习将输入压缩到低维表示(编码)，
然后再从低维表示重建原始输入(解码)。

结构：输入 → 编码器(降维) → 潜在空间(latent space) → 解码器(重建) → 输出

核心思想：强制网络学习数据的最重要特征，因为要从低维表示重建高维数据，
网络必须抓住数据的本质结构，丢弃噪声和冗余信息。

【编码器 vs 解码器】
┌─────────────┬─────────────────────────┬─────────────────────────┐
│    特性     │        编码器           │        解码器           │
├─────────────┼─────────────────────────┼─────────────────────────┤
│ 功能        │ 压缩数据到潜在空间       │ 从潜在空间重建数据       │
│ 维度变化    │ 高维 → 低维             │ 低维 → 高维             │
│ 类比        │ 压缩软件(如ZIP)         │ 解压软件                 │
│ 损失来源    │ 无(不直接输出)          │ 重建误差                 │
└─────────────┴─────────────────────────┴─────────────────────────┘

【为什么叫"无监督"学习？】
- 监督学习：需要输入-输出对（如图像+标签），告诉模型"正确答案"
- 无监督学习：只需要输入数据（如图像），目标是发现数据本身的结构
- 自编码器：输入=输出（重建自己），不需要人工标注的标签

【潜在空间(Latent Space)】
潜在空间是数据的压缩表示，具有有趣的几何性质：
- 相似的数据在潜在空间中距离近
- 可以对潜在向量做算术运算："戴眼镜的男人 - 男人 + 女人 = 戴眼镜的女人"
- 可以在潜在空间中插值，生成过渡样本

【应用场景】
- 降维可视化：将高维数据降到2D/3D，用t-SNE/UMAP可视化
- 特征学习：无监督预训练，提取数据的有效表示
- 去噪：训练时加入噪声，学习恢复干净数据
- 异常检测：重建误差大的样本可能是异常
- 数据压缩：用潜在向量代替原始数据，节省存储

【本数据集: MNIST】
- 将784维(28×28)降到32维，再重建回784维
- 压缩率: 784→32 = 24.5倍压缩
- 可视化原始图像、重建图像、潜在空间分布

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python dnn/autoencoder.py
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

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    data_dir = "data"
    image_size = 28
    flatten_dim = 784

    # test_size=0.1667: 从训练集划出验证集
    test_size = 0.1667
    random_state = 42

    # --- 模型相关 ---
    # latent_dim=32: 潜在空间维度
    #   【为什么选32？】
    #   784维 → 32维，压缩率约24.5倍
    #   32维足够表示10个数字的主要特征（每个数字约3维）
    #   太小（如2维）：信息损失太多，重建模糊
    #   太大（如128）：压缩效果不明显，潜在空间没有"提炼"特征
    latent_dim = 32

    # encoder_dims=[512, 256]: 编码器隐藏层
    #   784 → 512 → 256 → 32，逐层压缩到潜在空间
    encoder_dims = [512, 256]

    # decoder_dims=[256, 512]: 解码器隐藏层
    #   32 → 256 → 512 → 784，逐层扩展回原始维度
    #   通常与编码器对称（镜像结构）
    decoder_dims = [256, 512]

    # use_batch_norm=True: 使用批归一化
    use_batch_norm = True

    # --- 训练相关 ---
    batch_size = 256
    learning_rate = 1e-3
    epochs = 10
    weight_decay = 1e-5

    # --- 早停策略 ---
    early_stop_patience = 2

    # --- 学习率调度器 ---
    scheduler_type = "cosine"

    # --- 梯度裁剪 ---
    max_grad_norm = 1.0

    # --- 混合精度训练 ---
    use_amp = False

    # --- 可视化相关 ---
    # n_visualize=500: t-SNE可视化的样本数
    #   t-SNE计算复杂度高，太多样本会很慢
    n_visualize = 500
    # --- 保存相关 ---
    save_dir = "dnn/output/autoencoder"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 数据加载
# ============================================================
class FlattenMNIST(datasets.MNIST):
    """自定义MNIST，自动展平图像"""
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        img = img.view(-1)
        return img, target


def get_dataloaders(cfg):
    """加载MNIST数据集"""
    normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = FlattenMNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
    test_dataset = FlattenMNIST(root=cfg.data_dir, train=False, download=True, transform=transform)

    n_total = len(train_dataset)
    n_val = int(n_total * cfg.test_size)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(cfg.random_state)
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [n_train, n_val], generator=generator,
    )

    pin_mem = cfg.device.type == "cuda"
    pw = cfg.num_workers if hasattr(cfg, "num_workers") and cfg.num_workers > 0 else False
    num_workers = getattr(cfg, "num_workers", 0)

    train_loader = DataLoader(
        train_subset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem,
        persistent_workers=pw if pw else False,
    )
    val_loader = DataLoader(
        val_subset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
        persistent_workers=pw if pw else False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
        persistent_workers=pw if pw else False,
    )

    print(f"训练集: {n_train}张 | 验证集: {n_val}张 | 测试集: {len(test_dataset)}张")
    return train_loader, val_loader, test_loader


# ============================================================
# Step 4: 模型定义
# ============================================================
class Autoencoder(nn.Module):
    """
    自编码器模型

    【架构设计】
    编码器：784 → 512 → 256 → 32
    解码器：32 → 256 → 512 → 784 → Sigmoid

    【为什么是"沙漏"形状？】
    编码器：逐层压缩，提取越来越抽象的特征
    解码器：逐层扩展，从抽象特征重建细节

    【输出层为什么不加激活函数？】
    MNIST像素值经过标准化后有正有负，不在[0,1]范围内。
    因此使用MSELoss直接处理连续值重建，输出层不需要Sigmoid。

    【编码器和解码器必须对称吗？】
    不一定，但对称是常见设计：
    - 对称：编码器和解码器能力平衡，重建质量稳定
    - 不对称：如果更关注编码（特征提取），可以加深编码器
    """

    def __init__(self, cfg):
        super().__init__()
        self.latent_dim = cfg.latent_dim

        # ---- 编码器 ----
        encoder_dims = [cfg.flatten_dim] + cfg.encoder_dims + [cfg.latent_dim]
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
            if i < len(encoder_dims) - 2:  # 最后一层不加BN和ReLU
                if cfg.use_batch_norm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dims[i+1]))
                encoder_layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*encoder_layers)

        # ---- 解码器 ----
        decoder_dims = [cfg.latent_dim] + cfg.decoder_dims + [cfg.flatten_dim]
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
            if i < len(decoder_dims) - 2:
                if cfg.use_batch_norm:
                    decoder_layers.append(nn.BatchNorm1d(decoder_dims[i+1]))
                decoder_layers.append(nn.ReLU(inplace=True))
            else:
                # 输出层不加激活，MSELoss直接处理连续值
                pass
        self.decoder = nn.Sequential(*decoder_layers)

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
        前向传播：完整自编码器（编码+解码）

        输入: (batch, 784)
        输出: (batch, 784) — 重建的图像
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def encode(self, x):
        """只编码，返回潜在向量"""
        return self.encoder(x)

    def decode(self, z):
        """只解码，从潜在向量重建"""
        return self.decoder(z)


# ============================================================
# Step 5: 训练函数
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, cfg, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_samples = 0

    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for inputs, _ in loader:
        inputs = inputs.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(inputs)
        loss = criterion(outputs, inputs)  # 自编码器：输入=目标

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
    all_latent = []
    all_labels = []
    all_original = []
    all_reconstructed = []
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for inputs, labels in loader:
        inputs = inputs.to(cfg.device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(inputs)
        loss = criterion(outputs, inputs)

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        # 收集潜在向量用于可视化
        latent = model.encode(inputs)
        all_latent.append(latent.cpu())
        all_labels.append(labels)
        all_original.append(inputs.cpu())
        all_reconstructed.append(outputs.cpu())

    avg_loss = total_loss / total_samples
    all_latent = torch.cat(all_latent, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_original = torch.cat(all_original, dim=0).numpy()
    all_reconstructed = torch.cat(all_reconstructed, dim=0).numpy()

    # 计算MSE重建误差
    mse = np.mean((all_original - all_reconstructed) ** 2)

    return avg_loss, mse, all_latent, all_labels, all_original, all_reconstructed


def train(model, train_loader, val_loader, cfg):
    """完整训练流程"""
    # MSELoss: 均方误差损失
    #   自编码器重建的标准损失函数
    criterion = nn.MSELoss()

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

    history = {"train_loss": [], "val_loss": [], "val_mse": []}

    use_amp = cfg.use_amp and cfg.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")
    print(f"潜在空间维度: {cfg.latent_dim} (压缩率: {cfg.flatten_dim/cfg.latent_dim:.1f}x)")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg, scaler)
        val_loss, val_mse, _, _, _, _ = evaluate(model, val_loader, criterion, cfg)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mse"].append(val_mse)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg.epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val MSE: {val_mse:.6f}")

        scheduler.step()

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
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("训练/验证损失曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_mse"], "g-", label="Val MSE", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    ax2.set_title("验证重建误差(MSE)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()


def plot_reconstruction(original, reconstructed, cfg, num_samples=10):
    """
    可视化原始图像与重建图像的对比。

    【如何评估重建质量？】
    - 数字轮廓清晰 → 重建成功
    - 数字模糊/缺失 → 潜在空间维度太小，信息丢失
    - 有奇怪斑点 → 训练不充分或网络容量不足
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))

    for i in range(num_samples):
        # 原始图像
        img_orig = original[i].reshape(28, 28)
        axes[0, i].imshow(img_orig, cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("原始", fontsize=10)

        # 重建图像
        img_recon = reconstructed[i].reshape(28, 28)
        axes[1, i].imshow(img_recon, cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("重建", fontsize=10)

        # 计算单张重建误差
        mse = np.mean((img_orig - img_recon) ** 2)
        axes[1, i].text(14, 30, f"MSE:{mse:.4f}",
                        ha="center", fontsize=7, color="red")

    plt.suptitle(f"原始图像 vs 重建图像 (潜在维度={cfg.latent_dim})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "reconstruction.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 重建对比已保存: {save_path}")
    plt.close()


def plot_latent_space(latent, labels, cfg):
    """
    可视化潜在空间。

    【为什么潜在空间可视化很重要？】
    它告诉我们：网络是否学会了"有意义"的表示。
    - 同类数字聚在一起 → 网络学会了区分不同数字
    - 所有数字混在一起 → 网络没有学到有效特征
    - 聚类边界清晰 → 潜在空间质量好

    本代码使用PCA降到2D进行可视化（t-SNE太慢，用PCA更快）。
    """
    # 如果潜在维度>2，先用PCA降到2D
    if latent.shape[1] > 2:
        print(f"潜在维度{latent.shape[1]}>2，使用PCA降到2D可视化...")
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent)
        explained = pca.explained_variance_ratio_
        print(f"  PCA解释方差: PC1={explained[0]:.2%}, PC2={explained[1]:.2%}, 合计={sum(explained):.2%}")
    else:
        latent_2d = latent

    # 只可视化部分样本（避免点太多）
    n_show = min(cfg.n_visualize, len(latent_2d))
    indices = np.random.choice(len(latent_2d), n_show, replace=False)
    latent_show = latent_2d[indices]
    labels_show = labels[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        latent_show[:, 0], latent_show[:, 1],
        c=labels_show, cmap="tab10", s=15, alpha=0.7,
    )
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("潜在空间可视化 (PCA降维)")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("数字类别")

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "latent_space.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 潜在空间已保存: {save_path}")
    plt.close()


def plot_latent_interpolation(model, cfg):
    """
    潜在空间插值可视化。

    【原理】
    在潜在空间中取两个点，沿直线插值，观察解码结果。
    如果潜在空间"平滑"，插值结果应该产生有意义的过渡（如3→8的渐变）。
    """
    model.eval()

    # 随机生成两个潜在向量
    z1 = torch.randn(1, cfg.latent_dim).to(cfg.device)
    z2 = torch.randn(1, cfg.latent_dim).to(cfg.device)

    # 插值
    n_steps = 10
    alphas = np.linspace(0, 1, n_steps)

    fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 2))
    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            z = (1 - alpha) * z1 + alpha * z2
            recon = model.decode(z).cpu().numpy().reshape(28, 28)
            axes[i].imshow(recon, cmap="gray", vmin=0, vmax=1)
            axes[i].axis("off")
            axes[i].set_title(f"α={alpha:.1f}", fontsize=8)

    plt.suptitle("潜在空间插值（从左到右连续变化）", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "latent_interpolation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 潜在空间插值已保存: {save_path}")
    plt.close()


# ============================================================
# Step 7: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("DNN 自编码器 - MNIST降维与重建")
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
    model = Autoencoder(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: Autoencoder")
    print(f"总参数量: {total_params:,}")
    print(f"编码器: {cfg.flatten_dim} → {' → '.join(map(str, cfg.encoder_dims))} → {cfg.latent_dim}")
    print(f"解码器: {cfg.latent_dim} → {' → '.join(map(str, cfg.decoder_dims))} → {cfg.flatten_dim}")

    # 训练
    model, history = train(model, train_loader, val_loader, cfg)

    # 测试集评估
    print(f"\n{'='*60}")
    print("测试集评估...")
    criterion = nn.MSELoss()
    test_loss, test_mse, latent, labels, original, reconstructed = evaluate(
        model, test_loader, criterion, cfg,
    )
    print(f"测试集 BCE Loss: {test_loss:.6f}")
    print(f"测试集 MSE: {test_mse:.6f}")
    print(f"{'='*60}")

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "autoencoder.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
        "history": history,
    }, model_path)
    print(f"✓ 模型已保存: {model_path}")

    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, cfg)
    plot_reconstruction(original, reconstructed, cfg)
    plot_latent_space(latent, labels, cfg)
    plot_latent_interpolation(model, cfg)

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
