"""
=============================================================================
GAN 异常检测任务模板 (GAN for Anomaly Detection)
=============================================================================

【原理】
GAN异常检测的核心思想：只用正常数据训练GAN，让生成器学会正常数据的分布。
推理时，异常数据无法被生成器很好地重建，重建误差大的就是异常。

核心流程：
  训练阶段: 正常数据 → 训练GAN → 生成器学会正常模式
  推理阶段: 输入图像 → 寻找最佳z → 重建图像 → 计算重建误差
            重建误差大 → 异常！重建误差小 → 正常

【AnoGAN方法】
  1. 训练阶段: 用正常数据训练DCGAN
  2. 推理阶段(对每张测试图像):
     a. 随机初始化一个噪声向量z
     b. 生成图像 G(z)
     c. 计算损失 = 图像损失 + 特征损失
     d. 反向传播更新z(不更新G的权重！)
     e. 重复b-d若干步，找到最佳z*
     f. 用z*计算异常分数 = 重建误差

  【图像损失 vs 特征损失】
  - 图像损失: ||x - G(z*)||  像素级差异
  - 特征损失: ||D_feature(x) - D_feature(G(z*))||  高级语义差异
  - 特征损失更鲁棒，因为同一数字的像素可能有偏移，但语义特征一致

【异常分数】
  anomaly_score = (1-λ) × 图像损失 + λ × 特征损失
  λ通常=0.1，特征损失权重小但很关键

【应用场景】
- 工业缺陷检测 (产品表面缺陷) ← 类似本模板
- 医学影像异常 (X光/CT异常区域)
- 网络入侵检测
- 信用卡欺诈检测
- 设备故障预警

【本数据集: 合成几何图形】
- 正常数据: 圆形(1000个)
- 异常数据: 三角形/星形(各200个)
- 图像大小: 28×28灰度图
- 训练只用圆形，测试时检测三角形和星形

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python gan/anomaly_detection.py
3. 合成数据自动生成，无需下载
=============================================================================
"""

# ============================================================
# Step 1: 导入必要的库
# ============================================================
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

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
    image_size = 28
    in_channels = 1

    # num_normal=1000: 正常样本(圆形)数量
    num_normal = 1000

    # num_anomaly_test=200: 测试集异常样本数量
    num_anomaly_test = 200

    # random_state=42: 随机种子
    random_state = 42

    # --- 模型相关 ---
    latent_dim = 100
    gen_features = 64
    disc_features = 64

    # --- GAN训练相关 ---
    batch_size = 64
    learning_rate = 2e-4
    beta1 = 0.5
    epochs = 80              # 异常检测GAN训练轮数(正常数据少，不需要太多)

    # --- 异常检测推理相关 ---
    # anomaly_steps=50: 推理时优化z的迭代步数
    #   为什么50？AnoGAN论文推荐，足够找到好的z
    #   太少: z优化不充分，重建差，正常数据也被判为异常
    #   太多: 推理慢，且可能过拟合到异常模式
    anomaly_steps = 50

    # anomaly_lr=0.01: 推理时z的学习率
    #   为什么0.01？z优化需要较大LR才能快速收敛
    #   太小: 50步内z走不到最优
    anomaly_lr = 0.01

    # lambda_feature=0.1: 特征损失权重
    #   为什么0.1？图像损失是主要的，特征损失辅助
    #   太大: 特征损失主导，可能忽略像素级细节
    #   太小: 特征损失不起作用
    lambda_feature = 0.1

    # --- 保存相关 ---
    save_dir = "gan/output/anomaly_detection"

    # --- 数据加载优化 ---
    num_workers = min(2, os.cpu_count() or 1)

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 合成数据和数据加载
# ============================================================
def generate_shape_image(shape_type, size=28):
    """生成一个几何图形的28×28灰度图像。"""
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)

    margin = 4
    if shape_type == "circle":
        draw.ellipse([margin, margin, size - margin, size - margin], fill=255)
    elif shape_type == "triangle":
        draw.polygon([
            (size // 2, margin),
            (margin, size - margin),
            (size - margin, size - margin),
        ], fill=255)
    elif shape_type == "star":
        # 简单的五角星
        cx, cy = size // 2, size // 2
        r = (size // 2) - margin
        points = []
        for i in range(10):
            angle = np.pi / 2 + i * np.pi / 5
            radius = r if i % 2 == 0 else r * 0.4
            x = cx + int(radius * np.cos(angle))
            y = cy - int(radius * np.sin(angle))
            points.append((x, y))
        draw.polygon(points, fill=255)

    return np.array(img, dtype=np.float32) / 255.0


def generate_synthetic_data(cfg):
    """
    生成合成的几何图形数据。

    【为什么用几何图形而非真实数据？】
    - 真实异常检测数据需要专业标注
    - 几何图形直观: 圆=正常，三角形/星形=异常
    - 容易理解"为什么异常检测有效"
    - 学会原理后，替换为真实数据只需修改数据加载函数
    """
    np.random.seed(cfg.random_state)

    normal_data = []
    for _ in range(cfg.num_normal):
        # 正常: 圆形(加轻微噪声和位置变化)
        offset_x = np.random.randint(-2, 3)
        offset_y = np.random.randint(-2, 3)
        img = generate_shape_image("circle", cfg.image_size)
        # 加噪声
        img = img + 0.05 * np.random.randn(*img.shape)
        img = np.clip(img, 0, 1)
        normal_data.append(img)

    # 测试集: 50%正常 + 50%异常
    n_normal_test = cfg.num_anomaly_test
    n_anomaly_test = cfg.num_anomaly_test

    test_data = []
    test_labels = []  # 0=正常, 1=异常

    for _ in range(n_normal_test):
        img = generate_shape_image("circle", cfg.image_size)
        img = img + 0.05 * np.random.randn(*img.shape)
        img = np.clip(img, 0, 1)
        test_data.append(img)
        test_labels.append(0)

    for _ in range(n_anomaly_test // 2):
        img = generate_shape_image("triangle", cfg.image_size)
        img = img + 0.05 * np.random.randn(*img.shape)
        img = np.clip(img, 0, 1)
        test_data.append(img)
        test_labels.append(1)

    for _ in range(n_anomaly_test - n_anomaly_test // 2):
        img = generate_shape_image("star", cfg.image_size)
        img = img + 0.05 * np.random.randn(*img.shape)
        img = np.clip(img, 0, 1)
        test_data.append(img)
        test_labels.append(1)

    # 打乱测试集
    indices = list(range(len(test_data)))
    np.random.shuffle(indices)
    test_data = [test_data[i] for i in indices]
    test_labels = [test_labels[i] for i in indices]

    return normal_data, test_data, test_labels


class ShapeDataset(Dataset):
    """几何图形数据集。"""

    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        # 转为tensor并归一化到[-1, 1]
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, 28, 28)
        img_tensor = img_tensor * 2 - 1  # [0,1] → [-1,1]
        return img_tensor


def get_dataloaders(cfg):
    """生成合成数据并创建DataLoader。"""
    normal_data, test_data, test_labels = generate_synthetic_data(cfg)

    # 训练集只用正常数据
    train_dataset = ShapeDataset(normal_data)
    test_dataset = ShapeDataset(test_data)

    pin_mem = cfg.device.type == "cuda"
    pw = cfg.num_workers > 0

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=pw, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
    )

    print(f"训练集(正常): {len(normal_data)}个 | 测试集: {len(test_data)}个 (正常{sum(1-l for l in test_labels)}, 异常{sum(test_labels)})")

    return train_loader, test_loader, test_labels


# ============================================================
# Step 4: 模型定义 (复用DCGAN架构)
# ============================================================
class Generator(nn.Module):
    """DCGAN生成器(与image_generation.py相同架构)。"""

    def __init__(self, cfg):
        super().__init__()
        ngf = cfg.gen_features
        nc = cfg.in_channels
        self.init_size = cfg.image_size // 4

        self.fc = nn.Linear(cfg.latent_dim, ngf * 4 * self.init_size * self.init_size)

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), -1, self.init_size, self.init_size)
        out = self.main(out)
        return out


class Discriminator(nn.Module):
    """
    DCGAN判别器，增加特征提取功能。

    【为什么判别器要提取特征？】
    异常检测需要比较图像的深层特征差异:
    - 低级特征(边缘/纹理): 不同圆形之间也有差异
    - 高级特征(语义): 正常=圆形，异常=三角形，语义差异明显
    - 判别器的中间层输出就是"高级特征"
    """
    def __init__(self, cfg):
        super().__init__()
        ndf = cfg.disc_features
        nc = cfg.in_channels

        self.features = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf * 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, img):
        feat = self.features(img)
        validity = self.classifier(feat)
        return validity

    def extract_features(self, img):
        """提取判别器的中间特征(用于特征损失)。"""
        return self.features(img)


# ============================================================
# Step 5: 训练函数 (只用正常数据)
# ============================================================
def train_gan(G, D, dataloader, cfg):
    """用正常数据训练GAN。"""
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))

    history = {"G_loss": [], "D_loss": []}

    print(f"\n{'='*60}")
    print("训练GAN (只用正常数据)...")
    print(f"{'='*60}")

    for epoch in range(1, cfg.epochs + 1):
        G_losses, D_losses = [], []

        for real_imgs in dataloader:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(cfg.device)

            real_target = torch.full((batch_size, 1), 0.9, device=cfg.device)
            fake_target = torch.zeros(batch_size, 1, device=cfg.device)

            # 训练D
            optimizer_D.zero_grad()
            d_real = D(real_imgs)
            d_loss_real = criterion(d_real, real_target)

            z = torch.randn(batch_size, cfg.latent_dim, device=cfg.device)
            fake_imgs = G(z).detach()
            d_fake = D(fake_imgs)
            d_loss_fake = criterion(d_fake, fake_target)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # 训练G
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, cfg.latent_dim, device=cfg.device)
            fake_imgs = G(z)
            d_fake_for_g = D(fake_imgs)
            g_loss = criterion(d_fake_for_g, real_target)
            g_loss.backward()
            optimizer_G.step()

            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

        avg_g = np.mean(G_losses)
        avg_d = np.mean(D_losses)
        history["G_loss"].append(avg_g)
        history["D_loss"].append(avg_d)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg.epochs} | G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f}")

    return G, D, history


# ============================================================
# Step 6: 异常检测
# ============================================================
def compute_anomaly_score(G, D, test_img, cfg):
    """
    计算单张图像的异常分数(AnoGAN方法)。

    【推理过程详解】
    给定测试图像x，找到最优z*使得G(z*)最接近x:

    1. 初始化z(随机)
    2. 生成G(z)
    3. 计算损失:
       - 图像损失: ||x - G(z)||₁  (L1距离，比L2更鲁棒)
       - 特征损失: ||D_feat(x) - D_feat(G(z))||₂  (特征空间距离)
       - 总损失 = (1-λ)×图像损失 + λ×特征损失
    4. 反向传播更新z (不更新G和D！)
    5. 重复2-4若干步
    6. 最终的损失值就是异常分数

    【为什么优化z而不是直接计算？】
    - GAN的潜在空间是高维的，同一个图像可能对应多个z
    - 需要找到"最接近"的z，才能得到有意义的重建误差
    - 直接随机z的重建误差没有参考价值
    """
    G.eval()
    D.eval()

    # 初始化z(可学习)
    z = torch.randn(1, cfg.latent_dim, device=cfg.device, requires_grad=True)
    optimizer_z = optim.Adam([z], lr=cfg.anomaly_lr)

    for step in range(cfg.anomaly_steps):
        optimizer_z.zero_grad()

        # 生成重建图像
        recon_img = G(z)

        # 图像损失(L1)
        img_loss = torch.abs(recon_img - test_img).mean()

        # 特征损失(L2)
        feat_real = D.extract_features(test_img)
        feat_recon = D.extract_features(recon_img)
        feat_loss = ((feat_real - feat_recon) ** 2).mean()

        # 总损失
        total_loss = (1 - cfg.lambda_feature) * img_loss + cfg.lambda_feature * feat_loss

        # 反向传播只更新z
        total_loss.backward()
        optimizer_z.step()

    # 最终异常分数
    with torch.no_grad():
        recon_img = G(z)
        img_loss = torch.abs(recon_img - test_img).mean().item()
        feat_real = D.extract_features(test_img)
        feat_recon = D.extract_features(recon_img)
        feat_loss = ((feat_real - feat_recon) ** 2).mean().item()
        anomaly_score = (1 - cfg.lambda_feature) * img_loss + cfg.lambda_feature * feat_loss

    return anomaly_score, recon_img.detach()


def detect_anomalies(G, D, test_loader, test_labels, cfg):
    """对所有测试样本进行异常检测。"""
    print(f"\n{'='*60}")
    print("异常检测...")
    print(f"{'='*60}")

    anomaly_scores = []
    reconstructions = []

    for i, test_img in enumerate(test_loader):
        test_img = test_img.to(cfg.device)
        score, recon = compute_anomaly_score(G, D, test_img, cfg)
        anomaly_scores.append(score)
        reconstructions.append(recon.cpu())

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i+1}/{len(test_loader)} 个样本")

    anomaly_scores = np.array(anomaly_scores)
    test_labels = np.array(test_labels)

    # 计算AUC-ROC
    auc = roc_auc_score(test_labels, anomaly_scores)
    # 计算平均精度
    ap = average_precision_score(test_labels, anomaly_scores)

    print(f"\n异常检测结果:")
    print(f"  AUC-ROC: {auc:.4f} (1.0=完美, 0.5=随机)")
    print(f"  平均精度(AP): {ap:.4f}")

    # 正常/异常的分数分布
    normal_scores = anomaly_scores[test_labels == 0]
    anomaly_scores_pos = anomaly_scores[test_labels == 1]
    print(f"  正常样本平均分数: {normal_scores.mean():.4f} ± {normal_scores.std():.4f}")
    print(f"  异常样本平均分数: {anomaly_scores_pos.mean():.4f} ± {anomaly_scores_pos.std():.4f}")

    return anomaly_scores, reconstructions


# ============================================================
# Step 7: 可视化函数
# ============================================================
def plot_anomaly_results(anomaly_scores, test_labels, reconstructions, cfg):
    """可视化异常检测结果。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 异常分数分布
    normal_scores = anomaly_scores[test_labels == 0]
    anomaly_scores_pos = anomaly_scores[test_labels == 1]

    axes[0].hist(normal_scores.tolist(), bins=30, alpha=0.6, label="正常(圆形)", color="green")
    axes[0].hist(anomaly_scores_pos.tolist(), bins=30, alpha=0.6, label="异常(三角/星形)", color="red")
    axes[0].set_xlabel("异常分数")
    axes[0].set_ylabel("数量")
    axes[0].set_title("异常分数分布")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 正常样本的重建分数
    axes[1].barh(range(min(5, len(normal_scores))), [normal_scores[i] for i in range(min(5, len(normal_scores)))], color="green", alpha=0.6)
    axes[1].set_xlabel("异常分数")
    axes[1].set_ylabel("正常样本")
    axes[1].set_title("正常样本的异常分数")
    axes[1].grid(True, alpha=0.3)

    # 3. 异常样本的重建分数
    axes[2].barh(range(min(5, len(anomaly_scores_pos))), [anomaly_scores_pos[i] for i in range(min(5, len(anomaly_scores_pos)))], color="red", alpha=0.6)
    axes[2].set_xlabel("异常分数")
    axes[2].set_ylabel("异常样本")
    axes[2].set_title("异常样本的异常分数")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "anomaly_detection_results.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 异常检测结果已保存: {save_path}")
    plt.close()


def plot_reconstruction_comparison(test_labels, reconstructions, test_loader, cfg, num_samples=8):
    """对比原图和重建图。"""
    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))

    normal_count = 0
    anomaly_count = 0

    for i, (test_img, label) in enumerate(zip(test_loader, test_labels)):
        if label == 0 and normal_count < num_samples // 2:
            idx = normal_count
            # 原图
            orig = test_img[0, 0].numpy() * 0.5 + 0.5
            axes[0, idx].imshow(orig, cmap="gray")
            axes[0, idx].set_title("正常原图", fontsize=8, color="green")
            axes[0, idx].axis("off")
            # 重建
            recon = reconstructions[i][0, 0].numpy() * 0.5 + 0.5
            axes[1, idx].imshow(recon, cmap="gray")
            axes[1, idx].set_title("重建", fontsize=8, color="green")
            axes[1, idx].axis("off")
            normal_count += 1
        elif label == 1 and anomaly_count < num_samples // 2:
            idx = num_samples // 2 + anomaly_count
            orig = test_img[0, 0].numpy() * 0.5 + 0.5
            axes[0, idx].imshow(orig, cmap="gray")
            axes[0, idx].set_title("异常原图", fontsize=8, color="red")
            axes[0, idx].axis("off")
            recon = reconstructions[i][0, 0].numpy() * 0.5 + 0.5
            axes[1, idx].imshow(recon, cmap="gray")
            axes[1, idx].set_title("重建", fontsize=8, color="red")
            axes[1, idx].axis("off")
            anomaly_count += 1

        if normal_count >= num_samples // 2 and anomaly_count >= num_samples // 2:
            break

    plt.suptitle("原图 vs 重建图 (正常: 重建好 | 异常: 重建差)", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "reconstruction_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 重建对比已保存: {save_path}")
    plt.close()


# ============================================================
# Step 8: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("GAN 异常检测 - 合成几何图形数据")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    print("\n生成合成数据...")
    train_loader, test_loader, test_labels = get_dataloaders(cfg)

    G = Generator(cfg).to(cfg.device)
    D = Discriminator(cfg).to(cfg.device)

    print(f"\n生成器参数量: {sum(p.numel() for p in G.parameters()):,}")
    print(f"判别器参数量: {sum(p.numel() for p in D.parameters()):,}")

    # 训练GAN(只用正常数据)
    G, D, history = train_gan(G, D, train_loader, cfg)

    # 保存模型
    g_path = os.path.join(cfg.save_dir, "anomaly_generator.pth")
    d_path = os.path.join(cfg.save_dir, "anomaly_discriminator.pth")
    torch.save(G.state_dict(), g_path)
    torch.save(D.state_dict(), d_path)
    print(f"\n✓ 生成器已保存: {g_path}")
    print(f"✓ 判别器已保存: {d_path}")

    # 异常检测
    anomaly_scores, reconstructions = detect_anomalies(G, D, test_loader, test_labels, cfg)

    # 可视化
    print("\n生成可视化...")
    plot_anomaly_results(anomaly_scores, test_labels, reconstructions, cfg)
    plot_reconstruction_comparison(test_labels, reconstructions, test_loader, cfg)

    print(f"\n{'='*60}")
    print("异常检测完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
