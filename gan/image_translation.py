"""
=============================================================================
GAN 图像翻译任务模板 (Pix2Pix-style Image-to-Image Translation)
=============================================================================

【原理】
图像翻译是将一张图像转换为另一张图像的任务——输入和输出都是图像。
比如: 线稿→彩色图, 白天→夜晚, 语义图→真实图。

Pix2Pix是图像翻译的开山之作，核心思想:
  输入图像A → 生成器G → 输出图像B'
  判别器D判断: (A, B')是否像真实的(A, B)对

与普通GAN的区别:
  普通GAN: 随机噪声 → 生成图像
  Pix2Pix: 条件图像 → 生成对应图像

【Pix2Pix的关键创新】
1. U-Net生成器: 编码器-解码器+跳跃连接，保留输入的空间细节
2. PatchGAN判别器: 不判断整张图真假，而是判断每个小区域(N×N patch)真假
3. L1+对抗损失: L1保证全局一致性，对抗损失保证局部真实感

【U-Net生成器详解】
  编码器: 逐层下采样，提取高级语义特征
  解码器: 逐层上采样，恢复空间分辨率
  跳跃连接: 将编码器特征直接拼接到解码器，保留细节

  为什么需要跳跃连接？
  - 瓶颈层只有高级语义，丢失了空间细节(边缘/纹理)
  - 跳跃连接将低级特征直接传递给解码器
  - 相当于告诉解码器: "这里有边缘，请保留"

【PatchGAN判别器详解】
  普通判别器: 输入整张图 → 1个真/假判断
  PatchGAN: 输入整张图 → N×N个真/假判断(每个patch一个)

  为什么PatchGAN更好？
  1. 参数少: 只用小卷积，不需要全连接层
  2. 可处理任意尺寸图像
  3. 每个patch独立判断，关注局部纹理真实感
  4. 感受野通常70×70，即每个输出判断基于70×70区域

【损失函数】
  L_total = λ·L_L1 + L_GAN
  L_L1 = ||B - G(A)||₁  (L1重建损失，保持全局一致性)
  L_GAN = 对抗损失(让生成图像更真实)

  为什么用L1而非L2？
  - L2(MSE)生成模糊图像(取均值)
  - L1(MAE)生成更锐利的图像(取中值)
  - 经验: λ=100时效果好

【应用场景】
- 线稿→彩色图 (动漫上色)
- 语义分割图→真实图像 (城市景观生成)
- 白天→夜晚 (风格转换)
- 黑白→彩色 (老照片上色)
- 低分辨率→高分辨率 (超分辨率)

【本数据集: 合成边缘→填充图形】
- 输入: 几何图形的边缘轮廓(线稿)
- 输出: 填充后的几何图形(彩色)
- G学习: 从边缘线稿恢复原始图形

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python gan/image_translation.py
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
    image_size = 64          # Pix2Pix常用64或256
    in_channels = 1          # 输入通道(灰度)
    out_channels = 1         # 输出通道(灰度)
    num_samples = 1000       # 合成数据样本数
    test_size = 0.2
    random_state = 42

    # --- U-Net生成器相关 ---
    # unet_base=64: U-Net基础通道数
    #   编码器: 64→128→256→512
    #   解码器: 512→256→128→64
    unet_base = 64

    # --- PatchGAN判别器相关 ---
    # patch_size=70: PatchGAN的感受野
    #   为什么70？Pix2Pix论文推荐，对应4层卷积
    #   实际感受野取决于网络深度和卷积核大小
    disc_base = 64

    # --- 训练相关 ---
    batch_size = 16          # Pix2Pix通常用较小batch
    learning_rate = 2e-4
    beta1 = 0.5
    epochs = 100
    label_smoothing = 0.9    # 标签平滑，防D过强

    # lambda_l1=100: L1损失的权重
    #   为什么100？Pix2Pix论文推荐
    #   太小: 只有对抗损失，生成图像可能颜色不对
    #   太大: 只有L1损失，生成图像模糊
    lambda_l1 = 100

    # --- 保存相关 ---
    save_dir = "gan/output/image_translation"
    sample_interval = 10

    # --- 数据加载优化 ---
    num_workers = min(2, os.cpu_count() or 1)

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 合成数据和数据加载
# ============================================================
def generate_pair(shape_type, size=64):
    """
    生成一对(边缘图, 填充图)。

    边缘图: 只有轮廓线的图像(模拟线稿)
    填充图: 填充了颜色的图像(模拟上色结果)
    """
    # 填充图
    fill_img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(fill_img)

    margin = 8
    if shape_type == "circle":
        draw.ellipse([margin, margin, size - margin, size - margin], fill=200)
    elif shape_type == "rectangle":
        draw.rectangle([margin, margin, size - margin, size - margin], fill=180)
    elif shape_type == "triangle":
        draw.polygon([
            (size // 2, margin),
            (margin, size - margin),
            (size - margin, size - margin),
        ], fill=160)

    # 边缘图: 用边缘检测从填充图提取轮廓
    fill_arr = np.array(fill_img, dtype=np.float32) / 255.0
    # 简单的边缘检测: 与邻近像素的差
    edge_arr = np.zeros_like(fill_arr)
    edge_arr[1:, :] += np.abs(fill_arr[1:, :] - fill_arr[:-1, :])
    edge_arr[:, 1:] += np.abs(fill_arr[:, 1:] - fill_arr[:, :-1])
    edge_arr = np.clip(edge_arr * 5, 0, 1)  # 增强边缘

    return edge_arr, fill_arr


def generate_synthetic_data(cfg):
    """生成合成的边缘-填充图像对。"""
    np.random.seed(cfg.random_state)

    shapes = ["circle", "rectangle", "triangle"]
    pairs_A, pairs_B = [], []

    for _ in range(cfg.num_samples):
        shape = np.random.choice(shapes)
        edge, fill = generate_pair(shape, cfg.image_size)

        # 添加轻微噪声
        edge = edge + 0.02 * np.random.randn(*edge.shape)
        fill = fill / 255.0 + 0.02 * np.random.randn(*fill.shape) if fill.max() > 0 else fill
        edge = np.clip(edge, 0, 1)
        fill_arr = np.array(fill, dtype=np.float32) / 255.0 if isinstance(fill, np.ndarray) else fill_arr

        pairs_A.append(edge)
        # 重新生成以确保fill也是numpy
        fill_img = Image.new("L", (cfg.image_size, cfg.image_size), 0)
        draw = ImageDraw.Draw(fill_img)
        margin = 8
        if shape == "circle":
            draw.ellipse([margin, margin, cfg.image_size - margin, cfg.image_size - margin], fill=200)
        elif shape == "rectangle":
            draw.rectangle([margin, margin, cfg.image_size - margin, cfg.image_size - margin], fill=180)
        elif shape == "triangle":
            draw.polygon([
                (cfg.image_size // 2, margin),
                (margin, cfg.image_size - margin),
                (cfg.image_size - margin, cfg.image_size - margin),
            ], fill=160)
        fill_arr = np.array(fill_img, dtype=np.float32) / 255.0
        fill_arr = fill_arr + 0.02 * np.random.randn(*fill_arr.shape)
        fill_arr = np.clip(fill_arr, 0, 1)
        pairs_B.append(fill_arr)

    # 划分训练/测试集
    n_total = len(pairs_A)
    n_test = int(n_total * cfg.test_size)
    n_train = n_total - n_test

    return (pairs_A[:n_train], pairs_B[:n_train],
            pairs_A[n_train:], pairs_B[n_train:])


class ImagePairDataset(Dataset):
    """图像对数据集(A→B)。"""

    def __init__(self, images_a, images_b):
        self.images_a = images_a
        self.images_b = images_b

    def __len__(self):
        return len(self.images_a)

    def __getitem__(self, idx):
        a = torch.tensor(self.images_a[idx], dtype=torch.float32).unsqueeze(0)
        b = torch.tensor(self.images_b[idx], dtype=torch.float32).unsqueeze(0)
        # 归一化到[-1, 1]
        a = a * 2 - 1
        b = b * 2 - 1
        return a, b


def get_dataloaders(cfg):
    """生成合成数据并创建DataLoader。"""
    train_a, train_b, test_a, test_b = generate_synthetic_data(cfg)

    train_dataset = ImagePairDataset(train_a, train_b)
    test_dataset = ImagePairDataset(test_a, test_b)

    pin_mem = cfg.device.type == "cuda"
    pw = cfg.num_workers > 0

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=pw, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
    )

    print(f"训练集: {len(train_dataset)}对 | 测试集: {len(test_dataset)}对")
    return train_loader, test_loader


# ============================================================
# Step 4: 模型定义
# ============================================================
class UNetGenerator(nn.Module):
    """
    U-Net生成器 (Pix2Pix)

    【架构设计】
    编码器(下采样):
      input(1, 64, 64) → Conv(64) → pool → Conv(128) → pool → Conv(256) → pool → Conv(512)

    解码器(上采样):
      Conv(512) → up+cat(256) → Conv(256) → up+cat(128) → Conv(128) → up+cat(64) → Conv(64) → output

    跳跃连接: 编码器每层的输出直接拼接到解码器对应层
    """

    def __init__(self, cfg):
        super().__init__()
        nb = cfg.unet_base  # 64
        ic = cfg.in_channels
        oc = cfg.out_channels

        # ---- 编码器(下采样) ----
        self.enc1 = nn.Sequential(
            nn.Conv2d(ic, nb, 3, padding=1, bias=False),
            nn.BatchNorm2d(nb),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(nb, nb * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(nb * 2),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(nb * 2, nb * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(nb * 4),
            nn.ReLU(inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(nb * 4, nb * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(nb * 8),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool2d(2)

        # ---- 解码器(上采样) ----
        self.up3 = nn.ConvTranspose2d(nb * 8, nb * 4, 2, stride=2, bias=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(nb * 8, nb * 4, 3, padding=1, bias=False),  # 512+256=768→256, wait: up3 output 256 + skip 256 = 512
            nn.BatchNorm2d(nb * 4),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(nb * 4, nb * 2, 2, stride=2, bias=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(nb * 4, nb * 2, 3, padding=1, bias=False),  # 128+128=256→128
            nn.BatchNorm2d(nb * 2),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(nb * 2, nb, 2, stride=2, bias=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(nb * 2, nb, 3, padding=1, bias=False),  # 64+64=128→64
            nn.BatchNorm2d(nb),
            nn.ReLU(inplace=True),
        )

        # ---- 输出层 ----
        self.final = nn.Sequential(
            nn.Conv2d(nb, oc, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        前向传播(带跳跃连接)

        数据流动:
        x: (batch, 1, 64, 64)  ← 输入边缘图
          → enc1: 64×64, pool → 32×32
          → enc2: 32×32, pool → 16×16
          → enc3: 16×16, pool → 8×8
          → enc4: 8×8 (瓶颈)
          → up3+cat(enc3): 16×16
          → up2+cat(enc2): 32×32
          → up1+cat(enc1): 64×64
          → final: 1×64×64  ← 输出填充图
        """
        # 编码器
        e1 = self.enc1(x)       # (batch, 64, H, W)
        p1 = self.pool(e1)      # (batch, 64, H/2, W/2)

        e2 = self.enc2(p1)      # (batch, 128, H/2, W/2)
        p2 = self.pool(e2)      # (batch, 128, H/4, W/4)

        e3 = self.enc3(p2)      # (batch, 256, H/4, W/4)
        p3 = self.pool(e3)      # (batch, 256, H/8, W/8)

        e4 = self.enc4(p3)      # (batch, 512, H/8, W/8) [瓶颈]

        # 解码器(带跳跃连接)
        d3 = self.up3(e4)                                    # (batch, 256, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1)                     # (batch, 512, H/4, W/4)
        d3 = self.dec3(d3)                                   # (batch, 256, H/4, W/4)

        d2 = self.up2(d3)                                    # (batch, 128, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)                     # (batch, 256, H/2, W/2)
        d2 = self.dec2(d2)                                   # (batch, 128, H/2, W/2)

        d1 = self.up1(d2)                                    # (batch, 64, H, W)
        d1 = torch.cat([d1, e1], dim=1)                     # (batch, 128, H, W)
        d1 = self.dec1(d1)                                   # (batch, 64, H, W)

        out = self.final(d1)
        return out


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN判别器 (Pix2Pix)

    【PatchGAN原理】
    不判断整张图真假，而是输出N×N的判断矩阵，每个元素判断一个局部patch。

    为什么这样更好？
    1. 参数少: 不需要全连接层
    2. 任意尺寸: 输入多大图都行
    3. 关注细节: 每个patch独立判断，关注局部纹理
    4. 感受野: 每个输出元素基于约70×70的感受野

    输入: (条件图像A, 目标图像B) 拼接
    输出: (batch, 1, H', W') 每个元素=该patch的真假判断
    """

    def __init__(self, cfg):
        super().__init__()
        ndf = cfg.disc_base
        # 输入通道 = 条件图像 + 目标图像
        input_channels = cfg.in_channels + cfg.out_channels

        self.main = nn.Sequential(
            # 输入: (input_channels) x 64 x 64
            nn.Conv2d(input_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # → ndf x 32 x 32

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # → (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # → (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, 1, 4, 1, 1, bias=False),
            # → 1 x 7 x 7 (patch判断矩阵)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, input_img, target_img):
        """
        前向传播

        参数:
            input_img: 条件图像(边缘) (batch, 1, 64, 64)
            target_img: 目标图像(填充) (batch, 1, 64, 64)
        """
        # 拼接条件图像和目标图像
        x = torch.cat([input_img, target_img], dim=1)  # (batch, 2, 64, 64)
        return self.main(x)


# ============================================================
# Step 5: 训练函数
# ============================================================
def train(G, D, train_loader, cfg):
    """
    Pix2Pix训练循环。

    【Pix2Pix训练的独特之处】
    1. 生成器接收条件图像(边缘)而非噪声
    2. 判别器同时看条件图像和目标图像
    3. 损失 = L1重建损失 + 对抗损失

    【L1损失 vs 对抗损失的作用】
    - L1损失: 保证生成图像与真实图像全局一致(颜色/位置/大小)
    - 对抗损失: 让生成图像局部纹理更真实(不会模糊)
    - 只用L1: 清晰但缺少细节纹理
    - 只用对抗: 有纹理但全局结构可能错乱
    - 两者结合: 全局一致+局部真实 = 最佳效果
    """
    # 损失函数
    criterion_GAN = nn.BCEWithLogitsLoss()  # PatchGAN用logits更稳定
    criterion_L1 = nn.L1Loss()

    optimizer_G = optim.Adam(G.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))

    history = {"G_loss": [], "D_loss": [], "L1_loss": []}

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")

    for epoch in range(1, cfg.epochs + 1):
        G_losses, D_losses, L1_losses = [], [], []

        for input_a, target_b in train_loader:
            input_a = input_a.to(cfg.device)
            target_b = target_b.to(cfg.device)
            batch_size = input_a.size(0)

            # 真假标签(PatchGAN输出是矩阵)
            real_label = torch.full((batch_size, 1, 7, 7), cfg.label_smoothing, device=cfg.device)
            fake_label = torch.zeros(batch_size, 1, 7, 7, device=cfg.device)

            # =====================
            # 训练判别器D
            # =====================
            optimizer_D.zero_grad()

            # 真实对: (边缘, 真填充)
            d_real = D(input_a, target_b)
            d_loss_real = criterion_GAN(d_real, real_label)

            # 假对: (边缘, 生成填充)
            fake_b = G(input_a).detach()
            d_fake = D(input_a, fake_b)
            d_loss_fake = criterion_GAN(d_fake, fake_label)

            d_loss = (d_loss_real + d_loss_fake) * 0.5
            d_loss.backward()
            optimizer_D.step()

            # =====================
            # 训练生成器G
            # =====================
            optimizer_G.zero_grad()

            fake_b = G(input_a)
            d_fake_for_g = D(input_a, fake_b)

            # 对抗损失
            g_gan_loss = criterion_GAN(d_fake_for_g, real_label)
            # L1重建损失
            g_l1_loss = criterion_L1(fake_b, target_b)
            # 总损失
            g_loss = g_gan_loss + cfg.lambda_l1 * g_l1_loss

            g_loss.backward()
            optimizer_G.step()

            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            L1_losses.append(g_l1_loss.item())

        avg_g = np.mean(G_losses)
        avg_d = np.mean(D_losses)
        avg_l1 = np.mean(L1_losses)

        history["G_loss"].append(avg_g)
        history["D_loss"].append(avg_d)
        history["L1_loss"].append(avg_l1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg.epochs} | G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f} | L1: {avg_l1:.4f}")

        if epoch % cfg.sample_interval == 0 or epoch == 1:
            save_translation_samples(G, train_loader, cfg, epoch)

    return G, D, history


# ============================================================
# Step 6: 可视化函数
# ============================================================
def save_translation_samples(G, loader, cfg, epoch, num_samples=4):
    """保存翻译结果: 输入→生成→真实。"""
    G.eval()
    input_a, target_b = next(iter(loader))
    input_a = input_a[:num_samples].to(cfg.device)
    target_b = target_b[:num_samples]

    with torch.no_grad():
        fake_b = G(input_a).cpu()

    # 反归一化
    input_a = input_a.cpu() * 0.5 + 0.5
    fake_b = fake_b * 0.5 + 0.5
    target_b = target_b * 0.5 + 0.5

    fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 9))
    for i in range(num_samples):
        axes[0, i].imshow(input_a[i, 0].numpy(), cmap="gray")
        axes[0, i].set_title("输入(边缘)", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(fake_b[i, 0].numpy(), cmap="gray")
        axes[1, i].set_title("生成(填充)", fontsize=9)
        axes[1, i].axis("off")

        axes[2, i].imshow(target_b[i, 0].numpy(), cmap="gray")
        axes[2, i].set_title("真实(填充)", fontsize=9)
        axes[2, i].axis("off")

    plt.suptitle(f"Pix2Pix翻译结果 (Epoch {epoch})", fontsize=12)
    tag = f"epoch_{epoch:03d}" if epoch < cfg.epochs else "final"
    save_path = os.path.join(cfg.save_dir, f"translation_{tag}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(history, cfg):
    """绘制训练曲线。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["G_loss"]) + 1)

    axes[0].plot(epochs, history["G_loss"], "b-", label="G Loss", linewidth=1.5)
    axes[0].plot(epochs, history["D_loss"], "r-", label="D Loss", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("生成器/判别器损失")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["L1_loss"], "g-", label="L1 Loss", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("L1 Loss")
    axes[1].set_title("L1重建损失")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 组合损失
    axes[2].plot(epochs, history["G_loss"], "b-", label="G总损失", linewidth=1.5)
    axes[2].plot(epochs, history["L1_loss"], "g--", label="L1损失", linewidth=1.5, alpha=0.7)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("G损失 vs L1损失(展示L1主导)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

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
    print("GAN 图像翻译 - Pix2Pix (边缘→填充)")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    print("\n生成合成数据...")
    train_loader, test_loader = get_dataloaders(cfg)

    G = UNetGenerator(cfg).to(cfg.device)
    D = PatchGANDiscriminator(cfg).to(cfg.device)

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"\n生成器(U-Net)参数量: {g_params:,}")
    print(f"判别器(PatchGAN)参数量: {d_params:,}")

    G, D, history = train(G, D, train_loader, cfg)

    # 保存模型
    g_path = os.path.join(cfg.save_dir, "unet_generator.pth")
    d_path = os.path.join(cfg.save_dir, "patchgan_discriminator.pth")
    torch.save(G.state_dict(), g_path)
    torch.save(D.state_dict(), d_path)
    print(f"\n✓ 生成器已保存: {g_path}")
    print(f"✓ 判别器已保存: {d_path}")

    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, cfg)
    save_translation_samples(G, test_loader, cfg, cfg.epochs)

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
