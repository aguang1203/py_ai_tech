"""
=============================================================================
GAN 条件生成任务模板 (Conditional GAN for Targeted Generation)
=============================================================================

【原理】
普通GAN只能随机生成图像，无法控制生成内容。条件GAN(cGAN)在GAN基础上
加入条件信息(如类别标签)，实现"想生成什么就生成什么"。

核心思想：
  普通GAN:     噪声z → 生成器G → 随机图像
  条件GAN:     噪声z + 条件c → 生成器G → 指定条件的图像

例: 想生成数字"7" → 噪声z + 标签"7" → 生成器 → 数字"7"的图像

【cGAN的改进】
1. 生成器: 接收(噪声z + 条件标签y)作为输入
2. 判别器: 接收(图像x + 条件标签y)作为输入
3. 判别器不仅要判断图像真假，还要判断图像是否匹配条件

  判别器的4种判断:
    真图像 + 正确标签 → 真 (如: 真实数字7 + 标签7)
    真图像 + 错误标签 → 假 (如: 真实数字7 + 标签3)
    假图像 + 任意标签 → 假

【条件信息如何注入？】
方法1 - 拼接(本模板使用): 将条件标签嵌入后与噪声/特征拼接
  G: [z | embedding(y)] → 生成器
  D: [x | embedding(y)] → 判别器

方法2 - 条件BN: 用条件标签调制BatchNorm的参数
  更高级，效果更好，但实现更复杂

方法3 - 自注意力: 用条件标签生成注意力权重
  如SAGAN, 最新的方法

【应用场景】
- 按类别生成图像 (生成特定数字/物体) ← 本模板使用
- 文本到图像生成 (输入描述→输出图像)
- 图像到图像翻译 (素描→照片)
- 图像编辑 (修改年龄/发色/表情)
- 语义分割图→真实图像

【本数据集: MNIST】
- 条件: 数字标签(0-9)
- 输入: 噪声z + 目标数字标签
- 输出: 对应数字的手写图像

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python gan/conditional_generation.py
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
    in_channels = 1
    num_classes = 10       # MNIST有10个数字(0-9)

    # --- 模型相关 ---
    latent_dim = 100       # 噪声维度
    gen_features = 64      # 生成器基础通道数
    disc_features = 64     # 判别器基础通道数

    # embedding_dim=50: 条件标签嵌入维度
    #   为什么50？10个类别，50维足够编码类别信息
    #   经验: embedding_dim ≈ num_classes × 5
    #   太小: 条件信息不够，生成器忽略条件
    #   太大: 参数浪费，训练慢
    embedding_dim = 50

    # --- 训练相关 ---
    batch_size = 128
    learning_rate = 2e-4
    beta1 = 0.5
    epochs = 100
    label_smoothing = 0.9
    d_steps_per_g = 1
    noise_std = 0.1          # 给D输入加噪声，防D过强

    # --- 保存相关 ---
    save_dir = "gan/output/conditional_generation"
    sample_interval = 5
    num_samples = 100      # 10×10网格: 每个数字10个样本

    # --- 数据加载优化 ---
    num_workers = min(4, os.cpu_count() or 1)

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 数据加载和预处理
# ============================================================
def get_dataloaders(cfg):
    """加载MNIST数据集。cGAN同时需要图像和标签。"""
    transform = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = datasets.MNIST(
        root=cfg.data_dir, train=True, download=True, transform=transform,
    )

    pin_mem = cfg.device.type == "cuda"
    pw = cfg.num_workers > 0

    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=pw, drop_last=True,
    )

    print(f"训练集: {len(dataset)}张 | 类别数: {cfg.num_classes}")
    return dataloader


# ============================================================
# Step 4: 模型定义
# ============================================================
class ConditionalGenerator(nn.Module):
    """
    条件GAN生成器

    【与普通GAN的区别】
    普通GAN:  z(batch, 100)                        → 生成器 → 图像
    条件GAN:  z(batch, 100) + embedding(y)(batch, 50) → 生成器 → 图像

    条件标签通过nn.Embedding转换为向量，与噪声拼接后输入生成器。
    这样生成器同时知道"要生成什么(标签)"和"生成的多样性(噪声)"。

    【维度变化详解】
    z: (batch, 100)
    embedding(标签): (batch, 50)
    拼接: (batch, 150)
      → FC: (batch, 512*7*7)
      → reshape: (batch, 512, 7, 7)
      → ConvT1: (batch, 256, 14, 14)
      → ConvT2: (batch, 128, 28, 28)
      → ConvT3: (batch, 1, 28, 28)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ngf = cfg.gen_features
        nc = cfg.in_channels
        self.init_size = cfg.image_size // 4

        # 标签嵌入层
        # num_embeddings=10: 10个数字
        # embedding_dim=50: 每个数字的嵌入向量维度
        self.label_embedding = nn.Embedding(cfg.num_classes, cfg.embedding_dim)

        # 输入维度 = 噪声维度 + 嵌入维度
        input_dim = cfg.latent_dim + cfg.embedding_dim  # 100 + 50 = 150

        self.fc = nn.Linear(input_dim, ngf * 4 * self.init_size * self.init_size)

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
        """权重初始化。"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, z, labels):
        """
        前向传播

        参数:
            z: 噪声向量 (batch, latent_dim)
            labels: 条件标签 (batch,), 整数0-9
        """
        # 嵌入标签
        label_emb = self.label_embedding(labels)  # (batch, embedding_dim)

        # 拼接噪声和标签嵌入
        gen_input = torch.cat([z, label_emb], dim=1)  # (batch, 150)

        out = self.fc(gen_input)
        out = out.view(out.size(0), -1, self.init_size, self.init_size)
        out = self.main(out)
        return out


class ConditionalDiscriminator(nn.Module):
    """
    条件GAN判别器

    【与普通GAN的区别】
    普通GAN:  图像 → 判别器 → 真/假
    条件GAN:  图像 + embedding(标签) → 判别器 → 真/假+是否匹配

    条件注入方式:
    将标签嵌入扩展为与图像相同的空间维度，作为额外通道拼接到图像上。
    这样判别器可以同时看到图像内容和对应该图像的标签。

    例: 输入1通道图像 → 拼接嵌入通道 → 变成(1+embedding_dim)通道
    """

    def __init__(self, cfg):
        super().__init__()
        ndf = cfg.disc_features
        nc = cfg.in_channels
        self.cfg = cfg

        # 标签嵌入
        self.label_embedding = nn.Embedding(cfg.num_classes, cfg.embedding_dim)

        # 将嵌入扩展为图像空间维度并作为额外通道
        # 总输入通道 = 图像通道 + 嵌入维度
        total_channels = nc + cfg.embedding_dim  # 1 + 50 = 51

        self.main = nn.Sequential(
            nn.Conv2d(total_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf * 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """权重初始化。"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, img, labels):
        """
        前向传播

        参数:
            img: 图像 (batch, 1, 28, 28)
            labels: 条件标签 (batch,)
        """
        # 嵌入标签
        label_emb = self.label_embedding(labels)  # (batch, embedding_dim)

        # 将嵌入扩展到图像的空间维度
        # (batch, embedding_dim) → (batch, embedding_dim, 1, 1) → (batch, embedding_dim, H, W)
        label_emb = label_emb.unsqueeze(-1).unsqueeze(-1)
        label_emb = label_emb.expand(-1, -1, img.size(2), img.size(3))

        # 拼接图像和标签嵌入
        d_input = torch.cat([img, label_emb], dim=1)  # (batch, 1+50, 28, 28)

        return self.main(d_input)


# ============================================================
# Step 5: 训练函数
# ============================================================
def train(G, D, dataloader, cfg):
    """
    条件GAN训练循环。

    【与普通GAN训练的区别】
    1. 生成器: 输入(噪声z + 标签y)
    2. 判别器: 输入(图像x + 标签y)
    3. 真实数据: (真实图像, 正确标签) → 标签=1
    4. 生成数据: (生成图像, 生成时用的标签) → 标签=0

    【cGAN训练的一个关键细节】
    - 生成假图像时，同时记录使用了什么标签
    - 判别器判断假图像时，也传入同样的标签
    - 这样判别器不仅要判真假，还要判图像是否匹配标签
    """
    criterion = nn.BCELoss()

    optimizer_G = optim.Adam(G.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))

    # 固定噪声+标签用于可视化
    fixed_noise = torch.randn(cfg.num_samples, cfg.latent_dim, device=cfg.device)
    # 生成0-9每个数字各10个
    fixed_labels = torch.arange(cfg.num_classes, device=cfg.device).repeat_interleave(cfg.num_samples // cfg.num_classes)

    history = {
        "G_loss": [], "D_loss": [],
        "D_real_acc": [], "D_fake_acc": [],
    }

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")
    print(f"设备: {cfg.device} | G参数: {sum(p.numel() for p in G.parameters()):,} | "
          f"D参数: {sum(p.numel() for p in D.parameters()):,}")

    for epoch in range(1, cfg.epochs + 1):
        G_losses, D_losses = [], []
        D_real_accs, D_fake_accs = [], []

        for real_imgs, real_labels in dataloader:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(cfg.device)
            real_labels = real_labels.to(cfg.device)

            # 可选: 给D输入加噪声
            if cfg.noise_std > 0:
                real_imgs = real_imgs + cfg.noise_std * torch.randn_like(real_imgs)

            real_target = torch.full((batch_size, 1), cfg.label_smoothing, device=cfg.device)
            fake_target = torch.zeros(batch_size, 1, device=cfg.device)

            # =====================
            # 训练判别器D
            # =====================
            for _ in range(cfg.d_steps_per_g):
                optimizer_D.zero_grad()

                # 真实图像+正确标签
                d_real = D(real_imgs, real_labels)
                d_loss_real = criterion(d_real, real_target)

                # 随机选标签，生成假图像
                fake_labels = torch.randint(0, cfg.num_classes, (batch_size,), device=cfg.device)
                z = torch.randn(batch_size, cfg.latent_dim, device=cfg.device)
                fake_imgs = G(z, fake_labels).detach()

                d_fake = D(fake_imgs, fake_labels)
                d_loss_fake = criterion(d_fake, fake_target)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_D.step()

            with torch.no_grad():
                d_real_acc = d_real.mean().item()
                d_fake_acc = 1.0 - d_fake.mean().item()

            # =====================
            # 训练生成器G
            # =====================
            optimizer_G.zero_grad()

            fake_labels = torch.randint(0, cfg.num_classes, (batch_size,), device=cfg.device)
            z = torch.randn(batch_size, cfg.latent_dim, device=cfg.device)
            fake_imgs = G(z, fake_labels)

            d_fake_for_g = D(fake_imgs, fake_labels)
            g_loss = criterion(d_fake_for_g, real_target)

            g_loss.backward()
            optimizer_G.step()

            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            D_real_accs.append(d_real_acc)
            D_fake_accs.append(d_fake_acc)

        avg_g_loss = np.mean(G_losses)
        avg_d_loss = np.mean(D_losses)
        avg_d_real = np.mean(D_real_accs)
        avg_d_fake = np.mean(D_fake_accs)

        history["G_loss"].append(avg_g_loss)
        history["D_loss"].append(avg_d_loss)
        history["D_real_acc"].append(avg_d_real)
        history["D_fake_acc"].append(avg_d_fake)

        print(f"Epoch {epoch:3d}/{cfg.epochs} | "
              f"G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f} | "
              f"D(真): {avg_d_real:.2f} | D(假): {avg_d_fake:.2f}")

        if epoch % cfg.sample_interval == 0 or epoch == 1:
            save_conditional_samples(G, fixed_noise, fixed_labels, cfg, epoch)

    save_conditional_samples(G, fixed_noise, fixed_labels, cfg, cfg.epochs, final=True)

    return G, D, history


# ============================================================
# Step 6: 可视化函数
# ============================================================
def save_conditional_samples(G, fixed_noise, fixed_labels, cfg, epoch, final=False):
    """
    保存条件生成的样本: 每行一个数字(0-9)，每列不同的随机性。
    """
    G.eval()
    with torch.no_grad():
        fake_imgs = G(fixed_noise, fixed_labels).cpu()

    fake_imgs = fake_imgs * 0.5 + 0.5
    fake_imgs = fake_imgs.clamp(0, 1)

    n_per_class = cfg.num_samples // cfg.num_classes
    fig, axes = plt.subplots(cfg.num_classes, n_per_class, figsize=(n_per_class * 1.2, cfg.num_classes * 1.2))

    for digit in range(cfg.num_classes):
        for j in range(n_per_class):
            idx = digit * n_per_class + j
            if idx < fake_imgs.size(0):
                axes[digit, j].imshow(fake_imgs[idx, 0].numpy(), cmap="gray")
            axes[digit, j].axis("off")
            if j == 0:
                axes[digit, j].set_ylabel(str(digit), fontsize=10, rotation=0, labelpad=15)

    tag = "final" if final else f"epoch_{epoch:03d}"
    plt.suptitle(f"条件GAN生成样本 (Epoch {epoch})", fontsize=12)
    save_path = os.path.join(cfg.save_dir, f"conditional_samples_{tag}.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()

    if final:
        print(f"✓ 条件生成样本已保存: {save_path}")


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

    axes[1].plot(epochs, history["D_real_acc"], "g-", label="D(真实)", linewidth=1.5)
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("准确率")
    axes[1].set_title("判别器对真实图像的判断")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["D_fake_acc"], "m-", label="D(生成)", linewidth=1.5)
    axes[2].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("准确率")
    axes[2].set_title("判别器对生成图像的判断")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()


# ============================================================
# Step 7: 生成函数
# ============================================================
@torch.no_grad()
def generate_by_class(G, digit, num_images, cfg):
    """
    生成指定数字的图像。

    参数:
        G: 训练好的生成器
        digit: 目标数字(0-9)
        num_images: 生成数量
        cfg: 配置
    返回:
        images: 生成的图像张量
    """
    G.eval()
    z = torch.randn(num_images, cfg.latent_dim, device=cfg.device)
    labels = torch.full((num_images,), digit, dtype=torch.long, device=cfg.device)
    fake_imgs = G(z, labels)
    fake_imgs = fake_imgs * 0.5 + 0.5
    fake_imgs = fake_imgs.clamp(0, 1)
    return fake_imgs.cpu()


# ============================================================
# Step 8: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("条件GAN - 按数字标签生成MNIST手写数字")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    print("\n加载数据集...")
    dataloader = get_dataloaders(cfg)

    G = ConditionalGenerator(cfg).to(cfg.device)
    D = ConditionalDiscriminator(cfg).to(cfg.device)

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"\n生成器参数量: {g_params:,}")
    print(f"判别器参数量: {d_params:,}")

    G, D, history = train(G, D, dataloader, cfg)

    g_path = os.path.join(cfg.save_dir, "conditional_generator.pth")
    d_path = os.path.join(cfg.save_dir, "conditional_discriminator.pth")
    torch.save(G.state_dict(), g_path)
    torch.save(D.state_dict(), d_path)
    print(f"\n✓ 生成器已保存: {g_path}")
    print(f"✓ 判别器已保存: {d_path}")

    print("\n生成可视化...")
    plot_training_curves(history, cfg)

    # 演示: 按数字生成
    print("\n按数字生成演示:")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for digit in range(10):
        imgs = generate_by_class(G, digit, 1, cfg)
        row, col = digit // 5, digit % 5
        axes[row, col].imshow(imgs[0, 0].numpy(), cmap="gray")
        axes[row, col].set_title(f"数字 {digit}", fontsize=12)
        axes[row, col].axis("off")
    plt.suptitle("条件GAN: 按指定数字生成", fontsize=14)
    save_path = os.path.join(cfg.save_dir, "generate_by_digit.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 按数字生成图像已保存: {save_path}")
    plt.close()

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
