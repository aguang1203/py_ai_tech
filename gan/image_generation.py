"""
=============================================================================
GAN 图像生成任务模板 (DCGAN for Image Generation)
=============================================================================

【原理】
生成对抗网络(GAN)是深度学习中最具创意的发明之一——让两个神经网络"对抗"，
一个负责"造假"(生成器)，一个负责"打假"(判别器)，在对抗中共同进步。

核心思想：
  生成器G: 噪声z → 假图像    目标: 骗过判别器
  判别器D: 图像 → 真/假      目标: 识别真假图像

训练过程就像"造假者vs鉴定师"的博弈：
  第1轮: G造的图很丑，D一眼看穿
  第10轮: G造的图有点像了，D偶尔被骗
  第50轮: G造的图很逼真，D很难分辨
  最终:   G生成的图像几乎与真实图像无法区分

【DCGAN的关键改进】
原始GAN用全连接层，DCGAN用卷积/反卷积，效果大幅提升：
  1. 生成器用转置卷积(ConvTranspose2d)实现上采样
  2. 判别器用步进卷积(Conv2d stride=2)替代池化
  3. 用BatchNorm稳定训练
  4. 去除全连接层，全卷积结构

【GAN训练的难点】
1. 模式崩溃(Mode Collapse): 生成器只产生少数几种图像
2. 训练不稳定: D太强→G梯度消失；D太弱→G没有学习信号
3. 超参数敏感: 学习率、网络结构都需要仔细调整
4. 评估困难: 没有单一的损失值能衡量生成质量

【应用场景】
- 图像生成 (人脸/风景/艺术) ← 本模板使用
- 图像超分辨率 (低清→高清)
- 图像修复 (补全缺失部分)
- 数据增强 (生成训练样本)
- 风格迁移 (照片→油画风格)

【本数据集: MNIST】
- 10个类别: 数字0-9
- 70,000张 28×28 灰度图像
- DCGAN学习数字的分布，从随机噪声生成逼真的手写数字

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python gan/image_generation.py
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
# Step 2: 配置超参数 (修改这里即可适配你的数据)
# ============================================================
class CONFIG:
    """超参数配置中心 —— 所有可调参数集中在此，方便统一管理和实验对比。"""

    # --- 数据相关 ---
    # data_dir: 数据集存放目录
    data_dir = "data"

    # image_size=28: 生成图像尺寸(MNIST)
    image_size = 28

    # in_channels=1: 输入通道数(灰度图)
    in_channels = 1

    # --- 生成器相关 ---
    # latent_dim=100: 噪声向量维度(z的维度)
    #   为什么100？DCGAN论文的标配，足够编码丰富的图像信息
    #   太小(如32): 噪声信息不够，生成多样性差
    #   太大(如256): 训练慢，且容易模式崩溃
    latent_dim = 100

    # gen_features=64: 生成器基础通道数
    #   所有通道数以此为基数倍增: 64→128→256→512
    #   为什么64？MNIST图像简单，64足够
    #   更大图像(64×64+): 建议用128
    gen_features = 64

    # --- 判别器相关 ---
    # disc_features=64: 判别器基础通道数
    #   通常与生成器保持一致
    #   为什么不小一点？判别器需要足够强才能提供有效的梯度信号
    disc_features = 64

    # --- 训练相关 ---
    # batch_size=128: 每次梯度更新使用128张图
    batch_size = 128

    # learning_rate=2e-4: 学习率
    #   为什么2e-4？DCGAN论文推荐值
    #   GAN对学习率非常敏感！太大训练崩溃，太小米收敛
    #   注意: G和D使用相同的学习率
    learning_rate = 2e-4

    # beta1=0.5: Adam的β1参数
    #   为什么0.5而不是默认0.9？DCGAN论文发现0.5更稳定
    #   0.9会让动量项过大，导致GAN训练振荡
    beta1 = 0.5

    # epochs=100: 训练轮数
    #   GAN没有"收敛"概念，通常训练到生成质量满意为止
    #   MNIST+DCGAN: 50轮有明显效果，100轮质量较好
    epochs = 100

    # --- 训练技巧 ---
    # d_steps_per_g=1: 每训练1次G，训练几次D
    #   为什么1？大多数情况下1:1就够了
    #   如果D太弱(G生成差但D被骗): 增加D训练次数(如2或3)
    #   如果D太强(G梯度消失): 减少D训练次数或加噪声
    d_steps_per_g = 1

    # label_smoothing=0.9: 标签平滑
    #   真实标签从1.0→0.9，防止D过于自信
    #   为什么？D太自信→梯度消失→G学不动
    #   0.9是常用值，范围0.8-0.95
    label_smoothing = 0.9

    # noise_std=0.1: 判别器输入噪声标准差
    #   给D的输入加噪声，削弱D的判别能力
    #   为什么？防止D过强导致G梯度消失
    #   0.1是常用值，0表示不加噪声
    noise_std = 0.1

    # --- 保存相关 ---
    # save_dir: 模型和图表保存目录
    save_dir = "gan/output/image_generation"

    # sample_interval=5: 每隔几轮保存生成样本
    sample_interval = 5

    # num_samples=64: 每次生成的样本数量(用于可视化)
    num_samples = 64

    # --- 数据加载优化 ---
    num_workers = min(4, os.cpu_count() or 1)

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 数据加载和预处理
# ============================================================
def get_dataloaders(cfg):
    """
    加载MNIST数据集并创建DataLoader。

    【GAN只需要真实图像，不需要标签！】
    - 分类任务: (图像, 标签) 对
    - GAN任务: 只需要图像，标签由我们自己构造(真=1, 假=0)
    - MNIST的标签在这里被忽略，只取图像

    【为什么GAN用MNIST的统计值做Normalize？】
    - GAN不需要标准化，通常用[-1, 1]范围
    - transforms.Normalize([0.5], [0.5]) 将[0,1]映射到[-1,1]
    - 原因: tanh激活输出[-1,1]，Normalize后数据与输出范围一致
    - 这比用真实均值/标准差效果更好
    """
    transform = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        # 将[0,1]映射到[-1,1]，与生成器tanh输出一致
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = datasets.MNIST(
        root=cfg.data_dir, train=True, download=True, transform=transform,
    )

    # GAN不需要测试集，只取训练集
    pin_mem = cfg.device.type == "cuda"
    pw = cfg.num_workers > 0

    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=pw, drop_last=True,
        # drop_last=True: 丢弃最后不完整的batch
        # 为什么？GAN的batch需要固定大小，否则BN可能出错
    )

    print(f"训练集: {len(dataset)}张 | Batch大小: {cfg.batch_size}")
    return dataloader


# ============================================================
# Step 4: 模型定义
# ============================================================
class Generator(nn.Module):
    """
    DCGAN生成器

    【架构设计思路】
    输入: 噪声向量 z (batch, 100)
      → 全连接重塑为 (batch, 512, 4, 4)  ← 初始"种子图像"
      → ConvTranspose2d(512→256, stride=2) → (batch, 256, 7, 7)
      → ConvTranspose2d(256→128, stride=2) → (batch, 128, 14, 14)
      → ConvTranspose2d(128→1,   stride=2) → (batch, 1, 28, 28)

    【转置卷积(ConvTranspose2d)是什么？】
    普通卷积: 大图→小图(下采样)
    转置卷积: 小图→大图(上采样)，也叫"反卷积"

    直观理解: 转置卷积 = 在像素之间插入空隙 → 卷积填充
    例: 7×7输入，stride=2 → 每个像素间插入1个空格 → 14×14(有效填充)

    【为什么用BatchNorm？】
    - 稳定训练: 防止中间层激活值过大/过小
    - 加速收敛: 让每层输入分布稳定，梯度更平滑
    - DCGAN论文发现BN对训练稳定至关重要

    【为什么输出用Tanh而非Sigmoid？】
    - Tanh输出[-1, 1]，与Normalize后的真实数据范围一致
    - Sigmoid输出[0, 1]，需要修改Normalize参数
    - Tanh的梯度更强(在0附近梯度=1)，有利于训练

    【参数量计算】
    FC: 100 × (512×4×4) = 819,200
    ConvT1: 512×256×4×4 + 256 = 2,097,408
    ConvT2: 256×128×4×4 + 128 = 524,416
    ConvT3: 128×1×4×4 + 1 = 2,049
    总计 ≈ 3.4M
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ngf = cfg.gen_features  # 64
        nc = cfg.in_channels     # 1

        # 初始尺寸: 噪声→7×7特征图
        # MNIST是28×28，需要2次上采样: 7→14→28
        self.init_size = cfg.image_size // 4  # 7

        # 全连接层: 将噪声映射为初始特征图
        self.fc = nn.Linear(cfg.latent_dim, ngf * 4 * self.init_size * self.init_size)

        # 反卷积层: 逐步上采样
        # 7×7 → 14×14 → 28×28 (只需要2次上采样)
        self.main = nn.Sequential(
            # 状态: (ngf*4) x 7 x 7
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),

            # 上采样1: 7×7 → 14×14
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            # 上采样2: 14×14 → 28×28
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # Tanh输出[-1, 1]，与Normalize后的真实数据范围一致
        )

        self._init_weights()

    def _init_weights(self):
        """
        权重初始化: DCGAN论文推荐的方法。

        【为什么GAN的初始化特别重要？】
        - GAN训练本身就是不稳定的，好的初始化可以提供一个好的起点
        - DCGAN论文发现: 正态分布N(0, 0.02)效果最好
        - BatchNorm的γ=1, β=0(默认值即可)
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, z):
        """
        前向传播

        数据流动:
        z: (batch, 100)              ← 随机噪声
          → fc: (batch, 256*7*7)     ← 展平
          → reshape: (batch, 256, 7, 7) ← 重塑为特征图
          → main: (batch, 1, 28, 28) ← 逐步上采样生成图像
        """
        out = self.fc(z)
        out = out.view(out.size(0), -1, self.init_size, self.init_size)
        out = self.main(out)
        return out


class Discriminator(nn.Module):
    """
    DCGAN判别器

    【架构设计思路】
    输入: 图像 (batch, 1, 28, 28)
      → Conv2d(1→128, stride=2)  → (batch, 128, 14, 14)
      → Conv2d(128→256, stride=2) → (batch, 256, 7, 7)
      → Conv2d(256→512, stride=2) → (batch, 512, 4, 4)
      → Flatten → FC → 1维输出(真/假概率)

    【为什么判别器用LeakyReLU而非ReLU？】
    - ReLU将负值截断为0 → "死神经元"问题
    - LeakyReLU让负值通过一个小斜率(如0.2) → 梯度始终存在
    - 这对判别器尤其重要: 需要对"假图像"也产生梯度信号
    - 0.2是DCGAN论文推荐的斜率

    【为什么判别器不用BatchNorm？】
    - 实际上DCGAN判别器可以用BatchNorm，但不能用在第一层
    - 第一层用BN会导致: 所有样本的统计量被混合，D无法区分真假
    - 简化方案: 判别器使用InstanceNorm或不用BN
    - 本模板使用LayerNorm替代(更稳定)

    【为什么判别器用Sigmoid输出？】
    - 输出单个概率值: 0=假图像, 1=真图像
    - 配合BCELoss使用
    - 注意: 有些实现不加Sigmoid，用BCEWithLogitsLoss(更数值稳定)
    """

    def __init__(self, cfg):
        super().__init__()
        ndf = cfg.disc_features  # 64
        nc = cfg.in_channels      # 1

        self.main = nn.Sequential(
            # 输入: (nc) x 28 x 28
            # 第1层: 不用BN，stride=2下采样
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 28→14

            # 第2层: 下采样
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 14→7

            # 输出层: 压缩为1维
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf * 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """权重初始化: 与生成器相同策略。"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, img):
        """
        前向传播

        参数:
            img: 图像 (batch, 1, 28, 28)
        返回:
            validity: 真/假概率 (batch, 1), 范围[0, 1]
        """
        return self.main(img)


# ============================================================
# Step 5: 训练函数
# ============================================================
def train(G, D, dataloader, cfg):
    """
    GAN训练循环。

    【GAN训练的独特之处】
    与普通模型不同，GAN需要同时训练两个网络：

    判别器训练:
      1. 真实图像 → D → 应输出1(真)
      2. 噪声 → G → 假图像 → D → 应输出0(假)
      3. 最大化: log(D(real)) + log(1 - D(G(z)))

    生成器训练:
      1. 噪声 → G → 假图像 → D → 应输出1(骗过D)
      2. 最大化: log(D(G(z)))
      3. 即: 让D认为假图像是真的

    【为什么G不直接最小化log(1-D(G(z)))？】
    - 理论上G应最小化log(1-D(G(z)))
    - 但训练初期，G生成的图很差，D很容易判假→D(G(z))≈0
    - log(1-0)=0, 梯度也≈0 → G学不到东西(梯度消失)
    - 实际做法: G最大化log(D(G(z)))，等效于最小化-D(G(z))
    - 这样D(G(z))≈0时，-log(0)=∞，梯度很大→G能学习

    【标签构造】
    真实图像 → 标签=1(或label_smoothing=0.9)
    生成图像 → 标签=0

    【训练不稳定的常见现象】
    1. D_loss→0: D太强，G学不动(梯度消失)
    2. G_loss→0: G骗过了D，但可能只生成一种图(模式崩溃)
    3. 两个loss都振荡: 学习率太大
    4. 生成图像模糊: 训练不够或网络太小
    """
    # 损失函数: 二元交叉熵
    # BCELoss = -[y·log(p) + (1-y)·log(1-p)]
    # y=1(真): -log(p), 希望p→1
    # y=0(假): -log(1-p), 希望p→0
    criterion = nn.BCELoss()

    # 优化器: 两个网络各自有独立的优化器
    # 为什么用Adam而非SGD？GAN训练不稳定，Adam自适应LR更稳健
    optimizer_G = optim.Adam(G.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))

    # 固定噪声用于可视化(每次用同样的z，观察生成质量的演变)
    fixed_noise = torch.randn(cfg.num_samples, cfg.latent_dim, device=cfg.device)

    # 训练记录
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

        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(cfg.device)

            # 构造标签
            # 真实标签: 用label_smoothing(如0.9)，防止D过于自信
            real_label = torch.full((batch_size, 1), cfg.label_smoothing, device=cfg.device)
            fake_label = torch.zeros(batch_size, 1, device=cfg.device)

            # 可选: 给D输入加噪声
            if cfg.noise_std > 0:
                real_imgs = real_imgs + cfg.noise_std * torch.randn_like(real_imgs)

            # =====================
            # 训练判别器D
            # =====================
            for _ in range(cfg.d_steps_per_g):
                optimizer_D.zero_grad()

                # 1. 真实图像的损失
                d_real = D(real_imgs)
                d_loss_real = criterion(d_real, real_label)

                # 2. 生成假图像
                z = torch.randn(batch_size, cfg.latent_dim, device=cfg.device)
                fake_imgs = G(z).detach()
                # detach(): 切断梯度流，不让D的梯度传到G

                # 3. 假图像的损失
                d_fake = D(fake_imgs)
                d_loss_fake = criterion(d_fake, fake_label)

                # 4. 总损失
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_D.step()

            # 记录D的准确率
            with torch.no_grad():
                d_real_acc = d_real.mean().item()
                d_fake_acc = 1.0 - d_fake.mean().item()

            # =====================
            # 训练生成器G
            # =====================
            optimizer_G.zero_grad()

            # 生成假图像(这次不detach，需要梯度传到G)
            z = torch.randn(batch_size, cfg.latent_dim, device=cfg.device)
            fake_imgs = G(z)

            # G的目标: 让D认为假图像是真的
            # 注意: 标签用real_label(1)，不是fake_label(0)
            # 这就是"骗过判别器"
            d_fake_for_g = D(fake_imgs)
            g_loss = criterion(d_fake_for_g, real_label)

            g_loss.backward()
            optimizer_G.step()

            # 记录
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
            D_real_accs.append(d_real_acc)
            D_fake_accs.append(d_fake_acc)

        # Epoch统计
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

        # 定期保存生成样本
        if epoch % cfg.sample_interval == 0 or epoch == 1:
            save_generated_samples(G, fixed_noise, cfg, epoch)

    # 保存最终生成样本
    save_generated_samples(G, fixed_noise, cfg, cfg.epochs, final=True)

    return G, D, history


# ============================================================
# Step 6: 可视化函数
# ============================================================
def save_generated_samples(G, fixed_noise, cfg, epoch, final=False):
    """
    保存生成样本图像。

    【为什么要用固定噪声？】
    - 每次用相同的z生成图像，可以观察同一组z下生成质量的演变
    - 如果随机z，无法判断是"变好了"还是"只是换了一组z"
    - 固定z让对比更直观
    """
    G.eval()
    with torch.no_grad():
        fake_imgs = G(fixed_noise).cpu()

    # 反归一化: [-1, 1] → [0, 1]
    fake_imgs = fake_imgs * 0.5 + 0.5
    fake_imgs = fake_imgs.clamp(0, 1)

    # 网格布局
    n = int(np.sqrt(cfg.num_samples))
    fig, axes = plt.subplots(n, n, figsize=(n, n))
    for i, ax in enumerate(axes.flat):
        if i < fake_imgs.size(0):
            ax.imshow(fake_imgs[i, 0].numpy(), cmap="gray")
        ax.axis("off")

    tag = "final" if final else f"epoch_{epoch:03d}"
    plt.suptitle(f"DCGAN生成样本 (Epoch {epoch})", fontsize=10)
    save_path = os.path.join(cfg.save_dir, f"samples_{tag}.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()

    if final:
        print(f"✓ 最终生成样本已保存: {save_path}")


def plot_training_curves(history, cfg):
    """绘制GAN训练曲线。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["G_loss"]) + 1)

    # 损失曲线
    axes[0].plot(epochs, history["G_loss"], "b-", label="G Loss", linewidth=1.5)
    axes[0].plot(epochs, history["D_loss"], "r-", label="D Loss", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("生成器/判别器损失")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # D对真实图像的判别准确率
    axes[1].plot(epochs, history["D_real_acc"], "g-", label="D(真实)=1", linewidth=1.5)
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="理想平衡点")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("准确率")
    axes[1].set_title("判别器对真实图像的判断")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # D对假图像的判别准确率
    axes[2].plot(epochs, history["D_fake_acc"], "m-", label="D(生成)=0", linewidth=1.5)
    axes[2].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="理想平衡点")
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


def plot_interpolation(G, cfg, n_interpolations=10):
    """
    可视化潜在空间插值。

    【为什么看插值？】
    - 在两个随机噪声之间线性插值，观察生成图像的平滑过渡
    - 好的GAN: 数字形态平滑变化(如0慢慢变成6)
    - 差的GAN: 中间出现模糊或突变
    - 这验证了G是否学到了有意义的特征表示
    """
    G.eval()
    with torch.no_grad():
        z1 = torch.randn(1, cfg.latent_dim, device=cfg.device)
        z2 = torch.randn(1, cfg.latent_dim, device=cfg.device)

        # 线性插值
        alphas = torch.linspace(0, 1, n_interpolations, device=cfg.device)
        interpolated = []
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            img = G(z)
            interpolated.append(img)

        interpolated = torch.cat(interpolated, dim=0).cpu()
        interpolated = interpolated * 0.5 + 0.5
        interpolated = interpolated.clamp(0, 1)

    fig, axes = plt.subplots(1, n_interpolations, figsize=(2 * n_interpolations, 2))
    for i, ax in enumerate(axes):
        ax.imshow(interpolated[i, 0].numpy(), cmap="gray")
        ax.axis("off")
        ax.set_title(f"α={alphas[i]:.1f}", fontsize=8)

    plt.suptitle("潜在空间插值 (z1 → z2)", fontsize=12)
    save_path = os.path.join(cfg.save_dir, "interpolation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 插值可视化已保存: {save_path}")
    plt.close()


# ============================================================
# Step 7: 生成函数
# ============================================================
@torch.no_grad()
def generate(G, num_images, cfg):
    """
    生成新图像。

    参数:
        G: 训练好的生成器
        num_images: 生成图像数量
        cfg: 配置
    返回:
        images: 生成的图像张量 (num_images, 1, 28, 28), 范围[0,1]
    """
    G.eval()
    z = torch.randn(num_images, cfg.latent_dim, device=cfg.device)
    fake_imgs = G(z)
    # 反归一化: [-1, 1] → [0, 1]
    fake_imgs = fake_imgs * 0.5 + 0.5
    fake_imgs = fake_imgs.clamp(0, 1)
    return fake_imgs.cpu()


# ============================================================
# Step 8: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("GAN 图像生成 - DCGAN生成MNIST手写数字")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # 加载数据
    print("\n加载数据集...")
    dataloader = get_dataloaders(cfg)

    # 创建模型
    G = Generator(cfg).to(cfg.device)
    D = Discriminator(cfg).to(cfg.device)

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"\n生成器参数量: {g_params:,}")
    print(f"判别器参数量: {d_params:,}")
    print(f"\n生成器结构:\n{G}")
    print(f"\n判别器结构:\n{D}")

    # 训练
    G, D, history = train(G, D, dataloader, cfg)

    # 保存模型
    g_path = os.path.join(cfg.save_dir, "generator.pth")
    d_path = os.path.join(cfg.save_dir, "discriminator.pth")
    torch.save(G.state_dict(), g_path)
    torch.save(D.state_dict(), d_path)
    print(f"\n✓ 生成器已保存: {g_path}")
    print(f"✓ 判别器已保存: {d_path}")

    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, cfg)
    plot_interpolation(G, cfg)

    # 生成新图像
    print("\n生成新图像...")
    new_imgs = generate(G, 16, cfg)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(new_imgs[i, 0].numpy(), cmap="gray")
        ax.axis("off")
    plt.suptitle("DCGAN新生成的手写数字", fontsize=12)
    save_path = os.path.join(cfg.save_dir, "new_generated.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 新生成图像已保存: {save_path}")
    plt.close()

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
