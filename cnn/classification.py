"""
=============================================================================
CNN 图像分类任务模板 (Convolutional Neural Network for Image Classification)
=============================================================================

【原理】
卷积神经网络(CNN)通过"卷积核"在图像上滑动，自动提取从低级(边缘/纹理)到高级(形状/对象)的
层次化特征。相比全连接网络(FNN)，CNN有三大核心优势：
  1. 局部连接：每个神经元只关注局部区域，而非整张图(模拟人类视觉的局部感受野)
  2. 参数共享：同一个卷积核在整个图像上复用，大幅减少参数量
  3. 平移不变性：无论物体在图像的哪个位置，都能被检测到

典型CNN架构：输入图像 → [卷积+激活+池化]×N → 展平 → 全连接层 → 类别概率
  - 卷积层(Conv2d)：提取局部特征，输出特征图(feature map)
  - 濠活函数(ReLU)：引入非线性，让网络能学习复杂模式
  - 池化层(MaxPool)：降低特征图尺寸，扩大感受野，增强鲁棒性
  - 批归一化(BatchNorm)：稳定训练，加速收敛
  - 全连接层(Linear)：将高级特征映射到类别

【应用场景】
- 手写数字识别 (MNIST, 10类)
- 自然图像分类 (Fashion-MNIST, 本模板使用Fashion-MNIST)
- 医学影像分类 (X光/CT/病理切片)
- 商品分类 (电商/零售)
- 动植物识别

【本数据集: Fashion-MNIST】
- 10个类别: T恤、裤子、套头衫、连衣裙、外套、凉鞋、衬衫、运动鞋、包、短靴
- 70,000张 28×28 灰度图像 (训练60,000 + 测试10,000)
- 特点: MNIST的现代替代品，比CIFAR-10更小(30MB vs 170MB)，下载更快更稳定
- 相比CIFAR-10: 灰度图(1通道 vs 3通道)，尺寸更小(28 vs 32)，但分类难度适中
  本模板使用Fashion-MNIST(灰度28×28，10类)而非CIFAR-10

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python cnn/classification.py
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
from torchvision import datasets, transforms, utils

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
# Step 2: 配置超参数 (修改这里即可适配你的数据)
# ============================================================
class CONFIG:
    """超参数配置中心 —— 所有可调参数集中在此，方便统一管理和实验对比。"""

    # --- 数据相关 ---
    # data_dir: 数据集存放目录
    #   torchvision会自动下载Fashion-MNIST到此目录
    #   如果已有数据，直接指向数据目录即可
    data_dir = "data"

    # num_classes=10: Fashion-MNIST有10个类别
    #   如果用自己的数据集，改为实际类别数
    num_classes = 10

    # class_names: 类别名称(用于可视化)
    class_names = [
        "T恤", "裤子", "套头衫", "连衣裙", "外套",
        "凉鞋", "衬衫", "运动鞋", "包", "短靴",
    ]

    # image_size=28: 输入图像尺寸(Fashion-MNIST原始尺寸)
    #   如果用自己的数据，根据图像大小调整
    #   常见值: MNIST/Fashion-MNIST=28, CIFAR=32, ImageNet=224
    image_size = 28

    # in_channels=1: 输入通道数(灰度图)
    #   灰度图=1, RGB彩色图=3, RGBA=4
    #   Fashion-MNIST是灰度图，所以=1
    in_channels = 1

    # test_size=0.2: 验证集比例(从训练集中划出)
    #   Fashion-MNIST训练集60000张，划出20%=12000张作为验证集
    test_size = 0.2

    # random_state=42: 固定随机种子，确保每次运行结果可复现
    random_state = 42

    # --- 模型相关 ---
    # conv_channels: 各卷积块的输出通道数
    #   [32, 64, 128]: 逐层加倍，提取越来越丰富的特征
    #   为什么逐层加倍？浅层提取简单特征(边缘)需要少通道，深层提取复杂特征(对象)需要多通道
    #   为什么不是[16, 32, 64]？Fashion-MNIST虽小但模式复杂，太少通道欠拟合
    #   为什么不是[64, 128, 256]？28x28小图，太多通道容易过拟合+浪费计算
    conv_channels = [32, 64, 128]

    # fc_dims: 全连接层维度
    #   [256]: 只用一个全连接隐藏层
    #   为什么是256？最后一个卷积块输出 128×4×4=2048维，压缩到256足够
    #   为什么不多加几层？小图像分类，1层FC已足够，太多容易过拟合
    fc_dims = [256]

    # dropout_rate=0.5: 全连接层Dropout比例
    #   为什么0.5？全连接层参数多(2048×256=52万)，0.5是防过拟合的标准值
    #   CNN的卷积层通常不加Dropout(BN已有正则化效果)，只对FC层加
    dropout_rate = 0.5

    # --- 训练相关 ---
    # batch_size=128: 每次梯度更新使用128张图
    #   为什么128？Fashion-MNIST图小(28x28x1≈0.8KB)，128张≈100KB，GPU轻松处理
    #   较大batch_size训练更稳定，梯度估计更准确
    batch_size = 128

    # learning_rate=1e-3: 初始学习率
    #   为什么1e-3？CNN+BN+Adam的标配学习率
    #   比FNN(5e-4)大，因为BN让训练更稳定，可以用更大LR
    learning_rate = 1e-3

    # epochs=50: 最大训练轮数
    #   早停会自动控制，50是上限
    #   Fashion-MNIST+自定义CNN，通常20-30轮收敛
    epochs = 50

    # weight_decay=5e-4: L2正则化强度
    #   为什么比FNN(1e-4)更大？CNN参数多(约100万)，需要更强正则化
    #   图像分类的标配值，ImageNet训练也用5e-4
    weight_decay = 5e-4

    # --- 早停策略 ---
    # early_stop_patience=10: 验证损失连续10轮不下降就停止
    #   为什么比FNN(20)小？CNN收敛快，10轮足以判断是否过拟合
    early_stop_patience = 10

    # --- 学习率调度器 ---
    # scheduler_type="cosine": 余弦退火调度
    #   为什么不用ReduceLROnPlateau？
    #   余弦退火让学习率平滑下降，训练后期自动精细调优
    #   ReduceLROnPlateau需要监控指标，CNN训练波动大容易误触发
    scheduler_type = "cosine"  # "cosine" 或 "step"

    # lr_step_size=15: StepLR的步长(仅scheduler_type="step"时生效)
    lr_step_size = 15

    # lr_gamma=0.1: StepLR的衰减因子
    lr_gamma = 0.1

    # --- 梯度裁剪 ---
    # max_grad_norm=5.0: 梯度L2范数上限
    #   为什么比FNN(1.0)大？CNN的梯度通常更大(卷积反向传播的梯度聚合)
    #   5.0足以防止梯度爆炸，又不会过度限制学习
    max_grad_norm = 5.0

    # --- 数据增强 ---
    # use_augmentation=True: 是否使用数据增强
    #   数据增强是CNN防止过拟合的核心手段，相当于"免费"扩充数据
    #   对小数据集尤其重要
    use_augmentation = True

    # random_crop_padding=4: RandomCrop的填充像素
    #   先填充4像素(28→36)，再随机裁剪回28×28
    #   效果：图像内容略有偏移，模拟物体位置变化
    random_crop_padding = 4

    # random_hflip_prob=0.5: 随机水平翻转概率
    #   为什么0.5？50%概率翻转，最常用的增强方式
    #   注意：数字/文字识别不适合翻转(6和9会混淆)
    #   Fashion-MNIST: 衣服翻转还是衣服，所以可以翻转
    random_hflip_prob = 0.5

    # --- 混合精度训练(AMP) ---
    # use_amp=True: 启用自动混合精度(Automatic Mixed Precision)
    #   【什么是混合精度？】
    #   传统训练用float32(32位浮点数)，AMP自动将部分运算转为float16(16位)
    #   float16计算速度更快、显存占用更少，但精度略低
    #   AMP智能选择哪些运算用float16(如矩阵乘法)，哪些保持float32(如损失计算)
    #   【效果】训练速度提升1.5-2倍，显存减少30-50%，精度几乎不变
    #   【为什么初学者也能用？】PyTorch的AMP是全自动的，不需要手动管理精度
    #   仅在CUDA(GPU)上有效，CPU会自动降级为普通训练
    use_amp = True

    # --- 数据加载优化 ---
    # num_workers: DataLoader的子进程数
    #   0: 主进程加载(慢，但兼容性最好)
    #   2-4: 多进程并行加载(快，推荐值)
    #   太多(>8): 进程切换开销大，反而变慢
    #   这里根据CPU核心数自动选择，但不超过4
    num_workers = min(4, os.cpu_count() or 1)

    # --- 保存相关 ---
    # save_dir: 模型和图表保存目录
    save_dir = "cnn/output/classification"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 数据加载和预处理
# ============================================================
def get_transforms(cfg):
    """
    创建训练集和测试集的数据变换管道。

    【为什么训练集和测试集的变换不同？】
    - 训练集：数据增强 + 标准化，让模型看到更多变化，提高泛化能力
    - 测试集：只做标准化，评估模型在原始数据上的真实性能
    - 数据增强只在训练时使用，测试时不能用(否则评估结果不可靠)

    【Fashion-MNIST标准化参数】
    - mean=[0.2860]: Fashion-MNIST单通道的均值
    - std=[0.3530]: Fashion-MNIST单通道的标准差
    - 这些值是对整个Fashion-MNIST训练集统计得出的
    - 灰度图只有1个通道，所以均值和标准差各只有1个值
    - 用准确的统计值让标准化更有效，加速训练收敛
    """
    # Fashion-MNIST的均值和标准差(对训练集统计得出)
    normalize = transforms.Normalize(
        mean=[0.2860],
        std=[0.3530],
    )

    if cfg.use_augmentation:
        # 训练集变换：数据增强 + 标准化
        # 【数据增强原理】
        # 通过对训练图像做随机变换，人工扩充训练数据的多样性
        # 等价于告诉模型："同一个物体可能有不同位置/方向/颜色"
        train_transform = transforms.Compose([
            # 1. 随机裁剪：先填充再裁剪，模拟物体位置偏移
            #    填充4像素 → 28x28变成36x36 → 随机裁剪回28x28
            #    为什么这样做？物体不一定总在画面正中央
            transforms.RandomCrop(cfg.image_size, padding=cfg.random_crop_padding),

            # 2. 随机水平翻转：50%概率左右翻转
            #    为什么可以翻转？衣服翻转还是衣服
            #    什么时候不能翻？数字识别(6翻转变9)、文字
            transforms.RandomHorizontalFlip(p=cfg.random_hflip_prob),

            # 3. 转为张量(0~255 → 0~1) + 标准化(减均值除标准差 → 约-2~2)
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # 测试集变换：只做标准化，不做任何增强
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def get_dataloaders(cfg):
    """
    加载Fashion-MNIST数据集并创建DataLoader。

    【DataLoader参数说明】
    - batch_size: 每批样本数，GPU并行处理
    - shuffle: 训练集打乱(防止模型记住顺序)，测试集不打乱
    - num_workers: 数据加载的子进程数
      为什么用2？Windows上>0有时报错，Linux上2-4比较合适
    - pin_memory=True: 将数据固定在内存，加速CPU→GPU传输
      仅在GPU训练时有效，CPU训练时可设False

    【为什么换用Fashion-MNIST？】
    - 使用Fashion-MNIST: 仅30MB，下载快，不会超时
    - Fashion-MNIST仅30MB，下载快速稳定
    - 同样是10类分类任务，难度适中，更适合入门学习
    - MNIST的替代品：数字太简单(99%+准确率)，衣服更有挑战性
    """
    train_transform, val_transform = get_transforms(cfg)

    # 下载并加载训练集(60,000张)
    # train=True: 加载训练集；download=True: 自动下载
    train_dataset = datasets.FashionMNIST(
        root=cfg.data_dir, train=True, download=True, transform=train_transform,
    )

    # 下载并加载测试集(10,000张)
    test_dataset = datasets.FashionMNIST(
        root=cfg.data_dir, train=False, download=True, transform=val_transform,
    )

    # 从训练集中划出验证集
    # 【为什么要验证集？】
    # 训练集: 训练模型参数
    # 验证集: 监控过拟合，调超参数，决定何时早停
    # 测试集: 最终评估，训练过程中绝对不能用
    n_total = len(train_dataset)
    n_val = int(n_total * cfg.test_size)
    n_train = n_total - n_val

    # 固定随机种子确保划分一致
    generator = torch.Generator().manual_seed(cfg.random_state)
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [n_train, n_val], generator=generator,
    )

    # 验证集使用val_transform(不做数据增强)
    # 创建一个用val_transform的数据集，然后用相同索引划分
    val_dataset = datasets.FashionMNIST(
        root=cfg.data_dir, train=True, download=False, transform=val_transform,
    )
    val_subset = torch.utils.data.Subset(val_dataset, val_subset.indices)

    # 创建DataLoader
    pin_mem = cfg.device.type == "cuda"
    # persistent_workers=True: 保持数据加载进程活跃，不每轮重建
    #   为什么？每次启动子进程需要时间，保持活跃可加速后续epoch
    #   仅在num_workers>0时有效
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
class ConvBlock(nn.Module):
    """
    卷积块: Conv2d → BatchNorm2d → ReLU

    【为什么每个卷积块只包含Conv+BN+ReLU？】
    这是现代CNN的基本构建单元，称为"CBR"结构：
    - Conv2d: 提取特征
    - BatchNorm2d: 归一化特征，稳定训练，允许更大学习率
    - ReLU: 引入非线性

    为什么不在块内加池化？
    池化是空间维度的操作(降采样)，放在块之间更灵活
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        """
        参数:
            in_ch: 输入通道数
            out_ch: 输出通道数
            kernel_size=3: 3×3卷积核(最常用的尺寸)
              为什么3×3？VGG证明两个3×3等价于一个5×5，但参数更少且多一次非线性
            padding=1: 填充1像素
              为什么padding=1？3×3核+1padding=输出尺寸不变，方便设计网络
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            # bias=False: 为什么不要偏置？
            # BN的公式: y = γ·(x-μ)/σ + β，β已经起到了偏置的作用
            # 加偏置是多余的，且增加参数量
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # inplace=True: 原地操作，节省约30%内存
        )

    def forward(self, x):
        return self.block(x)


class CNNClassifier(nn.Module):
    """
    CNN图像分类模型

    【架构设计思路】
    输入 (1, 28, 28)   ← Fashion-MNIST: 灰度图, 28×28像素
      → ConvBlock1(1→32) + ConvBlock1b(32→32) + MaxPool → (32, 14, 14)
      → ConvBlock2(32→64) + ConvBlock2b(64→64) + MaxPool → (64, 7, 7)
      → ConvBlock3(64→128) + ConvBlock3b(128→128) + MaxPool → (128, 3, 3)
      → AdaptiveAvgPool → (128, 1, 1)
      → Flatten → 128
      → FC(128→256) → BN → ReLU → Dropout → FC(256→10)

    【维度变化详解】
    输入 28×28 → MaxPool(÷2) → 14×14 → MaxPool(÷2) → 7×7 → MaxPool(÷2) → 3×3
    为什么7÷2=3不是4？MaxPool2d做向下取整: floor(7/2)=3
    为什么用AdaptiveAvgPool？不管最终特征图是3×3还是4×4，都压缩到1×1

    【为什么每个阶段用2个卷积层？】
    VGG的核心理念：用多个小卷积(3×3)代替大卷积(7×7)
    - 2个3×3的感受野 = 1个5×5，但参数更少(2×9C² vs 25C²)
    - 多一次非线性变换，表达能力更强
    - 训练更容易(更平滑的梯度流)

    【为什么用AdaptiveAvgPool而不是直接展平？】
    - 直接展平: 最后特征图128×4×4=2048维，FC层参数量大
    - AdaptiveAvgPool(1,1): 对每个通道做全局平均，128×1×1=128维
    - 好处: (1)大幅减少FC参数 (2)对输入尺寸不敏感
    - 坏处: 可能丢失空间信息(但分类任务通常够了)
    """

    def __init__(self, cfg):
        super().__init__()
        c = cfg.conv_channels  # [32, 64, 128]
        in_ch = cfg.in_channels

        # ---- 卷积特征提取部分 ----
        # 每个stage: 2个ConvBlock + 1个MaxPool
        # 2个ConvBlock保证足够的特征提取深度
        self.features = nn.Sequential(
            # Stage 1: 输入(1,28,28) → 输出(32,14,14)
            ConvBlock(in_ch, c[0]),       # 1→32 (灰度图只有1通道)
            ConvBlock(c[0], c[0]),         # 32→32，同通道卷积，精炼特征
            nn.MaxPool2d(2, 2),           # 28×28 → 14×14，空间降采样

            # Stage 2: 输入(32,14,14) → 输出(64,7,7)
            ConvBlock(c[0], c[1]),         # 32→64，通道加倍
            ConvBlock(c[1], c[1]),         # 64→64
            nn.MaxPool2d(2, 2),           # 14×14 → 7×7

            # Stage 3: 输入(64,7,7) → 输出(128,3,3)
            ConvBlock(c[1], c[2]),         # 64→128
            ConvBlock(c[2], c[2]),         # 128→128
            nn.MaxPool2d(2, 2),           # 7×7 → 3×3 (floor(7/2)=3)
        )

        # 全局平均池化
        # 将4×4特征图压缩为1×1，每个通道只保留一个平均值
        # 【为什么用全局平均池化？】
        # 1. 替代Flatten+FC的暴力连接，大幅减少参数
        # 2. 对空间位置不敏感，天然具有平移不变性
        # 3. 网络架构对输入尺寸更灵活
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ---- 全连接分类头 ----
        self.classifier = nn.Sequential(
            nn.Linear(c[2], cfg.fc_dims[0]),  # 128 → 256
            nn.BatchNorm1d(cfg.fc_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_rate),      # 0.5 Dropout
            nn.Linear(cfg.fc_dims[0], cfg.num_classes),  # 256 → 10
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        权重初始化: He初始化(Kaiming Normal)

        【为什么用He初始化？】
        - ReLU激活函数会将负值截断为0，如果权重太小，信号会逐层衰减(梯度消失)
        - He初始化让每层输出的方差≈输入的方差，保持信号强度
        - 公式: W ~ N(0, sqrt(2/fan_in))
        - 这是ReLU网络的标准初始化方式
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                # BN的γ初始化为1，β初始化为0(已是默认值)
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        数据流动:
        x: (batch, 1, 28, 28)   ← Fashion-MNIST灰度图
          → features: (batch, 128, 3, 3)
          → avgpool: (batch, 128, 1, 1)
          → flatten: (batch, 128)
          → classifier: (batch, 10)  ← logits(未激活)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (batch, 128, 1, 1) → (batch, 128)
        x = self.classifier(x)
        return x


# ============================================================
# Step 5: 训练函数
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, cfg, scaler=None):
    """
    训练一个epoch。

    【CNN训练的注意事项】
    1. model.train(): 启用BN和Dropout(训练模式)
    2. 梯度裁剪: 防止梯度爆炸(CNN尤其重要)
    3. 数据类型: 输入是4D张量 (batch, channel, height, width)
    4. 混合精度: 用autocast自动选择float16/float32，加速训练
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # 判断是否启用混合精度(仅CUDA有效)
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for inputs, targets in loader:
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)

        # 前向传播(混合精度)
        # 【autocast做了什么？】
        # 自动将矩阵乘法、卷积等大运算转为float16(快)
        # 而损失计算、归一化等精度敏感操作保持float32(准)
        # 就像"该快的地方快，该准的地方准"，不需要手动指定
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        # 【GradScaler做了什么？】
        # float16的梯度值很小，直接用可能下溢(变成0)
        # Scaler先放大loss再反向传播，防止梯度消失
        # 然后再缩放回来更新参数
        if scaler is not None:
            scaler.scale(loss).backward()
            # 梯度裁剪: 先unscale再裁剪，否则裁剪阈值不对
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

        # 统计
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, cfg):
    """
    评估模型性能。

    @torch.no_grad(): 不计算梯度，节省GPU显存
    model.eval(): 切换到评估模式
      - BN使用全局统计量(训练时累积的running mean/var)
      - Dropout不生效(不丢弃任何神经元)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for inputs, targets in loader:
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        # 推理时也用autocast加速，不影响精度
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
    """
    完整训练流程: 训练 + 验证 + 早停 + 学习率调度

    【训练流程】
    每个epoch:
      1. 训练一个epoch (前向+反向+优化)
      2. 在验证集上评估
      3. 更新学习率
      4. 检查是否需要早停
      5. 保存最优模型
    """
    # 损失函数
    # CrossEntropyLoss: 内含Softmax，模型输出logits即可
    # 为什么不需要class_weight? Fashion-MNIST每类6000张(训练集)，完全均衡
    criterion = nn.CrossEntropyLoss()

    # 优化器
    # Adam: 自适应学习率，几乎不需要调参
    # weight_decay=5e-4: L2正则化，防止过拟合
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    # 学习率调度器
    if cfg.scheduler_type == "cosine":
        # 余弦退火: LR从初始值平滑降到接近0
        # 【为什么用余弦退火？】
        # 前期大LR快速收敛，后期小LR精细调优
        # 比阶梯式下降更平滑，训练曲线更好看
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    else:
        # 阶梯下降: 每step_size个epoch，LR乘以gamma
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    # 早停相关变量
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # 记录训练曲线
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # 混合精度训练的GradScaler
    # 【为什么需要Scaler？】
    # float16的数值范围小(最小约6e-8)，小梯度会变成0(下溢)
    # Scaler通过放大loss来放大梯度，防止下溢，再缩放回来更新参数
    use_amp = cfg.use_amp and cfg.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")
    print(f"设备: {cfg.device} | 优化器: Adam(lr={cfg.learning_rate}) | 调度器: {cfg.scheduler_type} | AMP: {use_amp}")

    for epoch in range(1, cfg.epochs + 1):
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, cfg, scaler)
        # 验证
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, cfg)

        # 记录
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]["lr"]

        # 打印
        print(f"Epoch {epoch:3d}/{cfg.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")

        # 更新学习率
        scheduler.step()

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ✓ 最优模型已更新 (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"\n⚠ 早停触发: 验证损失连续{cfg.early_stop_patience}轮未改善")
                break

    # 恢复最优模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(cfg.device)
        print(f"\n✓ 已恢复最优模型 (Val Loss: {best_val_loss:.4f})")

    return model, history


# ============================================================
# Step 6: 可视化函数
# ============================================================
def plot_training_curves(history, cfg):
    """绘制训练曲线(损失+准确率)"""
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
    """
    绘制混淆矩阵。

    【如何解读混淆矩阵？】
    - 对角线: 正确预测数(越亮越好)
    - 非对角线: 错误预测数(越暗越好)
    - 行=真实类别，列=预测类别
    - 例如: 真实是"猫"，被预测为"狗"，则在"猫"行"狗"列+1
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=cfg.class_names, yticklabels=cfg.class_names,
           ylabel="真实类别", xlabel="预测类别",
           title="混淆矩阵")

    # 在每个格子中显示数字
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
    """可视化预测结果：展示部分测试图像及模型预测"""
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_samples], labels[:num_samples]

    with torch.no_grad():
        outputs = model(images.to(cfg.device))
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(1).cpu()

    # 反标准化用于显示
    mean = torch.tensor([0.2860]).view(1, 1, 1)
    std = torch.tensor([0.3530]).view(1, 1, 1)

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    for i, ax in enumerate(axes.flat):
        if i >= num_samples:
            break
        img = images[i] * std + mean  # 反标准化
        img = img.squeeze().numpy().clip(0, 1)  # 灰度图去掉通道维

        ax.imshow(img, cmap="gray")
        true_name = cfg.class_names[labels[i]]
        pred_name = cfg.class_names[preds[i]]
        confidence = probs[i, preds[i]].item()

        color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(f"真实: {true_name}\n预测: {pred_name} ({confidence:.1%})",
                     color=color, fontsize=9)
        ax.axis("off")

    plt.suptitle("CNN图像分类预测结果", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "sample_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 预测结果已保存: {save_path}")
    plt.close()


def plot_feature_maps(model, test_loader, cfg, layer_idx=0, num_maps=8):
    """
    可视化卷积特征图。

    【为什么要看特征图？】
    - 理解CNN"看到了什么"
    - 第1层: 通常检测边缘、颜色
    - 中间层: 检测纹理、形状
    - 深层: 检测语义(物体的部件)

    - 全黑/全白/噪声: 训练有问题
    - 有意义的纹理/形状: 训练正常
    """
    model.eval()
    images, _ = next(iter(test_loader))
    img = images[0:1].to(cfg.device)

    # 获取第一个卷积块的输出
    with torch.no_grad():
        x = img
        for i, layer in enumerate(model.features):
            x = layer(x)
            if i == layer_idx:
                feature_maps = x
                break

    # 可视化前num_maps个特征图
    fig, axes = plt.subplots(1, num_maps, figsize=(2 * num_maps, 2))
    for i in range(num_maps):
        if i < feature_maps.shape[1]:
            axes[i].imshow(feature_maps[0, i].cpu().numpy(), cmap="viridis")
        axes[i].axis("off")
        axes[i].set_title(f"通道{i}", fontsize=8)

    plt.suptitle(f"第{layer_idx+1}层卷积特征图", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "feature_maps.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 特征图已保存: {save_path}")
    plt.close()


# ============================================================
# Step 7: 预测函数
# ============================================================
@torch.no_grad()
def predict(model, image_tensor, cfg):
    """
    对单张图像进行预测。

    参数:
        image_tensor: 预处理后的图像张量 (1, 3, 32, 32)
    返回:
        pred_class: 预测类别索引
        pred_name: 预测类别名称
        confidence: 预测置信度
        probabilities: 各类别概率
    """
    model.eval()
    output = model(image_tensor.to(cfg.device))
    probabilities = torch.softmax(output, dim=1)
    confidence, pred_class = probabilities.max(1)
    pred_name = cfg.class_names[pred_class.item()]

    return pred_class.item(), pred_name, confidence.item(), probabilities.cpu().numpy()


# ============================================================
# Step 8: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("CNN 图像分类 - Fashion-MNIST")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 创建输出目录
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 加载数据
    print("\n加载数据集...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # 创建模型
    model = CNNClassifier(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型: CNNClassifier")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"\n模型结构:\n{model}")

    # 训练
    model, history = train(model, train_loader, val_loader, cfg)

    # 在测试集上评估
    print(f"\n{'='*60}")
    print("测试集评估...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, cfg)
    print(f"测试集 Loss: {test_loss:.4f} | 准确率: {test_acc:.4f}")

    # 详细分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=cfg.class_names, digits=4))

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "cnn_classifier.pth")
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
    plot_feature_maps(model, test_loader, cfg)

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
