"""
=============================================================================
CNN 人脸识别任务模板 (Face Recognition with Embedding Network)
=============================================================================

【原理】
人脸识别的核心问题：给定一张人脸图像，判断这是谁。

两阶段流程：
  第1阶段 - 人脸检测: 从图像中找到人脸位置，裁剪对齐
  第2阶段 - 人脸识别: 将裁剪的人脸映射到特征向量(嵌入)，计算相似度

本模板聚焦第2阶段(人脸识别)，使用"分类预训练 + 嵌入提取"的方法：
  1. 训练一个CNN分类器，将人脸按身份分类
  2. 去掉最后的分类层，取倒数第二层作为"人脸嵌入向量"
  3. 用余弦相似度比较两张人脸的嵌入向量，判断是否为同一人

【为什么用嵌入而不是直接分类？】
- 分类: 只能识别训练过的N个人，新人员需要重新训练
- 嵌入: 将人脸映射到通用空间，新人员只需一张注册照片即可识别
- 嵌入向量维度低(128/256维)，存储和比较都很快
- 余弦相似度有明确的几何意义(向量夹角)

【两种识别模式】
1. 人脸验证(Face Verification): 这两张脸是同一人吗？→ 是/否
   应用: 人脸解锁、身份核验
2. 人脸辨识(Face Identification): 这张脸是谁？→ 在库中搜索最相似的人
   应用: 考勤打卡、安防监控

【应用场景】
- 手机人脸解锁
- 门禁/考勤系统
- 安防监控(人员识别)
- 社交媒体(自动标签)
- 金融(身份核验)

【本数据集: 合成人脸数据】
- 40个人，每人10张照片，共400张
- 图像尺寸: 64×64 灰度图
- 生成方式: 每个身份有独特基础模式 + 随机变化(噪声/亮度/平移)
- 特点: 无需下载，保证代码可运行；保留人脸识别任务的核心特性
- 替代原因: Olivetti Faces下载源(figshare)经常返回403错误

【使用方法】
1. 直接运行: python cnn/face_recognition.py
2. 数据集自动下载
=============================================================================
"""

# ============================================================
# Step 1: 导入必要的库
# ============================================================
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, pairwise_distances,
)
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
    """超参数配置中心 —— 人脸识别任务的所有可调参数。"""

    # --- 数据相关 ---
    # data_dir: 数据集存放目录
    data_dir = "data"

    # num_identities=40: 身份(人)的数量
    #   40个人，如果用自己的数据集，改为实际人数
    num_identities = 40

    # images_per_identity=10: 每个人的图像数
    #   每人生成10张不同变化的照片
    images_per_identity = 10

    # image_size=64: 输入图像尺寸
    #   64×64 灰度图
    image_size = 64

    # in_channels=1: 输入通道数(灰度图)
    #   人脸识别常用灰度图(颜色对身份识别帮助不大)
    #   如果用RGB图，改为3
    in_channels = 1

    # test_size=0.2: 测试集比例
    test_size = 0.2

    # random_state=42: 随机种子
    random_state = 42

    # --- 模型相关 ---
    # embedding_dim=128: 嵌入向量维度
    #   【为什么是128？】
    #   - 32维: 太小，信息不够区分不同人脸
    #   - 128维: 平衡点，足以编码人脸特征，又不太大
    #   - 512维: FaceNet用的512维，但数据量大(百万级)时才需要
    #   - 规则: 嵌入维度 ≈ 身份数 × 3~5 (40人 × 3 ≈ 120)
    embedding_dim = 128

    # conv_channels: 各卷积层输出通道数
    #   [32, 64, 128, 256]: 逐层加倍
    #   为什么4层？64×64小图，4层卷积后到4×4，足够提取人脸特征
    #   为什么通道逐层加倍？浅层提取简单特征(边缘/轮廓)，深层提取复杂特征(五官关系)
    conv_channels = [32, 64, 128, 256]

    # dropout_rate=0.5: Dropout比例
    dropout_rate = 0.5

    # --- 训练相关 ---
    # batch_size=32: 批次大小
    #   400张图，batch=32，每个epoch约12次更新
    batch_size = 32

    # learning_rate=1e-3: 初始学习率
    learning_rate = 1e-3

    # epochs=50: 最大训练轮数
    epochs = 50

    # weight_decay=1e-4: L2正则化
    weight_decay = 1e-4

    # --- 早停策略 ---
    early_stop_patience = 10

    # --- 学习率调度器 ---
    lr_factor = 0.5       # LR衰减因子
    lr_patience = 5       # 调度器耐心值
    lr_min = 1e-6         # LR下限

    # --- 梯度裁剪 ---
    max_grad_norm = 5.0

    # --- 混合精度训练(AMP) ---
    # use_amp=True: 启用自动混合精度，训练速度提升1.5-2倍
    #   仅CUDA(GPU)有效，CPU自动降级为普通训练
    #   原理: 自动将部分float32运算转为float16，加快速度、减少显存
    use_amp = True

    # --- 数据加载优化 ---
    # num_workers: 多进程并行加载数据
    #   0=主进程加载(慢)，2-4=推荐值
    num_workers = min(4, os.cpu_count() or 1)

    # --- 识别相关 ---
    # similarity_threshold=0.5: 余弦相似度阈值
    #   > 0.5: 同一人；<= 0.5: 不同人
    #   为什么0.5？经验值，实际应根据验证集ROC曲线选择最优阈值
    similarity_threshold = 0.5

    # top_k=3: 辨识模式返回最相似的K个人
    top_k = 3

    # --- 保存相关 ---
    save_dir = "cnn/output/face_recognition"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 数据加载
# ============================================================
class FaceDataset(Dataset):
    """
    人脸数据集。

    【合成人脸数据集】
    - 每个身份有独特的基础模式(模拟不同人的五官特征)
    - 同一身份的不同照片有变化(模拟表情/光照变化)
    """

    def __init__(self, images, labels, transform=None):
        """
        参数:
            images: numpy数组 (N, 64, 64)
            labels: numpy数组 (N,)，身份ID (0~39)
            transform: 图像变换
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 转为Tensor: (64, 64) → (1, 64, 64)
        # 人脸是灰度图，只有1个通道
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        # 归一化到[0,1]
        if image.max() > 1:
            image = image / 255.0

        if self.transform:
            image = self.transform(image)

        return image, label


def generate_synthetic_faces(cfg):
    """
    生成合成人脸数据(无需下载)。

    【为什么用合成数据替代Olivetti Faces？】
    Olivetti Faces数据集的下载源(figshare.com)经常返回HTTP 403错误，
    导致代码无法运行。合成数据保证了代码的可执行性。

    【合成策略】
    - 每个身份(id)用独特的随机基础模式表示(模拟不同人的面部特征)
    - 同一身份的不同照片添加随机噪声和亮度变化(模拟表情/光照变化)
    - 生成的数据保留人脸识别任务的核心特性：
      类内变化小(同一人相似)、类间差异大(不同人可区分)

    【合成方法】
    1. 为每个身份生成独特的基础模式：
       - 低频随机背景(先小图后放大，自然平滑)
       - 添加"五官"特征点(眼睛、鼻子、嘴巴)，位置/大小因人而异
    2. 为每张照片添加变化：
       - 高斯噪声: 模拟传感器噪声
       - 亮度偏移: 模拟不同光照条件
       - 微小平移(1像素): 模拟头部轻微移动
    """
    size = cfg.image_size  # 64
    n_id = cfg.num_identities  # 40
    n_per = cfg.images_per_identity  # 10
    n_total = n_id * n_per  # 400

    images = np.zeros((n_total, size, size), dtype=np.float32)
    labels = np.zeros(n_total, dtype=np.int64)

    for pid in range(n_id):
        # 每个身份的独特基础模式
        # 用固定种子确保同一身份每次生成相同的基础模式
        rng = np.random.RandomState(cfg.random_state + pid * 100)

        # 生成低频基础模式: 先小图后放大，自然平滑
        # 为什么先小图？小图只有8×8=64个值，放大后自然形成平滑的低频纹理
        # 这模拟了人脸的肤色/轮廓等大尺度特征
        small_size = size // 8  # 8
        small = rng.rand(small_size, small_size).astype(np.float32)
        # 最近邻放大: 8×8 → 64×64
        base = np.repeat(np.repeat(small, 8, axis=0), 8, axis=1)[:size, :size]
        # 归一化到[0.2, 0.8]范围(模拟肤色基础亮度)
        base = base * 0.6 + 0.2

        # 添加五官特征(每个人的位置/大小略有不同，模拟不同人的长相差异)
        y_grid, x_grid = np.ogrid[:size, :size]

        # 眼睛: 两个暗色圆形区域
        # 为什么暗色？眼球和眼眶比周围皮肤暗
        eye_y = int(size * (0.32 + rng.rand() * 0.06))
        eye_r = int(size * (0.03 + rng.rand() * 0.015))
        for ex in [int(size * (0.33 + rng.rand() * 0.04)),
                    int(size * (0.63 + rng.rand() * 0.04))]:
            eye_mask = (x_grid - ex)**2 + (y_grid - eye_y)**2 <= eye_r**2
            base[eye_mask] *= 0.4  # 暗化

        # 鼻子: 小暗色区域
        nose_y = int(size * (0.46 + rng.rand() * 0.04))
        nose_x = int(size * (0.47 + rng.rand() * 0.06))
        base[max(0, nose_y-1):nose_y+2, max(0, nose_x-1):nose_x+2] *= 0.65

        # 嘴巴: 水平暗色条
        # 为什么用条形？嘴巴呈水平延伸的形状
        mouth_y = int(size * (0.62 + rng.rand() * 0.06))
        mouth_w = int(size * (0.10 + rng.rand() * 0.04))
        mouth_cx = int(size * 0.5)
        x_start = max(0, mouth_cx - mouth_w)
        x_end = min(size, mouth_cx + mouth_w)
        base[max(0, mouth_y-1):mouth_y+1, x_start:x_end] *= 0.55

        # 生成该身份的多张照片(每张略有不同)
        for j in range(n_per):
            idx = pid * n_per + j

            # 高斯噪声: 模拟传感器噪声和细纹变化
            # 为什么σ=0.03？太小看不出变化，太大会掩盖身份特征
            noise = rng.randn(size, size) * 0.03

            # 亮度偏移: 模拟不同光照条件
            # 为什么±0.04？范围适中，模拟轻微光照变化
            brightness = rng.randn() * 0.04

            # 微小平移(±1像素): 模拟头部轻微移动/对齐偏差
            shift_x = rng.randint(-1, 2)
            shift_y = rng.randint(-1, 2)

            img = base + noise + brightness
            img = np.roll(np.roll(img, shift_x, axis=1), shift_y, axis=0)
            images[idx] = np.clip(img, 0, 1)
            labels[idx] = pid

    # 打乱顺序(确保训练时每个batch包含不同身份)
    shuffle_rng = np.random.RandomState(cfg.random_state)
    perm = shuffle_rng.permutation(n_total)
    return images[perm], labels[perm]


def load_data(cfg):
    """
    加载合成人脸数据集。

    【数据划分策略】
    - 400张图，8:2划分 → 训练320张，测试80张
    - 每人10张 → 训练8张，测试2张
    - 使用stratify确保每个人在训练/测试集都有代表
    """
    print("生成合成人脸数据集...")
    images, labels = generate_synthetic_faces(cfg)

    print(f"数据集: {images.shape[0]}张图, {len(np.unique(labels))}个人")

    # 划分训练/测试集
    from sklearn.model_selection import train_test_split
    train_img, test_img, train_lbl, test_lbl = train_test_split(
        images, labels, test_size=cfg.test_size,
        stratify=labels, random_state=cfg.random_state,
    )

    # 创建Dataset和DataLoader
    train_dataset = FaceDataset(train_img, train_lbl)
    test_dataset = FaceDataset(test_img, test_lbl)

    pin_mem = cfg.device.type == "cuda"
    pw = cfg.num_workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=pin_mem,
        num_workers=cfg.num_workers, persistent_workers=pw,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin_mem,
        num_workers=cfg.num_workers, persistent_workers=pw,
    )

    print(f"训练集: {len(train_dataset)}张 | 测试集: {len(test_dataset)}张")

    # 保存测试集用于验证和可视化
    return train_loader, test_loader, test_dataset


# ============================================================
# Step 4: 模型定义
# ============================================================
class FaceEmbeddingNet(nn.Module):
    """
    人脸嵌入网络: 输入人脸图像 → 输出128维嵌入向量

    【架构设计】
    输入 (1, 64, 64)
      → Conv(1→32) + BN + ReLU + MaxPool → (32, 32, 32)
      → Conv(32→64) + BN + ReLU + MaxPool → (64, 16, 16)
      → Conv(64→128) + BN + ReLU + MaxPool → (128, 8, 8)
      → Conv(128→256) + BN + ReLU + MaxPool → (256, 4, 4)
      → AdaptiveAvgPool(1,1) → (256, 1, 1)
      → Flatten → 256
      → FC(256→128) + BN + ReLU → 128维嵌入
      → FC(128→num_identities) → 分类logits

    【训练和推理使用不同层】
    训练时: 使用完整网络(含分类头)，用CrossEntropyLoss训练
    推理时: 只用嵌入层，取128维嵌入向量计算相似度

    【为什么用分类训练来获得嵌入？】
    这是"Proxy-based"方法的思路：
    - 分类层每个身份对应一个权重向量(代理向量/Proxy)
    - 训练时，同一个人的特征被拉向同一个Proxy
    - 不同人的特征被推开
    - 最终嵌入空间中：同一人距离近，不同人距离远
    """

    def __init__(self, cfg):
        super().__init__()
        c = cfg.conv_channels
        in_ch = cfg.in_channels

        # ---- 卷积特征提取 ----
        self.features = nn.Sequential(
            # Block 1: (1, 64, 64) → (32, 32, 32)
            nn.Conv2d(in_ch, c[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: (32, 32, 32) → (64, 16, 16)
            nn.Conv2d(c[0], c[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: (64, 16, 16) → (128, 8, 8)
            nn.Conv2d(c[1], c[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(c[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: (128, 8, 8) → (256, 4, 4)
            nn.Conv2d(c[2], c[3], 3, padding=1, bias=False),
            nn.BatchNorm2d(c[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 全局平均池化: (256, 4, 4) → (256, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # ---- 嵌入层 ----
        # 将256维特征压缩为128维嵌入
        # 【为什么需要嵌入层而不是直接用256维？】
        # 1. 降维: 128维比256维更紧凑，计算相似度更快
        # 2. 正则化: 瓶颈层迫使网络学习最本质的人脸特征
        # 3. 通用性: 嵌入维度与身份数解耦，新增身份不影响
        self.embedding = nn.Sequential(
            nn.Linear(c[3], cfg.embedding_dim),
            nn.BatchNorm1d(cfg.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_rate),
        )

        # ---- 分类头(仅训练时使用) ----
        # 映射嵌入到身份ID
        # 推理时不用这个层，只用嵌入层
        self.classifier = nn.Linear(cfg.embedding_dim, cfg.num_identities)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """He初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播(训练模式)。

        返回: (logits, embedding)
          logits: 分类得分 (batch, num_identities)
          embedding: 嵌入向量 (batch, embedding_dim)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        emb = self.embedding(x)
        logits = self.classifier(emb)
        return logits, emb

    def get_embedding(self, x):
        """
        提取嵌入向量(推理模式)。

        只返回嵌入向量，不经过分类头。
        用于人脸验证和辨识。
        """
        self.eval()
        with torch.no_grad():
            x = self.features(x)
            x = torch.flatten(x, 1)
            emb = self.embedding(x)
        return emb


# ============================================================
# Step 5: 训练函数
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, cfg, scaler=None):
    """训练一个epoch。"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for images, labels in loader:
        images = images.to(cfg.device)
        labels = labels.to(cfg.device)

        # 前向传播(混合精度)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _ = model(images)
            loss = criterion(logits, labels)

        # 反向传播
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

        # 统计
        total_loss += loss.item() * images.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, cfg):
    """评估模型(分类准确率)。"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_embeddings = []
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for images, labels in loader:
        images = images.to(cfg.device)
        labels_t = labels.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, embeddings = model(images)
            loss = criterion(logits, labels_t)

        total_loss += loss.item() * images.size(0)
        _, preds = logits.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_t.cpu().numpy())
        all_embeddings.append(embeddings.cpu())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    all_embeddings = torch.cat(all_embeddings, dim=0)

    return avg_loss, acc, all_preds, all_labels, all_embeddings


def train(model, train_loader, val_loader, cfg):
    """完整训练流程。"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor,
        patience=cfg.lr_patience, min_lr=cfg.lr_min,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # 混合精度GradScaler
    use_amp = cfg.use_amp and cfg.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")
    print(f"设备: {cfg.device} | 优化器: Adam(lr={cfg.learning_rate}) | AMP: {use_amp}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, cfg, scaler)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, cfg)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{cfg.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ✓ 最优模型已更新")
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
# Step 6: 人脸验证与辨识
# ============================================================
def cosine_similarity(emb1, emb2):
    """
    计算余弦相似度。

    【余弦相似度】
    cos(θ) = (A·B) / (||A|| × ||B||)
    - 值域: [-1, 1]
    - 1: 完全相同方向(最相似)
    - 0: 正交(无关)
    - -1: 完全相反方向

    【为什么用余弦相似度而不是欧氏距离？】
    - 余弦相似度衡量方向，对向量长度不敏感
    - 人脸嵌入中，同一人不同照片的嵌入长度可能不同(光照影响)
    - 但方向应该一致，所以余弦相似度更适合
    """
    # L2归一化
    emb1_norm = nn.functional.normalize(emb1, p=2, dim=-1)
    emb2_norm = nn.functional.normalize(emb2, p=2, dim=-1)
    # 点积 = 余弦相似度(归一化后)
    return (emb1_norm * emb2_norm).sum(dim=-1)


def verify_faces(model, img1, img2, cfg):
    """
    人脸验证: 判断两张人脸是否为同一人。

    参数:
        img1, img2: 图像张量 (1, 1, 64, 64)
    返回:
        is_same: bool，是否为同一人
        similarity: float，余弦相似度
    """
    emb1 = model.get_embedding(img1.to(cfg.device))
    emb2 = model.get_embedding(img2.to(cfg.device))

    sim = cosine_similarity(emb1, emb2).item()
    is_same = sim > cfg.similarity_threshold

    return is_same, sim


def identify_face(model, query_img, gallery_embeddings, gallery_labels, cfg):
    """
    人脸辨识: 在人脸库中找到最相似的人。

    参数:
        query_img: 查询图像 (1, 1, 64, 64)
        gallery_embeddings: 人脸库嵌入矩阵 (N, embedding_dim)
        gallery_labels: 人脸库标签 (N,)
    返回:
        top_k_labels: 最相似的K个身份ID
        top_k_sims: 对应的相似度
    """
    query_emb = model.get_embedding(query_img.to(cfg.device))

    # 计算与所有库中人脸的相似度
    similarities = cosine_similarity(query_emb, gallery_embeddings.to(cfg.device))

    # 取Top-K
    top_k = min(cfg.top_k, len(gallery_labels))
    top_vals, top_indices = similarities.topk(top_k)

    return gallery_labels[top_indices.cpu().numpy()], top_vals.cpu().numpy()


def build_gallery(model, dataset, cfg):
    """
    构建人脸库: 提取所有注册人脸的嵌入向量。

    【人脸库的概念】
    人脸库 = {(人名, 嵌入向量), ...}
    - 注册: 每个人提供1~N张照片，提取嵌入向量存入库中
    - 识别: 查询照片与库中所有嵌入比较，返回最相似的

    【批量处理优化】
    原始方法: 逐张图片调用model，每次只处理1张 → GPU利用率低
    批量方法: 将图片打包成batch，一次处理多张 → GPU利用率高，速度快5-10倍
    """
    model.eval()
    # 批量处理: 用DataLoader自动打包batch
    gallery_loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=cfg.device.type == "cuda",
    )
    all_embeddings = []
    all_labels = []
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    with torch.no_grad():
        for images, labels in gallery_loader:
            images = images.to(cfg.device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                embs = model.get_embedding(images)
            all_embeddings.append(embs.cpu())
            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)

    gallery_embeddings = torch.cat(all_embeddings, dim=0)
    gallery_labels = np.array(all_labels)

    print(f"人脸库: {len(gallery_labels)}张注册人脸, {len(np.unique(gallery_labels))}个身份")
    return gallery_embeddings, gallery_labels


# ============================================================
# Step 7: 可视化函数
# ============================================================
def plot_training_curves(history, cfg):
    """绘制训练曲线。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("训练/验证损失")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("训练/验证准确率")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "face_training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()


def plot_face_samples(dataset, cfg, num_per_person=5, num_persons=4):
    """展示数据集中的样本人脸。"""
    fig, axes = plt.subplots(num_persons, num_per_person, figsize=(3 * num_per_person, 3 * num_persons))

    shown = 0
    current_person = -1
    count = 0

    for i in range(len(dataset)):
        img, label = dataset[i]
        if label != current_person:
            current_person = label
            count = 0

        if count < num_per_person and shown < num_persons:
            axes[shown, count].imshow(img.squeeze().numpy(), cmap="gray")
            axes[shown, count].set_title(f"人{label} - 照片{count+1}", fontsize=9)
            axes[shown, count].axis("off")
            count += 1
            if count == num_per_person:
                shown += 1

        if shown >= num_persons:
            break

    plt.suptitle("合成人脸数据样本", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "face_samples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 人脸样本已保存: {save_path}")
    plt.close()


def plot_embedding_tsne(embeddings, labels, cfg):
    """
    用t-SNE将嵌入向量降维到2D，可视化嵌入空间。

    【t-SNE可视化原理】
    - 将128维嵌入向量降到2维，保留局部相似性结构
    - 同一人的嵌入应该聚集在一起(一团点)
    - 不同人的嵌入应该分开(不同团)
    - 如果所有人混在一起，说明嵌入质量差
    """
    print("计算t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=cfg.random_state, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings.numpy())

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="tab20", s=30, alpha=0.7)
    ax.set_title("人脸嵌入空间 (t-SNE可视化)", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 维度1")
    ax.set_ylabel("t-SNE 维度2")
    plt.colorbar(scatter, label="身份ID")

    save_path = os.path.join(cfg.save_dir, "embedding_tsne.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 嵌入可视化已保存: {save_path}")
    plt.close()


def plot_verification_results(model, test_dataset, cfg, num_pairs=8):
    """
    可视化人脸验证结果: 展示人脸对+相似度+判断结果。
    """
    n = len(test_dataset)
    fig, axes = plt.subplots(num_pairs, 2, figsize=(8, 3 * num_pairs))

    for i in range(num_pairs):
        # 随机选两张图
        idx1 = np.random.randint(n)
        idx2 = np.random.randint(n)

        img1, label1 = test_dataset[idx1]
        img2, label2 = test_dataset[idx2]

        # 验证
        is_same, sim = verify_faces(model, img1.unsqueeze(0), img2.unsqueeze(0), cfg)
        truth = "同一人" if label1 == label2 else "不同人"
        correct = (is_same == (label1 == label2))

        color = "green" if correct else "red"
        result = "✓ 正确" if correct else "X 错误"

        axes[i, 0].imshow(img1.squeeze().numpy(), cmap="gray")
        axes[i, 0].set_title(f"人{label1}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img2.squeeze().numpy(), cmap="gray")
        axes[i, 1].set_title(f"人{label2}", fontsize=9)
        axes[i, 1].axis("off")

        # 在右侧添加结果文字
        fig.text(0.92, 1 - (2 * i + 1) / (2 * num_pairs),
                 f"真实: {truth}\n相似度: {sim:.3f}\n判断: {'同一人' if is_same else '不同人'}\n{result}",
                 fontsize=8, color=color, ha="left", va="center",
                 transform=fig.transFigure)

    plt.suptitle("人脸验证结果", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    save_path = os.path.join(cfg.save_dir, "face_verification.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 验证结果已保存: {save_path}")
    plt.close()


def plot_identification_results(model, test_dataset, gallery_embeddings, gallery_labels, cfg):
    """
    可视化人脸辨识结果: 查询图 + Top-K最相似的人。
    """
    model.eval()
    fig, axes = plt.subplots(3, cfg.top_k + 1, figsize=(3 * (cfg.top_k + 1), 9))

    for row in range(3):
        # 随机选一张查询图
        idx = np.random.randint(len(test_dataset))
        query_img, query_label = test_dataset[idx]
        query_tensor = query_img.unsqueeze(0)

        # 辨识
        top_labels, top_sims = identify_face(
            model, query_tensor, gallery_embeddings, gallery_labels, cfg,
        )

        # 显示查询图
        axes[row, 0].imshow(query_img.squeeze().numpy(), cmap="gray")
        axes[row, 0].set_title(f"查询: 人{query_label}", fontsize=10, fontweight="bold")
        axes[row, 0].axis("off")

        # 显示Top-K结果
        for k in range(cfg.top_k):
            if k < len(top_labels):
                # 从库中找到该人的第一张图
                person_idx = np.where(gallery_labels == top_labels[k])[0][0]
                match_img, _ = test_dataset[person_idx % len(test_dataset)]
                match_label = top_labels[k]
                match_sim = top_sims[k]

                axes[row, k + 1].imshow(match_img.squeeze().numpy(), cmap="gray")
                correct = "✓" if match_label == query_label else "X"
                color = "green" if match_label == query_label else "red"
                axes[row, k + 1].set_title(
                    f"#{k+1}: 人{match_label}\n相似度: {match_sim:.3f} {correct}",
                    fontsize=9, color=color,
                )
                axes[row, k + 1].axis("off")

    plt.suptitle("人脸辨识结果 (Top-K搜索)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "face_identification.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 辨识结果已保存: {save_path}")
    plt.close()


# ============================================================
# Step 8: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("CNN 人脸识别 - Face Embedding Network")
    print("=" * 60)

    cfg = CONFIG()
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载数据
    train_loader, test_loader, test_dataset = load_data(cfg)

    # 创建模型
    model = FaceEmbeddingNet(cfg).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: FaceEmbeddingNet")
    print(f"总参数量: {total_params:,}")
    print(f"嵌入维度: {cfg.embedding_dim}")

    # 可视化数据样本
    plot_face_samples(test_dataset, cfg)

    # 训练
    model, history = train(model, train_loader, test_loader, cfg)

    # 最终评估
    print(f"\n{'='*60}")
    print("测试集评估...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_pred, y_true, test_embeddings = evaluate(
        model, test_loader, criterion, cfg,
    )
    print(f"分类准确率: {test_acc:.4f}")

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "face_embedding_net.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
    }, model_path)
    print(f"✓ 模型已保存: {model_path}")

    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, cfg)
    plot_embedding_tsne(test_embeddings, np.array(y_true), cfg)
    plot_verification_results(model, test_dataset, cfg)

    # 构建人脸库并演示辨识
    print("\n构建人脸库...")
    gallery_embeddings, gallery_labels = build_gallery(model, test_dataset, cfg)
    plot_identification_results(model, test_dataset, gallery_embeddings, gallery_labels, cfg)

    # 验证统计
    print("\n人脸验证统计:")
    n_verify = 100
    tp = fp = tn = fn = 0
    for _ in range(n_verify):
        idx1, idx2 = np.random.randint(len(test_dataset), size=2)
        img1, label1 = test_dataset[idx1]
        img2, label2 = test_dataset[idx2]
        is_same, _ = verify_faces(model, img1.unsqueeze(0), img2.unsqueeze(0), cfg)
        actually_same = (label1 == label2)

        if actually_same and is_same:
            tp += 1
        elif not actually_same and is_same:
            fp += 1
        elif not actually_same and not is_same:
            tn += 1
        else:
            fn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / n_verify
    print(f"  准确率: {accuracy:.2%} | 精确率: {precision:.2%} | 召回率: {recall:.2%}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    print(f"\n{'='*60}")
    print("人脸识别完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
