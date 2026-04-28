"""
=============================================================================
CNN 图像分割任务模板 (Semantic Segmentation with DeepLabV3)
=============================================================================

【原理】
图像分割是像素级的分类任务：对图像中**每个像素**都预测一个类别标签。
- 分类: 1张图 → 1个标签 (这张图里有猫)
- 检测: 1张图 → N个(框, 标签) (猫在(x1,y1,x2,y2)位置)
- 分割: 1张图 → H×W个标签 (每个像素是猫/狗/背景)

语义分割(Semantic Segmentation): 区分不同类别，同类不区分个体
  例: 图中有3只猫，分割结果只有"猫"区域，不区分哪只是哪只

实例分割(Instance Segmentation): 区分不同类别的不同个体
  例: 3只猫分别标记为猫1、猫2、猫3

本模板实现语义分割。

【DeepLabV3架构】
核心思想: 空洞卷积(Dilated/Atrous Convolution)
  普通卷积: 3×3核，感受野3×3
  空洞卷积(rate=2): 3×3核，感受野5×5(但参数量不变!)
  空洞卷积(rate=4): 3×3核，感受野9×9

为什么用空洞卷积？
  池化/步进卷积会降低分辨率(32×32→16×16)，丢失空间细节
  空洞卷积保持分辨率不变，同时扩大感受野
  → 不降低分辨率就能"看到"更大范围，实现精确的像素级预测

DeepLabV3 = Backbone(ResNet) + ASPP(多尺度空洞卷积) + 解码器

【应用场景】
- 自动驾驶(道路/车道/行人分割)
- 医学影像(器官/肿瘤分割)
- 遥感图像(建筑/农田/水体分割)
- 人像分割(背景替换/美颜)
- 工业检测(缺陷区域分割)

【本数据集: 合成分割数据集】
- 5个类别: 背景、圆形、矩形、三角形、条纹
- 即时生成，无需下载(替代VOC 2012，后者约2GB下载缓慢)
- 图像尺寸: 128×128 RGB
- 分割标注: 每个像素的类别索引(0=背景, 1-4=各类形状)
- 替代原因: VOC 2012数据集约2GB，下载经常超时；合成数据保证代码可运行

【使用方法】
1. 直接运行: python cnn/segmentation.py
2. 数据集自动下载到 data/ 目录
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
from torch.utils.data import DataLoader

import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import datasets, transforms
import torchvision.transforms.functional as F

# 设置中文字体
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# Step 2: 配置超参数
# ============================================================
class CONFIG:
    """超参数配置中心 —— 图像分割任务的所有可调参数。"""

    # --- 数据相关 ---
    # data_dir: 数据集存放目录(合成数据不需要，但保留接口)
    data_dir = "data"

    # num_classes=5: 背景 + 4种形状
    #   微调自己的数据时，改为你的类别数
    num_classes = 5

    # class_names: 类别名称(索引0=背景)
    class_names = ["背景", "圆形", "矩形", "三角形", "条纹"]

    # image_size=128: 训练时图像尺寸
    #   合成数据用128×128，比VOC(520)小但足够学习分割原理
    #   128是8的倍数(下采样倍率=8，128/8=16，整除)
    image_size = 128

    # --- 模型相关 ---
    # model_name: 分割模型名称
    #   "deeplabv3": DeepLabV3 + ResNet50 backbone
    #   【为什么选DeepLabV3？】
    #   - 精度高(PASCAL VOC mIoU约89%)
    #   - 架构清晰，适合学习分割原理
    #   - 空洞卷积是分割的经典技巧
    #   其他选择: FCN(更简单), UNet(医学影像常用), DeepLabV3+(更强)
    model_name = "deeplabv3"

    # use_pretrained=True: 使用COCO预训练权重初始化backbone
    #   预训练的ResNet50已学会提取边缘/纹理等底层特征
    #   即使数据集不同(形状vs COCO)，底层特征仍可复用
    use_pretrained = True

    # --- 训练相关 ---
    # batch_size=4: 每批图像数
    #   为什么比检测(2)多？分割图像裁剪到520×520，比检测(800×1333)小
    #   如果OOM，降到2
    batch_size = 4

    # learning_rate=1e-3: 初始学习率
    #   分割常用1e-3或7e-4，比检测的5e-3小
    learning_rate = 1e-3

    # epochs=30: 最大训练轮数
    #   VOC分割约20-30轮收敛
    epochs = 30

    # weight_decay=1e-4: L2正则化
    weight_decay = 1e-4

    # --- 早停策略 ---
    early_stop_patience = 10

    # --- 学习率调度器 ---
    # scheduler_type: "poly"(多项式衰减) 或 "cosine"
    #   【为什么分割常用poly？】
    #   poly调度: LR = base_LR × (1 - epoch/max_epoch)^power
    #   训练前期快速下降，后期非常平缓，适合精细调优
    #   power=0.9是DeepLab论文推荐值
    scheduler_type = "poly"

    # poly_power=0.9: 多项式衰减的指数
    poly_power = 0.9

    # --- 梯度裁剪 ---
    max_grad_norm = 5.0

    # --- 混合精度训练(AMP) ---
    # use_amp=True: 启用自动混合精度，训练速度提升1.5-2倍
    #   仅CUDA(GPU)有效，CPU自动降级为普通训练
    #   原理: 将部分float32运算自动转为float16，加快速度、减少显存
    use_amp = True

    # --- 数据加载优化 ---
    # num_workers: 多进程并行加载数据，加速训练
    #   0=主进程加载(慢)，2-4=推荐值
    num_workers = min(4, os.cpu_count() or 1)

    # --- 评估相关 ---
    # ignore_index=255: 忽略的标签值
    #   VOC标注中，边界像素标注为255(忽略区域)
    #   计算损失和指标时跳过这些像素
    ignore_index = 255

    # --- 保存相关 ---
    save_dir = "cnn/output/segmentation"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")



# ============================================================
# Step 3: 合成分割数据集
# ============================================================
class SyntheticSegmentationDataset(torch.utils.data.Dataset):
    """
    合成分割数据集: 生成含有几何形状的图像和对应的像素级标注。

    【为什么用合成数据替代VOC 2012？】
    VOC 2012数据集约2GB，下载经常超时或卡住。
    合成数据即时生成，无需下载，且保留分割任务的核心要素：
      1. 多类别(不同形状=不同类别)
      2. 像素级标注(每个像素一个类别标签)
      3. 需要精确的边界预测(形状边缘)

    【合成方法】
    - 在128×128画布上随机放置3-7个几何形状
    - 形状类型: 圆形(1)、矩形(2)、三角形(3)、条纹(4)
    - 每个形状有随机位置、大小、颜色
    - 标注掩码: 每个像素的类别索引(0=背景, 1-4=形状)
    - 后放置的形状会覆盖先放置的(模拟遮挡关系)
    """

    # 分割可视化颜色表(5类: 背景+4种形状)
    COLORS = np.array([
        [0, 0, 0],       # 0: 背景 - 黑色
        [200, 0, 0],     # 1: 圆形 - 红色
        [0, 200, 0],     # 2: 矩形 - 绿色
        [0, 0, 200],     # 3: 三角形 - 蓝色
        [200, 200, 0],   # 4: 条纹 - 黄色
    ], dtype=np.uint8)

    def __init__(self, num_samples=500, image_size=128, num_classes=5,
                 is_train=True, random_state=42):
        """
        参数:
            num_samples: 样本数量
            image_size: 图像尺寸(正方形)
            num_classes: 类别数(含背景)
            is_train: 是否为训练集(训练集添加噪声增强)
            random_state: 随机种子
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.is_train = is_train

        # 预生成所有样本的随机种子(确保可复现)
        rng = np.random.RandomState(random_state)
        self.seeds = rng.randint(0, 100000, num_samples)

    def __len__(self):
        return self.num_samples

    def _generate_sample(self, seed):
        """根据种子生成一个样本(图像+掩码)"""
        rng = np.random.RandomState(seed)
        size = self.image_size

        # 空白图像(灰色背景)和掩码
        image = np.ones((size, size, 3), dtype=np.float32) * 0.6
        mask = np.zeros((size, size), dtype=np.int64)

        # 随机放置3-7个形状
        # 为什么3-7？太少(1-2)场景太简单，太多(10+)形状重叠严重
        num_shapes = rng.randint(3, 8)
        for _ in range(num_shapes):
            # 类别1-4(跳过0=背景)
            class_id = rng.randint(1, self.num_classes)
            # 随机颜色(形状的颜色不影响分类，但增加视觉多样性)
            color = rng.rand(3) * 0.6 + 0.2

            if class_id == 1:  # 圆形
                cx = rng.randint(15, size - 15)
                cy = rng.randint(15, size - 15)
                r = rng.randint(8, max(9, size // 5))
                y, x = np.ogrid[:size, :size]
                circle = (x - cx)**2 + (y - cy)**2 <= r**2
                image[circle] = color
                mask[circle] = class_id

            elif class_id == 2:  # 矩形
                x1 = rng.randint(5, size - 30)
                y1 = rng.randint(5, size - 30)
                w = rng.randint(12, max(13, size // 4))
                h = rng.randint(12, max(13, size // 4))
                x2 = min(x1 + w, size)
                y2 = min(y1 + h, size)
                image[y1:y2, x1:x2] = color
                mask[y1:y2, x1:x2] = class_id

            elif class_id == 3:  # 三角形
                # 用三个顶点定义三角形
                pts = np.array([
                    [rng.randint(10, size - 10), rng.randint(10, size - 10)],
                    [rng.randint(10, size - 10), rng.randint(10, size - 10)],
                    [rng.randint(10, size - 10), rng.randint(10, size - 10)],
                ])
                # 使用向量化的点-in-三角形判断
                y, x = np.mgrid[:size, :size]
                coords = np.column_stack((x.ravel(), y.ravel()))
                # 三次叉积法判断点是否在三角形内
                v0, v1, v2 = pts[0], pts[1], pts[2]
                def sign(p1, p2, p3):
                    return (p1[:, 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[:, 1] - p3[1])
                d1 = sign(coords, v0, v1)
                d2 = sign(coords, v1, v2)
                d3 = sign(coords, v2, v0)
                has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
                has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
                triangle = (~(has_neg & has_pos)).reshape(size, size)
                image[triangle] = color
                mask[triangle] = class_id

            elif class_id == 4:  # 条纹(水平或垂直交替色带)
                is_horizontal = rng.rand() > 0.5
                stripe_width = rng.randint(4, max(5, size // 8))
                if is_horizontal:
                    for sy in range(0, size, stripe_width * 2):
                        y_start = sy
                        y_end = min(sy + stripe_width, size)
                        image[y_start:y_end, :] = color
                        mask[y_start:y_end, :] = class_id
                else:
                    for sx in range(0, size, stripe_width * 2):
                        x_start = sx
                        x_end = min(sx + stripe_width, size)
                        image[:, x_start:x_end] = color
                        mask[:, x_start:x_end] = class_id

        # 训练集添加轻微噪声(增强鲁棒性)
        if self.is_train:
            image += rng.randn(size, size, 3) * 0.02
            image = np.clip(image, 0, 1)

        return image, mask

    def __getitem__(self, idx):
        image, mask = self._generate_sample(self.seeds[idx])

        # 转为Tensor
        # image: (H, W, 3) → (3, H, W)
        image = torch.tensor(image).permute(2, 0, 1).float()
        # 标准化(ImageNet标准，与预训练backbone匹配)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


def get_dataloaders(cfg):
    """
    创建合成分割数据集的DataLoader。

    【数据量说明】
    - 训练集: 500张(即时生成)
    - 验证集: 100张(即时生成)
    - 无需下载，首次运行即用
    """
    train_dataset = SyntheticSegmentationDataset(
        num_samples=500, image_size=cfg.image_size,
        num_classes=cfg.num_classes, is_train=True, random_state=42,
    )
    val_dataset = SyntheticSegmentationDataset(
        num_samples=100, image_size=cfg.image_size,
        num_classes=cfg.num_classes, is_train=False, random_state=42,
    )

    # DataLoader
    pin_mem = cfg.device.type == "cuda"
    pw = cfg.num_workers > 0  # persistent_workers: 保持子进程活跃

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=pw,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=pw,
    )

    print(f"训练集: {len(train_dataset)}张 | 验证集: {len(val_dataset)}张")

    return train_loader, val_loader


# ============================================================
# Step 4: 模型构建
# ============================================================
def get_segmentation_model(cfg):
    """
    创建DeepLabV3分割模型。

    【DeepLabV3结构】
    输入图像 → ResNet50(骨干) → 特征图(1/8分辨率)
              → ASPP(多尺度空洞卷积) → 融合特征
              → 1×1卷积 → 上采样 → 像素级预测

    ASPP (Atrous Spatial Pyramid Pooling):
      并行5个分支:
        1. 1×1卷积(捕获全局上下文)
        2. 3×3空洞卷积 rate=6 (中距离)
        3. 3×3空洞卷积 rate=12 (远距离)
        4. 3×3空洞卷积 rate=18 (更远距离)
        5. 全局平均池化(图像级特征)
      5个分支concat → 1×1卷积 → 输出
      → 多尺度信息融合，同时检测大小不同的物体
    """
    # 使用COCO预训练的backbone，替换分类头为5类
    # 为什么用预训练？即使数据集不同(形状vs自然物体)，
    # ResNet50学到的边缘/纹理底层特征仍可复用，加速收敛
    weights = "DEFAULT" if cfg.use_pretrained else None
    model = deeplabv3_resnet50(weights=weights)

    # 替换分类头(5类: 背景+4种形状)
    # 预训练模型是21类(VOC)，需要替换为我们自己的类别数
    model.classifier = DeepLabHead(2048, cfg.num_classes)

    return model


# ============================================================
# Step 5: 训练函数
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, cfg, scaler=None):
    """
    训练一个epoch。

    【分割训练的特殊之处】
    1. 模型输出是字典，key="out"是主输出 (batch, num_classes, H, W)
    2. 损失是像素级的：每个像素一个CrossEntropy，取平均
    3. 忽略边界像素(ignore_index=255)
    """
    model.train()
    total_loss = 0
    total_pixels = 0
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for images, targets in loader:
        images = images.to(cfg.device)
        # targets: (batch, H, W)，每个值是类别索引
        targets = targets.to(cfg.device)

        # 前向传播(混合精度)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)["out"]  # (batch, num_classes, H, W)
            loss = criterion(outputs, targets)

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

        # 统计(只计非忽略像素)
        valid_mask = targets != cfg.ignore_index
        n_valid = valid_mask.sum().item()
        total_loss += loss.item() * n_valid
        total_pixels += n_valid

    return total_loss / max(total_pixels, 1)


@torch.no_grad()
def evaluate(model, loader, cfg):
    """
    评估分割模型: 计算mIoU(平均交并比)。

    【mIoU — 分割的核心指标】
    IoU(交并比) = |预测∩真实| / |预测∪真实|
    mIoU = 所有类别IoU的平均值

    为什么不用像素准确率？
    - 像素准确率会被大类别主导(背景占60%+，预测全背景也有60%)
    - mIoU对每个类别单独评估，小类别的性能也能体现

    【向量化计算 — 比逐像素循环快100倍】
    原始方法: for循环遍历每个像素，逐个更新混淆矩阵 → 极慢
    向量化方法: 用numpy批量操作一次更新所有像素 → 极快
    原理: np.add.at(conf_mat, (真实类别数组, 预测类别数组), 1)
      等价于: 对每个(真实, 预测)对，在conf_mat对应位置+1
      但用C层循环实现，比Python for循环快两个数量级
    """
    model.eval()
    # 用numpy数组存储混淆矩阵(比torch在CPU上更快)
    conf_mat = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
    use_amp = cfg.use_amp and cfg.device.type == "cuda"

    for images, targets in loader:
        images = images.to(cfg.device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)["out"]  # (batch, C, H, W)

        # 取每个像素的最大类别
        preds = outputs.argmax(dim=1).cpu().numpy()  # (batch, H, W)
        targets_np = targets.numpy()

        # 向量化更新混淆矩阵(替代原来的三层嵌套循环)
        # 1. 找出有效像素(非ignore_index)
        valid = targets_np != cfg.ignore_index
        valid_targets = targets_np[valid]
        valid_preds = preds[valid]

        # 2. 过滤超出类别范围的像素
        mask = (valid_targets < cfg.num_classes) & (valid_preds < cfg.num_classes)
        valid_targets = valid_targets[mask]
        valid_preds = valid_preds[mask]

        # 3. 批量更新混淆矩阵
        # np.add.at: 对每个(t, p)对，在conf_mat[t, p]位置+1
        # 这是向量化操作，比Python for循环快100倍以上
        np.add.at(conf_mat, (valid_targets, valid_preds), 1)

    # 计算mIoU
    iou_per_class = []
    for c in range(cfg.num_classes):
        tp = conf_mat[c, c].item()          # 真实为c且预测为c
        fp = conf_mat[:, c].sum().item() - tp  # 预测为c但真实不为c
        fn = conf_mat[c, :].sum().item() - tp  # 真实为c但预测不为c
        if tp + fp + fn > 0:
            iou = tp / (tp + fp + fn)
            iou_per_class.append(iou)

    miou = np.mean(iou_per_class) if iou_per_class else 0

    # 像素准确率
    total = conf_mat.sum().item()
    correct = conf_mat.diagonal().sum().item()
    pixel_acc = correct / max(total, 1)

    return miou, pixel_acc, iou_per_class


def train(model, train_loader, val_loader, cfg):
    """完整训练流程。"""
    # 损失函数
    # CrossEntropyLoss: 像素级分类，ignore_index忽略边界像素
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)

    # 优化器
    optimizer = optim.SGD(
        model.parameters(), lr=cfg.learning_rate,
        momentum=0.9, weight_decay=cfg.weight_decay,
    )
    # 【为什么分割用SGD而不是Adam？】
    # DeepLab论文推荐SGD+momentum
    # SGD在分割任务上收敛更稳定，泛化更好
    # Adam在分割上容易过拟合(训练loss低但mIoU不高)

    # 学习率调度器
    if cfg.scheduler_type == "poly":
        # 多项式衰减: LR = base_LR × (1 - epoch/max_epoch)^power
        # 自定义实现
        def poly_scheduler(epoch):
            return (1 - epoch / cfg.epochs) ** cfg.poly_power
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, poly_scheduler)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # 早停
    best_miou = 0
    patience_counter = 0
    best_model_state = None

    # 混合精度GradScaler
    use_amp = cfg.use_amp and cfg.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "val_miou": [], "val_pixel_acc": []}

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")
    print(f"设备: {cfg.device} | 优化器: SGD(lr={cfg.learning_rate}) | AMP: {use_amp}")

    for epoch in range(1, cfg.epochs + 1):
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg, scaler)
        # 验证
        miou, pixel_acc, _ = evaluate(model, val_loader, cfg)

        history["train_loss"].append(train_loss)
        history["val_miou"].append(miou)
        history["val_pixel_acc"].append(pixel_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{cfg.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"mIoU: {miou:.4f} | "
              f"PixelAcc: {pixel_acc:.4f} | "
              f"LR: {current_lr:.6f}")

        scheduler.step()

        # 早停(基于mIoU)
        if miou > best_miou:
            best_miou = miou
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ✓ 最优模型已更新 (mIoU: {miou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"\n⚠ 早停触发: mIoU连续{cfg.early_stop_patience}轮未改善")
                break

    # 恢复最优模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(cfg.device)
        print(f"\n✓ 已恢复最优模型 (mIoU: {best_miou:.4f})")

    return model, history


# ============================================================
# Step 6: 可视化函数
# ============================================================
def plot_training_curves(history, cfg):
    """绘制训练曲线。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("训练损失")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_miou"], "r-", label="mIoU", linewidth=2)
    ax2.plot(epochs, history["val_pixel_acc"], "g-", label="Pixel Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("验证指标")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "segmentation_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()


def plot_segmentation_result(model, val_loader, cfg, num_samples=4):
    """
    可视化分割结果: 原图 / 真实标注 / 预测结果 三列对比。
    """
    model.eval()
    images, targets = next(iter(val_loader))
    images = images[:num_samples].to(cfg.device)

    with torch.no_grad():
        outputs = model(images)["out"]
        preds = outputs.argmax(dim=1).cpu().numpy()

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        # 原图(反标准化)
        img = images[i].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img * std + mean).permute(1, 2, 0).numpy().clip(0, 1)

        # 真实标注 → 彩色图
        gt = targets[i].numpy()
        gt_color = SyntheticSegmentationDataset.COLORS[gt % len(SyntheticSegmentationDataset.COLORS)]

        # 预测 → 彩色图
        pred = preds[i]
        pred_color = SyntheticSegmentationDataset.COLORS[pred % len(SyntheticSegmentationDataset.COLORS)]

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("原图")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_color)
        axes[i, 1].set_title("真实标注")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_color)
        axes[i, 2].set_title("预测结果")
        axes[i, 2].axis("off")

    plt.suptitle("图像分割结果对比", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "segmentation_results.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 分割结果已保存: {save_path}")
    plt.close()


def plot_class_iou(iou_per_class, cfg):
    """绘制各类别IoU柱状图。"""
    valid_ious = [(cfg.class_names[i], iou) for i, iou in enumerate(iou_per_class)
                  if i > 0]  # 跳过背景
    names, ious = zip(*valid_ious) if valid_ious else ([], [])

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, ious, color="steelblue")
    ax.set_ylabel("IoU")
    ax.set_title("各类别IoU")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")

    # 在柱子上方标注数值
    for bar, iou in zip(bars, ious):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{iou:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "class_iou.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 类别IoU已保存: {save_path}")
    plt.close()


# ============================================================
# Step 7: 预测函数
# ============================================================
@torch.no_grad()
def predict(model, image, cfg):
    """
    对单张图像进行分割预测。

    参数:
        image: PIL Image 或 Tensor (C, H, W)
    返回:
        pred_mask: 预测的分割掩码 (H, W)，每个像素是类别索引
        pred_color: 彩色可视化 (H, W, 3)
    """
    model.eval()

    if isinstance(image, Image.Image):
        # 保存原始尺寸用于恢复
        orig_size = image.size  # (W, H)
        image_tensor = F.to_tensor(image).to(cfg.device)
        # 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(cfg.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(cfg.device)
        image_tensor = (image_tensor - mean) / std
    else:
        image_tensor = image.to(cfg.device)
        orig_size = None

    # 推理
    output = model(image_tensor.unsqueeze(0))["out"]  # (1, C, H, W)
    pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

    # 恢复原始尺寸
    if orig_size:
        pred_mask = np.array(Image.fromarray(pred_mask.astype(np.uint8)).resize(
            orig_size, Image.NEAREST,
        ))

    # 彩色可视化
    pred_color = SyntheticSegmentationDataset.COLORS[pred_mask % len(SyntheticSegmentationDataset.COLORS)]

    return pred_mask, pred_color


# ============================================================
# Step 8: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("CNN 图像分割 - DeepLabV3")
    print("=" * 60)

    cfg = CONFIG()
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载数据
    print("\n加载合成分割数据集...")
    train_loader, val_loader = get_dataloaders(cfg)

    # 创建模型
    print("\n加载DeepLabV3模型(ResNet50 backbone)...")
    model = get_segmentation_model(cfg).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 训练
    model, history = train(model, train_loader, val_loader, cfg)

    # 最终评估
    print(f"\n{'='*60}")
    print("最终评估...")
    miou, pixel_acc, iou_per_class = evaluate(model, val_loader, cfg)
    print(f"mIoU: {miou:.4f} | Pixel Accuracy: {pixel_acc:.4f}")

    # 各类别IoU
    print("\n各类别IoU:")
    for i, iou in enumerate(iou_per_class):
        name = cfg.class_names[i] if i < len(cfg.class_names) else f"类{i}"
        print(f"  {name}: {iou:.4f}")

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "deeplabv3_segmentation.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ 模型已保存: {model_path}")

    # 可视化
    plot_training_curves(history, cfg)
    plot_segmentation_result(model, val_loader, cfg)
    if iou_per_class:
        plot_class_iou(iou_per_class, cfg)

    print(f"\n{'='*60}")
    print("完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
