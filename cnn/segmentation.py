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

【本数据集: Pascal VOC 2012】
- 20个语义类别 + 1个背景类
- 约1,464张训练图像，1,449张验证图像
- 图像尺寸不固定(约500×300)
- 分割标注: 每个像素的类别索引(0=背景, 1-20=各类)

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

# 设置中文字体
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# VOC 20类名称 + 背景
VOC_CLASSES = [
    "背景", "飞机", "自行车", "鸟", "船", "瓶子", "公交车", "汽车",
    "猫", "椅子", "牛", "餐桌", "狗", "马", "摩托车", "人",
    "盆栽", "羊", "沙发", "火车", "电视",
]

# 可视化用的颜色表(VOC标准)
VOC_COLORS = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)


# ============================================================
# Step 2: 配置超参数
# ============================================================
class CONFIG:
    """超参数配置中心 —— 图像分割任务的所有可调参数。"""

    # --- 数据相关 ---
    # data_dir: VOC数据集存放目录
    data_dir = "data"

    # num_classes=21: VOC 20类 + 1背景
    #   微调自己的数据时，改为你的类别数+1
    num_classes = 21

    # class_names: 类别名称(索引0=背景)
    class_names = VOC_CLASSES

    # image_size=520: 训练时图像的裁剪尺寸
    #   【为什么是520？】
    #   VOC图像约500×300，需要统一尺寸输入网络
    #   520是8的倍数(下采样倍率=8，520/8=65，整除)
    #   为什么不裁剪到32×32？分割需要高分辨率保持空间细节
    #   520×520是在精度和显存之间的平衡
    image_size = 520

    # --- 模型相关 ---
    # model_name: 分割模型名称
    #   "deeplabv3": DeepLabV3 + ResNet50 backbone
    #   【为什么选DeepLabV3？】
    #   - 精度高(PASCAL VOC mIoU约89%)
    #   - 架构清晰，适合学习分割原理
    #   - 空洞卷积是分割的经典技巧
    #   其他选择: FCN(更简单), UNet(医学影像常用), DeepLabV3+(更强)
    model_name = "deeplabv3"

    # pretrained=True: 使用COCO预训练权重
    #   【分割预训练的特殊性】
    #   COCO预训练的分割模型已学会识别常见物体的轮廓
    #   微调时只需要学习新类别的边界
    pretrained = True

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
# Step 3: 数据加载和预处理
# ============================================================
class SegmentationTransform:
    """
    分割数据变换：同时变换图像和标注。

    【分割数据增强的特殊之处】
    - 分类: 只需变换图像
    - 分割: 图像和标注必须同步变换！
      如果图像水平翻转，标注也必须翻转，否则像素对不上
    """

    def __init__(self, image_size, is_train=True):
        self.image_size = image_size
        self.is_train = is_train

    def __call__(self, image, target):
        """
        参数:
            image: PIL Image
            target: PIL Image (分割标注，每个像素是类别索引)
        """
        # 随机裁剪(训练时)
        if self.is_train:
            # 随机裁剪到固定尺寸
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(self.image_size, self.image_size),
            )
            image = F.crop(image, i, j, h, w)
            target = F.crop(target, i, j, h, w)

            # 随机水平翻转(图像和标注同步)
            if torch.rand(1) < 0.5:
                image = F.hflip(image)
                target = F.hflip(target)
        else:
            # 验证/测试: 只做Resize
            image = F.resize(image, self.image_size)
            target = F.resize(target, self.image_size, interpolation=Image.NEAREST)
            # NEAREST: 分割标注必须用最近邻插值，不能用双线性
            # 因为标注是类别索引，双线性插值会产生非法的中间值

        # 图像转Tensor并标准化
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 标注转Tensor (保持为整数索引)
        target = torch.as_tensor(np.array(target), dtype=torch.long)

        return image, target


def get_dataloaders(cfg):
    """
    加载VOC分割数据集。

    【VOCSegmentation数据集说明】
    - 10,582张训练图像(含增广), 1,449张验证图像
    - 标注为PNG图像，每个像素值=类别索引
    - 255=忽略边界(不参与训练和评估)
    """
    from torchvision.transforms.functional import crop as F_crop
    import torchvision.transforms.functional as F

    train_transform = SegmentationTransform(cfg.image_size, is_train=True)
    val_transform = SegmentationTransform(cfg.image_size, is_train=False)

    try:
        train_dataset = datasets.VOCSegmentation(
            root=cfg.data_dir, year="2012", image_set="train",
            download=True, transforms=train_transform,
        )
        val_dataset = datasets.VOCSegmentation(
            root=cfg.data_dir, year="2012", image_set="val",
            download=True, transforms=val_transform,
        )
    except Exception as e:
        print(f"VOC数据集加载失败: {e}")
        print("请手动下载VOC2012数据集到 data/VOCdevkit/ 目录")
        raise

    # DataLoader
    # collate_fn=custom_collate: 处理不同尺寸的图像
    pin_mem = cfg.device.type == "cuda"

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=2, pin_memory=pin_mem,
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
    if cfg.pretrained and cfg.num_classes == 21:
        # VOC预训练
        model = deeplabv3_resnet50(pretrained=True)
    else:
        # 微调: 替换分类头
        model = deeplabv3_resnet50(pretrained=False, num_classes=cfg.num_classes)

    # 如果微调自己的数据(类别数!=21)，需要替换分类头
    if cfg.num_classes != 21:
        model.classifier = DeepLabHead(2048, cfg.num_classes)

    return model


# ============================================================
# Step 5: 训练函数
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, cfg):
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

    for images, targets in loader:
        images = images.to(cfg.device)
        # targets: (batch, H, W)，每个值是类别索引
        targets = targets.to(cfg.device)

        # 前向传播
        # 分割模型返回字典 {"out": main_output, "aux": auxiliary_output}
        outputs = model(images)["out"]  # (batch, num_classes, H, W)

        # 计算损失
        # outputs: (batch, num_classes, H, W) → 每个像素在各类的得分
        # targets: (batch, H, W) → 每个像素的真实类别
        # ignore_index=255: 边界像素不参与计算
        loss = criterion(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
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
    """
    model.eval()
    # 混淆矩阵: (num_classes, num_classes)
    # conf_mat[i][j] = 真实为i但预测为j的像素数
    conf_mat = torch.zeros(cfg.num_classes, cfg.num_classes)

    for images, targets in loader:
        images = images.to(cfg.device)
        outputs = model(images)["out"]  # (batch, C, H, W)

        # 取每个像素的最大类别
        preds = outputs.argmax(dim=1).cpu()  # (batch, H, W)
        targets_np = targets.numpy()
        preds_np = preds.numpy()

        # 更新混淆矩阵
        valid = targets_np != cfg.ignore_index
        for i in range(targets_np.shape[0]):  # batch
            t = targets_np[i][valid[i]]
            p = preds_np[i][valid[i]]
            for ti, pi in zip(t.flatten(), p.flatten()):
                if ti < cfg.num_classes and pi < cfg.num_classes:
                    conf_mat[ti, pi] += 1

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
    correct = conf_mat.diag().sum().item()
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

    history = {"train_loss": [], "val_miou": [], "val_pixel_acc": []}

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}")

    for epoch in range(1, cfg.epochs + 1):
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg)
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
        gt_color = VOC_COLORS[gt % len(VOC_COLORS)]

        # 预测 → 彩色图
        pred = preds[i]
        pred_color = VOC_COLORS[pred % len(VOC_COLORS)]

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
    pred_color = VOC_COLORS[pred_mask % len(VOC_COLORS)]

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
    print("\n加载VOC 2012分割数据集...")
    try:
        train_loader, val_loader = get_dataloaders(cfg)
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("尝试使用预训练模型进行推理演示...")
        train_loader, val_loader = None, None

    # 创建模型
    print("\n加载DeepLabV3模型(ResNet50 backbone)...")
    model = get_segmentation_model(cfg).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    if train_loader is not None:
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
    else:
        # 仅推理演示
        print("\n推理演示(无训练数据)...")

        # 使用VOC预训练模型直接推理
        demo_img = Image.new("RGB", (520, 520), color=(100, 150, 200))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(demo_img)
        draw.rectangle([100, 100, 400, 400], fill=(0, 128, 0))  # 模拟物体
        draw.ellipse([200, 200, 350, 350], fill=(128, 0, 0))

        pred_mask, pred_color = predict(model, demo_img, cfg)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(demo_img)
        ax1.set_title("输入图像")
        ax1.axis("off")
        ax2.imshow(pred_color)
        ax2.set_title("分割预测")
        ax2.axis("off")

        save_path = os.path.join(cfg.save_dir, "segmentation_demo.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ 推理演示已保存: {save_path}")
        plt.close()

    print(f"\n{'='*60}")
    print("完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
