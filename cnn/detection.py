"""
=============================================================================
CNN 目标检测任务模板 (Object Detection with Faster R-CNN)
=============================================================================

【原理】
目标检测 = 图像分类 + 定位，不仅要识别图中有"什么"，还要标出在"哪里"。
与图像分类(整张图一个标签)不同，目标检测输出多个检测框+类别+置信度。

Faster R-CNN 是经典的两阶段检测器：
  第1阶段 - RPN(区域建议网络): 在图像上生成候选框，判断是否包含物体
  第2阶段 - ROI Head: 对每个候选框精确分类和微调边界框

核心组件:
  - Backbone(骨干网络): 提取图像特征(通常用ResNet+FPN)
  - RPN: 生成候选区域(proposal)
  - ROI Pooling: 将不同大小的候选区域统一到固定尺寸
  - Classifier: 对候选区域分类
  - BBox Regressor: 微调候选框的位置

【与图像分类的区别】
  分类: 1张图 → 1个类别
  检测: 1张图 → N个(边界框, 类别, 置信度)
  分类损失: CrossEntropyLoss
  检测损失: RPN分类损失 + RPN回归损失 + ROI分类损失 + ROI回归损失

【应用场景】
- 自动驾驶(车辆/行人/信号灯检测)
- 安防监控(人员/异常检测)
- 工业质检(缺陷/零件检测)
- 医学影像(病灶定位)
- 零售(商品识别与计数)

【本模板特点】
- 使用torchvision预训练的Faster R-CNN，开箱即用
- 支持推理(直接检测)和微调(自定义数据集训练)
- 内置COCO 80类的预训练权重，可检测常见物体
- 提供自定义数据集模板，方便适配自己的数据

【使用方法】
1. 推理模式: 直接运行，对示例图像进行检测
   python cnn/detection.py
2. 微调模式: 准备标注数据，修改CONFIG，运行微调
=============================================================================
"""

# ============================================================
# Step 1: 导入必要的库
# ============================================================
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# 设置中文字体
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# COCO 80类名称(预训练模型使用)
COCO_CLASSES = [
    "人", "自行车", "汽车", "摩托车", "飞机", "公交车", "火车", "卡车", "船",
    "交通灯", "消防栓", "停止标志", "停车计时器", "长凳", "鸟", "猫", "狗",
    "马", "羊", "牛", "大象", "熊", "斑马", "长颈鹿", "背包", "雨伞",
    "手提包", "领带", "行李箱", "飞盘", "滑雪板", "滑雪单板", "运动球",
    "风筝", "棒球棒", "棒球手套", "滑板", "冲浪板", "网球拍", "瓶子",
    "红酒杯", "杯子", "叉子", "刀", "勺子", "碗", "香蕉", "苹果",
    "三明治", "橙子", "西兰花", "胡萝卜", "热狗", "披萨", "甜甜圈",
    "蛋糕", "椅子", "沙发", "盆栽", "床", "餐桌", "厕所", "电视",
    "笔记本电脑", "鼠标", "遥控器", "键盘", "手机", "微波炉", "烤箱",
    "烤面包机", "水槽", "冰箱", "书", "时钟", "花瓶", "剪刀",
    "泰迪熊", "吹风机", "牙刷",
]


# ============================================================
# Step 2: 配置超参数
# ============================================================
class CONFIG:
    """超参数配置中心 —— 目标检测任务的所有可调参数。"""

    # --- 数据相关 ---
    # data_dir: 数据集存放目录
    data_dir = "data"

    # num_classes: 检测类别数(含背景)
    #   微调时必须修改！= 你的类别数 + 1(背景类)
    #   例: 检测猫和狗 → num_classes=3 (猫/狗/背景)
    #   推理时使用COCO预训练=81类(80物体+1背景)
    num_classes = 81  # COCO预训练

    # class_names: 类别名称列表(索引0=背景)
    #   微调时替换为你的类别名称
    class_names = ["背景"] + COCO_CLASSES

    # --- 模型相关 ---
    # backbone: 骨干网络类型
    #   "resnet50_fpn": ResNet50 + 特征金字塔网络(FPN)
    #   【为什么用FPN？】
    #   多尺度特征融合：浅层特征分辨率高(检测小物体)，深层特征语义强(检测大物体)
    #   FPN将多层特征融合，实现多尺度检测
    backbone = "resnet50_fpn"

    # pretrained=True: 使用COCO预训练权重
    #   【为什么用预训练？】
    #   COCO预训练已学会通用视觉特征(边缘/纹理/形状)
    #   微调时只需学习新类别，比从零训练快10倍以上
    pretrained = True

    # min_size/max_size: 输入图像的最小/最大边长
    #   图像会被缩放到 [min_size, max_size] 范围内
    #   为什么min_size=800？检测任务需要较大分辨率看清细节
    #   为什么不是32x32(分类)？分类只需全局判断，检测需要精确定位
    min_size = 800
    max_size = 1333

    # --- 推理相关 ---
    # confidence_threshold=0.5: 置信度阈值
    #   只保留置信度>0.5的检测结果
    #   为什么0.5？过高会漏检，过低会误检
    #   可根据场景调整：安防(0.3宽松) vs 医疗(0.7严格)
    confidence_threshold = 0.5

    # nms_threshold=0.5: 非极大值抑制(NMS)的IoU阈值
    #   【NMS原理】
    #   同一物体可能被多个框检测到，NMS保留最优框，删除冗余框
    #   步骤: 按置信度排序 → 保留最高分框 → 删除与它IoU>0.5的框 → 重复
    #   为什么0.5？IoU>0.5说明两个框基本重叠，保留一个即可
    nms_threshold = 0.5

    # --- 训练相关(微调时使用) ---
    # batch_size=2: 每批图像数
    #   为什么只有2？检测模型的图像分辨率高(800x1333)，显存消耗大
    #   RTX 4070(8GB显存)通常只能batch=2，batch=4可能OOM
    batch_size = 2

    # learning_rate=5e-3: 微调学习率
    #   为什么比分类(1e-3)大5倍？微调时backbone用小LR，新层用大LR
    #   这里是全局LR，实际各层LR不同(后续可设置)
    learning_rate = 5e-3

    # epochs=10: 微调轮数
    #   为什么只10轮？微调不需要太多轮，预训练权重已很好
    #   太多轮会过拟合(检测标注数据通常不多)
    epochs = 10

    # weight_decay=5e-4: L2正则化
    weight_decay = 5e-4

    # --- 保存相关 ---
    save_dir = "cnn/output/detection"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 模型构建
# ============================================================
def get_detection_model(cfg):
    """
    创建Faster R-CNN检测模型。

    【Faster R-CNN架构】
    输入图像 → ResNet50+FPN(骨干) → 多尺度特征图
                                    ↓
                          RPN(区域建议) → 候选框(~2000个)
                                    ↓
                          ROI Pooling → 固定尺寸特征
                                    ↓
                          分类头 + 回归头 → (类别, 边界框)

    【为什么选择Faster R-CNN而不是YOLO？】
    - Faster R-CNN: 两阶段，精度高，适合学习检测原理
    - YOLO: 单阶段，速度快，但原理更复杂(锚框设计、损失函数等)
    - 初学者推荐先学Faster R-CNN，理解检测的完整流程
    """
    if cfg.pretrained and cfg.num_classes == 81:
        # 直接使用COCO预训练模型(81类)
        model = fasterrcnn_resnet50_fpn(
            pretrained=True,
            min_size=cfg.min_size,
            max_size=cfg.max_size,
        )
    else:
        # 微调: 加载预训练backbone，替换分类头
        # 【微调的原理】
        # 1. 保留backbone(已学会通用特征: 边缘/纹理/形状)
        # 2. 替换分类头(旧头是81类，新头是你的类别数)
        # 3. 只需少量数据就能训练出好模型
        model = fasterrcnn_resnet50_fpn(pretrained=True)

        # 替换分类头
        # 原始分类头: Linear(in_features, 81)
        # 新分类头: Linear(in_features, num_classes)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.num_classes)

    return model


# ============================================================
# Step 4: 自定义数据集(微调时使用)
# ============================================================
class DetectionDataset(Dataset):
    """
    目标检测自定义数据集模板。

    【标注格式要求】
    每张图像对应一个标注文件(JSON)，格式如下：
    {
        "boxes": [[x1, y1, x2, y2], ...],   # 边界框坐标(左上+右下)
        "labels": [1, 2, ...]                # 类别索引(从1开始，0=背景)
    }

    【目录结构】
    data/your_dataset/
    ├── images/          # 图像文件
    │   ├── 001.jpg
    │   ├── 002.jpg
    │   └── ...
    └── annotations/     # 标注文件(与图像同名，扩展名.json)
        ├── 001.json
        ├── 002.json
        └── ...

    【如何制作标注？】
    推荐工具: LabelImg(图形界面), CVAT(在线协作), Roboflow(云端)
    """

    def __init__(self, root_dir, transforms=None):
        """
        参数:
            root_dir: 数据集根目录(包含images/和annotations/)
            transforms: 图像变换(检测任务的变换需要同时变换边界框!)
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.img_dir = os.path.join(root_dir, "images")
        self.ann_dir = os.path.join(root_dir, "annotations")

        # 获取所有图像文件名
        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __getitem__(self, idx):
        import json

        # 加载图像
        img_path = os.path.join(self.img_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        # 加载标注
        ann_path = os.path.join(
            self.ann_dir, os.path.splitext(self.images[idx])[0] + ".json",
        )
        with open(ann_path, "r") as f:
            ann = json.load(f)

        # 转为张量
        # boxes: (N, 4) 格式 [x1, y1, x2, y2]
        # labels: (N,) 类别索引
        boxes = torch.as_tensor(ann["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(ann["labels"], dtype=torch.int64)

        # 构造target字典
        # torchvision检测模型要求target包含这些字段
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        # 如果有面积信息(用于COCO评估)
        if "area" in ann:
            target["area"] = torch.as_tensor(ann["area"], dtype=torch.float32)

        # 如果有iscrowd标注(拥挤人群标记，COCO格式)
        if "iscrowd" in ann:
            target["iscrowd"] = torch.as_tensor(ann["iscrowd"], dtype=torch.int64)

        # 变换
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return F.to_tensor(img), target

    def __len__(self):
        return len(self.images)


# ============================================================
# Step 5: 推理与可视化
# ============================================================
@torch.no_grad()
def detect(model, image, cfg):
    """
    对单张图像进行目标检测。

    参数:
        image: PIL Image 或 Tensor (C, H, W)
    返回:
        detections: 字典, 包含 boxes, labels, scores
    """
    model.eval()

    # 预处理
    if isinstance(image, Image.Image):
        image_tensor = F.to_tensor(image).to(cfg.device)
    else:
        image_tensor = image.to(cfg.device)

    # 推理
    # torchvision检测模型输入: List[Tensor]，每个Tensor是一张图
    outputs = model([image_tensor])[0]

    # 过滤低置信度检测
    # 【为什么需要过滤？】
    # Faster R-CNN默认输出所有候选框(可能上百个)
    # 大部分置信度很低(噪声)，只保留高置信度的
    keep = outputs["scores"] > cfg.confidence_threshold

    detections = {
        "boxes": outputs["boxes"][keep].cpu(),
        "labels": outputs["labels"][keep].cpu(),
        "scores": outputs["scores"][keep].cpu(),
    }

    return detections


def draw_detections(image, detections, cfg, save_path=None):
    """
    在图像上绘制检测结果(边界框+类别+置信度)。

    【边界框坐标格式】
    (x1, y1, x2, y2): 左上角+右下角坐标
      x1, y1: 左上角像素坐标
      x2, y2: 右下角像素坐标
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))

    # 显示图像
    if isinstance(image, Image.Image):
        ax.imshow(image)
    elif isinstance(image, torch.Tensor):
        # Tensor (C, H, W) → numpy (H, W, C)
        img_np = image.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img_np)

    # 为每个类别分配颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, len(cfg.class_names)))

    # 绘制每个检测框
    for box, label, score in zip(
        detections["boxes"], detections["labels"], detections["scores"]
    ):
        x1, y1, x2, y2 = box.numpy()
        class_idx = label.item()
        class_name = cfg.class_names[class_idx] if class_idx < len(cfg.class_names) else f"类{class_idx}"
        color = colors[class_idx % len(colors)]

        # 绘制边界框
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)

        # 添加标签
        ax.text(
            x1, y1 - 5, f"{class_name} {score:.2f}",
            fontsize=10, color="white",
            bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=2),
        )

    ax.axis("off")
    plt.title("目标检测结果", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ 检测结果已保存: {save_path}")
    plt.close()


def create_demo_image():
    """
    创建一张示例图像用于演示检测。

    在实际使用中，你会用自己的图像替换这里。
    """
    # 创建一张包含简单图形的图像
    img = Image.new("RGB", (640, 480), color=(200, 220, 240))
    draw = ImageDraw.Draw(img)

    # 画一些简单形状模拟物体
    # "汽车"(矩形)
    draw.rectangle([100, 300, 250, 380], fill=(50, 50, 200), outline=(0, 0, 0))
    # 车窗
    draw.rectangle([120, 310, 180, 340], fill=(180, 220, 255))
    # 车轮
    draw.ellipse([110, 360, 145, 395], fill=(40, 40, 40))
    draw.ellipse([205, 360, 240, 395], fill=(40, 40, 40))

    # "人"(简笔画)
    draw.ellipse([400, 200, 430, 230], fill=(255, 200, 160))  # 头
    draw.rectangle([405, 230, 425, 300], fill=(200, 50, 50))   # 身体
    draw.line([415, 300, 400, 370], fill=(0, 0, 100), width=3)  # 腿
    draw.line([415, 300, 430, 370], fill=(0, 0, 100), width=3)

    # "杯子"(梯形)
    draw.polygon([500, 250, 550, 250, 540, 350, 510, 350], fill=(255, 255, 100))

    return img


# ============================================================
# Step 6: 训练函数(微调时使用)
# ============================================================
def train_one_epoch(model, optimizer, data_loader, cfg):
    """
    训练一个epoch(微调)。

    【检测模型训练的特殊之处】
    1. 损失函数内置在模型中(model返回loss字典)，不需要外部定义criterion
    2. target必须包含 boxes, labels 等字段
    3. 模型在训练模式返回loss，在评估模式返回detections
    4. batch中的图像大小可以不同(检测模型内部处理)
    """
    model.train()
    total_loss = 0

    for i, (images, targets) in enumerate(data_loader):
        # 将数据移到设备
        images = [img.to(cfg.device) for img in images]
        targets = [{k: v.to(cfg.device) for k, v in t.items()} for t in targets]

        # 前向传播
        # 检测模型训练时返回loss字典
        loss_dict = model(images, targets)

        # 总损失 = 分类损失 + 回归损失(RPN + ROI)
        losses = sum(loss for loss in loss_dict.values())

        # 反向传播
        optimizer.zero_grad()
        losses.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += losses.item()

        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{len(data_loader)} | Loss: {losses.item():.4f}")

    return total_loss / len(data_loader)


def finetune(model, train_loader, cfg):
    """
    微调检测模型。

    【微调策略】
    1. 冻结backbone前几层(低级特征已学好，不需要再学)
    2. 用较大学习率训练新分类头，较小学习率训练backbone
    3. 微调轮数少(5-10轮)，过多会过拟合
    """
    # 冻结backbone的浅层(BN和前几个卷积块)
    # 【为什么冻结？】
    # 浅层提取通用特征(边缘/纹理)，这些特征对所有图像都有用
    # 冻结后减少参数量，训练更快，防止过拟合
    for name, param in model.backbone.body.named_parameters():
        if "layer2" not in name and "layer3" not in name and "layer4" not in name:
            param.requires_grad = False

    # 不同层使用不同学习率
    # 新分类头: 大LR(需要快速学习新类别)
    # backbone: 小LR(只需要微调)
    params = [
        {"params": model.roi_heads.box_predictor.parameters(), "lr": cfg.learning_rate},
        {"params": model.backbone.body.parameters(), "lr": cfg.learning_rate * 0.1},
        {"params": model.rpn.parameters(), "lr": cfg.learning_rate * 0.5},
    ]

    optimizer = optim.SGD(params, momentum=0.9, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print(f"\n微调开始 (共{cfg.epochs}轮)...")
    for epoch in range(cfg.epochs):
        avg_loss = train_one_epoch(model, optimizer, train_loader, cfg)
        scheduler.step()
        print(f"Epoch {epoch+1}/{cfg.epochs} | Avg Loss: {avg_loss:.4f}")

    # 解冻所有参数(推理时)
    for param in model.parameters():
        param.requires_grad = True

    return model


# ============================================================
# Step 7: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("CNN 目标检测 - Faster R-CNN")
    print("=" * 60)

    cfg = CONFIG()
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载模型
    print("\n加载Faster R-CNN模型(ResNet50+FPN, COCO预训练)...")
    model = get_detection_model(cfg).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # === 推理演示 ===
    print("\n" + "=" * 60)
    print("推理演示: 对示例图像进行目标检测")
    print("=" * 60)

    # 创建示例图像
    demo_img = create_demo_image()
    demo_path = os.path.join(cfg.save_dir, "demo_input.png")
    demo_img.save(demo_path)
    print(f"示例图像已保存: {demo_path}")

    # 检测
    print("正在检测...")
    detections = detect(model, demo_img, cfg)

    n_dets = len(detections["boxes"])
    print(f"检测到 {n_dets} 个目标:")
    for i in range(n_dets):
        label = detections["labels"][i].item()
        score = detections["scores"][i].item()
        box = detections["boxes"][i].numpy()
        name = cfg.class_names[label] if label < len(cfg.class_names) else f"类{label}"
        print(f"  [{name}] 置信度={score:.2f} 位置=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})")

    # 可视化
    output_path = os.path.join(cfg.save_dir, "detection_result.png")
    draw_detections(demo_img, detections, cfg, save_path=output_path)

    # === 微调说明 ===
    print("\n" + "=" * 60)
    print("微调指南")
    print("=" * 60)
    print("""
要使用自己的数据集微调检测模型:

1. 准备数据集:
   data/your_dataset/
   ├── images/          # 图像文件(.jpg/.png)
   └── annotations/     # 标注文件(.json)
       格式: {"boxes": [[x1,y1,x2,y2],...], "labels": [1,2,...]}

2. 修改CONFIG:
   num_classes = 你的类别数 + 1  # +1是背景类
   class_names = ["背景", "类别1", "类别2", ...]

3. 取消下方注释运行微调:

# dataset = DetectionDataset("data/your_dataset")
# train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
# model = get_detection_model(cfg)  # 会自动替换分类头
# model = finetune(model, train_loader, cfg)
# torch.save(model.state_dict(), os.path.join(cfg.save_dir, "detection_finetuned.pth"))

4. 推荐标注工具:
   - LabelImg: 图形界面，适合小项目
   - CVAT: 在线协作，适合团队
   - Roboflow: 云端服务，一键导出
""")


def collate_fn(batch):
    """
    自定义collate函数，用于DataLoader。

    【为什么检测任务需要自定义collate？】
    - 分类任务: 每张图大小相同，可以stack成tensor
    - 检测任务: 每张图大小可能不同，每张的检测框数量也不同
    - 默认collate会尝试stack，导致维度不匹配报错
    - 自定义collate: 保持List格式，让模型内部处理不同尺寸
    """
    return tuple(zip(*batch))


if __name__ == "__main__":
    main()
