# CNN 卷积神经网络 完全指南

---

## 目录

1. [基础知识](#1-基础知识)
2. [技术原理](#2-技术原理)
3. [四大任务类型](#3-四大任务类型)
4. [应用场景](#4-应用场景)
5. [使用说明](#5-使用说明)
6. [任务类型对比](#6-任务类型对比)
7. [常见问题与调优](#7-常见问题与调优)
8. [进阶扩展](#8-进阶扩展)

---

## 1. 基础知识

### 1.1 什么是卷积神经网络 (CNN)

卷积神经网络（Convolutional Neural Network，简称 CNN），是专门为处理**图像数据**设计的深度学习模型。

**核心思想**：通过"卷积核"在图像上滑动，自动提取从低级(边缘/纹理)到高级(形状/对象)的层次化特征。

```
输入图像         卷积核滑动         特征图           分类结果
┌─────────┐    ┌───┐            ┌─────────┐     ┌───────┐
│ ▓▓░░▓▓ │    │1 0│  → 滑动 →  │ ░░██░░ │  →  │ 猫:0.9│
│ ▓▓░░▓▓ │    │0 -1│            │ ░░██░░ │     │ 狗:0.1│
│ ░░▓▓░░ │    └───┘              │ ██░░██ │     └───────┘
│ ░░▓▓░░ │     垂直边缘检测        边缘特征图
└─────────┘
```

### 1.2 CNN vs FNN 的核心区别

| 对比项 | FNN (前馈网络) | CNN (卷积网络) |
|--------|---------------|---------------|
| **输入** | 1维特征向量 | 2D/3D图像(保留空间结构) |
| **连接方式** | 全连接(每个像素连到每个神经元) | 局部连接(只关注局部区域) |
| **参数共享** | 无(每对连接独立权重) | 有(同一卷积核在整个图像复用) |
| **参数量** | 极大(28×28图→784×256≈20万) | 极小(3×3×32核≈320) |
| **平移不变性** | 无(猫在左上≠猫在右下) | 有(卷积核全图滑动，任何位置都能检测) |
| **适用数据** | 表格/数值特征 | 图像/空间数据 |

**为什么图像不用FNN？**
- 32×32×3的图，展平为3072维向量，第一层全连接到256神经元 = 786,432个参数
- 而CNN第一层3×3×32个卷积核 = 只有864个参数，少了近1000倍
- 更关键：FNN展平图像后丢失了空间关系(相邻像素的信息)，CNN保留了它

### 1.3 CNN 的关键组成

| 组件 | 作用 | 类比 |
|------|------|------|
| **卷积层 (nn.Conv2d)** | 用卷积核提取局部特征 | 手电筒扫视：照亮局部区域 |
| **激活函数 (nn.ReLU)** | 引入非线性 | 开关：只让有用信号通过 |
| **池化层 (nn.MaxPool2d)** | 降采样，扩大感受野 | 缩略图：保留关键信息，减小尺寸 |
| **批归一化 (nn.BatchNorm2d)** | 稳定训练，加速收敛 | 标准化量杯：统一度量 |
| **Dropout / Dropout2d** | 随机丢弃，防止过拟合 | 团队备胎：不依赖单个人 |
| **全连接层 (nn.Linear)** | 将特征映射到输出 | 裁判：综合所有特征做判断 |

### 1.4 训练流程（与FNN相同的5步循环）

```
┌──────────────────────────────────────────────┐
│              每个 batch 重复执行               │
│                                              │
│  1. optimizer.zero_grad()  ← 清零梯度        │
│  2. outputs = model(x)     ← 前向传播        │
│  3. loss = criterion(…)    ← 计算损失        │
│  4. loss.backward()        ← 反向传播(求梯度) │
│  5. optimizer.step()       ← 更新参数        │
│                                              │
│  ⚠️ CNN额外: 梯度裁剪(防止梯度爆炸)            │
└──────────────────────────────────────────────┘
```

---

## 2. 技术原理

### 2.1 卷积操作 (Convolution)

卷积核在输入图像上滑动，逐元素相乘再求和，得到输出特征图：

```
输入 5×5              卷积核 3×3           输出 3×3
┌───────────┐        ┌───────┐          ┌───────┐
│ 1  0  1  0  1 │    │ 1  0  1 │        │ 4  3  4 │
│ 0  1  0  1  0 │  ⊗ │ 0  1  0 │  =    │ 2  4  3 │
│ 1  0  1  0  1 │    │ 1  0  1 │        │ 4  3  4 │
│ 0  1  0  1  0 │    └───────┘          └───────┘
│ 1  0  1  0  1 │
└───────────┘

计算示例(左上角):
1×1 + 0×0 + 1×1 + 0×0 + 1×1 + 0×0 + 1×1 + 0×0 + 1×1 = 4
```

**输出尺寸计算**：

```
output_size = (input_size + 2×padding - kernel_size) / stride + 1

例: 输入32×32, kernel=3, padding=1, stride=1
    output = (32 + 2×1 - 3) / 1 + 1 = 32  ← 尺寸不变
```

### 2.2 卷积核详解

| 卷积核大小 | 感受野 | 参数量(输入32通道) | 适用场景 |
|-----------|--------|-------------------|---------|
| 1×1 | 1×1 | 32×C×1×1 | 降维/升维(不改变空间) |
| 3×3 ⭐ | 3×3 | 32×C×9 | **最常用**，两个3×3=一个5×5但参数更少 |
| 5×5 | 5×5 | 32×C×25 | 较少用，可用两个3×3替代 |
| 7×7 | 7×7 | 32×C×49 | 仅第一层用(如ResNet)，后续用3×3 |

**为什么3×3最常用？**(VGG的发现)
- 两个3×3的感受野 = 一个5×5 (都是看5×5的区域)
- 但参数: 2×9C² = 18C² < 25C² (少28%)
- 而且多一次非线性变换(两个ReLU vs 一个ReLU)

### 2.3 池化操作 (Pooling)

```
最大池化 (MaxPool2d, 2×2):

输入 4×4              输出 2×2
┌───────────┐        ┌───────┐
│ 1  3 │ 2  1 │      │ 3  2 │   ← 每个区域取最大值
│ 2  3 │ 1  2 │  →   │ 4  3 │
│──────┼──────│
│ 0  1 │ 3  2 │
│ 1  4 │ 2  3 │
└───────────┘        └───────┘

作用:
1. 降采样: 4×4 → 2×2, 计算量减4倍
2. 扩大感受野: 后续层每个神经元"看到"更大区域
3. 平移不变性: 小幅位移不影响池化结果
```

### 2.4 空洞卷积 (Dilated Convolution)

分割任务中使用的核心技术，**不降低分辨率就能扩大感受野**：

```
普通卷积 rate=1:       空洞卷积 rate=2:       空洞卷积 rate=4:
┌───┐                 ┌─┬─┬─┐               ┌─┬─┬─┬─┬─┬─┬─┐
│●●●│  感受野 3×3     │●│●│●│  感受野 5×5    │●│ │ │●│ │ │●│  感受野 9×9
│●●●│                 │●│●│●│               │●│ │ │●│ │ │●│
│●●●│                 │●│●│●│               │●│ │ │●│ │ │●│
└───┘                 └─┴─┴─┘               └─┴─┴─┴─┴─┴─┴─┘
                      ●=有效位置,空=跳过

参数量相同！都是3×3核=9个参数
但感受野从3×3扩大到5×5甚至9×9
```

### 2.5 特征金字塔网络 (FPN)

检测任务中使用的多尺度特征融合技术：

```
                    自顶向下融合
ResNet层4 ─────────────────────→ P5 (强语义，大物体)
    ↓ 上采样 ↑                    ↑ 横向连接(1×1卷积)
ResNet层3 ─────────────────────→ P4 (中等语义+细节)
    ↓ 上采样 ↑                    ↑
ResNet层2 ─────────────────────→ P3 (弱语义，小物体)
    ↓ 上采样 ↑                    ↑
ResNet层1 ─────────────────────→ P2 (最底层特征)

为什么需要FPN？
- 浅层特征: 分辨率高，适合检测小物体，但语义弱
- 深层特征: 语义强，适合检测大物体，但分辨率低
- FPN融合两者: 每个尺度都有丰富的语义+足够的分辨率
```

### 2.6 全局平均池化 (AdaptiveAvgPool2d)

```
传统方式: Flatten + FC
  特征图 128×4×4 = 2048维 → FC(2048, 256) → 52万参数

全局平均池化:
  特征图 128×4×4 → 每个通道取平均值 → 128维 → FC(128, 256) → 3.3万参数

参数量减少16倍！而且对输入尺寸不敏感
```

### 2.7 数据增强 (Data Augmentation)

CNN防止过拟合的**核心手段**，相当于"免费"扩充数据：

| 增强方法 | 原理 | 适用场景 |
|---------|------|---------|
| RandomCrop | 随机裁剪，模拟物体位置变化 | 通用 |
| RandomHorizontalFlip | 随机水平翻转 | 自然图像(不能用于数字/文字) |
| ColorJitter | 随机改变亮度/对比度/饱和度 | 户外图像 |
| RandomRotation | 随机旋转 | 通用 |
| RandomErasing | 随机擦除区域，模拟遮挡 | 通用 |

**关键原则**：训练时增强，测试时不增强！

---

## 3. 四大任务类型

### 3.1 图像分类 (Image Classification)

**目标**：判断整张图像属于哪个类别

```
输入: 28×28灰度图 → CNN → 输出: [0.1, 0.8, 0.05, 0.05] → 预测: 裤子(类别1)
                                  (10个类别的概率)             (取最大值)
```

**输出层**：`num_classes` 个神经元，不加激活函数
**损失函数**：`CrossEntropyLoss`（内含 Softmax）
**评估指标**：准确率(Accuracy)、F1、混淆矩阵
**本模板数据**：Fashion-MNIST (70,000张 28×28 灰度图，10类)

### 3.2 目标检测 (Object Detection)

**目标**：找出图中所有物体的位置(边界框)和类别

```
输入: 图像 → Faster R-CNN → 输出: N个(边界框, 类别, 置信度)
                                     ┌─────────┐
                              猫 0.95│ ■       │
                                     │    ■    │狗 0.87
                                     └─────────┘
```

**输出**：多个(边界框[x1,y1,x2,y2], 类别, 置信度)
**损失函数**：RPN分类 + RPN回归 + ROI分类 + ROI回归（模型内置）
**评估指标**：mAP(平均精度均值)
**本模板**：Faster R-CNN (COCO预训练，80类)

**与分类的区别**：
- 分类: 1张图 → 1个标签
- 检测: 1张图 → N个(框, 标签, 置信度)

### 3.3 图像分割 (Semantic Segmentation)

**目标**：对图像中**每个像素**预测类别标签

```
输入: 图像 → DeepLabV3 → 输出: H×W的类别图
┌─────────┐              ┌─────────┐
│ 🐱      │              │ 2  2  0 │  2=猫, 0=背景
│    🌳   │      →       │ 0  0  3 │  3=树
│         │              │ 0  0  0 │
└─────────┘              └─────────┘
原图                      每个像素的类别
```

**输出**：与输入同尺寸的类别图 (H, W)，每个像素一个类别
**损失函数**：`CrossEntropyLoss`（像素级，`ignore_index=255`跳过边界）
**评估指标**：mIoU(平均交并比)、像素准确率
**本模板数据**：合成分割数据 (5类，含背景)

**与分类/检测的区别**：
- 分类: 1张图 → 1个标签
- 检测: 1张图 → N个(框, 标签)
- 分割: 1张图 → H×W个像素标签

### 3.4 人脸识别 (Face Recognition)

**目标**：识别人脸身份，判断两张脸是否为同一人

```
第1步 - 人脸检测: 从图像中裁剪出人脸
第2步 - 嵌入提取: 人脸 → CNN → 128维嵌入向量
第3步 - 相似度比较:

  人脸A → [0.2, 0.8, 0.1, ...] ─┐
                                  ├→ 余弦相似度 = 0.92 → 同一人 ✓
  人脸B → [0.25, 0.78, 0.12, ...]┘

两种模式:
  验证: 这两张脸是同一人吗？ → 是/否
  辨识: 这张脸是谁？ → 在库中搜索最相似的人
```

**输出**：128维嵌入向量 → 余弦相似度
**训练损失**：`CrossEntropyLoss`（分类训练，提取嵌入）
**评估指标**：验证准确率、辨识Top-K准确率
**本模板数据**：合成人脸数据 (40人×10张，64×64灰度图)

**与分类的区别**：
- 分类: 只能识别训练过的N个人，新人需重新训练
- 人脸识别: 提取通用嵌入，新人只需1张注册照即可识别

---

## 4. 应用场景

### 4.1 图像分类应用

| 场景 | 输入 | 类别数 | 说明 |
|------|------|--------|------|
| 手写数字识别 | 28×28灰度图 | 10 | MNIST，入门经典 |
| 自然图像分类 | 32×32彩色图 | 10/1000 | CIFAR-10/ImageNet |
| 医学影像分类 | CT/X光切片 | 2-10 | 正常/异常/疾病类型 |
| 商品分类 | 商品图片 | 数百 | 电商自动分类 |
| 农作物病害 | 叶片照片 | 数十 | 识别病害类型 |

### 4.2 目标检测应用

| 场景 | 检测目标 | 说明 |
|------|---------|------|
| 自动驾驶 | 车辆/行人/信号灯 | 实时性要求高 |
| 安防监控 | 人员/异常行为 | 7×24不间断 |
| 工业质检 | 缺陷/零件 | 精度要求高 |
| 零售分析 | 商品/货架 | 商品识别与计数 |
| 医学影像 | 病灶/器官 | 辅助诊断定位 |

### 4.3 图像分割应用

| 场景 | 分割目标 | 说明 |
|------|---------|------|
| 自动驾驶 | 道路/车道/行人 | 像素级场景理解 |
| 医学影像 | 器官/肿瘤 | 精确轮廓分割 |
| 遥感图像 | 建筑/农田/水体 | 大范围地表分类 |
| 人像分割 | 前景人像 | 背景替换/美颜 |
| 工业检测 | 缺陷区域 | 精确缺陷定位 |

### 4.4 人脸识别应用

| 场景 | 模式 | 说明 |
|------|------|------|
| 手机解锁 | 验证 | 1:1比对，安全第一 |
| 门禁考勤 | 辨识 | 1:N搜索，便捷第一 |
| 安防监控 | 辨识 | 1:N搜索，大库搜索 |
| 金融核验 | 验证 | 1:1比对，合规要求 |
| 社交标签 | 辨识 | 1:N搜索，自动标注 |

---

## 5. 使用说明

### 5.1 快速开始

```bash
# 进入项目根目录
cd py_ai_tech/

# 激活虚拟环境
source venv/bin/activate

# 运行图像分类模板（Fashion-MNIST，70000张28×28灰度图，10类）
python cnn/classification.py

# 运行目标检测模板（Faster R-CNN，COCO预训练推理演示）
python cnn/detection.py

# 运行图像分割模板（DeepLabV3，合成分割数据，5类）
python cnn/segmentation.py

# 运行人脸识别模板（合成人脸数据，40人×10张）
python cnn/face_recognition.py
```

### 5.2 使用自己的数据

修改 `CONFIG` 类和相关数据加载函数：

**图像分类**：
```python
class CONFIG:
    # 1. 修改数据相关参数
    num_classes = 5                    # 你的类别数
    class_names = ["猫","狗","鸟","鱼","兔"]
    image_size = 224                   # 根据你的图像大小调整
    in_channels = 3                    # RGB=3, 灰度=1

# 2. 修改 get_dataloaders() 函数
#    替换 datasets.FashionMNIST 为你自己的 Dataset
#    torchvision.datasets.ImageFolder 可直接读取目录结构:
#      data/my_dataset/
#      ├── cat/    (猫的图片)
#      ├── dog/    (狗的图片)
#      └── bird/   (鸟的图片)
train_dataset = datasets.ImageFolder("data/my_dataset", transform=train_transform)
```

**目标检测**：
```python
# 1. 修改 CONFIG
class CONFIG:
    num_classes = 你的类别数 + 1  # +1是背景类！
    class_names = ["背景", "缺陷A", "缺陷B"]
    pretrained = False            # 从零训练或微调

# 2. 准备标注数据
data/your_dataset/
├── images/
│   ├── 001.jpg
│   └── 002.jpg
└── annotations/
    ├── 001.json  → {"boxes": [[x1,y1,x2,y2],...], "labels": [1,2,...]}
    └── 002.json

# 3. 取消 main() 中微调代码的注释
```

**图像分割**：
```python
class CONFIG:
    num_classes = 你的类别数 + 1  # +1是背景类
    class_names = ["背景", "道路", "建筑", "植被"]
    image_size = 512              # 根据GPU显存调整

# VOC格式数据直接用 torchvision.datasets.VOCSegmentation
# 自定义数据需继承 Dataset，返回 (image, target_mask)
```

**人脸识别**：
```python
class CONFIG:
    num_identities = 100          # 身份(人)数量
    image_size = 64               # 人脸裁剪尺寸
    embedding_dim = 128           # 嵌入维度 ≈ 身份数 × 3

# 1. 先用人脸检测裁剪对齐人脸
# 2. 修改 FaceDataset 加载你的人脸数据
# 3. 注册: 提取人脸嵌入存入人脸库
# 4. 识别: 查询人脸与库中嵌入比较
```

### 5.3 修改超参数

修改各文件中的 `CONFIG` 类：

```python
class CONFIG:
    # --- 数据相关 ---
    image_size = 28               # 输入图像尺寸
    in_channels = 1               # 输入通道数(灰度=1, RGB=3)
    test_size = 0.2               # 验证集比例
    random_state = 42             # 随机种子

    # --- 模型相关 ---
    conv_channels = [32, 64, 128] # 卷积通道数(逐层加倍)
    dropout_rate = 0.5            # Dropout比例(FC层)
    embedding_dim = 128           # 嵌入维度(人脸识别)

    # --- 训练相关 ---
    batch_size = 32               # 批次大小
    learning_rate = 1e-3          # 初始学习率
    epochs = 50                   # 最大训练轮数
    weight_decay = 5e-4           # L2正则化强度

    # --- 早停 & LR调度 ---
    early_stop_patience = 10      # 早停耐心值
    scheduler_type = "cosine"     # 调度器类型

    # --- 梯度裁剪 ---
    max_grad_norm = 5.0           # 梯度L2范数上限

    # --- AMP混合精度 ---
    use_amp = True                # 启用混合精度(仅GPU有效，速度↑1.5-2x)

    # --- 数据加载优化 ---
    num_workers = min(4, os.cpu_count() or 1)  # 多进程并行加载(0=主进程)

    # --- 数据增强 ---
    use_augmentation = True       # 是否使用数据增强
    random_crop_padding = 4       # RandomCrop填充
    random_hflip_prob = 0.5       # 水平翻转概率
```

### 5.4 模型保存与加载

```python
# 保存模型
torch.save({
    "model_state_dict": model.state_dict(),
    "config": vars(cfg),
}, "model.pth")

# 加载模型（分类示例）
checkpoint = torch.load("model.pth", weights_only=True)
model = CNNClassifier(cfg).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

### 5.5 对新数据预测

```python
# 图像分类：预处理 → softmax → argmax
from PIL import Image
from torchvision import transforms
img = Image.open("test.jpg").convert("L")  # 转为灰度图
transform = transforms.Compose([
    transforms.Resize(28), transforms.ToTensor(),
    transforms.Normalize(mean=[0.2860], std=[0.3530]),  # Fashion-MNIST统计值
])
tensor = transform(img).unsqueeze(0).to(device)
with torch.no_grad():
    output = model(tensor)
    prob = torch.softmax(output, dim=1)
    pred = prob.argmax(dim=1)

# 目标检测：预处理 → 模型推理 → 过滤低置信度
from torchvision.transforms.functional import to_tensor
img = Image.open("test.jpg")
tensor = to_tensor(img).to(device)
with torch.no_grad():
    outputs = model([tensor])[0]
    keep = outputs["scores"] > 0.5
    boxes = outputs["boxes"][keep]
    labels = outputs["labels"][keep]
    scores = outputs["scores"][keep]

# 图像分割：预处理 → argmax → 彩色可视化
with torch.no_grad():
    output = model(tensor.unsqueeze(0))["out"]
    pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()
    pred_color = VOC_COLORS[pred_mask]

# 人脸识别：嵌入提取 → 余弦相似度
with torch.no_grad():
    emb1 = model.get_embedding(face1_tensor)  # (1, 128)
    emb2 = model.get_embedding(face2_tensor)  # (1, 128)
    sim = nn.functional.cosine_similarity(emb1, emb2)
    is_same = sim.item() > 0.5
```

---

## 6. 任务类型对比

### 6.1 核心差异一览

| 对比项 | 图像分类 | 目标检测 | 图像分割 | 人脸识别 |
|--------|---------|---------|---------|---------|
| **预测目标** | 整图类别 | 物体位置+类别 | 每个像素类别 | 人脸嵌入+相似度 |
| **输出粒度** | 1个标签/图 | N个框+标签/图 | H×W标签/图 | 1个128维向量/脸 |
| **数据集** | Fashion-MNIST(70K) | COCO(118K) | 合成数据 | 合成人脸(400) |
| **输入尺寸** | 28×28 | 800×1333 | 128×128 | 64×64 |
| **模型** | 自定义CNN | Faster R-CNN | DeepLabV3 | 嵌入网络 |
| **Backbone** | 自定义(3层) | ResNet50+FPN | ResNet50 | 自定义(4层) |
| **损失函数** | CrossEntropyLoss | 模型内置(4项) | CrossEntropyLoss(像素级) | CrossEntropyLoss(分类训练) |
| **核心指标** | Accuracy | mAP | mIoU | 验证准确率/辨识Top-K |
| **预训练** | 可选 | 推荐(必用) | 推荐(必用) | 不适用(小数据) |
| **数据增强** | 必须 | 必须 | 必须 | 有限(人脸变形有限) |

### 6.2 网络结构对比

| 对比项 | 图像分类 | 目标检测 | 图像分割 | 人脸识别 |
|--------|---------|---------|---------|---------|
| **模型类** | `CNNClassifier` | `Faster R-CNN` | `DeepLabV3` | `FaceEmbeddingNet` |
| **卷积通道** | [32,64,128] | ResNet50内置 | ResNet50+ASPP | [32,64,128,256] |
| **卷积块数** | 3 stage × 2层 | 50层ResNet | 50层+ASPP | 4 block |
| **池化** | MaxPool×3 | FPN多尺度 | 空洞卷积(无池化) | MaxPool×4 |
| **FC层** | 128→256→10 | ROI Head | 1×1卷积 | 256→128→40 |
| **特殊组件** | AdaptiveAvgPool | RPN+ROI Pool | ASPP(多尺度空洞) | 嵌入层+分类头 |
| **Dropout** | 0.5(FC层) | 无 | 无 | 0.5(嵌入层) |
| **权重初始化** | He初始化 | 预训练权重 | 预训练权重 | He初始化 |
| **AMP混合精度** | GradScaler+autocast | GradScaler+autocast | GradScaler+autocast | GradScaler+autocast |
| **num_workers** | min(4,cpu) | — | min(4,cpu) | min(4,cpu) |

**为什么分类和识别用自定义网络，检测和分割用预训练模型？**
- 分类(28×28): 小图自定义CNN足够，从头训练收敛快
- 识别(64×64): 小数据集(400张)，预训练容易过拟合，自定义更可控
- 检测/分割: 结构复杂(ResNet50+FPN/ASPP)，预训练权重提供通用特征，训练快10倍

### 6.3 训练超参数对比

| 超参数 | 图像分类 | 目标检测 | 图像分割 | 人脸识别 | 选择依据 |
|--------|---------|---------|---------|---------|---------|
| **batch_size** | 128 | 2 | 4 | 32 | 检测图大(800px)只能2; 分类图小可128 |
| **learning_rate** | 1e-3 | 5e-3 | 1e-3 | 1e-3 | 微调backbone小LR，新层大LR |
| **epochs** | 50 | 10 | 30 | 50 | 微调轮数少；从头训练需多轮 |
| **weight_decay** | 5e-4 | 5e-4 | 1e-4 | 1e-4 | ImageNet标配5e-4; 小数据用1e-4 |
| **early_stop** | 10 | — | 10 | 10 | CNN收敛快，10轮足够判断 |
| **optimizer** | Adam | SGD | SGD | Adam | 分割/检测论文推荐SGD; 分类用Adam更稳 |
| **scheduler** | Cosine | StepLR | Poly | ReduceLR | 分割用Poly; 检测用Step; 分类用Cosine |
| **max_grad_norm** | 5.0 | 5.0 | 5.0 | 5.0 | CNN梯度比FNN大，5.0更宽松 |
| **use_amp** | True | True | True | True | AMP混合精度(仅GPU有效，速度↑1.5-2x) |
| **num_workers** | min(4,cpu) | — | min(4,cpu) | min(4,cpu) | 多进程数据加载(0=主进程) |
| **image_size** | 28 | 800 | 128 | 64 | 分类用小图; 检测需大图保细节 |

**为什么batch_size差异这么大？**
- 检测(2): 800×1333分辨率，单张图约8MB显存，batch=2即16MB
- 分割(4): 128×128分辨率，合成数据较小
- 分类(128): 28×28灰度图，单张约0.8KB，128张≈100KB
- 识别(32): 64×64灰度图，单张约4KB

### 6.4 损失函数对比

```
分类 - CrossEntropyLoss:
  输入: logits (batch, num_classes)
  内部: Softmax → -Σ y_i · log(p_i)
  特点: 所有类别概率和=1(互斥)
  本模板: Fashion-MNIST均衡，不需要class_weight

检测 - 内置4项损失:
  1. RPN分类: 前景/背景二分类
  2. RPN回归: 候选框位置修正
  3. ROI分类: 具体类别分类
  4. ROI回归: 最终框位置修正
  总损失 = λ1·L_rpn_cls + λ2·L_rpn_reg + λ3·L_roi_cls + λ4·L_roi_reg
  特点: 模型训练时返回loss字典，无需外部定义criterion

分割 - CrossEntropyLoss (像素级):
  输入: (batch, num_classes, H, W) logits
  目标: (batch, H, W) 类别索引
  ignore_index=255: 边界像素不参与计算
  特点: 每个像素独立分类，一张图有H×W个预测

人脸识别 - CrossEntropyLoss (分类训练):
  训练阶段: 用分类损失训练，使同类嵌入靠近，异类远离
  推理阶段: 不用损失函数，直接计算余弦相似度
  特点: "分类训练 → 嵌入提取"的两阶段策略
```

### 6.5 数据预处理差异

```
分类:
  图像 → RandomCrop(填充+裁剪) → RandomHFlip → ColorJitter
       → ToTensor → Normalize(Fashion-MNIST均值/标准差)
  标签: 整数(0-9)
  特殊: 训练增强/测试不增强; 标准化参数来自数据集统计

检测:
  图像 → Resize(800×1333) → ToTensor → Normalize(ImageNet均值/标准差)
  标注: {boxes: [[x1,y1,x2,y2],...], labels: [1,2,...]}
  特殊: 标注必须与图像同步变换; collate_fn需自定义(不同尺寸)

分割:
  图像+标注 → RandomCrop(同步!) → RandomHFlip(同步!) → ToTensor → Normalize
  标注: (H, W) 每个像素类别索引，边界=255(忽略)
  特殊: 图像和标注必须同步变换! 标注用NEAREST插值(不能双线性); 合成数据无需下载

人脸识别:
  图像 → ToTensor → 归一化到[0,1]
  标签: 整数身份ID(0-39)
  特殊: 人脸已预先裁剪对齐; 灰度图(1通道); 合成数据(400张)
```

### 6.6 输出格式对比

```python
# 分类: (batch, num_classes) logits
outputs = model(images)               # shape: (128, 10)
probabilities = softmax(outputs, 1)   # shape: (128, 10)
preds = outputs.argmax(dim=1)         # shape: (128,)

# 检测: 字典，多个检测结果
outputs = model([image])[0]
# {
#   "boxes":   tensor [[x1,y1,x2,y2], ...],   # N个边界框
#   "labels":  tensor [3, 1, 8, ...],          # N个类别
#   "scores":  tensor [0.95, 0.87, 0.72, ...]  # N个置信度
# }

# 分割: (batch, num_classes, H, W) logits
outputs = model(images)["out"]        # shape: (4, 21, 520, 520)
pred_masks = outputs.argmax(dim=1)    # shape: (4, 520, 520)

# 人脸识别: (batch, embedding_dim) 嵌入向量
logits, embeddings = model(images)    # 训练: 返回分类+嵌入
embeddings = model.get_embedding(img) # 推理: 只返回嵌入 (1, 128)
similarity = cosine_similarity(emb1, emb2)  # 标量，范围[-1, 1]
```

### 6.7 评估指标对比

| 指标 | 图像分类 | 目标检测 | 图像分割 | 人脸识别 |
|------|---------|---------|---------|---------|
| **主要指标** | Accuracy | mAP | mIoU | 验证准确率 |
| **辅助指标** | F1, 混淆矩阵 | AP_per_class | Pixel Acc | 辨识Top-K, FAR/FRR |
| **指标含义** | 预测正确的比例 | 精度-召回曲线下面积 | 交并比的平均 | 相似度>阈值的判断准确率 |

**mAP (检测)**:
```
AP = 精度-召回曲线下面积(单个类别)
mAP = 所有类别AP的平均值
IoU阈值: 0.5(AP50), 0.75(AP75), 0.5:0.95(AP)
```

**mIoU (分割)**:
```
IoU_c = |预测∩真实|_c / |预测∪真实|_c   (类别c的交并比)
mIoU = 所有类别IoU的平均值

为什么不用像素准确率？
背景可能占60%+，全预测背景也有60%准确率
mIoU对每个类别单独评估，小类别也能体现
```

**FAR/FRR (人脸)**:
```
FAR (False Accept Rate): 不同人被判为同一人的比例(误识率)
FRR (False Reject Rate): 同一人被判为不同人的比例(拒识率)
阈值↓ → FAR↑, FRR↓ (宽松: 少拒认，多误识)
阈值↑ → FAR↓, FRR↑ (严格: 少误识，多拒认)
```

### 6.8 CONFIG参数全表对比

| 参数 | 分类 | 检测 | 分割 | 人脸识别 | 说明 |
|------|------|------|------|---------|------|
| `num_classes` | 10 | 81(含背景) | 5(含背景) | — | 类别数(检测/分割+1背景) |
| `num_identities` | — | — | — | 40 | 身份数量(人脸识别) |
| `image_size` | 28 | 800 | 128 | 64 | 输入图像尺寸 |
| `in_channels` | 1 | 3 | 3 | 1 | 输入通道(RGB=3,灰度=1) |
| `conv_channels` | [32,64,128] | — | — | [32,64,128,256] | 卷积通道数 |
| `embedding_dim` | — | — | — | 128 | 嵌入向量维度 |
| `fc_dims` | [256] | — | — | — | 全连接层维度 |
| `dropout_rate` | 0.5 | — | — | 0.5 | Dropout比例 |
| `batch_size` | 128 | 2 | 4 | 32 | 批次大小 |
| `learning_rate` | 1e-3 | 5e-3 | 1e-3 | 1e-3 | 初始学习率 |
| `epochs` | 50 | 10 | 30 | 50 | 最大训练轮数 |
| `weight_decay` | 5e-4 | 5e-4 | 1e-4 | 1e-4 | L2正则化强度 |
| `early_stop_patience` | 10 | — | 10 | 10 | 早停耐心值 |
| `scheduler_type` | cosine | step | poly | reduce_lr | 调度器类型 |
| `max_grad_norm` | 5.0 | 5.0 | 5.0 | 5.0 | 梯度裁剪阈值 |
| `optimizer` | Adam | SGD | SGD | Adam | 优化器 |
| `use_amp` | True | True | True | True | AMP混合精度(仅GPU有效) |
| `num_workers` | min(4,cpu) | — | min(4,cpu) | min(4,cpu) | 数据加载进程数 |
| `pretrained` | — | True | True | — | 是否用预训练 |
| `use_augmentation` | True | — | True | — | 数据增强 |
| `confidence_threshold` | — | 0.5 | — | — | 检测置信度阈值 |
| `nms_threshold` | — | 0.5 | — | — | NMS IoU阈值 |
| `ignore_index` | — | — | 255 | — | 分割忽略像素值 |
| `similarity_threshold` | — | — | — | 0.5 | 人脸验证阈值 |
| `top_k` | — | — | — | 3 | 人脸辨识Top-K |

### 6.9 代码逻辑流程对比

```
分类流程:
  get_dataloaders: Fashion-MNIST → 数据增强(训练)/标准化(测试)
                  → stratify划分 → DataLoader(num_workers, persistent_workers)
  train:     CrossEntropyLoss → Adam → CosineAnnealingLR → AMP混合精度 → 梯度裁剪 → 早停
  evaluate:  argmax → Accuracy/F1/混淆矩阵(AMP加速推理)
  predict:   softmax → argmax → 类别名 + 置信度

检测流程:
  get_model: 加载Faster R-CNN (COCO预训练, weights=...COCO_V1)
  detect:    图像 → 模型推理 → 过滤低置信度 → NMS去重
  finetune:  冻结backbone浅层 → SGD分层学习率 → StepLR → AMP混合精度
  特殊:      模型内置损失(4项); collate_fn自定义; 推理/训练模式不同

分割流程:
  get_dataloaders: 合成分割数据 → 同步变换(图像+标注) → DataLoader(num_workers, persistent_workers)
  train:     CrossEntropyLoss(ignore_index=255) → SGD → Poly调度 → AMP混合精度 → 梯度裁剪 → 早停
  evaluate:  argmax → 向量化混淆矩阵(np.add.at) → mIoU/像素准确率
  predict:   模型输出 → argmax → 彩色可视化
  特殊:      图像和标注同步变换; 标注用NEAREST插值; ASPP多尺度空洞卷积; 合成数据无需下载

人脸识别流程:
  load_data: 合成人脸数据 → 划分训练/测试 → DataLoader(num_workers, persistent_workers)
  train:     CrossEntropyLoss → Adam → ReduceLROnPlateau → AMP混合精度 → 梯度裁剪 → 早停
  evaluate:  分类准确率 + 嵌入可视化(t-SNE)
  verify:    嵌入提取 → 余弦相似度 → 阈值判断(同一人/不同人)
  identify:  嵌入提取 → 与人脸库比较(批量DataLoader) → Top-K最相似
  特殊:      训练用分类头,推理只用嵌入层; 人脸库注册/搜索机制; 合成数据无需下载
```

---

## 7. 常见问题与调优

### 7.1 过拟合（训练损失低，验证损失高）

**症状**：训练集准确率高，验证集准确率低

**解决方案**：
```python
# 1. 数据增强（CNN最有效的防过拟合手段）
use_augmentation = True
random_crop_padding = 4          # 增大裁剪范围
random_hflip_prob = 0.5
color_jitter = (0.3, 0.3, 0.3, 0.15)  # 增强颜色抖动

# 2. 增大 Dropout
dropout_rate = 0.3  →  0.5  →  0.6

# 3. 减小网络规模
conv_channels = [64, 128, 256]  →  [32, 64, 128]

# 4. 使用预训练模型(冻结backbone)
for param in model.backbone.parameters():
    param.requires_grad = False  # 只训练新层

# 5. 增大权重衰减
weight_decay = 1e-4  →  5e-4  →  1e-3
```

### 7.2 欠拟合（训练和验证损失都高）

**症状**：训练集和验证集的表现都很差

**解决方案**：
```python
# 1. 加大网络规模
conv_channels = [32, 64]  →  [32, 64, 128, 256]

# 2. 减小 Dropout
dropout_rate = 0.5  →  0.3  →  0.1

# 3. 增加训练轮数
epochs = 30  →  100

# 4. 使用预训练模型
pretrained = True  # 利用COCO/ImageNet预训练权重

# 5. 增大学习率
learning_rate = 1e-4  →  1e-3
```

### 7.3 OOM（显存不足）

**症状**：`CUDA out of memory` 错误

**解决方案**：
```python
# 1. 减小 batch_size
batch_size = 16  →  8  →  4  →  2  →  1

# 2. 减小图像尺寸
image_size = 512  →  384  →  256  # 分类/分割
min_size = 800  →  600  →  400    # 检测

# 3. 使用梯度累积(等效大batch_size)
accumulation_steps = 4
# 每4个batch才更新一次参数，等效batch_size×4

# 4. 使用混合精度训练
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda"):
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 5. 使用更轻量的backbone
ResNet50  →  ResNet18  →  MobileNetV3
```

### 7.4 检测/分割用预训练还是从头训练？

| 情况 | 策略 | 说明 |
|------|------|------|
| 数据少(<1K) + 类似COCO | 冻结backbone，只训练新层 | 最快最好 |
| 数据中等(1K-10K) | 微调所有层，backbone小LR | 平衡精度和速度 |
| 数据大(>10K) + 领域差异大 | 从头训练(或微调) | 医学/遥感等特殊领域 |
| 数据多(>100K) | 从头训练 | 数据足够，无需预训练 |

### 7.5 学习率选择指南

| 学习率 | 适用情况 | 本项目实际使用 |
|--------|----------|---------------|
| 5e-3 | 微调检测模型(新层) | 检测(微调) |
| 1e-3 | CNN+BN从头训练 / 分割微调 | 分类/分割/人脸识别 |
| 5e-4 | ImageNet分类标配 | — |
| 1e-4 | 保守微调 / 小数据集 | — |

> 经验：SGD通常用1e-2~5e-3，Adam通常用1e-4~1e-3

### 7.6 网络规模选择指南

| 图像尺寸 | 数据量 | 推荐 conv_channels | 本项目实际使用 |
|---------|--------|-------------------|---------------|
| 28×28 | <10K | [16, 32] | — |
| 28×28 | 10K-100K | [32, 64, 128] | 分类[32,64,128] |
| 64×64 | <1K | [32, 64, 128, 256] | 人脸[32,64,128,256] |
| 128×128 | <1K | [32, 64, 128, 256] | 分割(ResNet50预训练) |
| 224×224+ | >100K | ResNet18/34/50 | 检测(ResNet50) |

> 经验：小图用自定义CNN，大图用预训练ResNet；通道数逐层加倍

### 7.7 数据增强选择指南

| 任务 | 推荐增强 | 不推荐增强 |
|------|---------|-----------|
| 自然图像分类 | Crop, HFlip, ColorJitter | 垂直翻转(猫倒过来不像猫) |
| 目标检测 | Crop, HFlip, ColorJitter | 必须同步变换标注框！ |
| 图像分割 | Crop, HFlip | 必须同步变换标注掩码！ |
| 手写数字 | Crop, 旋转±15° | HFlip(6变9), 大角度旋转 |
| 人脸识别 | Crop, HFlip | 大变形(破坏人脸结构) |
| 医学影像 | Crop, 旋转, 弹性变形 | ColorJitter(颜色是诊断依据) |

---

## 8. 进阶扩展

### 8.1 经典CNN架构演进

```
LeNet (1998)     → 开山之作，5层卷积
AlexNet (2012)   → 深度学习复兴，8层，ReLU+Dropout+GPU
VGG (2014)       → 用小卷积(3×3)堆叠代替大卷积，16-19层
GoogLeNet (2014) → Inception模块，多尺度并行卷积
ResNet (2015)    → 残差连接，152层+，解决深层网络退化
DenseNet (2017)  → 密集连接，每层连接所有前层
EfficientNet (2019) → 复合缩放(深度+宽度+分辨率)
ConvNeXt (2022)  → 借鉴Transformer设计的纯CNN，性能媲美ViT
```

### 8.2 残差连接 (Residual Connection)

```python
class ResidualBlock(nn.Module):
    """
    残差连接: 输出 = F(x) + x
    解决深层网络梯度消失问题

    为什么有效？
    - 普通网络: 需要学习 H(x) = 期望输出
    - 残差网络: 只需学习 F(x) = H(x) - x (残差)
    - 如果某层没用，F(x)→0，等价于跳过该层(恒等映射)
    - 梯度可以直接通过 x �分支传回，不会消失
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x                        # 保存输入
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity                # 残差连接！
        out = self.relu(out)
        return out
```

### 8.3 迁移学习 (Transfer Learning)

```python
# 方法1: 特征提取(冻结backbone)
model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
for param in model.parameters():
    param.requires_grad = False  # 冻结所有参数
model.fc = nn.Linear(2048, num_classes)  # 只训练新分类头

# 方法2: 微调(分层学习率)
model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
model.fc = nn.Linear(2048, num_classes)
optimizer = optim.SGD([
    {"params": model.conv1.parameters(), "lr": 1e-4},    # 浅层: 小LR
    {"params": model.layer4.parameters(), "lr": 1e-3},   # 深层: 中LR
    {"params": model.fc.parameters(), "lr": 1e-2},       # 新层: 大LR
])

# 何时用哪种？
# 数据少 + 类似ImageNet → 方法1(冻结)
# 数据多 + 类似ImageNet → 方法2(微调)
# 数据多 + 差异大       → 方法2或从头训练
```

### 8.4 检测模型选择

| 模型 | 类型 | 速度 | 精度 | 适用场景 |
|------|------|------|------|---------|
| Faster R-CNN | 两阶段 | 慢 | 高 | 学习原理，高精度需求 |
| SSD | 单阶段 | 中 | 中 | 平衡速度和精度 |
| YOLOv5/v8 | 单阶段 | 快 | 中高 | 实时检测，部署 |
| DETR | Transformer | 慢 | 高 | 研究探索 |

### 8.5 分割模型选择

| 模型 | 特点 | 适用场景 |
|------|------|---------|
| FCN | 最简单，全卷积 | 入门学习 |
| UNet | 编码器-解码器+跳跃连接 | 医学影像(小数据) |
| DeepLabV3 | 空洞卷积+ASPP | 通用场景 |
| DeepLabV3+ | DeepLabV3+解码器 | 更高精度 |
| SegFormer | Transformer | 最新研究 |

### 8.6 人脸识别进阶

```python
# 1. 人脸检测(从图像中找到人脸)
#    推荐: MTCNN, RetinaFace, BlazeFace
#    输出: 人脸边界框 + 5个关键点(眼/鼻/嘴角)

# 2. 人脸对齐(根据关键点仿射变换)
#    将人脸旋转/缩放到标准位置，提高识别准确率

# 3. 更好的损失函数(替代简单分类)
#    ArcFace: 在嵌入空间增加角度margin，同一人更近，不同人更远
#    CosFace: 基于余弦距离的margin
#    Triplet Loss: 锚点+正样本+负样本，拉大正负距离

# 4. 更好的嵌入网络
#    FaceNet (Inception+Triplet Loss)
#    ArcFace (ResNet50+ArcFace Loss)
```

### 8.7 GPU 加速要点

```python
# 1. 数据移到GPU
images = images.to(device)
targets = targets.to(device)

# 2. pin_memory=True 加速CPU→GPU传输
train_loader = DataLoader(..., pin_memory=True)

# 3. num_workers=2-4 并行数据加载
train_loader = DataLoader(..., num_workers=4)

# 4. 混合精度训练(显存减半，速度翻倍)
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda"):
    output = model(input)

# 5. 梯度累积(等效更大batch_size)
if (i + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### 8.8 可复现性

```python
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭自动优化
```

---

## 文件结构

```
cnn/
├── classification.py      # 图像分类模板(Fashion-MNIST, 自定义CNN)
├── detection.py           # 目标检测模板(Faster R-CNN, COCO预训练)
├── segmentation.py        # 图像分割模板(DeepLabV3, 合成分割数据)
├── face_recognition.py    # 人脸识别模板(合成人脸数据, 嵌入网络)
└── CNN指南.md             # 本文档
```

---

> 💡 **提示**：四个模板文件中，分类使用公开数据集(Fashion-MNIST自动下载)，检测使用COCO预训练模型，分割和人脸识别使用合成数据(无需下载)。分类和人脸识别使用自定义CNN(从头训练)，检测和分割使用预训练模型(推理/微调)。所有模板均支持AMP混合精度训练(仅GPU有效)。所有可调参数集中在 `CONFIG` 类中，方便统一管理和实验对比。替换为自己的数据时，修改 `CONFIG` 和数据加载函数即可。
