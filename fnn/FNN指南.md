# FNN 前馈神经网络 完全指南

---

## 目录

1. [基础知识](#1-基础知识)
2. [技术原理](#2-技术原理)
3. [三大任务类型](#3-三大任务类型)
4. [应用场景](#4-应用场景)
5. [使用说明](#5-使用说明)
6. [任务类型对比](#6-任务类型对比)
7. [常见问题与调优](#7-常见问题与调优)
8. [进阶扩展](#8-进阶扩展)

---

## 1. 基础知识

### 1.1 什么是前馈神经网络 (FNN)

前馈神经网络（Feedforward Neural Network，简称 FNN），也叫多层感知机（MLP），是最基础的深度学习模型。

**核心思想**：数据从输入层 → 隐藏层（可以有多个） → 输出层，**单向流动，没有回路**（这就是"前馈"的含义）。

```
输入层       隐藏层1      隐藏层2      输出层
 x1 ──→  ○ ──→  ○ ──→  ○ ──→  y
 x2 ──→  ○ ──→  ○ ──→  ○
 x3 ──→  ○ ──→  ○
 x4 ──→  ○
```

### 1.2 FNN 的关键组成

| 组件 | 作用 | 类比 |
|------|------|------|
| **全连接层 (nn.Linear)** | 学习特征间的线性组合 `y = Wx + b` | 调色板：混合基础颜色 |
| **激活函数 (nn.ReLU)** | 引入非线性，让网络能学习复杂模式 | 非线性变换：直线变曲线 |
| **批归一化 (nn.BatchNorm)** | 稳定训练，加速收敛 | 标准化量杯：统一度量 |
| **Dropout** | 随机丢弃神经元，防止过拟合 | 团队备胎：不依赖单个人 |
| **损失函数** | 衡量预测与真实的差距 | 考试评分标准 |
| **优化器** | 根据梯度更新参数 | 学习策略：如何纠错 |

### 1.3 为什么需要激活函数？

如果只用全连接层（线性层）堆叠，无论多少层，等价于**一个**线性层：

```
Linear(线性) + Linear(线性) = Linear(线性)
y = W2(W1x + b1) + b2 = W2·W1·x + (W2·b1 + b2) = W'x + b'
```

加入 ReLU 后，网络才能学习非线性关系：

```
Linear + ReLU + Linear = 非线性
f(x) = W2·relu(W1·x + b1) + b2  ≠  W'x + b'
```

### 1.4 训练流程（5步循环）

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
└──────────────────────────────────────────────┘
```

---

## 2. 技术原理

### 2.1 前向传播

数据从输入层逐层传递到输出层的过程：

```python
def forward(self, x):
    x = self.relu(self.fc1(x))   # 输入 → 隐藏层1 → ReLU
    x = self.relu(self.fc2(x))   # 隐藏层1 → 隐藏层2 → ReLU
    x = self.fc3(x)              # 隐藏层2 → 输出层
    return x
```

数学表达：

```
h1 = ReLU(W1 · x + b1)
h2 = ReLU(W2 · h1 + b2)
output = W3 · h2 + b3
```

### 2.2 反向传播

根据损失函数，从输出层向输入层逐层计算梯度（链式法则），指导参数更新方向：

```
∂Loss/∂W3 → ∂Loss/∂W2 → ∂Loss/∂W1
(输出层)     (隐藏层2)    (隐藏层1)
```

### 2.3 梯度下降与优化器

| 优化器 | 特点 | 适用场景 |
|--------|------|----------|
| **SGD** | 最基础，需要手动调学习率 | 研究用途 |
| **SGD+Momentum** | 加入动量，加速收敛 | 凸优化问题 |
| **Adam** ⭐ | 自适应学习率，综合Momentum和RMSProp | **大多数任务首选** |
| **AdamW** | Adam + 权重衰减修正 | 大模型训练 |

**本模板默认使用 Adam**，这是工业界最常用的优化器。

### 2.4 损失函数选择

| 任务类型 | 损失函数 | 说明 |
|----------|----------|------|
| 多分类 | `CrossEntropyLoss` | 内含 Softmax，模型输出层**不加**激活函数 |
| 回归 | `MSELoss` | 均方误差，对大误差惩罚更大 |
| 回归(异常值多) | `L1Loss` 或 `SmoothL1Loss` | 对异常值更鲁棒 |
| 多标签 | `BCEWithLogitsLoss` | 内含 Sigmoid，数值稳定版 |

> ⚠️ **关键**：`CrossEntropyLoss` 和 `BCEWithLogitsLoss` 内置了激活函数，模型输出层不需要再加 Softmax/Sigmoid，否则会导致梯度计算错误。

### 2.5 批归一化 (BatchNorm) 原理

```
原始数据:  [100, 0.01, 50000]  ← 量纲差异巨大
标准化后:  [0.5, -0.3, 1.2]   ← 均值≈0，标准差≈1
```

**为什么需要？**

- 不同特征量纲差异大（年龄 0-100 vs 收入 0-1000000）
- 不标准化 → 梯度偏向大数值特征 → 训练不稳定
- 标准化后 → 梯度分布均匀 → 收敛更快

### 2.6 Dropout 原理

```
训练时：随机丢弃 30% 的神经元
  ○ → ○      ○ → ✗
  ○ → ✗  →   ○ → ○
  ○ → ○      ○ → ✗

评估时：所有神经元都工作，输出乘以 (1 - dropout_rate)
```

**为什么有效？** 防止神经元"偷懒"（co-adaptation），迫使每个神经元独立学习有用特征。

### 2.7 早停机制 (Early Stopping)

```
Epoch  训练损失  验证损失   状态
  1     1.200    1.180    ↓ 好转
  2     0.900    0.850    ↓ 好转
  ...
  20    0.150    0.200    ↑ 过拟合开始
  21    0.120    0.210    ↑ 继续
  ...
  30    0.050    0.250    ✗ 触发早停！
```

**原理**：监控验证损失，连续 `patience` 轮不下降则停止，恢复最佳模型。

---

## 3. 三大任务类型

### 3.1 分类任务 (Classification)

**目标**：预测样本属于哪个离散类别（互斥）

```
输入: 房屋特征 → FNN → 输出: [0.1, 0.7, 0.2] → 预测: 类别1
                       (3个类别的概率)          (取最大值)
```

**输出层**：`num_classes` 个神经元，不加激活函数
**损失函数**：`CrossEntropyLoss`（内含 Softmax）
**标签类型**：`torch.long`（整数）

### 3.2 回归任务 (Regression)

**目标**：预测连续数值

```
输入: 房屋特征 → FNN → 输出: 25.3 → 预测: 房价25.3万
                       (1个数值，无限制范围)
```

**输出层**：1 个神经元，**不加激活函数**（输出可以是任意实数）
**损失函数**：`MSELoss`
**标签类型**：`torch.float32`（浮点数）

**特有处理**：标签也需要标准化（`y_scaler`），预测后需反标准化还原真实值。

### 3.3 多标签分类 (Multi-Label Classification)

**目标**：预测样本同时属于哪些类别（不互斥）

```
输入: 新闻文本 → FNN → 输出: [2.1, -0.5, 1.8, -1.2, 0.3]
                           ↓ Sigmoid
                       概率: [0.89, 0.38, 0.86, 0.23, 0.57]
                           ↓ 阈值0.5
                       预测: [ 1,   0,   1,   0,   1  ]
                       标签: 体育    国际   财经
```

**输出层**：`num_labels` 个神经元，不加激活函数（`BCEWithLogitsLoss` 内含 Sigmoid）
**损失函数**：`BCEWithLogitsLoss`
**标签类型**：`torch.float32`（二维 0/1 矩阵）

---

## 4. 应用场景

### 4.1 分类任务应用

| 场景 | 输入特征 | 类别数 | 数据示例 |
|------|----------|--------|----------|
| 鸢尾花分类 | 花萼/花瓣长度宽度 | 3 | sklearn iris |
| 手写数字识别 | 28×28 像素 | 10 | MNIST |
| 客户流失预测 | 消费记录、行为数据 | 2 | 流失/不流失 |
| 新闻主题分类 | 文本 TF-IDF 向量 | 多类 | 政治/体育/科技 |
| 疾病诊断 | 体检指标 | 多类 | 健康/轻症/重症 |

### 4.2 回归任务应用

| 场景 | 输入特征 | 预测目标 | 数据示例 |
|------|----------|----------|----------|
| 房价预测 | 面积、位置、房龄 | 价格 | California Housing |
| 销量预测 | 历史销量、促销力度 | 销售数量 | 电商数据 |
| 气温预测 | 历史温度、气压、湿度 | 温度值 | 气象数据 |
| 能耗预测 | 时间、天气、设备状态 | 用电量 | 智能电网 |
| 股票预测 | 历史价格、成交量 | 价格 | 金融数据 |

### 4.3 多标签分类应用

| 场景 | 输入特征 | 标签 | 说明 |
|------|----------|------|------|
| 新闻标签 | 文本向量 | 体育/国际/财经/娱乐 | 一篇新闻可有多个标签 |
| 电影类型 | 剧情描述向量 | 动作/喜剧/科幻/爱情 | 一部电影多种类型 |
| 医学诊断 | 检验指标 | 高血压/糖尿病/心脏病 | 病人可同时患多种病 |
| 图片标签 | 图像特征向量 | 猫/狗/草地/蓝天 | 图片可有多个物体 |
| 文本情感 | 文本向量 | 愤怒/失望/焦虑/悲伤 | 文本可表达多种情绪 |

---

## 5. 使用说明

### 5.1 快速开始

```bash
# 进入 fnn 目录
cd fnn/

# 运行分类模板（使用模拟数据，可直接运行）
python classification.py

# 运行回归模板
python regression.py

# 运行多标签分类模板
python multilabel.py
```

### 5.2 使用自己的数据

只需修改 `load_data()` 函数中 `方式2` 的部分：

**分类任务**：
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("your_data.csv")
X = df.iloc[:, :-1].values       # 前 N-1 列为特征
y = df.iloc[:, -1].values        # 最后一列为标签

# 如果标签是字符串，需要编码为数字
le = LabelEncoder()
y = le.fit_transform(y)          # ["cat","dog","cat"] → [0, 1, 0]
```

**回归任务**：
```python
import pandas as pd

df = pd.read_csv("your_data.csv")
X = df.iloc[:, :-1].values       # 前 N-1 列为特征
y = df.iloc[:, -1].values        # 最后一列为连续目标值
# 标签会自动标准化，无需手动处理
```

**多标签任务**：
```python
import pandas as pd

df = pd.read_csv("your_data.csv")
X = df.iloc[:, :num_features].values    # 前 N 列为特征
y = df.iloc[:, num_features:].values    # 后 M 列为标签（0/1）

# 如果标签是文本列表，用 MultiLabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y_text_list)      # [["体育","国际"]] → [[1,1,0,0,0]]
```

### 5.3 修改超参数

修改各文件中的 `CONFIG` 类：

```python
class CONFIG:
    num_features = 20        # ← 改为你的特征数
    num_classes = 5          # ← 改为你的类别数（分类）
    num_labels = 8           # ← 改为你的标签数（多标签）
    hidden_dims = [256, 128] # ← 加宽/加深网络
    dropout_rate = 0.3       # ← 调整防过拟合强度
    learning_rate = 0.001    # ← 调整学习率
    epochs = 200             # ← 增加训练轮数
    batch_size = 64          # ← 加大batch（GPU显存够的话）
    patience = 15            # ← 早停耐心值
```

### 5.4 模型保存与加载

所有模板都包含模型保存/加载代码：

```python
# 保存模型
torch.save(model.state_dict(), "model.pth")

# 加载模型（必须确保模型结构一致）
loaded_model = FNNClassifier(input_dim=20, hidden_dims=[128, 64], num_classes=3)
loaded_model.load_state_dict(torch.load("model.pth", weights_only=True))
loaded_model.eval()
```

### 5.5 对新数据预测

```python
# 分类
new_data_scaled = feature_scaler.transform(new_data)
new_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    outputs = loaded_model(new_tensor)
    preds = torch.argmax(outputs, dim=1)

# 回归（需要反标准化）
with torch.no_grad():
    pred_scaled = loaded_model(new_tensor).cpu().numpy()
    pred_real = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

# 多标签（Sigmoid + 阈值）
with torch.no_grad():
    logits = loaded_model(new_tensor)
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
```

---

## 6. 任务类型对比

### 6.1 核心差异一览

| 对比项 | 分类 | 回归 | 多标签分类 |
|--------|------|------|------------|
| **预测目标** | 离散类别 | 连续数值 | 多个0/1标签 |
| **输出层神经元** | `num_classes` | 1 | `num_labels` |
| **输出层激活** | 无（CrossEntropyLoss内置Softmax） | 无（输出任意实数） | 无（BCEWithLogitsLoss内置Sigmoid） |
| **损失函数** | `CrossEntropyLoss` | `MSELoss` | `BCEWithLogitsLoss` |
| **标签 dtype** | `torch.long` | `torch.float32` | `torch.float32` |
| **标签 shape** | `(n,)` | `(n,)` | `(n, num_labels)` |
| **标签标准化** | 不需要 | **需要** (`y_scaler`) | 不需要 |
| **预测方式** | `argmax` | 直接输出 | `sigmoid` + 阈值 |
| **评估指标** | 准确率、F1 | MSE、RMSE、MAE、R² | Hamming Loss、F1 |

### 6.2 损失函数对比

```
分类 - CrossEntropyLoss:
  输入: logits (未激活的输出)
  内部: Softmax(logits) → -Σ y_i · log(p_i)
  特点: 所有类别概率和=1（互斥）

回归 - MSELoss:
  公式: mean((pred - true)²)
  特点: 对大误差惩罚更大

多标签 - BCEWithLogitsLoss:
  输入: logits (未激活的输出)
  内部: Sigmoid(logits) → -Σ [y·log(σ) + (1-y)·log(1-σ)]
  特点: 每个标签独立计算概率（不互斥）
```

### 6.3 预测结果对比

```python
# 分类：取概率最大的类别
outputs = model(x)                        # shape: (n, num_classes)
preds = torch.argmax(outputs, dim=1)      # shape: (n,)，每个值是类别索引

# 回归：直接输出数值
outputs = model(x).squeeze(-1)            # shape: (n,)，每个值是预测数值
preds_real = y_scaler.inverse_transform(...)  # 反标准化

# 多标签：Sigmoid + 阈值
outputs = model(x)                        # shape: (n, num_labels)
probs = torch.sigmoid(outputs)            # 每个标签的概率
preds = (probs > 0.5).float()             # 每个标签独立判断 0/1
```

### 6.4 数据格式对比

```python
# 分类标签 (1维整数)
y = [0, 2, 1, 0, 2]  # 类别索引

# 回归标签 (1维浮点)
y = [25.3, 18.7, 42.1, 15.0, 38.5]  # 连续数值

# 多标签标签 (2维0/1矩阵)
y = [[1, 0, 1, 0, 0],   # 样本1: 有标签0和2
     [0, 1, 0, 1, 1],   # 样本2: 有标签1、3、4
     [0, 0, 0, 0, 0]]   # 样本3: 无标签
```

---

## 7. 常见问题与调优

### 7.1 过拟合（训练损失低，验证损失高）

**症状**：训练集准确率高，测试集准确率低

**解决方案**：
```python
# 1. 增大 Dropout
dropout_rate = 0.3  →  0.5

# 2. 减小网络规模
hidden_dims = [256, 128]  →  [64, 32]

# 3. 增加数据量（数据增强、采集更多数据）

# 4. 使用 L2 正则化（权重衰减）
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 5. 降低早停耐心值
patience = 15  →  8
```

### 7.2 欠拟合（训练和验证损失都高）

**症状**：训练集和测试集的表现都很差

**解决方案**：
```python
# 1. 加大网络规模
hidden_dims = [64, 32]  →  [256, 128, 64]

# 2. 减小 Dropout
dropout_rate = 0.5  →  0.2

# 3. 增加训练轮数
epochs = 50  →  200

# 4. 调整学习率
learning_rate = 0.001  →  0.0001  # 或 0.01

# 5. 检查数据质量（是否有噪声、缺失值）
```

### 7.3 训练不稳定（损失忽高忽低）

**解决方案**：
```python
# 1. 减小学习率
learning_rate = 0.001  →  0.0001

# 2. 减小 batch_size
batch_size = 128  →  32

# 3. 使用 BatchNorm（模板已包含）

# 4. 使用梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 7.4 标签不平衡

**分类任务**：
```python
# 方法1: 设置类别权重
class_counts = np.bincount(y_train)
weights = 1.0 / class_counts
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 方法2: 分层采样
from torch.utils.data import WeightedRandomSampler
sample_weights = 1.0 / class_counts[y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = DataLoader(dataset, sampler=sampler)
```

**多标签任务**：
```python
# 设置 pos_weight (模板已自动计算)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
```

### 7.5 学习率选择指南

| 学习率 | 适用情况 |
|--------|----------|
| 0.01 | 数据简单、模型小 |
| 0.001 ⭐ | **大多数情况的首选** |
| 0.0001 | 数据复杂、模型大、微调 |
| 0.00001 | 精细微调 |

### 7.6 网络规模选择指南

| 数据量 | 推荐 hidden_dims |
|--------|------------------|
| < 1K | `[32, 16]` |
| 1K - 10K | `[128, 64]` |
| 10K - 100K | `[256, 128, 64]` |
| > 100K | `[512, 256, 128, 64]` |

> 经验法则：网络参数量 ≈ 数据量的 1/10 ~ 1 倍

---

## 8. 进阶扩展

### 8.1 学习率调度器

```python
# 1. ReduceLROnPlateau（模板已使用，推荐）
#    验证损失停滞时自动降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

# 2. CosineAnnealingLR（余弦退火）
#    学习率按余弦曲线衰减，适合固定轮数训练
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 3. OneCycleLR（超收敛）
#    先升后降，训练速度快
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, total_steps=epochs * len(train_loader)
)
```

### 8.2 多目标回归

```python
# 预测多个连续值（如同时预测房价和租金）
class FNNMultiTargetRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_targets):
        super().__init__()
        # ... 隐藏层同上 ...
        self.fc_out = nn.Linear(prev_dim, num_targets)  # 输出层=num_targets个神经元

    def forward(self, x):
        return self.network(x)  # shape: (batch_size, num_targets)
```

### 8.3 残差连接 (Residual Connection)

```python
class FNNWithResidual(nn.Module):
    """加入残差连接，解决深层网络梯度消失问题"""

    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

        # 如果输入维度和隐藏维度不同，需要投影层
        self.shortcut = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)           # 残差（跳跃连接）
        out = self.relu(self.bn(self.fc1(x)))
        out = self.relu(self.bn(self.fc2(out)))
        out = out + identity                  # 加上残差！
        out = self.fc3(out)
        return out
```

### 8.4 GPU 加速要点

```python
# 两步实现 CUDA 加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 模型移到 GPU
model = FNN().to(device)

# 2. 数据移到 GPU（在训练循环中）
for batch_x, batch_y in train_loader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    # ... 训练 ...
```

### 8.5 可复现性

```python
# 设置所有随机种子，保证结果可复现
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
```

---

## 文件结构

```
fnn/
├── classification.py    # 多分类任务模板
├── regression.py        # 回归任务模板
├── multilabel.py        # 多标签分类任务模板
├── FNN指南.md           # 本文档
├── main.py              # 原始学习代码
└── norm.py              # 原始学习代码
```

---

> 💡 **提示**：三个模板文件都使用 `sklearn.make_*` 生成模拟数据，**无需准备任何数据即可直接运行**。替换为自己的数据时，只需修改 `load_data()` 函数。
