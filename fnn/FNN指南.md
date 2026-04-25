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
| 多分类 | `CrossEntropyLoss` | 内含 Softmax，模型输出层**不加**激活函数；可设weight处理不平衡 |
| 回归(无异常值) | `MSELoss` | 均方误差，对大误差惩罚更大 |
| 回归(有异常值) | `SmoothL1Loss` ⭐ | 小误差类似MSE，大误差类似MAE，鲁棒抗异常值 |
| 多标签 | `BCEWithLogitsLoss` | 内含 Sigmoid，数值稳定版；可设pos_weight处理不平衡 |

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
**损失函数**：`SmoothL1Loss`（Huber Loss，抗异常值）
**标签类型**：`torch.float32`（浮点数）

**特有处理**：
- 标签需要log变换(右偏数据) + 标准化（`y_scaler`）
- 预测后需双重反变换：`y_scaler.inverse_transform` → `np.exp` 还原真实值
- OneHot编码分类特征时使用 `drop_first=True` 避免多重共线性

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
**损失函数**：`BCEWithLogitsLoss`（可设 `pos_weight` 处理标签不平衡）
**标签类型**：`torch.float32`（二维 0/1 矩阵）

**特有处理**：
- TF-IDF 提取文本特征（`TfidfVectorizer`）
- One-Hot 编码标签列（`pd.get_dummies`）
- 训练/验证使用不同 criterion（训练带 pos_weight，验证不带，避免验证损失不稳定）
- `single_label_mode`：当数据实际是单标签时，用 argmax 而非阈值判定
- 阈值调优：搜索 0.3~0.7 范围内最优阈值（仅多标签模式）

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
# 进入项目根目录
cd py_ai_tech/

# 激活虚拟环境
source venv/bin/activate

# 运行分类模板（红酒品质，1599条数据，3类）
python fnn/classification.py

# 运行回归模板（房价预测，22781条数据，41维特征）
python fnn/regression.py

# 运行多标签分类模板（道德基础新闻，2004条数据，TF-IDF特征）
python fnn/multilabel.py
```

### 5.2 使用自己的数据

只需修改 `CONFIG.datasets_path` 和 `load_data()` 函数：

**分类任务**：
```python
# 1. 修改 CONFIG
class CONFIG:
    datasets_path = "your_data.csv"
    target_col = "label"          # 标签列名或索引(-1=最后一列)
    num_classes = 5               # 类别数
    class_merge = None            # 不需要合并时设为None
    class_names = ["A","B","C","D","E"]

# 2. load_data() 自动处理：
#    - 检测分类列并One-Hot编码
#    - 计算类别权重(如启用)
#    - stratify划分保证类别比例
```

**回归任务**：
```python
# 1. 修改 CONFIG
class CONFIG:
    datasets_path = "your_data.csv"
    target_col = "price"          # 标签列名或索引(-1=最后一列)

# 2. load_data() 自动处理：
#    - 检测分类列并One-Hot编码(drop_first=True)
#    - log变换右偏标签 + y_scaler标准化
#    - 预测时自动反变换
# 注意：如果标签不右偏(偏度<1)，可在load_data中注释掉log变换
```

**多标签任务**：
```python
# 1. 修改 CONFIG
class CONFIG:
    datasets_path = "your_data.csv"
    target_col = "labels"         # 标签列名
    text_col = "text"             # 文本列名(如果用TF-IDF)
    num_labels = 5
    label_names = ["标签1","标签2","标签3","标签4","标签5"]
    single_label_mode = False     # 真正多标签时设为False

# 2. load_data() 自动处理：
#    - TfidfVectorizer提取文本特征
#    - pd.get_dummies将标签列One-Hot编码
#    - 计算pos_weight(sqrt平滑)
# 如果不用TF-IDF(直接数值特征)，需修改load_data跳过TfidfVectorizer
```

### 5.3 修改超参数

修改各文件中的 `CONFIG` 类：

```python
class CONFIG:
    # --- 数据相关 ---
    datasets_path = "your_data.csv"
    num_features = None          # None=自动检测
    target_col = -1              # -1=最后一列，也可指定列名
    test_size = 0.2              # 测试集比例
    random_state = 42            # 随机种子

    # --- 模型相关 ---
    hidden_dims = [256, 128, 64] # 隐藏层维度(漏斗结构)
    dropout_rate = 0.3           # Dropout比例

    # --- 训练相关 ---
    batch_size = 64              # 批次大小
    learning_rate = 3e-4         # 初始学习率
    epochs = 200                 # 最大训练轮数
    weight_decay = 1e-4          # L2正则化强度

    # --- 早停 & LR调度 ---
    early_stop_patience = 20     # 早停耐心值
    lr_factor = 0.5              # LR衰减因子
    lr_patience = 8              # 调度器耐心值
    lr_min = 1e-6                # LR下限

    # --- 梯度裁剪 ---
    max_grad_norm = 1.0          # 梯度L2范数上限
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
# 分类：特征标准化 → argmax
new_data_scaled = feature_scaler.transform(new_data)
new_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    outputs = loaded_model(new_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    preds = torch.argmax(probabilities, dim=1)

# 回归：特征标准化 → 模型输出 → 双重反变换
with torch.no_grad():
    pred_scaled = loaded_model(new_tensor).cpu().numpy()
    pred_log = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    pred_real = np.exp(pred_log)  # 还原log变换

# 多标签：TF-IDF + 标准化 → sigmoid → 阈值/argmax
new_tfidf = vectorizer.transform(new_texts).toarray()  # 用训练时的vectorizer
new_scaled = scaler.transform(new_tfidf)
new_tensor = torch.tensor(new_scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    logits = loaded_model(new_tensor)
    probs = torch.sigmoid(logits)
    if single_label_mode:
        # 单标签模式：argmax选最高概率
        preds = torch.zeros_like(probs)
        max_indices = probs.argmax(dim=1)
        preds.scatter_(1, max_indices.unsqueeze(1), 1.0)
    else:
        # 多标签模式：阈值判定
        preds = (probs > threshold).float()
```

---

## 6. 任务类型对比

### 6.1 核心差异一览

| 对比项 | 分类 | 回归 | 多标签分类 |
|--------|------|------|------------|
| **预测目标** | 离散类别(互斥) | 连续数值 | 多个0/1标签(不互斥) |
| **数据集** | 红酒品质(1599条) | 房价(22781条) | 道德基础新闻(2004条) |
| **输入特征** | 11维化学指标(数值) | 41维(数值+OneHot) | 500维(TF-IDF文本向量) |
| **特征预处理** | StandardScaler | StandardScaler | TfidfVectorizer → StandardScaler |
| **输出层神经元** | `num_classes`(3) | 1 | `num_labels`(4) |
| **输出层激活** | 无(CrossEntropyLoss内置Softmax) | 无(输出任意实数) | 无(BCEWithLogitsLoss内置Sigmoid) |
| **损失函数** | `CrossEntropyLoss`(可加权) | `SmoothL1Loss`(Huber) | `BCEWithLogitsLoss`(可加pos_weight) |
| **标签 dtype** | `torch.long` | `torch.float32` | `torch.float32` |
| **标签 shape** | `(n,)` | `(n,)` | `(n, num_labels)` |
| **标签预处理** | 类别合并(可选) | log变换 + y_scaler标准化 | One-Hot编码 |
| **标签反变换** | 不需要 | y_scaler逆变换 → exp | 不需要 |
| **预测方式** | `argmax`(取最大概率类别) | 直接输出(需反标准化) | `sigmoid`+阈值 或 argmax(单标签模式) |
| **评估指标** | 准确率、F1、混淆矩阵 | MSE、RMSE、MAE、R² | Hamming Loss、F1(micro/macro) |
| **不平衡处理** | class_weight(sqrt平滑) | SmoothL1Loss(抗异常值) | pos_weight(sqrt平滑) |
| **训练/验证criterion** | 相同 | 相同 | **不同**(训练带pos_weight,验证不带) |

### 6.2 网络结构与参数对比

| 对比项 | 分类 | 回归 | 多标签分类 |
|--------|------|------|------------|
| **模型类** | `FNNClassifier` | `FNNRegressor` | `FNNMultiLabel` |
| **hidden_dims** | `[128, 64, 32]` | `[256, 128, 64]` | `[128, 64]` |
| **网络深度** | 3层隐藏层 | 3层隐藏层 | 2层隐藏层 |
| **dropout_rate** | 0.3 | 0.2 | 0.4 |
| **BatchNorm位置** | Pre-Norm(BN→Linear→ReLU) | Pre-Norm | Pre-Norm |
| **权重初始化** | He初始化(Kaiming) | He初始化(Kaiming) | He初始化(Kaiming) |
| **forward输出** | `(batch, num_classes)` logits | `(batch,)` squeeze后 | `(batch, num_labels)` logits |

**为什么网络结构不同？**
- 分类：11维输入，3层[128,64,32]，中等深度适配小数据(1599条)
- 回归：41维输入，3层[256,128,64]，较宽网络适配丰富特征+大数据(22781条)
- 多标签：500维输入，2层[128,64]，浅网络防过拟合(2004条小样本+高维TF-IDF)

**为什么dropout不同？**
- 回归0.2：连续值数据更平滑，不需要太强正则化
- 分类0.3：分类边界更复杂，需要适度正则化
- 多标签0.4：小样本+高维稀疏特征，过拟合风险最高，需要最强正则化

### 6.3 训练超参数对比

| 超参数 | 分类 | 回归 | 多标签分类 | 选择依据 |
|--------|------|------|------------|----------|
| **batch_size** | 32 | 64 | 32 | 小数据集用小batch; 回归数据量大可用64 |
| **learning_rate** | 5e-4 | 3e-4 | 1e-4 | TF-IDF+pos_weight需更保守; 表格数据可用较高LR |
| **epochs** | 200 | 200 | 150 | 均配合早停，此为上限 |
| **weight_decay** | 1e-4 | 1e-4 | 5e-4 | 小样本+高维需更强L2正则化 |
| **early_stop_patience** | 20 | 20 | 20 | 配合LR调度器，需足够耐心 |
| **lr_factor** | 0.5 | 0.5 | 0.5 | 每次LR减半，通用值 |
| **lr_patience** | 8 | 8 | 8 | 连续8轮无改善才降LR |
| **lr_min** | 1e-6 | 1e-6 | 1e-6 | LR下限 |
| **max_grad_norm** | 1.0 | 1.0 | 1.0 | 梯度裁剪阈值，1.0是经验值 |
| **optimizer** | Adam | Adam | Adam | 均使用Adam+weight_decay |

### 6.4 损失函数对比

```
分类 - CrossEntropyLoss:
  输入: logits (未激活的输出)
  内部: Softmax(logits) → -Σ y_i · log(p_i)
  特点: 所有类别概率和=1（互斥）
  不平衡: 可设 weight 参数，给少数类更大权重

回归 - SmoothL1Loss (Huber Loss):
  公式: loss = { 0.5*x²     若 |x| < 1
               { |x| - 0.5  若 |x| ≥ 1
  特点: 小误差时类似MSE(精确收敛)，大误差时类似MAE(鲁棒抗异常值)
  为什么不用MSELoss？房价有极端高价异常值，MSE的平方会放大其影响

多标签 - BCEWithLogitsLoss:
  输入: logits (未激活的输出)
  内部: Sigmoid(logits) → -Σ [y·log(σ) + (1-y)·log(1-σ)]
  特点: 每个标签独立计算概率（不互斥）
  不平衡: 可设 pos_weight 参数，给正样本更大权重
  特殊处理: 训练criterion带pos_weight, 验证criterion不带(防止验证损失不稳定)
```

### 6.5 数据预处理差异

```
分类:
  CSV → 类别合并(可选) → OneHot编码(分类列) → StandardScaler(X)
  标签: 直接用整数(0,1,2)，dtype=long
  特殊: stratify=y 保证各类比例

回归:
  CSV → OneHot编码(分类列, drop_first=True) → StandardScaler(X)
  标签: log变换 → StandardScaler(y) → dtype=float32
  特殊: 标签双重变换(log+标准化)，预测需双重反变换(exp+inverse_transform)

多标签:
  CSV → TfidfVectorizer(文本→向量) → StandardScaler(X)
  标签: One-Hot编码(字符串→0/1矩阵) → dtype=float32
  特殊: TF-IDF提取文本特征；标签从单列字符串→多列0/1矩阵
```

### 6.6 预测结果对比

```python
# 分类：取概率最大的类别
outputs = model(x)                        # shape: (n, num_classes)
_, preds = torch.max(outputs, dim=1)      # shape: (n,)，每个值是类别索引

# 回归：直接输出数值（需反标准化）
outputs = model(x).squeeze(-1)            # shape: (n,)，标准化尺度
preds_log = y_scaler.inverse_transform(...)  # → log尺度
preds_real = np.exp(preds_log)             # → 真实价格

# 多标签：Sigmoid + 阈值（或argmax单标签模式）
outputs = model(x)                        # shape: (n, num_labels)
probs = torch.sigmoid(outputs)            # 每个标签的概率

# 模式1: 多标签模式——阈值判定
preds = (probs > threshold).float()       # 每个标签独立判断 0/1

# 模式2: 单标签模式——argmax（本数据集实际是单标签）
preds = torch.zeros_like(probs)
max_indices = probs.argmax(dim=1)
preds.scatter_(1, max_indices.unsqueeze(1), 1.0)  # 保证恰好1个标签
```

### 6.7 数据格式对比

```python
# 分类标签 (1维整数)
y = [0, 2, 1, 0, 2]  # 类别索引
# dtype: torch.long

# 回归标签 (1维浮点，标准化后)
y = [-0.52, 1.23, -0.11, 0.87, -1.45]  # 标准化后的log房价
# dtype: torch.float32

# 多标签标签 (2维0/1矩阵)
y = [[1, 0, 0, 1],   # 样本1: 有标签0和3
     [0, 1, 0, 0],   # 样本2: 只有标签1
     [0, 0, 1, 0]]   # 样本3: 只有标签2
# dtype: torch.float32
```

### 6.8 不平衡处理策略对比

| 策略 | 分类 | 多标签 |
|------|------|--------|
| **方法** | `CrossEntropyLoss(weight=...)` | `BCEWithLogitsLoss(pos_weight=...)` |
| **权重计算** | `sqrt(N / (C * n_i))` | `sqrt(负样本数 / 正样本数)` |
| **平滑方式** | sqrt平滑 | sqrt平滑 |
| **为何用sqrt** | 原始逆频率太激进，sqrt后更温和 | 同左 |
| **额外选项** | WeightedRandomSampler(本模板未启用) | 单标签模式(argmax) |
| **训练/验证criterion** | 相同 | **不同**(验证不带pos_weight) |

**为什么多标签训练/验证criterion不同？**
- pos_weight改变了损失尺度，少数类损失被放大
- 验证时用带pos_weight的criterion，损失可能从0.7飙升到3.0(表面恶化)
- 但实际F1在改善，导致早停误判
- 解决：验证用无权重的criterion，早停和LR调度器基于真实误差做决策

### 6.9 CONFIG参数全表对比

| 参数 | 分类 | 回归 | 多标签 | 说明 |
|------|------|------|--------|------|
| `datasets_path` | classification-red-winequality.csv | regression-house-clean.csv | multilabel-moral_foundation_news.csv | 数据集路径 |
| `num_features` | None(自动检测→11) | None(自动检测→41) | None(自动检测→500) | 输入特征维度 |
| `num_classes` | 3 | — | — | 分类类别数 |
| `num_labels` | — | — | 4 | 多标签数量 |
| `output_dim` | — | 1 | — | 回归输出维度 |
| `class_merge` | {3:0,4:0,5:1,6:1,7:2,8:2} | — | — | 类别合并规则 |
| `class_names` | ["低品质","中品质","高品质"] | — | — | 类别名称 |
| `label_names` | — | — | ["创新/守旧","非道德",...] | 标签名称 |
| `target_col` | -1 | -1 | "response" | 标签列(支持列名或索引) |
| `text_col` | — | — | "query" | 文本列名 |
| `test_size` | 0.2 | 0.2 | 0.2 | 测试集比例 |
| `random_state` | 42 | 42 | 42 | 随机种子 |
| `hidden_dims` | [128,64,32] | [256,128,64] | [128,64] | 隐藏层维度 |
| `dropout_rate` | 0.3 | 0.2 | 0.4 | Dropout比例 |
| `batch_size` | 32 | 64 | 32 | 批次大小 |
| `learning_rate` | 5e-4 | 3e-4 | 1e-4 | 初始学习率 |
| `epochs` | 200 | 200 | 150 | 最大训练轮数 |
| `weight_decay` | 1e-4 | 1e-4 | 5e-4 | L2正则化强度 |
| `early_stop_patience` | 20 | 20 | 20 | 早停耐心值 |
| `lr_factor` | 0.5 | 0.5 | 0.5 | LR衰减因子 |
| `lr_patience` | 8 | 8 | 8 | 调度器耐心值 |
| `lr_min` | 1e-6 | 1e-6 | 1e-6 | LR下限 |
| `max_grad_norm` | 1.0 | 1.0 | 1.0 | 梯度裁剪阈值 |
| `use_class_weight` | True | — | — | 是否使用类别权重 |
| `use_weighted_sampler` | False | — | — | 是否使用加权采样 |
| `use_pos_weight` | — | — | True | 是否使用pos_weight |
| `single_label_mode` | — | — | True | 单标签argmax模式 |
| `threshold` | — | — | 0.5 | 多标签预测阈值 |
| `criterion` | — | SmoothL1Loss | — | 回归损失函数 |
| `tfidf_max_features` | — | — | 500 | TF-IDF最大特征数 |
| `tfidf_ngram_range` | — | — | (1,2) | n-gram范围 |
| `tfidf_min_df` | — | — | 2 | 最小文档频率 |
| `tfidf_max_df` | — | — | 0.95 | 最大文档频率 |

### 6.10 代码逻辑流程对比

```
分类流程:
  load_data: CSV → 类别合并 → OneHot(分类列) → StandardScaler(X)
             → stratify划分 → torch.long标签 → DataLoader
  train:     CrossEntropyLoss(可加权) → Adam → ReduceLROnPlateau → 梯度裁剪 → 早停
  evaluate:  argmax取预测类别 → 准确率/F1/混淆矩阵
  predict:   softmax → argmax → 类别名 + 置信度

回归流程:
  load_data: CSV → OneHot(分类列,drop_first) → StandardScaler(X)
             → log变换(y) → StandardScaler(y) → torch.float32标签 → DataLoader
  train:     SmoothL1Loss → Adam → ReduceLROnPlateau → 梯度裁剪 → 早停
  evaluate:  y_scaler.inverse_transform → exp → RMSE/MAE/R²
  predict:   输出 → inverse_transform → exp → 真实价格

多标签流程:
  load_data: CSV → TfidfVectorizer(文本) → StandardScaler(X)
             → get_dummies(标签列) → reindex(label_names) → torch.float32标签 → DataLoader
  train:     BCEWithLogitsLoss(训练带pos_weight,验证不带) → Adam → ReduceLROnPlateau
             → 梯度裁剪 → 早停
  evaluate:  sigmoid → 阈值/argmax(单标签模式) → Hamming Loss/F1
  predict:   sigmoid → argmax/阈值 → 标签名 + 概率
  阈值调优:  仅多标签模式(非single_label_mode)时搜索0.3~0.7最优阈值
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
# 方法1: 设置类别权重(模板默认使用，推荐)
#   权重 = sqrt(N / (C * n_i))，sqrt平滑避免过度矫正
#   例：低品质63条→权重≈2.91，中品质1319条→权重≈0.63
weights = []
for c in unique_classes:
    n_c = (y_train == c).sum()
    w_raw = n_total / (n_classes * n_c)
    w = np.sqrt(w_raw)  # sqrt平滑
    weights.append(w)
class_weights = torch.tensor(weights, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# 方法2: 分层采样(模板可选，但不要与class_weight同时使用)
from torch.utils.data import WeightedRandomSampler
sample_weights = 1.0 / class_counts[y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
# 注意：sampler和shuffle互斥！sampler时必须shuffle=False
train_loader = DataLoader(dataset, sampler=sampler, shuffle=False)
```

**多标签任务**：
```python
# 设置 pos_weight (模板自动计算，sqrt平滑)
#   pos_weight[i] = sqrt(负样本数 / 正样本数)
pos_weights = []
for i in range(num_labels):
    pos_count = y_train[:, i].sum()
    neg_count = len(y_train) - pos_count
    pw = np.sqrt(neg_count / pos_count)  # sqrt平滑
    pos_weights.append(pw)
pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)

# 关键：训练和验证使用不同的criterion！
train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)  # 训练带权重
val_criterion = nn.BCEWithLogitsLoss()  # 验证不带权重
# 原因：pos_weight改变了损失尺度，导致验证损失不稳定，早停误判
# 验证用无权重的criterion，早停和LR调度器基于真实误差做决策
```

### 7.5 学习率选择指南

| 学习率 | 适用情况 | 本项目实际使用 |
|--------|----------|----------------|
| 5e-4 | 表格数据+小数据集+类别不平衡 | 分类(红酒1599条) |
| 3e-4 | 表格数据+中等数据集 | 回归(房价22781条) |
| 1e-4 | TF-IDF稀疏特征+pos_weight | 多标签(新闻2004条) |
| 1e-3 | 数据简单、模型小(不推荐) | — |

> 经验：TF-IDF特征+pos_weight时，学习率需要更保守(1e-4)，否则验证损失不稳定

### 7.6 网络规模选择指南

| 数据量 | 推荐 hidden_dims | 本项目实际使用 | 说明 |
|--------|------------------|----------------|------|
| < 1K | `[32, 16]` | — | 极小数据集 |
| 1K - 10K | `[128, 64]` | 分类[128,64,32], 多标签[128,64] | 分类3层因输入维度小(11维) |
| 10K - 100K | `[256, 128, 64]` | 回归[256,128,64] | 中等数据集，3层漏斗 |
| > 100K | `[512, 256, 128, 64]` | — | 大数据集 |

> 经验法则：网络参数量 ≈ 数据量的 1/10 ~ 1 倍；TF-IDF稀疏特征用浅网络防过拟合

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
├── classification.py    # 多分类任务模板(红酒品质)
├── regression.py        # 回归任务模板(房价预测)
├── multilabel.py        # 多标签分类任务模板(道德基础新闻)
└── FNN指南.md           # 本文档

datasets/
├── classification-red-winequality.csv   # 分类数据集(1599条)
├── regression-house-clean.csv           # 回归数据集(22781条)
└── multilabel-moral_foundation_news.csv # 多标签数据集(2004条)
```

---

> 💡 **提示**：三个模板文件均使用真实数据集。替换为自己的数据时，修改 `CONFIG.datasets_path` 和 `load_data()` 函数即可。所有可调参数集中在 `CONFIG` 类中，方便统一管理和实验对比。
