# DNN 深度神经网络 完全指南

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

### 1.1 什么是深度神经网络 (DNN)

深度神经网络（Deep Neural Network，简称 DNN），是由**多层全连接层**堆叠而成的神经网络，是最基础、最经典的神经网络形式。

**核心思想**：通过多层非线性变换，将输入特征逐层映射到输出空间。每一层学习不同抽象级别的特征表示。

```
输入层       隐藏层1      隐藏层2      隐藏层3      输出层
┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
│ x₁  │───→│     │───→│     │───→│     │───→│  ŷ  │
│ x₂  │───→│ h₁  │───→│ h₂  │───→│ h₃  │───→│     │
│ x₃  │───→│     │───→│     │───→│     │───→│     │
│ ... │    │     │    │     │    │     │    │     │
│ xₙ  │───→│     │    │     │    │     │    │     │
└─────┘    └─────┘    └─────┘    └─────┘    └─────┘
  ↓          ↓          ↓          ↓
原始特征   初级特征    中级特征    高级特征    预测结果
```

### 1.2 DNN vs FNN vs CNN

| 对比项 | FNN (浅层前馈) | DNN (深度前馈) | CNN (卷积网络) |
|--------|---------------|---------------|---------------|
| **层数** | 1-2层隐藏层 | 3层+隐藏层 | 卷积+全连接 |
| **连接方式** | 全连接 | 全连接 | 局部连接+参数共享 |
| **参数量** | 少 | 多（逐层累积） | 少（卷积核共享） |
| **特征学习** | 简单线性组合 | 层次化非线性特征 | 空间层次化特征 |
| **适用数据** | 表格数据 | 任何向量数据 | 图像/空间数据 |
| **训练难度** | 容易 | 需要技巧（BN/初始化） | 中等 |

**为什么需要"深度"？**
- 浅层网络（1-2层）只能学习简单的线性决策边界
- 深层网络（3层+）通过多层非线性变换，可以学习任意复杂的函数
- 这就是万能逼近定理的核心：深度带来表达能力

### 1.3 DNN 的关键组成

| 组件 | 作用 | 类比 |
|------|------|------|
| **全连接层 (nn.Linear)** | 学习特征的线性组合权重 | 投票器：综合所有输入投票 |
| **激活函数 (nn.ReLU)** | 引入非线性，让网络能学习复杂模式 | 开关：只让正向信号通过 |
| **批归一化 (nn.BatchNorm1d)** | 稳定训练，加速收敛 | 标准化量杯：统一度量 |
| **Dropout** | 随机丢弃神经元，防止过拟合 | 团队备胎：不依赖单个人 |
| **损失函数** | 衡量预测与真实的差距 | 评分标准：告诉模型错在哪 |

### 1.4 训练流程（5步循环）

```
┌──────────────────────────────────────────────┐
│              每个 batch 重复执行               │
│                                              │
│  1. optimizer.zero_grad()  ← 清零梯度        │
│  2. outputs = model(x)     ← 前向传播        │
│  3. loss = criterion(...)  ← 计算损失        │
│  4. loss.backward()        ← 反向传播(求梯度) │
│  5. optimizer.step()       ← 更新参数        │
│                                              │
│  ⚠️ DNN额外: 梯度裁剪(防止梯度爆炸)            │
└──────────────────────────────────────────────┘
```

---

## 2. 技术原理

### 2.1 全连接层 (Fully Connected Layer)

全连接层是最基础的神经网络组件，每个输入与每个输出都有连接：

```
输入 (3)              权重矩阵 W (3×2)         输出 (2)
┌─────┐              ┌───────────┐           ┌─────┐
│ x₁  │─────────────→│ w₁₁   w₁₂ │           │ h₁  │
│ x₂  │─────────────→│ w₂₁   w₂₂ │  + 偏置 = │ h₂  │
│ x₃  │─────────────→│ w₃₁   w₃₂ │           └─────┘
└─────┘              └───────────┘

计算: h = W·x + b
  h₁ = w₁₁·x₁ + w₂₁·x₂ + w₃₁·x₃ + b₁
  h₂ = w₁₂·x₁ + w₂₂·x₂ + w₃₂·x₃ + b₂
```

**参数量计算**：
- Linear(in, out) 的参数量 = in × out（权重）+ out（偏置）
- 例：Linear(784, 512) = 784×512 + 512 = 401,920 参数

### 2.2 激活函数 (Activation Function)

激活函数引入非线性，让网络能学习复杂模式。

**ReLU (Rectified Linear Unit)**：
```
f(x) = max(0, x)

  f(x)
   │    ╱
 1 ┤   ╱
   │  ╱
 0 ┼─●────────→ x
   │
```

**为什么ReLU最常用？**
- 计算简单：只需比较0，没有指数运算
- 缓解梯度消失：正区间的梯度恒为1
- 稀疏激活：约50%的神经元输出为0，提高效率

**其他激活函数**：

| 激活函数 | 公式 | 特点 | 适用场景 |
|---------|------|------|---------|
| ReLU | max(0,x) | 简单高效，可能"死亡" | 隐藏层默认选择 |
| LeakyReLU | max(αx,x) | 解决ReLU死亡问题 | ReLU死亡时使用 |
| Sigmoid | 1/(1+e⁻ˣ) | 输出[0,1]，梯度小 | 二分类输出层 |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | 输出[-1,1]，零中心化 | RNN隐藏层 |

### 2.3 批归一化 (Batch Normalization)

```
输入: x₁, x₂, ..., xₘ (一个batch的m个样本)

步骤1: 计算batch均值和方差
  μ = (1/m) Σxᵢ
  σ² = (1/m) Σ(xᵢ - μ)²

步骤2: 归一化
  x̂ᵢ = (xᵢ - μ) / √(σ² + ε)

步骤3: 缩放和平移（可学习的参数）
  yᵢ = γ·x̂ᵢ + β

输出: y₁, y₂, ..., yₘ
```

**为什么BN有效？**
1. **稳定训练**：固定每层的输入分布，避免"内部协变量偏移"
2. **允许更大学习率**：归一化后梯度更稳定
3. **轻微正则化效果**：batch统计量有噪声，相当于随机扰动

### 2.4 Dropout 正则化

```
训练时:                    测试时:
┌─────┐                   ┌─────┐
│ h₁  │──●──→              │ h₁  │───→ (权重×0.5)
│ h₂  │──●──→   p=0.5      │ h₂  │───→ (权重×0.5)
│ h₃  │──○──→  (h₃被丢弃)   │ h₃  │───→ (权重×0.5)
│ h₄  │──●──→              │ h₄  │───→ (权重×0.5)
└─────┘                   └─────┘

● = 保留(概率1-p)    ○ = 丢弃(概率p)
```

**原理**：训练时随机"杀死"一部分神经元，迫使网络不依赖任何单个神经元，提升泛化能力。

**注意**：只在训练时使用！测试时所有神经元都参与，但权重乘以(1-p)。

### 2.5 万能逼近定理 (Universal Approximation Theorem)

> "一个具有足够多隐藏层神经元的单隐藏层前馈神经网络，可以以任意精度逼近任意连续函数。"

**直观理解**：
- 每个ReLU神经元 = 一条"折线"
- 多个神经元 = 多条折线叠加
- 足够多的折线 = 可以逼近任意连续曲线

```
单个ReLU:              3个ReLU叠加:           10个ReLU叠加:
  │╱                    │╱╲╱                  │ 波浪线
  │╱                    │╱  ╲                 │ 更平滑
──┼──→ x              ──┼────→ x            ──┼────→ x
```

**为什么实际中需要多层？**
- 单层需要指数级神经元才能逼近复杂函数
- 多层可以用更少参数达到相同表达能力
- 多层结构更符合"层次化特征"的直觉

### 2.6 权重初始化

**为什么需要好的初始化？**
- 权重太大 → 信号放大 → 梯度爆炸
- 权重太小 → 信号衰减 → 梯度消失
- 好的初始化 → 每层信号强度稳定 → 训练顺利

**He初始化（Kaiming Normal）**：
```
W ~ N(0, √(2/fan_in))

其中 fan_in = 输入神经元数量
```

**为什么乘以√2？**
- ReLU会"杀死"约50%的负值信号
- 乘以√2补偿被杀死的信号，保持输出方差≈输入方差

### 2.7 梯度裁剪 (Gradient Clipping)

```
梯度 g = [∂L/∂w₁, ∂L/∂w₂, ...]
梯度范数 ‖g‖ = √(Σ(∂L/∂wᵢ)²)

如果 ‖g‖ > max_norm:
  g = g × (max_norm / ‖g‖)

效果：梯度方向不变，只限制步长
```

**什么时候需要？**
- 深层网络（梯度容易爆炸）
- 循环神经网络（RNN/LSTM）
- 不平衡数据（少数类产生大梯度）

---

## 3. 四大任务类型

### 3.1 图像分类 (Image Classification)

**目标**：将输入图像分类到预定义的类别中

```
输入: 28×28灰度图 → 展平为784维 → DNN → 输出: [0.1, 0.8, 0.05, ...]
                                              (10个类别的概率)
```

**本模板**：MNIST手写数字分类
- 输入：784维（28×28展平）
- 网络：784 → 512 → 256 → 128 → 10
- 输出：10个数字的logits
- 损失：CrossEntropyLoss
- 特点：纯全连接，与CNN对比展示DNN的优缺点

**与CNN的关键区别**：
- DNN展平图像，丢失空间结构
- DNN参数量大（55万 vs CNN的10万）
- DNN没有平移不变性

### 3.2 非线性回归 (Nonlinear Regression)

**目标**：学习从输入到连续数值的复杂映射

```
输入: x ∈ [-1, 1] → DNN → 输出: ŷ (连续值)

真实函数: y = x³ · sin(5πx) + noise
```

**本模板**：复杂函数逼近
- 输入：1维（x坐标）
- 网络：1 → 128 → 64 → 32 → 1
- 输出：1维预测值
- 损失：MSELoss
- 特点：展示万能逼近定理，DNN可以拟合任意连续函数

**为什么输出层不加激活函数？**
- 回归输出可以是任意实数（正数、负数、小数）
- ReLU会截断负数，Sigmoid限制在[0,1]，都不适合

### 3.3 自编码器 (Autoencoder)

**目标**：学习数据的高效压缩表示，并能从压缩表示重建原始数据

```
输入 (784) → 编码器 → 潜在向量 (32) → 解码器 → 重建 (784)
                ↓
            压缩表示
            (特征学习)
```

**本模板**：MNIST降维与重建
- 编码器：784 → 512 → 256 → 32
- 潜在空间：32维（压缩率24.5倍）
- 解码器：32 → 256 → 512 → 784
- 损失：MSELoss（重建误差）
- 特点：无监督学习，不需要标签

**潜在空间的性质**：
- 相似的数据在潜在空间中距离近
- 可以在潜在空间中插值生成过渡样本
- 潜在向量可以作为数据的紧凑特征表示

### 3.4 多任务学习 (Multi-Task Learning)

**目标**：一个网络同时学习多个相关任务，共享底层表示

```
       输入特征
          │
    ┌─────┴─────┐
    ▼           ▼
 共享层1      共享层2
    │           │
    └─────┬─────┘
          ▼
      共享表示
     ┌────┴────┐
     ▼         ▼
  分类头     回归头
     │         │
     ▼         ▼
   类别       数值
```

**本模板**：分类 + 回归
- 共享层：10 → 128 → 64
- 分类头：64 → 32 → 3（CrossEntropyLoss）
- 回归头：64 → 32 → 1（MSELoss）
- 总损失 = 0.5 × L_class + 0.5 × L_reg
- 特点：数据效率更高，泛化更好

**为什么共享层有帮助？**
- 任务之间共享统计强度
- 一个任务的数据帮助另一个任务学习
- 通用表示更难过拟合

---

## 4. 应用场景

### 4.1 图像分类应用

| 场景 | 输入 | 类别数 | 说明 |
|------|------|--------|------|
| MNIST数字识别 | 28×28灰度图 | 10 | 入门经典，本模板使用 |
| 特征向量分类 | 已提取的特征 | 任意 | PCA/LDA后的特征分类 |
| 快速原型验证 | 小图像 | 少量 | 快速验证数据可学习性 |

### 4.2 回归预测应用

| 场景 | 输入 | 输出 | 说明 |
|------|------|------|------|
| 函数逼近 | x坐标 | y值 | 学习未知函数关系 |
| 房价预测 | 房屋特征 | 价格 | 连续数值预测 |
| 物理建模 | 实验参数 | 测量值 | 拟合实验数据 |
| 销量预测 | 历史数据+特征 | 销量 | 时间序列回归 |

### 4.3 自编码器应用

| 场景 | 用途 | 说明 |
|------|------|------|
| 降维可视化 | 高维→2D/3D | 配合t-SNE/UMAP可视化 |
| 特征预训练 | 无监督特征 | 为下游任务提供初始化 |
| 去噪 | 噪声→干净 | 训练时加入噪声 |
| 异常检测 | 重建误差 | 误差大的=异常 |
| 数据压缩 | 存储潜在向量 | 节省存储空间 |

### 4.4 多任务学习应用

| 场景 | 任务1 | 任务2 | 说明 |
|------|-------|-------|------|
| 自动驾驶 | 物体检测 | 距离估计 | 共享视觉特征 |
| 医疗诊断 | 疾病分类 | 严重程度 | 共享症状特征 |
| 推荐系统 | 点击率 | 停留时长 | 共享用户特征 |
| 金融风控 | 违约判断 | 违约金额 | 共享信用特征 |

---

## 5. 使用说明

### 5.1 快速开始

```bash
# 进入项目根目录
cd py_ai_tech/

# 激活虚拟环境
source venv/bin/activate

# 运行图像分类模板（MNIST，纯全连接网络）
python dnn/classification.py

# 运行非线性回归模板（函数逼近，合成数据）
python dnn/regression.py

# 运行自编码器模板（MNIST降维，32维潜在空间）
python dnn/autoencoder.py

# 运行多任务学习模板（分类+回归，合成数据）
python dnn/multi_task.py
```

### 5.2 使用自己的数据

**图像分类（改为自己的图像数据）**：
```python
class CONFIG:
    # 1. 如果输入是特征向量而非图像
    flatten_dim = 100  # 你的特征维度
    num_classes = 5    # 你的类别数
    class_names = ["A", "B", "C", "D", "E"]

# 2. 修改 get_dataloaders() 函数
#    替换 FlattenMNIST 为你自己的 Dataset
#    确保返回展平后的向量 (batch, flatten_dim) 和标签
```

**回归（改为自己的数据）**：
```python
class CONFIG:
    n_samples = 1000  # 你的数据量
    n_features = 5    # 输入特征维度

# 2. 替换 generate_data() 为你自己的数据加载
#    X: (n_samples, n_features)
#    y: (n_samples, 1)
```

**自编码器（改为自己的数据）**：
```python
class CONFIG:
    flatten_dim = 100  # 你的数据维度
    latent_dim = 10    # 想要的压缩维度

# 2. 替换数据加载部分
#    输入数据需要展平为向量 (batch, flatten_dim)
```

**多任务学习（改为自己的数据）**：
```python
class CONFIG:
    n_features = 20
    num_classes = 4
    # 调整任务权重
    class_weight = 0.6  # 分类更重要
    reg_weight = 0.4

# 2. 替换 generate_data()
#    需要同时提供分类标签和回归目标
```

### 5.3 修改超参数

修改各文件中的 `CONFIG` 类：

```python
class CONFIG:
    # --- 数据相关 ---
    test_size = 0.2               # 验证集比例
    random_state = 42             # 随机种子

    # --- 模型相关 ---
    hidden_dims = [512, 256, 128] # 隐藏层维度
    dropout_rate = 0.3            # Dropout比例
    use_batch_norm = True         # 是否使用BN
    latent_dim = 32               # 自编码器潜在维度

    # --- 训练相关 ---
    batch_size = 128              # 批次大小
    learning_rate = 1e-3          # 初始学习率
    epochs = 30                   # 最大训练轮数
    weight_decay = 1e-4           # L2正则化强度

    # --- 早停 & LR调度 ---
    early_stop_patience = 5       # 早停耐心值
    scheduler_type = "cosine"     # 调度器类型

    # --- 梯度裁剪 ---
    max_grad_norm = 1.0           # 梯度L2范数上限

    # --- 多任务 ---
    class_weight = 0.5            # 分类损失权重
    reg_weight = 0.5              # 回归损失权重
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
model = DNNClassifier(cfg).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

### 5.5 对新数据预测

```python
# 图像分类
with torch.no_grad():
    output = model(image_flattened)  # (1, 784)
    prob = torch.softmax(output, dim=1)
    pred = prob.argmax(dim=1)

# 回归
with torch.no_grad():
    pred = model(x_tensor)  # (1, 1)

# 自编码器
with torch.no_grad():
    reconstructed = model(x)  # (1, 784)
    latent = model.encode(x)  # (1, 32)

# 多任务
with torch.no_grad():
    class_logits, reg_pred = model(x)  # (1, 3), (1, 1)
    class_prob = torch.softmax(class_logits, dim=1)
```

---

## 6. 任务类型对比

### 6.1 核心差异一览

| 对比项 | 图像分类 | 非线性回归 | 自编码器 | 多任务学习 |
|--------|---------|-----------|---------|-----------|
| **预测目标** | 离散类别 | 连续数值 | 重建输入 | 类别+数值 |
| **监督类型** | 有监督 | 有监督 | 无监督 | 有监督 |
| **数据集** | MNIST(70K) | 合成(1K) | MNIST(70K) | 合成(2K) |
| **输入维度** | 784 | 1 | 784 | 10 |
| **输出维度** | 10 | 1 | 784 | 3+1 |
| **模型** | DNNClassifier | DNNRegressor | Autoencoder | MultiTaskDNN |
| **隐藏层** | [512,256,128] | [128,64,32] | [512,256] | [128,64] |
| **输出激活** | 无 | 无 | Sigmoid→MSE | 分类无/回归无 |
| **损失函数** | CrossEntropyLoss | MSELoss | MSELoss | CrossEntropy+MSE |
| **核心指标** | Accuracy | MSE/MAE/R² | 重建MSE | Acc+MSE |
| **特殊组件** | 展平输入 | 无 | 编码器+解码器 | 共享层+多任务头 |
| **可视化** | 预测结果+权重 | 拟合曲线+神经元 | 重建+潜在空间 | 混淆矩阵+残差 |

### 6.2 网络结构对比

| 对比项 | 图像分类 | 非线性回归 | 自编码器 | 多任务学习 |
|--------|---------|-----------|---------|-----------|
| **输入** | 784 (MNIST展平) | 1 (x坐标) | 784 (MNIST展平) | 10 (特征) |
| **隐藏层1** | Linear(784→512) | Linear(1→128) | Encoder(784→512) | Shared(10→128) |
| **隐藏层2** | Linear(512→256) | Linear(128→64) | Encoder(512→256) | Shared(128→64) |
| **隐藏层3** | Linear(256→128) | Linear(64→32) | Encoder(256→32) | — |
| **瓶颈/共享** | — | — | Latent(32) | Shared(64) |
| **输出层** | Linear(128→10) | Linear(32→1) | Decoder(32→784) | ClassHead(64→3) + RegHead(64→1) |
| **参数量** | ~55万 | ~1.2万 | ~108万 | ~2.5万 |
| **激活** | ReLU+Dropout | ReLU+Dropout | ReLU | ReLU+Dropout |
| **BN** | 是 | 是 | 是 | 是 |
| **初始化** | He | He | He | He |

### 6.3 训练超参数对比

| 超参数 | 图像分类 | 非线性回归 | 自编码器 | 多任务学习 | 选择依据 |
|--------|---------|-----------|---------|-----------|---------|
| **batch_size** | 128 | 32 | 128 | 64 | 数据量大小 |
| **learning_rate** | 1e-3 | 1e-3 | 1e-3 | 1e-3 | Adam标准值 |
| **epochs** | 30 | 500 | 20 | 200 | 任务复杂度 |
| **weight_decay** | 1e-4 | 1e-5 | 1e-5 | 1e-4 | 过拟合风险 |
| **early_stop** | 5 | 30 | 5 | 20 | 收敛速度 |
| **scheduler** | Cosine | ReduceLR | Cosine | ReduceLR | 损失特性 |
| **max_grad_norm** | 1.0 | 1.0 | 1.0 | 1.0 | 标准值 |
| **dropout** | 0.3 | 0.1 | 0.0 | 0.2 | 网络/数据大小 |
| **use_amp** | True | True | False | True | BCE与AMP冲突 |

### 6.4 损失函数对比

```
分类 - CrossEntropyLoss:
  输入: logits (batch, num_classes)
  内部: Softmax → -Σ y_i · log(p_i)
  特点: 所有类别概率和=1（互斥）
  本模板: MNIST 10类

回归 - MSELoss:
  输入: 预测值 (batch, 1), 目标值 (batch, 1)
  公式: (1/N) Σ(y_pred - y_true)²
  特点: 对大误差惩罚更重
  本模板: 函数逼近

自编码器 - MSELoss:
  输入: 重建 (batch, 784), 原始 (batch, 784)
  公式: 像素级均方误差
  特点: 输入=目标（自监督）
  本模板: MNIST重建

多任务 - 组合损失:
  总损失 = w₁ × CrossEntropy + w₂ × MSE
  特点: 需要平衡两个任务的损失量级
  本模板: 分类(0.5) + 回归(0.5)
```

### 6.5 数据预处理差异

```
图像分类:
  图像 → ToTensor → Normalize(mean=0.1307, std=0.3081) → 展平(784)
  标签: 整数(0-9)
  特殊: 必须展平为1D向量

非线性回归:
  x ∈ [-1, 1]均匀分布
  y = x³ · sin(5πx) + noise
  特殊: 无需标准化，数据本身就是合成的小范围值

自编码器:
  图像 → ToTensor → Normalize → 展平(784)
  目标 = 输入（自编码器的特点）
  特殊: 无监督，不需要标签

多任务学习:
  X ~ N(0,1), 10维
  y_class = argmax(非线性组合)
  y_reg = sin(X·w) + X₀² + noise
  特殊: 同时有分类标签和回归目标
```

### 6.6 输出格式对比

```python
# 分类: (batch, num_classes) logits
outputs = model(images)              # shape: (128, 10)
probabilities = softmax(outputs, 1)  # shape: (128, 10)
preds = outputs.argmax(dim=1)        # shape: (128,)

# 回归: (batch, 1) 连续值
preds = model(x)                     # shape: (32, 1)

# 自编码器: (batch, input_dim) 重建
recon = model(x)                     # shape: (128, 784)
latent = model.encode(x)             # shape: (128, 32)

# 多任务: (class_logits, reg_value)
class_out, reg_out = model(x)        # (32, 3), (32, 1)
```

---

## 7. 常见问题与调优

### 7.1 过拟合（训练损失低，验证损失高）

**症状**：训练集表现好，验证集表现差

**解决方案**：
```python
# 1. 增大 Dropout
dropout_rate = 0.3 → 0.5

# 2. 减小网络规模
hidden_dims = [512, 256, 128] → [256, 128, 64]

# 3. 增大权重衰减
weight_decay = 1e-4 → 1e-3

# 4. 早停
early_stop_patience = 5  # 已经启用

# 5. 减少训练轮数
epochs = 30 → 20
```

### 7.2 欠拟合（训练和验证损失都高）

**症状**：训练集和验证集表现都差

**解决方案**：
```python
# 1. 加大网络规模
hidden_dims = [128, 64] → [512, 256, 128]

# 2. 减小 Dropout
dropout_rate = 0.5 → 0.1

# 3. 增加训练轮数
epochs = 30 → 100

# 4. 增大学习率
learning_rate = 1e-4 → 1e-3
```

### 7.3 梯度问题

**梯度消失**（深层网络常见）：
- 症状：训练损失不下降，或下降极慢
- 解决：使用BN、更好的初始化（He）、减少网络深度

**梯度爆炸**：
- 症状：损失突然变成NaN或极大值
- 解决：梯度裁剪、减小学习率、使用BN

```python
# 梯度裁剪（已在代码中启用）
max_grad_norm = 1.0

# 如果仍爆炸，减小学习率
learning_rate = 1e-3 → 1e-4
```

### 7.4 学习率选择指南

| 学习率 | 适用情况 | 本项目使用 |
|--------|----------|-----------|
| 1e-2 | 大数据+简单任务 | — |
| 1e-3 | 标准值（Adam+BN） | 所有模板 |
| 1e-4 | 保守训练/微调 | — |
| 1e-5 | 非常保守 | — |

> 经验：如果训练不稳定（损失震荡），降低LR；如果收敛太慢，增大LR。

### 7.5 网络规模选择指南

| 输入维度 | 数据量 | 推荐 hidden_dims | 说明 |
|---------|--------|-----------------|------|
| < 50 | < 1K | [64, 32] | 小数据用小网络 |
| 50-500 | 1K-10K | [256, 128, 64] | 中等规模 |
| 500+ | > 10K | [512, 256, 128] | 大图/高维数据 |
| 784 (MNIST) | 60K | [512, 256, 128] | 本分类模板 |

> 经验：总参数量不应超过训练样本数的1/10。

### 7.6 激活函数选择指南

| 场景 | 推荐 | 不推荐 |
|------|------|--------|
| 隐藏层 | ReLU | Sigmoid（梯度消失） |
| 输出-分类 | 无（CrossEntropy内置Softmax） | 手动Softmax |
| 输出-回归 | 无（线性输出） | ReLU/Sigmoid（限制范围） |
| 输出-自编码器[0,1] | Sigmoid | 无（可能超出范围） |
| 输出-自编码器(标准化后) | 无 + MSELoss | Sigmoid + BCELoss |

---

## 8. 进阶扩展

### 8.1 网络架构演进

```
感知机 (1958)     → 单个神经元，线性分类
MLP (1986)       → 多层感知机，反向传播算法
Deep Belief Net   → 预训练+微调策略
ResNet (2015)    → 残差连接，解决深层网络退化
DenseNet (2017)  → 密集连接，特征复用
Transformer (2017) → 注意力机制，取代RNN/CNN
MLP-Mixer (2021) → 纯MLP架构，挑战Transformer
```

### 8.2 残差连接 (Residual Connection)

```python
class ResidualBlock(nn.Module):
    """
    残差连接: 输出 = F(x) + x
    解决深层网络梯度消失问题
    """
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out = out + identity  # 残差连接！
        out = self.relu(out)
        return out
```

### 8.3 自编码器变体

**去噪自编码器 (Denoising Autoencoder)**：
```python
# 训练时加入噪声，学习恢复干净数据
noisy_input = input + 0.3 * torch.randn_like(input)
reconstruction = model(noisy_input)
loss = MSELoss(reconstruction, input)  # 目标是干净数据
```

**变分自编码器 (Variational Autoencoder, VAE)**：
```python
# 潜在空间变为概率分布，可以生成新样本
mu, log_var = encoder(x)
z = mu + exp(0.5*log_var) * eps  # 重参数化技巧
reconstruction = decoder(z)
# 损失 = 重建误差 + KL散度（让潜在分布接近标准正态）
```

### 8.4 多任务学习进阶

**硬参数共享 vs 软参数共享**：
```python
# 硬参数共享（本模板使用）
# 底层完全共享，顶层分叉

# 软参数共享（更灵活）
# 每个任务有自己的网络，但用正则化约束它们相似
loss = task_loss + λ * ||W₁ - W₂||²
```

**不确定性加权 (Uncertainty Weighting)**：
```python
# 自动学习任务权重，无需手动调参
log_sigma1 = nn.Parameter(torch.zeros(1))  # 分类不确定性
log_sigma2 = nn.Parameter(torch.zeros(1))  # 回归不确定性
loss = (1/(2*σ1²)) * L_class + (1/(2*σ2²)) * L_reg + log(σ1) + log(σ2)
```

### 8.5 GPU 加速要点

```python
# 1. 数据移到GPU
inputs = inputs.to(device)

# 2. pin_memory=True 加速CPU→GPU传输
train_loader = DataLoader(..., pin_memory=True)

# 3. 混合精度训练（仅部分任务适用）
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda"):
    output = model(input)

# 4. 注意：BCELoss + Sigmoid 与 AMP 不兼容！
#    改用 BCEWithLogitsLoss 或 MSELoss
```

### 8.6 可复现性

```python
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 文件结构

```
dnn/
├── classification.py      # 图像分类模板 (MNIST, 纯全连接DNN)
├── regression.py          # 非线性回归模板 (函数逼近, 合成数据)
├── autoencoder.py         # 自编码器模板 (MNIST降维, 32维潜在空间)
├── multi_task.py          # 多任务学习模板 (分类+回归, 合成数据)
└── DNN指南.md             # 本文档
```

---

> 💡 **提示**：四个模板中，分类和自编码器使用MNIST数据集（自动下载），回归和多任务使用合成数据（无需下载）。所有模板均支持GPU加速（CUDA）。所有可调参数集中在 `CONFIG` 类中，方便统一管理和实验对比。替换为自己的数据时，修改 `CONFIG` 和数据加载/生成函数即可。
