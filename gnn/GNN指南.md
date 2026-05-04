# GNN 图神经网络 完全指南

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

### 1.1 什么是图神经网络 (GNN)

图神经网络（Graph Neural Network，简称 GNN），是专门为处理**图结构数据**设计的深度学习模型。

**核心思想**：通过消息传递（Message Passing）机制，让每个节点聚合邻居的信息来更新自己的表示。

```
输入图             消息传递            节点嵌入          预测结果
┌─────┐         ┌───┐
│ A─B │         │ ↓ │  → 聚合 →  A:[0.3,0.7,...]  →  A: 类别1
│ │   │   →     │ ↓ │  邻居     B:[0.8,0.2,...]  →  B: 类别2
│ C─D │         │ ↓ │  信息     C:[0.1,0.9,...]  →  C: 类别1
└─────┘         └───┘           D:[0.6,0.4,...]  →  D: 类别3
```

### 1.2 GNN vs CNN/RNN 的核心区别

| 对比项 | CNN (卷积网络) | RNN (循环网络) | GNN (图网络) |
|--------|---------------|---------------|-------------|
| **数据结构** | 网格(图像像素) | 序列(时间步) | 图(节点+边) |
| **连接模式** | 规则网格 | 链式 | 任意拓扑 |
| **核心操作** | 卷积(局部区域) | 循环(时序依赖) | 消息传递(邻居聚合) |
| **参数共享** | 空间共享 | 时间共享 | 图结构共享 |
| **适用数据** | 图像/视频 | 文本/语音 | 社交网络/分子/知识图谱 |

**为什么需要GNN？**
- CNN处理规则网格(图像像素整齐排列)，但很多数据不是网格
- RNN处理序列(有先后顺序)，但很多数据没有明确的顺序
- 现实中的数据往往是图：社交网络(用户+好友关系)、分子(原子+化学键)、交通网(路口+道路)
- GNN能处理任意拓扑结构的图数据

### 1.3 图数据的基本概念

| 概念 | 说明 | 例子 |
|------|------|------|
| **节点(Node/Vertex)** | 图中的实体 | 用户、论文、原子 |
| **边(Edge)** | 节点间的关系 | 好友、引用、化学键 |
| **节点特征(Node Feature)** | 节点的属性 | 年龄、词袋向量、原子类型 |
| **边特征(Edge Feature)** | 边的属性 | 关系类型、距离、权重 |
| **邻接矩阵(Adjacency)** | A[i][j]=1表示i和j有边 | N×N的稀疏矩阵 |
| **度(Degree)** | 节点的邻居数 | 好友数量、引用次数 |

### 1.4 GNN的关键组成

| 组件 | 作用 | 类比 |
|------|------|------|
| **消息函数(Message)** | 计算要传递给邻居的信息 | 写信：准备要告诉邻居的内容 |
| **聚合函数(Aggregate)** | 收集所有邻居的消息 | 收信：汇总所有邻居寄来的信 |
| **更新函数(Update)** | 用聚合信息更新自己的表示 | 思考：根据收到的信更新认知 |
| **读出函数(Readout)** | 从节点表示得到图表示 | 投票：汇总所有人的意见 |

### 1.5 训练流程

```
┌──────────────────────────────────────────────┐
│              每个 epoch 重复执行               │
│                                              │
│  1. 消息传递: 每个节点聚合邻居信息             │
│  2. 节点更新: 用聚合信息更新节点表示           │
│  3. 计算损失: 在目标节点/边/图上计算Loss       │
│  4. 反向传播: 计算梯度                        │
│  5. 参数更新: 优化器更新模型参数               │
│                                              │
│  ⚠️ GNN特殊: 通常全图训练(不切batch)          │
└──────────────────────────────────────────────┘
```

---

## 2. 技术原理

### 2.1 消息传递框架 (Message Passing)

GNN的核心是消息传递，可以形式化为3个步骤：

```
1. 消息计算: m_ij = MSG(h_i, h_j, e_ij)   ← 计算邻居j传给i的消息
2. 消息聚合: M_i = AGG({m_ij : j ∈ N(i)})  ← 聚合所有邻居的消息
3. 节点更新: h_i' = UPDATE(h_i, M_i)        ← 更新节点表示
```

其中：
- h_i: 节点i的当前表示
- h_j: 邻居j的表示
- e_ij: 边(i,j)的特征
- N(i): 节点i的邻居集合

### 2.2 三大经典GNN模型

#### GCN (Graph Convolutional Network)

**公式**: H' = σ(D^(-1/2) Â D^(-1/2) H W)

**核心**: 对邻居做归一化平均 + 线性变换

```
GCN的消息传递:
  1. MSG:  m_j = (1/√d_i · 1/√d_j) · h_j · W  ← 归一化权重
  2. AGG:  M_i = Σ m_j                          ← 求和(包含自己)
  3. UPDATE: h_i' = σ(M_i)                       ← 激活

特点: 简单高效，所有邻居权重相同(由图结构决定)
缺点: 无法区分重要/不重要的邻居
```

#### GraphSAGE

**公式**: h' = σ(W · CONCAT(h, AGG(h_N)))

**核心**: 采样固定数量邻居 + 聚合

```
SAGE的消息传递:
  1. 采样: 从邻居中随机采样k个 (解决大图计算问题)
  2. AGG: 可选mean/lstm/pool聚合
  3. UPDATE: h_i' = σ(W · [h_i || AGG(h_N)])  ← 拼接自己+邻居

特点: 支持归纳学习(可预测新节点)，适合大图
```

#### GAT (Graph Attention Network)

**公式**: h' = σ(Σ α_ij W h_j), α_ij = softmax(LeakyReLU(a^T[Wh_i||Wh_j]))

**核心**: 用注意力机制学习邻居权重

```
GAT的消息传递:
  1. 计算注意力: e_ij = a^T[Wh_i || Wh_j]  ← 计算i和j的相关性
  2. softmax归一化: α_ij = softmax(e_ij)     ← 确保权重和为1
  3. 加权聚合: h_i' = σ(Σ α_ij W h_j)       ← 重要邻居权重更大

特点: 精度最高，可解释性强(注意力权重可视化)
缺点: 计算代价大，参数更多
```

### 2.3 过平滑问题 (Over-smoothing)

**问题**: GNN层数越多，节点表示越相似，最终所有节点表示趋同

```
0层: 每个节点特征不同，区分度高
1层: 相邻节点开始相似
2层: 同社区的节点相似
3层: 大部分节点相似
...
N层: 所有节点表示几乎相同(过平滑)
```

**解决方案**:
1. 控制层数: 2-3层通常最优
2. 残差连接: h' = h + GNN(h)
3. DropEdge: 训练时随机删边
4. PairNorm/JKNet: 专门设计的抗过平滑方法

### 2.4 图级别的Readout操作

图分类任务需要将节点表示聚合为图表示：

| Readout | 公式 | 特点 |
|---------|------|------|
| **Mean Pooling** | h_G = (1/N)Σh_i | 简单，但被普通节点稀释 |
| **Max Pooling** | h_G = max(h_1,...,h_N) | 捕捉极端特征 |
| **Attention Pool** | h_G = Σ α_i h_i | 学习节点重要性 |
| **Set2Set** | LSTM迭代聚合 | 最强但最复杂 |
| **虚拟节点** | 额外节点连接所有节点 | GNN自动学习聚合 |

### 2.5 GIN (Graph Isomorphism Network)

**为什么GIN表达力最强？**

WL测试是判断图同构的经典算法，GIN的表达力等于1-WL测试。

**GIN公式**: h'_i = MLP((1+ε) · h_i + Σ_{j∈N(i)} h_j)

- GCN用平均聚合: 两个不同的邻居集合可能产生相同的均值
- GIN用求和+MLP: 求和保留了邻居的多重性，MLP增加了非线性

```
例子: 
邻居集A = {度3, 度1}  → 平均=2, 求和=4
邻居集B = {度2, 度2}  → 平均=2, 求和=4  ← 平均无法区分!
但MLP(4) ≠ MLP(4)? 不对...

实际上GIN的ε参数和MLP可以区分这种情况:
h' = MLP((1+ε)h_self + Σ h_neighbors)
ε放大了自身信息，MLP学习非线性映射
```

---

## 3. 四大任务类型

### 3.1 节点分类 (Node Classification)

**任务**: 给定图中部分节点的标签，预测其余节点的标签

**模板文件**: `node_classification.py`

```
输入: 图 + 部分节点标签
输出: 每个节点的类别

例子: Cora引用网络
  - 2708篇论文(节点)
  - 5429条引用(边)
  - 7个主题类别
  - 只用140篇论文的标签训练，预测其余2568篇

评估: 准确率(Accuracy)
```

**关键特点**:
- 转导学习(Transductive): 训练时能看到所有节点和边
- 全图训练: 不切batch，一次前向/反向用整张图
- 掩码计算损失: 只用训练节点的标签

### 3.2 图分类 (Graph Classification)

**任务**: 给定多个图，预测每个图的整体类别

**模板文件**: `graph_classification.py`

```
输入: 多个图 + 每个图的标签
输出: 每个图的类别

例子: 分子属性预测
  - 每个分子是一个图(原子=节点, 化学键=边)
  - 预测分子的属性(有毒/无毒, 溶解度等)

评估: 准确率(Accuracy), F1-score
```

**关键特点**:
- 归纳学习(Inductive): 可以预测训练时没见过的新图
- 需要Readout操作: 将节点表示聚合为图表示
- DataLoader: 可以用mini-batch训练

### 3.3 链接预测 (Link Prediction)

**任务**: 预测图中两个节点之间是否存在(或将出现)边

**模板文件**: `link_prediction.py`

```
输入: 图 + 部分边
输出: 任意两个节点间存在边的概率

例子: 社交网络好友推荐
  - 已知部分好友关系
  - 预测两个用户是否可能成为好友

评估: AUC-ROC, Average Precision
```

**关键特点**:
- 编码器-解码器架构: GNN编码节点 + 解码器预测边
- 负采样: 生成不存在的边作为负样本
- 二分类: 有边/无边

### 3.4 知识图谱表示学习 (Knowledge Graph)

**任务**: 将实体和关系嵌入低维空间，用于知识推理

**模板文件**: `knowledge_graph.py`

```
输入: 知识图谱三元组 (头实体, 关系, 尾实体)
输出: 实体和关系的向量表示

例子: 知识推理
  - 已知: (北京, 是首都, 中国), (中国, 位于, 亚洲)
  - 推理: (北京, 位于, 亚洲)

评估: MRR, Hits@1/3/10
```

**关键特点**:
- 三元组形式: (h, r, t)而非简单的边
- 多种关系类型: 同一对节点间可以有多种关系
- RotatE模型: 在复数空间用旋转建模关系
- GNN增强: 利用图结构改进实体嵌入

---

## 4. 应用场景

### 4.1 场景-模型匹配

| 应用场景 | 数据特点 | 推荐模型 | 推荐任务模板 |
|---------|---------|---------|------------|
| 社交网络分析 | 用户+好友关系 | GCN/GAT | 节点分类/链接预测 |
| 推荐系统 | 用户+物品+交互 | GraphSAGE | 链接预测 |
| 药物发现 | 原子+化学键 | GIN | 图分类 |
| 知识问答 | 实体+关系三元组 | RotatE | 知识图谱 |
| 交通预测 | 路口+道路 | ST-GNN | 节点回归 |
| 异常检测 | 交易网络 | GAN+GNN | 节点分类 |
| 代码分析 | 函数+调用关系 | GGNN | 图分类 |

### 4.2 实际案例

1. **Pinterest推荐**: 用PinSage(GraphSAGE变体)推荐图片
2. **Uber打车**: 用GNN预测到达时间
3. **阿里推荐**: 用GNN建模用户-商品交互图
4. **DeepMind**: 用GNN预测材料性质
5. **百度**: 用GNN做搜索排序

---

## 5. 使用说明

### 5.1 环境要求

```bash
# 安装PyTorch Geometric
pip install torch_geometric

# 其他依赖(通常随PyG自动安装)
pip install networkx scikit-learn matplotlib
```

### 5.2 快速开始

```bash
# 节点分类(Cora)
python gnn/node_classification.py

# 图分类(合成数据)
python gnn/graph_classification.py

# 链接预测(Cora)
python gnn/link_prediction.py

# 知识图谱(合成数据)
python gnn/knowledge_graph.py
```

### 5.3 适配自己的数据

#### 节点分类

```python
# 1. 准备数据
x = torch.tensor(node_features)       # (N, feature_dim)
edge_index = torch.tensor(edges)       # (2, E)
y = torch.tensor(labels)              # (N,)
train_mask = torch.tensor(train_mask)  # (N,) bool

data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, ...)

# 2. 修改CONFIG
cfg.num_classes = 你的类别数
cfg.hidden_dim = 64  # 根据数据量调整
```

#### 图分类

```python
# 1. 准备多个图
graphs = []
for graph_data in your_dataset:
    data = Data(x=..., edge_index=..., y=...)
    graphs.append(data)

# 2. 用DataLoader
loader = DataLoader(graphs, batch_size=32, shuffle=True)
```

#### 链接预测

```python
# 1. 准备正负边
pos_edge_index = ...  # 真实存在的边
neg_edge_index = ...  # 不存在的边

# 2. 解码器选择
cfg.decoder_type = "mlp"  # 精度高
cfg.decoder_type = "dot"  # 速度快
```

---

## 6. 任务类型对比

| 特征 | 节点分类 | 图分类 | 链接预测 | 知识图谱 |
|------|---------|--------|---------|---------|
| **预测目标** | 节点类别 | 图的类别 | 边是否存在 | 三元组是否为真 |
| **输入** | 1个图 | 多个图 | 1个图 | 多关系图 |
| **输出** | N个类别 | 1个类别 | E个概率 | 三元组分数 |
| **学习模式** | 转导 | 归纳 | 转导 | 归纳 |
| **训练方式** | 全图训练 | Mini-batch | 全图+负采样 | Mini-batch+负采样 |
| **核心操作** | GNN层 | GNN+Readout | GNN+解码器 | 嵌入+评分函数 |
| **评估指标** | Accuracy | Accuracy, F1 | AUC-ROC, AP | MRR, Hits@K |
| **数据需求** | 中(1图) | 多(多图) | 中(1图) | 中(多关系图) |
| **代码文件** | node_classification.py | graph_classification.py | link_prediction.py | knowledge_graph.py |

---

## 7. 常见问题与调优

### 7.1 过平滑 (Over-smoothing)

**症状**: GNN层数>3后，性能反而下降

**解决方案**:
```python
# 1. 减少层数(最简单有效)
cfg.num_layers = 2  # 节点分类
cfg.num_layers = 5  # 图分类(需要更大感受野)

# 2. 增加残差连接
# 在自定义GNN层中: h' = h + GNN(h)

# 3. 使用JKNet (Jumping Knowledge)
# 拼接所有层的输出: h_final = [h_0, h_1, ..., h_L]
```

### 7.2 过拟合

**症状**: 训练集准确率很高，但验证/测试集差

**解决方案**:
```python
# 1. 增大Dropout
cfg.dropout_rate = 0.5  # GNN标准值

# 2. 增大weight_decay
cfg.weight_decay = 5e-4  # L2正则化

# 3. 减小模型容量
cfg.hidden_dim = 32  # 从64减小到32

# 4. 使用标签平滑
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 7.3 训练不稳定

**症状**: 损失震荡，不收敛

**解决方案**:
```python
# 1. 减小学习率
cfg.learning_rate = 0.001  # 从0.01减小

# 2. 使用学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

# 3. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 7.4 大图内存不足

**症状**: OOM(Out of Memory)

**解决方案**:
```python
# 1. 使用GraphSAGE(采样邻居)
cfg.model_type = "sage"

# 2. 使用Mini-batch训练(PyG的NeighborLoader)
from torch_geometric.loader import NeighborLoader

# 3. 减小隐藏层维度
cfg.hidden_dim = 32

# 4. 使用混合精度训练
with torch.amp.autocast("cuda"):
    output = model(x, edge_index)
```

### 7.5 GNN模型选择指南

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 小图(<1万节点) | GAT | 精度最高，计算可接受 |
| 中图(1-10万节点) | GCN | 精度不错，训练快 |
| 大图(>10万节点) | GraphSAGE | 支持采样，可扩展 |
| 图分类 | GIN | 表达力最强 |
| 需要解释性 | GAT | 注意力权重可解释 |
| 归纳学习 | GraphSAGE | 支持新节点预测 |

---

## 8. 进阶扩展

### 8.1 异构图神经网络

现实中的图通常有多种类型的节点和边(异构图):
- 学术图: 论文/作者/会议/关键词 + 写/发表/包含
- 电商图: 用户/商品/店铺/品牌 + 购买/收藏/浏览

**模型**: RGCN, HAN, HGT

### 8.2 图 Transformers

将Transformer的注意力机制应用到图上:
- 不受消息传递的局部性限制
- 可以关注远距离节点
- 代表: Graphormer, GPS

### 8.3 图生成

生成新的图结构(如新分子):
- VAE-based: GraphVAE
- GAN-based: MolGAN
- Autoregressive: GraphRNN, GCPN
- Diffusion: GDSS

### 8.4 时空图神经网络

处理随时间变化的图(如交通流):
- STGCN: 时空图卷积
- DCRNN: 扩散卷积RNN
- ASTGCN: 注意力时空图卷积

### 8.5 图对比学习

无需标签的图预训练:
- GraphCL: 图增强+对比
- SimGRACE: 模型扰动+对比
- BGRL: 双分支自举

### 8.6 推荐阅读

**入门**:
- GNN综述: "A Comprehensive Survey on Graph Neural Networks" (2019)
- PyG官方教程: https://pytorch-geometric.readthedocs.io/

**进阶**:
- GCN原论文: "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017)
- GAT原论文: "Graph Attention Networks" (ICLR 2018)
- GraphSAGE: "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
- GIN: "How Powerful are Graph Neural Networks?" (ICLR 2019)
- RotatE: "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space" (ICLR 2019)
