"""
=============================================================================
GNN 节点分类任务模板 (Graph Neural Network for Node Classification)
=============================================================================

【原理】
图神经网络(GNN)是专门处理**图结构数据**的深度学习模型。与CNN处理图像网格、
RNN处理序列不同，GNN处理的是由节点和边组成的图——社交网络、引用网络、
分子结构等都是典型的图数据。

节点分类是GNN最基础的任务：给定图中部分节点的标签，预测其余节点的标签。
核心思想——消息传递(Message Passing)：
  每个节点聚合邻居的信息 → 更新自己的表示 → 逐层传播

  第0层: 节点只有原始特征 (如论文的词向量)
  第1层: 节点聚合了1跳邻居的信息 (直接引用的论文)
  第2层: 节点聚合了2跳邻居的信息 (间接引用的论文)
  ...    感受野逐层扩大

【三大经典GNN模型对比】
  GCN (Graph Convolutional Network):
    公式: H' = σ(D^(-1/2) Â D^(-1/2) H W)
    特点: 对所有邻居做归一化平均，简单高效
    类比: "民主投票"——每个邻居平等贡献

  GraphSAGE (SAGE = SAmple and aggreGatE):
    公式: h' = σ(W · CONCAT(h, AGG(h_N)))
    特点: 采样固定数量邻居+聚合(均值/LSTM/池化)
    类比: "抽样调查"——从邻居中采样再汇总

  GAT (Graph Attention Network):
    公式: h' = σ(Σ α_ij W h_j), α_ij = attention(h_i, h_j)
    特点: 用注意力机制学习邻居的重要性权重
    类比: "加权投票"——重要邻居的票更值钱

【应用场景】
- 论文主题分类 (Cora/CiteSeer引用网络, 本模板使用)
- 社交网络用户分类 (预测兴趣/职业)
- 推荐系统 (预测用户偏好)
- 交通网络 (预测拥堵节点)
- 蛋白质功能预测

【本数据集: Cora引用网络】
- 2,708篇机器学习论文(节点)
- 5,429条引用关系(边, 无向)
- 7个主题类别: 神经网络/强化学习/遗传算法/概率/贝叶斯/理论/案例
- 每篇论文1,433维词袋特征
- 训练/验证/测试: 140/500/2,068 节点(标准划分)

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python gnn/node_classification.py
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
import torch.nn.functional as F
import torch.optim as optim

# PyTorch Geometric: 图神经网络的核心库
from torch_geometric.nn import GCNConv, GATConv, SAGEConv  # 三种经典GNN层
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid  # 引用网络数据集
from torch_geometric.transforms import NormalizeFeatures  # 特征归一化

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from sklearn.manifold import TSNE  # 降维可视化

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
    #   PyG会自动下载Cora到此目录
    data_dir = "data"

    # dataset_name="Cora": 使用哪个引用网络数据集
    #   可选: "Cora"(7类, 2708节点), "CiteSeer"(6类, 3327节点), "PubMed"(3类, 19717节点)
    #   Cora最经典，节点数适中，适合入门
    dataset_name = "Cora"

    # num_classes=7: Cora有7个主题类别
    #   不需要手动设，会从数据集自动获取
    num_classes = 7

    # class_names: 类别名称(用于可视化)
    class_names = [
        "神经网络", "强化学习", "遗传算法", "概率论",
        "贝叶斯", "理论", "案例",
    ]

    # --- 模型相关 ---
    # model_type="gcn": 使用哪种GNN模型
    #   "gcn": Graph Convolutional Network, 简单高效, 适合入门
    #   "gat": Graph Attention Network, 注意力机制, 精度更高
    #   "sage": GraphSAGE, 采样聚合, 适合大图
    model_type = "gat"

    # hidden_dim=64: 隐藏层维度
    #   为什么64？Cora特征1433维→64维，压缩但保留关键信息
    #   太小(16): 表达能力不足，欠拟合
    #   太大(256): 参数过多，容易过拟合(Cora只有2708节点)
    hidden_dim = 64

    # num_layers=2: GNN层数
    #   为什么2层？GNN的层数≠CNN的层数(越多越好)
    #   GNN存在过平滑问题: 层数越多，节点表示越相似，区分度下降
    #   2-3层是大多数GNN的最优选择
    num_layers = 2

    # dropout_rate=0.5: Dropout比例
    #   GNN的Dropout特别重要：图数据通常很小(Cora只有2708节点)
    #   0.5是GNN论文的标准值(GCN/GAT原论文都用0.5)
    dropout_rate = 0.5

    # gat_heads=8: GAT注意力头数(仅model_type="gat"时生效)
    #   多头注意力: 类似CNN的多通道，每个头关注不同的邻居模式
    #   8头×8维/头=64维隐藏层(与hidden_dim对应)
    #   为什么8？GAT原论文推荐8头
    gat_heads = 8

    # gat_heads_out=1: 输出层注意力头数
    #   最后一层通常用1头，做多类别分类
    gat_heads_out = 1

    # --- 训练相关 ---
    # batch_size: GNN节点分类通常用全图训练(不需要batch)
    #   因为消息传递需要所有邻居信息，不能像CNN那样切batch
    #   这里设为None表示全图训练
    batch_size = None  # 全图训练

    # learning_rate=0.005: 学习率
    #   为什么0.005？GNN的LR通常比CNN大(0.01-0.005)
    #   因为全图训练每个epoch只做1次更新(不是per-batch)
    learning_rate = 0.005

    # weight_decay=5e-4: L2正则化
    #   与CNN相同，防止过拟合
    weight_decay = 5e-4

    # epochs=200: 最大训练轮数
    #   GNN训练很快(Cora全图前向+反向<0.1秒)
    #   200轮通常足够收敛
    epochs = 200

    # --- 早停策略 ---
    # early_stop_patience=20: 验证损失连续20轮不下降就停止
    early_stop_patience = 20

    # --- 保存相关 ---
    save_dir = "gnn/output/node_classification"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 数据加载和预处理
# ============================================================
def load_data(cfg):
    """
    加载Cora引用网络数据集。

    【图数据的基本概念】
    - 节点(Node/Vertex): 图中的实体(如论文)
    - 边(Edge): 节点间的关系(如引用关系)
    - 节点特征(Node Feature): 每个节点的属性(如论文的词袋向量)
    - 邻接矩阵(Adjacency Matrix): A[i][j]=1表示节点i和j有边

    【Cora数据集结构】
    - x: 节点特征矩阵 (2708, 1433) - 每篇论文的词袋向量
    - edge_index: 边列表 (2, 10556) - COO格式的边，每条边存两个端点
      为什么5429条边变成10556？无向边存了两次(i→j 和 j→i)
    - y: 节点标签 (2708,) - 7个类别
    - train_mask: 训练集掩码 (2708,) - True=训练节点
    - val_mask: 验证集掩码 (2708,) - True=验证节点
    - test_mask: 测试集掩码 (2708,) - True=测试节点

    【为什么用掩码(mask)而非DataLoader？】
    - GNN节点分类是**转导学习**(Transductive):
      训练时能看到所有节点和边(包括测试节点的特征和边)
      但只能用训练节点的标签计算损失
    - 这与CNN的归纳学习(Inductive)不同: CNN训练时完全看不到测试数据
    - 原因: 消息传递需要邻居信息，如果删掉测试节点，邻居的消息就不完整了
    """
    # NormalizeFeatures: 对节点特征做L2归一化
    # 为什么？词袋特征量级差异大，归一化后训练更稳定
    dataset = Planetoid(
        root=cfg.data_dir,
        name=cfg.dataset_name,
        transform=NormalizeFeatures(),
    )

    # Planetoid只包含1个图
    data = dataset[0]

    # 更新实际类别数
    cfg.num_classes = dataset.num_classes
    num_nodes = data.num_nodes
    num_edges = data.num_edges // 2  # 无向边存了两次，除以2得到实际边数
    num_features = dataset.num_features

    print(f"数据集: {cfg.dataset_name}")
    print(f"节点数: {num_nodes} | 边数: {num_edges} | 特征维度: {num_features} | 类别数: {cfg.num_classes}")
    print(f"训练集: {data.train_mask.sum().item()} | 验证集: {data.val_mask.sum().item()} | 测试集: {data.test_mask.sum().item()}")

    return data, dataset


# ============================================================
# Step 4: 模型定义
# ============================================================
class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN)

    【核心公式】
    H^(l+1) = σ(D^(-1/2) Â D^(-1/2) H^(l) W^(l))

    拆解来看:
    - H^(l): 第l层的节点表示 (N, d_in)
    - Â = A + I: 邻接矩阵+自环(每个节点也聚合自己的信息)
    - D: 度矩阵, D^(-1/2) Â D^(-1/2) 是对称归一化
    - W^(l): 可学习权重 (d_in, d_out)
    - σ: 激活函数(ReLU)

    【对称归一化做了什么？】
    不归一化: h' = Σ h_j W  → 度大的节点特征值爆炸
    简单平均: h' = (1/|N|) Σ h_j W  → 忽略了邻居的重要性差异
    对称归一化: h' = Σ (1/√d_i · 1/√d_j) h_j W
      → 度大的节点被缩小，度小的节点被放大，保持特征尺度稳定

    【GCN的优缺点】
    优点: 简单高效，半监督学习的经典方法
    缺点: 所有邻居权重相同(无法区分重要/不重要的邻居)
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # 构建GCN层
        self.convs = nn.ModuleList()

        # 第1层: 输入特征 → 隐藏层
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # 中间层: 隐藏层 → 隐藏层
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # 最后一层: 隐藏层 → 输出类别
        self.convs.append(GCNConv(hidden_channels, out_channels))

        # BatchNorm: 稳定中间层的训练
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index):
        """
        前向传播

        参数:
            x: 节点特征 (N, in_channels)
            edge_index: 边索引 (2, E), COO格式

        数据流动:
        x: (2708, 1433)          ← 节点特征(词袋向量)
          → GCN1: (2708, 64)     ← 聚合1跳邻居
          → GCN2: (2708, 7)      ← 聚合2跳邻居+分类
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)  # 图卷积: 聚合邻居+线性变换
            x = self.bns[i](x)       # 批归一化
            x = F.relu(x)            # 非线性激活
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout

        # 最后一层不加BN和ReLU，直接输出logits
        x = self.convs[-1](x, edge_index)
        return x


class GAT(nn.Module):
    """
    Graph Attention Network (GAT)

    【核心思想】
    GCN的问题: 所有邻居权重相同 → 无法区分重要/不重要的邻居
    GAT的解决: 用注意力机制自动学习每个邻居的权重

    【注意力系数计算】
    1. 线性变换: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
       - W: 共享的线性变换
       - a: 注意力向量
       - ||: 拼接操作
    2. softmax归一化: α_ij = softmax_j(e_ij)
       只对邻居j归一化，确保所有邻居权重之和为1

    3. 加权聚合: h'_i = σ(Σ_j α_ij W h_j)

    【多头注意力】
    类似Transformer的多头注意力:
    - 用K组不同的注意力参数，独立计算K组权重
    - 拼接或平均K组结果
    - 好处: 不同头关注不同的邻居模式(如一个头关注同领域论文，另一个关注跨领域)

    【GAT vs GCN】
    - GCN: 邻居权重由图结构决定(归一化系数)，固定不可学习
    - GAT: 邻居权重由注意力学习，可以根据特征动态调整
    - 代价: GAT参数更多，训练更慢，但精度通常更高
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, heads=8, heads_out=1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 第1层: 输入 → hidden*heads (多头拼接)
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        # 多头输出维度 = hidden_channels * heads
        # 例如: 8维*8头 = 64维
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout,
            ))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # 输出层: 多头输出 → 类别数 (用平均而非拼接)
        self.convs.append(GATConv(
            hidden_channels * heads, out_channels, heads=heads_out,
            concat=False,  # concat=False: 多头取平均，输出维度=out_channels
            dropout=dropout,
        ))

    def forward(self, x, edge_index):
        """
        前向传播

        数据流动(GAT, heads=8):
        x: (2708, 1433)              ← 节点特征
          → GAT1: (2708, 64)          ← 8头注意力，8*8=64维
          → GAT2: (2708, 7)           ← 1头注意力，输出7类logits
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)  # GAT原论文用ELU
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE (SAmple and aggreGatE)

    【核心思想】
    GCN/GAT用所有邻居，在大图上计算代价高
    GraphSAGE: 采样固定数量的邻居 + 聚合 → 适合大图和归纳学习

    【与GCN的区别】
    - GCN: 用所有邻居，转导学习(测试时需要整张图)
    - SAGE: 可以采样邻居，归纳学习(可以预测训练时没见过的新节点)

    【聚合方式】
    - mean: 对邻居取平均(最常用，简单高效)
    - lstm: 用LSTM聚合(有顺序偏好，但表达力强)
    - pool: 对邻居做最大池化

    【适用场景】
    - 大规模图(百万级节点)
    - 归纳学习(需要预测新节点)
    - Pinterest推荐系统就用了GraphSAGE(PinSage)
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index):
        """前向传播"""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x


def create_model(cfg, num_features, num_classes):
    """
    根据配置创建GNN模型。

    【模型选择指南】
    - GCN:  入门首选，简单高效，Cora上约81%准确率
    - GAT:  精度最高(~83%)，但训练慢(注意力计算代价大)
    - SAGE: 大图首选，支持归纳学习
    """
    if cfg.model_type == "gcn":
        model = GCN(
            in_channels=num_features,
            hidden_channels=cfg.hidden_dim,
            out_channels=num_classes,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout_rate,
        )
    elif cfg.model_type == "gat":
        model = GAT(
            in_channels=num_features,
            hidden_channels=cfg.hidden_dim // cfg.gat_heads,  # 每头维度
            out_channels=num_classes,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout_rate,
            heads=cfg.gat_heads,
            heads_out=cfg.gat_heads_out,
        )
    elif cfg.model_type == "sage":
        model = GraphSAGEModel(
            in_channels=num_features,
            hidden_channels=cfg.hidden_dim,
            out_channels=num_classes,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout_rate,
        )
    else:
        raise ValueError(f"未知模型类型: {cfg.model_type}, 请选择 gcn/gat/sage")

    return model


# ============================================================
# Step 5: 训练和评估函数
# ============================================================
def train_one_epoch(model, data, optimizer, criterion, cfg):
    """
    训练一个epoch。

    【GNN训练 vs CNN训练的关键区别】
    1. 全图训练: GNN节点分类通常把整张图送入模型(不是batch)
       因为消息传递需要邻居信息，切batch会丢失邻居
    2. 掩码计算损失: 只用训练节点的标签算loss
       loss = criterion(output[train_mask], y[train_mask])
    3. 无数据增强: 图数据通常不做随机增强
    """
    model.train()
    optimizer.zero_grad()

    # 前向传播: 需要节点特征和边索引
    out = model(data.x, data.edge_index)

    # 只在训练节点上计算损失
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    # 反向传播
    loss.backward()
    optimizer.step()

    # 计算训练准确率
    pred = out[data.train_mask].argmax(dim=1)
    acc = (pred == data.y[data.train_mask]).float().mean().item()

    return loss.item(), acc


@torch.no_grad()
def evaluate(model, data, mask, criterion=None):
    """
    评估模型性能。

    参数:
        mask: 训练/验证/测试掩码
    """
    model.eval()
    out = model(data.x, data.edge_index)

    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum().item()
    acc = correct / mask.sum().item()

    if criterion is not None:
        loss = criterion(out[mask], data.y[mask]).item()
    else:
        loss = None

    return loss, acc, pred[mask].cpu().numpy(), data.y[mask].cpu().numpy()


def train(model, data, cfg):
    """
    完整训练流程。

    【GNN训练的常见问题】
    1. 过平滑(Over-smoothing): GNN层数太多，所有节点表示趋同
       解决: 2-3层即可，用残差连接
    2. 过拟合: 图数据通常很小，容易过拟合
       解决: Dropout + weight_decay + 早停
    3. 训练不稳定: 学习率太大导致震荡
       解决: 用较小的LR(0.005-0.01)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"开始训练 (模型: {cfg.model_type.upper()})...")
    print(f"{'='*60}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, data, optimizer, criterion, cfg)
        val_loss, val_acc, _, _ = evaluate(model, data, data.val_mask, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # 早停: 以验证准确率为准(节点分类常用准确率而非loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"\n⚠ 早停触发: 验证准确率连续{cfg.early_stop_patience}轮未提升")
                break

    # 恢复最优模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(cfg.device)
        print(f"\n✓ 已恢复最优模型 (Val Acc: {best_val_acc:.4f})")

    return model, history


# ============================================================
# Step 6: 可视化函数
# ============================================================
def plot_training_curves(history, cfg):
    """绘制训练曲线"""
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


def plot_node_embeddings(model, data, cfg):
    """
    可视化节点嵌入(使用t-SNE降维)。

    【t-SNE是什么？】
    高维数据(64维节点嵌入)无法直接可视化
    t-SNE将高维数据映射到2维，保持相似节点的邻近关系
    同类节点在图中应该聚集在一起

    【如何解读？】
    - 同色点聚在一起: 模型学到了有意义的类别特征
    - 同色点散开: 模型没学好，不同类别混淆
    - 训练前后对比: 训练前随机分散，训练后同类聚集
    """
    model.eval()
    with torch.no_grad():
        # 获取最后一层之前的嵌入
        out = model(data.x, data.edge_index)

    # t-SNE降维到2维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(out.cpu().numpy())

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. 训练前的嵌入(随机)
    np.random.seed(42)
    random_emb = np.random.randn(data.num_nodes, 2)
    colors = data.y.cpu().numpy()
    scatter1 = ax1.scatter(random_emb[:, 0], random_emb[:, 1],
                           c=colors, cmap="Set2", s=8, alpha=0.6)
    ax1.set_title("训练前(随机嵌入)", fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # 2. 训练后的嵌入
    scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                           c=colors, cmap="Set2", s=8, alpha=0.6)
    ax2.set_title(f"训练后({cfg.model_type.upper()}嵌入)", fontsize=12)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # 添加颜色条
    cbar = plt.colorbar(scatter2, ax=[ax1, ax2], shrink=0.8)
    cbar.set_ticks(range(cfg.num_classes))
    cbar.set_ticklabels(cfg.class_names[:cfg.num_classes])

    plt.suptitle("节点嵌入可视化(t-SNE)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "node_embeddings.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 节点嵌入已保存: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, cfg):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    names = cfg.class_names[:cfg.num_classes]
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=names, yticklabels=names,
           ylabel="真实类别", xlabel="预测类别",
           title="混淆矩阵")

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


def plot_attention_weights(model, data, cfg):
    """
    可视化GAT的注意力权重(仅GAT模型)。

    【注意力权重能告诉我们什么？】
    - 哪些邻居对当前节点最重要
    - 权重大: 该邻居对分类决策影响大
    - 权重均匀: 所有邻居贡献相同(类似GCN)
    - 权重集中在少数邻居: 模型学会了选择性关注
    """
    if cfg.model_type != "gat":
        print("⚠ 注意力权重可视化仅支持GAT模型，跳过")
        return

    model.eval()
    with torch.no_grad():
        # 获取第一层GAT的注意力权重
        # GATConv的forward返回(attentions=True时): (out, (edge_index, alpha))
        x = data.x
        for i, conv in enumerate(model.convs):
            if hasattr(conv, 'attentions') or isinstance(conv, GATConv):
                # GATConv支持return_attention_weights
                out, (edge_idx, attn) = conv(
                    x, data.edge_index, return_attention_weights=True
                )
                if i == 0:
                    # 只可视化第一层
                    break
                x = out
            else:
                x = conv(x, data.edge_index)

    # 平均所有头的注意力
    attn_mean = attn.mean(dim=1).cpu().numpy()  # (E',)

    # 注意: GATConv内部可能添加自环，edge_idx的边数可能>原始edge_index
    # 所以必须用edge_idx(返回的边索引)而非原始data.edge_index
    edge_idx_np = edge_idx.cpu().numpy()

    # 选择一个节点，可视化其邻居的注意力
    # 找一个训练节点
    train_nodes = data.train_mask.nonzero(as_tuple=True)[0]
    target_node = train_nodes[0].item()

    # 在GATConv返回的边索引中找邻居
    mask = edge_idx_np[1] == target_node
    neighbors = edge_idx_np[0, mask]
    neighbor_attns = attn_mean[mask]

    if len(neighbors) == 0:
        print("⚠ 所选节点没有邻居，跳过注意力可视化")
        return

    # 只取前20个邻居(避免太拥挤)
    top_k = min(20, len(neighbors))
    top_indices = np.argsort(neighbor_attns)[-top_k:]
    top_neighbors = neighbors[top_indices]
    top_attns = neighbor_attns[top_indices]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(range(top_k), top_attns, color="steelblue", alpha=0.8)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"节点{n}" for n in top_neighbors], fontsize=8)
    ax.set_xlabel("注意力权重")
    ax.set_ylabel("邻居节点")
    ax.set_title(f"节点{target_node}的邻居注意力权重(第一层GAT, 前{top_k}个)")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "attention_weights.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 注意力权重已保存: {save_path}")
    plt.close()


def plot_model_comparison(data, num_features, cfg):
    """
    对比三种GNN模型在Cora上的性能。

    【为什么要对比？】
    - GCN: 基线，最简单
    - GAT: 精度最高，但最慢
    - SAGE: 速度适中，适合大图
    - 对比帮助选择最适合任务的模型
    """
    models_config = [
        ("GCN", "gcn"),
        ("GAT", "gat"),
        ("GraphSAGE", "sage"),
    ]

    results = {}
    for name, model_type in models_config:
        # 临时修改配置
        cfg_copy = CONFIG()
        cfg_copy.model_type = model_type
        cfg_copy.epochs = 100  # 减少epoch加快对比

        model = create_model(cfg_copy, num_features, cfg.num_classes).to(cfg.device)
        data_dev = data.to(cfg.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg_copy.learning_rate,
                               weight_decay=cfg_copy.weight_decay)

        # 训练
        for epoch in range(cfg_copy.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data_dev.x, data_dev.edge_index)
            loss = criterion(out[data_dev.train_mask], data_dev.y[data_dev.train_mask])
            loss.backward()
            optimizer.step()

        # 测试
        model.eval()
        with torch.no_grad():
            out = model(data_dev.x, data_dev.edge_index)
            pred = out.argmax(dim=1)
            acc = (pred[data_dev.test_mask] == data_dev.y[data_dev.test_mask]).float().mean().item()

        results[name] = acc
        print(f"  {name}: 测试准确率 = {acc:.4f}")

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    accs = list(results.values())
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(names, accs, color=colors, alpha=0.8, width=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("测试准确率")
    ax.set_title("三种GNN模型在Cora上的性能对比")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "model_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 模型对比已保存: {save_path}")
    plt.close()


# ============================================================
# Step 7: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("GNN 节点分类 - Cora引用网络")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")
    if cfg.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # 加载数据
    print("\n加载数据集...")
    data, dataset = load_data(cfg)
    data = data.to(cfg.device)

    # 创建模型
    model = create_model(cfg, dataset.num_features, cfg.num_classes).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: {cfg.model_type.upper()}")
    print(f"总参数量: {total_params:,}")
    print(f"\n模型结构:\n{model}")

    # 训练
    model, history = train(model, data, cfg)

    # 测试集评估
    print(f"\n{'='*60}")
    print("测试集评估...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_pred, y_true = evaluate(model, data, data.test_mask, criterion)
    print(f"测试集 Loss: {test_loss:.4f} | 准确率: {test_acc:.4f}")

    # 分类报告
    print("\n分类报告:")
    names = cfg.class_names[:cfg.num_classes]
    print(classification_report(y_true, y_pred, target_names=names, digits=4, zero_division=0))

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "gnn_node_classifier.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_type": cfg.model_type,
        "num_features": dataset.num_features,
        "num_classes": cfg.num_classes,
    }, model_path)
    print(f"✓ 模型已保存: {model_path}")

    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, cfg)
    plot_node_embeddings(model, data, cfg)
    plot_confusion_matrix(y_true, y_pred, cfg)
    plot_attention_weights(model, data, cfg)

    # 模型对比
    print("\n三种GNN模型性能对比...")
    data_cpu = data.cpu()  # 对比时用CPU避免显存不足
    plot_model_comparison(data_cpu, dataset.num_features, cfg)

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
