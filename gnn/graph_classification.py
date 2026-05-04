"""
=============================================================================
GNN 图分类任务模板 (Graph Neural Network for Graph Classification)
=============================================================================

【原理】
图分类与节点分类不同：
  节点分类: 给定一个图，预测图中每个节点的类别
  图分类:   给定多个图，预测每个图的整体类别

核心区别在于如何从节点表示得到图表示：
  节点分类: 每个节点的输出就是预测
  图分类:   需要"读出"(Readout)操作，聚合所有节点表示为一个图表示

常见的Readout操作:
  1. 全局平均池化: h_G = (1/N) Σ h_i  (最简单)
  2. 全局最大池化: h_G = max(h_1, ..., h_N)
  3. 注意力池化: h_G = Σ α_i h_i  (学习每个节点的重要性)
  4. Set2Set: 更复杂的聚合(类似注意力机制)
  5. 虚拟节点: 添加一个特殊节点与所有节点相连

【本模板使用的模型: GIN (Graph Isomorphism Network)】
GIN的理论基础很强:
  - WL测试(Weisfeiler-Lehman): 判断两个图是否同构的经典算法
  - GIN的表达能力 = 1-WL测试(这是消息传递GNN的理论上限)
  - GCN的表达能力 < GIN(GCN的聚合是简单平均，会丢失信息)

GIN的核心公式: h'_i = MLP((1+ε) · h_i + Σ_{j∈N(i)} h_j)
  - (1+ε): 放大中心节点自身的信息(区分自己和邻居)
  - MLP: 多层感知机(比单层线性变换表达力更强)
  - 可学习ε: 让模型自己决定中心节点和邻居的重要性比例

【应用场景】
- 分子属性预测 (毒性/溶解度/药效) ← 最经典的图分类场景
- 蛋白质功能分类
- 社交网络分类 (社区类型)
- 程序图分类 (漏洞检测)
- 交通网络分类 (拥堵模式)

【本数据集: 合成图分类数据集】
- 4个类别: 环形图/星形图/路径图/完全图
- 每类100个图，共400个图
- 即时生成，无需下载
- 每个节点3维特征(拓扑特征)

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python gnn/graph_classification.py
=============================================================================
"""

# ============================================================
# Step 1: 导入必要的库
# ============================================================
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

# PyTorch Geometric
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.utils import to_networkx

import networkx as nx

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

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
    """超参数配置中心"""

    # --- 数据相关 ---
    data_dir = "data"

    # num_graphs_per_class: 每类生成的图数量
    num_graphs_per_class = 100

    # num_classes=4: 4种图类型
    num_classes = 4

    class_names = ["环形图", "星形图", "路径图", "完全图"]

    # node_feature_dim=3: 节点特征维度(拓扑特征)
    #   特征: [度中心性, 聚类系数, 是否为极端节点]
    node_feature_dim = 3

    # test_ratio=0.2: 测试集比例
    test_ratio = 0.2

    # val_ratio=0.1: 验证集比例
    val_ratio = 0.1

    # random_state=42: 随机种子
    random_state = 42

    # --- 模型相关 ---
    # model_type="gin": 图分类推荐GIN
    #   "gin": Graph Isomorphism Network, 表达能力最强
    #   "gcn": GCN + 全局池化, 简单基线
    model_type = "gin"

    # hidden_dim=64: 隐藏层维度
    hidden_dim = 64

    # num_layers=5: GNN层数
    #   为什么5层？图分类需要更大的感受野来捕捉全局结构
    #   节点分类2层就够了，图分类通常需要3-5层
    num_layers = 5

    # dropout_rate=0.3: Dropout
    dropout_rate = 0.3

    # readout="mean": 图级别的Readout操作
    #   "mean": 全局平均池化，最简单
    #   "max": 全局最大池化，捕捉极端特征
    #   "mean+max": 两者拼接，信息更丰富
    readout = "mean+max"

    # --- 训练相关 ---
    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 5e-4
    epochs = 150
    early_stop_patience = 20

    # --- 保存相关 ---
    save_dir = "gnn/output/graph_classification"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 合成数据集生成
# ============================================================
def generate_ring_graph(n_nodes, seed=None):
    """
    生成环形图: 每个节点连接前后两个邻居，首尾相连。

    环形图: 1-2-3-...-n-1
    特征: 每个节点度=2，有循环结构
    """
    if seed is not None:
        np.random.seed(seed)
    edges = []
    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        edges.append([i, j])
        edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def generate_star_graph(n_nodes, seed=None):
    """
    生成星形图: 一个中心节点连接所有其他节点。

    星形图: 中心-1, 中心-2, ..., 中心-(n-1)
    特征: 中心节点度=n-1，其余节点度=1
    """
    edges = []
    center = 0
    for i in range(1, n_nodes):
        edges.append([center, i])
        edges.append([i, center])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def generate_path_graph(n_nodes, seed=None):
    """
    生成路径图: 节点按顺序连接，首尾不相连。

    路径图: 1-2-3-...-n
    特征: 两端节点度=1，中间节点度=2
    """
    edges = []
    for i in range(n_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def generate_complete_graph(n_nodes, seed=None):
    """
    生成完全图: 每个节点连接所有其他节点。

    完全图: 所有节点两两相连
    特征: 每个节点度=n-1，密度最高
    """
    edges = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def compute_node_features(edge_index, num_nodes):
    """
    计算节点的拓扑特征。

    【为什么用拓扑特征而非随机特征？】
    - 随机特征: 模型可能忽略图结构，只靠特征就能分类
    - 拓扑特征: 让模型必须利用图结构来区分不同类型
    - 3个特征: [度/最大度, 聚类系数, 度方差贡献]

    【聚类系数】
    一个节点的邻居之间相互连接的比例
    - 完全图: 每个节点的邻居全部互连，聚类系数=1
    - 星形图: 中心节点聚类系数=0(邻居间无边)，叶节点聚类系数=0(没有邻居对)
    - 环形图: 每个节点的2个邻居互连(如果有三角形)，否则=0
    """
    # 度
    degree = torch.zeros(num_nodes)
    for i in range(edge_index.size(1)):
        degree[edge_index[0, i]] += 1

    # 用networkx计算聚类系数
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edge_list = edge_index.t().tolist()
    # 去重(无向边存了两次)
    edge_set = set()
    for e in edge_list:
        edge_set.add((min(e), max(e)))
    G.add_edges_from(edge_set)

    clustering = nx.clustering(G)
    clustering_vals = torch.tensor([clustering[i] for i in range(num_nodes)], dtype=torch.float)

    # 归一化度
    max_degree = degree.max().item() if degree.max() > 0 else 1.0
    norm_degree = degree / max_degree

    # 组合特征
    features = torch.stack([norm_degree, clustering_vals, degree / (num_nodes - 1)], dim=1)

    return features.float()


def generate_synthetic_dataset(cfg):
    """
    生成合成图分类数据集。

    生成4类图: 环形/星形/路径/完全
    每类图有不同的大小(节点数6-15)，增加多样性
    """
    np.random.seed(cfg.random_state)
    graphs = []

    generators = [
        (generate_ring_graph, 0),
        (generate_star_graph, 1),
        (generate_path_graph, 2),
        (generate_complete_graph, 3),
    ]

    for gen_func, label in generators:
        for i in range(cfg.num_graphs_per_class):
            # 随机节点数(6-15)
            n_nodes = np.random.randint(6, 16)
            # 完全图节点数不能太多(否则边太多)
            if label == 3:
                n_nodes = min(n_nodes, 10)

            edge_index = gen_func(n_nodes, seed=cfg.random_state * 100 + i)
            x = compute_node_features(edge_index, n_nodes)

            data = Data(x=x, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long))
            graphs.append(data)

    # 打乱
    np.random.shuffle(graphs)

    print(f"合成数据集: {len(graphs)}个图, {cfg.num_classes}类")
    print(f"各类图数量: {[sum(1 for g in graphs if g.y.item() == i) for i in range(cfg.num_classes)]}")

    return graphs


def split_dataset(graphs, cfg):
    """划分训练/验证/测试集"""
    n = len(graphs)
    n_test = int(n * cfg.test_ratio)
    n_val = int(n * cfg.val_ratio)
    n_train = n - n_test - n_val

    # 固定随机种子
    generator = torch.Generator().manual_seed(cfg.random_state)
    train_dataset, val_dataset, test_dataset = random_split(
        graphs, [n_train, n_val, n_test], generator=generator,
    )

    # PyG的DataLoader
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    print(f"训练集: {n_train} | 验证集: {n_val} | 测试集: {n_test}")

    return train_loader, val_loader, test_loader


# ============================================================
# Step 4: 模型定义
# ============================================================
class GINConv(nn.Module):
    """
    GIN卷积层 (Graph Isomorphism Network)

    【核心公式】
    h'_i = MLP((1+ε) · h_i + Σ_{j∈N(i)} h_j)

    为什么GIN比GCN表达力更强？
    - GCN用平均聚合: h' = σ(MEAN(h_self, h_neighbors) · W)
      问题: 不同的邻居集合可能产生相同的均值(信息丢失)
    - GIN用求和+MLP: h' = MLP((1+ε)·h_self + SUM(h_neighbors))
      优势: 求和保留了邻居的多重性(1个度2邻居 ≠ 2个度1邻居)
      MLP: 比单层变换表达力更强(可以学习任意函数)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # MLP: 两层全连接 + BN + ReLU
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )
        # 可学习的ε参数
        # 初始化为0，训练过程中自动调整
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index):
        """
        前向传播

        公式: h'_i = MLP((1+ε) · h_i + Σ_{j∈N(i)} h_j)
        - (1+ε) · h_i: 中心节点自身的信息，被放大
        - Σ h_j: 所有邻居信息的求和
        """
        # 消息传递: 邻居求和
        from torch_geometric.utils import scatter
        row, col = edge_index
        # 聚合邻居信息(求和)
        neighbor_sum = scatter(x[col], row, dim=0, reduce='sum', dim_size=x.size(0))

        # 组合中心节点和邻居
        out = (1 + self.eps) * x + neighbor_sum

        # MLP变换
        out = self.mlp(out)
        return out


class GINGraphClassifier(nn.Module):
    """
    GIN图分类模型

    【架构设计】
    输入图 → GIN层×5 → Readout → MLP分类器

    与节点分类的区别:
    - 节点分类: 输出每个节点的类别
    - 图分类: 需要Readout操作把节点表示聚合为图表示

    【Readout操作对比】
    - mean: h_G = (1/N)Σh_i  简单，但可能被大量普通节点稀释
    - max:  h_G = max(h_i)   捕捉极端特征，但可能忽略整体趋势
    - mean+max: concat(mean, max) 两者互补，信息最丰富
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, readout="mean"):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.readout_type = readout

        # GIN层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 第1层
        self.convs.append(GINConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # 中间层
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # 分类头
        # Readout后的维度
        if readout == "mean+max":
            readout_dim = hidden_channels * 2
        else:
            readout_dim = hidden_channels

        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def readout(self, x, batch):
        """
        图级别的Readout操作。

        参数:
            x: 所有节点的表示 (N_total, hidden)
            batch: 批次索引 (N_total,), batch[i]=k表示第i个节点属于第k个图

        【为什么需要batch索引？】
        PyG的DataLoader把多个图合并成一个大图(加速计算)
        batch记录了每个节点属于哪个图
        Readout时需要按batch分组聚合
        """
        if self.readout_type == "mean":
            return global_mean_pool(x, batch)
        elif self.readout_type == "max":
            return global_max_pool(x, batch)
        elif self.readout_type == "mean+max":
            # 拼接mean和max，信息更丰富
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            return torch.cat([x_mean, x_max], dim=1)
        else:
            return global_mean_pool(x, batch)

    def forward(self, x, edge_index, batch):
        """
        前向传播

        数据流动:
        x: (N, 3)              ← 节点特征(拓扑特征)
          → GIN1: (N, 64)      ← 聚合1跳邻居
          → GIN2: (N, 64)      ← 聚合2跳邻居
          → ...                  ← 逐层扩大感受野
          → Readout: (B, 128)  ← 聚合为图表示(B=batch_size)
          → MLP: (B, 4)        ← 图分类logits
        """
        # GNN层: 逐层聚合邻居信息
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Readout: 节点表示 → 图表示
        x = self.readout(x, batch)

        # 分类
        x = self.classifier(x)
        return x


class GCNGraphClassifier(nn.Module):
    """GCN图分类基线模型"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, readout="mean"):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.readout_type = readout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        if readout == "mean+max":
            readout_dim = hidden_channels * 2
        else:
            readout_dim = hidden_channels

        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def readout(self, x, batch):
        if self.readout_type == "mean+max":
            return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        elif self.readout_type == "max":
            return global_max_pool(x, batch)
        return global_mean_pool(x, batch)

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.readout(x, batch)
        x = self.classifier(x)
        return x


def create_model(cfg, num_features, num_classes):
    """根据配置创建图分类模型"""
    if cfg.model_type == "gin":
        model = GINGraphClassifier(
            in_channels=num_features,
            hidden_channels=cfg.hidden_dim,
            out_channels=num_classes,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout_rate,
            readout=cfg.readout,
        )
    elif cfg.model_type == "gcn":
        model = GCNGraphClassifier(
            in_channels=num_features,
            hidden_channels=cfg.hidden_dim,
            out_channels=num_classes,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout_rate,
            readout=cfg.readout,
        )
    else:
        raise ValueError(f"未知模型类型: {cfg.model_type}")
    return model


# ============================================================
# Step 5: 训练和评估
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, cfg):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(cfg.device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, cfg):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(cfg.device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)

        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


def train(model, train_loader, val_loader, cfg):
    """完整训练流程"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"开始训练 (模型: {cfg.model_type.upper()}, Readout: {cfg.readout})...")
    print(f"{'='*60}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, cfg)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, cfg)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"\n⚠ 早停触发")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(cfg.device)
        print(f"\n✓ 已恢复最优模型 (Val Acc: {best_val_acc:.4f})")

    return model, history


# ============================================================
# Step 6: 可视化
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


def plot_graph_samples(graphs, cfg, num_per_class=3):
    """
    可视化各类图的示例。

    帮助理解不同图类型的结构差异:
    - 环形图: 圆环状连接
    - 星形图: 中心辐射
    - 路径图: 线性排列
    - 完全图: 全部互连
    """
    fig, axes = plt.subplots(cfg.num_classes, num_per_class, figsize=(4 * num_per_class, 4 * cfg.num_classes))

    for label in range(cfg.num_classes):
        class_graphs = [g for g in graphs if g.y.item() == label]
        for j in range(num_per_class):
            ax = axes[label, j] if cfg.num_classes > 1 else axes[j]
            g = class_graphs[j]
            G = to_networkx(g, to_undirected=True)
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, ax=ax, node_size=80, node_color="steelblue",
                    edge_color="gray", width=1.5, with_labels=False)
            if j == 0:
                ax.set_ylabel(cfg.class_names[label], fontsize=12, fontweight="bold")
            ax.set_title(f"图{j+1} ({g.num_nodes}节点)")

    plt.suptitle("各类图示例", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "graph_samples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 图示例已保存: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, cfg):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=cfg.class_names, yticklabels=cfg.class_names,
           ylabel="真实类别", xlabel="预测类别", title="混淆矩阵")

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


# ============================================================
# Step 7: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("GNN 图分类 - 合成图数据集")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # 生成数据集
    print("\n生成合成数据集...")
    graphs = generate_synthetic_dataset(cfg)

    # 可视化图示例
    plot_graph_samples(graphs, cfg)

    # 划分数据集
    train_loader, val_loader, test_loader = split_dataset(graphs, cfg)

    # 创建模型
    model = create_model(cfg, cfg.node_feature_dim, cfg.num_classes).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: {cfg.model_type.upper()}")
    print(f"总参数量: {total_params:,}")
    print(f"\n模型结构:\n{model}")

    # 训练
    model, history = train(model, train_loader, val_loader, cfg)

    # 测试
    print(f"\n{'='*60}")
    print("测试集评估...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, cfg)
    print(f"测试集 Loss: {test_loss:.4f} | 准确率: {test_acc:.4f}")

    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=cfg.class_names, digits=4, zero_division=0))

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "gnn_graph_classifier.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
    }, model_path)
    print(f"✓ 模型已保存: {model_path}")

    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, cfg)
    plot_confusion_matrix(y_true, y_pred, cfg)

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
