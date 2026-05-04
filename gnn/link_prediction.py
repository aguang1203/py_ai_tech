"""
=============================================================================
GNN 链接预测任务模板 (Graph Neural Network for Link Prediction)
=============================================================================

【原理】
链接预测(Link Prediction): 给定图中的部分边，预测图中两个节点之间是否存在(或将出现)边。

这是GNN最实用的任务之一：
  节点分类: 预测节点的属性 (这个用户喜欢什么)
  图分类:   预测整个图的类别 (这个分子有毒吗)
  链接预测: 预测节点间的关系 (这两个用户会成为朋友吗？)

【链接预测的方法】
1. 基于启发式: Common Neighbors, Jaccard, Adamic-Adar
   只用图结构统计，不用特征，精度有限

2. 基于GNN:
   Step 1: 用GNN学习节点嵌入 h_i, h_j
   Step 2: 用解码器(Decoder)计算边存在的概率
     - 点积: p(i,j) = σ(h_i^T · h_j)  (最简单)
     - 拼接+MLP: p(i,j) = σ(MLP([h_i || h_j]))  (更灵活)
     - Hadamard: p(i,j) = σ(MLP(h_i ⊙ h_j))

【训练策略: 负采样】
- 正样本: 图中真实存在的边
- 负样本: 图中不存在的边(随机采样)
- 训练目标: 正样本概率高，负样本概率低
- 为什么需要负采样？所有可能的边中，存在的边极少(稀疏)
  如果只用正样本训练，模型会预测所有边都存在

【本模板使用的方法】
1. 在Cora图上随机删掉10%的边作为测试正样本
2. 随机采样同等数量的不存在的边作为负样本
3. 用GCN学习节点嵌入
4. 用MLP解码器预测边的存在性

【应用场景】
- 社交网络好友推荐 (你会和谁成为朋友？)
- 商品推荐 (你会买什么商品？)
- 知识图谱补全 (实体间的关系推理)
- 药物交互预测 (两种药物会相互作用吗？)
- 引文网络 (两篇论文会引用吗？)

【本数据集: Cora引用网络】
- 与节点分类使用同一个Cora数据集
- 但任务不同: 预测两个论文之间是否存在引用关系

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python gnn/link_prediction.py
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

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import negative_sampling, train_test_split_edges

from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    classification_report, roc_curve,
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
    dataset_name = "Cora"

    # test_edge_ratio=0.1: 划出10%的边作为测试集
    test_edge_ratio = 0.1

    # val_edge_ratio=0.05: 划出5%的边作为验证集
    val_edge_ratio = 0.05

    # random_state=42: 随机种子
    random_state = 42

    # --- 模型相关 ---
    # hidden_dim=128: 隐藏层维度
    #   链接预测的嵌入维度通常比节点分类更大
    #   因为需要编码更细粒度的节点关系信息
    hidden_dim = 64

    # num_layers=2: GNN层数
    num_layers = 2

    # dropout_rate=0.5: Dropout
    dropout_rate = 0.5

    # decoder_type="mlp": 解码器类型
    #   "dot": 点积解码 h_i^T h_j, 参数少，但表达力有限
    #   "mlp": MLP解码 MLP([h_i || h_j || h_i⊙h_j |h_i-h_j|])
    #          拼接多种交互特征，表达力更强
    decoder_type = "mlp"

    # --- 训练相关 ---
    # learning_rate=0.005: 学习率
    learning_rate = 0.005

    # weight_decay=1e-3: L2正则化(比节点分类更大，防止过拟合)
    weight_decay = 1e-3

    # epochs=200: 训练轮数
    epochs = 200

    # early_stop_patience=20: 早停耐心
    early_stop_patience = 20

    # --- 保存相关 ---
    save_dir = "gnn/output/link_prediction"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 数据准备
# ============================================================
def prepare_link_prediction_data(cfg):
    """
    准备链接预测数据。

    【关键步骤】
    1. 加载Cora图
    2. 划分边: 训练边/验证边/测试边
    3. 生成负样本: 不存在的边

    【为什么要划分边？】
    - 训练时: 只用训练边(不偷看测试边)
    - 验证时: 用验证边评估(调超参数)
    - 测试时: 用测试边评估(最终结果)
    - 不能用测试边训练，否则就是"作弊"

    【负采样】
    - 正样本: 真实存在的边(从划分的边集合中取)
    - 负样本: 随机采样的不存在的边
    - 比例: 1:1(正负样本数量相等)
    - 原理: 链接预测本质是二分类(有边/无边)
    """
    dataset = Planetoid(root=cfg.data_dir, name=cfg.dataset_name,
                        transform=NormalizeFeatures())
    data = dataset[0]

    # 使用train_test_split_edges自动划分边
    # 它会把边划分为: train_pos_edge_index, val_pos_edge_index, test_pos_edge_index
    # 同时生成对应的负样本边
    data = train_test_split_edges(data, val_ratio=cfg.val_edge_ratio,
                                  test_ratio=cfg.test_edge_ratio)

    print(f"数据集: {cfg.dataset_name}")
    print(f"节点数: {data.num_nodes} | 特征维度: {dataset.num_features}")
    print(f"训练正边: {data.train_pos_edge_index.size(1)}")
    print(f"验证正边: {data.val_pos_edge_index.size(1)} | 验证负边: {data.val_neg_edge_index.size(1)}")
    print(f"测试正边: {data.test_pos_edge_index.size(1)} | 测试负边: {data.test_neg_edge_index.size(1)}")

    return data, dataset


# ============================================================
# Step 4: 模型定义
# ============================================================
class DotDecoder(nn.Module):
    """
    点积解码器: p(i,j) = σ(h_i^T · h_j)

    【优缺点】
    优点: 无额外参数，计算快，有对称性(h_i^T h_j = h_j^T h_i)
    缺点: 表达力有限，只能捕捉线性关系

    【适用场景】
    - 节点嵌入维度较高时
    - 图的边关系比较简单时
    """

    def forward(self, z, edge_index):
        """
        参数:
            z: 节点嵌入 (N, d)
            edge_index: 要预测的边 (2, E)
        返回:
            每条边的存在概率 (E,)
        """
        # 取出边的两个端点的嵌入
        h_i = z[edge_index[0]]  # (E, d)
        h_j = z[edge_index[1]]  # (E, d)

        # 点积 + Sigmoid → 概率
        return (h_i * h_j).sum(dim=1).sigmoid()


class MLPDecoder(nn.Module):
    """
    MLP解码器: p(i,j) = σ(MLP([h_i || h_j || h_i⊙h_j || |h_i-h_j|]))

    【为什么拼接4种特征？】
    - h_i: 节点i的特征(单独看i是什么样的)
    - h_j: 节点j的特征(单独看j是什么样的)
    - h_i⊙h_j: 逐元素积(Hadamard积)，捕捉两节点的相似性
      乘法有交互效果: 相似节点乘积大，不相似节点乘积小
    - |h_i-h_j|: 差的绝对值，捕捉两节点的差异性
      距离近的节点差异小，距离远的差异大

    【为什么比点积好？】
    点积只能捕捉线性关系: h_i^T h_j = Σ h_ik * h_jk
    MLP可以学习非线性组合: 例如"h_i的第3维>0 且 h_j的第7维<0 则有边"
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # 输入维度 = 4 * hidden_dim (拼接4种特征)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z, edge_index):
        h_i = z[edge_index[0]]  # (E, d)
        h_j = z[edge_index[1]]  # (E, d)

        # 4种交互特征
        features = torch.cat([
            h_i,                           # 节点i
            h_j,                           # 节点j
            h_i * h_j,                     # Hadamard积(相似性)
            torch.abs(h_i - h_j),          # 差的绝对值(差异性)
        ], dim=1)

        return self.mlp(features).squeeze(-1).sigmoid()


class LinkPredictionModel(nn.Module):
    """
    链接预测模型: GCN编码器 + 解码器

    【架构】
    节点特征 → GCN层×2 → 节点嵌入z → 解码器 → 边概率

    【编码器-解码器架构】
    编码器(GCN): 将节点特征映射到低维嵌入空间
    解码器: 从节点嵌入计算边存在的概率

    为什么分离编码器和解码器？
    - 编码器学习节点的"语义表示"(每个节点是什么)
    - 解码器学习边的"关系模式"(什么样的节点对会有边)
    - 分离后可以独立改进(换更好的编码器或解码器)
    """

    def __init__(self, in_channels, hidden_channels, num_layers, dropout, decoder_type="mlp"):
        super().__init__()

        # GCN编码器
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # 解码器
        if decoder_type == "dot":
            self.decoder = DotDecoder()
        else:
            self.decoder = MLPDecoder(hidden_channels)

        self.dropout = dropout

    def encode(self, x, edge_index):
        """
        编码器: 学习节点嵌入

        数据流动:
        x: (2708, 1433)      ← 节点特征
          → GCN1: (2708, 128) ← 聚合1跳邻居
          → GCN2: (2708, 128) ← 聚合2跳邻居
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def decode(self, z, edge_index):
        """
        解码器: 从节点嵌入预测边的概率

        参数:
            z: 节点嵌入 (N, d)
            edge_index: 待预测的边 (2, E)
        返回:
            每条边的存在概率 (E,)
        """
        return self.decoder(z, edge_index)

    def forward(self, x, train_edge_index, pred_edge_index):
        """
        前向传播

        注意: 训练和预测使用不同的边集合
        - train_edge_index: 用于GNN消息传递的边(训练集边)
        - pred_edge_index: 要预测概率的边(可能是正/负样本)
        """
        z = self.encode(x, train_edge_index)
        return self.decode(z, pred_edge_index)


# ============================================================
# Step 5: 训练和评估
# ============================================================
def train_one_epoch(model, data, optimizer, criterion, cfg):
    """
    训练一个epoch。

    【链接预测训练的关键】
    1. 只用训练边做消息传递(不偷看验证/测试边)
    2. 负采样: 每个epoch采样不同的负样本(增加训练多样性)
    3. 正负样本1:1训练(防止模型偏向预测"无边")
    """
    model.train()
    optimizer.zero_grad()

    # 编码: 只用训练边做消息传递
    z = model.encode(data.x, data.train_pos_edge_index)

    # 正样本: 训练正边
    pos_edge_index = data.train_pos_edge_index

    # 负采样: 随机采样不存在的边
    # 采样数量 = 正边数量(1:1比例)
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1),
    )

    # 解码: 预测正负样本
    pos_pred = model.decode(z, pos_edge_index)
    neg_pred = model.decode(z, neg_edge_index)

    # 拼接预测和标签
    preds = torch.cat([pos_pred, neg_pred])
    labels = torch.cat([
        torch.ones(pos_pred.size(0), device=cfg.device),
        torch.zeros(neg_pred.size(0), device=cfg.device),
    ])

    # 计算损失
    loss = criterion(preds, labels)
    loss.backward()
    optimizer.step()

    # 计算准确率
    pred_binary = (preds > 0.5).float()
    acc = (pred_binary == labels).float().mean().item()

    return loss.item(), acc


@torch.no_grad()
def evaluate(model, data, cfg):
    """
    评估链接预测性能。

    【评估指标】
    - AUC-ROC: 正样本的预测分数高于负样本的概率
      0.5=随机猜测, 1.0=完美预测
    - AP(Average Precision): 精确率-召回率曲线下面积
      对正负样本不平衡更鲁棒
    - Accuracy: 预测正确的比例
    """
    model.eval()
    z = model.encode(data.x, data.train_pos_edge_index)

    # 验证集评估
    pos_pred_val = model.decode(z, data.val_pos_edge_index)
    neg_pred_val = model.decode(z, data.val_neg_edge_index)

    preds_val = torch.cat([pos_pred_val, neg_pred_val]).cpu().numpy()
    labels_val = torch.cat([
        torch.ones(pos_pred_val.size(0)),
        torch.zeros(neg_pred_val.size(0)),
    ]).numpy()

    auc_val = roc_auc_score(labels_val, preds_val)
    ap_val = average_precision_score(labels_val, preds_val)
    acc_val = accuracy_score(labels_val, (preds_val > 0.5).astype(int))

    # 测试集评估
    pos_pred_test = model.decode(z, data.test_pos_edge_index)
    neg_pred_test = model.decode(z, data.test_neg_edge_index)

    preds_test = torch.cat([pos_pred_test, neg_pred_test]).cpu().numpy()
    labels_test = torch.cat([
        torch.ones(pos_pred_test.size(0)),
        torch.zeros(neg_pred_test.size(0)),
    ]).numpy()

    auc_test = roc_auc_score(labels_test, preds_test)
    ap_test = average_precision_score(labels_test, preds_test)
    acc_test = accuracy_score(labels_test, (preds_test > 0.5).astype(int))

    return {
        "val_auc": auc_val, "val_ap": ap_val, "val_acc": acc_val,
        "test_auc": auc_test, "test_ap": ap_test, "test_acc": acc_test,
        "preds_test": preds_test, "labels_test": labels_test,
    }


def train(model, data, cfg):
    """完整训练流程"""
    criterion = nn.BCELoss()  # 二分类交叉熵
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_val_auc = 0
    patience_counter = 0
    best_model_state = None
    history = {"train_loss": [], "train_acc": [], "val_auc": [], "val_ap": []}

    print(f"\n{'='*60}")
    print(f"开始训练 (解码器: {cfg.decoder_type})...")
    print(f"{'='*60}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, data, optimizer, criterion, cfg)
        metrics = evaluate(model, data, cfg)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_auc"].append(metrics["val_auc"])
        history["val_ap"].append(metrics["val_ap"])

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val AUC: {metrics['val_auc']:.4f} AP: {metrics['val_ap']:.4f}")

        if metrics["val_auc"] > best_val_auc:
            best_val_auc = metrics["val_auc"]
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
        print(f"\n✓ 已恢复最优模型 (Val AUC: {best_val_auc:.4f})")

    return model, history


# ============================================================
# Step 6: 可视化
# ============================================================
def plot_training_curves(history, cfg):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0, 0].plot(epochs, history["train_loss"], "b-", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("训练损失")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["train_acc"], "b-", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("训练准确率")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["val_auc"], "r-", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("AUC-ROC")
    axes[1, 0].set_title("验证AUC-ROC")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history["val_ap"], "g-", linewidth=2)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Average Precision")
    axes[1, 1].set_title("验证AP")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("链接预测训练曲线", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()


def plot_roc_curve(preds, labels, cfg):
    """
    绘制ROC曲线。

    【如何解读ROC曲线？】
    - X轴: 假正率(FPR) = 预测为正但实际为负的比例
    - Y轴: 真正率(TPR) = 预测为正且实际为正的比例
    - 曲线越靠左上角，模型越好
    - 对角线=随机猜测(AUC=0.5)
    - AUC=1.0=完美预测
    """
    fpr, tpr, _ = roc_curve(labels, preds)
    auc = roc_auc_score(labels, preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"GNN (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="随机猜测 (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="blue")
    ax.set_xlabel("假正率 (FPR)")
    ax.set_ylabel("真正率 (TPR)")
    ax.set_title("ROC曲线 - 链接预测")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "roc_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ ROC曲线已保存: {save_path}")
    plt.close()


def plot_edge_predictions(model, data, cfg, num_examples=10):
    """
    可视化边的预测概率分布。

    正样本(真实存在的边)应该有高概率
    负样本(不存在的边)应该有低概率
    """
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)

        pos_pred = model.decode(z, data.test_pos_edge_index).cpu().numpy()
        neg_pred = model.decode(z, data.test_neg_edge_index).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 5))

    # 采样以避免太多点
    pos_sample = np.random.choice(pos_pred, min(num_examples * 10, len(pos_pred)), replace=False)
    neg_sample = np.random.choice(neg_pred, min(num_examples * 10, len(neg_pred)), replace=False)

    ax.hist(pos_sample, bins=30, alpha=0.6, label=f"正样本(均值={pos_sample.mean():.3f})", color="green")
    ax.hist(neg_sample, bins=30, alpha=0.6, label=f"负样本(均值={neg_sample.mean():.3f})", color="red")
    ax.axvline(x=0.5, color="black", linestyle="--", label="阈值=0.5")
    ax.set_xlabel("预测概率")
    ax.set_ylabel("数量")
    ax.set_title("边预测概率分布")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "edge_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 边预测分布已保存: {save_path}")
    plt.close()


# ============================================================
# Step 7: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("GNN 链接预测 - Cora引用网络")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # 加载数据
    print("\n加载数据集...")
    data, dataset = prepare_link_prediction_data(cfg)
    data = data.to(cfg.device)

    # 创建模型
    model = LinkPredictionModel(
        in_channels=dataset.num_features,
        hidden_channels=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout_rate,
        decoder_type=cfg.decoder_type,
    ).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: GCN编码器 + {cfg.decoder_type.upper()}解码器")
    print(f"总参数量: {total_params:,}")
    print(f"\n模型结构:\n{model}")

    # 训练
    model, history = train(model, data, cfg)

    # 测试集评估
    print(f"\n{'='*60}")
    print("测试集评估...")
    metrics = evaluate(model, data, cfg)
    print(f"测试集 AUC-ROC: {metrics['test_auc']:.4f} | AP: {metrics['test_ap']:.4f} | Acc: {metrics['test_acc']:.4f}")

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "gnn_link_predictor.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
    }, model_path)
    print(f"✓ 模型已保存: {model_path}")

    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, cfg)
    plot_roc_curve(metrics["preds_test"], metrics["labels_test"], cfg)
    plot_edge_predictions(model, data, cfg)

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
