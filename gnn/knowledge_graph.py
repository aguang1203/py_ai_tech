"""
=============================================================================
GNN 知识图谱表示学习模板 (Knowledge Graph Representation Learning)
=============================================================================

【原理】
知识图谱(Knowledge Graph, KG)是结构化的人类知识表示:
  三元组: (头实体, 关系, 尾实体)  即 (h, r, t)
  例子: (北京, 是首都, 中国), (爱因斯坦, 提出了, 相对论)

知识图谱表示学习: 将实体和关系映射到低维向量空间，
使得真实三元组的分数高于虚假三元组。

【经典方法对比】
1. TransE: h + r ≈ t
   最简单的翻译模型，将关系看作实体间的"平移"
   缺点: 无法处理1-to-N关系(一个头实体+关系对应多个尾实体)

2. DistMult: score = h^T · diag(r) · t
   用双线性模型，关系是对角矩阵
   缺点: 对称关系(无法区分方向)

3. ComplEx: 在复数空间做DistMult
   可以处理非对称关系
   是DistMult在复数域的推广

4. RotatE: h ∘ r ≈ t (复数空间旋转)
   将关系看作复数空间中的旋转
   能处理对称/反对称/组合/逆关系

【本模板使用的方法: RotatE + GNN增强】
Step 1: 用GNN(GCN)在知识图谱上传播，得到实体的结构嵌入
Step 2: 用RotatE的旋转模型计算三元组的分数
Step 3: 负采样训练(替换头实体或尾实体)

【应用场景】
- 智能问答 (知识推理)
- 推荐系统 (用户-物品-属性三元组)
- 药物发现 (药物-靶点-疾病)
- 搜索引擎 (实体关系理解)
- 社交网络 (人-关系-人)

【本数据集: 合成知识图谱】
- 5种关系类型: 属于/位于/创立于/属于行业/毕业于
- 约500个实体(人物/公司/城市/学校/行业)
- 约2000个三元组
- 即时生成，无需下载

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 直接运行: python gnn/knowledge_graph.py
=============================================================================
"""

# ============================================================
# Step 1: 导入必要的库
# ============================================================
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score

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
    # num_entities: 实体数量(从数据集自动获取)
    num_entities = 0

    # num_relations: 关系类型数量
    num_relations = 5

    # relation_names: 关系名称
    relation_names = ["属于", "位于", "创立于", "属于行业", "毕业于"]

    # entity_types: 实体类型
    entity_types = ["人物", "公司", "城市", "学校", "行业"]

    # test_ratio=0.1: 测试集比例
    test_ratio = 0.1

    # val_ratio=0.1: 验证集比例
    val_ratio = 0.1

    # random_state=42: 随机种子
    random_state = 42

    # --- 模型相关 ---
    # embedding_dim=64: 实体/关系嵌入维度
    #   RotatE在复数空间操作，所以实际维度=64(实部32+虚部32)
    embedding_dim = 64

    # gnn_hidden_dim=64: GNN隐藏层维度
    gnn_hidden_dim = 64

    # use_gnn=True: 是否使用GNN增强实体嵌入
    #   True: 实体嵌入 = 结构嵌入(GNN) + 可学习嵌入
    #   False: 只用可学习嵌入(类似标准RotatE)
    use_gnn = True

    # --- RotatE特定参数 ---
    # gamma=12.0: RotatE的间隔margin
    #   正样本分数应该 > 负样本分数 + gamma
    gamma = 12.0

    # epsilon=2.0: 嵌入范数的上界
    #   RotatE要求实体嵌入在超球面上，约束范数 ≤ epsilon
    epsilon = 2.0

    # --- 训练相关 ---
    # learning_rate=1e-3: 学习率
    learning_rate = 1e-3

    # weight_decay=1e-5: L2正则化(比分类任务小)
    weight_decay = 1e-5

    # epochs=200: 训练轮数
    epochs = 200

    # batch_size=512: 批次大小(三元组数量)
    batch_size = 512

    # neg_samples_per_pos=1: 每个正样本对应的负样本数
    neg_samples_per_pos = 1

    # early_stop_patience=100: 早停耐心(因为每5轮才评估一次)
    early_stop_patience = 100

    # --- 保存相关 ---
    save_dir = "gnn/output/knowledge_graph"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# Step 3: 合成知识图谱生成
# ============================================================
def generate_knowledge_graph(cfg):
    """
    生成合成知识图谱。

    【知识图谱结构】
    实体类型和数量:
    - 人物: 100人 (name_0 ~ name_99)
    - 公司: 50家 (company_0 ~ company_49)
    - 城市: 30个 (city_0 ~ city_29)
    - 学校: 20所 (school_0 ~ school_19)
    - 行业: 10个 (industry_0 ~ industry_9)
    总计: 210个实体

    关系类型:
    - 属于(0): 人物 → 公司 (员工所属公司)
    - 位于(1): 公司 → 城市 (公司所在城市)
    - 创立于(2): 公司 → 城市 (公司创立城市)
    - 属于行业(3): 公司 → 行业 (公司所属行业)
    - 毕业于(4): 人物 → 学校 (人物毕业学校)
    """
    np.random.seed(cfg.random_state)

    # 实体定义
    entities = {
        "人物": [f"人物_{i}" for i in range(100)],
        "公司": [f"公司_{i}" for i in range(50)],
        "城市": [f"城市_{i}" for i in range(30)],
        "学校": [f"学校_{i}" for i in range(20)],
        "行业": [f"行业_{i}" for i in range(10)],
    }

    # 为每个实体分配全局ID
    entity2id = {}
    id2entity = {}
    entity_type = {}  # entity_id → type_name
    idx = 0
    for type_name, entity_list in entities.items():
        for entity in entity_list:
            entity2id[entity] = idx
            id2entity[idx] = entity
            entity_type[idx] = type_name
            idx += 1

    num_entities = len(entity2id)
    cfg.num_entities = num_entities

    # 生成三元组
    triples = []

    # 关系0: 属于 (人物 → 公司)
    for i in range(100):
        h = entity2id[entities["人物"][i]]
        t = entity2id[entities["公司"][np.random.randint(0, 50)]]
        triples.append((h, 0, t))

    # 关系1: 位于 (公司 → 城市)
    for i in range(50):
        h = entity2id[entities["公司"][i]]
        t = entity2id[entities["城市"][np.random.randint(0, 30)]]
        triples.append((h, 1, t))

    # 关系2: 创立于 (公司 → 城市)
    for i in range(50):
        h = entity2id[entities["公司"][i]]
        t = entity2id[entities["城市"][np.random.randint(0, 30)]]
        triples.append((h, 2, t))

    # 关系3: 属于行业 (公司 → 行业)
    for i in range(50):
        h = entity2id[entities["公司"][i]]
        t = entity2id[entities["行业"][np.random.randint(0, 10)]]
        triples.append((h, 3, t))

    # 关系4: 毕业于 (人物 → 学校)
    for i in range(100):
        h = entity2id[entities["人物"][i]]
        t = entity2id[entities["学校"][np.random.randint(0, 20)]]
        triples.append((h, 4, t))

    # 添加更多三元组增加密度
    # 额外的"属于"关系(兼职等)
    for _ in range(50):
        h = entity2id[entities["人物"][np.random.randint(0, 100)]]
        t = entity2id[entities["公司"][np.random.randint(0, 50)]]
        triples.append((h, 0, t))

    # 额外的"毕业于"关系(双学位等)
    for _ in range(30):
        h = entity2id[entities["人物"][np.random.randint(0, 100)]]
        t = entity2id[entities["学校"][np.random.randint(0, 20)]]
        triples.append((h, 4, t))

    # 去重
    triples = list(set(triples))

    print(f"知识图谱统计:")
    print(f"  实体数: {num_entities}")
    print(f"  关系类型: {cfg.num_relations}")
    print(f"  三元组数: {len(triples)}")
    print(f"  实体类型分布: {', '.join(f'{k}:{len(v)}' for k, v in entities.items())}")

    # 统计每种关系的数量
    rel_counts = defaultdict(int)
    for h, r, t in triples:
        rel_counts[cfg.relation_names[r]] += 1
    print(f"  关系分布: {dict(rel_counts)}")

    return triples, entity2id, id2entity, entity_type, entities


def split_triples(triples, cfg):
    """划分训练/验证/测试三元组"""
    np.random.seed(cfg.random_state)
    indices = np.random.permutation(len(triples))
    n = len(triples)
    n_test = int(n * cfg.test_ratio)
    n_val = int(n * cfg.val_ratio)

    test_indices = indices[:n_test]
    val_indices = indices[n_test:n_test + n_val]
    train_indices = indices[n_test + n_val:]

    train_triples = [triples[i] for i in train_indices]
    val_triples = [triples[i] for i in val_indices]
    test_triples = [triples[i] for i in test_indices]

    print(f"  训练三元组: {len(train_triples)} | 验证: {len(val_triples)} | 测试: {len(test_triples)}")

    return train_triples, val_triples, test_triples


def build_kg_graph(triples, num_entities, cfg):
    """
    从三元组构建PyG图对象(用于GNN)。

    【如何把知识图谱转为普通图？】
    知识图谱是有向多重图(同一对节点间可以有多条不同类型的边)
    PyG的Data对象可以存储边类型信息(edge_attr)

    转换方法: 所有三元组的边合并为一个大图
    - 边: (h, t) 有向边
    - 边属性: 关系类型r
    """
    # 构建边
    edge_list = []
    edge_type_list = []
    for h, r, t in triples:
        edge_list.append([h, t])
        edge_type_list.append(r)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type_list, dtype=torch.long)

    # 节点特征: One-hot实体类型
    type_to_id = {t: i for i, t in enumerate(cfg.entity_types)}
    x = torch.zeros(num_entities, len(cfg.entity_types))
    # 需要entity_type信息来设置one-hot
    # 简化: 用度特征
    degree = torch.zeros(num_entities)
    for h, t in edge_list:
        degree[h] += 1
        degree[t] += 1
    x = torch.stack([degree, degree / degree.max()], dim=1)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    return data


# ============================================================
# Step 4: 模型定义
# ============================================================
class RotatEScore(nn.Module):
    """
    RotatE评分函数: 在复数空间中，关系是头实体到尾实体的旋转。

    【RotatE核心公式】
    score(h, r, t) = -||h ∘ r - t||²

    其中:
    - h, r, t ∈ C^d (复数向量)
    - ∘ 是逐元素复数乘法(Hadamard积)
    - h ∘ r: 头实体"旋转"关系r的角度

    【为什么用旋转？】
    不同的关系模式可以用不同的旋转角度表示:
    - 对称关系(配偶): r旋转180°, h∘r = t且t∘r = h
    - 反对称关系(父子): r旋转非0°, h∘r = t但t∘r ≠ h
    - 组合关系(祖父 = 父亲∘父亲): r3 = r1 ∘ r2
    - 逆关系(学生-老师): r2 = r1的逆旋转

    【复数乘法实现】
    (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    用实数表示: Re(h∘r) = Re(h)·Re(r) - Im(h)·Im(r)
                Im(h∘r) = Re(h)·Im(r) + Im(h)·Re(r)
    """

    def __init__(self, embedding_dim, gamma=12.0, epsilon=2.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.epsilon = epsilon

        # 嵌入维度在复数空间中是embedding_dim//2
        # 每个复数分量需要2个实数(实部+虚部)
        self.half_dim = embedding_dim // 2

    def forward(self, head_emb, relation_emb, tail_emb):
        """
        计算三元组分数。

        参数:
            head_emb: 头实体嵌入 (batch, embedding_dim)
                      前 half_dim 是实部，后 half_dim 是虚部
            relation_emb: 关系嵌入 (batch, embedding_dim)
                          关系只需角度，所以先归一化到单位圆
            tail_emb: 尾实体嵌入 (batch, embedding_dim)

        返回:
            分数: (batch,), 越大表示越可能是真实三元组
        """
        # 拆分实部和虚部
        h_re, h_im = head_emb[:, :self.half_dim], head_emb[:, self.half_dim:]
        r_re, r_im = relation_emb[:, :self.half_dim], relation_emb[:, self.half_dim:]
        t_re, t_im = tail_emb[:, :self.half_dim], tail_emb[:, self.half_dim:]

        # 关系嵌入归一化: 只保留方向(角度)，去掉模长
        # r = r / ||r||，确保关系是单位复数(在单位圆上)
        r_norm = torch.sqrt(r_re ** 2 + r_im ** 2 + 1e-10)
        r_re = r_re / r_norm
        r_im = r_im / r_norm

        # 复数乘法: h ∘ r
        # (h_re + h_im*i) * (r_re + r_im*i) = (h_re*r_re - h_im*r_im) + (h_re*r_im + h_im*r_re)*i
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        # 距离: ||h∘r - t||²
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im
        distance = diff_re ** 2 + diff_im ** 2

        # 分数: gamma - distance (距离越小分数越高)
        score = self.gamma - distance.sum(dim=1)

        return score


class KnowledgeGraphModel(nn.Module):
    """
    知识图谱表示学习模型: GNN增强的RotatE

    【架构】
    1. 实体嵌入: 可学习嵌入 + GNN结构嵌入(可选)
    2. 关系嵌入: 可学习嵌入
    3. 评分函数: RotatE

    【为什么用GNN增强？】
    标准RotatE: 每个实体只有一个可学习的嵌入向量
    GNN增强: 实体嵌入 = 可学习嵌入 + 从图结构中学习的嵌入
    - GNN能捕捉实体的邻居结构(连接模式)
    - 相似结构的实体(如同一行业的公司)嵌入更接近
    """

    def __init__(self, num_entities, num_relations, embedding_dim,
                 gnn_hidden_dim, use_gnn=True, gamma=12.0, epsilon=2.0):
        super().__init__()
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.use_gnn = use_gnn
        self.half_dim = embedding_dim // 2

        # 实体嵌入(可学习)
        # 初始化: 在超球面上均匀分布
        entity_emb = torch.zeros(num_entities, embedding_dim)
        nn.init.xavier_uniform_(entity_emb)
        # 约束嵌入范数 ≤ epsilon
        with torch.no_grad():
            norm = entity_emb.norm(p=2, dim=1, keepdim=True)
            entity_emb = entity_emb * (epsilon / norm.clamp(min=epsilon))
        self.entity_embedding = nn.Parameter(entity_emb)

        # 关系嵌入(可学习)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)

        # GNN编码器(可选)
        if use_gnn:
            self.gnn_conv1 = GCNConv(embedding_dim, gnn_hidden_dim)
            self.gnn_bn1 = nn.BatchNorm1d(gnn_hidden_dim)
            self.gnn_conv2 = GCNConv(gnn_hidden_dim, embedding_dim)
            self.gnn_bn2 = nn.BatchNorm1d(embedding_dim)

            # 拼接后的投影层
            self.combine = nn.Linear(embedding_dim * 2, embedding_dim)

        # 评分函数
        self.score_fn = RotatEScore(embedding_dim, gamma, epsilon)

    def get_entity_embeddings(self, edge_index=None):
        """
        获取实体嵌入(可学习 + GNN增强)

        如果use_gnn=True且提供了edge_index:
          实体嵌入 = 可学习嵌入 + GNN编码的结构嵌入
        否则:
          实体嵌入 = 可学习嵌入
        """
        base_emb = self.entity_embedding

        if self.use_gnn and edge_index is not None:
            # GNN编码
            x = self.gnn_conv1(base_emb, edge_index)
            x = self.gnn_bn1(x)
            x = F.relu(x)
            x = self.gnn_conv2(x, edge_index)
            x = self.gnn_bn2(x)
            x = F.relu(x)

            # 拼接可学习嵌入和GNN嵌入
            combined = torch.cat([base_emb, x], dim=1)
            return self.combine(combined)

        return base_emb

    def forward(self, head_ids, relation_ids, tail_ids, edge_index=None):
        """
        前向传播: 计算三元组分数

        参数:
            head_ids: 头实体ID (batch,)
            relation_ids: 关系ID (batch,)
            tail_ids: 尾实体ID (batch,)
            edge_index: 图的边(用于GNN)
        """
        entity_emb = self.get_entity_embeddings(edge_index)

        head_emb = entity_emb[head_ids]
        relation_emb = self.relation_embedding(relation_ids)
        tail_emb = entity_emb[tail_ids]

        scores = self.score_fn(head_emb, relation_emb, tail_emb)
        return scores

    def score_triples(self, triples, edge_index=None):
        """计算三元组分数的便捷方法"""
        head_ids = triples[:, 0]
        relation_ids = triples[:, 1]
        tail_ids = triples[:, 2]
        return self.forward(head_ids, relation_ids, tail_ids, edge_index)


# ============================================================
# Step 5: 训练和评估
# ============================================================
def generate_negative_samples(triples, num_entities, num_neg=1):
    """
    生成负样本: 替换头实体或尾实体。

    【负采样策略】
    对于正样本(h, r, t):
    - 替换头: (h', r, t), h' ≠ h
    - 替换尾: (h, r, t'), t' ≠ t

    随机选择替换头还是替换尾(各50%概率)
    替换的实体从所有实体中随机选择

    【为什么不替换关系？】
    替换关系容易产生"假负样本"(实际存在但不在训练集中的三元组)
    替换实体产生假负样本的概率更低(因为同一对实体+关系通常是唯一的)
    """
    neg_triples = []
    for h, r, t in triples:
        for _ in range(num_neg):
            if np.random.random() < 0.5:
                # 替换头实体
                h_new = np.random.randint(0, num_entities)
                while h_new == h:
                    h_new = np.random.randint(0, num_entities)
                neg_triples.append((h_new, r, t))
            else:
                # 替换尾实体
                t_new = np.random.randint(0, num_entities)
                while t_new == t:
                    t_new = np.random.randint(0, num_entities)
                neg_triples.append((h, r, t_new))

    return neg_triples


def train_one_epoch(model, train_triples, kg_data, optimizer, cfg):
    """训练一个epoch"""
    model.train()

    # 随机打乱训练数据
    indices = np.random.permutation(len(train_triples))
    total_loss = 0
    num_batches = 0

    for start in range(0, len(train_triples), cfg.batch_size):
        end = min(start + cfg.batch_size, len(train_triples))
        batch_indices = indices[start:end]

        # 正样本
        pos_batch = [train_triples[i] for i in batch_indices]
        pos_triples = torch.tensor(pos_batch, dtype=torch.long, device=cfg.device)

        # 负样本
        neg_batch = generate_negative_samples(pos_batch, cfg.num_entities, cfg.neg_samples_per_pos)
        neg_triples = torch.tensor(neg_batch, dtype=torch.long, device=cfg.device)

        # 计算分数
        edge_index = kg_data.edge_index.to(cfg.device) if kg_data is not None else None
        pos_scores = model.score_triples(pos_triples, edge_index)
        neg_scores = model.score_triples(neg_triples, edge_index)

        # Margin-based损失: max(0, γ + neg_score - pos_score)
        # 目标: 正样本分数 > 负样本分数 + margin
        loss = F.relu(cfg.gamma + neg_scores - pos_scores).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, triples, kg_data, cfg):
    """
    评估知识图谱模型。

    【评估指标】
    - MRR(Mean Reciprocal Rank): 正确实体的排名倒数的平均值
      越高越好，1.0=完美
    - Hits@1: 排名第1的比例
    - Hits@3: 排名前3的比例
    - Hits@10: 排名前10的比例

    【评估方式】
    对于每个测试三元组(h, r, t):
    1. 替换t为所有可能的实体，计算分数
    2. 按分数排序，看正确的t排第几
    3. 这叫"过滤"设置: 排序时去掉训练集中已知的正确三元组
    """
    model.eval()
    edge_index = kg_data.edge_index.to(cfg.device) if kg_data is not None else None

    all_triples = torch.tensor(triples, dtype=torch.long, device=cfg.device)
    ranks = []

    # 评估尾实体预测(替换t)
    for i in range(len(triples)):
        h, r, t = triples[i]

        # 构造查询: (h, r, ?)
        head_ids = torch.full((cfg.num_entities,), h, dtype=torch.long, device=cfg.device)
        relation_ids = torch.full((cfg.num_entities,), r, dtype=torch.long, device=cfg.device)
        tail_ids = torch.arange(cfg.num_entities, dtype=torch.long, device=cfg.device)

        scores = model(head_ids, relation_ids, tail_ids, edge_index)

        # 排名(分数越高越好，排名越前越好)
        _, indices = scores.sort(descending=True)
        rank = (indices == t).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    mrr = (1.0 / ranks).mean()
    hits1 = (ranks <= 1).mean()
    hits3 = (ranks <= 3).mean()
    hits10 = (ranks <= 10).mean()

    return {"mrr": mrr, "hits@1": hits1, "hits@3": hits3, "hits@10": hits10}


def train(model, train_triples, val_triples, kg_data, cfg):
    """完整训练流程"""
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_val_mrr = 0
    patience_counter = 0
    best_model_state = None
    history = {"train_loss": [], "val_mrr": [], "val_hits10": []}

    print(f"\n{'='*60}")
    print(f"开始训练 (GNN增强: {cfg.use_gnn})...")
    print(f"{'='*60}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_triples, kg_data, optimizer, cfg)

        # 每5轮验证一次(评估比较慢)
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = evaluate(model, val_triples, kg_data, cfg)
            history["train_loss"].append(train_loss)
            history["val_mrr"].append(val_metrics["mrr"])
            history["val_hits10"].append(val_metrics["hits@10"])

            print(f"Epoch {epoch:3d}/{cfg.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val MRR: {val_metrics['mrr']:.4f} | "
                  f"Hits@10: {val_metrics['hits@10']:.4f}")

            if val_metrics["mrr"] > best_val_mrr:
                best_val_mrr = val_metrics["mrr"]
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stop_patience // 5:
                    print(f"\n⚠ 早停触发")
                    break
        else:
            history["train_loss"].append(train_loss)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(cfg.device)
        print(f"\n✓ 已恢复最优模型 (Val MRR: {best_val_mrr:.4f})")

    return model, history


# ============================================================
# Step 6: 可视化
# ============================================================
def plot_training_curves(history, cfg):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs_loss = range(1, len(history["train_loss"]) + 1)
    epochs_val = range(1, len(history["val_mrr"]) * 10, 10)

    axes[0].plot(epochs_loss, history["train_loss"], "b-", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("训练损失")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_val, history["val_mrr"], "r-", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MRR")
    axes[1].set_title("验证MRR")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_val, history["val_hits10"], "g-", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Hits@10")
    axes[2].set_title("验证Hits@10")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("知识图谱表示学习训练曲线", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()


def plot_entity_embeddings(model, entity_type, cfg, num_samples=200):
    """
    可视化实体嵌入(t-SNE降维)。

    同类型的实体应该聚集在一起:
    - 所有"人物"聚在一起
    - 所有"公司"聚在一起
    - ...
    """
    model.eval()
    with torch.no_grad():
        entity_emb = model.entity_embedding.cpu().numpy()

    # t-SNE降维
    # 如果实体太多，采样一部分
    if entity_emb.shape[0] > num_samples:
        indices = np.random.choice(entity_emb.shape[0], num_samples, replace=False)
        emb_sample = entity_emb[indices]
        type_sample = [entity_type.get(i, "未知") for i in indices]
    else:
        emb_sample = entity_emb
        type_sample = [entity_type.get(i, "未知") for i in range(entity_emb.shape[0])]

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb_sample) - 1))
    emb_2d = tsne.fit_transform(emb_sample)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, len(cfg.entity_types)))
    type_to_color = {t: colors[i] for i, t in enumerate(cfg.entity_types)}

    for type_name in cfg.entity_types:
        mask = [t == type_name for t in type_sample]
        if sum(mask) == 0:
            continue
        x_vals = emb_2d[mask, 0] if hasattr(mask, '__iter__') else [emb_2d[i, 0] for i, m in enumerate(mask) if m]
        y_vals = emb_2d[mask, 1] if hasattr(mask, '__iter__') else [emb_2d[i, 1] for i, m in enumerate(mask) if m]

        # 手动筛选
        xs, ys = [], []
        for i, m in enumerate(mask):
            if m:
                xs.append(emb_2d[i, 0])
                ys.append(emb_2d[i, 1])

        ax.scatter(xs, ys, c=[type_to_color[type_name]], label=type_name, s=30, alpha=0.7)

    ax.set_title("实体嵌入可视化(t-SNE)", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "entity_embeddings.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 实体嵌入已保存: {save_path}")
    plt.close()


def plot_relation_examples(model, triples, id2entity, cfg, num_examples=5):
    """
    可视化关系推理示例。

    展示模型对三元组的预测分数:
    - 高分: 模型认为该三元组很可能为真
    - 低分: 模型认为该三元组很可能为假
    """
    model.eval()
    with torch.no_grad():
        triple_tensor = torch.tensor(triples[:num_examples], dtype=torch.long, device=cfg.device)
        edge_index = None
        scores = model.score_triples(triple_tensor, edge_index).cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    labels = []
    for i, (h, r, t) in enumerate(triples[:num_examples]):
        h_name = id2entity[h]
        r_name = cfg.relation_names[r]
        t_name = id2entity[t]
        labels.append(f"({h_name}, {r_name}, {t_name})")

    bars = ax.barh(range(num_examples), scores, color="steelblue", alpha=0.8)
    ax.set_yticks(range(num_examples))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("RotatE分数(越高越可能为真)")
    ax.set_title("三元组评分示例")
    ax.grid(True, alpha=0.3, axis="x")

    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(cfg.save_dir, "relation_examples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 关系示例已保存: {save_path}")
    plt.close()


# ============================================================
# Step 7: 主函数
# ============================================================
def main():
    print("=" * 60)
    print("GNN 知识图谱表示学习 - RotatE + GNN")
    print("=" * 60)

    cfg = CONFIG()
    print(f"设备: {cfg.device}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # 生成知识图谱
    print("\n生成合成知识图谱...")
    triples, entity2id, id2entity, entity_type, entities = generate_knowledge_graph(cfg)

    # 划分数据集
    train_triples, val_triples, test_triples = split_triples(triples, cfg)

    # 构建KG图(用于GNN)
    kg_data = build_kg_graph(train_triples, cfg.num_entities, cfg)

    # 创建模型
    model = KnowledgeGraphModel(
        num_entities=cfg.num_entities,
        num_relations=cfg.num_relations,
        embedding_dim=cfg.embedding_dim,
        gnn_hidden_dim=cfg.gnn_hidden_dim,
        use_gnn=cfg.use_gnn,
        gamma=cfg.gamma,
        epsilon=cfg.epsilon,
    ).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: RotatE + GNN({'启用' if cfg.use_gnn else '禁用'})")
    print(f"总参数量: {total_params:,}")

    # 训练
    model, history = train(model, train_triples, val_triples, kg_data, cfg)

    # 测试集评估
    print(f"\n{'='*60}")
    print("测试集评估...")
    test_metrics = evaluate(model, test_triples, kg_data, cfg)
    print(f"测试 MRR: {test_metrics['mrr']:.4f} | Hits@1: {test_metrics['hits@1']:.4f} | "
          f"Hits@3: {test_metrics['hits@3']:.4f} | Hits@10: {test_metrics['hits@10']:.4f}")

    # 保存模型
    model_path = os.path.join(cfg.save_dir, "gnn_kg_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
    }, model_path)
    print(f"✓ 模型已保存: {model_path}")

    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, cfg)
    plot_entity_embeddings(model, entity_type, cfg)
    plot_relation_examples(model, test_triples, id2entity, cfg)

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
