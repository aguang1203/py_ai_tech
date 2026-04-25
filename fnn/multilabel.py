"""
=============================================================================
FNN 多标签分类任务模板 (Feedforward Neural Network for Multi-Label Classification)
=============================================================================

【原理】
多标签分类：每个样本可以同时拥有多个标签(标签之间不互斥)。
例如：一篇新闻可以同时是"体育"+"国际"两个类别。

与多分类的关键区别：
- 多分类：每个样本只属于1个类别(互斥)，用Softmax(概率和=1)
- 多标签：每个标签独立判断0或1，用Sigmoid(每个标签概率独立，不做归一化)

【本数据集说明】
moral_foundation_news.csv 是"道德基础判断"数据集：
- 每条样本是一段环境新闻文本(query)
- 标签(response)有4种：责任/利益、实用/理想、创新/守旧、非道德
- 原始标签是单标签(每条只属于1类)，我们将其One-Hot编码为4维0/1向量
- 用BCEWithLogitsLoss训练，等效于多标签方式处理单标签分类

【文本特征提取：TF-IDF】
文本不能直接输入FNN，需要先转为数值向量。TF-IDF是最常用的方法：
- TF (Term Frequency): 词在文档中出现的频率
- IDF (Inverse Document Frequency): 越常见的词权重越低(如"的"、"是")
- TF-IDF = TF × IDF：既考虑词的重要性，又过滤掉常见停用词
- 本代码用sklearn的TfidfVectorizer自动完成

【应用场景】
- 新闻标签：一篇新闻可同时有"体育"、"国际"、"财经"标签
- 电影类型：一部电影可同时是"动作"、"喜剧"、"科幻"
- 医学诊断：一个病人可同时患"高血压"、"糖尿病"
- 图片标签：一张图片可同时包含"猫"、"草地"、"蓝天"
- 文本情感：一段文本可同时表达"愤怒"、"失望"

【与分类/回归的区别】
- 多标签: 输出层=标签数+Sigmoid，损失=BCEWithLogitsLoss，阈值=0.5判断
- 分类: 输出层=类别数，损失=CrossEntropyLoss，取argmax
- 回归: 输出层=1，损失=MSELoss，输出连续值

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 替换 load_data() 为你自己的数据加载逻辑
3. 直接运行: python multilabel.py
=============================================================================
"""

# ============================================================
# Step 1: 导入必要的库
# ============================================================
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    classification_report,
)

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

now = datetime.datetime.now()


# ============================================================
# Step 2: 配置超参数
# ============================================================
class CONFIG:
    """超参数配置中心 —— 所有可调参数集中在此，方便统一管理和实验对比。"""

    datasets_path = "datasets/multilabel-moral_foundation_news.csv"

    # --- 数据相关 ---
    # num_features: 输入特征维度，设为 None 表示自动检测
    #   原因：TF-IDF后特征数取决于词汇量，手动写容易出错
    #   自动检测在 load_data() 中完成：cfg.num_features = X.shape[1]
    num_features = None

    # num_labels=4: 标签数量(多标签任务中，每个标签独立判断0/1)
    #   本数据集有4种道德维度：责任/利益、实用/理想、创新/守旧、非道德
    #   注意：标签数 ≠ 类别数！多分类中4类只需1个标签列，多标签中4标签=4列0/1
    num_labels = 4

    # label_names: 标签名称列表(用于可视化输出，按One-Hot列的顺序)
    label_names = ["创新/守旧", "非道德", "实用/理想", "责任/利益"]

    # test_size=0.2: 训练集:测试集 = 8:2
    #   为什么不是7:3？2004条数据量偏小，8:2保证训练集足够(1600+)，
    #   测试集也有400条，评估指标方差不大
    test_size = 0.2

    # random_state=42: 固定随机种子，确保每次运行结果可复现
    random_state = 42

    # --- TF-IDF特征提取 ---
    # tfidf_max_features=500: 最多保留500个词的特征
    #   为什么不是2000？2004条数据(训练集1603)配2000维特征，特征数>样本数
    #   高维稀疏特征+小样本→快速过拟合(epoch15后F1开始下降)
    #   500维让特征数<样本数的1/3，模型更不容易死记硬背
    #   经验：max_features ≈ 样本数/3 ~ 样本数/2，小数据集偏小值
    tfidf_max_features = 500

    # tfidf_ngram_range=(1,2): 提取1-gram和2-gram
    #   1-gram: 单个词(如"碳","排放")
    #   2-gram: 连续两个词(如"碳排放","碳中和")
    #   为什么包含2-gram？很多道德判断依赖词组而非单字(如"碳中和"≠"碳"+"中和")
    #   为什么不包含3-gram？短文本(81字)中3-gram太稀疏，几乎不重复出现
    tfidf_ngram_range = (1, 2)

    # tfidf_min_df=2: 一个词至少在2个文档中出现才保留
    #   为什么？只在1个文档出现的词(错别字、极罕见专有名词)对分类无帮助
    #   过滤掉它们减少噪声和维度，2004条文档中min_df=2很合理
    tfidf_min_df = 2

    # tfidf_max_df=0.95: 在超过95%文档中出现的词被过滤
    #   为什么？几乎所有文档都出现的词(如"的","是")没有区分度
    #   类似停用词过滤，但基于统计而非预定义词表
    tfidf_max_df = 0.95

    # --- 模型相关 ---
    # hidden_dims: 隐藏层维度列表，"漏斗"型逐层压缩
    #   为什么是 [128, 64]？
    #     - TF-IDF后约500维输入 → 128：第一层适度压缩
    #     - 128 → 64：提取更抽象的语义特征
    #     - 为什么不是 [256, 128, 64]？500维输入到256太宽，小样本容易过拟合
    #   经验法则：第一层隐藏维度 ≈ 输入维度的1/4~1/2
    hidden_dims = [128, 64]

    # dropout_rate=0.4: 训练时随机丢弃40%的神经元
    #   为什么是0.4而不是0.3？
    #     - 2004条数据+TF-IDF特征，过拟合风险很高(epoch15就开始过拟合)
    #     - 需要更强的正则化，0.4比0.3丢弃更多，强制网络不依赖单一路径
    #   原理：Dropout迫使网络不依赖任何单个神经元，提升泛化能力
    dropout_rate = 0.4

    # threshold=0.5: 预测阈值，sigmoid概率>threshold则预测为1
    #   为什么是0.5？默认值，适合标签分布较均衡的情况
    #   可以在阈值调优步骤中搜索最优值(见Step 7)
    threshold = 0.5

    # single_label_mode=True: 单标签模式(每样本只选概率最高的1个标签)
    #   为什么需要？本数据集每条样本只有1个标签(互斥类别)
    #   多标签模式可能预测0个或2+个标签，对单标签数据不合理
    #   单标签模式下，不管概率多低，都选最高的那个标签
    #   如果你的数据是真正的多标签(可同时有多个1)，设为False
    single_label_mode = True

    # --- 训练相关 ---
    # batch_size=32: 每次梯度更新使用32个样本
    #   为什么不是64？2004条数据量小，batch=32时每个epoch更新50次，
    #   batch=64只有25次，更新频率低，收敛慢
    #   小数据集用小batch更合适
    batch_size = 32

    # learning_rate=1e-4: Adam优化器的初始学习率
    #   为什么不是3e-4？TF-IDF特征+pos_weight加权，3e-4导致验证损失不稳定
    #     实测：3e-4下验证损失从1.8飙升到4.1，22轮就早停
    #   为什么不是2e-4？实测2e-4的F1(0.54)略低于1e-4(0.56)
    #   1e-4虽慢但稳，配合500维特征+强正则化，训练52轮充分收敛
    learning_rate = 1e-4

    # epochs=150: 最大训练轮数
    #   早停机制会在验证损失不再下降时自动停止，150只是上限
    epochs = 150

    # weight_decay=5e-4: L2正则化强度
    #   原理：在损失函数中加入 λ·Σ(w²)，惩罚过大的权重
    #   防止模型"记住"训练数据的噪声(过拟合)
    #   为什么是5e-4而不是1e-4？小样本+高维特征需要更强的正则化
    weight_decay = 5e-4

    # --- 早停策略 ---
    # early_stop_patience=20: 验证损失连续20轮不下降就停止
    #   为什么是20？配合学习率调度器，降低LR后改善缓慢，需要足够耐心
    early_stop_patience = 20

    save_best_only = True  # 只保存验证集最优模型(而非最后一轮)

    # --- 学习率调度器 ---
    # lr_factor=0.5: 每次LR减半
    # lr_patience=8: 连续8轮无改善才降LR
    # lr_min=1e-6: LR下限
    lr_factor = 0.5
    lr_patience = 8
    lr_min = 1e-6

    # --- 梯度裁剪 ---
    # max_grad_norm=1.0: 梯度L2范数的上限
    #   不平衡数据下少数类梯度可能很大，裁剪防止训练崩溃
    max_grad_norm = 1.0

    # --- 类别不平衡处理 ---
    # use_pos_weight=True: 是否在BCEWithLogitsLoss中使用pos_weight
    #   原理：给正样本更大的权重，让模型关注少数类
    #   pos_weight[i] = sqrt(负样本数 / 正样本数)，sqrt平滑避免过度矫正
    use_pos_weight = True

    # --- 目标列 ---
    # target_col="response": 标签列的列名
    #   本数据集的标签在"response"列，文本在"query"列
    target_col = "response"

    # --- 文本列 ---
    # text_col="query": 文本特征的列名
    text_col = "query"

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Step 3: 加载和预处理数据
# ============================================================
def load_data(cfg):
    """
    加载多标签数据并预处理。

    【多标签数据格式】
    X: (n_samples, n_features) — 特征矩阵(TF-IDF稀疏矩阵→密集)
    y: (n_samples, n_labels)   — 标签矩阵，每个标签是0或1

    【文本→特征：TF-IDF流程】
    1. TfidfVectorizer将文本转为TF-IDF稀疏矩阵
    2. StandardScaler标准化特征(中心化+缩放)
    3. 转为PyTorch密集张量

    【标签→多标签格式：One-Hot编码】
    原始标签如"责任/利益" → [0,0,0,1]
    每个类别独占一列，不存在虚假序关系
    """

    df = pd.read_csv(cfg.datasets_path)
    print(f"数据集大小: {df.shape}")
    print(f"标签分布:\n{df[cfg.target_col].value_counts()}")

    # ========== Step 3.1: TF-IDF 文本特征提取 ==========
    #
    # 为什么用TF-IDF而不是词袋(Bag of Words)？
    #   词袋: 只计算词出现次数，常见词("的","是")权重过高
    #   TF-IDF: TF×IDF，常见词的IDF低→权重低，关键词的IDF高→权重高
    #   效果：TF-IDF自动"降权"停用词，突出有区分度的词
    #
    # 参数说明：
    #   max_features=2000: 只保留TF-IDF值最高的2000个词/词组
    #   ngram_range=(1,2): 提取1-gram(单字)和2-gram(两字词组)
    #   min_df=2: 至少在2个文档中出现才保留(过滤极罕见词)
    #   max_df=0.95: 在超过95%文档中出现的词被过滤(过滤常见词)
    #   sublinear_tf=True: 用1+log(tf)替代原始tf，压缩高频词的影响
    #     为什么？一个词出现10次 ≠ 重要性是出现1次的10倍
    #     log压缩后，10次→2.3倍，更符合实际
    vectorizer = TfidfVectorizer(
        max_features=cfg.tfidf_max_features,
        ngram_range=cfg.tfidf_ngram_range,
        min_df=cfg.tfidf_min_df,
        max_df=cfg.tfidf_max_df,
        sublinear_tf=True,
    )

    # fit_transform: 在训练文本上学习词汇表+计算TF-IDF
    # 注意：这里先对全部文本fit，后面split后再分别transform
    # 更安全的做法是split后再fit(避免数据泄露)，但TF-IDF的词汇表泄露影响很小
    texts = df[cfg.text_col].fillna("").values
    X = vectorizer.fit_transform(texts).toarray()  # 稀疏矩阵→密集矩阵
    print(f"TF-IDF特征维度: {X.shape[1]} (max_features={cfg.tfidf_max_features})")

    # 自动更新特征维度
    if cfg.num_features is None:
        cfg.num_features = X.shape[1]
        print(f"自动检测特征维度: {cfg.num_features}")

    # ========== Step 3.2: 标签 One-Hot 编码 ==========
    #
    # 为什么用One-Hot而不是LabelEncoder？
    #   LabelEncoder: "责任/利益"→0, "创新/守旧"→1, ...
    #   暗示了虚假序关系(2>1>0)，模型会错误地认为编号大的类别"更大"
    #   One-Hot: 每个类别独占一列[0,0,0,1]，不存在大小关系
    #
    # 为什么这里用One-Hot而不是直接做多分类(CrossEntropyLoss)？
    #   本代码是多标签模板，用BCEWithLogitsLoss + Sigmoid
    #   对于单标签数据，One-Hot + BCEWithLogitsLoss仍然有效
    #   而且代码可以直接扩展到真正的多标签数据(多个1)
    y_series = df[cfg.target_col]
    # pd.get_dummies自动按字母序排列类别，与label_names顺序对应
    y_df = pd.get_dummies(y_series, dtype=float)
    # 确保列顺序与label_names一致
    y_df = y_df.reindex(columns=cfg.label_names, fill_value=0)
    y = y_df.values.astype(np.float32)

    print(f"标签矩阵形状: {y.shape}")
    for i, name in enumerate(cfg.label_names):
        pos_count = y[:, i].sum()
        print(f"  {name}: 正样本={int(pos_count)}, 比例={pos_count/len(y):.3f}")

    # ========== Step 3.3: 划分训练集和测试集 ==========
    # stratify=y: 分层抽样，保证训练集和测试集中各类比例一致
    #   为什么必须？数据不平衡(责任/利益55% vs 非道德13%)
    #   不用stratify可能测试集里"非道德"类别极少，评估不可靠
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    # ========== Step 3.4: 特征标准化 (Z-score标准化) ==========
    # 公式: x' = (x - μ) / σ
    # TF-IDF值范围差异大(0~0.8)，标准化后梯度更稳定
    # fit_transform vs transform:
    #   fit_transform: 在训练集上计算μ和σ，然后转换(训练集用)
    #   transform: 直接用已计算的μ和σ转换(测试集用)
    #   关键：测试集绝不能fit！否则会"偷看"测试集信息(数据泄露)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ========== Step 3.5: 转为PyTorch张量 ==========
    # 多标签: 标签是 float32 的二维张量 (n_samples, n_labels)
    # 为什么用float32而不是long？BCEWithLogitsLoss要求标签为float
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # ========== Step 3.6: 封装为DataLoader ==========
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return (
        train_loader,
        test_loader,
        X_test.to(cfg.device),
        y_test.to(cfg.device),
        scaler,
        vectorizer,
    )


# ============================================================
# Step 4: 定义FNN多标签分类模型
# ============================================================
class FNNMultiLabel(nn.Module):
    """
    前馈神经网络多标签分类器。

    【网络结构: "漏斗"型逐层压缩】
    输入(~2000维) → [BN→Linear→ReLU→Dropout](256) → [BN→Linear→ReLU→Dropout](128) →
    [BN→Linear→ReLU→Dropout](64) → Linear(4)

    【每一层的作用】
    - BatchNorm1d: 批归一化，稳定每层输入的分布，加速收敛
    - Linear: 全连接层，学习特征的线性组合权重
    - ReLU: 非线性激活，让网络能学习复杂的非线性关系
    - Dropout: 随机丢弃神经元，防止过拟合
    - 输出层Linear(4): 4个神经元对应4个标签，不加激活函数

    【为什么 BatchNorm 放在 Linear 之前？(Pre-Norm)】
    Pre-Norm: BatchNorm → Linear → ReLU (本代码采用)
    Post-Norm: Linear → ReLU → BatchNorm (原始论文)
    Pre-Norm优势：梯度流更稳定，对学习率更鲁棒，训练初期更稳定

    【为什么输出层不加Sigmoid？】
    BCEWithLogitsLoss = Sigmoid + BCELoss 的数值稳定版本
    直接用logits(未激活的输出)计算损失，避免Sigmoid的数值溢出问题
    预测时再手动加Sigmoid即可

    【Sigmoid vs Softmax：多标签为什么用Sigmoid？】
    - Sigmoid: 每个输出独立计算σ(x)，值域(0,1)，各标签概率互不影响
      适合多标签：一篇新闻可以同时"责任/利益"和"实用/理想"
    - Softmax: 所有输出归一化为概率分布，和=1，标签间互斥
      适合多分类：一篇新闻只能属于1个类别
    """

    def __init__(self, input_dim, hidden_dims, num_labels, dropout_rate=0.3):
        super(FNNMultiLabel, self).__init__()

        layers = []
        prev_dim = input_dim

        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.BatchNorm1d(prev_dim))
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # 输出层: num_labels个神经元，不加激活函数(BCEWithLogitsLoss内置Sigmoid)
        layers.append(nn.Linear(prev_dim, num_labels))

        self.network = nn.Sequential(*layers)

        # He初始化(Kaiming初始化)：专为ReLU激活函数设计的权重初始化方法
        #
        # 为什么需要初始化？默认初始化可能让每层输出的方差逐层放大或缩小
        #   → 梯度爆炸(方差放大)或梯度消失(方差缩小) → 训练失败
        #
        # He初始化原理：让每层输出的方差 ≈ 输入的方差(保持信号强度)
        #   具体做法：权重 W ~ N(0, sqrt(2/fan_in))
        #   fan_in = 输入神经元数，fan_in越大 → 权重越小 → 信号不会放大
        #   乘以2是因为ReLU会"杀死"一半的负值信号，需要补偿
        #
        # 与Xavier初始化的区别：
        #   Xavier: W ~ N(0, sqrt(1/fan_in))，适合Sigmoid/Tanh(不杀死信号)
        #   He:     W ~ N(0, sqrt(2/fan_in))，适合ReLU(补偿被杀死的信号)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 偏置初始化为0，让初始输出以0为中心

    def forward(self, x):
        # 输出shape: (batch_size, num_labels)，每个值是logit(未激活)
        return self.network(x)


# ============================================================
# Step 5: 训练和评估函数
# ============================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm=1.0):
    """
    训练一个epoch。

    标准训练5步循环（每个batch执行一次）：
    1. optimizer.zero_grad()  → 清零上一步的梯度(否则梯度会累加)
    2. outputs = model(x)     → 前向传播：输入x，计算预测值
    3. loss = criterion(...)  → 计算损失：衡量预测值与真实值的差距
    4. loss.backward()        → 反向传播：计算每个参数的梯度(∂Loss/∂w)
    5. optimizer.step()       → 更新参数：w = w - lr * gradient
    """
    model.train()  # 切换到训练模式(启用Dropout和BatchNorm的训练行为)
    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()       # Step 1: 清零梯度
        outputs = model(batch_x)    # Step 2: 前向传播
        loss = criterion(outputs, batch_y)  # Step 3: 计算损失
        loss.backward()             # Step 4: 反向传播，计算梯度

        # 梯度裁剪：在Step 4和Step 5之间插入
        # 原理：当所有参数梯度的L2范数 > max_grad_norm 时，等比缩放梯度
        #   ‖g‖ = sqrt(Σgᵢ²)，若 ‖g‖ > max_grad_norm，则 g = g * (max_grad_norm / ‖g‖)
        # 为什么需要？不平衡数据下，少数类样本可能产生很大的梯度
        #   不裁剪：参数一步跨太远，损失飙升
        #   裁剪后：梯度方向不变，只限制步长，训练更稳定
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()            # Step 5: 更新参数

        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    return total_loss / total_samples


def evaluate(model, test_loader, criterion, device, threshold=0.5, single_label_mode=False):
    """
    评估多标签模型。

    与训练的区别：
    - model.eval(): 切换到评估模式
      → Dropout关闭(所有神经元都参与)
      → BatchNorm使用全局统计量(而非当前batch的)
    - torch.no_grad(): 禁用梯度计算
      → 节省内存(不存储中间激活值)
      → 加速计算(不需要反向传播)

    【多标签评估指标】
    - Hamming Loss: 错误预测的标签比例，越小越好
    - F1(micro): 全局计算所有标签的F1，适合标签分布均衡
    - F1(macro): 每个标签单独算F1再平均，适合标签分布不均衡
    - Precision/Recall: 各标签的精确率和召回率

    【single_label_mode】
    当数据集实际是单标签(互斥类别)时，设为True：
    - 不用阈值判定，而是选概率最高的1个标签(argmax)
    - 避免预测0个或2+个标签的情况
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

            # Sigmoid → 概率
            probs = torch.sigmoid(outputs)

            if single_label_mode:
                # 单标签模式：每样本选概率最高的1个标签(argmax)
                # 为什么不用阈值？单标签数据每样本恰好1个标签
                #   阈值模式可能预测0个(概率都<0.5)或多个(概率都>0.5)
                #   argmax保证恰好1个，更符合单标签数据特性
                preds = torch.zeros_like(probs)
                max_indices = probs.argmax(dim=1)
                preds.scatter_(1, max_indices.unsqueeze(1), 1.0)
            else:
                # 多标签模式：概率 > 阈值 → 预测为1
                preds = (probs > threshold).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / total_samples
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return avg_loss, all_preds, all_labels


# ============================================================
# Step 6: 主训练流程
# ============================================================
def main():
    cfg = CONFIG()
    print(f"使用设备: {cfg.device}")

    # --- 加载数据 ---
    train_loader, test_loader, X_test, y_test, scaler, vectorizer = load_data(cfg)
    print(
        f"训练集大小: {len(train_loader.dataset)}, 测试集大小: {len(test_loader.dataset)}"
    )
    print(f"标签数量: {cfg.num_labels}")

    # 统计每个标签的正样本比例
    y_train_np = train_loader.dataset.tensors[1].numpy()
    for i, name in enumerate(cfg.label_names):
        pos_ratio = y_train_np[:, i].mean()
        print(f"  {name} 正样本比例: {pos_ratio:.2%}")

    # --- 创建模型 ---
    model = FNNMultiLabel(
        input_dim=cfg.num_features,
        hidden_dims=cfg.hidden_dims,
        num_labels=cfg.num_labels,
        dropout_rate=cfg.dropout_rate,
    ).to(cfg.device)
    print(f"\n模型结构:\n{model}")

    # --- 损失函数和优化器 ---
    # BCEWithLogitsLoss: 二元交叉熵 + Sigmoid 的数值稳定版本
    # 公式: -[y*log(σ(x)) + (1-y)*log(1-σ(x))]，对每个标签分别计算
    #
    # 【处理标签不平衡】设置pos_weight参数
    # pos_weight[i] = 负样本数 / 正样本数 (对每个标签分别计算)
    # 作用：增加正样本的权重，缓解正负样本不平衡问题
    #
    # 为什么用sqrt平滑？
    #   直接逆频率权重过于激进：
    #     "责任/利益"正样本多(55%)→pos_weight≈0.8，"非道德"少(13%)→pos_weight≈5.7
    #     5.7/0.8 = 7倍差距，模型会过度偏向少数类
    #   sqrt平滑后：sqrt(0.8)≈0.9, sqrt(5.7)≈2.4，2.4/0.9 = 2.7倍，更温和
    pos_weights = []
    for i in range(cfg.num_labels):
        pos_count = y_train_np[:, i].sum()
        neg_count = len(y_train_np) - pos_count
        if pos_count > 0:
            pw_raw = neg_count / pos_count  # 原始逆频率权重
            pw = np.sqrt(pw_raw)             # sqrt平滑
        else:
            pw = 1.0
        pos_weights.append(pw)
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(cfg.device)
    print(f"\n各标签pos_weight(sqrt平滑): {dict(zip(cfg.label_names, [f'{w:.2f}' for w in pos_weights]))}")

    if cfg.use_pos_weight:
        train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        train_criterion = nn.BCEWithLogitsLoss()
    # 验证损失函数：不带pos_weight
    # 为什么训练和验证用不同的criterion？
    #   pos_weight改变了损失的尺度(少数类损失被放大)，使验证损失不稳定
    #   例：pos_weight让验证损失从0.7→3.0(表面恶化)，但实际F1在改善
    #   用无权重的criterion评估，验证损失更准确反映真实误差
    #   早停和LR调度器基于真实误差做决策，更稳定
    val_criterion = nn.BCEWithLogitsLoss()

    # Adam优化器：结合Momentum(动量)和RMSprop(自适应学习率)
    # 优势：对学习率不太敏感，几乎不需要调
    # weight_decay: L2正则化，防止过拟合
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # ReduceLROnPlateau: 验证损失停滞时自动降低学习率
    # LR只降不升，比CosineAnnealingWarmRestarts更稳定
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.lr_min
    )

    # --- 训练循环 ---
    train_losses = []
    val_losses = []
    val_f1s = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # 早停机制(Early Stopping)原理：
    #   目的：防止过拟合——训练太久会让模型"死记"训练数据的噪声
    #   方法：监控验证损失，如果连续patience轮没有改善，说明模型开始过拟合
    #   恢复：加载验证损失最低时的模型参数(best_model_state)
    #
    #   训练损失↓ 验证损失↓ → 模型还在学习，继续训练(好)
    #   训练损失↓ 验证损失→ → 模型开始过拟合，应该停止(早停触发)

    print("\n开始训练...")
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(
            model, train_loader, train_criterion, optimizer, cfg.device, cfg.max_grad_norm
        )
        val_loss, val_preds, val_labels = evaluate(
            model, test_loader, val_criterion, cfg.device, cfg.threshold,
            single_label_mode=cfg.single_label_mode
        )

        # 计算验证F1
        val_f1 = f1_score(val_labels, val_preds, average="micro", zero_division=0)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        scheduler.step(val_loss)  # ReduceLROnPlateau: 传入验证损失

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"\n早停触发！在第 {epoch + 1} 轮停止训练")
                break

        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch + 1}/{cfg.epochs}] "
                f"训练损失: {train_loss:.4f} | "
                f"验证损失: {val_loss:.4f} | "
                f"验证F1(micro): {val_f1:.4f} | "
                f"LR: {current_lr:.2e}"
            )

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n已恢复最佳模型(验证损失: {best_val_loss:.4f})")

    # ============================================================
    # Step 7: 阈值调优(在最终评估之前，仅多标签模式需要)
    # ============================================================
    # 单标签模式下不需要阈值调优(直接用argmax)，跳过
    best_threshold = cfg.threshold
    if cfg.single_label_mode:
        print("\n单标签模式: 使用argmax预测，无需阈值调优")
    else:
        # 默认阈值0.5不一定最优，搜索最佳阈值
        # 原理：sigmoid输出是概率，阈值决定"多少概率算正类"
        #   阈值低(0.3)→更多预测为正→召回率高但精确率低
        #   阈值高(0.7)→更少预测为正→精确率高但召回率低
        #   搜索范围0.3~0.7：太低/太高都不实际
        print("\n阈值调优:")
        model.eval()
        with torch.no_grad():
            all_logits = model(X_test)
            all_probs = torch.sigmoid(all_logits).cpu().numpy()
        all_labels_np = y_test.cpu().numpy()

        best_f1 = 0
        for t in np.arange(0.3, 0.7, 0.05):
            preds_t = (all_probs > t).astype(float)
            f1_t = f1_score(all_labels_np, preds_t, average="micro", zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = t
        print(f"  最佳阈值: {best_threshold:.2f}, 对应F1(micro): {best_f1:.4f}")
        print(f"  (默认阈值0.5, 实际使用{best_threshold:.2f})")

    # ============================================================
    # Step 8: 最终评估
    # ============================================================
    _, all_preds, all_labels = evaluate(
        model, test_loader, val_criterion, cfg.device, best_threshold,
        single_label_mode=cfg.single_label_mode
    )

    # 多标签评估指标
    #
    # Hamming Loss: 错误预测的标签比例
    #   = (预测错误的标签数) / (总标签数)
    #   例：400个测试样本×4标签=1600个预测，错200个 → HL=0.125
    #   越低越好，0=完美，0.5=随机猜
    #
    # F1 (micro): 全局计算所有标签的TP/FP/FN，再算F1
    #   适合标签分布均衡时，受多数类影响更大
    #
    # F1 (macro): 每个标签单独算F1再平均
    #   适合标签分布不均衡时，每个标签权重相同
    #
    # Precision: 精确率 = TP / (TP + FP)
    #   预测为正的样本中，真正为正的比例
    #
    # Recall: 召回率 = TP / (TP + FN)
    #   真正为正的样本中，被正确预测的比例
    h_loss = hamming_loss(all_labels, all_preds)
    f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision_micro = precision_score(all_labels, all_preds, average="micro", zero_division=0)
    recall_micro = recall_score(all_labels, all_preds, average="micro", zero_division=0)

    print(f"\n{'='*50}")
    print(f"多标签分类评估指标:")
    print(f"  Hamming Loss:        {h_loss:.4f}  (越低越好，0=完美)")
    print(f"  F1 (micro):          {f1_micro:.4f}  (全局F1，受多数类影响大)")
    print(f"  F1 (macro):          {f1_macro:.4f}  (各标签F1平均，关注少数类)")
    print(f"  Precision (micro):   {precision_micro:.4f}")
    print(f"  Recall (micro):      {recall_micro:.4f}")
    print(f"{'='*50}")

    # 各标签详细报告
    print("\n各标签详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=cfg.label_names, zero_division=0))

    # ============================================================
    # Step 9: 可视化
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    # 横轴=训练轮数，纵轴=损失值
    # 理想情况：训练和验证损失都持续下降，最终趋于平稳
    # 过拟合信号：训练损失↓ 但 验证损失↑
    axes[0].plot(train_losses, label="训练损失")
    axes[0].plot(val_losses, label="验证损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("损失曲线")
    axes[0].legend()

    # F1曲线
    # 理想情况：F1持续上升，最终趋于平稳
    axes[1].plot(val_f1s, label="验证F1(micro)", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("F1曲线")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"fnn/fnn_multilabel_training_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png", dpi=150
    )
    plt.show()

    # ============================================================
    # Step 10: 单样本预测示例
    # ============================================================
    model.eval()
    with torch.no_grad():
        sample_x = X_test[:5]
        logits = model(sample_x)
        probs = torch.sigmoid(logits)     # logits → 概率

        if cfg.single_label_mode:
            # 单标签模式：选概率最高的标签
            preds = torch.zeros_like(probs)
            max_indices = probs.argmax(dim=1)
            preds.scatter_(1, max_indices.unsqueeze(1), 1.0)
        else:
            preds = (probs > best_threshold).float()

        print("\n单样本预测示例:")
        for i in range(min(5, len(sample_x))):
            true_labels = y_test[i].cpu().numpy().astype(int)
            pred_labels = preds[i].cpu().numpy().astype(int)
            prob_values = probs[i].cpu().numpy()
            true_names = [cfg.label_names[j] for j in range(cfg.num_labels) if true_labels[j] == 1]
            pred_names = [cfg.label_names[j] for j in range(cfg.num_labels) if pred_labels[j] == 1]
            print(
                f"  样本{i}: 真实={true_names}, "
                f"预测={pred_names}, "
                f"各标签概率={{ {', '.join([f'{cfg.label_names[j]}:{prob_values[j]:.3f}' for j in range(cfg.num_labels)])} }}"
            )

    # ============================================================
    # Step 11: 模型保存与加载
    # ============================================================
    model_path = f"fnn/fnn_multilabel_model_{now.strftime('%Y-%m-%d_%H-%M-%S')}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")

    # loaded_model = FNNMultiLabel(
    #     input_dim=cfg.num_features,
    #     hidden_dims=cfg.hidden_dims,
    #     num_labels=cfg.num_labels,
    #     dropout_rate=cfg.dropout_rate,
    # ).to(cfg.device)
    # loaded_model.load_state_dict(
    #     torch.load(model_path, weights_only=True, map_location=cfg.device)
    # )
    # loaded_model.eval()
    # print("模型加载成功！")

    # ============================================================
    # Step 12: 对新数据进行预测(生产环境用法)
    # ============================================================
    # new_texts = ["新的新闻文本1", "新的新闻文本2"]
    # new_tfidf = vectorizer.transform(new_texts).toarray()  # 用训练时的vectorizer
    # new_scaled = scaler.transform(new_tfidf)
    # new_tensor = torch.tensor(new_scaled, dtype=torch.float32).to(cfg.device)
    # with torch.no_grad():
    #     logits = loaded_model(new_tensor)
    #     probs = torch.sigmoid(logits)
    #     preds = (probs > best_threshold).float()
    # for i, text in enumerate(new_texts):
    #     pred_names = [cfg.label_names[j] for j in range(cfg.num_labels) if preds[i][j] == 1]
    #     print(f"  '{text[:30]}...' → 预测标签: {pred_names}")

    print("\n多标签分类训练和评估完成！")


if __name__ == "__main__":
    main()
