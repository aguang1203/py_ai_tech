"""
=============================================================================
FNN 多分类任务模板 (Feedforward Neural Network for Classification)
=============================================================================

【原理】
前馈神经网络(FNN)通过多层全连接层 + 非线性激活函数，学习输入特征到离散类别的映射。
数据从输入层 → 隐藏层(逐层) → 输出层，单向流动，没有回路(前馈)。
输出层使用 Softmax 将 logits 转换为概率分布，取概率最大的类别作为预测结果。

【应用场景】
- 红酒品质分类 (3类: 低/中/高, 本模板使用的数据集)
- 鸢尾花分类 (3类)
- 手写数字识别 (10类, MNIST)
- 客户流失预测 (2类: 流失/不流失)
- 疾病诊断 (多类: 健康/轻症/重症)

【与回归/多标签的区别】
- 分类: 每个样本只属于1个类别，输出层维度=类别数，损失=CrossEntropyLoss
- 回归: 预测连续数值，输出层维度=1，损失=MSELoss
- 多标签: 每个样本可同时属于多个类别，输出层维度=标签数，损失=BCEWithLogitsLoss

【本数据集: 红酒品质分类】
- 数据来源: UCI Red Wine Quality (1599条)
- 特征: 11个化学指标(固定酸度、挥发酸度、柠檬酸、残糖、氯化物等)
- 原始标签: 3~8分(6类)，极度不平衡(3分仅10条, 8分仅18条)
- 优化策略: 合并为3类(低/中/高品质) + 加权损失函数
  → 品质3-4: 低品质(63条), 品质5-6: 中品质(1319条), 品质7-8: 高品质(217条)

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 替换 load_data() 为你自己的数据加载逻辑
3. 直接运行: python classification.py
=============================================================================
"""

# ============================================================
# Step 1: 导入必要的库
# ============================================================
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",
    "SimHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

now = datetime.datetime.now()


# ============================================================
# Step 2: 配置超参数 (修改这里即可适配你的数据)
# ============================================================
class CONFIG:
    """超参数配置中心 —— 所有可调参数集中在此，方便统一管理和实验对比。"""

    datasets_path = "/home/hjg/dev/datasets/red-winequality.csv"

    # --- 数据相关 ---
    # num_features: 输入特征维度，设为 None 表示自动检测
    #   原因：不同数据集特征数不同，自动检测避免手动写错
    num_features = None

    # num_classes=3: 本数据集将6类品质合并为3类(低/中/高)
    #   原因：原始6类(3~8分)极度不平衡：
    #     3分(10条), 4分(53条), 5分(681条), 6分(638条), 7分(199条), 8分(18条)
    #   问题1: 类别3和8各不到20条，模型无法有效学习这两个类别的特征
    #   问题2: 类别5和6占82%，模型倾向于全预测5或6就能得到82%准确率
    #   合并方案: 3+4→0(低品质63条), 5+6→1(中品质1319条), 7+8→2(高品质217条)
    #   合并后: 3类都有足够样本，且3类有明确的语义(低/中/高品质)
    #   如果你的数据类别均衡，设为实际类别数即可，不需要合并
    num_classes = 3

    # class_merge: 类别合并规则 {原始标签: 合并后标签}
    #   key=原始标签(红酒评分3~8), value=合并后标签(0/1/2)
    #   None表示不合并(直接用原始标签)
    class_merge = {3: 0, 4: 0, 5: 1, 6: 1, 7: 2, 8: 2}
    # class_names: 合并后各类别的名称(用于报告展示)
    class_names = ["低品质(3-4分)", "中品质(5-6分)", "高品质(7-8分)"]

    # test_size=0.2: 训练集:测试集 = 8:2
    #   1599条数据，8:2 → 训练1279条，测试320条
    #   为什么不是7:3？数据量本就不大，训练集太少模型学不好
    test_size = 0.2

    # random_state=42: 固定随机种子，确保每次运行结果可复现
    random_state = 42

    # --- 模型相关 ---
    # hidden_dims=[128, 64, 32]: 3层隐藏层，逐层压缩"漏斗"结构
    #   为什么是3层而不是2层？
    #     11维输入 → 3类输出，中间需要足够的非线性变换来学习复杂决策边界
    #     品质分类不是简单的线性可分问题(酸度和品质的关系是非线性的)
    #   为什么是[128, 64, 32]而不是[256, 128]？
    #     11维输入，第一层128已经足够宽(约12倍输入维度)
    #     3层比2层多一次非线性变换，在小数据集上泛化更好
    #   经验法则：总参数量不应超过训练样本数的1/10
    #     本模型参数 ≈ 11×128 + 128×64 + 64×32 + 32×3 ≈ 11,871
    #     训练样本1279条，比例 ≈ 9:1，合理
    hidden_dims = [128, 64, 32]

    # dropout_rate=0.3: 训练时随机丢弃30%的神经元
    #   为什么是0.3？
    #     分类任务比回归更需要正则化(决策边界更复杂，容易过拟合)
    #     0.3是分类任务的常用值，0.5太激进(小数据集会欠拟合)
    #   原理：Dropout迫使网络不依赖任何单个神经元，提升泛化能力
    #   类比：团队中随机有人请假，其他人必须学会替代，整体更健壮
    dropout_rate = 0.3

    # --- 训练相关 ---
    # batch_size=32: 每次梯度更新使用32个样本
    #   为什么不是64？只有1279个训练样本，batch_size=64每个epoch只更新20次
    #     batch_size=32每个epoch更新40次，更频繁的更新有助于收敛
    #   为什么不是16？太小梯度噪声大，训练不稳定
    batch_size = 32

    # learning_rate=5e-4: Adam优化器的初始学习率
    #   为什么不是1e-3(Adam默认)？小数据集+类别不平衡，1e-3容易震荡
    #   5e-4更保守，给模型更多时间适应少数类的梯度
    learning_rate = 5e-4

    # epochs=200: 最大训练轮数
    #   早停会自动控制，200只是上限
    epochs = 200

    # weight_decay=1e-4: L2正则化强度(权重衰减)
    #   原理：在损失函数中加入 λ·Σ(w²)，惩罚过大的权重
    #   效果：防止模型"记住"训练数据的噪声(过拟合)
    #   为什么是1e-4？经验值，太大(1e-2)会欠拟合，太小(1e-6)等于没有
    weight_decay = 1e-4

    # --- 早停策略 ---
    # early_stop_patience=20: 验证损失连续20轮不下降就停止
    #   为什么是20而不是10？配合学习率调度器，降低LR后改善缓慢
    #   需要足够耐心等它收敛
    early_stop_patience = 20

    # --- 类别不平衡处理 ---
    # use_class_weight=True: 是否在损失函数中使用类别权重
    #   原理：给少数类更大的权重，多数类更小的权重
    #   效果：模型不再"偏心"于多数类，被迫也关注少数类
    #   计算方式：weight_i = N / (C * n_i)，N=总样本数，C=类别数，n_i=第i类样本数
    #   例：低品质63条 → weight ≈ 1599/(3×63) ≈ 8.46
    #       中品质1319条 → weight ≈ 1599/(3×1319) ≈ 0.40
    #       高品质217条 → weight ≈ 1599/(3×217) ≈ 2.46
    #   含义：模型预测错1条低品质样本的惩罚 ≈ 预测错21条中品质样本
    use_class_weight = True

    # use_weighted_sampler=False: 是否使用加权随机采样器
    #   原理：训练时让少数类被采样的概率更高，多数类更低
    #   与class_weight的区别：
    #     class_weight: 改变损失函数，预测错少数类惩罚更大(软约束)
    #     weighted_sampler: 改变数据采样，少数类被看到次数更多(硬约束)
    #   为什么不用sampler？
    #     sampler + class_weight 双管齐下会过度矫正，模型疯狂预测少数类
    #     实测：同时使用时准确率仅15%，模型把中品质全预测为低品质
    #     只用class_weight更稳定，足以让模型关注少数类
    use_weighted_sampler = False

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Step 3: 加载和预处理数据
# ============================================================
def load_data(cfg):
    """
    加载分类数据并预处理。

    【分类数据预处理要点】
    1. 类别合并(可选)：将稀有类别合并，解决类别不平衡问题
    2. 特征标准化：与回归相同，必须标准化
    3. 类别权重计算：为不平衡数据集提供加权损失函数
    4. 加权采样器：让少数类在训练时被更频繁地采样
    """

    # --- 加载数据 ---
    df = pd.read_csv(cfg.datasets_path)
    print(f"原始数据: {df.shape[0]}条, {df.shape[1]}列")
    print(f"列名: {list(df.columns)}")

    # 分离特征和标签
    target_col = df.columns[-1]  # 最后一列是标签(quality)
    X = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].values

    # ========== 关键优化1: 类别合并 ==========
    #
    # 为什么要合并？原始6类(3~8分)的分布：
    #   3分: 10条(0.6%)   ← 太少，模型学不到
    #   4分: 53条(3.3%)   ← 太少
    #   5分: 681条(42.6%) ← 占主导
    #   6分: 638条(39.9%) ← 占主导
    #   7分: 199条(12.4%)
    #   8分: 18条(1.1%)   ← 太少
    #
    # 问题1: 类别3和8各不到20条，一个batch(32条)可能1条都没有
    #   → 模型在这些类别上的梯度估计极不准确
    # 问题2: 类别5+6占82%，模型全预测5或6就能82%准确率
    #   → 模型没有动力学习区分低品质和高品质
    #
    # 合并方案: 3+4→低品质(0), 5+6→中品质(1), 7+8→高品质(2)
    #   合并后: 63条(3.9%), 1319条(82.5%), 217条(13.6%)
    #   虽然仍不平衡，但每类至少60+条，模型可以学习了
    #   语义也更清晰: "这酒品质如何？低/中/高"比"这酒是3分还是4分"更实用
    if cfg.class_merge is not None:
        print(f"\n类别合并: {cfg.class_merge}")
        y = np.array([cfg.class_merge[label] for label in y])
        # 打印合并后的分布
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            name = cfg.class_names[u] if cfg.class_names else f"类别{u}"
            print(f"  {name}: {c}条({c/len(y)*100:.1f}%)")

    # 处理分类特征(如果有的话)
    # 本数据集全是数值特征，无需编码
    cat_cols = df.drop(columns=[target_col]).select_dtypes(exclude="number").columns.tolist()
    if cat_cols:
        print(f"\n检测到分类列: {cat_cols}，使用One-Hot编码")
        X_df = pd.get_dummies(df.drop(columns=[target_col]), columns=cat_cols, dtype=float, drop_first=True)
        X = X_df.values.astype(np.float32)
    else:
        print("\n所有特征为数值型，无需One-Hot编码")

    # 自动更新特征维度
    if cfg.num_features is None:
        cfg.num_features = X.shape[1]
        print(f"自动检测特征维度: {cfg.num_features}")

    # Step 3.1: 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    # stratify=y: 保证训练集和测试集中各类别的比例与原始数据一致
    #   为什么必须？不平衡数据下，不stratify可能测试集完全没有低品质样本
    #   原理：按y的分布等比例抽样，确保每个子集中类别比例一致

    # Step 3.2: 特征标准化 (Z-score标准化)
    # 公式: x' = (x - μ) / σ，其中μ=均值，σ=标准差
    # 变换后: 均值=0，标准差=1
    #
    # 为什么必须标准化？
    #   本数据集中特征范围差异巨大：
    #     氯化物: 0.01~0.61，总二氧化硫: 6~289
    #   不标准化时，大数值特征主导梯度，小数值特征被忽略
    #
    # fit_transform vs transform 的区别：
    #   fit_transform: 在训练集上计算μ和σ，然后转换(训练集用)
    #   transform:     直接用已计算的μ和σ转换(测试集用)
    #   关键：测试集绝不能fit！否则"偷看"测试集信息(数据泄露)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ========== 关键优化2: 计算类别权重 ==========
    #
    # 类别权重公式: weight_i = N / (C * n_i)
    #   N = 总训练样本数, C = 类别数, n_i = 第i类的训练样本数
    #
    # 直觉：少数类权重高 → 预测错少数类 → 损失大 → 梯度大 → 模型更努力学少数类
    #   例：低品质63条 → weight≈8.5，中品质1319条 → weight≈0.4
    #   预测错1条低品质的惩罚 ≈ 预测错21条中品质的惩罚
    #   这样模型不会只盯着中品质预测了
    class_weights = None
    if cfg.use_class_weight:
        unique_classes = np.unique(y_train)
        n_total = len(y_train)
        n_classes = len(unique_classes)
        weights = []
        for c in unique_classes:
            n_c = (y_train == c).sum()
            # 逆频率权重: w = N / (C * n_c)
            # 例：低品质63条 → w ≈ 8.5，中品质1319条 → w ≈ 0.4
            w_raw = n_total / (n_classes * n_c)
            # 平滑处理: 使用 sqrt(原始权重) 而非原始权重
            # 为什么？直接用逆频率权重过于激进：
            #   低品质权重8.5 >> 中品质权重0.4 → 模型过度偏向少数类
            #   sqrt(8.5)≈2.9, sqrt(0.4)≈0.6 → 更温和的调节
            # 效果：模型仍然关注少数类，但不会牺牲多数类的准确率
            w = np.sqrt(w_raw)
            weights.append(w)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"\n类别权重(sqrt平滑): {dict(zip(unique_classes, [f'{w:.2f}' for w in weights]))}")

    # ========== 关键优化3: 加权随机采样器 ==========
    #
    # 原理：给每个样本分配采样权重，少数类样本权重高 → 被采到的概率大
    #   例：低品质样本权重8.5，中品质0.4
    #   一个epoch中，低品质样本会被看到约8.5次，中品质约0.4次
    #   等效于"复制"低品质样本让模型多看几遍
    #
    # 与class_weight的区别：
    #   class_weight: 改变损失函数(同一个样本，算损失时乘以权重)
    #   WeightedRandomSampler: 改变数据分布(少数类被采样更多次)
    #   两者互补：class_weight让模型"更重视"少数类的错误
    #             sampler让模型"更多看到"少数类的样本
    train_sampler = None
    if cfg.use_weighted_sampler:
        sample_weights = []
        for label in y_train:
            class_idx = list(np.unique(y_train)).index(label)
            sample_weights.append(class_weights[class_idx].item() if class_weights is not None else 1.0)
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        # num_samples=len(y_train): 一个epoch的总采样数=训练集大小
        # replacement=True: 有放回采样(同一样本可能被多次采到，这是sampler的核心)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(y_train),
            replacement=True,
        )
        print("已启用加权随机采样器(少数类被更频繁采样)")

    # Step 3.3: 转为PyTorch张量
    # 分类任务: 标签必须是 long 类型(CrossEntropyLoss要求)
    #   为什么？CrossEntropyLoss内部用long类型做索引，查找对应类别的log概率
    #   如果是float会报错！这是分类和回归的关键区别
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Step 3.4: 封装为DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # 注意：sampler和shuffle互斥！
    #   使用sampler时必须shuffle=False，否则报错
    #   sampler已经负责"打乱"(通过随机采样)，不需要额外的shuffle
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),  # 有sampler时不用shuffle
        sampler=train_sampler,
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return (
        train_loader,
        test_loader,
        X_test.to(cfg.device),
        y_test.to(cfg.device),
        scaler,
        class_weights,
    )


# ============================================================
# Step 4: 定义FNN分类模型
# ============================================================
class FNNClassifier(nn.Module):
    """
    前馈神经网络分类器。

    【网络结构: "漏斗"型逐层压缩】
    输入(11维) → [BN→Linear→ReLU→Dropout](128) → [BN→Linear→ReLU→Dropout](64)
    → [BN→Linear→ReLU→Dropout](32) → Linear(3)

    【每一层的作用】
    - BatchNorm1d: 批归一化，稳定每层输入的分布，加速收敛
    - Linear: 全连接层，学习特征的线性组合权重
    - ReLU: 非线性激活，让网络能学习复杂的非线性关系
    - Dropout: 随机丢弃神经元，防止过拟合
    - 输出层Linear: 无激活函数，输出logits(未经softmax的原始分数)

    【为什么 BatchNorm 放在 Linear 之前？(Pre-Norm)】
    有两种常见顺序：
      Post-Norm: Linear → ReLU → BatchNorm  (原始论文)
      Pre-Norm:  BatchNorm → Linear → ReLU  (本代码采用)
    Pre-Norm的优势：
      - 梯度流更稳定(梯度不经过BN的缩放)
      - 对学习率更鲁棒
      - 训练初期更稳定(不会出现loss spike)

    【输出层为什么不加Softmax？】
    因为 nn.CrossEntropyLoss 内部已经包含了 LogSoftmax + NLLLoss
    如果再加 Softmax 就是重复计算，且会导致梯度计算错误
    正确做法：模型输出logits → CrossEntropyLoss内部做softmax
    如需概率：torch.softmax(outputs, dim=1)
    """

    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.3):
        super(FNNClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.BatchNorm1d(prev_dim))          # 批归一化(放在Linear前)
            layers.append(nn.Linear(prev_dim, hidden_dim))    # 全连接层
            layers.append(nn.ReLU())                          # 激活函数
            layers.append(nn.Dropout(dropout_rate))           # Dropout防过拟合
            prev_dim = hidden_dim

        # 输出层(不加激活函数，CrossEntropyLoss内置Softmax)
        layers.append(nn.Linear(prev_dim, num_classes))

        # 用nn.Sequential将所有层串联起来
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
        return self.network(x)


# ============================================================
# Step 5: 训练和评估函数
# ============================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
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
        # 原理：当所有参数梯度的L2范数 > max_norm 时，等比缩放梯度
        #   ‖g‖ = sqrt(Σgᵢ²)，若 ‖g‖ > 1.0，则 g = g * (1.0 / ‖g‖)
        # 为什么需要？不平衡数据下，少数类样本可能产生很大的梯度
        #   不裁剪：参数一步跨太远，损失飙升
        #   裁剪后：梯度方向不变，只限制步长，训练更稳定
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()            # Step 5: 更新参数

        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    return total_loss / total_samples


def evaluate(model, test_loader, criterion, device):
    """
    评估模型，返回损失、准确率和预测结果。

    与训练的区别：
    - model.eval(): 切换到评估模式
      → Dropout关闭(所有神经元都参与)
      → BatchNorm使用全局统计量(而非当前batch的)
    - torch.no_grad(): 禁用梯度计算
      → 节省内存(不存储中间激活值)
      → 加速计算(不需要反向传播)
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

            # 取概率最大的类别作为预测结果
            # torch.max(outputs, dim=1) 返回 (最大值, 最大值索引)
            # 我们只需要索引(即预测的类别)
            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_preds, all_labels


# ============================================================
# Step 6: 主训练流程
# ============================================================
def main():
    cfg = CONFIG()
    print(f"使用设备: {cfg.device}")

    # --- 加载数据 ---
    train_loader, test_loader, X_test, y_test, scaler, class_weights = load_data(cfg)
    print(
        f"训练集大小: {len(train_loader.dataset)}, 测试集大小: {len(test_loader.dataset)}"
    )

    # --- 创建模型 ---
    model = FNNClassifier(
        input_dim=cfg.num_features,
        hidden_dims=cfg.hidden_dims,
        num_classes=cfg.num_classes,
        dropout_rate=cfg.dropout_rate,
    ).to(cfg.device)
    print(f"\n模型结构:\n{model}")

    # --- 损失函数 ---
    # CrossEntropyLoss: 分类任务的标准损失函数
    # 内部 = LogSoftmax + NLLLoss，所以模型输出不需要Softmax
    #
    # 加权CrossEntropyLoss: 给少数类更大的损失权重
    #   原理：loss = -weight[y] * log(softmax(output)[y])
    #   效果：预测错低品质(权重8.5)的惩罚 >> 预测错中品质(权重0.4)
    #   结果：模型被迫学习所有类别，不再只预测多数类
    if cfg.use_class_weight and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(cfg.device))
        print(f"使用加权CrossEntropyLoss(权重已在GPU上)")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用标准CrossEntropyLoss(无类别权重)")

    # --- 优化器 ---
    # Adam优化器：目前最常用的深度学习优化器
    # 原理：结合Momentum(动量)和RMSprop(自适应学习率)
    #   - Momentum: 梯度方向一致时加速，方向摇摆时减速(像球滚下坡)
    #   - 自适应学习率: 每个参数有独立的学习率，梯度大的参数步长小
    # 优势：对学习率不太敏感，几乎不需要调
    # weight_decay=1e-4: L2正则化，与Adam配合防止过拟合
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # --- 学习率调度器 ---
    # ReduceLROnPlateau: 验证损失停滞时自动降低学习率
    # 原理：监控验证损失，如果连续patience轮没有改善，LR乘以factor
    #   例：初始LR=5e-4，停滞8轮后 → LR=2.5e-4，再停滞8轮 → LR=1.25e-4
    # 参数说明：
    #   mode="min": 监控指标越小越好(损失)
    #   factor=0.5: 每次LR减半
    #   patience=8: 连续8轮无改善才降LR
    #   min_lr=1e-6: LR下限
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    # --- 训练循环 ---
    train_losses = []
    val_losses = []
    val_accuracies = []
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
            model, train_loader, criterion, optimizer, cfg.device
        )
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, cfg.device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step(val_loss)  # ReduceLROnPlateau: 传入验证损失

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"\n早停触发！在第 {epoch + 1} 轮停止训练")
                break

        # 打印训练进度
        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch + 1}/{cfg.epochs}] "
                f"训练Loss: {train_loss:.4f} | "
                f"验证Loss: {val_loss:.4f} | "
                f"验证准确率: {val_acc:.4f} | "
                f"LR: {current_lr:.2e}"
            )

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n已恢复最佳模型(验证Loss: {best_val_loss:.4f})")

    # ============================================================
    # Step 7: 最终评估
    # ============================================================
    _, final_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, cfg.device)

    print(f"\n{'='*50}")
    print(f"最终测试准确率: {final_acc * 100:.2f}%")
    print(f"{'='*50}")

    # 详细分类报告(精确率、召回率、F1)
    # 精确率(Precision): 预测为该类的样本中，真正是该类的比例 → "预测的有多准"
    # 召回率(Recall):    真正是该类的样本中，被正确预测的比例 → "找回了多少"
    # F1:                精确率和召回率的调和平均 → 综合指标
    #   F1 = 2 × P × R / (P + R)
    #   为什么用调和平均而非算术平均？调和平均对短板更敏感
    #   P=1.0, R=0.01 → 算术平均=0.505(看着还行), 调和平均=0.02(暴露问题)
    print("\n分类报告:")
    target_names = cfg.class_names if cfg.class_names else None
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

    # 混淆矩阵
    # 行=真实类别，列=预测类别
    # 对角线=正确预测，非对角线=错误预测
    # 例：第0行第2列=真实为低品质但被预测为高品质的样本数
    print("混淆矩阵 (行=真实, 列=预测):")
    cm = confusion_matrix(all_labels, all_preds)
    if cfg.class_names:
        # 带标签的混淆矩阵
        header = "        " + "  ".join([f"预{i}" for i in range(cfg.num_classes)])
        print(header)
        for i, row in enumerate(cm):
            label = cfg.class_names[i][:4] if len(cfg.class_names[i]) > 4 else cfg.class_names[i]
            print(f"真{i}({label}) {row}")
    else:
        print(cm)

    # ============================================================
    # Step 8: 可视化训练过程
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    # 理想情况：训练和验证损失都持续下降，最终趋于平稳
    # 过拟合信号：训练损失↓ 但 验证损失↑ (验证曲线开始上升)
    axes[0].plot(train_losses, label="训练损失")
    axes[0].plot(val_losses, label="验证损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("损失曲线")
    axes[0].legend()

    # 准确率曲线
    axes[1].plot(val_accuracies, label="验证准确率", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("准确率曲线")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"fnn/fnn_classification_training_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png", dpi=150
    )
    plt.show()

    # ============================================================
    # Step 9: 单样本预测示例
    # ============================================================
    model.eval()
    with torch.no_grad():
        sample_x = X_test[:5]
        outputs = model(sample_x)
        probabilities = torch.softmax(outputs, dim=1)  # 手动转为概率
        predicted_classes = torch.argmax(probabilities, dim=1)

        print("\n单样本预测示例:")
        for i in range(min(5, len(X_test))):
            true_label = y_test[i].item()
            pred_label = predicted_classes[i].item()
            confidence = probabilities[i][pred_label].item()
            true_name = cfg.class_names[true_label] if cfg.class_names else f"类别{true_label}"
            pred_name = cfg.class_names[pred_label] if cfg.class_names else f"类别{pred_label}"
            print(
                f"  样本{i}: 真实={true_name}, "
                f"预测={pred_name}, 置信度={confidence:.4f}"
            )

    # ============================================================
    # Step 10: 模型保存与加载
    # ============================================================
    model_path = f"fnn/fnn_classification_model_{now.strftime('%Y-%m-%d_%H-%M-%S')}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")

    # 加载模型(使用时需要确保模型结构一致)
    loaded_model = FNNClassifier(
        input_dim=cfg.num_features,
        hidden_dims=cfg.hidden_dims,
        num_classes=cfg.num_classes,
        dropout_rate=cfg.dropout_rate,
    ).to(cfg.device)
    loaded_model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=cfg.device)
    )
    loaded_model.eval()
    print("模型加载成功！")

    # ============================================================
    # Step 11: 对新数据进行预测(生产环境用法)
    # ============================================================
    # new_data = np.array([[7.4, 0.70, 0.00, 1.9, 0.076, 11, 34, 0.998, 3.51, 0.56, 9.4]])
    # new_data_scaled = scaler.transform(new_data)
    # new_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).to(cfg.device)
    # with torch.no_grad():
    #     outputs = loaded_model(new_tensor)
    #     probs = torch.softmax(outputs, dim=1)
    #     pred = torch.argmax(probs, dim=1).item()
    #     print(f"预测品质: {cfg.class_names[pred]}, 置信度: {probs[0][pred].item():.4f}")

    print("\n分类任务训练和评估完成！")
    plt.close()


if __name__ == "__main__":
    main()
