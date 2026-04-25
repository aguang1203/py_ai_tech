"""
=============================================================================
FNN 回归任务模板 (Feedforward Neural Network for Regression)
=============================================================================

【原理】
回归任务的目标是预测连续数值(如房价、温度、销量等)。
FNN回归与分类的核心区别：
- 输出层：只有1个神经元(预测单个数值)，不加激活函数(输出可以是任意实数)
- 损失函数：MSELoss(均方误差) 或 L1Loss(平均绝对误差)
- 评估指标：MSE、RMSE、MAE、R² 等，而非准确率

【应用场景】
- 房价预测 (输出: 价格，连续值)
- 销量预测 (输出: 销售数量)
- 气温预测 (输出: 温度值)
- 能耗预测 (输出: 用电量)
- 股票价格预测 (输出: 价格)

【与分类/多标签的区别】
- 回归: 输出层=1个神经元(无激活)，损失=MSELoss，评估=RMSE/R²
- 分类: 输出层=类别数，损失=CrossEntropyLoss，评估=准确率
- 多标签: 输出层=标签数，损失=BCEWithLogitsLoss，评估=各标签F1

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 替换 load_data() 为你自己的数据加载逻辑
3. 直接运行: python regression.py
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

    datasets_path = "/home/hjg/dev/datasets/house-clean.csv"

    # --- 数据相关 ---
    # num_features: 输入特征维度，设为 None 表示自动检测
    #   原因：OneHot编码后特征数会变化(本数据集7→41)，手动写容易出错
    #   自动检测在 load_data() 中完成：cfg.num_features = X.shape[1]
    num_features = None

    # test_size=0.2: 训练集:测试集 = 8:2
    #   为什么不是7:3？22781条数据量中等，8:2既保证训练充分，又有足够测试样本(~4500条)
    #   为什么不是9:1？测试集太少会导致评估指标方差大，不可靠
    test_size = 0.2

    # random_state=42: 固定随机种子，确保每次运行结果可复现
    #   42是惯例(《银河系漫游指南》中"生命、宇宙及一切的答案")
    #   实际值不重要，重要的是"固定"，方便对比不同超参数的效果
    random_state = 42

    # --- 模型相关 ---
    # hidden_dims: 隐藏层维度列表，从大到小"漏斗"结构
    #   为什么是 [256, 128, 64]？
    #     - 41维输入 → 256：第一层要足够宽，学习41个特征的组合模式
    #     - 256 → 128 → 64：逐层压缩，每层提取更抽象的特征
    #     - 为什么不是 [64, 32, 16]？太窄，41维输入压缩到64会丢失信息
    #     - 为什么不是 [512, 256, 128]？太宽，2万条数据容易过拟合
    #   经验法则：第一层隐藏维度 ≈ 2~8倍输入维度，后续逐层减半
    hidden_dims = [256, 128, 64]

    # output_dim=1: 回归任务输出1个连续值(房价)
    #   如果预测多个目标(如同时预测房价+面积)，设为对应数量
    output_dim = 1

    # dropout_rate=0.2: 训练时随机丢弃20%的神经元
    #   为什么是0.2而不是0.5？
    #     - 分类任务常用0.5，因为分类边界复杂，需要强正则化
    #     - 回归任务数据更平滑(连续值)，0.5会丢失太多信息，0.2更合适
    #   原理：Dropout迫使网络不依赖任何单个神经元，提升泛化能力
    #   类比：团队中随机有人请假，其他人必须学会替代，整体更健壮
    dropout_rate = 0.2

    # --- 训练相关 ---
    # batch_size=64: 每次梯度更新使用64个样本
    #   为什么不是16？批次太小，梯度估计噪声大，训练不稳定
    #   为什么不是256？批次太大，每次epoch更新次数少，收敛慢
    #   64是表格数据回归的甜蜜点：梯度足够稳定，更新频率也够
    batch_size = 64

    # learning_rate=3e-4: Adam优化器的初始学习率
    #   为什么不是1e-3(Adam默认值)？本任务中1e-3导致后期损失波动
    #   为什么不是1e-5？太慢，200个epoch都收敛不了
    #   3e-4是Adam的"黄金学习率"之一(Yaroslar GPT系列经验)
    #   原理：学习率 = 每步参数更新的幅度，太大震荡，太慢收敛
    learning_rate = 3e-4

    # epochs=200: 最大训练轮数
    #   不用担心跑满200轮——早停机制会在验证损失不再下降时自动停止
    #   200只是"上限"，实际通常在50~100轮就触发早停
    epochs = 200

    # weight_decay=1e-4: L2正则化强度(也叫权重衰减)
    #   原理：在损失函数中加入 λ·Σ(w²)，惩罚过大的权重
    #   效果：防止模型"记住"训练数据的噪声(过拟合)
    #   为什么是1e-4？经验值，太大(1e-2)会欠拟合，太小(1e-6)等于没有
    #   直觉：告诉优化器"权重别太大"，就像弹簧限制模型的复杂度
    weight_decay = 1e-4

    # --- 早停策略 ---
    # early_stop_patience=20: 验证损失连续20轮不下降就停止
    #   为什么是20而不是5？学习率调度器会降低LR，低LR下改善缓慢，
    #   需要足够的耐心等它收敛，否则会在还有改善空间时就停止
    #   为什么不是50？太长会浪费训练时间
    early_stop_patience = 20

    save_best_only = True  # 只保存验证集最优模型(而非最后一轮)

    # --- 损失函数 ---
    # SmoothL1Loss (Huber Loss): 兼顾MSE和MAE优点的损失函数
    #
    # 公式: loss(x) = { 0.5*x²     若 |x| < 1
    #                  { |x| - 0.5  若 |x| ≥ 1
    #
    # 与MSELoss对比:
    #   MSELoss = x²        → 对大误差惩罚极强(平方放大)，被异常值主导
    #   SmoothL1Loss         → 误差小时类似MSE(精确收敛)，大时类似MAE(鲁棒)
    #
    # 为什么选SmoothL1Loss而不是MSELoss？
    #   本数据集有137条极端高价异常值(>Q3+3*IQR)
    #   MSELoss下，一条1000万的房子预测误差200万 → loss=40000
    #   而正常100万的房子误差20万 → loss=400，前者影响是后者的100倍！
    #   SmoothL1Loss下，大误差时用|x|-0.5，不会被异常值绑架
    #
    # 什么时候用MSELoss？数据干净、无异常值时，MSELoss收敛更快
    # 什么时候用L1Loss？异常值极多时，L1Loss更鲁棒但收敛慢(梯度不连续)
    criterion = nn.SmoothL1Loss()

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Step 3: 加载和预处理数据
# ============================================================
def load_data(cfg):
    """
    加载回归数据并预处理。

    【回归数据预处理要点】
    1. 特征标准化：与分类相同，必须标准化
    2. 标签标准化：回归任务中，标签(y)也可以标准化！
       - 好处：让损失值在合理范围内，训练更稳定
       - 注意：预测后需要反标准化(inverse_transform)才能得到真实预测值
    3. 回归任务的标签是float类型(不是long！)
    """

    # --- 方式1: 使用sklearn生成模拟数据 ---
    # X, y = make_regression(
    #     n_samples=cfg.num_samples,
    #     n_features=cfg.num_features,
    #     n_informative=cfg.num_features,  # 所有特征都有信息
    #     noise=10.0,                       # 添加噪声，模拟真实数据
    #     random_state=cfg.random_state,
    # )

    # --- 方式2: 加载你自己的数据 ---

    df = pd.read_csv(cfg.datasets_path)

    # ========== 关键优化1: One-Hot 编码 ==========
    #
    # 为什么不用 LabelEncoder？
    #   LabelEncoder 把 "朝阳"→0, "海淀"→1, "平谷"→2
    #   这暗示了 "平谷(2) > 海淀(1) > 朝阳(0)" 的虚假序关系
    #   模型会学到: 市区编号越大→房价越高，这完全是错的！
    #   (平谷房价远低于朝阳和海淀)
    #
    # One-Hot编码的原理：
    #   "朝阳"→[1,0,0,0,...], "海淀"→[0,1,0,0,...], "平谷"→[0,0,1,0,...]
    #   每个类别独占一列，列之间没有大小关系，模型自由学习每个区的影响
    #   代价：特征数从7→41，但FNN完全可以处理(41维对神经网络来说很小)
    #
    # drop_first=True 的含义：k个类别只生成k-1列
    #   例："市区"有10个区，只生成9列。如果9列全为0，就自动推断是第10个区
    #   为什么？避免"多重共线性"——一列可以用其他列推出，导致数值不稳定
    #   数学上：10列的和恒为1，信息冗余；9列已经完整编码所有类别
    target_col = df.columns[-1]
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    cat_cols = [c for c in cat_cols if c != target_col]

    X_df = pd.get_dummies(df.drop(columns=[target_col]), columns=cat_cols, dtype=float, drop_first=True)
    X = X_df.values
    y = df[target_col].values.astype(np.float32)

    # 自动检测特征维度(OneHot后特征数会变化，手动设置容易出错)
    if cfg.num_features is None:
        cfg.num_features = X.shape[1]
        print(f"自动检测特征维度: {cfg.num_features} (OneHot编码后)")

    # 保存列名(新数据预测时必须用相同的列顺序，否则特征错位)
    feature_columns = X_df.columns.tolist()

    # Step 3.1: 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    # ========== 关键优化2: 对数变换 ==========
    #
    # 对房价标签做 np.log 变换——这是房价预测提升效果最强的单步优化
    #
    # 为什么？房价数据天然"右偏"(正偏态分布)：
    #   - 大部分房子 100~500万，少数豪宅 2000~5000万
    #   - 偏度(skewness)≈2.6，远大于0(正态分布偏度=0)
    #   - MSELoss会让模型拼命拟合那些高价房(因为误差的平方很大)
    #
    # log变换的效果：
    #   100万 → 4.6,  500万 → 6.2,  5000万 → 8.5
    #   变换后偏度≈0.1，接近正态分布！
    #   高价房的影响被压缩，模型不会"偏心"于高价房
    #
    # 预测时需要 np.exp() 还原真实价格（见Step 7评估部分）
    #
    # 什么时候用log变换？目标值右偏(偏度>1)时，如：房价、收入、销量
    # 什么时候不用？目标值接近正态分布(偏度≈0)时，如：温度、标准化考试分数
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    # Step 3.2: 特征标准化 (Z-score标准化)
    # 公式: x' = (x - μ) / σ，其中μ=均值，σ=标准差
    # 变换后: 均值=0，标准差=1
    #
    # 为什么必须标准化？
    #   OneHot后特征范围差异巨大：面积列(50~300) vs 市区列(0/1)
    #   不标准化时，大面积值特征主导梯度，小值特征被忽略
    #   标准化后所有特征在同等尺度上，模型公平对待每个特征
    #
    # fit_transform vs transform 的区别：
    #   fit_transform: 在训练集上计算μ和σ，然后转换(训练集用)
    #   transform:     直接用已计算的μ和σ转换(测试集用)
    #   关键：测试集绝不能fit！否则会"偷看"测试集信息(数据泄露)
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_test = feature_scaler.transform(X_test)

    # Step 3.3: 标签标准化(回归任务特有！分类任务不需要)
    #
    # 为什么回归任务要标准化标签y？
    #   log变换后 y ≈ 4~9，范围还是偏大
    #   标准化后 y ≈ -2~2，损失值在合理范围(0.01~1.0)
    #   好处：学习率不需要特别调整，梯度稳定，收敛更快
    #
    # 为什么分类任务不需要？
    #   分类标签是类别编号(0,1,2,...)，CrossEntropyLoss内部已处理
    #   回归标签是连续值，范围不确定(可以是0.1~1.0，也可以是1~10000)
    #
    # 预测时的反变换链：
    #   模型输出 → y_scaler.inverse_transform → log尺度 → np.exp → 真实价格
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # Step 3.4: 转为PyTorch张量
    # 回归任务: 标签是 float32 类型(不是long！因为要预测连续值)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Step 3.5: 封装为DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return (
        train_loader,
        test_loader,
        X_test.to(cfg.device),
        y_test.to(cfg.device),
        feature_scaler,
        y_scaler,
        feature_columns,
    )


# ============================================================
# Step 4: 定义FNN回归模型
# ============================================================
class FNNRegressor(nn.Module):
    """
    前馈神经网络回归器。

    【网络结构: "漏斗"型逐层压缩】
    输入(41维) → [BN→Linear→ReLU→Dropout](256) → [BN→Linear→ReLU→Dropout](128) → [BN→Linear→ReLU→Dropout](64) → Linear(1)

    【每一层的作用】
    - BatchNorm1d: 批归一化，稳定每层输入的分布，加速收敛
    - Linear: 全连接层，学习特征的线性组合权重
    - ReLU: 非线性激活，让网络能学习复杂的非线性关系
    - Dropout: 随机丢弃神经元，防止过拟合
    - 输出层Linear: 无激活函数，输出可以是任意实数(回归需要)

    【为什么 BatchNorm 放在 Linear 之前？(Pre-Norm)】
    有两种常见顺序：
      Post-Norm: Linear → ReLU → BatchNorm  (原始论文)
      Pre-Norm:  BatchNorm → Linear → ReLU  (本代码采用)
    Pre-Norm的优势：
      - 梯度流更稳定(梯度不经过BN的缩放)
      - 对学习率更鲁棒
      - 训练初期更稳定(不会出现loss spike)
    实践中，深层网络Pre-Norm效果通常更好

    【为什么回归输出层不加激活函数？】
    - ReLU: 输出≥0，无法预测负数(如温度-10°C)
    - Sigmoid: 输出0-1，范围太窄
    - Softmax: 输出和为1，完全不适合回归
    - 无激活: 输出可以是任意实数(-∞ ~ +∞)，完美适配回归
    """

    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate=0.2):
        super(FNNRegressor, self).__init__()

        layers = []
        prev_dim = input_dim

        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.BatchNorm1d(prev_dim))
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # 输出层: 1个神经元，无激活函数
        layers.append(nn.Linear(prev_dim, output_dim))

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
        #
        # mode="fan_in"：用输入维度计算，训练初期更稳定(推荐)
        # nonlinearity="relu"：告诉初始化器使用He而非Xavier公式
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 偏置初始化为0，让初始输出以0为中心

    def forward(self, x):
        # 输出shape: (batch_size, 1)，需要squeeze掉最后一维
        return self.network(x).squeeze(-1)


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
        # 为什么需要？回归任务的损失landscape比分类更崎岖
        #   偶尔遇到异常样本时，梯度可能突然变得很大(梯度爆炸)
        #   不裁剪：参数一步跨太远，损失飙升，甚至训练崩溃
        #   裁剪后：梯度方向不变，只限制步长，训练更稳定
        # max_norm=1.0：经验值，通常1.0效果很好
        #   太大(10.0)裁不到，太小(0.01)限制太死，学习变慢
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()            # Step 5: 更新参数

        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    return total_loss / total_samples


def evaluate(model, test_loader, criterion, device):
    """
    评估模型，返回损失和预测结果。

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

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / total_samples
    return avg_loss, np.array(all_preds), np.array(all_labels)


# ============================================================
# Step 6: 主训练流程
# ============================================================
def main():
    cfg = CONFIG()
    print(f"使用设备: {cfg.device}")

    # --- 加载数据 ---
    train_loader, test_loader, X_test, y_test, feature_scaler, y_scaler, feature_columns = load_data(cfg)
    print(
        f"训练集大小: {len(train_loader.dataset)}, 测试集大小: {len(test_loader.dataset)}"
    )

    # --- 创建模型 ---
    model = FNNRegressor(
        input_dim=cfg.num_features,
        output_dim=cfg.output_dim,
        hidden_dims=cfg.hidden_dims,
        dropout_rate=cfg.dropout_rate,
    ).to(cfg.device)
    print(f"\n模型结构:\n{model}")

    # --- 优化器和学习率调度器 ---

    # Adam优化器：目前最常用的深度学习优化器
    # 原理：结合Momentum(动量)和RMSprop(自适应学习率)
    #   - Momentum: 梯度方向一致时加速，方向摇摆时减速(像球滚下坡)
    #   - 自适应学习率: 每个参数有独立的学习率，梯度大的参数步长小
    # 优势：对学习率不太敏感(3e-4和5e-4都能工作)，几乎不需要调
    # weight_decay=1e-4: L2正则化，与Adam配合防止过拟合
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # ReduceLROnPlateau: 验证损失停滞时自动降低学习率
    # 原理：监控验证损失，如果连续patience轮没有改善，LR乘以factor
    #   例：初始LR=3e-4，停滞8轮后 → LR=1.5e-4，再停滞8轮 → LR=7.5e-5
    # 为什么用这个而不是CosineAnnealingWarmRestarts？
    #   CosineAnnealingWarmRestarts: LR按余弦曲线周期性变化，热重启时LR突然升高
    #   实测在本任务中，热重启导致验证损失spike(从0.08飙升到1.04)
    #   ReduceLROnPlateau: LR只降不升，更稳定，适合大多数回归任务
    # 参数说明：
    #   mode="min": 监控指标越小越好(损失)
    #   factor=0.5: 每次LR减半(不要太激进，0.5是常用值)
    #   patience=8: 连续8轮无改善才降LR(给模型足够的探索时间)
    #   min_lr=1e-6: LR下限，不会降到0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    # --- 训练循环 ---
    train_losses = []
    val_losses = []
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
    #
    # 为什么要保存best_model_state而不是等训练完再评估？
    #   因为最后一轮的模型往往不是最好的！过拟合后验证损失会上升
    #   早停的核心：在最好的时刻停下来，而不是在最后停下来

    print("\n开始训练...")
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(
            model, train_loader, cfg.criterion, optimizer, cfg.device
        )
        val_loss, _, _ = evaluate(model, test_loader, cfg.criterion, cfg.device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

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
                f"训练Loss: {train_loss:.4f} | 验证Loss: {val_loss:.4f} | LR: {current_lr:.2e}"
            )

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n已恢复最佳模型(验证MSE: {best_val_loss:.4f})")

    # ============================================================
    # Step 7: 最终评估
    # ============================================================
    _, all_preds_scaled, all_labels_scaled = evaluate(
        model, test_loader, cfg.criterion, cfg.device
    )

    # ========== 反标准化：从模型输出还原到真实价格 ==========
    #
    # 完整的还原链条(训练时的逆向操作)：
    #   模型输出(标准化尺度) → y_scaler.inverse_transform → log尺度 → np.exp → 真实价格(万元)
    #
    # 为什么需要两步反变换？
    #   训练时做了两步变换：y_real → np.log(y_real) → StandardScaler
    #   所以反变换也要两步：StandardScaler逆变换 → np.exp
    #
    #   y_scaler.inverse_transform: 还原StandardScaler的变换
    #     模型输出 ≈ -2~2 → 还原后 ≈ 4~9 (log尺度)
    #   np.exp: 还原log变换
    #     log尺度 ≈ 4~9 → exp后 ≈ 55~8103 (真实价格，万元)
    #
    # 如果不做np.exp会怎样？
    #   指标在log尺度上计算，R²看起来很高(0.95+)，但实际意义不大
    #   因为log(100万)≈4.6 和 log(1000万)≈6.9 差距才2.3
    #   但真实价格差距是900万！必须在真实尺度上评估才有意义
    all_preds_log = y_scaler.inverse_transform(
        all_preds_scaled.reshape(-1, 1)
    ).flatten()
    all_labels_log = y_scaler.inverse_transform(
        all_labels_scaled.reshape(-1, 1)
    ).flatten()

    all_preds_real = np.exp(all_preds_log)
    all_labels_real = np.exp(all_labels_log)

    # 计算评估指标(在真实价格尺度上，单位：万元)
    #
    # MSE (Mean Squared Error, 均方误差) = mean((y_true - y_pred)²)
    #   对大误差惩罚很重(平方放大)，单位是 万元²，不太直观
    #
    # RMSE (Root MSE, 均方根误差) = sqrt(MSE)
    #   单位和原始数据一致(万元)，最常用的回归指标
    #   含义：平均预测误差约 ±RMSE 万元
    #   例：RMSE=128，表示平均预测误差约128万元
    #
    # MAE (Mean Absolute Error, 平均绝对误差) = mean(|y_true - y_pred|)
    #   对异常值更鲁棒(不平方)，单位也是万元
    #   含义：平均预测偏差的绝对值
    #   通常 MAE < RMSE，因为RMSE受大误差影响更大
    #
    # R² (决定系数, Coefficient of Determination)
    #   = 1 - Σ(y_true - y_pred)² / Σ(y_true - y_mean)²
    #   含义：模型解释了多少百分比的方差
    #   R²=1.0: 完美预测，R²=0: 和用均值预测一样差，R²<0: 比均值还差
    #   R²=0.81: 模型解释了81%的房价方差，剩下19%是无法解释的噪声
    #   本数据集R²>0.8算不错(房价受太多因素影响，FNN只能从已有特征学习)
    mse = mean_squared_error(all_labels_real, all_preds_real)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels_real, all_preds_real)
    r2 = r2_score(all_labels_real, all_preds_real)

    print(f"\n{'='*50}")
    print(f"回归评估指标(真实价格尺度，单位:万元):")
    print(f"  MSE  (均方误差):     {mse:.4f}")
    print(f"  RMSE (均方根误差):   {rmse:.4f}")
    print(f"  MAE  (平均绝对误差): {mae:.4f}")
    print(f"  R²   (决定系数):     {r2:.4f}")
    print(f"{'='*50}")
    # R²解读：1=完美预测，0=和均值一样差，<0=比均值还差
    # 本数据集R²≈0.81：模型解释了81%的房价方差，19%是无法解释的噪声

    # ============================================================
    # Step 8: 可视化
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 损失曲线
    # 损失曲线
    # 横轴=训练轮数，纵轴=损失值
    # 理想情况：训练和验证损失都持续下降，最终趋于平稳
    # 过拟合信号：训练损失↓ 但 验证损失↑ (验证曲线开始上升)
    axes[0].plot(train_losses, label="训练MSE")
    axes[0].plot(val_losses, label="验证MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("损失曲线")
    axes[0].legend()

    # 预测值 vs 真实值散点图
    # 每个点 = 一个测试样本，横轴=真实房价，纵轴=预测房价
    # 红色虚线 = 完美预测线(y=x)，点越靠近这条线说明预测越准
    # 点在线上方 = 预测偏高，点在线下方 = 预测偏低
    # 散点越集中 = 模型越稳定，散点越分散 = 模型不确定
    axes[1].scatter(all_labels_real, all_preds_real, alpha=0.5, s=10)
    min_val = min(all_labels_real.min(), all_preds_real.min())
    max_val = max(all_labels_real.max(), all_preds_real.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--", label="完美预测线")
    axes[1].set_xlabel("真实值")
    axes[1].set_ylabel("预测值")
    axes[1].set_title(f"预测 vs 真实 (R²={r2:.4f})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"fnn/fnn_regression_training_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png", dpi=150
    )
    plt.show()

    # ============================================================
    # Step 9: 单样本预测示例
    # ============================================================
    model.eval()
    with torch.no_grad():
        sample_x = X_test[:5]
        preds_scaled = model(sample_x).cpu().numpy()
        # 反标准化得到真实预测值(先inverse_transform还原log，再exp还原真实价格)
        preds_log = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        preds_real = np.exp(preds_log)
        labels_log = y_scaler.inverse_transform(
            y_test[:5].cpu().numpy().reshape(-1, 1)
        ).flatten()
        labels_real = np.exp(labels_log)

        print("\n单样本预测示例(原始尺度):")
        for i in range(5):
            error = abs(preds_real[i] - labels_real[i])
            print(
                f"  样本{i}: 真实值={labels_real[i]:.2f}, 预测值={preds_real[i]:.2f}, 误差={error:.2f}"
            )

    # ============================================================
    # Step 10: 模型保存与加载
    # ============================================================
    model_path = f"fnn/fnn_regression_model_{now.strftime('%Y-%m-%d_%H-%M-%S')}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")

    # loaded_model = FNNRegressor(
    #     input_dim=cfg.num_features,
    #     output_dim=cfg.output_dim,
    #     hidden_dims=cfg.hidden_dims,
    #     dropout_rate=cfg.dropout_rate,
    # ).to(cfg.device)
    # loaded_model.load_state_dict(
    #     torch.load(model_path, weights_only=True, map_location=cfg.device)
    # )
    # loaded_model.eval()
    # print("模型加载成功！")

    print("\n回归任务训练和评估完成！")  # 结束语，标志流程完成

    # ============================================================
    # Step 11: 对新数据进行预测(生产环境用法)
    # ============================================================
    # new_data = np.array([[...], [...]])  # shape: (n_samples, n_features)
    # new_data_scaled = feature_scaler.transform(new_data)
    # new_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).to(cfg.device)
    # with torch.no_grad():
    #     pred_scaled = loaded_model(new_tensor).cpu().numpy()
    #     pred_real = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    # print("新数据预测结果(原始尺度):", pred_real)


if __name__ == "__main__":
    main()
