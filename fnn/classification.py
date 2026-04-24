"""
=============================================================================
FNN 多分类任务模板 (Feedforward Neural Network for Classification)
=============================================================================

【原理】
前馈神经网络(FNN)通过多层全连接层 + 非线性激活函数，学习输入特征到离散类别的映射。
数据从输入层 → 隐藏层(逐层) → 输出层，单向流动，没有回路(前馈)。
输出层使用 Softmax 将 logits 转换为概率分布，取概率最大的类别作为预测结果。

【应用场景】
- 鸢尾花分类 (3类)
- 手写数字识别 (10类, MNIST)
- 客户流失预测 (2类: 流失/不流失)
- 新闻主题分类 (多类)
- 疾病诊断 (多类: 健康/轻症/重症)

【与回归/多标签的区别】
- 分类: 每个样本只属于1个类别，输出层维度=类别数，损失=CrossEntropyLoss
- 回归: 预测连续数值，输出层维度=1，损失=MSELoss
- 多标签: 每个样本可同时属于多个类别，输出层维度=标签数，损失=BCEWithLogitsLoss

【使用方法】
1. 修改 CONFIG 部分的超参数
2. 替换 load_data() 为你自己的数据加载逻辑
3. 直接运行: python classification.py
=============================================================================
"""

# ============================================================
# Step 1: 导入必要的库
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.datasets import make_classification  # 生成模拟分类数据
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt

# 设置中文字体(如果系统没有SimHei，可以换成其他中文字体或注释掉)
plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# Step 2: 配置超参数 (修改这里即可适配你的数据)
# ============================================================
class CONFIG:
    # --- 数据相关 ---
    num_samples = 1000       # 样本总数
    num_features = 20        # 输入特征维度
    num_classes = 3          # 分类类别数 (2=二分类, >2=多分类)
    test_size = 0.2          # 测试集比例
    random_state = 42        # 随机种子(保证可复现)

    # --- 模型相关 ---
    hidden_dims = [128, 64]  # 隐藏层维度列表，如 [128, 64] 表示两层隐藏层
    dropout_rate = 0.3       # Dropout 比率，防止过拟合(0=不用, 0.3=随机丢弃30%)

    # --- 训练相关 ---
    batch_size = 32          # 每次训练的样本数
    learning_rate = 0.001    # 学习率(Adam推荐0.001)
    epochs = 100             # 训练轮数
    patience = 10            # 早停耐心值(验证损失连续patience轮不下降则停止)

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Step 3: 加载和预处理数据
# ============================================================
def load_data(cfg):
    """
    加载数据并预处理。
    
    【数据预处理4步曲】(分类任务必备):
    1. 划分训练集/测试集
    2. 特征标准化 (StandardScaler: 均值→0, 标准差→1)
    3. 转为PyTorch张量
    4. 封装为DataLoader
    
    为什么要标准化？
    - 不同特征的量纲可能差异巨大(如年龄0-100 vs 收入0-1000000)
    - 不标准化会导致梯度下降偏向大数值特征，训练不稳定
    """

    # --- 方式1: 使用sklearn生成模拟数据(可直接运行) ---
    X, y = make_classification(
        n_samples=cfg.num_samples,
        n_features=cfg.num_features,
        n_classes=cfg.num_classes,
        n_informative=cfg.num_features // 2,  # 有信息的特征数
        n_redundant=cfg.num_features // 4,     # 冗余特征数
        random_state=cfg.random_state,
    )

    # --- 方式2: 加载你自己的数据(替换上面的make_classification) ---
    # import pandas as pd
    # df = pd.read_csv("your_data.csv")
    # X = df.iloc[:, :-1].values          # 除最后一列外的所有列作为特征
    # y = df.iloc[:, -1].values           # 最后一列作为标签
    #
    # 如果标签是字符串(如"cat"/"dog"), 需要编码为数字:
    # from sklearn.preprocessing import LabelEncoder
    # le = LabelEncoder()
    # y = le.fit_transform(y)

    # Step 3.1: 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    # stratify=y: 保证训练集和测试集中各类别的比例与原始数据一致

    # Step 3.2: 特征标准化(只用训练集fit，测试集用相同的参数transform)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit: 计算均值和标准差
    X_test = scaler.transform(X_test)         # transform: 用训练集的均值和标准差标准化

    # Step 3.3: 转为PyTorch张量
    # 分类任务: 标签必须是 long 类型(CrossEntropyLoss要求)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Step 3.4: 封装为DataLoader
    # DataLoader: 自动分批、打乱、并行加载
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    # 训练集 shuffle=True: 每轮打乱顺序，防止模型记住数据顺序
    # 测试集 shuffle=False: 不需要打乱，评估时顺序无关

    return train_loader, test_loader, X_test.to(cfg.device), y_test.to(cfg.device), scaler


# ============================================================
# Step 4: 定义FNN模型
# ============================================================
class FNNClassifier(nn.Module):
    """
    前馈神经网络分类器。
    
    【网络结构】
    输入层 → [BatchNorm → Linear → ReLU → Dropout] × N → 输出层
    
    【关键组件解析】
    - nn.Linear:      全连接层，学习特征之间的线性组合 y = Wx + b
    - nn.BatchNorm1d: 批归一化，稳定训练过程，加速收敛
                      原理：对每个mini-batch的数据做标准化(均值0,方差1)
                      作用：防止梯度消失/爆炸，允许更大学习率
    - nn.ReLU:        激活函数，引入非线性 max(0, x)
                      为什么需要？纯线性层堆叠等价于一个线性层，无法学习复杂模式
    - nn.Dropout:     随机丢弃神经元，防止过拟合
                      原理：训练时随机将部分神经元输出置0，迫使网络不依赖某些神经元
                      注意：只在训练时生效，评估时自动关闭
    
    【输出层】
    分类任务输出层不加激活函数！
    因为 nn.CrossEntropyLoss 内部已经包含了 Softmax 操作，
    如果再加 Softmax 就是重复计算，会导致梯度计算错误。
    """

    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.3):
        super(FNNClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.BatchNorm1d(prev_dim))         # 批归一化(放在Linear前)
            layers.append(nn.Linear(prev_dim, hidden_dim))   # 全连接层
            layers.append(nn.ReLU())                         # 激活函数
            layers.append(nn.Dropout(dropout_rate))           # Dropout防过拟合
            prev_dim = hidden_dim

        # 输出层(不加激活函数，CrossEntropyLoss内置Softmax)
        layers.append(nn.Linear(prev_dim, num_classes))

        # 用nn.Sequential将所有层串联起来
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================
# Step 5: 训练函数
# ============================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch，返回平均训练损失"""
    model.train()  # 切换到训练模式(启用Dropout和BatchNorm的训练行为)
    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # --- 训练4步曲(每个batch必须执行) ---
        optimizer.zero_grad()          # 1. 清零梯度(防止梯度累积)
        outputs = model(batch_x)       # 2. 前向传播：输入→输出
        loss = criterion(outputs, batch_y)  # 3. 计算损失：预测vs真实
        loss.backward()                # 4. 反向传播：计算梯度
        optimizer.step()               # 5. 更新参数：w = w - lr * grad

        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    return total_loss / total_samples


def evaluate(model, test_loader, criterion, device):
    """评估模型，返回平均损失和准确率"""
    model.eval()  # 切换到评估模式(关闭Dropout，BatchNorm用全局统计量)
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 不计算梯度，节省内存和加速
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

            # 取概率最大的类别作为预测结果
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
    train_loader, test_loader, X_test, y_test, scaler = load_data(cfg)
    print(f"训练集大小: {len(train_loader.dataset)}, 测试集大小: {len(test_loader.dataset)}")

    # --- 创建模型 ---
    model = FNNClassifier(
        input_dim=cfg.num_features,
        hidden_dims=cfg.hidden_dims,
        num_classes=cfg.num_classes,
        dropout_rate=cfg.dropout_rate,
    ).to(cfg.device)
    print(f"\n模型结构:\n{model}")

    # --- 损失函数和优化器 ---
    # CrossEntropyLoss: 交叉熵损失，分类任务标准选择
    # 内部 = LogSoftmax + NLLLoss，所以模型输出不需要Softmax
    criterion = nn.CrossEntropyLoss()

    # Adam: 自适应学习率优化器，综合了Momentum和RMSProp的优点
    # 大多数情况下Adam是最佳选择，无需手动调学习率衰减策略
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # 学习率调度器：验证损失停滞时自动降低学习率
    # ReduceLROnPlateau: 当监控指标不再下降时，将学习率乘以factor
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # --- 训练循环 ---
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print("\n开始训练...")
    for epoch in range(cfg.epochs):
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        # 评估
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, cfg.device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 学习率调度
        scheduler.step(val_loss)

        # 早停机制(Early Stopping)
        # 原理：如果验证损失连续patience轮不再下降，说明模型开始过拟合，应停止训练
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()  # 保存最佳模型参数
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"\n早停触发！在第 {epoch + 1} 轮停止训练")
                break

        # 打印训练进度
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch + 1}/{cfg.epochs}] "
                f"训练损失: {train_loss:.4f} | "
                f"验证损失: {val_loss:.4f} | "
                f"验证准确率: {val_acc:.4f}"
            )

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n已恢复最佳模型(验证损失: {best_val_loss:.4f})")

    # ============================================================
    # Step 7: 最终评估
    # ============================================================
    _, final_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, cfg.device)

    print(f"\n{'='*50}")
    print(f"最终测试准确率: {final_acc * 100:.2f}%")
    print(f"{'='*50}")

    # 详细分类报告(精确率、召回率、F1)
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    # 混淆矩阵
    print("混淆矩阵:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # ============================================================
    # Step 8: 可视化训练过程
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
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
    plt.savefig("fnn_classification_training.png", dpi=150)
    plt.show()

    # ============================================================
    # Step 9: 单样本预测示例
    # ============================================================
    model.eval()
    with torch.no_grad():
        # 取测试集前5个样本做预测演示
        sample_x = X_test[:5]
        outputs = model(sample_x)
        probabilities = torch.softmax(outputs, dim=1)  # 转为概率
        predicted_classes = torch.argmax(probabilities, dim=1)

        print("\n单样本预测示例:")
        for i in range(5):
            true_label = y_test[i].item()
            pred_label = predicted_classes[i].item()
            confidence = probabilities[i][pred_label].item()
            print(
                f"  样本{i}: 真实类别={true_label}, "
                f"预测类别={pred_label}, 置信度={confidence:.4f}"
            )

    # ============================================================
    # Step 10: 模型保存与加载
    # ============================================================
    model_path = "fnn_classification_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")

    # 加载模型(使用时需要确保模型结构一致)
    loaded_model = FNNClassifier(
        input_dim=cfg.num_features,
        hidden_dims=cfg.hidden_dims,
        num_classes=cfg.num_classes,
        dropout_rate=cfg.dropout_rate,
    ).to(cfg.device)
    loaded_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=cfg.device))
    loaded_model.eval()
    print("模型加载成功！")

    # ============================================================
    # Step 11: 对新数据进行预测(生产环境用法)
    # ============================================================
    # new_data = np.array([[...], [...]])  # 你的新数据，shape: (n_samples, n_features)
    # new_data_scaled = scaler.transform(new_data)  # 用训练时的scaler标准化
    # new_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).to(cfg.device)
    # with torch.no_grad():
    #     outputs = loaded_model(new_tensor)
    #     probs = torch.softmax(outputs, dim=1)
    #     preds = torch.argmax(probs, dim=1)
    # print("新数据预测结果:", preds.cpu().numpy())


if __name__ == "__main__":
    main()
