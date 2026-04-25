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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.datasets import make_multilabel_classification  # 生成模拟多标签数据
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    "SimHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# Step 2: 配置超参数
# ============================================================
class CONFIG:
    # --- 数据相关 ---
    num_samples = 1000       # 样本总数
    num_features = 20        # 输入特征维度
    num_labels = 5           # 标签数量(注意：不是类别数！每个标签是0/1)
    test_size = 0.2          # 测试集比例
    random_state = 42        # 随机种子

    # --- 模型相关 ---
    hidden_dims = [128, 64]  # 隐藏层维度列表
    dropout_rate = 0.3       # Dropout比率
    threshold = 0.5          # 预测阈值: sigmoid概率>threshold则预测为1

    # --- 训练相关 ---
    batch_size = 32
    learning_rate = 0.001
    epochs = 100
    patience = 15            # 早停耐心值

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Step 3: 加载和预处理数据
# ============================================================
def load_data(cfg):
    """
    加载多标签数据并预处理。
    
    【多标签数据格式】
    X: (n_samples, n_features) — 特征矩阵，与分类/回归相同
    y: (n_samples, n_labels)   — 标签矩阵，每个标签是0或1
    
    示例：
        y = [[1, 0, 1, 0, 0],   # 样本1有标签0和标签2
             [0, 1, 0, 1, 1],   # 样本2有标签1、3、4
             [0, 0, 0, 0, 0]]   # 样本3没有标签
    
    【标签预处理】
    多标签任务中，标签已经是0/1矩阵，通常不需要额外编码。
    如果原始标签是文本(如["体育","国际"])，需要转为0/1矩阵：
        使用 sklearn.preprocessing.MultiLabelBinarizer
    """

    # --- 方式1: 使用sklearn生成模拟数据 ---
    X, y = make_multilabel_classification(
        n_samples=cfg.num_samples,
        n_features=cfg.num_features,
        n_labels=cfg.num_labels,
        n_classes=cfg.num_labels,      # 每个标签至少出现一次
        allow_unlabeled=False,          # 不允许样本没有标签
        random_state=cfg.random_state,
    )

    # --- 方式2: 加载你自己的数据 ---
    # import pandas as pd
    # df = pd.read_csv("your_data.csv")
    # X = df.iloc[:, :cfg.num_features].values  # 前N列为特征
    # y = df.iloc[:, cfg.num_features:].values   # 后M列为标签(0/1)
    #
    # 如果标签是文本列表，用MultiLabelBinarizer转换:
    # from sklearn.preprocessing import MultiLabelBinarizer
    # mlb = MultiLabelBinarizer()
    # y = mlb.fit_transform(y_text_list)  # y_text_list如 [["体育","国际"], ["财经"]]

    # Step 3.1: 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    # Step 3.2: 特征标准化(标签不需要标准化，因为已经是0/1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 3.3: 转为PyTorch张量
    # 多标签: 标签是 float32 的二维张量 (n_samples, n_labels)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)   # float32！(BCEWithLogitsLoss要求)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Step 3.4: 封装为DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, test_loader, X_test.to(cfg.device), y_test.to(cfg.device), scaler


# ============================================================
# Step 4: 定义FNN多标签分类模型
# ============================================================
class FNNMultiLabel(nn.Module):
    """
    前馈神经网络多标签分类器。
    
    【与分类模型的关键区别】
    1. 输出层维度 = 标签数(num_labels)，不是类别数
    2. 输出层后接 Sigmoid(不是Softmax！)
       - Sigmoid: 每个输出独立计算，值域(0,1)，各标签概率互不影响
       - Softmax: 所有输出归一化为概率分布，和=1，标签间互斥(不适合多标签)
    3. 但注意：代码中输出层不加Sigmoid，因为BCEWithLogitsLoss内置了Sigmoid
    
    【为什么不直接在模型里加Sigmoid？】
    BCEWithLogitsLoss = Sigmoid + BCELoss 的数值稳定版本
    直接用logits(未激活的输出)计算损失，避免Sigmoid的数值溢出问题
    预测时再手动加Sigmoid即可
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

    def forward(self, x):
        # 输出shape: (batch_size, num_labels)，每个值是logit(未激活)
        return self.network(x)


# ============================================================
# Step 5: 训练和评估函数
# ============================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)            # (batch_size, num_labels) — logits
        loss = criterion(outputs, batch_y)   # BCEWithLogitsLoss自动加Sigmoid
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    return total_loss / total_samples


def evaluate(model, test_loader, criterion, device, threshold=0.5):
    """
    评估多标签模型。
    
    【多标签评估指标】
    - Hamming Loss: 错误预测的标签比例，越小越好
    - F1(micro): 全局计算所有标签的F1，适合标签分布均衡
    - F1(macro): 每个标签单独算F1再平均，适合标签分布不均衡
    - Precision/Recall: 各标签的精确率和召回率
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

            # Sigmoid + 阈值判定
            probs = torch.sigmoid(outputs)                # logits → 概率
            preds = (probs > threshold).float()            # 概率 > 阈值 → 预测为1

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
    train_loader, test_loader, X_test, y_test, scaler = load_data(cfg)
    print(f"训练集大小: {len(train_loader.dataset)}, 测试集大小: {len(test_loader.dataset)}")
    print(f"标签数量: {cfg.num_labels}")
    # 统计每个标签的正样本比例
    y_train_np = train_loader.dataset.tensors[1].numpy()
    for i in range(cfg.num_labels):
        pos_ratio = y_train_np[:, i].mean()
        print(f"  标签{i} 正样本比例: {pos_ratio:.2%}")

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
    # 【处理标签不平衡】可以设置pos_weight参数
    # pos_weight[i] = 负样本数 / 正样本数 (对每个标签分别计算)
    # 作用：增加正样本的权重，缓解正负样本不平衡问题
    pos_weights = []
    for i in range(cfg.num_labels):
        pos_count = y_train_np[:, i].sum()
        neg_count = len(y_train_np) - pos_count
        # 如果正样本为0，权重设为1；否则计算负/正比
        pw = neg_count / max(pos_count, 1)
        pos_weights.append(pw)
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(cfg.device)
    print(f"\n各标签pos_weight: {[f'{w:.2f}' for w in pos_weights]}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # --- 训练循环 ---
    train_losses = []
    val_losses = []
    val_f1s = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print("\n开始训练...")
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        val_loss, val_preds, val_labels = evaluate(
            model, test_loader, criterion, cfg.device, cfg.threshold
        )

        # 计算验证F1
        val_f1 = f1_score(val_labels, val_preds, average="micro", zero_division=0)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        scheduler.step(val_loss)

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"\n早停触发！在第 {epoch + 1} 轮停止训练")
                break

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch + 1}/{cfg.epochs}] "
                f"训练损失: {train_loss:.4f} | "
                f"验证损失: {val_loss:.4f} | "
                f"验证F1(micro): {val_f1:.4f}"
            )

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n已恢复最佳模型(验证损失: {best_val_loss:.4f})")

    # ============================================================
    # Step 7: 最终评估
    # ============================================================
    _, all_preds, all_labels = evaluate(model, test_loader, criterion, cfg.device, cfg.threshold)

    # 多标签评估指标
    h_loss = hamming_loss(all_labels, all_preds)
    f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision_micro = precision_score(all_labels, all_preds, average="micro", zero_division=0)
    recall_micro = recall_score(all_labels, all_preds, average="micro", zero_division=0)

    print(f"\n{'='*50}")
    print(f"多标签分类评估指标:")
    print(f"  Hamming Loss:        {h_loss:.4f}  (越低越好)")
    print(f"  F1 (micro):          {f1_micro:.4f}  (全局F1)")
    print(f"  F1 (macro):          {f1_macro:.4f}  (各标签F1平均)")
    print(f"  Precision (micro):   {precision_micro:.4f}")
    print(f"  Recall (micro):      {recall_micro:.4f}")
    print(f"{'='*50}")

    # 各标签详细报告
    label_names = [f"标签{i}" for i in range(cfg.num_labels)]
    print("\n各标签详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))

    # ============================================================
    # Step 8: 可视化
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axes[0].plot(train_losses, label="训练损失")
    axes[0].plot(val_losses, label="验证损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("损失曲线")
    axes[0].legend()

    # F1曲线
    axes[1].plot(val_f1s, label="验证F1(micro)", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("F1曲线")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("fnn_multilabel_training.png", dpi=150)
    plt.show()

    # ============================================================
    # Step 9: 单样本预测示例
    # ============================================================
    model.eval()
    with torch.no_grad():
        sample_x = X_test[:5]
        logits = model(sample_x)
        probs = torch.sigmoid(logits)     # logits → 概率
        preds = (probs > cfg.threshold).float()

        print("\n单样本预测示例:")
        for i in range(5):
            true_labels = y_test[i].cpu().numpy().astype(int)
            pred_labels = preds[i].cpu().numpy().astype(int)
            prob_values = probs[i].cpu().numpy()
            predicted_label_indices = np.where(pred_labels == 1)[0]
            true_label_indices = np.where(true_labels == 1)[0]
            print(
                f"  样本{i}: 真实标签={true_label_indices.tolist()}, "
                f"预测标签={predicted_label_indices.tolist()}, "
                f"各标签概率={[f'{p:.3f}' for p in prob_values]}"
            )

    # ============================================================
    # Step 10: 阈值调优(可选)
    # ============================================================
    # 默认阈值0.5不一定最优，可以搜索最佳阈值
    print("\n阈值调优:")
    model.eval()
    with torch.no_grad():
        all_logits = model(X_test)
        all_probs = torch.sigmoid(all_logits).cpu().numpy()
    all_labels_np = y_test.cpu().numpy()

    best_threshold = 0.5
    best_f1 = 0
    for t in np.arange(0.3, 0.7, 0.05):
        preds_t = (all_probs > t).astype(float)
        f1_t = f1_score(all_labels_np, preds_t, average="micro", zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = t
    print(f"  最佳阈值: {best_threshold:.2f}, 对应F1(micro): {best_f1:.4f}")
    print(f"  (默认阈值0.5, 可根据需求调整)")

    # ============================================================
    # Step 11: 模型保存与加载
    # ============================================================
    model_path = "fnn_multilabel_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")

    loaded_model = FNNMultiLabel(
        input_dim=cfg.num_features,
        hidden_dims=cfg.hidden_dims,
        num_labels=cfg.num_labels,
        dropout_rate=cfg.dropout_rate,
    ).to(cfg.device)
    loaded_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=cfg.device))
    loaded_model.eval()
    print("模型加载成功！")

    # ============================================================
    # Step 12: 对新数据进行预测(生产环境用法)
    # ============================================================
    # new_data = np.array([[...], [...]])  # shape: (n_samples, n_features)
    # new_data_scaled = scaler.transform(new_data)
    # new_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).to(cfg.device)
    # with torch.no_grad():
    #     logits = loaded_model(new_tensor)
    #     probs = torch.sigmoid(logits)
    #     preds = (probs > best_threshold).float()
    # print("新数据预测标签:", preds.cpu().numpy().astype(int))


if __name__ == "__main__":
    main()
