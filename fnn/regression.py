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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.datasets import make_regression  # 生成模拟回归数据
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# Step 2: 配置超参数
# ============================================================
class CONFIG:
    # --- 数据相关 ---
    num_samples = 1000       # 样本总数
    num_features = 10        # 输入特征维度
    test_size = 0.2          # 测试集比例
    random_state = 42        # 随机种子

    # --- 模型相关 ---
    hidden_dims = [128, 64]  # 隐藏层维度列表
    dropout_rate = 0.2       # Dropout比率(回归任务通常比分类小，因为数据更平滑)

    # --- 训练相关 ---
    batch_size = 32
    learning_rate = 0.001
    epochs = 100
    patience = 15            # 早停耐心值(回归任务通常给更多耐心，因为损失波动更大)

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
    X, y = make_regression(
        n_samples=cfg.num_samples,
        n_features=cfg.num_features,
        n_informative=cfg.num_features,  # 所有特征都有信息
        noise=10.0,                       # 添加噪声，模拟真实数据
        random_state=cfg.random_state,
    )

    # --- 方式2: 加载你自己的数据 ---
    # import pandas as pd
    # df = pd.read_csv("your_data.csv")
    # X = df.iloc[:, :-1].values
    # y = df.iloc[:, -1].values  # 标签是连续数值

    # Step 3.1: 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    # Step 3.2: 特征标准化
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_test = feature_scaler.transform(X_test)

    # Step 3.3: 标签标准化(回归任务特有！)
    # 将y也标准化，让目标值在0附近，训练更稳定
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

    return train_loader, test_loader, X_test.to(cfg.device), y_test.to(cfg.device), feature_scaler, y_scaler


# ============================================================
# Step 4: 定义FNN回归模型
# ============================================================
class FNNRegressor(nn.Module):
    """
    前馈神经网络回归器。
    
    【与分类模型的关键区别】
    1. 输出层只有1个神经元(回归预测单个连续值)
    2. 输出层不加激活函数(因为预测值可以是任意实数，不能限制在0-1之间)
    3. 如果多目标回归(预测多个数值)，输出层神经元数=目标数
    
    【为什么回归输出层不加激活函数？】
    - ReLU: 输出≥0，无法预测负数(如温度-10°C)
    - Sigmoid: 输出0-1，范围太窄
    - Softmax: 输出和为1，完全不适合回归
    - 无激活: 输出可以是任意实数(-∞ ~ +∞)，完美适配回归
    """

    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
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
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # 输出shape: (batch_size, 1)，需要squeeze掉最后一维
        return self.network(x).squeeze(-1)


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
        outputs = model(batch_x)           # (batch_size,)
        loss = criterion(outputs, batch_y)  # 标签也是1维的
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    return total_loss / total_samples


def evaluate(model, test_loader, criterion, device):
    """评估模型，返回损失和预测结果"""
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
    train_loader, test_loader, X_test, y_test, feature_scaler, y_scaler = load_data(cfg)
    print(f"训练集大小: {len(train_loader.dataset)}, 测试集大小: {len(test_loader.dataset)}")

    # --- 创建模型 ---
    model = FNNRegressor(
        input_dim=cfg.num_features,
        hidden_dims=cfg.hidden_dims,
        dropout_rate=cfg.dropout_rate,
    ).to(cfg.device)
    print(f"\n模型结构:\n{model}")

    # --- 损失函数和优化器 ---
    # MSELoss: 均方误差 = mean((pred - true)²)
    # 优点：对大误差惩罚更大，适合大多数回归任务
    # 缺点：对异常值敏感(因为平方会放大异常值的影响)
    criterion = nn.MSELoss()

    # 如果数据中有较多异常值，可以改用 L1Loss (MAE):
    # criterion = nn.L1Loss()  # 平均绝对误差 = mean(|pred - true|)

    # 或者使用 SmoothL1Loss (Huber Loss)，对异常值更鲁棒:
    # criterion = nn.SmoothL1Loss()  # 误差小时用平方，误差大时用绝对值

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # --- 训练循环 ---
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print("\n开始训练...")
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        val_loss, _, _ = evaluate(model, test_loader, criterion, cfg.device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

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
                f"训练MSE: {train_loss:.4f} | 验证MSE: {val_loss:.4f}"
            )

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n已恢复最佳模型(验证MSE: {best_val_loss:.4f})")

    # ============================================================
    # Step 7: 最终评估
    # ============================================================
    _, all_preds_scaled, all_labels_scaled = evaluate(model, test_loader, criterion, cfg.device)

    # 反标准化：将预测值和真实值还原到原始尺度
    # 因为我们对y做了标准化，所以需要用y_scaler反变换才能看到真实数值
    all_preds_real = y_scaler.inverse_transform(all_preds_scaled.reshape(-1, 1)).flatten()
    all_labels_real = y_scaler.inverse_transform(all_labels_scaled.reshape(-1, 1)).flatten()

    # 计算评估指标(在原始尺度上)
    mse = mean_squared_error(all_labels_real, all_preds_real)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels_real, all_preds_real)
    r2 = r2_score(all_labels_real, all_preds_real)

    print(f"\n{'='*50}")
    print(f"回归评估指标(原始尺度):")
    print(f"  MSE  (均方误差):     {mse:.4f}")
    print(f"  RMSE (均方根误差):   {rmse:.4f}")
    print(f"  MAE  (平均绝对误差): {mae:.4f}")
    print(f"  R²   (决定系数):     {r2:.4f}")
    print(f"{'='*50}")
    # R² 解释：1=完美预测，0=和均值一样差，<0=比均值还差

    # ============================================================
    # Step 8: 可视化
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 损失曲线
    axes[0].plot(train_losses, label="训练MSE")
    axes[0].plot(val_losses, label="验证MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("损失曲线")
    axes[0].legend()

    # 预测值 vs 真实值散点图(越接近对角线越好)
    axes[1].scatter(all_labels_real, all_preds_real, alpha=0.5, s=10)
    min_val = min(all_labels_real.min(), all_preds_real.min())
    max_val = max(all_labels_real.max(), all_preds_real.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--", label="完美预测线")
    axes[1].set_xlabel("真实值")
    axes[1].set_ylabel("预测值")
    axes[1].set_title(f"预测 vs 真实 (R²={r2:.4f})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("fnn_regression_training.png", dpi=150)
    plt.show()

    # ============================================================
    # Step 9: 单样本预测示例
    # ============================================================
    model.eval()
    with torch.no_grad():
        sample_x = X_test[:5]
        preds_scaled = model(sample_x).cpu().numpy()
        # 反标准化得到真实预测值
        preds_real = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        labels_real = y_scaler.inverse_transform(y_test[:5].cpu().numpy().reshape(-1, 1)).flatten()

        print("\n单样本预测示例(原始尺度):")
        for i in range(5):
            error = abs(preds_real[i] - labels_real[i])
            print(f"  样本{i}: 真实值={labels_real[i]:.2f}, 预测值={preds_real[i]:.2f}, 误差={error:.2f}")

    # ============================================================
    # Step 10: 模型保存与加载
    # ============================================================
    model_path = "fnn_regression_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")

    loaded_model = FNNRegressor(
        input_dim=cfg.num_features,
        hidden_dims=cfg.hidden_dims,
        dropout_rate=cfg.dropout_rate,
    ).to(cfg.device)
    loaded_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=cfg.device))
    loaded_model.eval()
    print("模型加载成功！")

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
