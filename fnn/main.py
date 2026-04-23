# 1. 纯 PyTorch+CUDA 固定流程：配置设备 → 数据转张量→移到 CUDA → 定义 FNN 模型→移到 CUDA → 训练循环（梯度清零→前向→损失→反向→更新）；
# 2. CUDA 加速只需两步：model.to(device) + 张量.to(device)；
# 3. 回归 / 分类切换仅修改输出层 + 损失函数；
# 4. 所有代码完全脱离 TensorFlow，纯 PyTorch 生态，工业界标准写法。

# 1-start: 导入必要的库和模块

# 导入PyTorch库及其相关模块，用于构建和训练神经网络模型。
from ast import mod

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 数据处理（通用，仅用于加载/划分/标准化）
import numpy as np
import pandas as pd
# from sklearn.datasets import fetch_california_housing, make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# 数据可视化
import matplotlib.pyplot as plt

#  1-end: 导入必要的库和模块

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# 2-start: 配置设备（自动选择 CUDA / CPU）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备: ", device)

# 2-end: 配置设备（自动选择 CUDA / CPU）

# 3-start: 加载数据 + 预处理

# 加载数据
# data = fetch_california_housing()
# x, y = data.data, data.target


# 划分训练集/测试集,将数据集划分为训练集和测试集，测试集占比20%，设置随机种子保证结果可复现
# 总结：x_train 是"因"，y_train 是"果"，神经网络的任务就是学习这个因果关系。
# x_train: 特征数据（输入）
# y_train: 标签数据（输出/目标）
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, random_state=42
# )

df = pd.read_csv("/home/hjg/dev/datasets/iris.data.txt")
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 特征标准化（FNN必须）
# 标准化目的(需要分场景,有些情况不需要特征标准化)：
# 1. 梯度下降会偏向大数值特征（如Population）
# 2. 训练速度慢，可能无法收敛
# 3. 模型性能差
# 1. 创建编码器

# 神经网络只支持32位浮点型和整数型标签，字符串标签必须转换为数字。LabelEncoder是一个工具类，用于将分类标签（如字符串）转换为整数编码，方便神经网络处理。
le = LabelEncoder()

# 2. 把字符串标签变成数字
y_train = le.fit_transform(y_train)  
y_test = le.transform(y_test)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 【关键】转为PyTorch张量，并移到CUDA设备
# 将训练数据转换为32位浮点型张量并移至指定设备。
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)

# reshape(-1, 1) 的主要目的是将标签数据重塑为列向量（二维张量），以匹配模型输出和损失函数的维度要求。
# 这个操作是为了确保 y_train 从 [样本数] 变成 [样本数, 1]，使其成为一个标准的列向量，从而能与模型输出的二维张量在维度上完全对齐，顺利进行数学运算或损失计算。
y_train = torch.tensor(y_train, dtype=torch.long).reshape(-1, 1).to(device)

x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).reshape(-1, 1).to(device)

# 3-end: 加载数据 + 预处理

# 4-start 封装数据加载器（PyTorch 标准用法）

# 加载数据加载器,封装数据集
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# 创建数据加载器,使用DataLoader封装数据集，设置批量大小为32，训练集启用随机打乱（shuffle=True），测试集不打乱（shuffle=False）。
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4-end 封装数据加载器（PyTorch 标准用法）

# 5-start: 定义 FNN 模型（纯 PyTorch）


# 定义一个简单的前馈神经网络（FNN）模型，包含两层全连接层和ReLU激活函数。
class FNN(nn.Module):

    # 定义网络结构
    def __init__(self, input_dim=8):
        super(FNN, self).__init__()

        # 定义网络结构: 输入层 -> 隐藏层 -> 输出层
        # 输入：每个样本有 8 个特征值
        # 输出：每个样本被映射到 64 维的隐藏空间
        # 本质：学习 8 个原始特征之间的非线性组合关系

        # nn.Linear(8, 64) 的本质：
        # 不是简单的特征转换，而是学习到 64 种不同的特征组合方式
        # 每个神经元都学习捕捉输入特征的不同非线性组合模式
        # 维度扩展是为了让网络有足够的表达能力去拟合复杂的关系
        # 后续层（64→32→1）逐步提取最关键的信息，最终输出预测结果
        # 这就像把 8 种基础颜料混合调配，创造出 64 种新的色彩，再用这些新色彩绘制出最终的画作（房价预测）。
        self.fc1 = nn.Linear(
            input_dim, 64
        )  # 输入层到隐藏层，输入特征数为8，隐藏层神经元数为64

        self.fc2 = nn.Linear(64, 32)  # 隐藏层到输出层，输出一个连续值（房价）
        self.fc3 = nn.Linear(32, 1)  # 隐藏层到输出层，输出一个连续值（房价）
        self.relu = nn.ReLU()  # ReLU激活函数

    # （FNN核心逻辑）定义前向传播,在前向传播过程中，输入数据依次通过全连接层和ReLU激活函数，最终输出预测结果。
    def forward(self, x):
        x = self.relu(self.fc1(x))  # 输入层 -> 隐藏层1 -> ReLU激活
        x = self.relu(self.fc2(x))  # 隐藏层1 -> 隐藏层2 -> ReLU激活
        x = self.fc3(x)  # 隐藏层2 -> 输出层（线性输出）
        return x


# 初始化模型 + 移到CUDA
# 鸢尾花特征：4个特征
model = FNN(input_dim=4).to(device)
print(model)

# 5-end: 定义 FNN 模型（纯 PyTorch）

# 6-start: 定义损失函数 + 优化器

# 衡量模型预测有多"错"
criterion = nn.MSELoss()  # 均方误差损失函数

optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 6-end: 定义损失函数 + 优化器

# 7-start: CUDA 加速训练循环（核心！）

# 训练循环,训练模型，迭代指定的训练轮数（epochs），在每轮中遍历训练数据加载器，计算损失并更新模型参数。
epochs = 50

train_loss_list = []
val_loss_list = []

for epoch in range(epochs):

    # 训练模式
    model.train()
    train_loss = 0.0

    for batch_x, batch_y in train_loader:

        # zero_grad(): 防止梯度累积,确保每次计算都是"干净的"
        # 前向传播: 数据从输入流向输出,产生预测结果
        # 损失计算: 量化预测与真实的差距,指导模型改进方向
        # 三者缺一不可: 这是每个训练batch必须执行的标准化流程

        # 反向传播,计算梯度,在每个训练批次开始时，调用optimizer.zero_grad()清除之前计算的梯度，确保当前批次的梯度计算不受之前批次的影响。
        optimizer.zero_grad()

        # 前向传播,计算预测结果，根据输入数据计算模型输出，并计算损失。
        outputs = model(batch_x)

        # 损失计算,计算预测结果与真实值的差距，并计算损失。
        loss = criterion(outputs, batch_y)

        # 反向传播,计算梯度，根据当前的损失值计算模型参数的梯度，这些梯度将用于更新模型参数以最小化损失。
        loss.backward()

        # 更新参数,根据计算得到的梯度更新模型参数，优化器会根据设定的学习率和算法规则调整权重。
        optimizer.step()

        train_loss += loss.item() * batch_x.size(0)  # 累加损失

    # 平均训练损失
    train_loss = train_loss / len(train_loader.dataset)
    train_loss_list.append(train_loss)

    # 验证模式,不更新参数, 不计算梯度, 目的是提速和评估模型性能
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)

    # 平均验证损失
    val_loss = val_loss / len(test_loader.dataset)
    val_loss_list.append(val_loss)

    # 输出训练进度和损失,每5轮输出一次当前的训练损失和验证损失，帮助监控模型的训练过程和性能。
    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"训练损失: {train_loss:.4f} 测试损失: {val_loss:.4f}"
        )

# 7-end: CUDA 加速训练循环（核心！）

# 8-start: 可视化训练过程

plt.plot(train_loss_list, label="训练损失")
plt.plot(val_loss_list, label="测试损失")
plt.xlabel("轮数")
plt.ylabel("损失")
plt.legend()
plt.show()

# 8-end: 可视化训练过程

# 9-start: 模型评估

# 计算平均绝对误差（MAE），直观看预测精度
model.eval()
with torch.no_grad():
    y_pred = model(x_test)
    mae = torch.mean(torch.abs(y_pred - y_test))
    print(f"测试MAE: {mae.item():.4f} (10万美元)")

# 9-end: 模型评估

# 10-start: 真实样本预测

# 这里预测前5个样本
model.eval()
with torch.no_grad():
    pred = model(x_test[:5]).cpu().numpy()  # 转回CPU用于打印
    true = y_test[:5].cpu().numpy()

print("预测房价:", pred.flatten())
print("真实房价:", true.flatten())

# 10-end: 真实样本预测

# start: 模型保存和加载

torch.save(model.state_dict(), "model.pth")

# 模型加载
model = FNN(input_dim=8)
model.load_state_dict(torch.load("model.pth"))

# end: 模型保存和加载

# start: 延伸

# 分类任务（快速修改，纯 PyTorch）
# 如果做分类（如鸢尾花），只改 3 处：
# 输出层神经元 = 类别数
# 输出层加 nn.Softmax(dim=1)
# 损失函数改为 nn.CrossEntropyLoss()

# end: 延伸


# start: 进阶,优化 FNN（防过拟合 + 更强模型）


class FNN_Improved(nn.Module):
    def __init__(self, input_dim=8):
        super(FNN_Improved, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.drop1 = nn.Dropout(0.2)  # 随机丢弃20%神经元
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# end: 进阶,优化 FNN（防过拟合 + 更强模型）
