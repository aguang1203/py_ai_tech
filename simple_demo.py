"""
简单PyTorch Demo - 验证PyTorch安装并快速展示基本功能

【学习目标】
本demo旨在帮助你理解深度学习的核心概念：
1. 张量(Tensor) - 深度学习的基本数据结构
2. 自动求导(Autograd) - PyTorch的核心特性，自动计算梯度
3. 神经网络构建 - 如何定义和搭建网络结构
4. 训练流程 - 前向传播、损失计算、反向传播、参数更新

【深度学习基础概念】
- 张量: 多维数组，是标量(0维)、向量(1维)、矩阵(2维)的推广
- 梯度: 函数在某点的变化率，用于指导参数更新方向
- 前向传播: 数据从输入层经过各层计算得到输出
- 反向传播: 根据输出误差，从后往前计算每层参数的梯度
- 损失函数: 衡量预测值与真实值的差距
- 优化器: 根据梯度更新模型参数的算法
"""

import torch              # PyTorch主库，提供张量运算和自动求导
import torch.nn as nn     # 神经网络模块，包含各种层和损失函数
import numpy as np        # 数值计算库，用于数据处理

print("="*60)
print("PyTorch 基础功能演示")
print("="*60)

# ============================================================================
# 第一步：环境检查
# ============================================================================

# 1. 检查PyTorch版本
print(f"\n✓ PyTorch版本: {torch.__version__}")

# 2. 检查计算设备（GPU/CPU）
# 【原理说明】
# GPU(Graphics Processing Unit)拥有大量并行计算单元，适合矩阵运算
# CUDA是NVIDIA推出的并行计算平台和编程模型
# PyTorch会自动检测可用设备，优先使用GPU加速训练
if torch.cuda.is_available():
    device = torch.device('cuda')  # 使用NVIDIA GPU
    print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA版本: {torch.version.cuda}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')   # 使用Apple Silicon芯片的MPS加速
    print("✓ MPS可用 (Apple Silicon)")
else:
    device = torch.device('cpu')   # 使用CPU
    print("⚠ 使用CPU")

# ============================================================================
# 第二步：张量操作 - 深度学习的数据基础
# ============================================================================
# 【什么是张量？】
# 张量(Tensor)是多维数组，是深度学习中最基本的数据结构
# - 0维张量: 标量 (scalar)，如: 5
# - 1维张量: 向量 (vector)，如: [1, 2, 3]
# - 2维张量: 矩阵 (matrix)，如: [[1,2], [3,4]]
# - 3维及以上: 高阶张量，如图像(batch, channel, height, width)
#
# 【为什么用张量？】
# 1. 支持GPU加速计算
# 2. 自动记录运算历史，用于反向传播
# 3. 提供丰富的数学运算接口

print("\n" + "-"*60)
print("1. 张量操作")
print("-"*60)

# 创建一个随机张量 (从标准正态分布中采样)
# torch.randn生成均值为0，方差为1的随机数
x = torch.randn(3, 4)  # 3行4列的矩阵
print(f"\n随机张量 (3x4):\n{x}")
# 输出示例:
# tensor([[ 0.5, -1.2,  0.8,  0.3],
#         [-0.7,  1.1, -0.4,  0.9],
#         [ 0.2, -0.6,  1.5, -0.1]])

# 张量运算 - 元素级别的加法
y = torch.ones(3, 4)   # 创建全1张量
z = x + y              # 对应位置相加
print(f"\n张量加法结果:\n{z}")
# 原理: z[i,j] = x[i,j] + y[i,j]

# 矩阵乘法 - 神经网络中的核心运算
# 【原理说明】
# 矩阵乘法是线性变换的基础，在神经网络中用于：
# - 全连接层: output = input @ weight.T + bias
# - 卷积运算可以转化为矩阵乘法
# - Attention机制中的Q@K^T
A = torch.randn(2, 3)  # 2×3矩阵
B = torch.randn(3, 4)  # 3×4矩阵
C = torch.mm(A, B)     # 矩阵乘法，结果维度: (2×3) × (3×4) = (2×4)
print(f"\n矩阵乘法 (2x3) × (3x4) = (2x4):\n{C.shape}")
# 计算公式: C[i,j] = Σ(A[i,k] * B[k,j]), k从0到2

# ============================================================================
# 第三步：自动求导 - PyTorch的核心魔法
# ============================================================================
# 【什么是自动求导？】
# 自动求导(Autograd)是PyTorch最核心的功能之一
# 它会自动记录所有对张量的操作，构建计算图(Computational Graph)
# 当调用backward()时，利用链式法则自动计算梯度
#
# 【链式法则】
# 如果 y = f(u), u = g(x)，那么 dy/dx = dy/du * du/dx
# 对于多层神经网络，梯度需要从输出层逐层反向传播到输入层
#
# 【为什么需要梯度？】
# 梯度告诉我们应该如何调整参数来减小损失
# - 梯度为正: 减小参数可以减小损失
# - 梯度为负: 增大参数可以减小损失
# - 梯度大小: 调整的幅度

print("\n" + "-"*60)
print("2. 自动求导 (Autograd)")
print("-"*60)

# 创建需要计算梯度的张量
# requires_grad=True 告诉PyTorch跟踪这个张量的所有操作
x = torch.tensor([2.0], requires_grad=True)

# 定义一个函数: y = x² + 3x + 1
# PyTorch会记录这个计算过程，构建计算图
y = x ** 2 + 3 * x + 1

# 反向传播 - 计算梯度
# 这会计算 dy/dx 并存储在 x.grad 中
y.backward()

# 手动验证: dy/dx = 2x + 3
# 当 x=2.0 时，dy/dx = 2*2 + 3 = 7
print(f"\nx = 2.0")
print(f"y = x² + 3x + 1 = {y.item()}")  # item()将单元素张量转为Python数值
print(f"dy/dx = 2x + 3 = {x.grad.item()}")  # 应该等于7.0

# 【计算图可视化】
# x --(平方)--> x² --\
#                     +--(加法)--> x²+3x --(加法)--> y
# x --(乘3)--> 3x --/                    /
# 常数1 ---------------------------------/

# ============================================================================
# 第四步：构建神经网络 - 从理论到实践
# ============================================================================
# 【神经网络基本原理】
# 神经网络由多层神经元组成，每一层执行：
# output = activation(input @ weight + bias)
# 
# 关键组件：
# 1. 线性层(Linear): 执行线性变换 y = xW + b
# 2. 激活函数(Activation): 引入非线性，如ReLU、Sigmoid
# 3. 前向传播(forward): 定义数据如何通过网络

print("\n" + "-"*60)
print("3. 简单神经网络")
print("-"*60)

# 定义神经网络类
# 【nn.Module】
# PyTorch中所有神经网络的基类
# 必须实现两个方法：
# - __init__: 定义网络结构（层）
# - forward: 定义数据流动方式（前向传播）
class SimpleNet(nn.Module):
    def __init__(self):
        """
        初始化网络结构
        
        网络架构:
        输入(10维) → Linear(10→5) → ReLU → Linear(5→2) → 输出(2维)
        
        参数说明：
        - Linear(10, 5): 将10维输入映射到5维隐藏层
          包含权重矩阵 W(5×10) 和偏置 b(5)
        - ReLU: 激活函数，f(x) = max(0, x)
          作用：引入非线性，使网络能学习复杂模式
        - Linear(5, 2): 将5维隐藏层映射到2维输出
          用于二分类任务
        """
        super(SimpleNet, self).__init__()  # 调用父类构造函数
        
        # 第一层：全连接层，10个输入神经元 → 5个隐藏神经元
        # 参数量: 10×5(weights) + 5(biases) = 55
        self.fc1 = nn.Linear(10, 5)
        
        # 第二层：全连接层，5个隐藏神经元 → 2个输出神经元
        # 参数量: 5×2(weights) + 2(biases) = 12
        self.fc2 = nn.Linear(5, 2)
        
        # ReLU激活函数
        # 【为什么需要激活函数？】
        # 如果没有激活函数，多层线性变换等价于单层线性变换
        # 激活函数引入非线性，使网络能够拟合任意复杂函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        前向传播函数 - 定义数据如何通过网络
        
        参数:
            x: 输入张量，形状为 (batch_size, 10)
        
        返回:
            输出张量，形状为 (batch_size, 2)
        
        数据流动:
        输入 → fc1 → ReLU → fc2 → 输出
        """
        # 第一层：线性变换 + 激活函数
        # x: (batch, 10) → fc1 → (batch, 5) → ReLU → (batch, 5)
        x = self.relu(self.fc1(x))
        
        # 第二层：线性变换（输出层通常不加激活函数）
        # x: (batch, 5) → fc2 → (batch, 2)
        x = self.fc2(x)
        
        return x

# 创建模型实例并移动到指定设备（GPU或CPU）
model = SimpleNet().to(device)
print(f"\n模型结构:\n{model}")
# 输出会显示网络层级结构和参数维度

# 创建随机输入数据进行测试
# batch_size=1, input_dim=10
input_data = torch.randn(1, 10).to(device)
output = model(input_data)  # 前向传播
print(f"\n输入形状: {input_data.shape}")  # torch.Size([1, 10])
print(f"输出形状: {output.shape}")        # torch.Size([1, 2])
print(f"输出值: {output.detach().cpu().numpy()}")
# detach(): 从计算图中分离，不计算梯度
# cpu(): 转移到CPU（如果在GPU上）
# numpy(): 转换为NumPy数组便于打印

# ============================================================================
# 第五步：训练流程 - 深度学习的核心循环
# ============================================================================
# 【训练四步骤】
# 1. 前向传播(Forward): 计算模型预测值
# 2. 计算损失(Loss): 衡量预测与真实的差距
# 3. 反向传播(Backward): 计算梯度
# 4. 参数更新(Step): 根据梯度更新权重
#
# 【优化器工作原理】
# SGD: w = w - lr * gradient
# Adam: 更复杂的自适应学习率算法
# lr(learning rate): 学习率，控制参数更新的步长

print("\n" + "-"*60)
print("4. 训练步骤演示")
print("-"*60)

# 模拟训练数据
# X_train: 100个样本，每个样本10个特征
X_train = torch.randn(100, 10).to(device)
# y_train: 100个标签，值为0或1（二分类）
y_train = torch.randint(0, 2, (100,)).to(device)

# 定义损失函数
# CrossEntropyLoss: 交叉熵损失，常用于分类任务
# 【交叉熵原理】
# Loss = -Σ(y_true * log(y_pred))
# 衡量预测概率分布与真实分布的差异
criterion = nn.CrossEntropyLoss()

# 定义优化器
# Adam: 自适应矩估计优化器
# - 结合 Momentum 和 RMSprop 的优点
# - 为每个参数维护独立的学习率
# - lr=0.01: 学习率，控制更新步长
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# model.parameters(): 返回模型所有可学习参数（weights和biases）

# 开始训练演示
print("\n开始训练...")
for step in range(3):  # 只训练3步作为演示
    # ========== 步骤1: 前向传播 ==========
    # 取一批数据（这里每次10个样本）
    batch_X = X_train[step*10:(step+1)*10]  # (10, 10)
    batch_y = y_train[step*10:(step+1)*10]  # (10,)
    
    # 通过模型计算预测值
    outputs = model(batch_X)  # (10, 2)
    
    # ========== 步骤2: 计算损失 ==========
    # 比较预测值和真实标签
    loss = criterion(outputs, batch_y)
    
    # ========== 步骤3: 反向传播 ==========
    # 清空之前的梯度（重要！否则会累积）
    optimizer.zero_grad()
    
    # 计算当前损失对所有参数的梯度
    loss.backward()
    
    # ========== 步骤4: 更新参数 ==========
    # 根据梯度和学习率更新权重
    optimizer.step()
    
    # 打印训练进度
    print(f"Step {step+1}/3, Loss: {loss.item():.4f}")

print("\n✅ 训练演示完成!")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*60)
print("总结")
print("="*60)
print("✓ PyTorch安装正常")
print("✓ 张量操作正常")
print("✓ 自动求导正常")
print("✓ 神经网络构建和训练正常")
print("\n🎉 一切就绪!可以开始使用PyTorch了!")
print("="*60)

# 【下一步学习建议】
# 1. 运行 fnn_demo.py - 学习前馈神经网络
# 2. 运行 cnn_demo.py - 学习卷积神经网络
# 3. 运行 rnn_demo.py - 学习循环神经网络
# 4. 运行 transformer_demo.py - 学习Transformer
# 5. 阅读 GUIDE.md 了解详细理论