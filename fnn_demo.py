"""
FNN (Feedforward Neural Network) - MNIST 手写数字识别

【学习目标】
本demo将帮助你深入理解前馈神经网络（全连接网络）的原理：
1. 多层感知机(MLP)的架构设计
2. Batch Normalization的作用和原理
3. Dropout正则化防止过拟合
4. 学习率调度策略
5. 完整的训练流程

【FNN核心概念】
- 前馈神经网络: 数据从输入层单向流向输出层，无反馈连接
- 全连接层: 每个神经元与前一层所有神经元相连
- 隐藏层: 位于输入层和输出层之间的中间层
- 激活函数: 引入非线性，使网络能学习复杂模式
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os


# ==================== 配置参数 ====================
# 【超参数说明】
# 超参数(Hyperparameters)是在训练前设置的参数，不是通过训练学习的
# 它们控制模型的学习过程和容量

BATCH_SIZE = 128        # 批次大小：每次训练使用的样本数
                        # 较大batch: 训练快，梯度估计准，但需要更多显存
                        # 较小batch: 训练慢，有更多噪声帮助逃离局部最优

EPOCHS = 10             # 训练轮数：整个数据集被遍历的次数
                        # 太少: 欠拟合；太多: 过拟合

LEARNING_RATE = 0.001   # 学习率：参数更新的步长
                        # 太大: 可能发散；太小: 收敛慢

HIDDEN_UNITS = [512, 256, 128]  # 隐藏层结构：3层，神经元数递减
                                # 这种"金字塔"结构有助于特征提取

# 设备选择：优先使用GPU，其次MPS(Apple)，最后CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 
                     'mps' if torch.backends.mps.is_available() else 'cpu')


# ==================== 模型定义 ====================
class FNNModel(nn.Module):
    """
    前馈神经网络模型（多层感知机 MLP）
    
    【网络架构】
    输入(784维) → Linear(784→512) → ReLU → BN → Dropout →
                Linear(512→256) → ReLU → BN → Dropout →
                Linear(256→128) → ReLU → BN → Dropout →
                Linear(128→10) → 输出
    
    【为什么这样设计？】
    1. 逐层减少神经元数量：从低级特征到高级抽象
    2. 每层后加ReLU：引入非线性
    3. BatchNorm：稳定训练，加速收敛
    4. Dropout：随机丢弃神经元，防止过拟合
    """
    
    def __init__(self, input_size=784, num_classes=10, hidden_units=None):
        """
        初始化网络结构
        
        参数:
            input_size: 输入维度，MNIST图像28×28=784
            num_classes: 分类类别数，数字0-9共10类
            hidden_units: 隐藏层神经元列表，如[512, 256, 128]
        """
        super(FNNModel, self).__init__()
        
        if hidden_units is None:
            hidden_units = [512, 256, 128]
        
        layers = []  # 用于存储网络各层的列表
        prev_size = input_size  # 前一层的神经元数量
        
        # ========== 构建隐藏层 ==========
        # 【循环构建网络的原理】
        # 使用循环可以灵活地创建任意深度的网络
        # 每一层都包含：Linear → ReLU → BatchNorm → Dropout
        for i, hidden_size in enumerate(hidden_units):
            # 线性变换层: y = xW + b
            # 输入维度: prev_size, 输出维度: hidden_size
            # 参数量: prev_size × hidden_size (weights) + hidden_size (biases)
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # ReLU激活函数: f(x) = max(0, x)
            # 【为什么用ReLU？】
            # 1. 计算简单，只需阈值判断
            # 2. 缓解梯度消失问题（正区间梯度为1）
            # 3. 产生稀疏激活，提高表达能力
            layers.append(nn.ReLU())
            
            # Batch Normalization（批归一化）
            # 【BN的作用】
            # 1. 将每层的输入归一化为均值0、方差1的分布
            # 2. 允许使用更大的学习率，加速训练
            # 3. 有轻微的正则化效果
            # 4. 减少对初始化的敏感度
            # 公式: BN(x) = γ * (x - μ) / √(σ² + ε) + β
            # 其中γ和β是可学习参数，μ和σ是batch的统计量
            layers.append(nn.BatchNorm1d(hidden_size))
            
            # Dropout正则化
            # 【Dropout原理】
            # 训练时以概率p随机"丢弃"（置零）神经元
            # 作用：
            # 1. 防止神经元之间的共适应(co-adaptation)
            # 2. 相当于训练多个子网络的集成
            # 3. 有效防止过拟合
            # 测试时会关闭Dropout，所有神经元都参与
            layers.append(nn.Dropout(0.3))  # 30%的丢弃率
            
            prev_size = hidden_size  # 更新为当前层的大小，供下一层使用
        
        # ========== 输出层 ==========
        # 最后一层不需要激活函数、BN和Dropout
        # 直接输出logits（未归一化的对数概率）
        # CrossEntropyLoss会在内部应用softmax
        layers.append(nn.Linear(prev_size, num_classes))
        
        # 将所有层组合成顺序容器
        # nn.Sequential会按顺序执行所有层
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入张量，形状为 (batch_size, 1, 28, 28)
               - batch_size: 批次中的样本数
               - 1: 通道数（灰度图）
               - 28×28: 图像尺寸
        
        返回:
            输出张量，形状为 (batch_size, 10)
            每个样本对应10个类别的得分（logits）
        """
        # 将2D图像展平为1D向量
        # x.view(): 重塑张量形状
        # x.size(0): 获取batch_size
        # -1: 自动计算该维度大小
        # 例如: (128, 1, 28, 28) → (128, 784)
        x = x.view(x.size(0), -1)
        
        # 通过网络层进行前向传播
        # Sequential会自动按顺序执行所有层
        return self.network(x)


# ==================== 训练函数 ====================
def train(model, train_loader, optimizer, criterion, epoch):
    """
    训练一个epoch
    
    【训练流程详解】
    一个epoch = 遍历完整个训练集一次
    由于使用mini-batch，实际上是多次迭代完成
    
    参数:
        model: 神经网络模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        epoch: 当前epoch编号
    
    返回:
        avg_loss: 平均损失
        accuracy: 准确率（百分比）
    """
    # 设置为训练模式
    # 【train() vs eval()】
    # train(): 启用Dropout和BatchNorm的训练行为
    # eval(): 禁用Dropout，BatchNorm使用运行统计量
    model.train()
    
    total_loss = 0    # 累计损失
    correct = 0       # 正确预测数
    total = 0         # 总样本数
    
    # 遍历训练数据
    # enumerate返回索引和数据
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据移动到指定设备（GPU/CPU）
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        # ===== 步骤1: 清空梯度 =====
        # 【为什么要清空梯度？】
        # PyTorch默认会累积梯度
        # 如果不清空，新计算的梯度会与之前的累加
        # 这会导致错误的参数更新
        optimizer.zero_grad()
        
        # ===== 步骤2: 前向传播 =====
        # 将输入数据传入模型，得到预测输出
        # output形状: (batch_size, 10)
        output = model(data)
        
        # ===== 步骤3: 计算损失 =====
        # 比较预测值和真实标签
        # CrossEntropyLoss结合了LogSoftmax和NLLLoss
        # 公式: Loss = -log(exp(output[target]) / Σexp(output))
        loss = criterion(output, target)
        
        # ===== 步骤4: 反向传播 =====
        # 计算损失对所有参数的梯度
        # 利用链式法则，从输出层向输入层传播
        loss.backward()
        
        # ===== 步骤5: 更新参数 =====
        # 根据梯度和学习率更新权重
        # Adam更新规则: w = w - lr * m / (√v + ε)
        # 其中m是一阶矩估计，v是二阶矩估计
        optimizer.step()
        
        # 统计指标
        total_loss += loss.item()  # item()将单元素张量转为Python数值
        pred = output.argmax(dim=1)  # 取最大值的索引作为预测类别
        correct += pred.eq(target).sum().item()  # 计算正确预测数
        total += len(target)  # 累加样本总数
        
        # 每100个batch打印一次进度
        if (batch_idx + 1) % 100 == 0:
            print(f'  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%')
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# ==================== 测试函数 ====================
def test(model, test_loader, criterion):
    """
    测试模型性能
    
    【测试与训练的区别】
    1. 不计算梯度（节省内存和计算）
    2. 不更新参数
    3. 使用eval()模式（关闭Dropout等）
    4. 用于评估模型泛化能力
    
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        criterion: 损失函数
    
    返回:
        avg_loss: 平均测试损失
        accuracy: 测试准确率
    """
    # 设置为评估模式
    # 【eval()的作用】
    # 1. Dropout层：停止随机丢弃，所有神经元都参与
    # 2. BatchNorm层：使用训练时统计的均值和方差，而非当前batch的
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    # 【torch.no_grad()上下文管理器】
    # 在此上下文中的所有操作都不会记录梯度
    # 好处：
    # 1. 节省内存（不需要存储中间变量用于反向传播）
    # 2. 加快计算速度
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            total_loss += criterion(output, target).item()
            
            # 统计准确率
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# ==================== 可视化函数 ====================
def plot_training_curve(train_losses, train_accs, test_losses, test_accs, save_path='fnn_training_curve.png'):
    """
    绘制训练曲线
    
    【为什么要可视化训练曲线？】
    1. 监控训练过程，及时发现问题
    2. 判断是否过拟合或欠拟合
    3. 决定何时停止训练（早停）
    
    【如何解读曲线？】
    - 正常: train和test曲线都下降且接近
    - 过拟合: train loss继续下降，但test loss开始上升
    - 欠拟合: 两条曲线都很高且不再下降
    
    参数:
        train_losses: 训练损失列表
        train_accs: 训练准确率列表
        test_losses: 测试损失列表
        test_accs: 测试准确率列表
        save_path: 保存路径
    """
    # 创建1行2列的子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== 左图：损失曲线 =====
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(test_losses, label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('FNN Training Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)  # 添加网格，透明度0.3
    
    # ===== 右图：准确率曲线 =====
    ax2.plot(train_accs, label='Train Accuracy', linewidth=2)
    ax2.plot(test_accs, label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('FNN Training Accuracy', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 自动调整布局，避免重叠
    plt.tight_layout()
    # 保存图片，dpi=150保证清晰度
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'✓ 训练曲线已保存为 {save_path}')
    plt.close()  # 关闭图形，释放内存


def plot_predictions(model, test_loader, num_samples=16, save_path='fnn_predictions.png'):
    """
    可视化预测结果
    
    【可视化的意义】
    1. 直观了解模型的预测能力
    2. 发现模型的错误模式
    3. 展示给他人看
    
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        num_samples: 显示的样本数
        save_path: 保存路径
    """
    model.eval()
    
    # 从测试集中获取一批数据
    data_iter = iter(test_loader)  # 创建迭代器
    images, labels = next(data_iter)  # 获取下一个batch
    images, labels = images[:num_samples], labels[:num_samples]  # 取前num_samples个
    
    # 进行预测
    with torch.no_grad():
        images = images.to(DEVICE)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()  # 转回CPU以便绘图
    
    # 创建4×4的子图网格
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    # 遍历每个样本并绘制
    for i, ax in enumerate(axes.flat):
        # 显示图像
        # squeeze()移除大小为1的维度
        # cmap='gray'使用灰度 colormap
        ax.imshow(images[i].cpu().squeeze(), cmap='gray')
        
        # 根据预测是否正确设置颜色
        color = 'green' if predictions[i] == labels[i] else 'red'
        
        # 显示预测和真实标签
        ax.set_title(f'Pred: {predictions[i]} | True: {labels[i]}', 
                    color=color, fontsize=10)
        ax.axis('off')  # 关闭坐标轴
    
    # 添加总标题
    plt.suptitle('FNN Predictions (Green=Correct, Red=Wrong)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'✓ 预测结果已保存为 {save_path}')
    plt.close()


# ==================== 主函数 ====================
def main():
    """
    主函数 - 协调整个训练流程
    
    【完整训练流程】
    1. 环境准备（设备检测）
    2. 数据准备（加载和预处理）
    3. 模型构建
    4. 训练配置（优化器、损失函数）
    5. 训练循环
    6. 模型保存
    7. 结果可视化
    """
    print("=" * 60)
    print("FNN (Feedforward Neural Network) - MNIST 手写数字识别")
    print("=" * 60)
    
    # ===== 第1步：设备检测 =====
    print(f"✓ 使用设备: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # ===== 第2步：数据加载 =====
    print("\n📥 正在加载MNIST数据集...")
    
    # 【数据预处理管道】
    # transforms.Compose将多个变换组合成一个管道
    transform = transforms.Compose([
        # ToTensor: 将PIL图像或NumPy数组转换为PyTorch张量
        # - 值范围从[0, 255]变为[0.0, 1.0]
        # - 维度从(H, W, C)变为(C, H, W)
        transforms.ToTensor(),
        
        # Normalize: 标准化
        # 公式: output = (input - mean) / std
        # MNIST的均值和标准差是预先计算好的
        # 标准化的好处：
        # 1. 加速收敛
        # 2. 提高数值稳定性
        # 3. 使不同特征的尺度一致
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 下载并加载训练集
    # root: 数据存储路径
    # train=True: 加载训练集
    # download=True: 如果不存在则自动下载
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # 加载测试集
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # 【DataLoader的作用】
    # 1. 批量加载数据（batch）
    # 2. 打乱数据顺序（shuffle）
    # 3. 多进程加载（num_workers）
    # 4. 自动collate（将多个样本堆叠成batch）
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"✓ 训练集大小: {len(train_dataset)}")  # 60000
    print(f"✓ 测试集大小: {len(test_dataset)}")   # 10000
    
    # ===== 第3步：模型初始化 =====
    print("\n📊 模型结构:")
    model = FNNModel(input_size=784, num_classes=10, hidden_units=HIDDEN_UNITS).to(DEVICE)
    print(model)  # 打印模型结构
    
    # 统计参数量
    # numel(): 返回张量中元素的总数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # ===== 第4步：配置优化器和损失函数 =====
    
    # 【Adam优化器】
    # Adam (Adaptive Moment Estimation) 结合了：
    # 1. Momentum: 利用历史梯度的指数加权平均
    # 2. RMSprop: 自适应调整每个参数的学习率
    # 优点：
    # - 收敛快
    # - 对学习率不太敏感
    # - 适合大多数任务
    # weight_decay: L2正则化系数，防止过拟合
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # 【学习率调度器】
    # StepLR: 每隔step_size个epoch，学习率乘以gamma
    # 例如: step_size=5, gamma=0.5
    # - Epoch 1-5: lr = 0.001
    # - Epoch 6-10: lr = 0.0005
    # 作用：后期减小学习率，精细调整参数
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 【交叉熵损失】
    # 适用于多分类任务
    # 内部包含Softmax + NLLLoss
    criterion = nn.CrossEntropyLoss()
    
    # ===== 第5步：训练循环 =====
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    # 用于记录训练历史
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    # 逐个epoch训练
    for epoch in range(1, EPOCHS + 1):
        # 训练一个epoch
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)
        
        # 在测试集上评估
        test_loss, test_acc = test(model, test_loader, criterion)
        
        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 打印epoch总结
        print(f"\n📈 Epoch {epoch}/{EPOCHS} 总结:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print("-" * 60)
        
        # 更新学习率
        scheduler.step()
    
    # ===== 第6步：保存模型 =====
    # state_dict(): 返回模型的所有参数（weights和biases）
    # 只保存参数，不保存模型结构
    # 加载时需要先创建模型实例，再load_state_dict
    torch.save(model.state_dict(), 'fnn_mnist.pth')
    print("\n✓ 模型已保存为 fnn_mnist.pth")
    
    # ===== 第7步：可视化 =====
    print("\n📊 生成可视化结果...")
    plot_training_curve(train_losses, train_accs, test_losses, test_accs)
    plot_predictions(model, test_loader)
    
    # ===== 最终结果 =====
    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print("=" * 60)
    print(f"\n最终测试结果:")
    print(f"  测试准确率: {test_accs[-1]:.2f}%")
    print(f"  测试损失: {test_losses[-1]:.4f}")
    print(f"\n生成的文件:")
    print(f"  - fnn_mnist.pth (模型权重)")
    print(f"  - fnn_training_curve.png (训练曲线)")
    print(f"  - fnn_predictions.png (预测可视化)")


if __name__ == '__main__':
    main()
