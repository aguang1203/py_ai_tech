"""
CNN (Convolutional Neural Network) - MNIST 手写数字识别

【学习目标】
本demo将帮助你深入理解卷积神经网络(CNN)的核心原理：
1. 卷积操作(Convolution)的数学原理和物理意义
2. 池化(Pooling)的作用和类型
3. CNN架构设计原则（卷积块+全连接层）
4. 数据增强(Data Augmentation)技术
5. 卷积核可视化方法

【CNN核心概念】
- 卷积: 用小的卷积核在图像上滑动，提取局部特征
- 特征图(Feature Map): 卷积后的输出，表示检测到的特征
- 感受野(Receptive Field): 神经元能"看到"的输入区域大小
- 参数共享: 同一个卷积核在整个图像上使用，大幅减少参数量
- 平移不变性: 无论物体在图像的哪个位置都能检测到
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
BATCH_SIZE = 128        # 批次大小
EPOCHS = 10             # 训练轮数
LEARNING_RATE = 0.001   # 学习率

# 设备选择：GPU > MPS(Apple) > CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 
                     'mps' if torch.backends.mps.is_available() else 'cpu')


# ==================== 模型定义 ====================
class CNNModel(nn.Module):
    """
    改进的卷积神经网络模型
    
    【网络架构设计】
    输入(1×28×28) → ConvBlock1 → ConvBlock2 → Flatten → FC Layers → 输出(10)
    
    【为什么这样设计？】
    1. 使用两个卷积块，逐步提取高级特征
    2. 每个卷积块包含2个卷积层，增强特征表达能力
    3. BatchNorm稳定训练，加速收敛
    4. MaxPool降采样，减少计算量，提高鲁棒性
    5. Dropout防止过拟合
    6. 最后用全连接层进行分类
    
    【参数计算示例】
    第一个卷积层: Conv2d(1, 32, 3×3)
    - 输入通道: 1 (灰度图)
    - 输出通道: 32 (32个不同的卷积核)
    - 卷积核大小: 3×3
    - 参数量: 1×32×3×3 + 32(bias) = 320
    
    第二个卷积层: Conv2d(32, 32, 3×3)
    - 输入通道: 32
    - 输出通道: 32
    - 参数量: 32×32×3×3 + 32 = 9,248
    """
    
    def __init__(self, num_classes=10):
        """
        初始化CNN模型
        
        参数:
            num_classes: 分类类别数，MNIST为10（数字0-9）
        """
        super(CNNModel, self).__init__()
        
        # ========== 第一个卷积块 ==========
        # 【卷积块的作用】
        # 提取低级特征：边缘、角点、纹理等
        # 输入: (batch, 1, 28, 28)
        # 输出: (batch, 32, 14, 14) 经过MaxPool后尺寸减半
        self.conv_block1 = nn.Sequential(
            # 第一层卷积
            # 【Conv2d参数详解】
            # in_channels=1: 输入通道数（灰度图为1）
            # out_channels=32: 输出通道数（32个卷积核，提取32种特征）
            # kernel_size=3: 卷积核大小3×3
            # padding=1: 填充1像素，保持输出尺寸不变
            # 
            # 【输出尺寸计算】
            # output_size = (input_size + 2*padding - kernel_size) / stride + 1
            # = (28 + 2*1 - 3) / 1 + 1 = 28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            
            # Batch Normalization（批归一化）
            # 【BN在CNN中的作用】
            # 1. 对每个通道的特征图进行归一化
            # 2. 加速训练，允许更大的学习率
            # 3. 有正则化效果，减少对Dropout的依赖
            nn.BatchNorm2d(32),
            
            # ReLU激活函数
            # inplace=True: 原地操作，节省内存
            nn.ReLU(inplace=True),
            
            # 第二层卷积
            # 继续提取更复杂的特征组合
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 最大池化
            # 【MaxPool的作用】
            # 1. 降采样：减少特征图尺寸，降低计算量
            # 2. 扩大感受野：让后续层能看到更大的区域
            # 3. 提供平移不变性：小幅度的位置变化不影响结果
            # 4. 保留最显著的特征：取最大值
            #
            # 输出尺寸: (28 + 0 - 2) / 2 + 1 = 14
            nn.MaxPool2d(2, 2),
            
            # 2D Dropout
            # 【Dropout2d vs Dropout】
            # Dropout2d: 随机丢弃整个通道（feature map）
            # Dropout: 随机丢弃单个元素
            # Dropout2d更适合CNN，保持空间结构
            nn.Dropout2d(0.25)  # 25%的通道被随机丢弃
        )
        
        # ========== 第二个卷积块 ==========
        # 【深层卷积的作用】
        # 提取高级特征：形状、部件、对象等
        # 输入: (batch, 32, 14, 14)
        # 输出: (batch, 64, 7, 7) 经过MaxPool后尺寸再次减半
        self.conv_block2 = nn.Sequential(
            # 第三层卷积
            # 增加通道数到64，提取更多种类的特征
            # 输入: 32通道, 输出: 64通道
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第四层卷积
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第二次池化
            # 输出尺寸: (14 + 0 - 2) / 2 + 1 = 7
            nn.MaxPool2d(2, 2),
            
            # Dropout
            nn.Dropout2d(0.25)
        )
        
        # ========== 全连接层 ==========
        # 【从卷积特征到分类】
        # 卷积层提取的特征需要映射到类别
        # 输入: 64×7×7 = 3136维特征向量
        # 输出: 10维（10个类别的得分）
        self.fc_layers = nn.Sequential(
            # 第一层全连接
            # 将3136维特征压缩到256维
            # 这是一个瓶颈层，迫使网络学习紧凑的特征表示
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            # Dropout比例更高（0.5）
            # 全连接层参数多，容易过拟合，需要更强的正则化
            nn.Dropout(0.5),
            
            # 输出层
            # 映射到10个类别
            # 不使用Softmax，因为CrossEntropyLoss内部会处理
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, 1, 28, 28)
               - batch_size: 批次大小
               - 1: 通道数（灰度图）
               - 28×28: 图像尺寸
        
        返回:
            输出张量，形状为 (batch_size, 10)
            每个样本对应10个类别的得分（logits）
        
        【数据流动过程】
        输入: (batch, 1, 28, 28)
          ↓ conv_block1
        (batch, 32, 14, 14)  # 提取低级特征，尺寸减半
          ↓ conv_block2
        (batch, 64, 7, 7)    # 提取高级特征，尺寸再减半
          ↓ flatten
        (batch, 3136)        # 展平为一维向量
          ↓ fc_layers
        (batch, 10)          # 分类得分
        """
        # 通过第一个卷积块
        x = self.conv_block1(x)
        
        # 通过第二个卷积块
        x = self.conv_block2(x)
        
        # 展平操作
        # 将多维特征图展平为一维向量
        # x.size(0): batch_size
        # -1: 自动计算，这里是 64×7×7 = 3136
        x = x.view(x.size(0), -1)
        
        # 通过全连接层
        x = self.fc_layers(x)
        
        return x


# ==================== 训练函数 ====================
def train(model, train_loader, optimizer, criterion, epoch):
    """
    训练一个epoch
    
    【CNN训练的特殊之处】
    1. 输入是4D张量: (batch, channel, height, width)
    2. 卷积操作自动利用GPU并行加速
    3. 反向传播时，梯度会通过卷积核传播
    
    参数:
        model: CNN模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        epoch: 当前epoch编号
    
    返回:
        avg_loss: 平均损失
        accuracy: 准确率（百分比）
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 数据移动到指定设备
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        # 清空梯度
        optimizer.zero_grad()
        
        # 前向传播
        # data形状: (batch_size, 1, 28, 28)
        output = model(data)  # 输出形状: (batch_size, 10)
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 统计指标
        total_loss += loss.item()
        pred = output.argmax(dim=1)  # 取最大值的索引
        correct += pred.eq(target).sum().item()
        total += len(target)
        
        # 打印进度
        if (batch_idx + 1) % 100 == 0:
            print(f'  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# ==================== 测试函数 ====================
def test(model, test_loader, criterion):
    """
    测试模型性能
    
    参数:
        model: 训练好的CNN模型
        test_loader: 测试数据加载器
        criterion: 损失函数
    
    返回:
        avg_loss: 平均测试损失
        accuracy: 测试准确率
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不计算梯度，节省内存
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# ==================== 可视化函数 ====================
def plot_training_curve(train_losses, train_accs, test_losses, test_accs, save_path='cnn_training_curve.png'):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(test_losses, label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('CNN Training Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Train Accuracy', linewidth=2)
    ax2.plot(test_accs, label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('CNN Training Accuracy', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'✓ 训练曲线已保存为 {save_path}')
    plt.close()


def plot_predictions(model, test_loader, num_samples=16, save_path='cnn_predictions.png'):
    """可视化预测结果"""
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images[:num_samples], labels[:num_samples]
    
    with torch.no_grad():
        images = images.to(DEVICE)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].cpu().squeeze(), cmap='gray')
        color = 'green' if predictions[i] == labels[i] else 'red'
        ax.set_title(f'Pred: {predictions[i]} | True: {labels[i]}', 
                    color=color, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('CNN Predictions (Green=Correct, Red=Wrong)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'✓ 预测结果已保存为 {save_path}')
    plt.close()


def plot_conv_filters(model, save_path='cnn_filters.png'):
    """
    可视化卷积核（滤波器）
    
    【为什么要可视化卷积核？】
    1. 理解CNN学到了什么特征
    2. 调试模型，检查是否正常训练
    3. 直观展示卷积的工作原理
    
    【如何解读？】
    - 第一层卷积核通常学习到边缘、角点等低级特征
    - 深层卷积核学习到更复杂的模式
    - 如果卷积核是全零或噪声，说明训练有问题
    """
    # 获取第一个卷积层的权重
    # conv_block1[0] 是第一个Conv2d层
    conv1_weight = model.conv_block1[0].weight.detach().cpu()
    
    # 卷积核形状: (out_channels, in_channels, kernel_h, kernel_w)
    # 对于我们的模型: (32, 1, 3, 3)
    print(f"卷积核形状: {conv1_weight.shape}")
    
    # 可视化所有32个卷积核
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    
    for i, ax in enumerate(axes.flat):
        if i < conv1_weight.shape[0]:  # 只可视化存在的卷积核
            # 获取第i个卷积核
            # 形状: (1, 3, 3)，因为是灰度图，只有1个输入通道
            kernel = conv1_weight[i, 0]
            
            # 显示卷积核
            ax.imshow(kernel.numpy(), cmap='gray')
            ax.set_title(f'Filter {i+1}', fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle('CNN Conv1 Filters (3x3 Kernels)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'✓ 卷积核可视化已保存为 {save_path}')
    plt.close()


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("CNN (Convolutional Neural Network) - MNIST 手写数字识别")
    print("=" * 60)
    
    # 设备检测
    print(f"✓ 使用设备: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # 数据加载
    print("\n📥 正在加载MNIST数据集...")
    
    # 【数据增强管道】
    # 训练集使用数据增强，提高模型泛化能力
    # 测试集只做标准化，不做增强
    train_transform = transforms.Compose([
        # 数据增强：随机旋转±10度
        # 【为什么需要数据增强？】
        # 1. 人工扩充训练数据
        # 2. 让模型学会旋转不变性
        # 3. 防止过拟合
        # 4. 模拟真实场景中的变化
        transforms.RandomRotation(10),
        
        # 转换为张量并标准化
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 测试集变换（无数据增强）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"✓ 训练集大小: {len(train_dataset)}")
    print(f"✓ 测试集大小: {len(test_dataset)}")
    
    # 模型初始化
    print("\n📊 模型结构:")
    model = CNNModel(num_classes=10).to(DEVICE)
    print(model)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 【参数量对比】
    # CNN: ~400K参数
    # FNN: ~500K参数
    # CNN参数更少但性能更好，因为：
    # 1. 参数共享：同一个卷积核在整个图像上使用
    # 2. 局部连接：每个神经元只连接局部区域
    # 3. 稀疏连接：不是全连接
    
    # 优化器和损失函数
    # Adam优化器，带权重衰减（L2正则化）
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Cosine Annealing学习率调度
    # 【Cosine Annealing的优势】
    # 1. 平滑地降低学习率
    # 2. 避免学习率突变
    # 3. 有助于跳出局部最优
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    train_losses, train_accs = [], []
    test_losses, test_accs = []
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, test_loader, criterion)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"\n📈 Epoch {epoch}/{EPOCHS} 总结:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print("-" * 60)
        
        # 更新学习率
        scheduler.step()
    
    # 保存模型
    torch.save(model.state_dict(), 'cnn_mnist.pth')
    print("\n✓ 模型已保存为 cnn_mnist.pth")
    
    # 可视化
    print("\n📊 生成可视化结果...")
    plot_training_curve(train_losses, train_accs, test_losses, test_accs)
    plot_predictions(model, test_loader)
    plot_conv_filters(model)  # CNN特有的卷积核可视化
    
    # 最终结果
    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print("=" * 60)
    print(f"\n最终测试结果:")
    print(f"  测试准确率: {test_accs[-1]:.2f}%")
    print(f"  测试损失: {test_losses[-1]:.4f}")
    print(f"\n生成的文件:")
    print(f"  - cnn_mnist.pth (模型权重)")
    print(f"  - cnn_training_curve.png (训练曲线)")
    print(f"  - cnn_predictions.png (预测可视化)")
    print(f"  - cnn_filters.png (卷积核可视化)")


if __name__ == '__main__':
    main()
