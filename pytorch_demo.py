"""
PyTorch Demo - 使用CNN进行MNIST手写数字识别
这个demo展示了PyTorch的基本用法:
1. 数据加载和预处理
2. 构建卷积神经网络
3. 模型训练
4. 模型评估
5. 可视化结果
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ==================== 1. 设备配置 ====================
def get_device():
    """获取训练设备 (CUDA/MPS/CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ 使用 NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ 使用 Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("⚠ 使用 CPU")
    return device


# ==================== 2. 数据准备 ====================
def prepare_data(batch_size=64):
    """准备MNIST数据集"""
    # 定义数据预处理步骤
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
    ])
    
    # 下载并加载训练集和测试集
    print("\n📥 正在下载MNIST数据集...")
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1000, 
        shuffle=False
    )
    
    print(f"✓ 训练集大小: {len(train_dataset)}")
    print(f"✓ 测试集大小: {len(test_dataset)}")
    
    return train_loader, test_loader


# ==================== 3. 构建模型 ====================
class CNNModel(nn.Module):
    """卷积神经网络模型"""
    
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输入通道1, 输出32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Dropout2d(0.25)
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  # 10个类别 (0-9)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x


# ==================== 4. 训练函数 ====================
def train_model(model, train_loader, optimizer, criterion, epoch, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据移动到设备
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计信息
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 打印进度
        if (batch_idx + 1) % 100 == 0:
            print(f'  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Acc: {100.*correct/total:.2f}%')
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# ==================== 5. 测试函数 ====================
def test_model(model, test_loader, criterion, device):
    """在测试集上评估模型"""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return avg_loss, accuracy


# ==================== 6. 可视化 ====================
def plot_training_curve(train_losses, train_accs, test_losses, test_accs):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss曲线
    ax1.plot(train_losses, label='Training Loss', marker='o')
    ax1.plot(test_losses, label='Test Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy曲线
    ax2.plot(train_accs, label='Training Acc', marker='o')
    ax2.plot(test_accs, label='Test Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
    print("\n✓ 训练曲线已保存为 training_curve.png")
    plt.close()


def visualize_predictions(model, test_loader, device, num_samples=10):
    """可视化预测结果"""
    model.eval()
    
    # 获取一批测试数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # 进行预测
    with torch.no_grad():
        images_subset = images[:num_samples].to(device)
        outputs = model(images_subset)
        _, predictions = outputs.max(1)
    
    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # 反归一化
        image = images[i].squeeze().numpy()
        image = image * 0.3081 + 0.1307  # 反归一化
        
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'Pred: {predictions[i].item()} | True: {labels[i].item()}',
                         fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('MNIST Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    print("✓ 预测结果已保存为 predictions.png")
    plt.close()


# ==================== 7. 主函数 ====================
def main():
    """主训练流程"""
    print("="*60)
    print("PyTorch CNN - MNIST 手写数字识别")
    print("="*60)
    
    # 配置参数
    EPOCHS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    
    # 获取设备
    device = get_device()
    
    # 准备数据
    train_loader, test_loader = prepare_data(BATCH_SIZE)
    
    # 创建模型
    model = CNNModel().to(device)
    print(f"\n📊 模型结构:\n{model}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    
    # 训练记录
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # 开始训练
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60)
    
    for epoch in range(1, EPOCHS + 1):
        # 训练
        train_loss, train_acc = train_model(
            model, train_loader, optimizer, criterion, epoch, device
        )
        
        # 测试
        test_loss, test_acc = test_model(
            model, test_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 打印epoch总结
        print(f'\n📈 Epoch {epoch}/{EPOCHS} 总结:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%')
        print('-'*60)
    
    # 保存模型
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("\n✓ 模型已保存为 mnist_cnn.pth")
    
    # 可视化
    print("\n📊 生成可视化结果...")
    plot_training_curve(train_losses, train_accs, test_losses, test_accs)
    visualize_predictions(model, test_loader, device)
    
    print("\n" + "="*60)
    print("✅ 训练完成!")
    print("="*60)
    print(f"\n最终测试结果:")
    print(f"  测试准确率: {test_accs[-1]:.2f}%")
    print(f"  测试损失: {test_losses[-1]:.4f}")
    print(f"\n生成的文件:")
    print(f"  - mnist_cnn.pth (模型权重)")
    print(f"  - training_curve.png (训练曲线)")
    print(f"  - predictions.png (预测可视化)")


if __name__ == '__main__':
    main()
