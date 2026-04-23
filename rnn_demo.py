"""
RNN (Recurrent Neural Network) - MNIST 手写数字识别

【学习目标】
本demo将帮助你深入理解循环神经网络(RNN)和LSTM的核心原理：
1. RNN处理序列数据的基本思想
2. LSTM如何解决长期依赖问题
3. 如何将图像视为序列数据
4. 门控机制（遗忘门、输入门、输出门）的工作原理
5. 梯度裁剪防止梯度爆炸

【RNN核心概念】
- 序列建模: 数据有先后顺序，当前输出依赖于之前的输入
- 隐藏状态(Hidden State): 网络的"记忆"，携带历史信息
- 时间步(Time Step): 序列中的每个位置
- 长期依赖: 早期信息对后期预测的影响
- 梯度消失/爆炸: RNN训练中的常见问题

【为什么用RNN处理图像？】
虽然CNN更适合图像，但将MNIST视为序列可以：
1. 学习RNN处理序列数据的思想
2. 理解LSTM的门控机制
3. 为NLP等真正的序列任务打基础
4. 展示不同架构的灵活性
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

# LSTM特定参数
HIDDEN_SIZE = 128       # LSTM隐藏单元数量
                        # 越大：表达能力越强，但更容易过拟合
                        # 越小：泛化能力更好，但可能欠拟合

NUM_LAYERS = 2          # LSTM层数（堆叠深度）
                        # 多层LSTM可以学习更抽象的特征
                        # 第1层：低级时序模式
                        # 第2层：高级时序模式

# 【如何将图像转为序列？】
# MNIST图像尺寸: 28×28
# 我们将每一行视为一个时间步
SEQUENCE_LENGTH = 28    # 序列长度 = 图像行数 = 28
INPUT_SIZE = 28         # 每个时间步的输入维度 = 图像列数 = 28

# 设备选择
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 
                     'mps' if torch.backends.mps.is_available() else 'cpu')


# ==================== 模型定义 ====================
class RNNModel(nn.Module):
    """
    基于LSTM的循环神经网络模型
    
    【网络架构】
    输入序列(28步×28维) → LSTM(2层) → 最后时间步输出 → FC(128→64→10) → 输出
    
    【为什么这样设计？】
    1. 使用LSTM而非普通RNN：解决梯度消失问题
    2. 2层LSTM：学习多层次的时序特征
    3. dropout=0.3：防止过拟合
    4. 只取最后时间步：整个序列的信息已聚合到隐藏状态
    5. FC层加Dropout：进一步正则化
    
    【参数量计算】
    LSTM参数量 = 4 × [(input_size + hidden_size) × hidden_size + hidden_size]
               = 4 × [(28 + 128) × 128 + 128]
               = 4 × [156 × 128 + 128]
               = 4 × 20,096
               = 80,384 （每层）
    
    2层LSTM总参数 ≈ 160K
    FC层参数 = 128×64 + 64 + 64×10 + 10 = 8,586
    总计 ≈ 170K
    """
    
    def __init__(self, input_size=28, hidden_size=128, num_layers=2, num_classes=10):
        """
        初始化LSTM模型
        
        参数:
            input_size: 每个时间步的输入维度（28，图像的列数）
            hidden_size: LSTM隐藏单元数量（128）
            num_layers: LSTM层数（2）
            num_classes: 分类类别数（10）
        """
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # ========== LSTM层 ==========
        # 【LSTM vs 普通RNN】
        # 普通RNN: h_t = tanh(W_h @ h_{t-1} + W_x @ x_t)
        # 问题：梯度容易消失或爆炸，无法捕捉长期依赖
        #
        # LSTM通过3个门控机制解决这个问题：
        # 1. 遗忘门(Forget Gate): 决定丢弃什么信息
        # 2. 输入门(Input Gate): 决定存储什么新信息
        # 3. 输出门(Output Gate): 决定输出什么信息
        #
        # 还有细胞状态(Cell State)作为"信息高速公路"
        self.lstm = nn.LSTM(
            input_size=input_size,      # 输入维度: 28
            hidden_size=hidden_size,    # 隐藏层维度: 128
            num_layers=num_layers,      # 层数: 2
            batch_first=True,           # 输入格式: (batch, seq, feature)
                                        # 如果False: (seq, batch, feature)
            dropout=0.3 if num_layers > 1 else 0
            # dropout只在多层时使用
            # 在LST M层之间添加dropout，防止过拟合
        )
        
        # ========== 全连接分类层 ==========
        # 【为什么需要FC层？】
        # LSTM输出的是隐藏状态，需要映射到类别空间
        # 使用两层FC+Dropout增强泛化能力
        self.fc = nn.Sequential(
            # 第一层：128 → 64
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # 第二层：64 → 10（类别数）
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, seq_length, input_size)
               对于MNIST: (batch_size, 28, 28)
               - batch_size: 批次大小
               - seq_length: 序列长度（28行）
               - input_size: 每个时间步的维度（28列）
        
        返回:
            输出张量，形状为 (batch_size, num_classes)
            每个样本对应10个类别的得分
        
        【数据流动过程】
        输入: (batch, 28, 28)  # 28个时间步，每步28维
          ↓ LSTM
        输出: (batch, 28, 128)  # 28个时间步，每步128维隐藏状态
          ↓ 取最后时间步
        (batch, 128)  # 只取最后一个时间步的输出
          ↓ FC层
        (batch, 10)   # 分类得分
        """
        # x shape: (batch_size, seq_length, input_size)
        
        # ========== 初始化隐藏状态和细胞状态 ==========
        # 【为什么要初始化？】
        # LSTM需要初始的隐藏状态h0和细胞状态c0
        # 通常初始化为全零
        # 形状: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # ========== LSTM前向传播 ==========
        # 【LSTM的内部工作原理】
        # 对于每个时间步t：
        # 1. 遗忘门: f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)
        # 2. 输入门: i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)
        # 3. 候选细胞: C̃_t = tanh(W_C @ [h_{t-1}, x_t] + b_C)
        # 4. 更新细胞: C_t = f_t * C_{t-1} + i_t * C̃_t
        # 5. 输出门: o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)
        # 6. 隐藏状态: h_t = o_t * tanh(C_t)
        #
        # lstm_out: 所有时间步的隐藏状态 (batch, seq_len, hidden_size)
        # _: 最后的(h_n, c_n)，我们不需要
        lstm_out, _ = self.lstm(x, (h0, c0))
        # lstm_out shape: (batch_size, 28, 128)
        
        # ========== 提取最后时间步的输出 ==========
        # 【为什么只取最后时间步？】
        # LSTM的设计使得信息会在时间步之间传递
        # 最后一个时间步的隐藏状态已经包含了整个序列的信息
        # 这类似于RNN的"记忆"功能
        out = lstm_out[:, -1, :]  # 取最后一个时间步
        # out shape: (batch_size, 128)
        
        # ========== 分类 ==========
        out = self.fc(out)
        # out shape: (batch_size, 10)
        
        return out


# ==================== 训练函数 ====================
def train(model, train_loader, optimizer, criterion, epoch):
    """
    训练一个epoch
    
    【RNN训练的特殊之处】
    1. 输入是3D张量: (batch, seq_len, input_size)
    2. LSTM内部会按时间步逐步处理
    3. BPTT(Backpropagation Through Time): 反向传播通过时间
    4. 可能需要梯度裁剪防止梯度爆炸
    
    参数:
        model: LSTM模型
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
        # 【数据重塑】
        # 原始data形状: (batch_size, 1, 28, 28)  # (batch, channel, height, width)
        # 需要转换为: (batch_size, 28, 28)       # (batch, seq_len, input_size)
        # 方法：将每一行视为一个时间步
        data = data.squeeze(1)  # 移除channel维度: (batch, 28, 28)
        
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        # 清空梯度
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 【梯度裁剪】
        # 【为什么要梯度裁剪？】
        # RNN/LSTM容易出现梯度爆炸问题
        # 因为梯度要在时间步上连乘
        # 限制梯度的最大范数，防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        # 统计指标
        total_loss += loss.item()
        pred = output.argmax(dim=1)
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
        model: 训练好的LSTM模型
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
    
    with torch.no_grad():
        for data, target in test_loader:
            # 同样需要重塑数据
            data = data.squeeze(1)  # (batch, 28, 28)
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
def plot_training_curve(train_losses, train_accs, test_losses, test_accs, save_path='rnn_training_curve.png'):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(test_losses, label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('RNN Training Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Train Accuracy', linewidth=2)
    ax2.plot(test_accs, label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('RNN Training Accuracy', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'✓ 训练曲线已保存为 {save_path}')
    plt.close()


def plot_predictions(model, test_loader, num_samples=16, save_path='rnn_predictions.png'):
    """可视化预测结果"""
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images[:num_samples], labels[:num_samples]
    
    with torch.no_grad():
        # 重塑数据用于预测
        images_reshaped = images.squeeze(1).to(DEVICE)
        outputs = model(images_reshaped)
        predictions = outputs.argmax(dim=1).cpu()
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].cpu().squeeze(), cmap='gray')
        color = 'green' if predictions[i] == labels[i] else 'red'
        ax.set_title(f'Pred: {predictions[i]} | True: {labels[i]}', 
                    color=color, fontsize=10)
        ax.axis('off')
    
    plt.suptitle('RNN Predictions (Green=Correct, Red=Wrong)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'✓ 预测结果已保存为 {save_path}')
    plt.close()


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("RNN (LSTM) - MNIST 手写数字识别（序列方式）")
    print("=" * 60)
    
    # 设备检测
    print(f"✓ 使用设备: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # 数据加载
    print("\n📥 正在加载MNIST数据集...")
    
    # 【注意】
    # RNN不需要特殊的数据增强
    # 因为我们是按行处理，旋转会破坏序列结构
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"✓ 训练集大小: {len(train_dataset)}")
    print(f"✓ 测试集大小: {len(test_dataset)}")
    
    # 模型初始化
    print("\n📊 模型结构:")
    model = RNNModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=10
    ).to(DEVICE)
    print(model)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 【参数量对比】
    # RNN(LSTM): ~170K参数
    # CNN: ~400K参数
    # FNN: ~500K参数
    # RNN参数适中，但训练较慢（序列处理难以并行）
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # StepLR学习率调度
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
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
        
        scheduler.step()
    
    # 保存模型
    torch.save(model.state_dict(), 'rnn_mnist.pth')
    print("\n✓ 模型已保存为 rnn_mnist.pth")
    
    # 可视化
    print("\n📊 生成可视化结果...")
    plot_training_curve(train_losses, train_accs, test_losses, test_accs)
    plot_predictions(model, test_loader)
    
    # 最终结果
    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print("=" * 60)
    print(f"\n最终测试结果:")
    print(f"  测试准确率: {test_accs[-1]:.2f}%")
    print(f"  测试损失: {test_losses[-1]:.4f}")
    print(f"\n生成的文件:")
    print(f"  - rnn_mnist.pth (模型权重)")
    print(f"  - rnn_training_curve.png (训练曲线)")
    print(f"  - rnn_predictions.png (预测可视化)")
    print(f"\n💡 提示: RNN将图像视为序列，适合理解时序建模思想")


if __name__ == '__main__':
    main()
