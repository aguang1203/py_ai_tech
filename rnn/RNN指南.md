# RNN 循环神经网络 完全指南

---

## 目录

1. [基础知识](#1-基础知识)
2. [技术原理](#2-技术原理)
3. [四大任务类型](#3-四大任务类型)
4. [应用场景](#4-应用场景)
5. [使用说明](#5-使用说明)
6. [任务类型对比](#6-任务类型对比)
7. [常见问题与调优](#7-常见问题与调优)
8. [进阶扩展](#8-进阶扩展)

---

## 1. 基础知识

### 1.1 什么是循环神经网络 (RNN)

循环神经网络（Recurrent Neural Network，简称 RNN），是专门为处理**序列数据**设计的深度学习模型。

**核心思想**：网络在时间步之间共享参数，通过"隐藏状态"(hidden state)传递历史信息，相当于网络拥有了"记忆"。

```
输入序列           RNN展开             输出序列
[x₁, x₂, x₃] → [h₁→h₂→h₃]  →  [y₁, y₂, y₃]
                 ↑  ↑  ↑
                 x₁ x₂ x₃    每步接收输入+上一步的隐藏状态

时间步1: h₁ = f(W·x₁ + U·h₀)     h₀通常为全零
时间步2: h₂ = f(W·x₂ + U·h₁)     h₁携带了x₁的信息
时间步3: h₃ = f(W·x₃ + U·h₂)     h₂携带了x₁,x₂的信息
```

### 1.2 RNN vs CNN vs FNN 的核心区别

| 对比项 | FNN (前馈网络) | CNN (卷积网络) | RNN (循环网络) |
|--------|---------------|---------------|---------------|
| **输入** | 1维特征向量 | 2D/3D图像 | 序列(文本/时间序列/音频) |
| **参数共享** | 无 | 空间共享(卷积核) | 时间共享(时间步间共享) |
| **记忆能力** | 无 | 局部(感受野) | 有(隐藏状态传递) |
| **输出** | 固定大小 | 固定大小 | 可变长度序列 |
| **适用数据** | 表格/数值 | 图像/空间 | 文本/语音/时序 |
| **核心问题** | 无时序建模 | 无长程依赖 | 梯度消失/爆炸 |

**为什么序列数据不用FNN/CNN？**
- FNN把序列展平，丢失了顺序信息("我爱你"和"你爱我"展平后一样)
- CNN只能捕获局部模式，无法理解"远距离依赖"(句子首尾的关系)
- RNN天然处理序列：每步读入一个元素，用隐藏状态记住之前的内容

### 1.3 RNN 的关键组成

| 组件 | 作用 | 类比 |
|------|------|------|
| **隐藏状态 (h_t)** | 携带历史信息的"记忆" | 大脑的短期记忆 |
| **词嵌入 (nn.Embedding)** | 将离散词索引映射为连续向量 | 字典：查词→含义 |
| **RNN层 (nn.LSTM/GRU)** | 按时间步处理序列 | 逐字阅读理解 |
| **门控机制 (LSTM)** | 控制信息的遗忘和保留 | 大脑的注意力开关 |
| **全连接层 (nn.Linear)** | 将隐藏状态映射到输出 | 做出最终判断 |
| **pack_padded_sequence** | 压缩变长序列的padding | 跳过空白部分 |

### 1.4 训练流程

```
┌──────────────────────────────────────────────┐
│              每个 batch 重复执行               │
│                                              │
│  1. optimizer.zero_grad()  ← 清零梯度        │
│  2. outputs = model(x)     ← 前向传播        │
│  3. loss = criterion(…)    ← 计算损失        │
│  4. loss.backward()        ← BPTT(时间反向传播)│
│  5. clip_grad_norm_(…)     ← 梯度裁剪(防爆炸) │
│  6. optimizer.step()       ← 更新参数        │
│                                              │
│  ⚠️ RNN特有: BPTT + 梯度裁剪                  │
└──────────────────────────────────────────────┘
```

**BPTT (Backpropagation Through Time)**：
- 将RNN按时间步展开，就像一个很深的FNN
- 梯度需要从最后的时间步传回第一个
- 这是RNN梯度消失/爆炸的根本原因

---

## 2. 技术原理

### 2.1 原始RNN

最简单的循环神经网络：

```
h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
y_t = W_hy · h_t + b_y

参数: W_xh(输入→隐藏), W_hh(隐藏→隐藏), W_hy(隐藏→输出)
```

**问题**：梯度在时间步上连乘
```
∂L/∂h_1 = ∂L/∂h_T × ∏(t=2→T) ∂h_t/∂h_{t-1}

如果 ∂h_t/∂h_{t-1} < 1 → 连乘 → 梯度消失 (长期信息丢失)
如果 ∂h_t/∂h_{t-1} > 1 → 连乘 → 梯度爆炸 (训练不稳定)
```

### 2.2 LSTM (长短期记忆网络)

LSTM通过3个门控 + 细胞状态解决梯度问题：

```
遗忘门: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)       ← 决定丢弃什么
输入门: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)       ← 决定存储什么
候选值: C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)   ← 新候选信息
细胞状态: C_t = f_t * C_{t-1} + i_t * C̃_t         ← 更新细胞状态
输出门: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)       ← 决定输出什么
隐藏状态: h_t = o_t * tanh(C_t)                    ← 最终输出
```

**为什么LSTM能解决梯度消失？**

```
细胞状态的梯度传播:
∂C_t/∂C_{t-1} = f_t   (遗忘门直接控制)

如果 f_t ≈ 1 → 梯度几乎无损传递 (信息被保留)
如果 f_t ≈ 0 → 梯度被截断 (信息被遗忘)

关键: 遗忘门是可学习的，模型自己决定哪些信息要保留
这比原始RNN的固定连乘好得多
```

**LSTM的3个门详解**：

| 门 | 公式 | 作用 | 类比 |
|----|------|------|------|
| 遗忘门 | f_t = σ(...) | 0=完全遗忘, 1=完全保留 | 决定忘掉什么旧记忆 |
| 输入门 | i_t = σ(...) | 0=不接收, 1=完全接收 | 决定记住什么新信息 |
| 输出门 | o_t = σ(...) | 0=不输出, 1=完全输出 | 决定透露什么信息 |

### 2.3 GRU (门控循环单元)

GRU是LSTM的简化版，只有2个门：

```
重置门: r_t = σ(W_r · [h_{t-1}, x_t])   ← 控制忽略多少旧信息
更新门: z_t = σ(W_z · [h_{t-1}, x_t])   ← 控制新旧信息混合比例
候选值: h̃_t = tanh(W · [r_t * h_{t-1}, x_t])
隐藏状态: h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
```

**LSTM vs GRU**：

| 对比项 | LSTM | GRU |
|--------|------|-----|
| 门数量 | 3个(遗忘/输入/输出) | 2个(重置/更新) |
| 细胞状态 | 有(C_t独立) | 无(合并到h_t) |
| 参数量 | 较多 | 约少25% |
| 训练速度 | 较慢 | 较快约20% |
| 表达能力 | 更强 | 略弱但通常够用 |
| 推荐场景 | 需要精确控制记忆 | 追求速度/简单 |

### 2.4 双向RNN (Bidirectional RNN)

```
正向: x₁ → x₂ → x₃ → ... → x_T    从前往后读
反向: x₁ ← x₂ ← x₃ ← ... ← x_T    从后往前读

输出: y_t = f(h_t_forward, h_t_backward)  拼接双向隐藏状态

为什么需要双向？
- "不好用" → 正向读到"不好"时还不知道后面的"用"
- 双向同时拥有前后文，理解更完整
- 分类任务推荐用双向，生成任务不能用(未来信息不可知)
```

### 2.5 词嵌入 (Word Embedding)

```
原始输入: 词索引 (如 "好"=2, "差"=51)
               ↓ nn.Embedding
嵌入向量: "好" → [0.23, -0.15, 0.82, ...]  (64维)
          "差" → [-0.31, 0.42, -0.18, ...]  (64维)

为什么需要词嵌入？
- 词索引是任意整数，没有语义信息 (2和3不一定相似)
- 词嵌入让相似的词在向量空间中接近
- "好"和"棒"的嵌入会很近，"好"和"差"的嵌入会很远
- nn.Embedding 是可学习的，训练过程中自动优化

维度选择经验:
- 词汇量 < 1000: 32-64维
- 词汇量 1K-10K: 64-128维
- 词汇量 10K+: 128-300维
- 预训练词向量(GloVe/fastText): 通常300维
```

### 2.6 变长序列处理 (pack_padded_sequence)

```
问题: 不同句子长度不同，需要padding对齐
  ["好",   "产品"]  → [2,   101, 0,   0]   (padding到长度4)
  ["这个", "产品", "质量", "好"]  → [107, 101, 102, 2]

  LSTM会计算padding位置 → 浪费计算 + 污染隐藏状态

解决: pack_padded_sequence
  告诉LSTM每个序列的真实长度，跳过padding

  步骤:
  1. 按长度从长到短排序batch
  2. pack_padded_sequence(embedded, lengths)
  3. LSTM处理packed序列
  4. pad_packed_sequence解压回原始格式

  注意: 本模板的合成数据使用了固定长度，简化了这一步
        真实NLP任务中，变长序列处理是必须的
```

### 2.7 序列到序列模型 (Seq2Seq)

```
编码器-解码器架构 (Encoder-Decoder):

编码器: 输入序列 → 上下文向量(context vector)
  "这个产品很好" → LSTM → h_T (整个句子的压缩表示)

解码器: 上下文向量 → 输出序列
  h_T → LSTM → "This product is good"

应用: 机器翻译、文本摘要、对话系统

进阶: 注意力机制(Attention)
  - 问题: 固定长度的上下文向量是信息瓶颈
  - 解决: 解码每步都"看"编码器的所有隐藏状态
  - 注意力权重: 决定当前步应该关注输入的哪个部分
```

---

## 3. 四大任务类型

### 3.1 序列分类 (Sequence Classification)

**目标**：对整个输入序列判断一个类别

```
输入: 28×28 MNIST图像 → 视为28步序列 → LSTM → 最后隐藏状态 → 类别
  [行1, 行2, ..., 行28] → LSTM → [0.01, 0.95, 0.02, ...] → 预测: 数字1

输入: 评论文本 → 词嵌入 → BiLSTM → 最大池化 → 情感类别
  "产品非常好" → BiLSTM → [0.1, 0.9] → 预测: 正面
```

**输出层**：`num_classes` 个神经元，不加激活函数
**损失函数**：`CrossEntropyLoss`（内含 Softmax）
**评估指标**：准确率(Accuracy)、F1、混淆矩阵
**本模板数据**：MNIST手写数字 (70,000张 28×28，10类)

### 3.2 文本分类 (Text Classification / Sentiment Analysis)

**目标**：对文本进行分类，如情感分析、主题分类

```
输入: 文本 → 分词 → 词嵌入 → BiLSTM → 分类
  "这个 产品 质量 很 好"  →  正面 ✓
  "这个 产品 质量 很 差"  →  负面 ✗

与序列分类的区别:
  - 序列分类: 输入可以是任何序列(如图像行)
  - 文本分类: 输入是自然语言文本，需要词嵌入
  - 文本分类推荐用双向RNN(前后文都很重要)
```

**关键技术**：词嵌入 + 双向LSTM + pack_padded_sequence
**损失函数**：`CrossEntropyLoss`
**评估指标**：准确率、精确率、召回率、F1
**本模板数据**：合成中文评论 (1000条，正面/负面各500)

### 3.3 时间序列预测 (Time Series Forecasting)

**目标**：根据过去的值预测未来的值

```
输入: 过去30个时间步 → LSTM → 未来10个时间步
  [x₁, x₂, ..., x₃₀] → LSTM → [x₃₁, x₃₂, ..., x₄₀]

两种预测模式:
  单步预测: 只预测下一个值 (x₁,...,xₜ → x_{t+1})
  多步预测: 预测未来多个值 (x₁,...,xₜ → x_{t+1},...,x_{t+k}) ← 本模板使用

与分类的区别:
  - 分类: 输出是类别(离散)
  - 预测: 输出是数值(连续)
  - 损失函数不同: 分类用CrossEntropy，预测用MSE
```

**损失函数**：`MSELoss`（均方误差）
**评估指标**：MAE(平均绝对误差)、RMSE(均方根误差)
**本模板数据**：合成正弦波 (2000个窗口，30步输入→10步预测)

### 3.4 文本生成 (Text Generation)

**目标**：学习文本模式，逐字符生成新文本

```
训练阶段: 学习"给定前面的字符，预测下一个字符"
  输入: [S, 王, 伟]    → 目标: [王, 伟, E]
  模型学到: "S"后面常跟姓氏，"王"后面常跟名字...

生成阶段(自回归):
  第1步: 输入S → 预测"王" (采样)
  第2步: 输入S王 → 预测"伟" (采样)
  第3步: 输入S王伟 → 预测E (遇到结束符，停止)

温度控制:
  温度=0.5: "王伟", "李强", "张敏"  (保守，常见名字)
  温度=1.0: "王伟", "陈思怡", "赵磊" (平衡)
  温度=1.5: "黄博宇", "吴然浩", "罗鹏飞" (创意，更多样)
```

**损失函数**：`CrossEntropyLoss`（字符级分类）
**评估指标**：困惑度(Perplexity, PPL)
**本模板数据**：合成中文姓名 (500个，字符级生成)

---

## 4. 应用场景

### 4.1 序列分类应用

| 场景 | 输入 | 类别数 | 说明 |
|------|------|--------|------|
| 手写数字识别 | 28×28图像(序列) | 10 | MNIST，入门经典 |
| 语音命令识别 | 音频序列 | 数十 | "播放音乐"/"打开灯" |
| 心电信号分类 | ECG信号 | 2-5 | 正常/异常/疾病类型 |
| 活动识别 | 加速度序列 | 数个 | 走路/跑步/坐着 |
| 网络流量分类 | 数据包序列 | 数种 | 正常/攻击/异常 |

### 4.2 文本分类应用

| 场景 | 输入 | 类别数 | 说明 |
|------|------|--------|------|
| 情感分析 | 评论文本 | 2-5 | 正面/负面/中性 |
| 垃圾邮件 | 邮件内容 | 2 | 正常/垃圾 |
| 新闻分类 | 新闻文本 | 数十 | 体育/科技/娱乐等 |
| 意图识别 | 用户输入 | 数个 | 查询/下单/投诉 |
| 问答匹配 | 问题+答案 | 2 | 相关/不相关 |

### 4.3 时间序列预测应用

| 场景 | 输入 | 预测目标 | 说明 |
|------|------|---------|------|
| 股票预测 | 历史价格 | 未来价格 | 高波动，难度大 |
| 天气预测 | 温度/湿度等 | 未来天气 | 多变量，周期性强 |
| 电力负荷 | 历史用电量 | 未来需求 | 电力调度关键 |
| 交通流量 | 历史车流量 | 未来拥堵 | 智慧交通 |
| 设备故障 | 传感器数据 | 剩余寿命 | 预测性维护 |

### 4.4 文本生成应用

| 场景 | 输入 | 生成目标 | 说明 |
|------|------|---------|------|
| 姓名生成 | 起始符 | 新姓名 | 本模板使用 |
| 诗歌创作 | 主题/首句 | 完整诗歌 | 文学创作 |
| 代码补全 | 代码前缀 | 后续代码 | IDE辅助 |
| 对话生成 | 用户输入 | 回复内容 | 聊天机器人 |
| 摘要生成 | 长文本 | 短摘要 | 信息压缩 |

---

## 5. 使用说明

### 5.1 快速开始

```bash
# 进入项目根目录
cd py_ai_tech/

# 激活虚拟环境
source venv/bin/activate

# 运行序列分类模板（MNIST手写数字，70000张28×28图像，10类）
python rnn/classification.py

# 运行文本分类模板（合成中文评论，1000条，2类情感）
python rnn/text_classification.py

# 运行时间序列预测模板（合成正弦波，2000个窗口，30步→10步）
python rnn/sequence_prediction.py

# 运行文本生成模板（合成中文姓名，500个，字符级生成）
python rnn/text_generation.py
```

### 5.2 使用自己的数据

修改 `CONFIG` 类和相关数据加载函数：

**序列分类**：
```python
class CONFIG:
    # 1. 修改序列参数
    sequence_length = 100      # 你的序列长度
    input_size = 5             # 每个时间步的特征数
    num_classes = 3            # 你的类别数
    class_names = ["A", "B", "C"]

# 2. 修改 get_dataloaders() 函数
#    创建自定义Dataset，返回 (序列张量, 标签)
#    输入形状: (batch, seq_len, input_size)
class MyDataset(Dataset):
    def __init__(self, data_path):
        # 加载你的数据
        self.sequences = ...  # (N, seq_len, input_size)
        self.labels = ...     # (N,)
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
```

**文本分类**：
```python
class CONFIG:
    # 1. 修改文本相关参数
    vocab_size = 5000         # 你的词汇表大小
    max_seq_length = 50       # 最大句子长度
    num_classes = 5           # 类别数
    class_names = ["体育", "科技", "娱乐", "财经", "教育"]
    embedding_dim = 128       # 词嵌入维度
    bidirectional = True      # 推荐双向

# 2. 准备数据
#    需要分词工具(jieba)和词表构建
#    本模板使用词索引序列，你需要:
#    (1) 分词: "这个产品很好" → ["这个", "产品", "很", "好"]
#    (2) 转索引: [107, 101, 112, 2]
#    (3) padding: [107, 101, 112, 2, 0, 0, ...]
```

**时间序列预测**：
```python
class CONFIG:
    # 1. 修改预测参数
    input_size = 3             # 多变量(如温度+湿度+风速)
    seq_length = 60            # 回看窗口
    pred_length = 24           # 预测长度(如未来24小时)

# 2. 准备数据
#    CSV/数据库中的时间序列数据
#    需要做滑动窗口切分:
#    [x_1,...,x_60] → [x_61,...,x_84]
#    [x_2,...,x_61] → [x_62,...,x_85]
#    ...
#    注意: 必须用训练集的统计量做标准化!
```

**文本生成**：
```python
class CONFIG:
    # 1. 修改生成参数
    hidden_size = 256          # 更大数据集需要更大隐藏层
    embedding_dim = 64         # 字符级用32-64，词级用128-300
    temperature = 0.8          # 控制生成多样性

# 2. 准备数据
#    将文本转为字符/词索引序列
#    添加起始<S>和结束<E>符号
#    构建输入-目标对: 输入左移一位
```

### 5.3 修改超参数

```python
class CONFIG:
    # --- 序列参数 ---
    sequence_length = 28       # 输入序列长度
    input_size = 28            # 每个时间步的输入维度
    max_seq_length = 50        # 最大文本长度(文本分类)

    # --- 词嵌入 ---
    embedding_dim = 64         # 词嵌入维度
    vocab_size = 5000          # 词汇表大小

    # --- 模型相关 ---
    rnn_type = "lstm"          # "lstm"/"gru"/"rnn"
    hidden_size = 128          # 隐藏状态维度
    num_layers = 2             # RNN层数
    bidirectional = True       # 是否双向(分类推荐True)
    dropout_rate = 0.3         # Dropout比例

    # --- 训练相关 ---
    batch_size = 32            # 批次大小
    learning_rate = 1e-3       # 初始学习率
    epochs = 50                # 最大训练轮数
    weight_decay = 1e-5        # L2正则化

    # --- 早停 & LR调度 ---
    early_stop_patience = 7    # 早停耐心值
    scheduler_type = "step"    # 调度器类型

    # --- 梯度裁剪 ---
    max_grad_norm = 1.0        # 梯度L2范数上限(RNN标准值)

    # --- AMP混合精度 ---
    use_amp = True             # 启用混合精度(仅GPU有效)

    # --- 生成相关(仅text_generation) ---
    temperature = 0.8          # 采样温度
```

### 5.4 模型保存与加载

```python
# 保存模型
torch.save({
    "model_state_dict": model.state_dict(),
    "config": vars(cfg),
}, "model.pth")

# 加载模型（分类示例）
checkpoint = torch.load("model.pth", weights_only=True)
model = RNNClassifier(cfg).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

### 5.5 对新数据预测

```python
# 序列分类: 图像序列 → squeeze → softmax → argmax
from torchvision import transforms
img = Image.open("digit.png").convert("L")
transform = transforms.Compose([
    transforms.Resize(28), transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081]),
])
tensor = transform(img).unsqueeze(0)  # (1, 1, 28, 28)
tensor = tensor.squeeze(1)            # (1, 28, 28) ← RNN输入格式
with torch.no_grad():
    output = model(tensor.to(device))
    pred = output.argmax(dim=1)

# 文本分类: 分词 → 词索引 → padding → 模型推理
text = "这个产品很好"
indices = [char2idx.get(w, 1) for w in jieba.cut(text)]
indices = indices + [0] * (max_len - len(indices))
tensor = torch.tensor([indices], dtype=torch.long)
length = torch.tensor([len([w for w in indices if w != 0])])
with torch.no_grad():
    output = model(tensor.to(device), length)
    pred = output.argmax(dim=1)

# 时间序列预测: 标准化 → 模型推理 → 反标准化
history = np.array([...])  # 过去的值
history_norm = (history - train_mean) / train_std
x = torch.tensor(history_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
with torch.no_grad():
    pred_norm = model(x.to(device)).cpu().numpy()
    pred = pred_norm * train_std + train_mean  # 反标准化

# 文本生成: 种子 → 自回归生成
generated = model.generate(char2idx["<S>"], max_length=20, temperature=0.8)
text = "".join(idx2char[i] for i in generated if i not in [0, char2idx["<S>"]])
```

---

## 6. 任务类型对比

### 6.1 核心差异一览

| 对比项 | 序列分类 | 文本分类 | 时间序列预测 | 文本生成 |
|--------|---------|---------|------------|---------|
| **预测目标** | 序列类别 | 文本类别 | 未来数值 | 下一个字符 |
| **输出粒度** | 1个标签/序列 | 1个标签/文本 | k个数值/序列 | 1个字符/步 |
| **数据集** | MNIST(70K) | 合成评论(1K) | 合成正弦波(2K) | 合成姓名(500) |
| **输入格式** | (batch,seq,feat) | (batch,seq)词索引 | (batch,seq,1) | (batch,seq)字符索引 |
| **RNN类型** | LSTM | BiLSTM | LSTM | LSTM |
| **损失函数** | CrossEntropyLoss | CrossEntropyLoss | MSELoss | CrossEntropyLoss |
| **核心指标** | Accuracy | Accuracy/F1 | MAE/RMSE | Perplexity |
| **双向RNN** | 可选 | 推荐 | 不适用 | 不适用 |
| **词嵌入** | 不需要 | 需要 | 不需要 | 字符嵌入 |

### 6.2 网络结构对比

| 对比项 | 序列分类 | 文本分类 | 时间序列预测 | 文本生成 |
|--------|---------|---------|------------|---------|
| **模型类** | `RNNClassifier` | `TextClassifier` | `SequencePredictor` | `CharRNN` |
| **嵌入层** | 无 | nn.Embedding | 无 | nn.Embedding |
| **RNN类型** | LSTM单向 | BiLSTM | LSTM单向 | LSTM单向 |
| **隐藏层维度** | 128 | 128 | 64 | 128 |
| **层数** | 2 | 2 | 2 | 2 |
| **FC层** | 128→64→10 | 256→64→2 | 64→64→10 | 128→vocab_size |
| **输出** | 每序列1个标签 | 每文本1个标签 | 每序列k个数值 | 每步1个字符分布 |

### 6.3 训练超参数对比

| 超参数 | 序列分类 | 文本分类 | 时间序列预测 | 文本生成 | 选择依据 |
|--------|---------|---------|------------|---------|---------|
| **batch_size** | 128 | 32 | 64 | 32 | 分类用大batch更稳定; 生成/predict用中等 |
| **learning_rate** | 1e-3 | 1e-3 | 1e-3 | 5e-3 | 生成任务收敛快，可用稍大LR |
| **epochs** | 30 | 30 | 50 | 50 | 预测/生成需要更多轮; 分类收敛快 |
| **weight_decay** | 1e-5 | 1e-4 | 1e-5 | 1e-5 | RNN参数少，不需要太强L2 |
| **early_stop** | 7 | 7 | 10 | — | 分类/预测有早停; 生成通常不用 |
| **optimizer** | Adam | Adam | Adam | Adam | RNN推荐Adam，比SGD更稳定 |
| **scheduler** | StepLR | StepLR | Cosine | StepLR | 分类用Step; 预测用Cosine |
| **max_grad_norm** | 1.0 | 1.0 | 1.0 | 5.0 | RNN标准值1.0; 生成可以用更大 |
| **hidden_size** | 128 | 128 | 64 | 128 | 简单任务64; 中等128; 复杂256 |
| **dropout** | 0.3 | 0.3 | 0.2 | 0.1 | RNN对Dropout敏感，不宜太大 |

### 6.4 损失函数对比

```
序列/文本分类 - CrossEntropyLoss:
  输入: logits (batch, num_classes)
  内部: Softmax → -Σ y_i · log(p_i)
  特点: 所有类别概率和=1(互斥)

时间序列预测 - MSELoss:
  输入: 预测值 (batch, pred_length) vs 真实值
  公式: (1/n) Σ (pred - target)²
  特点: 对大误差惩罚更重，适合回归

文本生成 - CrossEntropyLoss (字符级):
  输入: logits (batch×seq_len, vocab_size)
  目标: 字符索引 (batch×seq_len)
  ignore_index=0: 忽略PAD位置的损失
  特点: 每个位置都计算损失，整体平均
  评估: Perplexity = exp(loss)，越低越好
```

### 6.5 数据预处理差异

```
序列分类:
  图像 → ToTensor → Normalize → squeeze(去掉channel维)
  标签: 整数(0-9)
  特殊: 4D图像 → 3D序列; 不需要数据增强

文本分类:
  文本 → 分词 → 词索引 → padding
  标签: 整数(0或1)
  特殊: 需要词嵌入; pack_padded_sequence处理变长; 按长度排序

时间序列预测:
  原始序列 → 滑动窗口 → 标准化(训练集统计量)
  标签: 连续值(未来值)
  特殊: 标准化必须用训练集统计量; 滑动窗口切分; 回归任务

文本生成:
  文本 → 字符索引 → 添加<S><E> → padding
  标签: 输入序列左移一位
  特殊: 起始/结束符; 自回归训练; 温度采样
```

### 6.6 评估指标对比

| 指标 | 序列/文本分类 | 时间序列预测 | 文本生成 |
|------|-------------|------------|---------|
| **主要指标** | Accuracy | MAE | Perplexity |
| **辅助指标** | F1, 混淆矩阵 | RMSE | 人工评估 |
| **指标含义** | 预测正确的比例 | 预测值与真实值的偏差 | 模型对数据的"惊讶程度" |
| **理想值** | 1.0 (100%) | 0 | 1.0 |

**Perplexity (困惑度)**:
```
PPL = exp(cross_entropy_loss)

PPL = 1:   模型完美预测每个字符
PPL = 10:  平均每个字符有10个等概率候选
PPL = vocab_size: 模型等于随机猜

经验:
  字符级中文: PPL < 5 算不错
  词级英文:   PPL < 50 算不错
  大语言模型: PPL < 10 (GPT系列)
```

---

## 7. 常见问题与调优

### 7.1 梯度消失/爆炸

**症状**：
- 消失: 训练不收敛，loss不下降，或长序列的早期信息丢失
- 爆炸: loss突然变成NaN，梯度值极大

**解决方案**：
```python
# 1. 使用LSTM/GRU替代原始RNN (解决梯度消失)
rnn_type = "lstm"  # ✅ 推荐
rnn_type = "rnn"   # ❌ 容易梯度消失

# 2. 梯度裁剪 (解决梯度爆炸)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# RNN标准值: 1.0-5.0
# 越小越保守，太大可能裁不住

# 3. 遗忘门偏置初始化为1 (LSTM技巧)
# 让遗忘门初始倾向于"记住"，防止训练初期信息丢失
for name, param in model.named_parameters():
    if "bias" in name and isinstance(model.rnn, nn.LSTM):
        n = param.size(0)
        param.data[n // 4:n // 2].fill_(1.0)

# 4. 正交初始化 (隐藏-隐藏权重)
nn.init.orthogonal_(rnn.weight_hh_l0)
# 正交矩阵连乘不会放大/缩小，有利于梯度传播

# 5. 减小学习率
learning_rate = 1e-3  →  5e-4  →  1e-4
```

### 7.2 过拟合（训练损失低，验证损失高）

**症状**：训练集准确率高，验证集准确率低

**解决方案**：
```python
# 1. 增大Dropout (但RNN不宜太大)
dropout_rate = 0.1  →  0.3  →  0.5  # 最大0.5

# 2. 减小网络规模
hidden_size = 256  →  128  →  64
num_layers = 3  →  2

# 3. 增大weight_decay
weight_decay = 1e-5  →  1e-4  →  1e-3

# 4. 减小词嵌入维度
embedding_dim = 128  →  64  →  32

# 5. 使用预训练词嵌入 (不更新)
embedding = nn.Embedding.from_pretrained(pretrained_vectors, freeze=True)

# 6. 数据增强 (文本)
# 同义词替换、随机删除、回译等
```

### 7.3 欠拟合（训练和验证损失都高）

**症状**：训练集和验证集的表现都很差

**解决方案**：
```python
# 1. 加大网络规模
hidden_size = 64  →  128  →  256
num_layers = 1  →  2  →  3

# 2. 使用双向RNN (分类任务)
bidirectional = True  # 准确率通常提升2-5%

# 3. 增加训练轮数
epochs = 20  →  50  →  100

# 4. 使用LSTM而非GRU/RNN
rnn_type = "lstm"  # 表达能力最强

# 5. 增大词嵌入维度
embedding_dim = 32  →  64  →  128

# 6. 减小Dropout
dropout_rate = 0.5  →  0.3  →  0.1
```

### 7.4 RNN类型选择

| 场景 | 推荐类型 | 原因 |
|------|---------|------|
| 入门学习 | RNN | 最简单，理解基本原理 |
| 几乎所有实际任务 | LSTM | 解决梯度消失，最稳定 |
| 追求速度 | GRU | 比LSTM快约20%，参数少25% |
| 文本分类 | BiLSTM | 捕获前后文，效果最好 |
| 时间序列预测 | LSTM | GRU通常也够用 |
| 文本生成 | LSTM/GRU | 不能用双向(未来信息不可知) |

### 7.5 隐藏层维度选择

| 任务 | 序列长度 | 数据量 | 推荐hidden_size | 本模板使用 |
|------|---------|--------|----------------|-----------|
| MNIST分类 | 28 | 70K | 64-128 | 128 |
| 文本分类 | 20 | 1K | 128-256 | 128 |
| 时间序列 | 30 | 2K | 64-128 | 64 |
| 文本生成 | 5 | 500 | 128-256 | 128 |
| 长文本 | 200+ | 10K+ | 256-512 | — |

经验：`hidden_size ≈ 4~8 × input_size` 是好的起点

### 7.6 序列长度问题

**问题**：序列太长 → 训练慢 + 梯度消失

**解决方案**：
```python
# 1. 截断长序列
max_seq_length = 50  # 只取前50个词

# 2. 梯度裁剪
max_grad_norm = 1.0

# 3. 使用LSTM(比RNN好得多)
rnn_type = "lstm"

# 4. 多层RNN(每层处理一部分序列长度)
num_layers = 2  # 每层学不同层次的时序特征

# 5. 注意力机制(进阶)
# 不受序列长度限制，可以"跳着看"
```

### 7.7 训练速度优化

```python
# 1. 使用GRU替代LSTM (快约20%)
rnn_type = "gru"

# 2. AMP混合精度 (速度↑1.5-2x)
use_amp = True

# 3. 减小batch中序列长度差异
# 按长度分桶(batch内长度相近) → 减少padding浪费

# 4. pin_memory + num_workers
DataLoader(..., pin_memory=True, num_workers=4)

# 5. 预提取词嵌入
# 将Embedding查找提前到数据加载阶段
```

---

## 8. 进阶扩展

### 8.1 经典RNN架构演进

```
RNN (1986)         → 开山之作，但梯度消失严重
LSTM (1997)        → 引入门控机制，解决长期依赖
GRU (2014)         → LSTM简化版，2个门，更快
BiLSTM (2005)      → 双向LSTM，分类任务标配
Seq2Seq (2014)     → 编码器-解码器，机器翻译突破
Attention (2015)   → 注意力机制，解决信息瓶颈
Transformer (2017) → 自注意力，完全替代RNN
BERT (2018)        → 双向Transformer，NLP里程碑
GPT (2018-2024)    → 单向Transformer，生成式AI
```

### 8.2 注意力机制 (Attention)

```python
class Attention(nn.Module):
    """
    注意力机制: 让模型"关注"输入序列中最重要的部分

    为什么需要注意力？
    - 普通RNN只用最后的隐藏状态，信息被压缩到固定维度
    - 注意力让每步输出都能"看"输入的所有位置
    - 不受序列长度限制，长序列效果更好

    工作原理:
    1. 计算查询(Q)与所有键(K)的相似度
    2. softmax得到注意力权重(和为1)
    3. 用权重对值(V)加权求和
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden_size) - 当前查询
        # encoder_outputs: (batch, seq_len, hidden_size) - 所有编码器输出

        # 计算注意力权重
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attention = torch.softmax(self.v(energy).squeeze(2), dim=1)
        # attention: (batch, seq_len) - 每个位置的权重

        # 加权求和
        context = torch.bmm(attention.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention
```

### 8.3 Transformer vs RNN

| 对比项 | RNN/LSTM | Transformer |
|--------|----------|-------------|
| **并行度** | 低(必须顺序处理) | 高(所有位置同时计算) |
| **长程依赖** | 有限(梯度传播距离) | 无限制(自注意力直接连接) |
| **计算复杂度** | O(n) | O(n²) (n=序列长度) |
| **位置信息** | 天然有序(时间步) | 需要位置编码 |
| **短序列** | 更快(计算量小) | 略慢(注意力开销) |
| **长序列** | 慢+效果差 | 快+效果好(但内存大) |
| **参数效率** | 较高 | 较低(需要更多数据) |
| **推荐场景** | 短序列、小数据 | 长序列、大数据 |

**何时用RNN vs Transformer？**
- 数据少(<10K) + 短序列(<100) → RNN/LSTM
- 数据多(>100K) + 长序列 → Transformer
- 实时/低延迟 → RNN(流式处理)
- 追求最高精度 → Transformer + 预训练

### 8.4 预训练语言模型

```python
# 使用预训练BERT做文本分类
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese", num_labels=2
)

# 编码文本
inputs = tokenizer("这个产品很好", return_tensors="pt", padding=True, truncation=True)

# 推理
with torch.no_grad():
    outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=1)

# 微调: 将BERT的输出接入自定义分类头
# 冻结BERT: for p in model.bert.parameters(): p.requires_grad = False
# 只训练分类头
```

### 8.5 Seq2Seq与机器翻译

```python
class Encoder(nn.Module):
    """编码器: 将源语言序列压缩为上下文向量"""
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden  # hidden传递给解码器

class Decoder(nn.Module):
    """解码器: 从上下文向量逐步生成目标语言"""
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        # x: 上一步生成的词
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output)
        return prediction, hidden
```

### 8.6 实际NLP项目建议

```
从简单到复杂的推荐路径:

1. 入门: 本模板的字符级RNN → 理解序列建模原理
2. 进阶: 词级BiLSTM + 词嵌入 → 真实文本分类
3. 中级: 加Attention的BiLSTM → 更好的分类/匹配
4. 高级: 预训练BERT微调 → 最强分类效果
5. 生成: Seq2Seq + Attention → 翻译/摘要
6. 前沿: GPT/LLM微调 → 对话/生成

关键工具链:
- 分词: jieba(中文), spaCy(英文)
- 预训练模型: Hugging Face Transformers
- 词向量: GloVe, fastText, Word2Vec
- 数据集: Hugging Face Datasets
```

### 8.7 GPU 加速要点

```python
# 1. 数据移到GPU
inputs = inputs.to(device)
targets = targets.to(device)

# 2. pin_memory=True 加速CPU→GPU传输
DataLoader(..., pin_memory=True)

# 3. 混合精度训练(显存减半，速度翻倍)
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda"):
    output = model(input)

# 4. 梯度累积(等效更大batch_size)
if (i + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()

# 5. RNN特有: 按序列长度分桶
# 减少短序列的padding浪费，加速训练
```

### 8.8 可复现性

```python
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 文件结构

```
rnn/
├── classification.py         # 序列分类模板(MNIST手写数字, LSTM)
├── text_classification.py    # 文本分类模板(合成评论数据, BiLSTM+词嵌入)
├── sequence_prediction.py    # 时间序列预测模板(合成正弦波, LSTM多步预测)
├── text_generation.py        # 文本生成模板(合成中文姓名, 字符级RNN)
└── RNN指南.md                # 本文档
```

---

> 💡 **提示**：四个模板文件中，序列分类使用公开数据集(MNIST自动下载)，文本分类、时间序列预测和文本生成使用合成数据(无需下载)。序列分类和文本分类是分类任务(输出类别)，时间序列预测是回归任务(输出连续值)，文本生成是生成任务(输出序列)。文本分类推荐使用双向RNN(BiLSTM)，生成任务不能使用双向。所有模板均支持AMP混合精度训练(仅GPU有效)。所有可调参数集中在 `CONFIG` 类中，方便统一管理和实验对比。替换为自己的数据时，修改 `CONFIG` 和数据加载函数即可。
