# GAN 生成对抗网络 完全指南

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

### 1.1 什么是生成对抗网络 (GAN)

生成对抗网络（Generative Adversarial Network，简称 GAN），是由Ian Goodfellow在2014年提出的生成模型，通过两个神经网络的对抗训练来学习数据分布。

**核心思想**：让"造假者"(生成器)和"鉴定师"(判别器)相互竞争，在对抗中共同进步。

```
┌─────────────┐     假图像      ┌─────────────┐
│   生成器G    │ ──────────────→ │  判别器D     │
│  (造假者)    │                 │  (鉴定师)    │
└──────┬──────┘                 └──────┬──────┘
       │ ↑                             │
    噪声z                       真图像 │ 假图像
                                     ↓ ↓
                                 真(1)/假(0)
```

### 1.2 GAN vs CNN/RNN 的核心区别

| 对比项 | CNN (分类) | RNN (序列) | GAN (生成) |
|--------|-----------|-----------|-----------|
| **目标** | 预测标签 | 预测序列 | 生成新数据 |
| **网络数量** | 1个 | 1个 | 2个(G+D) |
| **损失函数** | CrossEntropy | CrossEntropy/MSE | 对抗损失 |
| **训练方式** | 单网络优化 | 单网络优化 | 极小极大博弈 |
| **输出** | 类别概率 | 序列值 | 新图像/数据 |
| **评估** | 准确率/F1 | 困惑度/MAE | FID/IS(间接) |
| **稳定性** | 稳定 | 较稳定 | 不稳定！ |
| **是否需要标签** | 是 | 是/否 | 否(无监督) |

**GAN最独特之处**：训练没有明确的"损失最小化"目标，而是两个网络的博弈平衡。

### 1.3 GAN的关键组成

| 组件 | 作用 | 类比 |
|------|------|------|
| **生成器G** | 从噪声生成假数据 | 造假者：伪造名画 |
| **判别器D** | 区分真假数据 | 鉴定师：辨别真伪 |
| **噪声z** | 生成器的随机输入 | 灵感来源：每次创作不同 |
| **对抗损失** | 驱动两个网络竞争 | 博弈规则：优胜劣汰 |

### 1.4 GAN训练流程

```
┌──────────────────────────────────────────────┐
│           每个 batch 重复执行                  │
│                                              │
│  1. 训练判别器D:                              │
│     a. 真实图像 → D → 应输出1(真)             │
│     b. 噪声 → G → 假图像 → D → 应输出0(假)    │
│     c. D_loss = BCE(D(real),1) + BCE(D(fake),0)│
│     d. 更新D的参数                            │
│                                              │
│  2. 训练生成器G:                              │
│     a. 噪声 → G → 假图像 → D → 应输出1(骗过D) │
│     b. G_loss = BCE(D(G(z)), 1)              │
│     c. 更新G的参数                            │
│                                              │
│  ⚠️ 关键: G和D需要保持力量平衡!               │
└──────────────────────────────────────────────┘
```

---

## 2. 技术原理

### 2.1 GAN的数学基础

GAN的训练目标是一个极小极大博弈(Minimax Game)：

```
min_G max_D V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]

解释:
- max_D: 判别器D要最大化辨别真假的能力
- min_G: 生成器G要最小化被识别的概率
- 理论最优: D(x) = 0.5 (完全分不清真假)
```

### 2.2 训练不稳定的根本原因

```
理想训练曲线:           实际训练曲线:
D_loss ───────          D_loss ╱╲╱╲╱╲
G_loss ───────          G_loss    ╲╱╱╲

为什么不稳定？
1. 梯度消失: D太强 → G(z)总被判0 → G梯度≈0 → 学不动
2. 模式崩溃: G只学会生成1-2种图 → D无法阻止 → 多样性丢失
3. 震荡: G和D交替占优，无法达到平衡
4. 超参数敏感: LR、网络结构对训练影响极大
```

### 2.3 DCGAN: 用卷积改造GAN

```
原始GAN(MLP):                    DCGAN(卷积):
生成器:                          生成器:
FC → FC → FC → 图像              FC → Reshape → ConvT → ConvT → ConvT → 图像
(参数多，效果差)                  (参数少，效果好)

判别器:                          判别器:
图像 → FC → FC → 1               图像 → Conv → Conv → Conv → 1
(丢失空间信息)                    (保留空间结构)
```

DCGAN的关键规则:
1. 用步进卷积替代池化(让网络自己学习下采样)
2. 生成器和判别器都用BatchNorm
3. 去除全连接层
4. G用ReLU(输出层Tanh)，D用LeakyReLU
5. 权重初始化: N(0, 0.02)

### 2.4 转置卷积(ConvTranspose2d)

```
普通卷积(下采样):              转置卷积(上采样):
输入 4×4 → Conv → 输出 2×2    输入 2×2 → ConvT → 输出 4×4

转置卷积的直观理解:
1. 在输入像素之间插入空格(stride-1个)
2. 用卷积核填充空格
3. 等效于"反向"的普通卷积

输出尺寸计算:
output = (input - 1) × stride - 2×padding + kernel_size

例: input=7, stride=2, padding=1, kernel=4
    output = (7-1)×2 - 2×1 + 4 = 14
```

### 2.5 条件GAN(cGAN)

```
普通GAN:     噪声z ──────→ G → 随机图像(无法控制)
条件GAN:     噪声z + 条件y → G → 指定条件图像(可控)

条件注入方式:
1. 拼接法(本模板使用):
   G: [z | embedding(y)] → 生成器
   D: [x | embedding(y)] → 判别器

2. 条件BN法(更高级):
   用条件y调制BatchNorm的γ和β参数
   效果更好但实现更复杂

3. 自注意力法(最新):
   用条件y生成空间注意力图
```

### 2.6 U-Net生成器

```
编码器(下采样)            解码器(上采样)
┌───┐                     ┌───┐
│64 │ ─── 跳跃连接 ────→ │64 │
└─┬─┘                     └─↑─┘
  ↓pool                     ↑up
┌───┐                     ┌───┐
│128│ ─── 跳跃连接 ────→ │128│
└─┬─┘                     └─↑─┘
  ↓pool                     ↑up
┌───┐                     ┌───┐
│256│ ─── 跳跃连接 ────→ │256│
└─┬─┘                     └─↑─┘
  ↓pool                     ↑up
┌───┐                     ┌───┐
│512│ ──────────────────→ │512│
└───┘     瓶颈层           └───┘

为什么跳跃连接重要？
- 瓶颈层只有高级语义，丢失了空间细节
- 跳跃连接直接传递低级特征(边缘/纹理)
- 类似ResNet的短路，但跨层更多
```

### 2.7 PatchGAN判别器

```
普通判别器:  图像 → 网络 → 1个值(整张图真/假)
PatchGAN:   图像 → 网络 → N×N矩阵(每个patch真/假)

┌─────────────┐          ┌───┬───┬───┐
│   输入图像    │  →  D →  │0.9│0.2│0.8│  每个值=该patch的真假判断
│   64×64     │          ├───┼───┼───┤
│             │          │0.7│0.3│0.9│  0.2=这个patch很假
│             │          ├───┼───┼───┤
│             │          │0.8│0.1│0.7│  0.9=这个patch很真
└─────────────┘          └───┴───┴───┘

优点:
1. 参数少: 不需要全连接层
2. 任意尺寸: 输入多大图都行
3. 关注细节: 每个patch独立判断
4. 感受野: 约70×70，关注局部纹理
```

### 2.8 GAN的评估指标

```
1. FID (Fréchet Inception Distance):
   - 用Inception网络提取真实/生成图像的特征
   - 计算两个特征分布的Wasserstein距离
   - 越低越好(0=完全一致)
   - 目前最常用的指标

2. IS (Inception Score):
   - 衡量生成图像的清晰度和多样性
   - 越高越好
   - 但不检测模式崩溃

3. 视觉检查(最直接):
   - 看生成图像是否清晰
   - 看生成图像是否有多样性
   - 看生成图像是否有奇怪伪影
```

---

## 3. 四大任务类型

### 3.1 图像生成 (Image Generation)

**目标**：从随机噪声生成逼真的图像

```
输入: 随机噪声 z(100维) → DCGAN → 输出: 手写数字图像(28×28)

训练数据: MNIST(只有图像，不需要标签)
损失函数: BCELoss(二元交叉熵)
模型: DCGAN(生成器+判别器)
```

### 3.2 条件生成 (Conditional Generation)

**目标**：按指定条件(类别)生成图像

```
输入: 噪声z + 条件标签(如"数字7") → cGAN → 输出: 数字7的图像

训练数据: MNIST(图像+标签)
损失函数: BCELoss
模型: 条件GAN(条件生成器+条件判别器)
```

### 3.3 异常检测 (Anomaly Detection)

**目标**：只用正常数据训练，检测异常数据

```
训练: 正常数据(圆形) → 训练GAN
推理: 测试图像 → AnoGAN → 重建误差 → 异常分数

损失函数: BCELoss(训练) + L1+L2(推理)
模型: DCGAN + AnoGAN推理
评估: AUC-ROC
```

### 3.4 图像翻译 (Image Translation)

**目标**：将一种图像风格转换为另一种

```
输入: 边缘线稿(64×64) → Pix2Pix → 输出: 填充图像(64×64)

训练数据: (边缘图, 填充图)图像对
损失函数: L1损失 + 对抗损失
模型: U-Net生成器 + PatchGAN判别器
```

---

## 4. 应用场景

### 4.1 图像生成应用

| 场景 | 输入 | 输出 | 说明 |
|------|------|------|------|
| 人脸生成 | 随机噪声 | 人脸图像 | StyleGAN |
| 风景生成 | 随机噪声 | 风景图像 | GauGAN |
| 艺术创作 | 噪声+风格 | 艺术图像 | 创意AI |
| 数据增强 | 随机噪声 | 训练样本 | 扩充数据集 |

### 4.2 条件生成应用

| 场景 | 条件 | 输出 | 说明 |
|------|------|------|------|
| 文本到图像 | 文字描述 | 对应图像 | StackGAN |
| 标签到图像 | 类别标签 | 指定类别图像 | cGAN |
| 草图到图像 | 线稿草图 | 彩色图像 | Scribble2Img |
| 语义图到图像 | 语义分割图 | 真实图像 | SPADE |

### 4.3 异常检测应用

| 场景 | 正常数据 | 异常类型 | 说明 |
|------|---------|---------|------|
| 工业检测 | 合格产品 | 缺陷产品 | 表面缺陷检测 |
| 医学影像 | 健康组织 | 病变区域 | 早期疾病筛查 |
| 网络安全 | 正常流量 | 攻击流量 | 入侵检测 |
| 金融风控 | 正常交易 | 欺诈交易 | 实时风控 |

### 4.4 图像翻译应用

| 场景 | 输入→输出 | 说明 |
|------|----------|------|
| 黑白上色 | 黑白→彩色 | 老照片修复 |
| 白天→夜晚 | 白天→夜晚 | 数据增强 |
| 线稿上色 | 线稿→彩色 | 动漫上色 |
| 超分辨率 | 低清→高清 | 图像增强 |
| 风格迁移 | 照片→油画 | 艺术创作 |

---

## 5. 使用说明

### 5.1 快速开始

```bash
# 进入项目根目录
cd py_ai_tech/

# 激活虚拟环境
source venv/bin/activate

# 运行图像生成模板(DCGAN, MNIST, 随机生成手写数字)
python gan/image_generation.py

# 运行条件生成模板(cGAN, MNIST, 按标签生成指定数字)
python gan/conditional_generation.py

# 运行异常检测模板(AnoGAN, 合成几何图形)
python gan/anomaly_detection.py

# 运行图像翻译模板(Pix2Pix, 边缘→填充)
python gan/image_translation.py
```

### 5.2 使用自己的数据

**图像生成**：
```python
class CONFIG:
    image_size = 64         # 根据你的图像大小调整
    in_channels = 3         # RGB=3, 灰度=1
    latent_dim = 100        # 噪声维度
    gen_features = 64       # 基础通道数

# 修改 get_dataloaders():
# 替换 datasets.MNIST 为你自己的 Dataset
dataset = datasets.ImageFolder("data/my_images", transform=transform)
```

**异常检测**：
```python
class CONFIG:
    image_size = 64
    # 只需要正常数据！
    # 修改 generate_synthetic_data() 为你的正常数据加载函数
```

**图像翻译**：
```python
class CONFIG:
    image_size = 256        # Pix2Pix常用256
    in_channels = 3         # RGB输入
    out_channels = 3        # RGB输出

# 准备图像对: (输入图像A, 目标图像B)
# 修改 ImagePairDataset 加载你的图像对
```

### 5.3 修改超参数

```python
class CONFIG:
    # --- 模型相关 ---
    latent_dim = 100        # 噪声维度(50-200常用)
    gen_features = 64       # 生成器基础通道(64/128)
    disc_features = 64      # 判别器基础通道(64/128)
    embedding_dim = 50      # 条件嵌入维度(仅cGAN)

    # --- 训练相关 ---
    batch_size = 128        # 批次大小
    learning_rate = 2e-4    # 学习率(1e-4~5e-4)
    beta1 = 0.5             # Adam β1(GAN专用，不是0.9！)
    epochs = 100            # 训练轮数
    label_smoothing = 0.9   # 标签平滑(防止D过强)
    d_steps_per_g = 1       # 每训练G一次，训练D几次

    # --- Pix2Pix专用 ---
    lambda_l1 = 100         # L1损失权重
    unet_base = 64          # U-Net基础通道
    disc_base = 64          # PatchGAN基础通道

    # --- AnoGAN专用 ---
    anomaly_steps = 50      # z优化步数
    anomaly_lr = 0.01       # z学习率
    lambda_feature = 0.1    # 特征损失权重
```

### 5.4 模型保存与加载

```python
# 保存
torch.save(G.state_dict(), "generator.pth")
torch.save(D.state_dict(), "discriminator.pth")

# 加载
G = Generator(cfg).to(device)
G.load_state_dict(torch.load("generator.pth", weights_only=True))
G.eval()

# 生成新图像
z = torch.randn(16, 100, device=device)
with torch.no_grad():
    fake_imgs = G(z)  # (16, 1, 28, 28), 范围[-1, 1]
    fake_imgs = fake_imgs * 0.5 + 0.5  # 反归一化到[0, 1]
```

---

## 6. 任务类型对比

### 6.1 核心差异一览

| 对比项 | 图像生成 | 条件生成 | 异常检测 | 图像翻译 |
|--------|---------|---------|---------|---------|
| **输入** | 随机噪声 | 噪声+条件 | 测试图像 | 条件图像 |
| **输出** | 随机图像 | 指定条件图像 | 异常分数 | 翻译图像 |
| **可控性** | 不可控 | 类别可控 | — | 输入可控 |
| **数据集** | MNIST(70K) | MNIST(70K) | 合成(1K) | 合成(1K) |
| **需要标签** | 否 | 是(类别) | 否 | 是(图像对) |
| **模型** | DCGAN | cGAN | AnoGAN | Pix2Pix |
| **生成器** | 反卷积网络 | 条件反卷积 | 反卷积网络 | U-Net |
| **判别器** | 卷积网络 | 条件卷积 | 卷积+特征 | PatchGAN |
| **损失函数** | BCELoss | BCELoss | BCE+L1+L2 | L1+BCE |

### 6.2 网络结构对比

| 对比项 | 图像生成 | 条件生成 | 异常检测 | 图像翻译 |
|--------|---------|---------|---------|---------|
| **G类型** | FC+ConvT | Embed+FC+ConvT | FC+ConvT | U-Net |
| **G输入** | z(100) | z(100)+emb(50) | z(100) | 图像(1×64×64) |
| **G输出** | 1×28×28 | 1×28×28 | 1×28×28 | 1×64×64 |
| **D类型** | Conv+Sigmoid | CondConv+Sigmoid | Conv+Feature | PatchGAN |
| **D输入** | 图像 | 图像+emb | 图像 | 图像对(拼接) |
| **D输出** | 1个概率 | 1个概率 | 1概率+特征 | N×N矩阵 |
| **特殊组件** | — | Embedding层 | 特征提取层 | 跳跃连接 |

### 6.3 训练超参数对比

| 超参数 | 图像生成 | 条件生成 | 异常检测 | 图像翻译 | 说明 |
|--------|---------|---------|---------|---------|------|
| batch_size | 128 | 128 | 64 | 16 | 翻译图大，batch小 |
| learning_rate | 2e-4 | 2e-4 | 2e-4 | 2e-4 | GAN标配 |
| beta1 | 0.5 | 0.5 | 0.5 | 0.5 | GAN必须用0.5 |
| epochs | 100 | 100 | 80 | 100 | 异常数据少，不需要太多 |
| label_smoothing | 0.9 | 0.9 | 0.9 | — | 翻译用BCEWithLogits |
| lambda_l1 | — | — | — | 100 | Pix2Pix论文推荐 |

### 6.4 代码逻辑流程对比

```
图像生成流程:
  数据: MNIST → Normalize([-1,1]) → DataLoader
  训练: BCELoss → Adam(β1=0.5) → D训练→G训练交替
  生成: z → G → 反归一化 → 图像
  特殊: 固定噪声可视化; 潜在空间插值

条件生成流程:
  数据: MNIST(images+labels) → Normalize → DataLoader
  训练: BCELoss → 条件嵌入拼接 → D(图+标签)→G(z+标签)
  生成: z+目标标签 → G → 指定类别图像
  特殊: 每行一个数字的网格可视化

异常检测流程:
  数据: 合成几何图形(只正常数据训练) → Normalize → DataLoader
  训练: BCELoss → 只用正常数据训练GAN
  检测: 测试图 → 优化z → 重建误差 → 异常分数
  评估: AUC-ROC, 分数分布, 原图vs重建对比
  特殊: 推理时优化z不更新G; 特征损失辅助

图像翻译流程:
  数据: 合成(边缘,填充)图像对 → Normalize → DataLoader
  训练: L1+BCEWithLogits → U-Net+PatchGAN → G+D交替
  翻译: 条件图像 → G → 翻译后图像
  特殊: PatchGAN输出矩阵; L1权重=100; 跳跃连接
```

---

## 7. 常见问题与调优

### 7.1 模式崩溃(Mode Collapse)

**症状**：生成器只产生少数几种图像，缺乏多样性

**识别方法**：生成100张图像，如果大部分相同→模式崩溃

**解决方案**：
```python
# 1. 减小学习率
learning_rate = 2e-4  →  1e-4

# 2. 增加判别器训练次数(让D更强，迫使G多样化)
d_steps_per_g = 1  →  2

# 3. 使用梯度惩罚(WGAN-GP)
# 4. 使用Minibatch Discrimination
# 5. 增加噪声到判别器输入
noise_std = 0.0  →  0.1

# 6. 使用WGAN替代原始GAN
```

### 7.2 判别器过强(G梯度消失)

**症状**：D_loss→0，G_loss→很大，生成图像无改善

**识别方法**：D(real)>0.99, D(fake)<0.01 → D太强

**解决方案**：
```python
# 1. 标签平滑
label_smoothing = 0.9  # 真实标签从1.0→0.9

# 2. 给D输入加噪声
noise_std = 0.1  # 削弱D的判别能力

# 3. 减少D训练次数
d_steps_per_g = 2  →  1

# 4. 增大G网络
gen_features = 64  →  128

# 5. 使用不同的学习率
G_lr = 2e-4, D_lr = 1e-4  # D学慢一点
```

### 7.3 生成图像模糊

**症状**：生成图像不清晰，缺乏细节

**解决方案**：
```python
# 1. 训练更久
epochs = 50  →  100  →  200

# 2. 增大网络
gen_features = 64  →  128

# 3. 增大噪声维度
latent_dim = 100  →  200

# 4. 检查Normalize参数
# 确保真实图像和生成图像范围一致[-1,1]

# 5. 对于图像翻译: 增大对抗损失权重
lambda_l1 = 100  →  50  # 相对增大对抗损失
```

### 7.4 训练不稳定(loss剧烈震荡)

**症状**：G_loss和D_loss上下跳动，不收敛

**解决方案**：
```python
# 1. 减小学习率
learning_rate = 2e-4  →  1e-4

# 2. 调整Adam参数
beta1 = 0.5   # 必须是0.5，不是0.9！
beta2 = 0.999

# 3. 使用学习率调度
scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

# 4. 增大batch_size(如果显存允许)
batch_size = 64  →  128

# 5. 使用WGAN-GP(更稳定的GAN变体)
```

### 7.5 GAN超参数选择指南

| 超参数 | 推荐值 | 范围 | 说明 |
|--------|--------|------|------|
| learning_rate | 2e-4 | 1e-4~5e-4 | GAN对LR非常敏感 |
| beta1 | 0.5 | 0.0~0.5 | DCGAN论文推荐0.5 |
| latent_dim | 100 | 50~200 | 100是标配 |
| label_smoothing | 0.9 | 0.8~0.95 | 防D过强 |
| gen_features | 64 | 32~128 | 小图用64，大图用128 |
| batch_size | 64~128 | 16~256 | 越大越稳定 |

### 7.6 何时停止训练？

```
GAN没有"收敛"的概念，判断停止时机:
1. 生成质量满意(目视检查)
2. FID不再下降
3. D(real)稳定在0.5~0.7(理想平衡)
4. 训练轮数达到预设上限

注意: G_loss和D_loss不能用来判断质量！
- G_loss下降≠质量变好(可能模式崩溃)
- D_loss下降≠质量变好(D变强，G反而学不动)
```

---

## 8. 进阶扩展

### 8.1 GAN演进历程

```
GAN (2014)        → 开山之作，理论证明
DCGAN (2016)      → 卷积+BN，实用化
cGAN (2014)       → 条件控制
Pix2Pix (2016)    → 图像翻译，U-Net+PatchGAN
CycleGAN (2017)   → 无配对图像翻译
WGAN (2017)       → Wasserstein距离，训练更稳定
WGAN-GP (2017)    → 梯度惩罚，替代权重裁剪
Progressive GAN (2018) → 逐步增加分辨率(4→8→...→1024)
StyleGAN (2019)   → 风格控制，生成高分辨率人脸
StyleGAN2 (2020)  → 改进训练，消除伪影
```

### 8.2 WGAN: 更稳定的训练

```python
# WGAN的核心改进:
# 1. 用Wasserstein距离替代JS散度
# 2. 判别器改为"评论家"(Critic)，输出实数而非概率
# 3. 权重裁剪或梯度惩罚(保证Lipschitz条件)

# WGAN-GP实现要点:
class Critic(nn.Module):
    # 与D相同，但最后不加Sigmoid
    def forward(self, x):
        return self.main(x)  # 输出实数，不加Sigmoid

# 梯度惩罚
def gradient_penalty(critic, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    crit_interpolated = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=crit_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(crit_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
```

### 8.3 StyleGAN: 高分辨率生成

```
StyleGAN的核心创新:
1. 映射网络: z → 8层FC → w(中间潜在空间)
2. 自适应实例归一化(AdaIN): 用w调制每层的风格
3. 随机噪声注入: 每层加入不同的随机噪声(细节)
4. 渐进式训练: 从4×4逐步增长到1024×1024

StyleGAN生成过程:
  z → 映射网络 → w
  常数输入(4×4) → AdaIN(w) → 上采样 → AdaIN(w) → ... → 1024×1024

为什么StyleGAN效果更好？
- w空间比z空间更解耦(每个维度独立控制一个属性)
- AdaIN允许逐层控制风格
- 噪声注入控制细节(头发丝、雀斑)
```

### 8.4 FID评估指标

```python
# FID (Fréchet Inception Distance) 计算
from scipy.linalg import sqrtm

def calculate_fid(real_features, fake_features):
    """
    用Inception网络提取特征，计算两个分布的距离。
    FID越低越好(0=完全一致)。

    步骤:
    1. 用Inception V3提取真实/生成图像的2048维特征
    2. 分别计算均值μ和协方差Σ
    3. FID = ||μ1-μ2||² + Tr(Σ1+Σ2 - 2√(Σ1Σ2))
    """
    mu1, sigma1 = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(0), np.cov(fake_features, rowvar=False)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid
```

### 8.5 GAN训练技巧总结

```
1. 数据预处理:
   - Normalize到[-1, 1](配合Tanh)
   - 不要用ImageNet的均值/标准差

2. 优化器:
   - 用Adam, β1=0.5, β2=0.999
   - LR=2e-4是起点
   - G和D用相同的LR

3. 稳定训练:
   - 标签平滑(0.9代替1.0)
   - 给D输入加噪声
   - D和G训练次数1:1或1:2
   - 使用梯度裁剪

4. 评估:
   - 目视检查最重要
   - FID作为辅助指标
   - loss值不能直接用来判断质量

5. 调试:
   - 先让D单独训练(不加G)，确认D能区分真假
   - 检查G输出范围是否与真实数据一致
   - 检查D(real)和D(fake)的分布
```

---

## 文件结构

```
gan/
├── image_generation.py       # 图像生成模板(DCGAN, MNIST, 随机生成)
├── conditional_generation.py # 条件生成模板(cGAN, MNIST, 按标签生成)
├── anomaly_detection.py      # 异常检测模板(AnoGAN, 合成几何图形)
├── image_translation.py      # 图像翻译模板(Pix2Pix, 边缘→填充)
└── GAN指南.md                # 本文档
```

---

> 💡 **提示**：四个模板文件中，图像生成和条件生成使用MNIST数据集(自动下载)，异常检测和图像翻译使用合成数据(无需下载)。GAN训练本质上是不稳定的，如果第一次运行效果不理想，请尝试调整学习率、增加训练轮数、或修改网络结构。所有可调参数集中在 `CONFIG` 类中，方便统一管理和实验对比。替换为自己的数据时，修改 `CONFIG` 和数据加载函数即可。
