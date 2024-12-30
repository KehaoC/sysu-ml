# Project 2 - DeepLearning

探索神经网络在图像分类任务上的应用。在给定数据集CIFAR-10的训练集上训练模型，并在测试集上验证其性能。

> 学号：22336018
>
> 姓名：蔡可豪

## Requirements: 

1. 在给定的训练数据集上，分别训练一个线性分类器（Softmax 分类器），多层感知机（MLP）和卷积神经网络（CNN） 

2) 在MLP 实验中，研究使用不同网络层数和不同神经元数量对模型性能的影响 
3) 在 CNN 实验中，以 LeNet 模型为基础，

4) 分别使用 SGD 算法、SGD  Momentum 算法和 Adam 算法训练模型，观察并讨论他们对模型训练速度和性能的影响 

5) 比较并讨论线性分类器、MLP 和CNN 模型在CIFAR-10 图像分类任务上的性能区别 
6) 学习一种主流的深度学习框架（如：Tensorfolw，PyTorch，MindSpore），并用其中

实验报告需包含（但不限于）: 
1）采用的模型结构和训练方法（包括数据预处理的方法、模型参数初始化方法、超参数
选择、优化方法及其它用到的训练技巧） 
2）实验结果，及对观察结果的充分讨论 
将实验报告（.doc 或.pdf）和代码（不要数据）打包成一个文件，  文件包的命名规则为：
学号+姓名.tar 或.zip，并上传到课程FTP：ftp://172.18.167.164/Assignment2/report   
Due: 2024.11.30 

# 实验过程

## 1 Softmax分类器

写了一个非常简单的线性分类模型，然后开始训练。

```python
class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, num_classes=NUM_CLASSES):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)
```

本质上，线性分类器其实就是一个维度转换的模型，表现能力非常有限。

只可以描述线性边界，对更复杂的数据无能为力。所以先尝试了一下比较小的训练论数看看训练效果。

```python
    model = SoftmaxClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```



![Screenshot 2024-11-28 at 20.52.32](./Screenshot%202024-11-28%20at%2020.52.32.png)

训练轮次 50 轮，发现 `loss` 已经接近收敛，也比较平滑，但是测试准确率的浮动是比较大，但是基本上都维持在了 0.38 上下。上升空间很有限。

如果还要提高模型性能，就得考虑怎么让 loss 继续下降。显然，提高训练时长是一个方法，但必然不显著，因为看起来已经快要收敛了。

**有一种可能是卡在了局部最优解上下不去。**试着解决这个问题。

1. 把 Adam 优化器换成 SGD+Momentum。
2. 提高初始学习率。

![Screenshot 2024-11-28 at 21.00.58](./Screenshot%202024-11-28%20at%2021.00.58.png)

对比 Adam 和 SGD 的训练结果。发现

1. Adam 在早期下降的非常快，远快于 SGD
1. SGD 在后期依旧平稳下降，Adam 则过早地收敛。

应该是因为 SGD 保留了先前的动量信息，一阶和二阶的动量，因此模型训练到后期之后依旧保留了历史动量的惯性

因为假设 Adam 在探索到了一个局部最优解后，一个比较深的局部最优解，那么前期保留的惯性会使得模型失去探索的机会。

就像骑自行车下坡的时候，由于速度太快，无法在中途急转弯去探索另外一个方向，而那个方向恰巧是全局最优。

而采用了 SGD+Momentum+WeightDecay 的训练方法。相当于是在训练后期小步下降，减少惯性，所以表现出了上述特征。

下面提高训练时长看看。

![Screenshot 2024-11-28 at 21.15.03](./Screenshot%202024-11-28%20at%2021.15.03.png)

可以看到基本上到了一百轮的时候，测试准确率也基本不变了。

基本上来说这里就是全局最优了，但是为了防止因为后期动量依旧很大在跨越了全局最优，设置一个动量衰减的机制。让动量从 0.9 逐渐下降到 0.5。

```python
    # 动态调整动量
    def adjust_momentum(epoch):
        return INITIAL_MOMENTUM * (1 - epoch/NUM_EPOCHS) + FINAL_MOMENTUM * (epoch/NUM_EPOCHS)
    
    for epoch in range(num_epochs):
        # 更新动量
        optimizer.param_groups[0]['momentum'] = adjust_momentum(epoch)
```

![](./Screenshot%202024-11-28%20at%2021.20.09.png)s

可以看到后期甚至是中期明显地更加平滑了。而准确率依旧在0.40左右。最终准确率的具体数值是`0.415`。

由此可以发现线性分类器的局限性。显然，对于图像分类，比如猫和狗，会比较相似，在高维空间中可能局部重叠，用一个简单的线性分类无法区分。

下面继续尝试多层感知机。

## 2 MLP

首先是模型定义

```python
class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            # Input layer -> First hidden layer (3072 -> 512)
            nn.Linear(3072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            # First hidden layer -> Second hidden layer (512 -> 256)
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Second hidden layer -> Output layer (256 -> 10)
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.network(x)
```

输入层: 3072 个神经元 (32x32x3 CIFAR-10 图像展平)
第一隐藏层: 512 个神经元 + BatchNorm + ReLU + Dropout(0.5)
第二隐藏层: 256 个神经元 + BatchNorm + ReLU + Dropout(0.5)
输出层: 10 个神经元 (对应 CIFAR-10 的 10 个类别)

总共有 3 层参数层（不计激活函数和正则化层），参数数量为：
第一层: 3072 × 512 + 512 (权重 + 偏置)
第二层: 512 × 256 + 256 (权重 + 偏置)
第三层: 256 × 10 + 10 (权重 + 偏置)

先训练一下看一看效果。（用的是Adam）训练个10轮次。（因为发现训练速度明显变慢了）

![Screenshot 2024-11-28 at 21.32.12](./Screenshot%202024-11-28%20at%2021.32.12.png)

显然，训练还不够充分，那么提高至 50 次训练。

要看一下曲线的特征才方便去调整模型或者训练方式。不然无从下手。

![Screenshot 2024-11-28 at 23.05.37](./Screenshot%202024-11-28%20at%2023.05.37.png)

发现相对于训练轮次为 10 的时候，早期模型震荡非常明显，显然是没有进入一个合适的方向，虽然模型依旧保持Loss 下降的趋势，但是现在可以视图提高一下收敛速度了。

（这个时候电脑已经有一点点发烫了）

![Screenshot 2024-11-28 at 23.07.10](./Screenshot%202024-11-28%20at%2023.07.10.png)

于是去查了一下`mac air m2`有没有什么加速方法：

```python
import torch
import platform
import sys

def test_pytorch_environment():
    print("\n=== PyTorch Environment Test ===\n")
    
    # 基本系统信息
    print("System Information:")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # CUDA 可用性
    print("\nCUDA Support:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    
    # MPS (Metal Performance Shaders) 可用性
    print("\nMPS Support:")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"MPS Built: {torch.backends.mps.is_built()}")
    
    # 确定最佳可用设备
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"\nBest Available Device: {device}")
    
    # 简单的张量运算测试
    print("\nTensor Operations Test:")
    try:
        # 创建测试张量
        x = torch.randn(1000, 1000)
        if device != "cpu":
            x = x.to(device)
        
        # 进行一些基本运算
        start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        
        if device == "cuda":
            start_time.record()
        
        # 执行一些计算密集型操作
        for _ in range(100):
            y = torch.matmul(x, x)
        
        if device == "cuda":
            end_time.record()
            torch.cuda.synchronize()
            print(f"CUDA Computation Time: {start_time.elapsed_time(end_time):.2f} ms")
        
        print("Tensor operations completed successfully!")
        
    except Exception as e:
        print(f"Error during tensor operations: {str(e)}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_pytorch_environment()
```

测试的结果是如下，说明可以根据平台来优化一下

![Screenshot 2024-11-28 at 23.12.12](./Screenshot%202024-11-28%20at%2023.12.12.png)

对应做了一些修改，让设备发挥作用，发现好像训练数据没有变快，但是至少 CPU 发烫没那么严重了。于是乎就在这个设备上面继续训练吧！

![Screenshot 2024-11-28 at 23.23.48](./Screenshot%202024-11-28%20at%2023.23.48.png)

为了提高早期收敛速度，做出了如下更改：

1. **网络架构优化：**

> 增加网络宽度（第一层增加到1024个神经元）

提升最一开始网络对特征的捕捉能力，提高学习的“广度”

> 使用 LeakyReLU 替代 ReLU，避免死亡 ReLU 问题

因为 ReLU 可能因为在某些区域，神经元输出一直是 0，导致压根就没有学习，然后导致收敛速度过慢。Leaky 在输出小于 0 的时候还是有一定的梯度的，避免了所谓的死亡 ReLU 问题。

> 减小 Dropout 比例到0.3，让网络在早期能更好地学习

过高的 Dropout 会让模型学不到东西，因为全都drop 调了。

> 添加一个额外的隐藏层

对应第一层的神经元增多，再加一个层，意味着提高学习的“深度”

> 使用 Kaiming 初始化改善深层网络的训练

Kaiming 初始化是专门给 ReLU 类激活函数用的初始化器，可以让模型在早期更稳定，处于一个可以快速学习且找到方向的初始化位置。

2. **优化器和学习率调整：**

>  增加初始学习率到 0.003

早期学习率高一些，买的步子也大一些。

>  使用 AdamW 优化器替代 Adam

早期权重衰减太明显会抑制学习能力，而 AdamW 解耦了权重正则化和梯度的更新。不然正则化会被梯度更新策略所干扰。

> 使用 OneCycleLR 学习率调度器，实现超级收敛。

早期提高学习率到一个很高的值进行快速探索，后期见效学习率缓慢探索。

上面的图是之前的结果下面是跑了一百轮之后的结果。下面的图是这一轮的结果。

<img src="./Screenshot%202024-11-28%20at%2023.05.37.png" alt="Screenshot 2024-11-28 at 23.05.37" style="zoom:50%;" />

![Screenshot 2024-11-28 at 23.31.10](./Screenshot%202024-11-28%20at%2023.31.10.png)

发现啊早期学习率下降非常非常明显，对比之前的图像。说明之前那些组合调整是有效的。但是观察准确率图像发现了一个很大的问题，就是训练准确率远高于测试准确率。

经典过拟合。

解决这个问题，还能进一步提高。于是乎做出了下面这些优化

1. **数据增强**：增强数据的多样性（适用于 CIFAR-10）。

也就是把原先的训练集改大小啊旋转啊拉伸变换，从数据层面提高模型的识别能力，避免学到一些非正常特征。

2. **标签平滑**：避免模型过于自信，提升泛化能力。

3. **增加权重衰减和适度调整 Dropout**：提升正则化效果。

4. **早停机制**：避免模型过拟合。这个机制让模型在早期发现过拟合就立马停止，减少时间的浪费。

5. **学习率调度器**：优化训练过程中的学习率变化。

![Screenshot 2024-11-28 at 23.58.12](./Screenshot%202024-11-28%20at%2023.58.12.png)

怀疑是最简单的因素，也就是模型太复杂了，于是下面降低一下模型的复杂度再试一试。

```python
class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            # Input layer -> First hidden layer (3072 -> 512)
            nn.Linear(3072, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),  # 减小 Dropout 率
            
            # First hidden layer -> Second hidden layer (512 -> 256)
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            # Second hidden layer -> Output layer (256 -> 10)
            nn.Linear(256, 10)
        )
        
        # 使用 Kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)
```

现在依旧在过拟合，早期的时候模型在train上就表现的非常好。试着继续降低一下模型的复杂度。

![Screenshot 2024-11-29 at 00.02.34](./Screenshot%202024-11-29%20at%2000.02.34.png)

依旧是早停然后过拟合...于是尝试使用SGD作为优化器，SGD的泛化性能通常会比Adam要好一些。同时进一步降低了模型的复杂度。

最终采用了如下的架构，解决了过拟合问题：

1. 采用逐层递减的网络，并且使用了50%的dropout，每层都用 batchnorm 正则化

```python
self.network = nn.Sequential(
    nn.Linear(3072, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(DROPOUT_RATE),  # 0.5
    
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(DROPOUT_RATE),
    
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(DROPOUT_RATE),
    
    nn.Linear(128, NUM_CLASSES)
)
```

2. 数据增广技术保留了

```python
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
    transforms.Normalize(...)
])
```

这个无论如何都是好的，因为扩充了训练数据，并且避免模型学到一些不是真正有用的特征。

3. 使用了 SGD+Momentum

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=MOMENTUM,  # 0.9
    weight_decay=WEIGHT_DECAY,
    nesterov=True
)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 60, 90],
    gamma=0.2
)
```

普通SGD容易卡在鞍点上，加Momentum可以帮助模型冲过鞍点。

Nesterov 动量是动量法的改进版，先根据动量进行一次预更新，然后在预更新点计算梯度，通常能获得更好的收敛性

对于学习率调度

```python
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 60, 90],  # 在这些epoch时降低学习率
    gamma=0.2                 # 每次降低为原来的20%
)
```

初始：lr = 0.01
30 epoch后：lr = 0.002
60 epoch后：lr = 0.0004
90 epoch后：lr = 0.00008

显然，模型可以在早期自由探索，然后在晚期慢慢摸索。

目前训练到大概五十轮，发现训练准确率低于测试准确率，而测试准确率在大概 45% 左右。应该是训练不充分。

![Screenshot 2024-11-29 at 00.30.18](./Screenshot%202024-11-29%20at%2000.30.18.png)

但是设备已经不支持继续跑了...于是继续尝试CNN吧。

## 3 CNN

```python

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.features = nn.Sequential(
            # First conv block: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # Second conv block: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            # Third conv block: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the features
        x = self.classifier(x)
        return x
```

特征提取层（Features）:3个卷积块，每个块包含：

Conv2d层 + BatchNorm2d + ReLU激活 +MaxPool2d

通道数演进：3 → 32 → 64 → 128

图像尺寸演进：32x32 → 16x16 → 8x8 → 4x4

分类层的话:

两个全连接层：(128*4*4) → 512 → 10
使用了Dropout(0.5)防止过拟合

训练的话先训个十轮，因为怕太久了电脑还是跑不动。如下是训练结果，最终准确率到了79%，而且才训练了十轮，而且训练速度，由于用到了mac的mps，并没有训练非常久（比MLP快多了）就拿到了这个结果还是非常优秀的。

![Screenshot 2024-11-30 at 16.18.51](./Screenshot%202024-11-30%20at%2016.18.51.png)

下面想要继续提高训练轮次，因为从上图看到模型还没有完全收敛。但是提高训练轮次之后又要跑很久，所以先优化了一下模型的运行速度。

1. 提高了num_workers的数量到4个，并且使用persistant_worker放置反复销毁占用时间。
2. 尝试了一下混合精度：使用了FP16，应该可以减少内存占用提高计算速度，并且使用 GradScaler 自动处理数值稳定性。
3. 使用 prefetch_factor 预加载数据

提高到 30 轮，等等结果...

![Screenshot 2024-11-30 at 16.25.36](./Screenshot%202024-11-30%20at%2016.25.36.png)

从 loss 上可以看到基本是收敛了的，而且早期 loss 下降的很快，说明立马找到了优化方向，后面缓慢逼近，效果非常好。

比较一下他和 LeNet-5 模型（经典模型）

```python
Input (32x32) -> Conv1 (6@28x28) -> Pool1 (6@14x14) -> Conv2 (16@10x10) -> Pool2 (16@5x5) -> 
FC1 (120) -> FC2 (84) -> Output (10)
```

我现在的模型

```
Input (32x32x3) -> Conv1 (32@32x32) -> BN1 -> Pool1 (32@16x16) -> 
Conv2 (64@16x16) -> BN2 -> Pool2 (64@8x8) -> 
Conv3 (128@8x8) -> BN3 -> Pool3 (128@4x4) -> 
FC1 (512) -> FC2 (10)
```

模型都采用了卷积层+池化层的基本组合，都使用全连接层作为最终分类器，都是逐层降低特征图尺寸的金字塔结构。

不过现在的CNN可以看作是LeNet的现代化版本，保留了LeNet的基本设计理念，但是使用了更多稍微现代一些的方案。

比如老的 LeNEt 用的是 Tanh 做激活函数，现在用的是 ReLU，显然 ReLU 会更快一些。以及一些BatchNorm和 Dropout 策略。都是一些近些年才成为常态的好东西。不过基本的思想，比如参数共享啊比如局部捕捉特征这些，都是由 LeNet 所奠定的基础。影响深渊。

下面比价一下不同模型结构的影响：

```python
    configs = [
        CNNConfig("Shallow", conv_layers=2, filters=[32, 64]),
        CNNConfig("Deep", conv_layers=4, filters=[32, 64, 128, 256]),
        CNNConfig("Wide", conv_layers=3, filters=[64, 128, 256]),
        CNNConfig("AvgPool", conv_layers=3, filters=[32, 64, 128], pool_type='avg'),
        CNNConfig("Large_Kernel", conv_layers=3, filters=[32, 64, 128], kernel_size=5)
    ]
```

测试不同配置下的结果，如果还要跑完整的三十轮要太久了，这里索性就跑个 15 轮吧。还可以接受。

1. **Shallow（浅层网络）**
```python
CNNConfig("Shallow", conv_layers=2, filters=[32, 64])
```
- 只有2层卷积层
- 过滤器数量较少（32->64）
- 适合简单的特征提取
- 参数量最少，训练速度最快
- 可能欠拟合复杂数据集

2. **Deep（深层网络）**
```python
CNNConfig("Deep", conv_layers=4, filters=[32, 64, 128, 256])
```
- 4层卷积层，网络最深
- 过滤器数量逐层翻倍（32->64->128->256）
- 可以提取更复杂的特征层次
- 参数量较大，训练时间较长
- 可能需要更多的训练数据来避免过拟合

3. **Wide（宽网络）**
```python
CNNConfig("Wide", conv_layers=3, filters=[64, 128, 256])
```
- 3层卷积层，适中的深度
- 起始过滤器数量较大（64开始）
- 每层都有较多的特征图
- 可以捕获更多并行的特征模式
- 参数量大，但深度适中

4. **AvgPool（平均池化）**
```python
CNNConfig("AvgPool", conv_layers=3, filters=[32, 64, 128], pool_type='avg')
```
- 使用平均池化而不是最大池化
- 保留更多的空间信息
- 对噪声更不敏感
- 可能对纹理特征的提取更好
- 可能会丢失一些显著特征

5. **Large_Kernel（大卷积核）**
```python
CNNConfig("Large_Kernel", conv_layers=3, filters=[32, 64, 128], kernel_size=5)
```
- 使用5x5的卷积核（而不是默认的3x3）
- 每层的感受野更大
- 可以直接捕获更大范围的特征
- 参数量增加
- 计算成本更高

所有配置的共同特点：
- 都使用了批量归一化（BatchNorm）
- 都使用了 ReLU 激活函数
- 都包含 Dropout（率为0.5）用于防止过拟合
- 最后都使用相同的分类器结构（512个神经元的全连接层）

>  研究网络深度的影响（Shallow vs Deep）

深度的网络显然造成了更慢的学习速度，不过也可以学习到更加复杂的特征层次。因为有更大的模型容量，所以也天然适合更加复杂的数据集。

但是相应的也需要更多的数据和更长的时间才可以让这个模型表现的更好。

>  研究网络宽度的影响（Wide vs 其他）

宽网络的话可以并行学习更多的特征，信息流通的通道多了，可以对某些特征的学习更加直接。但是参数量显著的增加了，而且计算开销非常大，也更容易导致过拟合。

所以要尤其考虑计算资源和正则化（避免过拟合）

>  研究池化方式的影响（AvgPool vs 默认的MaxPool）

平均池化保留更多的背景信息，对噪声更加 robust，而最大池化保留的是比较显著的特征，对于 CIFAR-10，学习图像差异，学习的更多的应该是显著特征。所以 MaxPool 应该会表现的更好。

> 研究卷积核大小的影响（Large_Kernel vs 其他）

大卷积核的化，显然，训练速度会变慢，但是更大的感受野，也可以直接捕获更大范围的特征（也有可能捕捉到无关特征）

综合来看，还是适中的深度，三四层左右，适当的宽度，32 或者 64 过滤，最大池化，kernel 用 3*3，最终效果最好。

![Screenshot 2024-11-30 at 16.43.59](./Screenshot%202024-11-30%20at%2016.43.59.png)

![Screenshot 2024-11-30 at 16.44.13](./Screenshot%202024-11-30%20at%2016.44.27.png)

![Screenshot 2024-11-30 at 16.44.49](./Screenshot%202024-11-30%20at%2016.44.49.png)

![Screenshot 2024-11-30 at 16.45.02](./Screenshot%202024-11-30%20at%2016.45.02.png)

![Screenshot 2024-11-30 at 16.50.00](./Screenshot%202024-11-30%20at%2016.50.00.png)

至此，完成了整个实验项目，学到了很多东西，电脑也跑的老热乎了。