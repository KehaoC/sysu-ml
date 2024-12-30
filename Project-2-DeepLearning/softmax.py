import torch
import torch.nn as nn
import torch.optim as optim
from data.loader import load_data
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.01
INPUT_DIM = 3072  # CIFAR-10: 32x32x3 = 3072
NUM_CLASSES = 10
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
INITIAL_MOMENTUM = 0.9
FINAL_MOMENTUM = 0.5
WEIGHT_DECAY = 5e-4

class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, num_classes=NUM_CLASSES):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

def train_cifar10(data_dir=DATA_DIR, num_epochs=NUM_EPOCHS, 
                  batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Record start time
    start_time = datetime.now()
    
    # 1. 加载数据
    X_train, Y_train, X_test, Y_test = load_data(data_dir)
    
    # 2. 数据预处理
    # 归一化到 [0,1] 区间
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # 转换为 PyTorch 张量
    X_train = torch.FloatTensor(X_train)
    Y_train = torch.LongTensor(Y_train)
    X_test = torch.FloatTensor(X_test)
    Y_test = torch.LongTensor(Y_test)
    
    # 3. 创建数据集
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    # 4. 初始化模型、损失函数和优化器
    model = SoftmaxClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), 
                         lr=learning_rate,
                         momentum=INITIAL_MOMENTUM,
                         weight_decay=WEIGHT_DECAY)
    
    # 使用学习率衰减策略
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                    T_max=num_epochs,
                                                    eta_min=1e-6)
    
    # 添加记录训练历史的列表
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # 动态调整动量
    def adjust_momentum(epoch):
        return INITIAL_MOMENTUM * (1 - epoch/NUM_EPOCHS) + FINAL_MOMENTUM * (epoch/NUM_EPOCHS)
    
    # 5. 训练循环
    for epoch in range(num_epochs):
        # 更新动量
        optimizer.param_groups[0]['momentum'] = adjust_momentum(epoch)
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 打印训练进度
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # 每个epoch结束后评估模型
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs.data, 1)
            test_accuracy = (predicted == Y_test).sum().item() / len(Y_test)
            train_accuracy = correct_train / total_train
            
            # 记录训练指标
            train_losses.append(total_loss/len(train_loader))
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Average Loss: {total_loss/len(train_loader):.4f}, '
                  f'Train Accuracy: {train_accuracy:.4f}, '
                  f'Test Accuracy: {test_accuracy:.4f}')
        
        scheduler.step()
    
    # 训练结束后绘制最终结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制训练损失
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Epochs')
    ax1.legend()
    
    # 绘制训练和测试准确率
    ax2.plot(train_accuracies, label='Train Accuracy', color='blue')
    ax2.plot(test_accuracies, label='Test Accuracy', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Train and Test Accuracy over Epochs')
    ax2.legend()
    
    plt.tight_layout()
    
    # 添加超参数文本到图像底部
    duration = datetime.now() - start_time
    hyperparams_text = f'Epochs: {num_epochs} | Batch Size: {batch_size} | Learning Rate: {learning_rate} | ' \
                      f'Duration: {duration} | Final Accuracy: {test_accuracies[-1]:.4f}'
    plt.figtext(0.5, 0.01, hyperparams_text, ha='center', fontsize=10, wrap=True)
    
    # 调整图像布局以为底部文本留出空间
    plt.subplots_adjust(bottom=0.2)
    
    # Create detailed filename with hyperparameters
    filename = f'softmax_{datetime.now().strftime("%Y%m%d_%H%M%S")}_' \
              f'e{num_epochs}_b{batch_size}_lr{learning_rate}_' \
              f't{duration.total_seconds():.0f}s'
    
    # Save the plot with detailed filename
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename}.png'))
    plt.show()
    plt.close()
    
    return model, train_losses, test_accuracies

if __name__ == '__main__':
    train_cifar10()
