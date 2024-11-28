import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data.loader import load_data
import matplotlib.pyplot as plt
from IPython.display import clear_output

class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim=3072, num_classes=10):  # CIFAR-10: 32x32x3 = 3072
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

def train_cifar10(data_dir, num_epochs=100, batch_size=128, learning_rate=0.001):
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 添加记录训练历史的列表
    train_losses = []
    test_accuracies = []
    
    # 5. 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, target)
            
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
            accuracy = (predicted == Y_test).sum().item() / len(Y_test)
            
            # 记录训练指标
            train_losses.append(total_loss/len(train_loader))
            test_accuracies.append(accuracy)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Average Loss: {total_loss/len(train_loader):.4f}, '
                  f'Test Accuracy: {accuracy:.4f}')
    
    # 训练结束后绘制最终结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制训练损失
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Epochs')
    ax1.legend()
    
    # 绘制测试准确率
    ax2.plot(test_accuracies, label='Test Accuracy', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Test Accuracy over Epochs')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    plt.close()
    
    return model, train_losses, test_accuracies

if __name__ == '__main__':
    data_dir = 'data'  # CIFAR-10数据集所在目录
    train_cifar10(data_dir)
