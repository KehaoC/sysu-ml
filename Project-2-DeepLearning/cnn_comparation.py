import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
import os
from data.loader import load_data
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# 配置
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
NUM_EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DATA_DIR = 'data'
OUTPUT_DIR = 'experiments_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class CNNConfig:
    def __init__(self, name, conv_layers, filters, pool_type='max', 
                 dropout_rate=0.5, kernel_size=3):
        self.name = name
        self.conv_layers = conv_layers
        self.filters = filters
        self.pool_type = pool_type
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size

class FlexibleCNN(nn.Module):
    def __init__(self, config):
        super(FlexibleCNN, self).__init__()
        self.config = config
        self.padding = config.kernel_size // 2
        self.features = self._make_features()
        
        feature_size = 32 // (2 ** config.conv_layers)
        final_filters = config.filters[config.conv_layers-1]
        flatten_size = final_filters * (feature_size ** 2)
        
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(512, 10)
        )
    
    def _make_features(self):
        layers = []
        in_channels = 3
        
        for i in range(self.config.conv_layers):
            layers.extend([
                nn.Conv2d(in_channels, self.config.filters[i], 
                         kernel_size=self.config.kernel_size, 
                         padding=self.padding),
                nn.BatchNorm2d(self.config.filters[i]),
                nn.ReLU()
            ])
            
            if self.config.pool_type == 'max':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.append(nn.AvgPool2d(2, 2))
            
            in_channels = self.config.filters[i]
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, test_data, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler()
    
    X_test, Y_test = test_data
    X_test, Y_test = X_test.to(DEVICE), Y_test.to(DEVICE)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            with autocast():
                outputs = model(data)
                loss = criterion(outputs, target)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                      f'Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs.data, 1)
            test_accuracy = (predicted == Y_test).sum().item() / len(Y_test)
            
            avg_loss = total_loss/len(train_loader)
            train_losses.append(avg_loss)
            test_accuracies.append(test_accuracy)
            
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                  f'Loss: {avg_loss:.4f}, '
                  f'Test Accuracy: {test_accuracy:.4f}')
        
        scheduler.step()
    
    return train_losses, test_accuracies

def plot_results(results, filename):
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 1, 1)
    for name, (losses, _) in results.items():
        plt.plot(losses, label=name)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot test accuracy
    plt.subplot(2, 1, 2)
    for name, (_, accuracies) in results.items():
        plt.plot(accuracies, label=name)
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def run_experiments():
    # 加载数据
    X_train, Y_train, X_test, Y_test = load_data(DATA_DIR)
    
    # 预处理数据
    X_train = torch.FloatTensor(X_train.reshape(-1, 3, 32, 32) / 255.0)
    Y_train = torch.LongTensor(Y_train)
    X_test = torch.FloatTensor(X_test.reshape(-1, 3, 32, 32) / 255.0)
    Y_test = torch.LongTensor(Y_test)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )
    
    # 定义不同的模型配置
    configs = [
        # CNNConfig("Shallow", conv_layers=2, filters=[32, 64]),
        # CNNConfig("Deep", conv_layers=4, filters=[32, 64, 128, 256]),
        # CNNConfig("Wide", conv_layers=3, filters=[64, 128, 256]),
        # CNNConfig("AvgPool", conv_layers=3, filters=[32, 64, 128], pool_type='avg'),
        CNNConfig("Large_Kernel", conv_layers=3, filters=[32, 64, 128], kernel_size=5)
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTraining {config.name} model...")
        model = FlexibleCNN(config).to(DEVICE)
        train_losses, test_accuracies = train_model(
            model, train_loader, (X_test, Y_test), config
        )
        results[config.name] = (train_losses, test_accuracies)
        
        # 保存最终结果
        print(f"{config.name} - Final Test Accuracy: {test_accuracies[-1]:.4f}")
    
    # 绘制结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_results(results, f'model_comparison_{timestamp}.png')
    
    return results

if __name__ == '__main__':
    results = run_experiments()
    
    # 打印最终结果总结
    print("\nFinal Results Summary:")
    for name, (_, accuracies) in results.items():
        print(f"{name:12} - Final Accuracy: {accuracies[-1]:.4f}")