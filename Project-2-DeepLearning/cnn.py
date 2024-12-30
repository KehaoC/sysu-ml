import torch
import torch.nn as nn
import torch.optim as optim
from data.loader import load_data
import matplotlib.pyplot as plt
import os
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn

# Hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
WEIGHT_DECAY = 5e-4

# 优化 CUDA 性能设置
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    # MPS 特定优化
else:
    DEVICE = torch.device("cpu")
    cudnn.benchmark = True  # 如果输入尺寸固定，启用这个可以提升性能

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

def train_cifar10(data_dir=DATA_DIR, num_epochs=NUM_EPOCHS, 
                  batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = datetime.now()
    
    # Load and process data
    X_train, Y_train, X_test, Y_test = load_data(data_dir)
    
    # Reshape data for CNN (N, C, H, W)
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)
    
    X_train = torch.FloatTensor(X_train.astype('float32') / 255.0)
    Y_train = torch.LongTensor(Y_train)
    X_test = torch.FloatTensor(X_test.astype('float32') / 255.0)
    Y_test = torch.LongTensor(Y_test)
    
    # 使用 pin_memory 和更多 workers 加速数据加载
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=4,  # 增加 worker 数量
        persistent_workers=True,  # 保持 worker 进程存活
        prefetch_factor=2  # 预加载 batch 数量
    )
    
    # 初始化混合精度训练
    scaler = GradScaler()
    
    # 将模型和数据移至 GPU
    model = CNNClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), 
                          lr=learning_rate,
                          weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                    T_max=num_epochs,
                                                    eta_min=1e-6)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # 使用混合精度训练
            with autocast():
                outputs = model(data)
                loss = criterion(outputs, target)
            
            optimizer.zero_grad(set_to_none=True)  # 更快的梯度清零
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            X_test = X_test.to(DEVICE)
            Y_test = Y_test.to(DEVICE)
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs.data, 1)
            test_accuracy = (predicted == Y_test).sum().item() / len(Y_test)
            train_accuracy = correct_train / total_train
            
            train_losses.append(total_loss/len(train_loader))
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Average Loss: {total_loss/len(train_loader):.4f}, '
                  f'Train Accuracy: {train_accuracy:.4f}, '
                  f'Test Accuracy: {test_accuracy:.4f}')
        
        scheduler.step()
    
    # Plotting code
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Epochs')
    ax1.legend()
    
    ax2.plot(train_accuracies, label='Train Accuracy', color='blue')
    ax2.plot(test_accuracies, label='Test Accuracy', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Train and Test Accuracy over Epochs')
    ax2.legend()
    
    plt.tight_layout()
    
    duration = datetime.now() - start_time
    hyperparams_text = f'Epochs: {num_epochs} | Batch Size: {batch_size} | Learning Rate: {learning_rate} | ' \
                      f'Duration: {duration} | Final Accuracy: {test_accuracies[-1]:.4f}'
    plt.figtext(0.5, 0.01, hyperparams_text, ha='center', fontsize=10, wrap=True)
    
    plt.subplots_adjust(bottom=0.2)
    
    filename = f'cnn_{datetime.now().strftime("%Y%m%d_%H%M%S")}_' \
              f'e{num_epochs}_b{batch_size}_lr{learning_rate}_' \
              f't{duration.total_seconds():.0f}s'
    
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename}.png'))
    plt.show()
    plt.close()
    
    return model, train_losses, test_accuracies

if __name__ == '__main__':
    train_cifar10()