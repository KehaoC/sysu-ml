import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from datetime import datetime
from data.loader import load_data
import numpy as np

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 0.01
INPUT_DIM = 3072  # CIFAR-10: 32x32x3 = 3072
NUM_CLASSES = 10
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.1
PATIENCE = 5
MOMENTUM = 0.9
DROPOUT_RATE = 0.5

# Device configuration
device = (
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
)
print(f"Using device: {device}")

# Define data transformations
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010))
])

# Replace CIFAR10 dataset loading with custom loader
X_train, Y_train, X_test, Y_test = load_data(DATA_DIR)

# Convert numpy arrays to PyTorch tensors and create custom datasets
class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.FloatTensor(data.reshape(-1, 3, 32, 32)) / 255.0
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.data)

# Create custom datasets
train_dataset = CIFAR10Dataset(X_train, Y_train, transform=transform_train)
test_dataset = CIFAR10Dataset(X_test, Y_test, transform=transform_test)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True
)

class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
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
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten images
        x = self.network(x)
        return x

class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter +=1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_cifar10(num_epochs=NUM_EPOCHS, 
                  batch_size=BATCH_SIZE, 
                  learning_rate=LEARNING_RATE):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = datetime.now()
    
    # Initialize model, criterion, optimizer
    model = MLPClassifier().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[30, 60, 90],
        gamma=0.2
    )
    
    early_stopping = EarlyStopping()
    
    # Training records
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Evaluation function
    def evaluate():
        model.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                outputs = model(data)
                loss = criterion(outputs, target)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        return accuracy, avg_loss
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # Flatten images
            outputs = model(data)
            loss = criterion(outputs, target)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        test_accuracy, test_loss = evaluate()
        
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Average Loss: {avg_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, '
              f'Test Accuracy: {test_accuracy:.4f}, '
              f'Learning Rate: {current_lr:.6f}')
        
        early_stopping(test_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Plotting
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
    hyperparams_text = f'Epochs: {epoch+1} | Batch Size: {batch_size} | Learning Rate: {learning_rate} | ' \
                      f'Duration: {duration} | Final Test Accuracy: {test_accuracies[-1]:.4f}'
    plt.figtext(0.5, 0.01, hyperparams_text, ha='center', fontsize=10, wrap=True)
    
    plt.subplots_adjust(bottom=0.2)
    
    filename = f'mlp_{datetime.now().strftime("%Y%m%d_%H%M%S")}_' \
              f'e{epoch+1}_b{batch_size}_lr{learning_rate}_' \
              f't{duration.total_seconds():.0f}s'
    
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename}.png'))
    plt.show()
    plt.close()
    
    return model, train_losses, test_accuracies

if __name__ == '__main__':
    train_cifar10()
