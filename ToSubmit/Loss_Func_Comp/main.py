import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
import os
import time

# Hinge Loss 分类器实现
class LinearClassifierHinge:
    def __init__(self, learning_rate=0.001, epochs=100, C=1.0):
        # 初始化分类器参数
        self.learning_rate = learning_rate  # 学习率
        self.epochs = epochs  # 训练轮数
        self.C = C  # 正则化参数

    def fit(self, X, y):
        # 训练模型
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.w = np.random.randn(n_features) * 0.01  # 初始化权重
        self.b = 0  # 初始化偏置
        self.loss_history = []  # 记录损失历史
        self.time_per_epoch = []  # 记录每个epoch的训练时间

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # 向量化计算预测值
            scores = np.dot(X, self.w) + self.b
            # 计算 margin 违反条件的样本
            margin = y * scores
            mask = margin < 1
            
            # 向量化计算梯度（使用平均值）
            grad_w = self.w - self.C * np.mean(y[mask].reshape(-1, 1) * X[mask], axis=0)
            grad_b = -self.C * np.mean(y[mask])
            
            # 更新参数
            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b

            # 计算损失（保持不变）
            distances = 1 - y * (np.dot(X, self.w) + self.b)
            distances = np.maximum(0, distances)
            hinge_loss = self.C * np.mean(distances)
            loss = 0.5 * np.dot(self.w, self.w) + hinge_loss
            self.loss_history.append(loss)

            epoch_end_time = time.time()
            self.time_per_epoch.append(epoch_end_time - epoch_start_time)

            # 每10轮打印一次损失
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss}")
            
    def predict(self, X):
        # 预测样本类别
        return np.sign(np.dot(X, self.w) + self.b)

# Cross Entropy Loss 分类器实现
class LinearClassifierCrossEntropy:
    def __init__(self, learning_rate=0.001, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def sigmoid(self, z):
        # 避免数值溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)  # 初始化为零向量
        self.b = 0
        self.loss_history = []
        self.time_per_epoch = []
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            # 前向传播
            z = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(z)
            
            # 计算交叉熵损失
            epsilon = 1e-15  # 防止log(0)
            loss = -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
            self.loss_history.append(loss)
            
            # 反向传播
            dz = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(dz)
            
            # 更新参数
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # 添加L2正则化
            lambda_reg = 0.01  # 正则化强度
            self.w -= self.learning_rate * lambda_reg * self.w
            
            epoch_end_time = time.time()
            self.time_per_epoch.append(epoch_end_time - epoch_start_time)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss}")
                
    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int)

# 1. 数据加载和预处理
train_data = pd.read_csv('Project-1-SVM/data/mnist_01_train.csv', dtype=np.float32, skiprows=1)
test_data = pd.read_csv('Project-1-SVM/data/mnist_01_test.csv', dtype=np.float32, skiprows=1)

X_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]
X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]

# 特征缩放
X_train = X_train.values / 255.0
X_test = X_test.values / 255.0

# 创建实验结果目录
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Hinge Loss
print("\nTraining Hinge Loss Classifier...")
y_train_hinge = np.where(y_train == 0, -1, 1)
y_test_hinge = np.where(y_test == 0, -1, 1)
hinge_classifier = LinearClassifierHinge(
    learning_rate=0.00001,  # 稍微增加学习率
    epochs=200,           # 增加训练轮数
    C=0.1                # 减小正则化强度
)
hinge_classifier.fit(X_train, y_train_hinge)
y_pred_hinge = hinge_classifier.predict(X_test)
y_pred_hinge = np.where(y_pred_hinge == -1, 0, 1)
hinge_accuracy = accuracy_score(y_test, y_pred_hinge)

# Cross Entropy Loss
print("\nTraining Cross Entropy Loss Classifier...")
ce_classifier = LinearClassifierCrossEntropy(
    learning_rate=0.01,   # 调整学习率
    epochs=200,           # 增加训练轮数
)
ce_classifier.fit(X_train, y_train)
y_pred_ce = ce_classifier.predict(X_test)
ce_accuracy = accuracy_score(y_test, y_pred_ce)

# 创建实验结果目录
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = os.path.join('results', f'experiment_{timestamp}')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(hinge_classifier.loss_history) + 1), hinge_classifier.loss_history, label='Hinge Loss')
plt.plot(range(1, len(ce_classifier.loss_history) + 1), ce_classifier.loss_history, label='Cross Entropy Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.savefig(os.path.join(results_dir, 'loss_comparison.png'))
plt.close()

# 绘制每个epoch的训练时间比较
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(hinge_classifier.time_per_epoch) + 1), hinge_classifier.time_per_epoch, label='Hinge Loss')
plt.plot(range(1, len(ce_classifier.time_per_epoch) + 1), ce_classifier.time_per_epoch, label='Cross Entropy Loss')
plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.title('Training Time per Epoch Comparison')
plt.legend()
plt.savefig(os.path.join(results_dir, 'time_per_epoch_comparison.png'))
plt.close()

# 保存实验结果
with open(os.path.join(results_dir, 'experiment_results.txt'), 'w') as f:
    f.write("=== Hinge Loss Classifier ===\n")
    f.write(f"Learning Rate: {hinge_classifier.learning_rate}\n")
    f.write(f"Epochs: {hinge_classifier.epochs}\n")
    f.write(f"C: {hinge_classifier.C}\n")
    f.write(f"Average Time per Epoch: {np.mean(hinge_classifier.time_per_epoch):.4f} seconds\n")
    f.write(f"Accuracy: {hinge_accuracy:.4f}\n\n")
    
    f.write("=== Cross Entropy Loss Classifier ===\n")
    f.write(f"Learning Rate: {ce_classifier.learning_rate}\n")
    f.write(f"Epochs: {ce_classifier.epochs}\n")
    f.write(f"Average Time per Epoch: {np.mean(ce_classifier.time_per_epoch):.4f} seconds\n")
    f.write(f"Accuracy: {ce_accuracy:.4f}\n")

print(f"\nExperiment results saved to {os.path.join(results_dir, 'experiment_results.txt')}")
print(f"Loss comparison plot saved to {os.path.join(results_dir, 'loss_comparison.png')}")
print(f"Time per epoch comparison plot saved to {os.path.join(results_dir, 'time_per_epoch_comparison.png')}")
