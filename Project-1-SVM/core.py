import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
import os

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

        for epoch in range(self.epochs):
            # 随机打乱训练数据
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_samples):
                xi = X_shuffled[i]
                yi = y_shuffled[i]
                # 判断是否满足 margin 条件
                condition = yi * (np.dot(xi, self.w) + self.b) < 1
                if condition:
                    # 更新梯度（违反 margin 条件）
                    grad_w = self.w - self.C * yi * xi
                    grad_b = -self.C * yi
                else:
                    # 更新梯度（满足 margin 条件）
                    grad_w = self.w
                    grad_b = 0
                # 更新权重和偏置
                self.w -= self.learning_rate * grad_w
                self.b -= self.learning_rate * grad_b

            # 计算 hinge loss
            distances = 1 - y * (np.dot(X, self.w) + self.b)
            distances = np.maximum(0, distances)
            hinge_loss = self.C * np.mean(distances)
            # 计算总损失（包括正则化项）
            loss = 0.5 * np.dot(self.w, self.w) + hinge_loss
            self.loss_history.append(loss)

            # 每10轮打印一次损失
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss}")

    def predict(self, X):
        # 预测样本类别
        return np.sign(np.dot(X, self.w) + self.b)

# 1. 数据加载和预处理
train_data = pd.read_csv('data/mnist_01_train.csv', dtype=np.float32, skiprows=1)
test_data = pd.read_csv('data/mnist_01_test.csv', dtype=np.float32, skiprows=1)

X_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]
X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]

# 特征缩放
X_train = X_train.values / 255.0
X_test = X_test.values / 255.0

# 2. 模型训练和预测

# Linear SVM
print("\nTraining Linear SVM...")
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

# RBF SVM
print("\nTraining RBF SVM...")
rbf_svm = SVC(kernel='rbf')
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

# Logistic Regression
print("\nTraining Logistic Regression...")
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_logreg = log_reg.predict(X_test)

# Hinge Loss
print("\nTraining Hinge Loss Classifier...")
y_train_hinge = np.where(y_train == 0, -1, 1)
y_test_hinge = np.where(y_test == 0, -1, 1)
hinge_classifier = LinearClassifierHinge(learning_rate=0.00001, epochs=100, C=1.0)
hinge_classifier.fit(X_train, y_train_hinge)
y_pred_hinge = hinge_classifier.predict(X_test)
y_pred_hinge = np.where(y_pred_hinge == -1, 0, 1)

# 3. 创建实验记录
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
experiment_dir = f'experiments/experiment_{timestamp}'
if not os.path.exists('experiments'):
    os.makedirs('experiments')
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

# 创建或更新实验概述文件
summary_file = 'experiments/summary.txt'
with open(summary_file, 'a', encoding='utf-8') as f:
    f.write(f"\n=== 实验 {timestamp} ===\n")
    f.write(f"数据集大小: 训练集 {X_train.shape}, 测试集 {X_test.shape}\n")
    f.write("模型配置:\n")
    f.write(f"- Linear SVM: kernel='linear'\n")
    f.write(f"- RBF SVM: kernel='rbf'\n")
    f.write(f"- Logistic Regression: solver='lbfgs', max_iter=1000\n")
    f.write(f"- Hinge Loss: learning_rate={hinge_classifier.learning_rate}, epochs={hinge_classifier.epochs}, C={hinge_classifier.C}\n")

# 各个模型的结果
models = {
    "线性 SVM": (y_pred_linear, ""),
    "RBF SVM": (y_pred_rbf, ""),
    "逻辑回归 (Cross-Entropy)": (y_pred_logreg, ""),
    "Hinge Loss": (y_pred_hinge, f"学习率: {hinge_classifier.learning_rate}\n训练轮数: {hinge_classifier.epochs}\n正则化参数 C: {hinge_classifier.C}\n")
}

for name, (predictions, config) in models.items():
    with open(f'{experiment_dir}/results.txt', 'a', encoding='utf-8') as f:
        f.write(f"=== {name} ===\n")
        if config:
            f.write(config)
        f.write(f"准确率: {accuracy_score(y_test, predictions)}\n")
        f.write("分类报告:\n")
        f.write(classification_report(y_test, predictions))
        f.write("\n")

# 对于 Hinge Loss，额外保存损失历史
with open(f'{experiment_dir}/results.txt', 'a', encoding='utf-8') as f:
    f.write("=== Hinge Loss 损失历史 ===\n")
    for epoch, loss in enumerate(hinge_classifier.loss_history, 1):
        f.write(f"Epoch {epoch}: {loss}\n")

# 更新实验概述文件中的结果
with open(summary_file, 'a', encoding='utf-8') as f:
    f.write("\n准确率结果:\n")
    f.write(f"- Linear SVM: {accuracy_score(y_test, y_pred_linear):.4f}\n")
    f.write(f"- RBF SVM: {accuracy_score(y_test, y_pred_rbf):.4f}\n")
    f.write(f"- Logistic Regression: {accuracy_score(y_test, y_pred_logreg):.4f}\n")
    f.write(f"- Hinge Loss: {accuracy_score(y_test, y_pred_hinge):.4f}\n")
    f.write(f"详细结果保存在: {experiment_dir}\n")
    f.write("-" * 50 + "\n")

# 保存本次实验的详细结果
results_file = f'{experiment_dir}/results.txt'
with open(results_file, 'w', encoding='utf-8') as f:
    # 数据集信息
    f.write("=== 数据集信息 ===\n")
    f.write(f"训练集形状: {X_train.shape}\n")
    f.write(f"测试集形状: {X_test.shape}\n")
    f.write(f"训练集标签分布:\n{pd.Series(y_train).value_counts()}\n")
    f.write(f"测试集标签分布:\n{pd.Series(y_test).value_counts()}\n\n")

    # 各个模型的结果
    models = {
        "线性 SVM": (y_pred_linear, ""),
        "RBF SVM": (y_pred_rbf, ""),
        "逻辑回归 (Cross-Entropy)": (y_pred_logreg, ""),
        "Hinge Loss": (y_pred_hinge, f"学习率: {hinge_classifier.learning_rate}\n训练轮数: {hinge_classifier.epochs}\n正则化参数 C: {hinge_classifier.C}\n")
    }

    for name, (predictions, config) in models.items():
        f.write(f"=== {name} ===\n")
        if config:
            f.write(config)
        f.write(f"准确率: {accuracy_score(y_test, predictions)}\n")
        f.write("分类报告:\n")
        f.write(classification_report(y_test, predictions))
        f.write("\n")

    # 对于 Hinge Loss，额外保存损失历史
    f.write("=== Hinge Loss 损失历史 ===\n")
    for epoch, loss in enumerate(hinge_classifier.loss_history, 1):
        f.write(f"Epoch {epoch}: {loss}\n")

print(f"\n实验结果已保存到: {results_file}")

# 4. 绘制并保存图像
plt.figure(figsize=(15, 10))

# 混淆矩阵
plt.subplot(2, 2, 1)
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_linear)).plot(ax=plt.gca(), values_format='d')
plt.title('Linear SVM Confusion Matrix')

plt.subplot(2, 2, 2)
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_rbf)).plot(ax=plt.gca(), values_format='d')
plt.title('RBF SVM Confusion Matrix')

plt.subplot(2, 2, 3)
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_logreg)).plot(ax=plt.gca(), values_format='d')
plt.title('Cross-Entropy Classifier Confusion Matrix')

plt.subplot(2, 2, 4)
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_hinge)).plot(ax=plt.gca(), values_format='d')
plt.title('Hinge Loss Classifier Confusion Matrix')

plt.tight_layout()
plt.savefig(f'{experiment_dir}/confusion_matrices.png')

# Hinge Loss 学习曲线
plt.figure()
plt.plot(range(1, hinge_classifier.epochs + 1), hinge_classifier.loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Hinge Loss Learning Curve')
plt.savefig(f'{experiment_dir}/hinge_loss_curve.png')

plt.show()

# 创建实验索引文件
index_file = f'{experiment_dir}/index.txt'
with open(index_file, 'w', encoding='utf-8') as f:
    f.write("=== 实验文件索引 ===\n")
    f.write("1. config.json - 实验配置参数\n")
    f.write("2. results.txt - 详细实验结果\n")
    f.write("3. confusion_matrices.png - 混淆矩阵可视化\n")
    f.write("4. hinge_loss_curve.png - Hinge Loss 学习曲线\n")
    f.write("5. predictions.npz - 模型预测结果\n")
    f.write("6. hinge_model_params.npz - Hinge Loss 模型参数\n")