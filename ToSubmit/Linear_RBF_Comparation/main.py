import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
import os
import time

# 1. 数据加载和预处理
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

train_data = pd.read_csv('Project-1-SVM/data/mnist_01_train.csv', dtype=np.float32, skiprows=1)
test_data = pd.read_csv('Project-1-SVM/data/mnist_01_test.csv', dtype=np.float32, skiprows=1)

X_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]
X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]

# 特征缩放
X_train = X_train.values / 255.0
X_test = X_test.values / 255.0

# Linear SVM
print("\nTraining Linear SVM...")
linear_svm = SVC(kernel='linear')
start_time = time.time()
linear_svm.fit(X_train, y_train)
linear_train_time = time.time() - start_time

start_time = time.time()
y_pred_linear = linear_svm.predict(X_test)
linear_predict_time = time.time() - start_time

# RBF SVM
print("\nTraining RBF SVM...")
rbf_svm = SVC(kernel='rbf')
start_time = time.time()
rbf_svm.fit(X_train, y_train)
rbf_train_time = time.time() - start_time

start_time = time.time()
y_pred_rbf = rbf_svm.predict(X_test)
rbf_predict_time = time.time() - start_time

# 性能比较
print("\n性能比较：")
print("Linear SVM:")
print(f"训练时间: {linear_train_time:.4f} 秒")
print(f"预测时间: {linear_predict_time:.4f} 秒")
print(f"准确率: {accuracy_score(y_test, y_pred_linear):.4f}")
print("分类报告:")
print(classification_report(y_test, y_pred_linear))

print("\nRBF SVM:")
print(f"训练时间: {rbf_train_time:.4f} 秒")
print(f"预测时间: {rbf_predict_time:.4f} 秒")
print(f"准确率: {accuracy_score(y_test, y_pred_rbf):.4f}")
print("分类报告:")
print(classification_report(y_test, y_pred_rbf))

# 绘制混淆矩阵
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_linear, ax=ax1, cmap='Blues')
ax1.set_title("Linear SVM Confusion Matrix")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rbf, ax=ax2, cmap='Blues')
ax2.set_title("RBF SVM Confusion Matrix")

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'svm_comparison_confusion_matrix.png'))
plt.close()

# 计算并绘制ROC曲线
from sklearn.metrics import roc_curve, auc

fpr_linear, tpr_linear, _ = roc_curve(y_test, linear_svm.decision_function(X_test))
fpr_rbf, tpr_rbf, _ = roc_curve(y_test, rbf_svm.decision_function(X_test))

roc_auc_linear = auc(fpr_linear, tpr_linear)
roc_auc_rbf = auc(fpr_rbf, tpr_rbf)

plt.figure()
plt.plot(fpr_linear, tpr_linear, color='darkorange', lw=2, label=f'Linear SVM ROC curve (AUC = {roc_auc_linear:.2f})')
plt.plot(fpr_rbf, tpr_rbf, color='green', lw=2, label=f'RBF SVM ROC curve (AUC = {roc_auc_rbf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(results_dir, 'svm_comparison_roc_curve.png'))
plt.close()

# 绘制训练时间和预测时间的对比图
plt.figure(figsize=(10, 6))
models = ['Linear SVM', 'RBF SVM']
train_times = [linear_train_time, rbf_train_time]
predict_times = [linear_predict_time, rbf_predict_time]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, train_times, width, label='Training Time')
rects2 = ax.bar(x + width/2, predict_times, width, label='Prediction Time')

ax.set_ylabel('Time (seconds)')
ax.set_title('Training and Prediction Times Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()
plt.savefig(os.path.join(results_dir, 'svm_comparison_times.png'))
plt.close()

print("\n性能比较结果已保存在 'results' 目录下。")
