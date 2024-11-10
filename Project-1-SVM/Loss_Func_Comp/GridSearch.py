from main import LinearClassifierHinge, LinearClassifierCrossEntropy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def grid_search():
    """
    对Hinge Loss和Cross Entropy Loss分类器进行网格搜索，寻找最优超参数
    """
    # Hinge Loss 网格搜索参数
    hinge_params = {
        'learning_rate': [1e-5, 1e-4, 1e-3],
        'epochs': [100, 200],
        'C': [0.01, 0.1, 1.0]
    }
    
    # Cross Entropy Loss 网格搜索参数
    ce_params = {
        'learning_rate': [0.001, 0.01, 0.1],
        'epochs': [100, 200]
    }
    
    best_hinge_params = {
        'learning_rate': None,
        'epochs': None,
        'C': None,
        'accuracy': 0
    }
    
    best_ce_params = {
        'learning_rate': None,
        'epochs': None,
        'accuracy': 0
    }
    
    # 加载数据
    train_data = pd.read_csv('Project-1-SVM/data/mnist_01_train.csv', dtype=np.float32, skiprows=1)
    test_data = pd.read_csv('Project-1-SVM/data/mnist_01_test.csv', dtype=np.float32, skiprows=1)
    
    X_train = train_data.iloc[:, 1:].values / 255.0
    y_train = train_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values / 255.0
    y_test = test_data.iloc[:, 0].values
    
    # Hinge Loss 网格搜索
    y_train_hinge = np.where(y_train == 0, -1, 1)
    y_test_hinge = np.where(y_test == 0, -1, 1)
    
    for lr in hinge_params['learning_rate']:
        for epochs in hinge_params['epochs']:
            for c in hinge_params['C']:
                clf = LinearClassifierHinge(learning_rate=lr, epochs=epochs, C=c)
                clf.fit(X_train, y_train_hinge)
                y_pred = clf.predict(X_test)
                y_pred = np.where(y_pred == -1, 0, 1)
                acc = accuracy_score(y_test, y_pred)
                
                if acc > best_hinge_params['accuracy']:
                    best_hinge_params = {
                        'learning_rate': lr,
                        'epochs': epochs,
                        'C': c,
                        'accuracy': acc
                    }
    
    # Cross Entropy Loss 网格搜索
    for lr in ce_params['learning_rate']:
        for epochs in ce_params['epochs']:
            clf = LinearClassifierCrossEntropy(learning_rate=lr, epochs=epochs)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            if acc > best_ce_params['accuracy']:
                best_ce_params = {
                    'learning_rate': lr,
                    'epochs': epochs,
                    'accuracy': acc
                }
    
    return best_hinge_params, best_ce_params

if __name__ == "__main__":
    best_hinge_params, best_ce_params = grid_search()
    print("Best Hinge Loss Parameters:", best_hinge_params)
    print("Best Cross Entropy Loss Parameters:", best_ce_params)
