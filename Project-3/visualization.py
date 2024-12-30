import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import json

def set_style():
    """Set the style for all plots"""
    # 使用默认样式
    plt.style.use('default')
    
    # 设置颜色主题
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    sns.set_palette(colors)
    
    # 设置图表样式
    plt.rcParams.update({
        'figure.figsize': [12, 8],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def plot_accuracy_comparison(kmeans_results, gmm_results, save_dir):
    """Plot accuracy comparison between K-means and GMM"""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prepare data
    methods = []
    train_accs = []
    test_accs = []
    
    # K-means results
    for init_method, results in kmeans_results.items():
        methods.append(f'K-means\n({init_method})')
        train_accs.append(results['train_acc'])
        test_accs.append(results['test_acc'])
    
    # GMM results
    for config, results in gmm_results.items():
        cov_type, init_method = config.split('_')
        methods.append(f'GMM\n({cov_type},\n{init_method})')
        train_accs.append(results['train_acc'])
        test_accs.append(results['test_acc'])
    
    # Bar plot
    x = np.arange(len(methods))
    width = 0.35
    
    ax1.bar(x - width/2, train_accs, width, label='Training')
    ax1.bar(x + width/2, test_accs, width, label='Test')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    
    # Training time comparison
    times = []
    for results in [*kmeans_results.values(), *gmm_results.values()]:
        times.append(results['train_time'])
    
    ax2.bar(x, times, width)
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cluster_sizes(model, X, save_dir, model_name):
    """Plot cluster size distribution"""
    set_style()
    plt.figure(figsize=(10, 6))
    
    # Get cluster assignments
    labels = model.predict(X)
    unique_labels, counts = torch.unique(labels, return_counts=True)
    
    # Plot
    plt.bar(unique_labels.cpu().numpy(), counts.cpu().numpy())
    plt.xlabel('Cluster Index')
    plt.ylabel('Number of Samples')
    plt.title(f'Cluster Size Distribution - {model_name}')
    
    plt.savefig(Path(save_dir) / f'cluster_sizes_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_convergence(log_file, save_dir):
    """Plot convergence curves from log file"""
    set_style()
    
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    plt.figure(figsize=(12, 6))
    
    # 分别绘制 K-means 和 GMM 的曲线
    for model_name, log_data in logs.items():
        if 'convergence' in log_data:
            iterations = range(1, len(log_data['convergence']) + 1)
            if model_name.startswith('kmeans'):
                label = f"K-means ({model_name.split('_')[1]})"
                plt.plot(iterations, log_data['convergence'], label=label, linestyle='-')
            else:  # GMM
                cov_type, init = model_name.split('_')[1:]
                label = f"GMM ({cov_type}, {init})"
                plt.plot(iterations, log_data['convergence'], label=label, linestyle='--')
    
    plt.xlabel('Iteration')
    plt.ylabel('Metric Value (Inertia / Log-likelihood)')
    plt.title('Convergence Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'convergence.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cluster_visualization(model, X, y, save_dir, model_name):
    """Plot 2D visualization of clusters using PCA"""
    set_style()
    from sklearn.decomposition import PCA
    
    # Convert to CPU for sklearn
    X_cpu = X.cpu().numpy()
    y_cpu = y.cpu().numpy()
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cpu)
    
    # Get cluster assignments
    labels = model.predict(X).cpu().numpy()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot clusters
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6)
    ax1.set_title(f'Cluster Assignments - {model_name}')
    ax1.set_xlabel('First Principal Component')
    ax1.set_ylabel('Second Principal Component')
    plt.colorbar(scatter1, ax=ax1)
    
    # Plot true labels
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_cpu, cmap='tab10', alpha=0.6)
    ax2.set_title('True Labels')
    ax2.set_xlabel('First Principal Component')
    ax2.set_ylabel('Second Principal Component')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f'cluster_visualization_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(model, X, y, save_dir, model_name):
    """Plot confusion matrix between predicted clusters and true labels"""
    set_style()
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Get predictions
    pred = model.predict(X).cpu().numpy()
    true = y.cpu().numpy()
    
    # Compute confusion matrix
    cm = confusion_matrix(true, pred)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.savefig(Path(save_dir) / f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_figure(kmeans_results, gmm_results, models, data, save_dir):
    """Create a comprehensive figure combining all visualizations with explanations"""
    set_style()
    
    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 25))
    gs = plt.GridSpec(4, 2, height_ratios=[1, 1, 1, 0.3])
    
    # 1. Model Performance Comparison (Top Row)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Prepare accuracy data
    methods = []
    train_accs = []
    test_accs = []
    times = []
    
    for init_method, results in kmeans_results.items():
        methods.append(f'K-means\n({init_method})')
        train_accs.append(results['train_acc'])
        test_accs.append(results['test_acc'])
        times.append(results['train_time'])
    
    for config, results in gmm_results.items():
        cov_type, init_method = config.split('_')
        methods.append(f'GMM\n({cov_type},\n{init_method})')
        train_accs.append(results['train_acc'])
        test_accs.append(results['test_acc'])
        times.append(results['train_time'])
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax1.bar(x - width/2, train_accs, width, label='Training')
    ax1.bar(x + width/2, test_accs, width, label='Test')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    
    ax2.bar(x, times, width)
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    
    # 2. Convergence Curves (Second Row)
    ax3 = fig.add_subplot(gs[1, :])
    log_file = Path(save_dir) / 'training_log.json'
    if log_file.exists():
        with open(log_file, 'r') as f:
            logs = json.load(f)
        
        for model_name, log_data in logs.items():
            if 'convergence' in log_data:
                iterations = range(1, len(log_data['convergence']) + 1)
                if model_name.startswith('kmeans'):
                    label = f"K-means ({model_name.split('_')[1]})"
                    ax3.plot(iterations, log_data['convergence'], label=label, linestyle='-')
                else:  # GMM
                    cov_type, init = model_name.split('_')[1:]
                    label = f"GMM ({cov_type}, {init})"
                    ax3.plot(iterations, log_data['convergence'], label=label, linestyle='--')
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Metric Value (Inertia / Log-likelihood)')
        ax3.set_title('Convergence Curves')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True)
    
    # 3. Cluster Visualization (Third Row)
    # Choose one example from each model type
    kmeans_model = models['kmeans_kmeans++']
    gmm_model = models['gmm_spherical_kmeans']
    
    from sklearn.decomposition import PCA
    X_cpu = data['X_test'].cpu().numpy()
    y_cpu = data['y_test'].cpu().numpy()
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cpu)
    
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    
    # K-means visualization
    kmeans_labels = kmeans_model.predict(data['X_test']).cpu().numpy()
    scatter1 = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.6)
    ax4.set_title('K-means++ Clustering')
    ax4.set_xlabel('First Principal Component')
    ax4.set_ylabel('Second Principal Component')
    plt.colorbar(scatter1, ax=ax4)
    
    # GMM visualization
    gmm_labels = gmm_model.predict(data['X_test']).cpu().numpy()
    scatter2 = ax5.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='tab10', alpha=0.6)
    ax5.set_title('GMM Clustering (Spherical, K-means init)')
    ax5.set_xlabel('First Principal Component')
    ax5.set_ylabel('Second Principal Component')
    plt.colorbar(scatter2, ax=ax5)
    
    # 4. Add explanatory text (Bottom Row)
    explanation = """
    How to Read This Figure:
    
    1. Model Performance (Top Row):
       - Left: Accuracy comparison shows both training and test accuracies for different models
       - Right: Training time comparison reveals computational efficiency
       - K-means++ generally achieves better initialization than random
       - GMM with spherical covariance shows competitive performance
    
    2. Convergence Analysis (Middle Row):
       - Shows how quickly each model converges to a stable solution
       - Steeper slopes indicate faster convergence
       - K-means typically converges faster than GMM
       - GMM's log-likelihood provides a probabilistic measure of fit
    
    3. Cluster Visualization (Bottom Row):
       - 2D projection of high-dimensional MNIST data using PCA
       - Left: K-means++ clustering shows clear cluster boundaries
       - Right: GMM clustering reveals more nuanced cluster assignments
       - Colors represent different digit clusters (0-9)
    
    Key Observations:
    - K-means++ provides more stable and better-quality clusters than random initialization
    - GMM offers probabilistic cluster assignments but requires longer training time
    - Both methods successfully identify the underlying digit clusters in the MNIST dataset
    """
    
    fig.text(0.1, 0.02, explanation, fontsize=12, wrap=True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the text
    plt.savefig(Path(save_dir) / 'comprehensive_report.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_visualization_report(kmeans_results, gmm_results, models, data, save_dir):
    """Create comprehensive visualization report"""
    # 确保输出目录存在
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建综合报告图
    create_comprehensive_figure(kmeans_results, gmm_results, models, data, save_dir)
    
    # 创建单独的详细图
    # 1. 模型性能对比
    plot_accuracy_comparison(kmeans_results, gmm_results, save_dir)
    
    # 2. 每个模型的聚类大小分布
    for name, model in models.items():
        plot_cluster_sizes(model, data['X_test'], save_dir, name)
    
    # 3. 收敛曲线
    if (save_dir / 'training_log.json').exists():
        plot_convergence(save_dir / 'training_log.json', save_dir)
    
    # 4. 聚类可视化
    for name, model in models.items():
        plot_cluster_visualization(model, data['X_test'], data['y_test'], save_dir, name)
    
    # 5. 混淆矩阵
    for name, model in models.items():
        plot_confusion_matrix(model, data['X_test'], data['y_test'], save_dir, name) 