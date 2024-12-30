import torch
import time
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
import numpy as np

from kmeans import KMeans
from gmm import GMM
from utils import load_mnist, clustering_accuracy, get_device
from visualization import create_visualization_report

def run_kmeans_experiment(init_method, X_train, y_train, X_test, y_test, device):
    """Run a single K-means experiment"""
    # Train model
    kmeans = KMeans(n_clusters=10, init_method=init_method, device=device)
    
    start_time = time.time()
    kmeans.fit(X_train)
    train_time = time.time() - start_time
    
    # Evaluate
    train_labels = kmeans.predict(X_train)
    test_labels = kmeans.predict(X_test)
    
    train_acc = clustering_accuracy(y_train, train_labels)
    test_acc = clustering_accuracy(y_test, test_labels)
    
    return {
        'model': kmeans,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_time': train_time,
        'convergence': kmeans.inertia_history if hasattr(kmeans, 'inertia_history') else None
    }

def run_gmm_experiment(config, X_train, y_train, X_test, y_test, device):
    """Run a single GMM experiment"""
    cov_type, init_method = config
    
    # Train model
    gmm = GMM(
        n_components=10,
        covariance_type=cov_type,
        init_method=init_method,
        device=device,
        max_iters=100,
        tol=1e-5
    )
    
    start_time = time.time()
    gmm.fit(X_train)
    train_time = time.time() - start_time
    
    # Evaluate
    train_labels = gmm.predict(X_train)
    test_labels = gmm.predict(X_test)
    
    train_acc = clustering_accuracy(y_train, train_labels)
    test_acc = clustering_accuracy(y_test, test_labels)
    
    return {
        'model': gmm,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_time': train_time,
        'convergence': gmm.log_likelihood_history if hasattr(gmm, 'log_likelihood_history') else None
    }

def save_training_log(results, save_dir):
    """Save training logs to JSON file"""
    log_data = {}
    
    # Process K-means results
    for init_method, result in results['kmeans'].items():
        if 'convergence' in result and result['convergence'] is not None:
            # Convert tensor to list if necessary
            convergence = result['convergence']
            if isinstance(convergence, torch.Tensor):
                convergence = convergence.cpu().numpy().tolist()
            elif isinstance(convergence, np.ndarray):
                convergence = convergence.tolist()
            
            log_data[f'kmeans_{init_method}'] = {
                'convergence': convergence,
                'train_acc': float(result['train_acc']),
                'test_acc': float(result['test_acc']),
                'train_time': float(result['train_time'])
            }
    
    # Process GMM results
    for config, result in results['gmm'].items():
        if 'convergence' in result and result['convergence'] is not None:
            # Convert tensor to list if necessary
            convergence = result['convergence']
            if isinstance(convergence, torch.Tensor):
                convergence = convergence.cpu().numpy().tolist()
            elif isinstance(convergence, np.ndarray):
                convergence = convergence.tolist()
            
            log_data[f'gmm_{config}'] = {
                'convergence': convergence,
                'train_acc': float(result['train_acc']),
                'test_acc': float(result['test_acc']),
                'train_time': float(result['train_time'])
            }
    
    # Save to file
    log_file = Path(save_dir) / 'training_log.json'
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)

def run_experiments(output_dir):
    """Run all experiments and generate visualizations"""
    device = get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading MNIST dataset...")
    X_train, y_train = load_mnist(train=True, device=device)
    X_test, y_test = load_mnist(train=False, device=device)
    
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }
    
    # Run K-means experiments
    print("\nRunning K-means experiments...")
    kmeans_results = {}
    init_methods = ['random', 'kmeans++']
    
    with tqdm(total=len(init_methods), desc="K-means") as pbar:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for init_method in init_methods:
                future = executor.submit(
                    run_kmeans_experiment,
                    init_method,
                    X_train, y_train,
                    X_test, y_test,
                    device
                )
                futures.append((init_method, future))
            
            for init_method, future in futures:
                result = future.result()
                kmeans_results[init_method] = result
                acc = result['test_acc']
                pbar.set_postfix({'method': init_method, 'acc': f'{acc:.4f}'})
                pbar.update(1)
    
    # Run GMM experiments
    print("\nRunning GMM experiments...")
    gmm_results = {}
    configs = [
        ('spherical', 'random'), ('spherical', 'kmeans'),
        ('diag', 'random'), ('diag', 'kmeans'),
        ('full', 'random'), ('full', 'kmeans')
    ]
    
    with tqdm(total=len(configs), desc="GMM") as pbar:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for config in configs:
                future = executor.submit(
                    run_gmm_experiment,
                    config,
                    X_train, y_train,
                    X_test, y_test,
                    device
                )
                futures.append((f"{config[0]}_{config[1]}", future))
            
            for config_name, future in futures:
                result = future.result()
                gmm_results[config_name] = result
                acc = result['test_acc']
                pbar.set_postfix({'config': config_name, 'acc': f'{acc:.4f}'})
                pbar.update(1)
    
    # Prepare results for visualization
    results = {
        'kmeans': kmeans_results,
        'gmm': gmm_results
    }
    
    # Save training logs
    save_training_log(results, output_dir)
    
    # Extract models for visualization
    models = {}
    for init_method, result in kmeans_results.items():
        models[f'kmeans_{init_method}'] = result['model']
    for config, result in gmm_results.items():
        models[f'gmm_{config}'] = result['model']
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualization_report(
        {k: {k2: v2 for k2, v2 in v.items() if k2 != 'model'} for k, v in kmeans_results.items()},
        {k: {k2: v2 for k2, v2 in v.items() if k2 != 'model'} for k, v in gmm_results.items()},
        models,
        data,
        output_dir
    )
    
    print(f"\nExperiments completed. Results saved to {output_dir}")
    return results 