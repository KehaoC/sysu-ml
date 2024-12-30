import torch
import torchvision
import numpy as np
from scipy.optimize import linear_sum_assignment

def get_device():
    """Get the best available device (MPS for M1/M2 Macs, CUDA for NVIDIA GPUs, or CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_mnist(train=True, device=None):
    """
    Load MNIST dataset
    
    Parameters:
    -----------
    train : bool, default=True
        If True, load training set, else load test set
    device : torch.device, default=None
        Device to load the data to
        
    Returns:
    --------
    data : torch.Tensor of shape (n_samples, n_features)
        Flattened images
    targets : torch.Tensor of shape (n_samples,)
        True labels
    """
    if device is None:
        device = get_device()
        
    dataset = torchvision.datasets.MNIST(
        root='./data',
        train=train,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    
    data = dataset.data.float() / 255.0  # Normalize to [0, 1]
    data = data.reshape(len(dataset), -1)  # Flatten images
    return data.to(device), dataset.targets.to(device)

def process_in_batches(X, batch_size=1000, func=None):
    """Process data in batches to avoid memory issues
    
    Parameters:
    -----------
    X : torch.Tensor
        Input data
    batch_size : int
        Size of each batch
    func : callable
        Function to apply to each batch
        
    Returns:
    --------
    results : list
        List of results for each batch
    """
    results = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        if func is not None:
            result = func(batch)
            results.append(result)
    return results

def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy using the Hungarian algorithm
    
    Parameters:
    -----------
    y_true : torch.Tensor of shape (n_samples,)
        True labels
    y_pred : torch.Tensor of shape (n_samples,)
        Predicted cluster labels
        
    Returns:
    --------
    accuracy : float
        Clustering accuracy
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # Create contingency matrix
    n_classes = max(np.max(y_true), np.max(y_pred)) + 1
    contingency = np.zeros((n_classes, n_classes))
    for i in range(len(y_true)):
        contingency[y_true[i], y_pred[i]] += 1
    
    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(-contingency)
    
    # Calculate accuracy
    accuracy = contingency[row_ind, col_ind].sum() / len(y_true)
    return accuracy 