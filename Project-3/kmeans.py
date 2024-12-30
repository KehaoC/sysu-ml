import torch
import numpy as np
from tqdm import tqdm
from utils import process_in_batches, get_device

class KMeans:
    def __init__(self, n_clusters=10, max_iters=100, init_method='random', batch_size=1000, device=None, tol=1e-4):
        """
        Parameters:
        -----------
        n_clusters : int, default=10
            Number of clusters
        max_iters : int, default=100
            Maximum number of iterations
        init_method : str, default='random'
            Initialization method ('random' or 'kmeans++')
        batch_size : int, default=1000
            Size of batches for processing large datasets
        device : torch.device, default=None
            Device to run the computations on
        tol : float, default=1e-4
            Tolerance for convergence
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.init_method = init_method
        self.batch_size = batch_size
        self.device = device if device is not None else get_device()
        self.tol = tol
        self.centroids = None
        self.labels = None
        
    def _init_centroids(self, X):
        """Initialize centroids based on the specified method"""
        X = X.to(self.device)
        if self.init_method == 'random':
            self._init_random(X)
        elif self.init_method == 'kmeans++':
            self._init_kmeans_plus_plus(X)
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")
        
    def _init_random(self, X):
        """Random initialization of centroids"""
        n_samples = X.shape[0]
        idx = torch.randperm(n_samples, device=self.device)[:self.n_clusters]
        self.centroids = X[idx].clone()
        
    def _init_kmeans_plus_plus(self, X):
        """K-means++ initialization of centroids"""
        n_samples = X.shape[0]
        
        # Choose first centroid randomly
        idx = torch.randint(0, n_samples, (1,), device=self.device)
        self.centroids = X[idx].clone()
        
        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            # Calculate distances to nearest centroid for each point using batches
            distances = torch.cat(process_in_batches(
                X, 
                self.batch_size,
                lambda x: torch.min(torch.cdist(x, self.centroids), dim=1)[0]
            ))
            
            # Calculate probabilities proportional to squared distances
            probs = distances ** 2
            probs /= probs.sum()
            
            # Choose next centroid
            new_centroid_idx = torch.multinomial(probs, 1)
            self.centroids = torch.cat([self.centroids, X[new_centroid_idx]])
    
    def _compute_distances(self, X):
        """Compute distances between points and centroids using batches"""
        distances = torch.cat(process_in_batches(
            X,
            self.batch_size,
            lambda x: torch.cdist(x, self.centroids)
        ))
        return distances
    
    def fit(self, X):
        """
        Fit K-means clustering to the data
        
        Parameters:
        -----------
        X : torch.Tensor of shape (n_samples, n_features)
            Training data
        """
        X = X.to(self.device)
        self._init_centroids(X)
        self.inertia_history = []
        
        prev_centroids = None
        for _ in range(self.max_iters):
            # Assign points to clusters
            distances = self._compute_distances(X)
            labels = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = torch.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                mask = (labels == k)
                if torch.sum(mask) > 0:
                    new_centroids[k] = torch.mean(X[mask], dim=0)
                else:
                    # Handle empty clusters
                    new_centroids[k] = self.centroids[k]
            
            # Calculate inertia (within-cluster sum of squares)
            inertia = torch.sum(torch.min(distances, dim=1)[0])
            self.inertia_history.append(float(inertia))
            
            # Check convergence
            if prev_centroids is not None:
                diff = torch.max(torch.abs(new_centroids - prev_centroids))
                if diff < self.tol:
                    break
            
            self.centroids = new_centroids
            prev_centroids = new_centroids.clone()
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for X
        
        Parameters:
        -----------
        X : torch.Tensor of shape (n_samples, n_features)
            New data to predict
            
        Returns:
        --------
        labels : torch.Tensor of shape (n_samples,)
            Predicted cluster labels
        """
        X = X.to(self.device)
        distances = self._compute_distances(X)
        return torch.argmin(distances, dim=1) 