import torch
import numpy as np
from tqdm import tqdm
from utils import process_in_batches, get_device

class GMM:
    def __init__(self, n_components=10, max_iters=100, tol=1e-5, 
                 covariance_type='full', init_method='random', batch_size=2000, device=None):
        """
        Parameters:
        -----------
        n_components : int, default=10
            Number of mixture components
        max_iters : int, default=100
            Maximum number of iterations
        tol : float, default=1e-5
            Convergence threshold
        covariance_type : str, default='full'
            Type of covariance parameters:
            'spherical' - single variance per component
            'diag' - diagonal covariance matrix
            'full' - full covariance matrix
        init_method : str, default='random'
            Initialization method ('random' or 'kmeans')
        batch_size : int, default=2000
            Size of batches for processing large datasets
        device : torch.device, default=None
            Device to run the computations on
        """
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.covariance_type = covariance_type
        self.init_method = init_method
        self.batch_size = batch_size
        self.device = device if device is not None else get_device()
        
        # Model parameters
        self.weights = None  # mixture weights
        self.means = None    # component means
        self.covs = None     # component covariances
        
        # 数值稳定性参数
        self._eps = 1e-6  # 基础epsilon值
        self._min_covar = 1e-7  # 最小协方差值
        self._max_covar = 1e7   # 最大协方差值
        self._min_weight = 1e-7  # 最小权重值
        
        # 缓存
        self._log_det_cache = None
        self._inv_cov_cache = None
        self._const_cache = None
        
        # 设备相关
        self.use_cpu_fallback = self.device.type == 'mps'
        if self.use_cpu_fallback:
            self.cpu_device = torch.device('cpu')
    
    def _init_parameters(self, X):
        """Initialize GMM parameters with improved numerical stability"""
        n_samples, n_features = X.shape
        
        # 计算数据的基本统计量
        data_mean = torch.mean(X, dim=0)
        data_var = torch.var(X, dim=0)
        
        # 确保方差不会太小
        data_var = torch.clamp(data_var, min=self._min_covar)
        
        # 初始化权重（添加小的扰动避免完全相等）
        self.weights = (torch.ones(self.n_components, device=self.device) / self.n_components +
                       torch.rand(self.n_components, device=self.device) * self._eps)
        self.weights = self.weights / self.weights.sum()  # 重新归一化
        
        if self.init_method == 'random':
            # 随机初始化均值，但确保它们不会太远离数据中心
            idx = torch.randperm(n_samples, device=self.device)[:self.n_components]
            self.means = X[idx].clone()
            # 添加小的随机扰动
            self.means += torch.randn_like(self.means) * torch.sqrt(data_var) * 0.1
        else:  # kmeans initialization
            from kmeans import KMeans
            kmeans = KMeans(n_clusters=self.n_components, init_method='kmeans++', device=self.device)
            kmeans.fit(X)
            self.means = kmeans.centroids
        
        # 初始化协方差，使用数据的实际方差
        if self.covariance_type == 'spherical':
            base_var = torch.mean(data_var)  # 使用平均方差
            self.covs = torch.full((self.n_components,), base_var, device=self.device)
        elif self.covariance_type == 'diag':
            self.covs = data_var.repeat(self.n_components, 1)
        else:  # full
            # 使用数据协方差矩阵的缩放版本
            data_cov = torch.cov(X.T)
            # 确保协方差矩阵是正定的
            if self.use_cpu_fallback:
                min_eig = torch.linalg.eigvalsh(data_cov.cpu()).min().to(self.device)
            else:
                min_eig = torch.linalg.eigvalsh(data_cov).min()
            
            if min_eig < self._min_covar:
                data_cov.diagonal().add_(self._min_covar - min_eig)
            self.covs = torch.stack([data_cov.clone() for _ in range(self.n_components)])
        
        # 应用协方差限制
        self._clamp_covariances()
        
        # 初始化缓存
        self._update_cov_cache()
        
    def _clamp_covariances(self):
        """限制协方差值在合理范围内"""
        if self.covariance_type == 'spherical':
            self.covs.clamp_(self._min_covar, self._max_covar)
        elif self.covariance_type == 'diag':
            self.covs.clamp_(self._min_covar, self._max_covar)
        else:  # full
            for k in range(self.n_components):
                # 确保对角线元素不会太小
                self.covs[k].diagonal().clamp_(self._min_covar, self._max_covar)
                # 确保矩阵是正定的
                try:
                    if self.use_cpu_fallback:
                        eigvals = torch.linalg.eigvalsh(self.covs[k].cpu())
                        min_eig = eigvals.min().to(self.device)
                    else:
                        eigvals = torch.linalg.eigvalsh(self.covs[k])
                        min_eig = eigvals.min()
                        
                    if min_eig < self._min_covar:
                        self.covs[k].diagonal().add_(self._min_covar - min_eig)
                except RuntimeError:
                    # 如果特征值计算失败，添加更大的正则化
                    self.covs[k].diagonal().add_(self._min_covar * 10)

    def _update_cov_cache(self):
        """更新协方差矩阵的缓存值，包含更好的数值稳定性检查"""
        n_features = self.means.shape[1]
        self._log_det_cache = torch.zeros(self.n_components, device=self.device)
        self._const_cache = -0.5 * n_features * np.log(2 * np.pi)
        
        if self.covariance_type == 'full':
            self._inv_cov_cache = torch.zeros(self.n_components, n_features, n_features, device=self.device)
            
            for k in range(self.n_components):
                cov_k = self.covs[k].clone()
                # 添加小的对角正则化
                cov_k.diagonal().add_(self._eps)
                
                try:
                    if self.use_cpu_fallback:
                        cov_k_cpu = cov_k.cpu()
                        # 使用Cholesky分解计算行列式和逆矩阵
                        chol = torch.linalg.cholesky(cov_k_cpu)
                        self._log_det_cache[k] = 2 * torch.sum(torch.log(torch.diagonal(chol))).to(self.device)
                        self._inv_cov_cache[k] = torch.cholesky_inverse(chol).to(self.device)
                    else:
                        chol = torch.linalg.cholesky(cov_k)
                        self._log_det_cache[k] = 2 * torch.sum(torch.log(torch.diagonal(chol)))
                        self._inv_cov_cache[k] = torch.cholesky_inverse(chol)
                except RuntimeError:
                    # 如果Cholesky分解失败，回退到传统方法并添加更强的正则化
                    cov_k.diagonal().add_(self._min_covar)
                    if self.use_cpu_fallback:
                        cov_k_cpu = cov_k.cpu()
                        self._log_det_cache[k] = torch.log(torch.det(cov_k_cpu) + self._eps).to(self.device)
                        self._inv_cov_cache[k] = torch.inverse(cov_k_cpu).to(self.device)
                    else:
                        self._log_det_cache[k] = torch.log(torch.det(cov_k) + self._eps)
                        self._inv_cov_cache[k] = torch.inverse(cov_k)
        else:
            # 对于spherical和diagonal协方差，确保值不会太小
            self.covs = torch.clamp(self.covs, min=self._min_covar, max=self._max_covar)
            self._inv_cov_cache = 1.0 / self.covs
            if self.covariance_type == 'spherical':
                self._log_det_cache = n_features * torch.log(self.covs)
            else:  # diagonal
                self._log_det_cache = torch.sum(torch.log(self.covs), dim=1)

    def _e_step_batch(self, X):
        """E-step for a batch of data with improved numerical stability"""
        batch_size = X.shape[0]
        log_resp = torch.zeros(batch_size, self.n_components, device=self.device)
        log_weights = torch.log(torch.clamp(self.weights, min=self._min_weight))
        
        for k in range(self.n_components):
            diff = X - self.means[k]
            
            if self.covariance_type == 'spherical':
                log_resp[:, k] = log_weights[k] + self._const_cache - (
                    0.5 * self._log_det_cache[k] +
                    0.5 * torch.sum(diff * diff, dim=1) * self._inv_cov_cache[k]
                )
            elif self.covariance_type == 'diag':
                log_resp[:, k] = log_weights[k] + self._const_cache - (
                    0.5 * self._log_det_cache[k] +
                    0.5 * torch.sum(diff * diff * self._inv_cov_cache[k], dim=1)
                )
            else:  # full
                log_resp[:, k] = log_weights[k] + self._const_cache - (
                    0.5 * self._log_det_cache[k] +
                    0.5 * torch.sum(torch.mm(diff, self._inv_cov_cache[k]) * diff, dim=1)
                )
        
        # 使用log-sum-exp技巧进行数值稳定的归一化
        log_resp_max, _ = torch.max(log_resp, dim=1, keepdim=True)
        log_resp_norm = log_resp - log_resp_max
        resp = torch.exp(log_resp_norm)
        resp_sum = resp.sum(dim=1, keepdim=True)
        resp = resp / resp_sum
        
        # 处理数值不稳定性
        resp = torch.where(torch.isnan(resp), torch.ones_like(resp) / self.n_components, resp)
        resp = torch.where(torch.isinf(resp), torch.zeros_like(resp), resp)
        resp = resp / resp.sum(dim=1, keepdim=True)  # 再次归一化
        
        return resp
    
    def _e_step(self, X):
        """E-step: compute responsibilities using batches"""
        resp_list = process_in_batches(X, self.batch_size, self._e_step_batch)
        return torch.cat(resp_list)
    
    def _m_step(self, X, resp):
        """M-step with improved numerical stability"""
        n_samples = X.shape[0]
        resp_sum = torch.sum(resp, dim=0)
        
        # 更新权重（添加平滑）
        self.weights = (resp_sum + self._min_weight) / (n_samples + self.n_components * self._min_weight)
        self.weights = self.weights / self.weights.sum()  # 确保精确归一化
        
        # 更新均值
        resp_sum_expanded = resp_sum.unsqueeze(1)
        self.means = torch.mm(resp.T, X) / torch.clamp(resp_sum_expanded, min=self._eps)
        
        # 更新协方差
        if self.covariance_type == 'spherical':
            diff = X.unsqueeze(1) - self.means.unsqueeze(0)
            weighted_diff_sq = resp.unsqueeze(2) * (diff ** 2)
            self.covs = torch.sum(weighted_diff_sq, dim=(0, 2)) / (resp_sum * X.shape[1])
        elif self.covariance_type == 'diag':
            diff = X.unsqueeze(1) - self.means.unsqueeze(0)
            weighted_diff_sq = resp.unsqueeze(2) * (diff ** 2)
            self.covs = torch.sum(weighted_diff_sq, dim=0) / torch.clamp(resp_sum_expanded, min=self._eps)
        else:  # full
            diff = X.unsqueeze(1) - self.means.unsqueeze(0)
            weighted_diff = resp.unsqueeze(2) * diff
            for k in range(self.n_components):
                self.covs[k] = torch.mm(weighted_diff[:, k].T, diff[:, k]) / torch.clamp(resp_sum[k], min=self._eps)
        
        # 应用协方差限制和更新缓存
        self._clamp_covariances()
        self._update_cov_cache()
    
    def fit(self, X):
        """训练GMM模型，包含改进的收敛检查和错误处理"""
        X = X.to(self.device)
        self._init_parameters(X)
        
        prev_log_likelihood = -float('inf')
        best_params = None
        best_log_likelihood = -float('inf')
        no_improvement_count = 0
        self.log_likelihood_history = []
        
        for iter_idx in range(self.max_iters):
            try:
                # E-step
                resp = self._e_step(X)
                
                # M-step
                self._m_step(X, resp)
                
                # 计算对数似然
                log_likelihood = self._compute_log_likelihood(X)
                self.log_likelihood_history.append(float(log_likelihood))
                
                # 检查是否是最佳结果
                if log_likelihood > best_log_likelihood and not torch.isnan(log_likelihood):
                    best_log_likelihood = log_likelihood
                    best_params = {
                        'weights': self.weights.clone(),
                        'means': self.means.clone(),
                        'covs': self.covs.clone(),
                        'log_det_cache': self._log_det_cache.clone(),
                        'inv_cov_cache': self._inv_cov_cache.clone()
                    }
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # 计算改进
                improvement = log_likelihood - prev_log_likelihood
                
                # 检查数值问题
                if torch.isnan(log_likelihood) or torch.isinf(log_likelihood):
                    if best_params is not None:
                        self._restore_parameters(best_params)
                    break
                
                # 检查收敛
                if abs(improvement) < self.tol:
                    break
                
                # 检查是否长时间没有改进
                if no_improvement_count >= 10:
                    if best_params is not None:
                        self._restore_parameters(best_params)
                    break
                
                prev_log_likelihood = log_likelihood
                
            except RuntimeError as e:
                if best_params is not None:
                    self._restore_parameters(best_params)
                break
        
        # 确保使用最佳参数
        if best_params is not None:
            self._restore_parameters(best_params)
        
        return self
    
    def _compute_log_likelihood(self, X):
        """计算对数似然，使用批处理和改进的数值稳定性"""
        log_likelihood = 0.0
        n_batches = (X.shape[0] + self.batch_size - 1) // self.batch_size
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, X.shape[0])
            batch = X[start_idx:end_idx]
            
            # 计算每个样本对每个组件的对数似然
            log_probs = torch.zeros(batch.shape[0], self.n_components, device=self.device)
            
            for k in range(self.n_components):
                diff = batch - self.means[k]
                if self.covariance_type == 'spherical':
                    log_probs[:, k] = (
                        torch.log(self.weights[k] + self._eps) +
                        self._const_cache -
                        0.5 * (self._log_det_cache[k] + 
                              torch.sum(diff * diff, dim=1) * self._inv_cov_cache[k])
                    )
                elif self.covariance_type == 'diag':
                    log_probs[:, k] = (
                        torch.log(self.weights[k] + self._eps) +
                        self._const_cache -
                        0.5 * (self._log_det_cache[k] + 
                              torch.sum(diff * diff * self._inv_cov_cache[k], dim=1))
                    )
                else:  # full
                    log_probs[:, k] = (
                        torch.log(self.weights[k] + self._eps) +
                        self._const_cache -
                        0.5 * (self._log_det_cache[k] + 
                              torch.sum(torch.mm(diff, self._inv_cov_cache[k]) * diff, dim=1))
                    )
            
            # 使用log-sum-exp技巧计算批次的对数似然
            batch_log_likelihood = torch.logsumexp(log_probs, dim=1)
            log_likelihood += torch.sum(batch_log_likelihood)
        
        return log_likelihood / X.shape[0]  # 返回平均对数似然
    
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
        resp = self._e_step(X)
        return torch.argmax(resp, dim=1) 
    
    def _restore_parameters(self, params):
        """恢复模型参数"""
        self.weights = params['weights']
        self.means = params['means']
        self.covs = params['covs']
        self._log_det_cache = params['log_det_cache']
        self._inv_cov_cache = params['inv_cov_cache'] 