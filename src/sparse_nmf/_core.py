"""
Sparse Non-negative Matrix Factorization (NMF) module with GPU acceleration.

This module provides GPU-accelerated NMF that works directly with sparse matrices,
avoiding the need to convert large sparse matrices to dense format (which can be
memory-prohibitive for very sparse data).

The implementation uses PyTorch sparse tensors and processes data in batches to
maintain memory efficiency while leveraging GPU acceleration.

Two modes are available:
1. **Standalone NMF**: Use SparseNMF for dimensionality reduction, then pass to autoencoder
2. **Joint Model**: Use SparseNMF_Autoencoder for end-to-end training (recommended)

Examples:
---------
Standalone NMF:
    >>> from sparse_nmf import SparseNMF
    >>> from scipy.sparse import csr_matrix
    >>> 
    >>> # Create a sparse matrix (e.g., from gene associations)
    >>> X_sparse = csr_matrix((data, (rows, cols)), shape=(n_samples, n_features))
    >>> 
    >>> # Fit NMF model
    >>> nmf = SparseNMF(n_components=256, max_iter=500, device='cuda:0')
    >>> X_reduced = nmf.fit_transform(X_sparse)
    >>> 
    >>> # X_reduced is now (n_samples, 256) dense array ready for autoencoder

Joint Model (Recommended):
    >>> from sparse_nmf import train_joint_model
    >>> 
    >>> # Train joint model end-to-end
    >>> z, model = train_joint_model(
    ...     X_sparse,
    ...     n_samples=114171,
    ...     n_features=31263,
    ...     nmf_components=256,
    ...     latent_dim=2,
    ...     device='cuda:0',
    ...     n_epochs=100
    ... )
    >>> 
    >>> # z is now (n_samples, 2) 2D embeddings
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
from scipy.sparse import spmatrix, csr_matrix, coo_matrix
import torch
import torch.nn as nn
from tqdm import tqdm


def _compute_recon_values_chunked(
    W_rows: torch.Tensor,
    H: torch.Tensor,
    col_idx: torch.Tensor,
    chunk_size: int = 50000,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute reconstructed values (W_rows * H_cols).sum(dim=1) in chunks to avoid OOM.
    
    Uses a memory-efficient approach that processes the sum in sub-chunks to avoid
    creating large intermediate tensors.
    
    Parameters
    ----------
    W_rows : torch.Tensor
        W matrix rows of shape (nnz, n_components).
    H : torch.Tensor
        H matrix of shape (n_components, n_features).
    col_idx : torch.Tensor
        Column indices of shape (nnz,).
    chunk_size : int
        Maximum number of entries to process at once.
    device : torch.device, optional
        Device to use for clearing cache. If None, uses W_rows.device.
        
    Returns
    -------
    torch.Tensor
        Reconstructed values of shape (nnz,).
    """
    nnz = W_rows.shape[0]
    n_components = H.shape[0]
    if device is None:
        device = W_rows.device
    
    # For very large n_components, use much smaller chunks
    # Estimate memory: chunk_size * n_components * 4 bytes (float32) * 2 (for intermediate)
    # Try to keep chunks under ~50MB for large n_components
    estimated_memory_per_entry = n_components * 4 * 2  # bytes (with intermediate)
    max_chunk_memory = 50 * 1024 * 1024  # 50MB (more conservative)
    adaptive_chunk_size = min(chunk_size, max_chunk_memory // estimated_memory_per_entry)
    adaptive_chunk_size = max(500, adaptive_chunk_size)  # At least 500 entries
    
    # For very large n_components, use even smaller chunks
    if n_components > 512:
        adaptive_chunk_size = min(adaptive_chunk_size, 5000)
    
    if nnz <= adaptive_chunk_size:
        # Small enough to process all at once, but still use memory-efficient approach
        # Use unique columns to reduce memory
        unique_cols, inverse_idx = torch.unique(col_idx, return_inverse=True)
        H_cols_unique = H[:, unique_cols].t()  # (n_unique, n_components)
        H_cols = H_cols_unique[inverse_idx]  # (nnz, n_components)
        
        # For large n_components, compute sum in sub-chunks to avoid large intermediate tensor
        if n_components > 512:
            # Process sum in sub-chunks across components dimension
            result = torch.zeros(nnz, device=W_rows.device, dtype=W_rows.dtype)
            component_chunk_size = 256
            for comp_start in range(0, n_components, component_chunk_size):
                comp_end = min(comp_start + component_chunk_size, n_components)
                comp_idx = slice(comp_start, comp_end)
                result += (W_rows[:, comp_idx] * H_cols[:, comp_idx]).sum(dim=1)
                if device.type == 'cuda' and comp_start % (component_chunk_size * 4) == 0:
                    torch.cuda.empty_cache()
        else:
            result = (W_rows * H_cols).sum(dim=1)
        
        del H_cols_unique, H_cols, unique_cols, inverse_idx
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return result
    
    # Process in chunks
    X_recon_values = torch.zeros(nnz, device=W_rows.device, dtype=W_rows.dtype)
    for chunk_start in range(0, nnz, adaptive_chunk_size):
        chunk_end = min(chunk_start + adaptive_chunk_size, nnz)
        chunk_idx = slice(chunk_start, chunk_end)
        
        W_rows_chunk = W_rows[chunk_idx]
        col_idx_chunk = col_idx[chunk_idx]
        
        # Get unique columns to reduce memory
        unique_cols, inverse_idx = torch.unique(col_idx_chunk, return_inverse=True)
        H_cols_unique = H[:, unique_cols].t()  # (n_unique, n_components)
        H_cols_chunk = H_cols_unique[inverse_idx]  # (chunk_size, n_components)
        
        # For large n_components, compute sum in sub-chunks
        if n_components > 512:
            chunk_result = torch.zeros(chunk_end - chunk_start, device=W_rows.device, dtype=W_rows.dtype)
            component_chunk_size = 256
            for comp_start in range(0, n_components, component_chunk_size):
                comp_end = min(comp_start + component_chunk_size, n_components)
                comp_idx = slice(comp_start, comp_end)
                chunk_result += (W_rows_chunk[:, comp_idx] * H_cols_chunk[:, comp_idx]).sum(dim=1)
            X_recon_values[chunk_idx] = chunk_result
            del chunk_result
        else:
            X_recon_values[chunk_idx] = (W_rows_chunk * H_cols_chunk).sum(dim=1)
        
        # Clear intermediate tensors aggressively
        del H_cols_unique, H_cols_chunk, W_rows_chunk, col_idx_chunk, unique_cols, inverse_idx
        
        # Clear cache after every chunk when memory is tight
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Final cache clear
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return X_recon_values


class SparseNMF:
    """
    GPU-accelerated Non-negative Matrix Factorization for sparse matrices.
    
    This class implements NMF using multiplicative update rules while working
    directly with sparse matrices, avoiding memory-intensive dense conversions.
    
    When R² loss is enabled (r2_weight > 0), the optimization switches to
    gradient-based Adam optimizer to handle the normalized R² loss function.
    
    Parameters
    ----------
    n_components : int, default 256
        Number of components (latent dimensions) to extract.
    max_iter : int, default 500
        Maximum number of iterations for optimization.
    device : str, default 'cuda'
        Device to use for computation ('cuda', 'cuda:0', 'cpu', etc.).
        If 'cuda' is specified but CUDA is not available, falls back to CPU.
    batch_size : int, optional
        Batch size for processing rows. If None, auto-determines based on
        GPU memory and matrix size.
    verbose : bool, default True
        Whether to print progress information.
    random_state : int, optional
        Random seed for initialization. If None, uses random initialization.
    tol : float, default 1e-4
        Tolerance for convergence checking. Training stops if change in
        reconstruction error is below this threshold.
    patience : int, optional
        Number of iterations to wait without improvement before early stopping.
        If None, only uses tolerance-based convergence. If specified, stops training
        if the error doesn't improve for `patience` consecutive iterations.
    mse_weight : float, default 1.0
        Weight for MSE (Mean Squared Error) loss component.
        total_loss = mse_weight * MSE + r2_weight * (1 - R²)
    r2_weight : float, default 0.0
        Weight for R² (coefficient of determination) loss component.
        When > 0, switches to gradient-based optimization.
        R² loss is computed as (1 - R²) so that minimizing loss maximizes R².
    learning_rate : float, default 0.01
        Learning rate for gradient-based optimizer (used when r2_weight > 0).
    nonzero_mse_weight : float, default 0.0
        Controls whether MSE loss includes zeros or only non-zero values.
        When > 0: MSE computed only on non-zero positions (ignores zeros).
        When 0: MSE computed on all positions including zeros (learns sparsity patterns).
        When > 0, forces gradient descent (multiplicative updates always include zeros).
    nonzero_r2_weight : float, default 0.0
        Controls whether R² loss includes zeros or only non-zero values.
        When > 0: R² computed only on non-zero positions (ignores zeros).
        When 0: R² computed on all positions including zeros (learns sparsity patterns).
        Only affects training when r2_weight > 0.
        For final reporting, both R² (all values) and R² (non-zero only) are always computed.
    
    Attributes
    ----------
    W : torch.Tensor
        Basis matrix of shape (n_samples, n_components).
    H : torch.Tensor
        Coefficient matrix of shape (n_components, n_features).
    reconstruction_error_ : float
        Final reconstruction error (Frobenius norm).
    r2_score_ : float
        R² (coefficient of determination) score on all values (including zeros),
        computed on a sample of the data. Values closer to 1.0 indicate better
        reconstruction quality.
    r2_score_nonzero_ : float
        R² (coefficient of determination) score on non-zero values only,
        computed on a sample of the data. More meaningful for highly sparse data.
    n_iter_ : int
        Number of iterations completed.
    
    Examples
    --------
    >>> from sparse_nmf import SparseNMF
    >>> from scipy.sparse import random
    >>> 
    >>> # Create a sparse matrix
    >>> X = random(1000, 5000, density=0.01, format='csr')
    >>> 
    >>> # Fit NMF with MSE only (default, uses multiplicative updates)
    >>> nmf = SparseNMF(n_components=128, max_iter=200, device='cuda:0')
    >>> X_reduced = nmf.fit_transform(X)
    >>> 
    >>> # Fit NMF with weighted MSE + R² loss (uses gradient descent)
    >>> nmf = SparseNMF(
    ...     n_components=128, max_iter=200, device='cuda:0',
    ...     mse_weight=0.5, r2_weight=0.5  # Equal weighting
    ... )
    >>> X_reduced = nmf.fit_transform(X)
    >>> print(f"R² score: {nmf.r2_score_:.4f}")
    """
    
    def __init__(
        self,
        n_components: int = 256,
        max_iter: int = 500,
        device: str = 'cuda',
        batch_size: Optional[int] = None,
        verbose: bool = True,
        random_state: Optional[int] = None,
        tol: float = 1e-4,
        mse_weight: float = 1.0,
        r2_weight: float = 0.0,
        learning_rate: float = 0.01,
        nonzero_mse_weight: float = 0.0,
        nonzero_r2_weight: float = 0.0,
        patience: Optional[int] = None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.tol = tol
        self.mse_weight = mse_weight
        self.r2_weight = r2_weight
        self.learning_rate = learning_rate
        self.nonzero_mse_weight = nonzero_mse_weight
        self.nonzero_r2_weight = nonzero_r2_weight
        self.patience = patience
        
        # Set device
        if device.startswith('cuda') and not torch.cuda.is_available():
            if verbose:
                print("CUDA not available, falling back to CPU")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Model parameters (set during fit)
        self.W = None
        self.H = None
        self.reconstruction_error_ = None
        self.r2_score_ = None
        self.r2_score_nonzero_ = None
        self.n_iter_ = None
        
        # Set random seed if provided
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
    
    def _sparse_to_torch(self, X_sparse: spmatrix) -> torch.Tensor:
        """
        Convert scipy sparse matrix to PyTorch sparse tensor.
        
        Parameters
        ----------
        X_sparse : scipy.sparse matrix
            Sparse matrix to convert.
            
        Returns
        -------
        torch.Tensor
            Sparse COO tensor on the specified device.
        """
        # Convert to COO format for efficient conversion
        if not isinstance(X_sparse, coo_matrix):
            coo = X_sparse.tocoo()
        else:
            coo = X_sparse
        
        indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
        values = torch.from_numpy(coo.data).float()
        shape = torch.Size(coo.shape)
        
        # Create sparse tensor on device
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, shape, device=self.device
        )
        return sparse_tensor
    
    def _sparse_matmul(self, sparse_A: torch.Tensor, dense_B: torch.Tensor) -> torch.Tensor:
        """
        Efficient sparse-dense matrix multiplication.
        
        Parameters
        ----------
        sparse_A : torch.Tensor (sparse)
            Sparse matrix.
        dense_B : torch.Tensor (dense)
            Dense matrix.
            
        Returns
        -------
        torch.Tensor
            Result of sparse_A @ dense_B.
        """
        return torch.sparse.mm(sparse_A, dense_B)
    
    def _auto_batch_size(self, n_samples: int) -> int:
        """
        Automatically determine batch size based on GPU memory and matrix size.
        
        Parameters
        ----------
        n_samples : int
            Number of samples (rows) in the matrix.
            
        Returns
        -------
        int
            Recommended batch size.
        """
        if self.device.type == 'cuda':
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                # Conservative estimate: use ~20% of GPU memory for batches
                # Rough estimate: each batch needs ~batch_size * n_components * 4 bytes
                estimated_batch_memory_gb = (self.n_components * 4) / 1e9  # per sample
                max_batch_size = int((gpu_memory_gb * 0.2) / estimated_batch_memory_gb)
                batch_size = max(1000, min(10000, max_batch_size, n_samples))
            except:
                batch_size = 5000
        else:
            batch_size = 2000  # Smaller batches for CPU
        
        return batch_size
    
    def _compute_recon_values_chunked(
        self,
        W_rows: torch.Tensor,
        H: torch.Tensor,
        col_idx: torch.Tensor,
        chunk_size: int = 50000,
    ) -> torch.Tensor:
        """
        Compute reconstructed values (W_rows * H_cols).sum(dim=1) in chunks to avoid OOM.
        
        Parameters
        ----------
        W_rows : torch.Tensor
            W matrix rows of shape (nnz, n_components).
        H : torch.Tensor
            H matrix of shape (n_components, n_features).
        col_idx : torch.Tensor
            Column indices of shape (nnz,).
        chunk_size : int
            Maximum number of entries to process at once.
            
        Returns
        -------
        torch.Tensor
            Reconstructed values of shape (nnz,).
        """
        return _compute_recon_values_chunked(W_rows, H, col_idx, chunk_size, self.device)
    
    def _compute_r2_loss(
        self,
        X_values: torch.Tensor,
        X_recon_values: torch.Tensor,
        X_mean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute R² loss (1 - R²) for optimization.
        
        R² = 1 - (SS_res / SS_tot)
        R² loss = 1 - R² = SS_res / SS_tot
        
        Parameters
        ----------
        X_values : torch.Tensor
            Original values (from sparse matrix).
        X_recon_values : torch.Tensor
            Reconstructed values.
        X_mean : torch.Tensor
            Mean of original values.
            
        Returns
        -------
        torch.Tensor
            R² loss value (1 - R²).
        """
        SS_res = torch.sum((X_values - X_recon_values) ** 2)
        SS_tot = torch.sum((X_values - X_mean) ** 2)
        r2 = 1 - (SS_res / (SS_tot + 1e-10))
        return 1 - r2  # R² loss (to minimize)
    
    def fit_transform(
        self,
        X_sparse: spmatrix,
    ) -> np.ndarray:
        """
        Fit NMF model and return transformed matrix.
        
        Parameters
        ----------
        X_sparse : scipy.sparse matrix
            Sparse input matrix of shape (n_samples, n_features).
            Must be non-negative (values >= 0).
            
        Returns
        -------
        np.ndarray
            Transformed matrix W of shape (n_samples, n_components).
            This is the reduced representation ready for further processing
            (e.g., autoencoder).
        """
        n_samples, n_features = X_sparse.shape
        
        if self.verbose:
            print(f"Sparse NMF on {self.device}")
            print(f"  Input shape: {X_sparse.shape}")
            print(f"  Sparsity: {(1 - X_sparse.nnz / (n_samples * n_features)):.2%}")
            print(f"  Components: {self.n_components}")
            print(f"  Max iterations: {self.max_iter}")
            print(f"  Loss weights: MSE={self.mse_weight}, R²={self.r2_weight}, nonzero_MSE={self.nonzero_mse_weight}, nonzero_R²={self.nonzero_r2_weight}")
            if self.r2_weight > 0 or self.nonzero_mse_weight > 0 or self.nonzero_r2_weight > 0:
                print(f"  Using gradient-based optimization")
                if self.r2_weight > 0:
                    print(f"    R² loss enabled")
                if self.nonzero_mse_weight > 0:
                    print(f"    MSE computed on non-zero values only (ignores zeros)")
                elif self.mse_weight > 0:
                    print(f"    MSE computed on all values (including zeros, learns sparsity)")
                if self.r2_weight > 0:
                    if self.nonzero_r2_weight > 0:
                        print(f"    R² computed on non-zero values only (ignores zeros)")
                    else:
                        print(f"    R² computed on all values (including zeros, learns sparsity)")
                print(f"  Learning rate: {self.learning_rate}")
        
        # Check for negative values
        if (X_sparse.data < 0).any():
            if self.verbose:
                print("Warning: Input matrix contains negative values. Taking absolute value.")
            X_sparse = X_sparse.copy()
            X_sparse.data = np.abs(X_sparse.data)
        
        # Initialize W and H with random positive values
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        
        W = torch.rand(n_samples, self.n_components, device=self.device) * 0.1 + 0.01
        H = torch.rand(self.n_components, n_features, device=self.device) * 0.1 + 0.01
        
        # Convert sparse matrix to PyTorch sparse tensor
        if self.verbose:
            print(f"Converting sparse matrix to PyTorch sparse tensor...")
        X_sparse_torch = self._sparse_to_torch(X_sparse)
        
        # Determine batch size
        if self.batch_size is None:
            self.batch_size = self._auto_batch_size(n_samples)
        
        if self.verbose:
            print(f"Using batch size: {self.batch_size}")
            print(f"Training NMF...")
        
        # Convert to CSR for efficient row slicing
        X_csr = X_sparse.tocsr()
        
        # Choose optimization strategy based on loss weights
        # If nonzero_mse_weight > 0 or nonzero_r2_weight > 0, we need gradient descent to control which positions are included
        # (multiplicative updates always optimize full Frobenius norm including zeros)
        use_gradient_descent = self.r2_weight > 0 or self.nonzero_mse_weight > 0 or self.nonzero_r2_weight > 0
        
        if use_gradient_descent:
            # Gradient-based optimization for weighted MSE + R² loss
            W, H, iteration = self._fit_gradient_descent(
                X_sparse, X_csr, W, H, n_samples, n_features
            )
        else:
            # Multiplicative updates (efficient for MSE-only)
            W, H, iteration = self._fit_multiplicative_updates(
                X_sparse, X_csr, W, H, n_samples, n_features
            )
        
        # Store model parameters
        self.W = W
        self.H = H
        self.n_iter_ = iteration + 1
        
        # Compute final reconstruction error and R²
        self._compute_final_metrics(X_csr, W, H, n_samples)
        
        # Return W as numpy array
        return W.cpu().numpy()
    
    def _fit_multiplicative_updates(
        self,
        X_sparse: spmatrix,
        X_csr: spmatrix,
        W: torch.Tensor,
        H: torch.Tensor,
        n_samples: int,
        n_features: int,
    ) -> tuple:
        """
        Fit NMF using multiplicative update rules (efficient for MSE-only loss).
        """
        prev_error = None
        best_error = None
        patience_counter = 0
        error_report_interval = max(1, min(10, self.max_iter // 20))
        iterator = tqdm(range(self.max_iter), disable=not self.verbose, desc="NMF iterations (multiplicative)")
        
        for iteration in iterator:
            # Update H: H = H * (W^T @ X) / (W^T @ W @ H)
            WTX = torch.zeros(n_features, self.n_components, device=self.device)
            
            for i in range(0, n_samples, self.batch_size):
                end = min(i + self.batch_size, n_samples)
                X_batch_sparse = X_csr[i:end]
                if X_batch_sparse.nnz > 0:
                    X_batch_torch = self._sparse_to_torch(X_batch_sparse)
                    WTX += self._sparse_matmul(X_batch_torch.t(), W[i:end])
            
            WTW = torch.mm(W.t(), W)
            WTWH = torch.mm(WTW, H)
            H = H * (WTX.t() / (WTWH + 1e-10))
            H = torch.clamp(H, min=1e-10)
            
            # Update W: W = W * (X @ H^T) / (W @ H @ H^T)
            for i in range(0, n_samples, self.batch_size):
                end = min(i + self.batch_size, n_samples)
                X_batch_sparse = X_csr[i:end]
                
                if X_batch_sparse.nnz > 0:
                    X_batch_torch = self._sparse_to_torch(X_batch_sparse)
                    W_batch = W[i:end]
                    XHT_batch = self._sparse_matmul(X_batch_torch, H.t())
                    WHHT_batch = torch.mm(torch.mm(W_batch, H), H.t())
                    W[i:end] = W_batch * (XHT_batch / (WHHT_batch + 1e-10))
            
            W = torch.clamp(W, min=1e-10)
            
            # Monitor progress
            if self.verbose and ((iteration + 1) % error_report_interval == 0 or iteration == 0):
                error, prev_error, converged = self._check_convergence(
                    X_csr, W, H, n_samples, prev_error, iterator, iteration, 
                    patience=self.patience, best_error=best_error, patience_counter=patience_counter
                )
                if converged:
                    break
                
                # Check patience-based early stopping
                if self.patience is not None:
                    if best_error is None or error < best_error:
                        best_error = error
                        patience_counter = 0
                    else:
                        patience_counter += error_report_interval
                    
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"\nEarly stopping: no improvement for {self.patience} iterations (best error: {best_error:.6f})")
                        break
        
        return W, H, iteration
    
    def _fit_gradient_descent(
        self,
        X_sparse: spmatrix,
        X_csr: spmatrix,
        W: torch.Tensor,
        H: torch.Tensor,
        n_samples: int,
        n_features: int,
    ) -> tuple:
        """
        Fit NMF using gradient descent (supports weighted MSE + R² loss).
        """
        import torch.nn.functional as F
        
        # Clear CUDA cache aggressively to free up memory before training
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            # Force garbage collection and clear cache multiple times
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Make W and H require gradients
        W = W.requires_grad_(True)
        H = H.requires_grad_(True)
        
        # Use Adam optimizer
        optimizer = torch.optim.Adam([W, H], lr=self.learning_rate)
        
        # Patience tracking for gradient descent
        best_loss = None
        patience_counter = 0
        
        # Compute global mean for R² calculation
        # If nonzero_r2_weight > 0, use mean of non-zero values only
        # If nonzero_r2_weight = 0, use mean of all values (including zeros)
        if self.r2_weight > 0:
            if self.nonzero_r2_weight > 0:
                X_mean = torch.tensor(X_sparse.data.mean(), device=self.device, dtype=torch.float32)
            else:
                # Mean of all values including zeros (sparsity-aware)
                total_sum = X_sparse.data.sum()
                total_elements = X_sparse.shape[0] * X_sparse.shape[1]
                X_mean = torch.tensor(total_sum / total_elements, device=self.device, dtype=torch.float32)
        else:
            # Not used if r2_weight = 0, but initialize to avoid errors
            X_mean = torch.tensor(X_sparse.data.mean(), device=self.device, dtype=torch.float32)
        
        prev_loss = None
        loss_report_interval = max(1, min(10, self.max_iter // 20))
        iterator = tqdm(range(self.max_iter), disable=not self.verbose, desc="NMF iterations (gradient)")
        
        # Determine effective MSE weight: use nonzero_mse_weight as the weight if mse_weight is 0
        # This allows users to set mse_weight=0 and nonzero_mse_weight=1.0 to train with non-zero MSE only
        effective_mse_weight = self.mse_weight if self.mse_weight > 0 else self.nonzero_mse_weight
        
        # Similarly for R² weight
        effective_r2_weight = self.r2_weight if self.r2_weight > 0 else self.nonzero_r2_weight
        
        for iteration in iterator:
            optimizer.zero_grad()
            
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
            total_mse = torch.tensor(0.0, device=self.device)
            total_r2_loss = torch.tensor(0.0, device=self.device)
            n_batches = 0
            
            # Process in batches for memory efficiency
            for i in range(0, n_samples, self.batch_size):
                end = min(i + self.batch_size, n_samples)
                X_batch_sparse = X_csr[i:end]
                
                if X_batch_sparse.nnz == 0:
                    continue
                
                # Convert batch to torch sparse
                X_batch_torch = self._sparse_to_torch(X_batch_sparse)
                coo = X_batch_torch.coalesce()
                X_values = coo.values()
                row_idx = coo.indices()[0]
                col_idx = coo.indices()[1]
                
                W_batch = W[i:end]
                
                # Pre-compute X_recon_values on non-zero positions if needed (for MSE or R²)
                X_recon_values_nonzero = None
                if self.nonzero_mse_weight > 0 or self.nonzero_r2_weight > 0:
                    # Compute reconstructed values on non-zero positions (chunked to avoid OOM)
                    W_rows = W_batch[row_idx]
                    # Use much smaller chunk size when nonzero_mse_weight is enabled to be more conservative
                    # Especially important for large n_components
                    chunk_size = 5000 if self.nonzero_mse_weight > 0 else 20000
                    X_recon_values_nonzero = _compute_recon_values_chunked(W_rows, H, col_idx, chunk_size, self.device)
                
                # MSE loss - controlled by nonzero_mse_weight
                if effective_mse_weight > 0:
                    if self.nonzero_mse_weight > 0:
                        # Compute MSE on non-zero values only
                        if X_recon_values_nonzero is None:
                            W_rows = W_batch[row_idx]
                            chunk_size = 5000  # Very conservative chunk size for nonzero_mse_weight
                            X_recon_values = _compute_recon_values_chunked(W_rows, H, col_idx, chunk_size, self.device)
                        else:
                            X_recon_values = X_recon_values_nonzero
                        mse_loss = F.mse_loss(X_recon_values, X_values)
                    else:
                        # Compute MSE on all values (including zeros) - need full dense matrix
                        X_batch_dense = X_batch_torch.to_dense()
                        X_recon_batch = torch.mm(W_batch, H)
                        mse_loss = F.mse_loss(X_recon_batch, X_batch_dense)
                    total_loss = total_loss + effective_mse_weight * mse_loss
                    total_mse = total_mse + mse_loss.detach()
                
                # R² loss - controlled by nonzero_r2_weight
                if effective_r2_weight > 0:
                    if self.nonzero_r2_weight > 0:
                        # Compute R² on non-zero values only
                        if X_recon_values_nonzero is None:
                            W_rows = W_batch[row_idx]
                            chunk_size = 5000  # Very conservative chunk size
                            X_recon_values = _compute_recon_values_chunked(W_rows, H, col_idx, chunk_size, self.device)
                        else:
                            X_recon_values = X_recon_values_nonzero
                        r2_loss = self._compute_r2_loss(X_values, X_recon_values, X_mean)
                    else:
                        # Compute R² on all values (including zeros) - need full dense matrix
                        X_batch_dense = X_batch_torch.to_dense()
                        X_recon_batch = torch.mm(W_batch, H)
                        X_batch_flat = X_batch_dense.flatten()
                        X_recon_flat = X_recon_batch.flatten()
                        # Use mean of all values in this batch (including zeros) for R² calculation
                        X_mean_batch = X_batch_flat.mean()
                        r2_loss = self._compute_r2_loss(X_batch_flat, X_recon_flat, X_mean_batch)
                    total_loss = total_loss + effective_r2_weight * r2_loss
                    total_r2_loss = total_r2_loss + r2_loss.detach()
                
                # Clear intermediate tensors and cache after each batch when using nonzero_mse_weight
                # (memory-intensive mode) - helps prevent OOM
                if self.nonzero_mse_weight > 0 or self.nonzero_r2_weight > 0:
                    # Clear cache periodically to prevent memory fragmentation
                    if self.device.type == 'cuda' and n_batches % 5 == 0:
                        torch.cuda.empty_cache()
                
                n_batches += 1
            
            if n_batches == 0:
                continue
            
            # Average loss across batches
            total_loss = total_loss / n_batches
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([W, H], max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Enforce non-negativity
            with torch.no_grad():
                W.clamp_(min=1e-10)
                H.clamp_(min=1e-10)
            
            # Monitor progress
            if self.verbose and ((iteration + 1) % loss_report_interval == 0 or iteration == 0):
                avg_mse = (total_mse / n_batches).item()
                # Compute R² for display if either r2_weight > 0 or nonzero_r2_weight > 0
                should_show_r2 = self.r2_weight > 0 or self.nonzero_r2_weight > 0
                avg_r2_loss = (total_r2_loss / n_batches).item() if should_show_r2 else 0
                current_r2 = 1 - avg_r2_loss  # Convert loss back to R²
                
                postfix = {'loss': f'{total_loss.item():.6f}', 'MSE': f'{avg_mse:.6f}'}
                if should_show_r2:
                    # Label R² based on whether it's computed on non-zero values only
                    r2_label = 'R²(nonzero)' if self.nonzero_r2_weight > 0 else 'R²'
                    postfix[r2_label] = f'{current_r2:.4f}'
                
                if prev_loss is not None:
                    loss_change = abs(prev_loss - total_loss.item()) / (prev_loss + 1e-10)
                    postfix['change'] = f'{loss_change:.2%}'
                    
                    if loss_change < self.tol:
                        if self.verbose:
                            print(f"\nConverged at iteration {iteration + 1}")
                        break
                
                # Check patience-based early stopping
                if self.patience is not None:
                    current_loss = total_loss.item()
                    if best_loss is None or current_loss < best_loss:
                        best_loss = current_loss
                        patience_counter = 0
                    else:
                        patience_counter += loss_report_interval
                    
                    # Add patience counter to progress bar
                    postfix['patience'] = f'{patience_counter}/{self.patience}'
                    
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"\nEarly stopping: no improvement for {self.patience} iterations (best loss: {best_loss:.6f})")
                        break
                
                iterator.set_postfix(postfix)
                prev_loss = total_loss.item()
        
        # Detach from computation graph
        W = W.detach()
        H = H.detach()
        
        return W, H, iteration
    
    def _check_convergence(
        self,
        X_csr: spmatrix,
        W: torch.Tensor,
        H: torch.Tensor,
        n_samples: int,
        prev_error: Optional[float],
        iterator,
        iteration: int,
        patience: Optional[int] = None,
        best_error: Optional[float] = None,
        patience_counter: int = 0,
    ) -> tuple:
        """Check convergence and update progress bar."""
        sample_size = min(1000, n_samples)
        sample_indices = torch.randperm(n_samples, device=self.device)[:sample_size]
        
        X_sample_sparse = X_csr[sample_indices.cpu().numpy()]
        if X_sample_sparse.nnz > 0:
            X_sample_torch = self._sparse_to_torch(X_sample_sparse)
            W_sample = W[sample_indices]
            X_recon_sample = torch.mm(W_sample, H)
            X_sample_dense = X_sample_torch.to_dense()
            error = torch.mean((X_sample_dense - X_recon_sample) ** 2).item()
            
            if prev_error is not None:
                error_change = abs(prev_error - error) / (prev_error + 1e-10)
                postfix = {'error': f'{error:.6f}', 'change': f'{error_change:.2%}'}
                if patience is not None:
                    postfix['patience'] = f'{patience_counter}/{patience}'
                iterator.set_postfix(postfix)
                
                if error_change < self.tol:
                    if self.verbose:
                        print(f"\nConverged at iteration {iteration + 1}")
                    return error, error, True
            else:
                postfix = {'error': f'{error:.6f}', 'change': 'N/A'}
                if patience is not None:
                    postfix['patience'] = f'{patience_counter}/{patience}'
                iterator.set_postfix(postfix)
            
            return error, error, False
        
        return prev_error, prev_error, False
    
    def _compute_final_metrics(
        self,
        X_csr: spmatrix,
        W: torch.Tensor,
        H: torch.Tensor,
        n_samples: int,
    ) -> None:
        """Compute final reconstruction error and R² score (both all values and non-zero only)."""
        if self.verbose:
            print("Computing final reconstruction error and R²...")
        
        sample_size = min(5000, n_samples)
        sample_indices = torch.randperm(n_samples, device=self.device)[:sample_size]
        X_sample_sparse = X_csr[sample_indices.cpu().numpy()]
        if X_sample_sparse.nnz > 0:
            X_sample_torch = self._sparse_to_torch(X_sample_sparse)
            W_sample = W[sample_indices]
            X_recon_sample = torch.mm(W_sample, H)
            X_sample_dense = X_sample_torch.to_dense()
            self.reconstruction_error_ = torch.mean((X_sample_dense - X_recon_sample) ** 2).item()
            
            # Compute R² on all values (including zeros)
            X_sample_flat = X_sample_dense.flatten()
            X_recon_flat = X_recon_sample.flatten()
            X_mean_all = X_sample_flat.mean()
            
            SS_res_all = torch.sum((X_sample_flat - X_recon_flat) ** 2).item()
            SS_tot_all = torch.sum((X_sample_flat - X_mean_all) ** 2).item()
            
            # R² can be negative when model performs worse than baseline (predicting mean)
            # For highly sparse data, NMF with non-negative constraints cannot produce exact zeros.
            # The model W @ H produces a dense matrix with small positive values everywhere,
            # causing large errors on zero positions. This makes R² (all values) very negative
            # even though R² (non-zero only) may be good.
            if SS_tot_all > 0:
                self.r2_score_ = 1 - (SS_res_all / SS_tot_all)
                # Clamp extremely negative values to prevent numerical issues
                # R² below -1000 is likely a numerical artifact or indicates severe model failure
                if self.r2_score_ < -1000:
                    self.r2_score_ = -1000.0
            else:
                self.r2_score_ = 0.0
            
            # Compute R² on non-zero values only (sparse-aware)
            coo = X_sample_torch.coalesce()
            X_values = coo.values()  # Only non-zero values
            row_idx = coo.indices()[0]
            col_idx = coo.indices()[1]
            
            # Get reconstructed values at non-zero positions only
            W_rows = W_sample[row_idx]
            X_recon_values = _compute_recon_values_chunked(W_rows, H, col_idx)
            
            # R² on non-zero values only
            X_mean_nonzero = X_values.mean()
            SS_res_nonzero = torch.sum((X_values - X_recon_values) ** 2).item()
            SS_tot_nonzero = torch.sum((X_values - X_mean_nonzero) ** 2).item()
            
            self.r2_score_nonzero_ = 1 - (SS_res_nonzero / SS_tot_nonzero) if SS_tot_nonzero > 0 else 0.0
        else:
            self.reconstruction_error_ = 0.0
            self.r2_score_ = 0.0
            self.r2_score_nonzero_ = 0.0
        
        if self.verbose:
            print(f"Final reconstruction error (MSE): {self.reconstruction_error_:.6f}")
            print(f"Final R² (all values): {self.r2_score_:.6f}")
            if self.r2_score_ < -10:
                print(f"  ⚠️  Warning: Very negative R² (all values) is expected for highly sparse data.")
                print(f"     NMF with non-negative constraints cannot produce exact zeros - the model")
                print(f"     predicts small positive values everywhere, causing large errors on zero")
                print(f"     positions. R² (non-zero only) is the relevant metric for sparse data.")
            print(f"Final R² (non-zero values only): {self.r2_score_nonzero_:.6f}")
            print(f"  (computed on {sample_size:,} sample rows out of {n_samples:,} total)")
    
    def transform(self, X_sparse: spmatrix) -> np.ndarray:
        """
        Transform new data using the fitted model.
        
        Note: This is a simplified version that fits W for new data while
        keeping H fixed. For true out-of-sample transformation, you would
        need to solve: W_new = argmin ||X_new - W_new @ H||^2
        
        Parameters
        ----------
        X_sparse : scipy.sparse matrix
            New sparse matrix to transform.
            
        Returns
        -------
        np.ndarray
            Transformed matrix of shape (n_samples, n_components).
        """
        if self.H is None:
            raise ValueError("Model must be fitted before transform. Call fit_transform first.")
        
        n_samples, n_features = X_sparse.shape
        
        if n_features != self.H.shape[1]:
            raise ValueError(
                f"Input matrix has {n_features} features, but model expects {self.H.shape[1]}"
            )
        
        # Initialize W for new data
        W_new = torch.rand(n_samples, self.n_components, device=self.device) * 0.1 + 0.01
        
        # Convert to CSR for efficient row slicing
        X_csr = X_sparse.tocsr()
        
        # Update W with fixed H (fewer iterations needed)
        for iteration in range(min(100, self.max_iter // 5)):
            for i in range(0, n_samples, self.batch_size):
                end = min(i + self.batch_size, n_samples)
                X_batch_sparse = X_csr[i:end]
                
                if X_batch_sparse.nnz > 0:
                    X_batch_torch = self._sparse_to_torch(X_batch_sparse)
                    W_batch = W_new[i:end]
                    
                    XHT_batch = self._sparse_matmul(X_batch_torch, self.H.t())
                    WHHT_batch = torch.mm(torch.mm(W_batch, self.H), self.H.t())
                    
                    W_new[i:end] = W_batch * (XHT_batch / (WHHT_batch + 1e-10))
            
            W_new = torch.clamp(W_new, min=1e-10)
        
        return W_new.cpu().numpy()


class SparseNMF_Autoencoder(nn.Module):
    """
    Joint SparseNMF + Autoencoder model for end-to-end training.
    
    This model combines sparse NMF with an autoencoder in a single trainable architecture.
    The key advantage is that sparse operations are used throughout NMF, and only the
    reduced W matrix (n_samples × n_components) is converted to dense for the autoencoder.
    
    Architecture:
        Sparse X (n_samples, n_features) 
          → NMF: X ≈ W @ H (sparse operations)
          → Dense W (n_samples, n_components) 
          → Autoencoder: W → z (n_samples, latent_dim)
    
    Parameters
    ----------
    n_samples : int
        Number of samples (rows) in the input matrix.
        Can be inferred from X_sparse.shape[0] in train_joint_model.
    n_features : int
        Number of features (columns) in the input matrix.
        Can be inferred from X_sparse.shape[1] in train_joint_model.
    nmf_components : int, default 256
        Number of NMF components (intermediate dimension).
    latent_dim : int, default 2
        Final latent dimension (output of autoencoder).
    hidden_dims : tuple of int, default (256, 128, 64, 16)
        Hidden layer dimensions for autoencoder. Matches two-step approach default.
    activation : str, default "relu"
        Activation function for autoencoder.
    dropout : float, default 0.0
        Dropout rate for autoencoder.
    use_vae : bool, default False
        Whether to use Variational Autoencoder.
    use_feature_attention : bool, default False
        If True, learn attention weights for each NMF component based on reconstruction importance.
        Disabled by default for joint training - use two-step approach for feature attention.
    feature_attention_weight : float, default 1.0
        Weight for mixing original input with attended input (0 = no attention, 1 = full attention).
    feature_attention_temperature : float, default 1.0
        Temperature for attention weights. Use higher values (1.0+) for joint training stability.
    normalize_nmf_components : bool, default False
        Whether to L2-normalize NMF components (W) before passing to autoencoder.
        When True, matches `normalize_input=True` in two-step approach. This is crucial
        for proper clustering and prevents radial patterns. When False, preserves the
        original magnitude information in W.
    device : str, default 'cuda'
        Device to use for computation.
    
    Examples
    --------
    >>> from sparse_nmf import SparseNMF_Autoencoder
    >>> from scipy.sparse import csr_matrix
    >>> 
    >>> # Create and train model (dimensions inferred automatically)
    >>> z, model = train_joint_model(
    ...     X_sparse,  # n_samples and n_features inferred from X_sparse.shape
    ...     nmf_components=256,
    ...     latent_dim=2,
    ...     device='cuda:0',
    ...     n_epochs=100
    ... )
    """
    
    def __init__(
        self,
        n_samples: int,
        n_features: int,
        nmf_components: int = 256,
        latent_dim: int = 2,
        hidden_dims: tuple = (256, 128, 64, 16),  # Match two-step approach default
        activation: str = "relu",
        dropout: float = 0.0,
        use_vae: bool = False,
        use_feature_attention: bool = False,
        feature_attention_weight: float = 1.0,
        feature_attention_temperature: float = 1.0,  # Higher = more gradual (safer for joint training)
        normalize_nmf_components: bool = False,  # Whether to L2-normalize W before autoencoder
        device: str = 'cuda',
        random_state: Optional[int] = None,
    ):
        super().__init__()
        
        self.n_samples = n_samples
        self.n_features = n_features
        self.nmf_components = nmf_components
        self.latent_dim = latent_dim
        self.use_vae = use_vae
        self.use_feature_attention = use_feature_attention
        self.feature_attention_weight = feature_attention_weight
        self.feature_attention_temperature = feature_attention_temperature
        self.normalize_nmf_components = normalize_nmf_components
        
        # Set device
        if device.startswith('cuda') and not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Set random seed
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
        # NMF parameters (learnable)
        # W: (n_samples, nmf_components) - this is the only dense matrix
        # Better initialization: use Xavier uniform for better gradient flow
        W_init = torch.empty(n_samples, nmf_components, device=self.device)
        nn.init.xavier_uniform_(W_init, gain=0.1)
        self.W = nn.Parameter(W_init + 0.01)  # Ensure positive
        
        # H: (nmf_components, n_features)
        H_init = torch.empty(nmf_components, n_features, device=self.device)
        nn.init.xavier_uniform_(H_init, gain=0.1)
        self.H = nn.Parameter(H_init + 0.01)  # Ensure positive
        
        # Feature attention mechanism (matches two-step autoencoder)
        # Learns which NMF components are important for reconstruction
        if use_feature_attention:
            self.feature_attention_net = nn.Sequential(
                nn.Linear(nmf_components, max(64, nmf_components // 4)),
                nn.ReLU(),
                nn.Linear(max(64, nmf_components // 4), nmf_components),
                # No sigmoid here - we'll apply sigmoid with temperature in forward()
            )
        
        # Autoencoder
        # Activation function
        if activation == "relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu" or activation == "swish":
            self.activation = nn.SiLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build encoder
        encoder_layers = []
        prev_dim = nmf_components
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(self.activation)
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.encoder_base = nn.Sequential(*encoder_layers)
        
        # Encoder output
        if use_vae:
            self.encoder_mu = nn.Linear(prev_dim, latent_dim)
            self.encoder_logvar = nn.Linear(prev_dim, latent_dim)
        else:
            self.encoder_latent = nn.Linear(prev_dim, latent_dim)
        
        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(self.activation)
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, nmf_components))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def _sparse_to_torch(self, X_sparse: spmatrix) -> torch.Tensor:
        """Convert scipy sparse matrix to PyTorch sparse tensor."""
        if not isinstance(X_sparse, coo_matrix):
            coo = X_sparse.tocoo()
        else:
            coo = X_sparse
        
        indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
        values = torch.from_numpy(coo.data).float()
        shape = torch.Size(coo.shape)
        
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, shape, device=self.device
        )
        return sparse_tensor
    
    def forward(self, X_sparse_torch: torch.Tensor):
        """
        Forward pass through the joint model.
        
        Parameters
        ----------
        X_sparse_torch : torch.sparse.FloatTensor
            Sparse input matrix (n_samples, n_features)
        
        Returns
        -------
        z : torch.Tensor
            Latent embeddings (n_samples, latent_dim)
        W_recon : torch.Tensor
            Reconstructed W from autoencoder (n_samples, nmf_components)
        X_recon : torch.Tensor
            NMF reconstruction (n_samples, n_features) - for loss computation
        W : torch.Tensor
            Current W matrix (n_samples, nmf_components)
        """
        # Ensure non-negativity
        W = torch.clamp(self.W, min=1e-10)
        H = torch.clamp(self.H, min=1e-10)
        
        # NMF reconstruction: X ≈ W @ H
        # Don't compute full dense matrix - we'll compute only needed values in loss
        # X_recon = torch.mm(W, H)  # (n_samples, n_features) - TOO LARGE!
        # Instead, we'll compute W @ H only for non-zero elements in loss function
        X_recon = None  # Will be computed on-demand in loss function
        
        # Optionally normalize W before autoencoder (matches normalize_input=True in two-step)
        # This is crucial for proper clustering and prevents radial patterns when enabled
        if self.normalize_nmf_components:
            W_for_encoder = torch.nn.functional.normalize(W, p=2, dim=1)
        else:
            W_for_encoder = W
        
        # Apply feature attention if enabled (matches two-step autoencoder)
        if self.use_feature_attention:
            attention_logits = self.feature_attention_net(W_for_encoder)
            attention_weights = torch.sigmoid(attention_logits / self.feature_attention_temperature)
            attended_W = W_for_encoder * attention_weights
            W_for_encoder = (1.0 - self.feature_attention_weight) * W_for_encoder + self.feature_attention_weight * attended_W
        
        # Pass through autoencoder
        encoded_base = self.encoder_base(W_for_encoder)
        
        if self.use_vae:
            mu = self.encoder_mu(encoded_base)
            logvar = self.encoder_logvar(encoded_base)
            z = self.reparameterize(mu, logvar)
            W_recon = self.decoder(z)
            return z, W_recon, None, W, H, mu, logvar  # X_recon=None, pass H instead
        else:
            z = self.encoder_latent(encoded_base)
            W_recon = self.decoder(z)
            return z, W_recon, None, W, H  # X_recon=None, pass H instead
    
    def encode(self, X_sparse_torch: Optional[torch.Tensor] = None):
        """
        Encode to latent space.
        
        Parameters
        ----------
        X_sparse_torch : torch.sparse.FloatTensor, optional
            If provided, uses current W. Otherwise uses stored W.
        
        Returns
        -------
        z : torch.Tensor
            Latent embeddings (n_samples, latent_dim)
        """
        W = torch.clamp(self.W, min=1e-10)
        # Optionally normalize W before encoding (matches normalize_input=True in two-step)
        if self.normalize_nmf_components:
            W_for_encoder = torch.nn.functional.normalize(W, p=2, dim=1)
        else:
            W_for_encoder = W
        
        # Apply feature attention if enabled (matches two-step autoencoder)
        if self.use_feature_attention:
            attention_logits = self.feature_attention_net(W_for_encoder)
            attention_weights = torch.sigmoid(attention_logits / self.feature_attention_temperature)
            attended_W = W_for_encoder * attention_weights
            W_for_encoder = (1.0 - self.feature_attention_weight) * W_for_encoder + self.feature_attention_weight * attended_W
        
        encoded_base = self.encoder_base(W_for_encoder)
        
        if self.use_vae:
            mu = self.encoder_mu(encoded_base)
            logvar = self.encoder_logvar(encoded_base)
            if self.training:
                z = self.reparameterize(mu, logvar)
            else:
                z = mu  # Use mean at inference
            return z
        else:
            return self.encoder_latent(encoded_base)


def compute_joint_loss(
    model: SparseNMF_Autoencoder,
    X_sparse_torch: torch.Tensor,
    z: torch.Tensor,
    W_recon: torch.Tensor,
    X_recon: Optional[torch.Tensor],
    W: torch.Tensor,
    H: torch.Tensor,
    mu: Optional[torch.Tensor] = None,
    logvar: Optional[torch.Tensor] = None,
    nmf_weight: float = 1.0,
    ae_weight: float = 1.0,
    kl_weight: float = 0.01,
    use_sparse_loss: bool = True,
    use_contrastive: bool = True,
    contrastive_weight: float = 0.25,
    contrastive_temperature: float = 0.5,
    use_cosine_loss: bool = True,
    dimension_reg_weight: float = 0.1,
) -> tuple:
    """
    Compute combined loss for joint model.
    
    Parameters
    ----------
    model : SparseNMF_Autoencoder
        The model instance.
    X_sparse_torch : torch.sparse.FloatTensor
        Sparse input matrix.
    z : torch.Tensor
        Latent embeddings.
    W_recon : torch.Tensor
        Reconstructed W from autoencoder.
    X_recon : torch.Tensor
        NMF reconstruction.
    W : torch.Tensor
        Current W matrix.
    mu : torch.Tensor, optional
        VAE mean (if using VAE).
    logvar : torch.Tensor, optional
        VAE log variance (if using VAE).
    nmf_weight : float, default 1.0
        Weight for NMF reconstruction loss.
    ae_weight : float, default 1.0
        Weight for autoencoder reconstruction loss.
    kl_weight : float, default 0.01
        Weight for KL divergence (VAE only).
    use_sparse_loss : bool, default True
        If True, compute NMF loss only on non-zero elements (more efficient).
    
    Returns
    -------
    total_loss : torch.Tensor
        Total combined loss.
    loss_dict : dict
        Dictionary of individual loss components.
    """
    import torch.nn.functional as F
    
    losses = {}
    
    # 1. NMF reconstruction loss
    if use_sparse_loss:
        # Only compute loss on non-zero elements (memory efficient)
        coo = X_sparse_torch.coalesce()
        X_values = coo.values()
        row_idx = coo.indices()[0]
        col_idx = coo.indices()[1]
        
        # Compute W @ H only for non-zero positions (memory efficient)
        # X_recon[row, col] = sum_k W[row, k] * H[k, col]
        # We can compute this efficiently using batched operations
        W_rows = W[row_idx]  # (nnz, nmf_components)
        X_recon_values = _compute_recon_values_chunked(W_rows, H, col_idx)  # (nnz,)
        
        nmf_loss = F.mse_loss(X_recon_values, X_values)
    else:
        # Compute full dense loss (memory intensive - not recommended for large matrices)
        if X_recon is None:
            # Compute on the fly
            X_recon = torch.mm(W, H)
        X_dense = X_sparse_torch.to_dense()
        nmf_loss = F.mse_loss(X_recon, X_dense)
    
    losses['nmf'] = nmf_loss
    
    # 2. Autoencoder reconstruction loss
    if use_cosine_loss:
        # Use cosine loss (works better with normalized inputs)
        # Normalize both W and W_recon for cosine similarity
        W_norm = torch.nn.functional.normalize(W, p=2, dim=1)
        W_recon_norm = torch.nn.functional.normalize(W_recon, p=2, dim=1)
        # Cosine loss: 1 - cosine_similarity (mean over batch)
        cosine_sim = (W_norm * W_recon_norm).sum(dim=1)
        ae_loss = (1.0 - cosine_sim).mean()
    else:
        # Standard MSE loss
        ae_loss = F.mse_loss(W_recon, W)
    losses['ae'] = ae_loss
    
    # 3. KL divergence (VAE only)
    kl_loss = None
    if model.use_vae and mu is not None and logvar is not None:
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        losses['kl'] = kl_loss
    
    # 4. Contrastive loss (InfoNCE-style, matches autoencoder implementation)
    # Memory-efficient: sample subset for large datasets to avoid O(n^2) memory
    contrastive_loss = None
    if use_contrastive:
        n_samples = W.shape[0]
        if n_samples < 2:
            contrastive_loss = torch.tensor(0.0, device=W.device)
        else:
            # For large datasets, sample a subset to avoid O(n^2) memory
            # Sample up to 1000 samples for contrastive loss computation
            max_contrastive_samples = min(1000, n_samples)
            
            if n_samples > max_contrastive_samples:
                # Randomly sample indices
                indices = torch.randperm(n_samples, device=W.device)[:max_contrastive_samples]
                W_sample = W[indices]
                z_sample = z[indices]
            else:
                W_sample = W
                z_sample = z
            
            batch_size = W_sample.shape[0]
            
            # Normalize for cosine similarity (matches autoencoder)
            W_norm = torch.nn.functional.normalize(W_sample, p=2, dim=1)
            z_norm = torch.nn.functional.normalize(z_sample, p=2, dim=1)
            
            # Compute pairwise similarities in W space (original)
            W_sim = torch.mm(W_norm, W_norm.t())  # (batch, batch)
            
            # Compute pairwise similarities in z space (latent)
            z_sim = torch.mm(z_norm, z_norm.t()) / contrastive_temperature  # (batch, batch)
            
            # Convert W similarities to soft targets (probability distribution)
            W_targets = torch.nn.functional.softmax(W_sim / contrastive_temperature, dim=1)
            
            # Remove diagonal (self-similarity)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=W.device)
            
            # Compute cross-entropy: z similarities should match W similarities
            z_log_softmax = torch.nn.functional.log_softmax(z_sim, dim=1)
            
            # KL divergence: sum over pairs, mean over batch
            contrastive_loss = -torch.sum(W_targets * z_log_softmax * mask.float()) / batch_size
        
        losses['contrastive'] = contrastive_loss
    
    # 5. Dimension regularization (prevents collapse to 1D line)
    dim_reg_loss = None
    if dimension_reg_weight > 0:
        # Encourage both dimensions to be used
        # Penalize if variance in one dimension is much smaller than the other
        z_var = z.var(dim=0)  # Variance per dimension
        if z_var.min() > 0:
            dim_ratio = z_var.max() / (z_var.min() + 1e-8)
            # Penalize if ratio is too large (one dimension dominates)
            dim_reg_loss = torch.clamp(dim_ratio - 5.0, min=0.0)  # Only penalize if ratio > 5
        else:
            dim_reg_loss = torch.tensor(1.0, device=z.device)  # Heavy penalty if one dim has zero variance
        losses['dim_reg'] = dim_reg_loss
    
    # Total loss
    total_loss = nmf_weight * nmf_loss + ae_weight * ae_loss
    if kl_loss is not None:
        total_loss += kl_weight * kl_loss
    if contrastive_loss is not None:
        total_loss += contrastive_weight * contrastive_loss
    if dim_reg_loss is not None:
        total_loss += dimension_reg_weight * dim_reg_loss
    
    return total_loss, losses


def train_joint_model(
    X_sparse: spmatrix,
    model: Optional[SparseNMF_Autoencoder] = None,
    n_samples: Optional[int] = None,
    n_features: Optional[int] = None,
    nmf_components: int = 256,
    latent_dim: int = 2,
    hidden_dims: tuple = (256, 128, 64, 16),  # Match two-step approach default
    activation: str = "relu",
    dropout: float = 0.0,
    use_vae: bool = True,
    use_feature_attention: bool = False,  # Disabled by default - problematic for joint training from scratch
    feature_attention_weight: float = 1.0,
    feature_attention_temperature: float = 1.0,  # Higher temp for joint model (more gradual)
    normalize_nmf_components: bool = False,  # Whether to L2-normalize W before autoencoder
    device: str = 'cuda',
    n_epochs: int = 200,
    learning_rate: float = 0.0005,
    nmf_weight: float = 1.0,
    ae_weight: float = 1.0,
    kl_weight: float = 0.01,
    use_contrastive: bool = False,  # Matches working two-step config
    contrastive_weight: float = 0.25,
    contrastive_temperature: float = 0.5,
    use_cosine_loss: bool = True,
    dimension_reg_weight: float = 0.0,  # Disabled - can cause instability
    weight_decay: float = 1e-4,
    batch_size: Optional[int] = None,
    verbose: bool = True,
    random_state: Optional[int] = None,
    save_path: Optional[str] = None,
    force: bool = False,
) -> tuple:
    """
    Train the joint SparseNMF + Autoencoder model.
    
    Parameters
    ----------
    X_sparse : scipy.sparse matrix
        Sparse input matrix (n_samples, n_features).
    model : SparseNMF_Autoencoder, optional
        Pre-initialized model. If None, creates a new model.
    n_samples : int, optional
        Number of samples. If None, automatically inferred from X_sparse.shape[0].
        Only specify if you want to validate the shape matches.
    n_features : int, optional
        Number of features. If None, automatically inferred from X_sparse.shape[1].
        Only specify if you want to validate the shape matches.
    nmf_components : int, default 256
        Number of NMF components.
    latent_dim : int, default 2
        Final latent dimension.
    hidden_dims : tuple, default (256, 128, 64, 16)
        Autoencoder hidden dimensions. Matches two-step approach default.
    activation : str, default "relu"
        Activation function.
    dropout : float, default 0.0
        Dropout rate.
    use_vae : bool, default True
        Whether to use VAE.
    use_feature_attention : bool, default False
        If True, learn attention weights for each NMF component. Disabled by default for
        joint training because randomly initialized attention can destabilize training.
        Use two-step approach if you need feature attention.
    feature_attention_weight : float, default 1.0
        Weight for mixing original input with attended input (0 = no attention, 1 = full).
    feature_attention_temperature : float, default 1.0
        Temperature for attention weights. Higher values for joint training stability.
    normalize_nmf_components : bool, default False
        Whether to L2-normalize NMF components (W) before passing to autoencoder.
        When True, matches `normalize_input=True` in two-step approach. This is crucial
        for proper clustering and prevents radial patterns. When False, preserves the
        original magnitude information in W.
    device : str, default 'cuda'
        Device to use.
    n_epochs : int, default 200
        Number of training epochs.
    learning_rate : float, default 0.0005
        Learning rate.
    nmf_weight : float, default 1.0
        Weight for NMF loss.
    ae_weight : float, default 1.0
        Weight for AE loss.
    kl_weight : float, default 0.01
        Weight for KL loss (VAE only).
    use_contrastive : bool, default False
        Whether to use contrastive loss. Default False matches working two-step config.
    contrastive_weight : float, default 0.25
        Weight for contrastive loss term.
    contrastive_temperature : float, default 0.5
        Temperature for contrastive loss (lower = sharper distinctions).
    use_cosine_loss : bool, default True
        Whether to use cosine loss for autoencoder reconstruction.
        Works better with normalized inputs (matches two-step approach).
    dimension_reg_weight : float, default 0.1
        Weight for dimension regularization (prevents collapse to 1D).
    weight_decay : float, default 1e-4
        L2 weight regularization for optimizer.
    batch_size : int, optional
        Batch size for autoencoder training. Default 256.
    verbose : bool, default True
        Whether to print progress.
    random_state : int, optional
        Random seed.
    save_path : str, optional
        Path to save model and embeddings.
    force : bool, default False
        If True, retrain even if save_path exists.
    
    Returns
    -------
    z : np.ndarray
        Final latent embeddings (n_samples, latent_dim).
    model : SparseNMF_Autoencoder
        Trained model.
    """
    import os
    import pickle
    from pathlib import Path
    import torch.nn.functional as F
    
    # Set random seed
    if random_state is not None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    # Infer dimensions from X_sparse (always, even if provided - for validation)
    inferred_n_samples = X_sparse.shape[0]
    inferred_n_features = X_sparse.shape[1]
    
    if n_samples is None:
        n_samples = inferred_n_samples
    elif n_samples != inferred_n_samples:
        raise ValueError(
            f"n_samples={n_samples} doesn't match X_sparse.shape[0]={inferred_n_samples}"
        )
    
    if n_features is None:
        n_features = inferred_n_features
    elif n_features != inferred_n_features:
        raise ValueError(
            f"n_features={n_features} doesn't match X_sparse.shape[1]={inferred_n_features}"
        )
    
    # Check for saved model
    if save_path is not None and not force:
        save_path_obj = Path(save_path)
        if save_path_obj.exists():
            if verbose:
                print(f"Loading saved model from {save_path}...")
            try:
                checkpoint = torch.load(save_path, map_location=device)
                model = SparseNMF_Autoencoder(
                    n_samples=n_samples,
                    n_features=n_features,
                    nmf_components=nmf_components,
                    latent_dim=latent_dim,
                    hidden_dims=hidden_dims,
                    activation=activation,
                    dropout=dropout,
                    use_vae=use_vae,
                    use_feature_attention=use_feature_attention,
                    feature_attention_weight=feature_attention_weight,
                    feature_attention_temperature=feature_attention_temperature,
                    normalize_nmf_components=normalize_nmf_components,
                    device=device,
                    random_state=random_state,
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                
                # Load embeddings if available
                embeddings_path = save_path_obj.with_suffix('.npy')
                if embeddings_path.exists():
                    z = np.load(embeddings_path)
                    if verbose:
                        print(f"Loaded embeddings from {embeddings_path}")
                    return z, model
            except Exception as e:
                if verbose:
                    print(f"Error loading saved model: {e}. Training new model...")
    
    # Create model if not provided
    if model is None:
        model = SparseNMF_Autoencoder(
            n_samples=n_samples,
            n_features=n_features,
            nmf_components=nmf_components,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            use_vae=use_vae,
            use_feature_attention=use_feature_attention,
            feature_attention_weight=feature_attention_weight,
            feature_attention_temperature=feature_attention_temperature,
            normalize_nmf_components=normalize_nmf_components,
            device=device,
            random_state=random_state,
        )
    
    model.to(device)
    model.train()
    
    # Convert sparse matrix to PyTorch sparse tensor
    if verbose:
        print(f"Converting sparse matrix to PyTorch sparse tensor...")
    coo = X_sparse.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
    values = torch.from_numpy(coo.data).float()
    X_sparse_torch = torch.sparse_coo_tensor(
        indices, values, X_sparse.shape, device=device
    )
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Determine batch size for autoencoder training
    # Use smaller batches to match two-step approach and prevent OOM
    if batch_size is None:
        ae_batch_size = 256  # Match autoencoder default
    else:
        ae_batch_size = batch_size
    
    # Training loop with batching (like two-step approach)
    if verbose:
        print(f"Training joint model for {n_epochs} epochs...")
        print(f"  NMF components: {nmf_components}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Autoencoder batch size: {ae_batch_size}")
        print(f"  Device: {device}")
    
    # Create data loader for batching (shuffle for better training)
    from torch.utils.data import TensorDataset, DataLoader
    # Create dummy dataset (we'll use indices to slice W)
    dataset = TensorDataset(torch.arange(n_samples, device=device))
    dataloader = DataLoader(dataset, batch_size=ae_batch_size, shuffle=True)
    
    iterator = tqdm(range(n_epochs), disable=not verbose, desc="Training")
    
    for epoch in iterator:
        epoch_loss = 0.0
        epoch_losses = {}
        n_batches = 0
        
        # Process in batches (like two-step autoencoder)
        for batch_idx, (batch_indices_tensor,) in enumerate(dataloader):
            optimizer.zero_grad()
            
            batch_indices = batch_indices_tensor.long()
            batch_size_actual = len(batch_indices)
            
            # Get batch of W (this is the key - we only process a batch at a time)
            # Don't clamp here - let gradients flow, clamp only after optimizer.step()
            W_batch = model.W[batch_indices]  # (batch_size, nmf_components)
            H = model.H
            
            # Get corresponding sparse rows
            # Extract batch rows from sparse matrix
            X_batch_sparse = X_sparse[batch_indices.cpu().numpy()]
            if X_batch_sparse.nnz == 0:
                continue  # Skip empty batches
            
            # Convert batch sparse to torch
            coo_batch = X_batch_sparse.tocoo()
            indices_batch = torch.from_numpy(np.vstack([coo_batch.row, coo_batch.col])).long().to(device)
            values_batch = torch.from_numpy(coo_batch.data).float().to(device)
            X_batch_sparse_torch = torch.sparse_coo_tensor(
                indices_batch, values_batch, X_batch_sparse.shape, device=device
            )
            
            # Normalize W_batch before autoencoder (matches normalize_input=True)
            # Optionally normalize W_batch before autoencoder (matches normalize_input=True in two-step)
            if model.normalize_nmf_components:
                W_for_encoder = torch.nn.functional.normalize(W_batch, p=2, dim=1)
            else:
                W_for_encoder = W_batch
            
            # Apply feature attention if enabled (matches two-step autoencoder)
            if model.use_feature_attention:
                attention_logits = model.feature_attention_net(W_for_encoder)
                attention_weights = torch.sigmoid(attention_logits / model.feature_attention_temperature)
                attended_W = W_for_encoder * attention_weights
                W_for_encoder = (1.0 - model.feature_attention_weight) * W_for_encoder + model.feature_attention_weight * attended_W
            
            # Forward pass through autoencoder only (NMF reconstruction computed separately)
            encoded_base = model.encoder_base(W_for_encoder)
            
            if model.use_vae:
                mu = model.encoder_mu(encoded_base)
                logvar = model.encoder_logvar(encoded_base)
                z_batch = model.reparameterize(mu, logvar)
                W_recon_batch = model.decoder(z_batch)
                mu_batch, logvar_batch = mu, logvar
            else:
                z_batch = model.encoder_latent(encoded_base)
                W_recon_batch = model.decoder(z_batch)
                mu_batch, logvar_batch = None, None
            
            # Compute NMF reconstruction for this batch (sparse-aware)
            # W_batch @ H for non-zero positions only
            coo = X_batch_sparse_torch.coalesce()
            X_values = coo.values()
            row_idx_batch = coo.indices()[0]  # Local row indices (0 to batch_size-1)
            col_idx = coo.indices()[1]
            
            # Map local row indices back to global for W
            W_rows = W_batch[row_idx_batch]  # (nnz_batch, nmf_components)
            X_recon_values = _compute_recon_values_chunked(W_rows, H, col_idx)  # (nnz_batch,)
            
            # Compute losses for this batch
            # 1. NMF reconstruction loss
            nmf_loss_batch = F.mse_loss(X_recon_values, X_values)
            
            # 2. Autoencoder reconstruction loss
            if use_cosine_loss:
                W_batch_norm = torch.nn.functional.normalize(W_batch, p=2, dim=1)
                W_recon_batch_norm = torch.nn.functional.normalize(W_recon_batch, p=2, dim=1)
                cosine_sim = (W_batch_norm * W_recon_batch_norm).sum(dim=1)
                ae_loss_batch = (1.0 - cosine_sim).mean()
            else:
                ae_loss_batch = F.mse_loss(W_recon_batch, W_batch)
            
            # 3. KL divergence (VAE only)
            kl_loss_batch = None
            if model.use_vae and mu_batch is not None and logvar_batch is not None:
                kl_loss_batch = -0.5 * torch.sum(1 + logvar_batch - mu_batch.pow(2) - logvar_batch.exp(), dim=1).mean()
            
            # 4. Contrastive loss (on batch - matches autoencoder implementation)
            contrastive_loss_batch = None
            if use_contrastive and batch_size_actual >= 2:
                # Normalize for cosine similarity (matches autoencoder)
                W_batch_norm = torch.nn.functional.normalize(W_batch, p=2, dim=1)
                z_batch_norm = torch.nn.functional.normalize(z_batch, p=2, dim=1)
                
                # Compute pairwise similarities in W space (original)
                W_sim = torch.mm(W_batch_norm, W_batch_norm.t())  # (batch, batch)
                
                # Compute pairwise similarities in z space (latent)
                z_sim = torch.mm(z_batch_norm, z_batch_norm.t()) / contrastive_temperature  # (batch, batch)
                
                # Convert W similarities to soft targets (probability distribution)
                W_targets = torch.nn.functional.softmax(W_sim / contrastive_temperature, dim=1)
                
                # Remove diagonal (self-similarity)
                mask = ~torch.eye(batch_size_actual, dtype=torch.bool, device=device)
                
                # Compute cross-entropy: z similarities should match W similarities
                z_log_softmax = torch.nn.functional.log_softmax(z_sim, dim=1)
                
                # KL divergence: sum over pairs, mean over batch
                # This matches the autoencoder implementation exactly
                # Note: The division by batch_size_actual is already in the sum, so this gives mean
                contrastive_loss_batch = -torch.sum(W_targets * z_log_softmax * mask.float()) / (batch_size_actual * (batch_size_actual - 1))
            
            # 5. Dimension regularization (on full z, but compute on batch for efficiency)
            dim_reg_loss_batch = None
            if dimension_reg_weight > 0:
                z_var = z_batch.var(dim=0)
                if z_var.min() > 0:
                    dim_ratio = z_var.max() / (z_var.min() + 1e-8)
                    dim_reg_loss_batch = torch.clamp(dim_ratio - 5.0, min=0.0)
                else:
                    dim_reg_loss_batch = torch.tensor(1.0, device=device)
            
            # Total batch loss
            batch_loss = nmf_weight * nmf_loss_batch + ae_weight * ae_loss_batch
            if kl_loss_batch is not None:
                batch_loss += kl_weight * kl_loss_batch
            if contrastive_loss_batch is not None:
                batch_loss += contrastive_weight * contrastive_loss_batch
            if dim_reg_loss_batch is not None:
                batch_loss += dimension_reg_weight * dim_reg_loss_batch
            
            # Backward pass
            batch_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Accumulate losses
            epoch_loss += batch_loss.item()
            if 'nmf' not in epoch_losses:
                epoch_losses = {'nmf': 0.0, 'ae': 0.0}
            epoch_losses['nmf'] += nmf_loss_batch.item()
            epoch_losses['ae'] += ae_loss_batch.item()
            if kl_loss_batch is not None:
                if 'kl' not in epoch_losses:
                    epoch_losses['kl'] = 0.0
                epoch_losses['kl'] += kl_loss_batch.item()
            if contrastive_loss_batch is not None:
                if 'contrastive' not in epoch_losses:
                    epoch_losses['contrastive'] = 0.0
                epoch_losses['contrastive'] += contrastive_loss_batch.item()
            if dim_reg_loss_batch is not None:
                if 'dim_reg' not in epoch_losses:
                    epoch_losses['dim_reg'] = 0.0
                epoch_losses['dim_reg'] += dim_reg_loss_batch.item()
            
            # Update parameters after EACH batch (critical fix - was outside loop!)
            optimizer.step()
            
            # Clamp parameters to maintain non-negativity after each batch update
            with torch.no_grad():
                model.W.clamp_(min=1e-8)
                model.H.clamp_(min=1e-8)
            
            n_batches += 1
        
        # Update progress bar every epoch
        if verbose:
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            loss_str = f"Loss={avg_loss:.6f}"
            for key, val in epoch_losses.items():
                avg_val = val / n_batches if n_batches > 0 else 0.0
                loss_str += f", {key}={avg_val:.6f}"
            iterator.set_postfix_str(loss_str)
            
            # Check gradient norms for debugging (every 10 epochs)
            if (epoch + 1) % 10 == 0:
                total_norm = 0.0
                param_count = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                total_norm = total_norm ** (1. / 2)
                if verbose >= 2:
                    print(f"\n  Gradient norm: {total_norm:.6f} (from {param_count} parameters)")
    
    # Get final embeddings
    model.eval()
    with torch.no_grad():
        z = model.encode()
        z_np = z.cpu().numpy()
    
    # Save model and embeddings
    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"Saving model to {save_path}...")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'nmf_components': nmf_components,
            'latent_dim': latent_dim,
            'hidden_dims': hidden_dims,
            'activation': activation,
            'dropout': dropout,
            'use_vae': use_vae,
        }, save_path)
        
        # Save embeddings
        embeddings_path = save_path_obj.with_suffix('.npy')
        np.save(embeddings_path, z_np)
        
        if verbose:
            print(f"Saved embeddings to {embeddings_path}")
    
    return z_np, model


def train_sparse_nmf(
    X_sparse: Optional[spmatrix] = None,
    n_components: Optional[int] = None,
    max_iter: int = 500,
    device: str = 'cuda',
    batch_size: Optional[int] = None,
    verbose: bool = True,
    random_state: Optional[int] = None,
    mse_weight: float = 1.0,
    r2_weight: float = 0.0,
    learning_rate: float = 0.01,
    nonzero_mse_weight: float = 0.0,
    nonzero_r2_weight: float = 0.0,
    normalize_inputs: bool = True,  # Whether to L2-normalize input X before training
    normalize_outputs: bool = False,  # Whether to L2-normalize output W matrix
    patience: Optional[int] = 10,
    embeddings_save_path: Optional[str] = None,
    model_save_path: Optional[str] = None,
    force: bool = False,
) -> Tuple[np.ndarray, SparseNMF]:
    """
    Train sparse NMF model with automatic saving of embeddings and model.
    
    This is a convenience wrapper that handles model creation, training, and saving.
    If save paths are provided and files exist, loads from disk unless force=True.
    When loading from disk, X_sparse is not required.
    
    Parameters
    ----------
    X_sparse : scipy.sparse matrix, optional
        Sparse input matrix of shape (n_samples, n_features).
        Required when training a new model (force=True or files don't exist).
        Optional when loading from disk (files exist and force=False).
    n_components : int, optional
        Number of components to extract. If ``None`` (default), is
        auto-selected from input shape as
        ``clip(min(n_samples, n_features) // 8, 32, 1024)`` — a
        heuristic that lands in the empirically-good range from small
        synthetic data through phenome-scale matrices. Pass an explicit
        integer to override.
    max_iter : int, default 500
        Maximum number of iterations.
    device : str, default 'cuda'
        Device to use ('cuda', 'cpu', etc.).
    batch_size : int, optional
        Batch size for processing. If None, auto-determines.
    verbose : bool, default True
        Whether to print progress.
    random_state : int, optional
        Random seed for reproducibility.
    mse_weight : float, default 1.0
        Weight for MSE loss component.
    r2_weight : float, default 0.0
        Weight for R² loss component. When > 0, uses gradient descent.
    learning_rate : float, default 0.01
        Learning rate for gradient descent (used when r2_weight > 0 or nonzero_mse_weight > 0).
    nonzero_mse_weight : float, default 0.0
        Controls whether MSE loss includes zeros or only non-zero values.
        When > 0: MSE computed only on non-zero positions (ignores zeros).
        When 0: MSE computed on all positions including zeros (learns sparsity patterns).
    nonzero_r2_weight : float, default 0.0
        Controls whether R² loss includes zeros or only non-zero values.
        When > 0: R² computed only on non-zero positions (ignores zeros).
        When 0: R² computed on all positions including zeros (learns sparsity patterns).
        Only affects training when r2_weight > 0.
    normalize_inputs : bool, default True
        Whether to L2-normalize the input X matrix before training.
        When True, each row of X is normalized to unit length before NMF.
        This is the main mechanism that decouples NMF from per-row
        magnitude (e.g., library depth in single-cell data) and is
        enabled by default for "just works" behavior on sparse biomedical
        data. Set to False only if your rows already have comparable
        magnitudes (e.g., pre-normalized embeddings).
    normalize_outputs : bool, default False
        Whether to L2-normalize the output W matrix (embeddings) before returning.
        When True, each row of the output will have unit length. This is useful if you
        plan to pass the embeddings to an autoencoder with `normalize_input=True` or
        `use_cosine_loss=True`. When False, preserves the original magnitude information.
    patience : int, optional
        Number of iterations to wait without improvement before early
        stopping. Defaults to 10, which is fast for typical workloads
        and prevents over-training. Pass ``None`` to fall back to
        tolerance-based convergence only (legacy behavior).
    embeddings_save_path : str, optional
        Path to save the transformed embeddings (W matrix) as a .npy file.
        If provided and file exists and force=False, loads embeddings from disk.
    model_save_path : str, optional
        Path to save the model (W and H matrices) as a .pkl file.
        If provided and file exists and force=False, loads model from disk.
    force : bool, default False
        If True, force retraining even if save files exist.
        If False and both save files exist, loads from disk instead of retraining.
        
    Returns
    -------
    X_reduced : np.ndarray
        Transformed matrix of shape (n_samples, n_components).
    model : SparseNMF
        Trained SparseNMF model.
        
    Examples
    --------
    >>> from sparse_nmf import train_sparse_nmf
    >>> 
    >>> # Train and save model
    >>> X_reduced, model = train_sparse_nmf(
    ...     X_sparse,
    ...     n_components=256,
    ...     device='cuda:0',
    ...     embeddings_save_path='embeddings.npy',
    ...     model_save_path='model.pkl'
    ... )
    >>> 
    >>> # Load from disk (if files exist and force=False)
    >>> # X_sparse is not required when loading from disk
    >>> X_reduced, model = train_sparse_nmf(
    ...     embeddings_save_path='embeddings.npy',
    ...     model_save_path='model.pkl'
    ... )
    """
    from pathlib import Path
    import pickle
    import os
    
    # Check if we can load from disk
    can_load = False
    if not force and embeddings_save_path is not None and model_save_path is not None:
        if os.path.exists(embeddings_save_path) and os.path.exists(model_save_path):
            can_load = True
            if verbose:
                print(f"Loading embeddings and model from disk...")
                print(f"  Embeddings: {embeddings_save_path}")
                print(f"  Model: {model_save_path}")
    
    if can_load:
        # Load embeddings
        X_reduced = np.load(embeddings_save_path)
        
        # Check if embeddings were saved with normalization by checking if they're already normalized
        # (embeddings saved with normalize_outputs=True are already normalized)
        # Only normalize if they're not already normalized (check if mean norm is close to 1.0)
        if normalize_outputs:
            # Check if embeddings are already normalized (mean L2 norm per row should be ~1.0)
            row_norms = np.linalg.norm(X_reduced, axis=1)
            mean_norm = np.mean(row_norms)
            # If mean norm is close to 1.0 (within 0.01), assume already normalized
            if abs(mean_norm - 1.0) > 0.01:
                from sparse_nmf.utils import l2_normalize
                X_reduced = l2_normalize(X_reduced)
                if verbose:
                    print(f"  Normalized embeddings to unit length")
            elif verbose:
                print(f"  Embeddings already normalized (mean norm: {mean_norm:.4f})")
        
        # Load model
        with open(model_save_path, 'rb') as f:
            model_dict = pickle.load(f)
        
        # Guard against silent reuse of a model trained under a different
        # preprocessing regime. The normalize_inputs default flipped from
        # False → True; caches written before that may still be on disk.
        # If the stored value is present and disagrees with what the
        # caller requested, warn loudly — the returned W will not be what
        # they think it is.
        for _flag in ('normalize_inputs', 'normalize_outputs'):
            _stored = model_dict.get(_flag)
            _requested = locals()[_flag]
            if _stored is not None and _stored != _requested:
                import warnings as _warnings
                _warnings.warn(
                    f"Loaded model was trained with {_flag}={_stored} but the "
                    f"current call requested {_flag}={_requested}. Returning "
                    f"the stored embeddings as-is. Pass force=True to retrain "
                    f"under the requested setting, or pass {_flag}={_stored} "
                    f"to silence this warning.",
                    stacklevel=2,
                )

        # Reconstruct model object
        model = SparseNMF(
            n_components=model_dict['n_components'],
            max_iter=max_iter,  # Use provided max_iter (might differ from saved)
            device=device,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state,
            mse_weight=mse_weight,
            r2_weight=r2_weight,
            learning_rate=learning_rate,
            nonzero_mse_weight=model_dict.get('nonzero_mse_weight', nonzero_mse_weight),
            nonzero_r2_weight=model_dict.get('nonzero_r2_weight', nonzero_r2_weight),
        )
        # Restore model state
        model.W = torch.from_numpy(model_dict['W']).to(device)
        model.H = torch.from_numpy(model_dict['H']).to(device)
        model.reconstruction_error_ = model_dict.get('reconstruction_error', 0.0)
        model.r2_score_ = model_dict.get('r2_score', 0.0)
        model.r2_score_nonzero_ = model_dict.get('r2_score_nonzero', 0.0)
        model.n_iter_ = model_dict.get('n_iter', 0)
        
        if verbose:
            print(f"  Loaded embeddings shape: {X_reduced.shape}")
            print(f"  Model components: {model.n_components}")
            print(f"  Reconstruction error: {model.reconstruction_error_:.6f}")
            print(f"  R² score (all values): {model.r2_score_:.6f}")
            print(f"  R² score (non-zero only): {model.r2_score_nonzero_:.6f}")
        
        return X_reduced, model
    
    # Train new model
    if X_sparse is None:
        raise ValueError(
            "X_sparse is required when training a new model. "
            "Either provide X_sparse or ensure embeddings_save_path and model_save_path "
            "point to existing files to load from disk."
        )

    # Auto-size n_components if not specified. The heuristic
    # min(n_samples, n_features) // 8 clamped to [32, 1024] tracks the
    # range that empirically performs well: ~k=32-64 for hundreds of
    # rows, ~k=128-256 for tens of thousands, ~k=512-1024 at phenome
    # scale (100k+ rows). The user can always pass an integer to
    # override.
    if n_components is None:
        n_samples_, n_features_ = X_sparse.shape
        n_components = int(np.clip(min(n_samples_, n_features_) // 8, 32, 1024))
        if verbose:
            print(f"Auto-selected n_components={n_components} (from input shape {X_sparse.shape})")

    if verbose:
        print("Training new NMF model...")

    # Normalize inputs if requested (before training)
    if normalize_inputs:
        from sklearn.preprocessing import normalize
        from scipy import sparse as sp
        if sp.issparse(X_sparse):
            X_sparse = X_sparse.tocsr()
            X_sparse = sp.csr_matrix(normalize(X_sparse, norm='l2', axis=1))
        else:
            X_sparse = normalize(X_sparse, norm='l2', axis=1)
        if verbose:
            print("L2 normalized input matrix (each row has unit length)")
    
    model = SparseNMF(
        n_components=n_components,
        max_iter=max_iter,
        device=device,
        batch_size=batch_size,
        verbose=verbose,
        random_state=random_state,
        mse_weight=mse_weight,
        r2_weight=r2_weight,
        learning_rate=learning_rate,
        nonzero_mse_weight=nonzero_mse_weight,
        nonzero_r2_weight=nonzero_r2_weight,
        patience=patience,
    )
    
    X_reduced = model.fit_transform(X_sparse)
    
    # Normalize outputs if requested
    if normalize_outputs:
        from sparse_nmf.utils import l2_normalize
        X_reduced = l2_normalize(X_reduced)
        if verbose:
            print(f"Normalized output embeddings to unit length")
    
    # Save embeddings if path provided
    if embeddings_save_path is not None:
        Path(embeddings_save_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_save_path, X_reduced)
        if verbose:
            print(f"Saved embeddings to {embeddings_save_path}")
    
    # Save model if path provided
    if model_save_path is not None:
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        model_dict = {
            'W': model.W.cpu().numpy(),
            'H': model.H.cpu().numpy(),
            'n_components': model.n_components,
            'reconstruction_error': model.reconstruction_error_,
            'r2_score': model.r2_score_,
            'r2_score_nonzero': model.r2_score_nonzero_,
            'n_iter': model.n_iter_,
            'nonzero_mse_weight': model.nonzero_mse_weight,
            'nonzero_r2_weight': model.nonzero_r2_weight,
            # Persist the preprocessing flags so a later load can verify
            # the caller is asking for the same regime the model was
            # trained under. Critical for the normalize_inputs default
            # flip — old caches were trained with normalize_inputs=False
            # but the new default is True, so silent reuse would return
            # a model that doesn't match the caller's expectation.
            'normalize_inputs': normalize_inputs,
            'normalize_outputs': normalize_outputs,
        }
        with open(model_save_path, 'wb') as f:
            pickle.dump(model_dict, f)
        if verbose:
            print(f"Saved model to {model_save_path}")
    
    return X_reduced, model


def sparse_nmf(
    X_sparse: spmatrix,
    n_components: int = 256,
    max_iter: int = 500,
    device: str = 'cuda',
    batch_size: Optional[int] = None,
    verbose: bool = True,
    random_state: Optional[int] = None,
    mse_weight: float = 1.0,
    r2_weight: float = 0.0,
    learning_rate: float = 0.01,
    nonzero_mse_weight: float = 0.0,
    nonzero_r2_weight: float = 0.0,
) -> np.ndarray:
    """
    Convenience function for sparse NMF (without saving).
    
    Parameters
    ----------
    X_sparse : scipy.sparse matrix
        Sparse input matrix of shape (n_samples, n_features).
    n_components : int, default 256
        Number of components to extract.
    max_iter : int, default 500
        Maximum number of iterations.
    device : str, default 'cuda'
        Device to use ('cuda', 'cpu', etc.).
    batch_size : int, optional
        Batch size for processing. If None, auto-determines.
    verbose : bool, default True
        Whether to print progress.
    random_state : int, optional
        Random seed for reproducibility.
    mse_weight : float, default 1.0
        Weight for MSE loss component.
    r2_weight : float, default 0.0
        Weight for R² loss component. When > 0, uses gradient descent.
    learning_rate : float, default 0.01
        Learning rate for gradient descent (used when r2_weight > 0 or nonzero_mse_weight > 0).
    nonzero_mse_weight : float, default 0.0
        Controls whether MSE loss includes zeros or only non-zero values.
        When > 0: MSE computed only on non-zero positions (ignores zeros).
        When 0: MSE computed on all positions including zeros (learns sparsity patterns).
    nonzero_r2_weight : float, default 0.0
        Controls whether R² loss includes zeros or only non-zero values.
        When > 0: R² computed only on non-zero positions (ignores zeros).
        When 0: R² computed on all positions including zeros (learns sparsity patterns).
        Only affects training when r2_weight > 0.
        
    Returns
    -------
    np.ndarray
        Transformed matrix of shape (n_samples, n_components).
        
    Examples
    --------
    >>> from sparse_nmf import sparse_nmf
    >>> # MSE only (default, fast multiplicative updates)
    >>> X_reduced = sparse_nmf(X, n_components=256, device='cuda:0')
    >>> 
    >>> # Weighted MSE + R² loss
    >>> X_reduced = sparse_nmf(X, n_components=256, mse_weight=0.5, r2_weight=0.5)
    """
    nmf = SparseNMF(
        n_components=n_components,
        max_iter=max_iter,
        device=device,
        batch_size=batch_size,
        verbose=verbose,
        random_state=random_state,
        mse_weight=mse_weight,
        r2_weight=r2_weight,
        learning_rate=learning_rate,
        nonzero_mse_weight=nonzero_mse_weight,
        nonzero_r2_weight=nonzero_r2_weight,
    )
    return nmf.fit_transform(X_sparse)


def extract_attention_weights(model, X_nmf, batch_size=256, device=None, verbose=False):
    """
    Extract per-sample feature attention weights from trained autoencoder.
    
    Supports both feature attention and transformer attention modes.
    Optimized for very large datasets (1M+ samples) with memory-efficient processing.
    
    Parameters
    ----------
    model : Autoencoder
        Trained autoencoder model with either:
        - use_feature_attention=True: Feature-wise attention (MLP-based)
        - use_transformer=True: Transformer attention (self-attention between features)
    X_nmf : np.ndarray or torch.Tensor
        NMF-transformed embeddings, shape (n_samples, n_nmf_components).
        These are the inputs to the autoencoder.
        For very large datasets, keep as numpy array on CPU to avoid OOM.
    batch_size : int, default 256
        Batch size for processing. Auto-increased for large datasets.
        For 1M+ samples, uses 4096-8192 for efficiency.
    device : str, optional
        Device to use. If None, uses model's device.
    verbose : bool, default False
        If True, print progress information.
    
    Returns
    -------
    attention_weights : np.ndarray
        Attention weights for each sample and NMF component, 
        shape (n_samples, n_nmf_components).
        For feature attention: values in [0, 1] range (from sigmoid).
        For transformer: values in [0, 1] range (normalized attention scores).
    
    Examples
    --------
    >>> from sparse_nmf import extract_attention_weights
    >>> # Works with feature attention
    >>> attention_weights = extract_attention_weights(model, X_nmf, batch_size=1024)
    >>> # Also works with transformer attention
    >>> attention_weights = extract_attention_weights(model, X_nmf, batch_size=4096)
    >>> print(f"Attention shape: {attention_weights.shape}")  # (n_samples, n_components)
    """
    # Check which attention mode is enabled
    has_feature_attention = hasattr(model, 'use_feature_attention') and model.use_feature_attention
    has_transformer = hasattr(model, 'use_transformer') and model.use_transformer
    
    if not has_feature_attention and not has_transformer:
        raise ValueError(
            "Model does not have attention enabled. "
            "Enable either use_feature_attention=True or use_transformer=True."
        )
    
    if has_feature_attention and has_transformer:
        raise ValueError(
            "Model has both feature attention and transformer enabled. "
            "These are mutually exclusive. Choose one."
        )
    
    # Get device - ensure model and input are on the same device
    model_device = next(model.parameters()).device
    if device is None:
        device = model_device
    else:
        device = torch.device(device)
        # If user specified a different device, use the model's device instead
        # (moving the model would be expensive and might cause issues)
        if device != model_device:
            if verbose:
                print(f"  Warning: Model is on {model_device}, but device={device} specified. Using model's device {model_device}.")
            device = model_device
    
    # Keep X_nmf on CPU for very large datasets to avoid OOM
    # Only move batches to GPU during processing
    if isinstance(X_nmf, np.ndarray):
        X_nmf_numpy = X_nmf
        keep_on_cpu = len(X_nmf) > 100000  # For 100k+ samples, keep on CPU
    else:
        X_nmf_numpy = X_nmf.cpu().numpy() if X_nmf.is_cuda else X_nmf.numpy()
        keep_on_cpu = len(X_nmf_numpy) > 100000
    
    model.eval()
    n_samples = len(X_nmf_numpy)
    n_components = X_nmf_numpy.shape[1]
    
    # Auto-increase batch size for large datasets (more aggressive for speed)
    if n_samples > 500000:  # 500k+ samples
        batch_size = max(8192, batch_size * 16)
    elif n_samples > 100000:  # 100k+ samples
        batch_size = max(4096, batch_size * 8)
    elif n_samples > 10000:
        batch_size = max(2048, batch_size * 4)
    
    if verbose:
        print(f"Extracting attention weights for {n_samples:,} samples...")
        print(f"  Mode: {'Feature Attention' if has_feature_attention else 'Transformer Attention'}")
        print(f"  Batch size: {batch_size:,}")
        print(f"  Keep on CPU: {keep_on_cpu}")
    
    # Pre-allocate output array (on CPU for very large datasets)
    attention_weights = np.empty((n_samples, n_components), dtype=np.float32)
    
    try:
        from tqdm.auto import tqdm
        progress_iter = tqdm(range(0, n_samples, batch_size), desc="Extracting attention", disable=not verbose)
    except ImportError:
        progress_iter = range(0, n_samples, batch_size)
    
    with torch.no_grad():
        for batch_start in progress_iter:
            batch_end = min(batch_start + batch_size, n_samples)
            batch_indices = slice(batch_start, batch_end)
            
            # Get batch (keep on CPU, move to GPU only for processing)
            batch_numpy = X_nmf_numpy[batch_indices]
            batch_tensor = torch.from_numpy(batch_numpy).float().to(device)
            
            if has_feature_attention:
                # Feature attention: compute per-feature attention weights
                attention_logits = model.feature_attention_net(batch_tensor)
                temperature = model.feature_attention_temperature
                batch_attention = torch.sigmoid(attention_logits / temperature)
                # Shape: (batch_size, n_components)
                
            elif has_transformer:  # pragma: no cover
                # Transformer attention: extract from last transformer
                # block. ``SparseNMF_Autoencoder`` in this package
                # never sets ``use_transformer=True`` — no transformer
                # architecture is shipped here. The branch is preserved
                # so external callers that subclass the model with a
                # transformer stack can reuse ``extract_attention_weights``;
                # coverage-excluded since it can't be reached without
                # such a subclass.
                # Reshape to transformer format: (batch_size, n_components, 1)
                batch_expanded = batch_tensor.unsqueeze(-1)  # (batch_size, n_components, 1)
                
                # Project to transformer dimension
                x_proj = model.input_projection(batch_expanded)  # (batch_size, n_components, transformer_hidden_dim)
                
                # Process through all transformer blocks except the last
                for block in model.transformer_blocks[:-1]:
                    x_proj = block(x_proj, need_weights=False)
                
                # Get attention weights from last block
                last_block = model.transformer_blocks[-1]
                normed = last_block.norm1(x_proj)
                _, attn_weights = last_block.attention(normed, normed, normed, need_weights=True)
                # attn_weights shape: (batch_size, num_heads, n_components, n_components) or (batch_size, n_components, n_components)
                
                # Average across heads if needed
                if attn_weights.dim() == 4:
                    attn_weights = attn_weights.mean(dim=1)  # (batch_size, n_components, n_components)
                
                # Convert attention matrix to per-feature importance
                # Option 1: Mean attention received by each feature (how much others attend to it)
                # Option 2: Max attention received by each feature
                # Option 3: Sum of attention received (total importance)
                # We use mean for interpretability: average attention each feature receives
                batch_attention = attn_weights.mean(dim=2)  # (batch_size, n_components)
                # Normalize to [0, 1] range
                batch_attention = (batch_attention - batch_attention.min(dim=1, keepdim=True)[0]) / (
                    batch_attention.max(dim=1, keepdim=True)[0] - batch_attention.min(dim=1, keepdim=True)[0] + 1e-8
                )
            
            # Move back to CPU and store
            attention_weights[batch_indices] = batch_attention.cpu().numpy()
            
            # Clear GPU memory
            del batch_tensor, batch_attention
            if has_transformer:
                del x_proj, attn_weights
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    if verbose:
        print(f"✓ Extracted attention weights: shape {attention_weights.shape}")
    
    return attention_weights


def trace_attention_to_genes(attention_weights_nmf, nmf_H, normalize=True):
    """
    Trace attention weights from NMF components back to original gene features.
    
    This uses the principled approach of matrix multiplication through the 
    linear transformation (NMF H matrix). Since NMF decomposition is:
        X ≈ W @ H
    where W is (n_samples, n_components) and H is (n_components, n_genes),
    the attention on NMF components can be propagated to genes via:
        gene_attention = attention_weights_nmf @ H
    
    This is mathematically sound because:
    1. Attention weights represent importance of each NMF component
    2. H matrix maps each NMF component to original gene features
    3. Matrix multiplication properly aggregates attention across components
    
    This approach is equivalent to computing gradients through linear layers
    and is the standard method for propagating importance/attention through
    linear transformations in neural networks.
    
    Parameters
    ----------
    attention_weights_nmf : np.ndarray
        Attention weights on NMF components, shape (n_samples, n_nmf_components).
        Typically obtained from extract_attention_weights().
    nmf_H : np.ndarray or torch.Tensor
        NMF H matrix mapping components to genes, shape (n_nmf_components, n_genes).
        This is the coefficient matrix from NMF decomposition (X ≈ W @ H).
        Can be obtained from SparseNMF.H or SparseNMF_Autoencoder.H.
    normalize : bool, default True
        If True, normalize attention weights per sample so they sum to 1.
        This makes the weights interpretable as a probability distribution.
    
    Returns
    -------
    gene_attention_weights : np.ndarray
        Attention weights for each sample and gene, shape (n_samples, n_genes).
        Higher values indicate genes that are more important for that sample.
    
    Examples
    --------
    >>> from sparse_nmf import extract_attention_weights, trace_attention_to_genes
    >>> 
    >>> # Extract attention on NMF components
    >>> attention_nmf = extract_attention_weights(model, X_nmf)
    >>> 
    >>> # Trace back to genes
    >>> gene_attention = trace_attention_to_genes(attention_nmf, nmf.H)
    >>> 
    >>> # Get top genes for a sample
    >>> sample_idx = 0
    >>> top_genes = np.argsort(gene_attention[sample_idx])[-10:][::-1]
    """
    # Convert to numpy if needed (handle both CPU and GPU tensors)
    if isinstance(nmf_H, torch.Tensor):
        nmf_H = nmf_H.detach().cpu().numpy()
    
    # Matrix multiplication: (n_samples, n_components) @ (n_components, n_genes)
    # = (n_samples, n_genes)
    gene_attention = attention_weights_nmf @ nmf_H
    
    # Normalize per sample if requested (so each sample's attention sums to 1)
    if normalize:
        # Use efficient in-place normalization
        row_sums = gene_attention.sum(axis=1, keepdims=True)
        # Clip to avoid division by zero (set minimum to 1.0)
        np.clip(row_sums, a_min=1.0, a_max=None, out=row_sums)
        # In-place division (much faster than creating new array)
        np.divide(gene_attention, row_sums, out=gene_attention)
    
    return gene_attention


def extract_and_aggregate_attention(
    model,
    X_nmf,
    nmf_H,
    batch_size=256,
    device=None,
    normalize=True,
    gene_feature_names=None,
    nmf_feature_names=None,
    sample_names=None,
    metadata=None,
    verbose=True,
    nonzero_threshold=None,
    save_dir=None,
    force=False,
    return_attention_matrices=False,
):
    """
    Extract attention weights for all samples, trace to genes, and aggregate statistics.
    
    This is a high-level wrapper that:
    1. Extracts attention weights for all samples on NMF components
    2. Traces attention weights back to original gene features
    3. Aggregates statistics across samples for both gene and NMF features
    
    Supports both feature attention and transformer attention modes.
    Optimized for very large datasets (1M+ samples): uses GPU operations, larger batches,
    memory-efficient processing, and vectorized aggregations.
    
    Parameters
    ----------
    model : Autoencoder
        Trained autoencoder model with either feature attention or transformer attention enabled.
    X_nmf : np.ndarray or torch.Tensor
        NMF-transformed embeddings, shape (n_samples, n_nmf_components).
    nmf_H : np.ndarray or torch.Tensor
        NMF H matrix mapping components to genes, shape (n_nmf_components, n_genes).
    batch_size : int, default 256
        Batch size for extracting attention weights. Auto-increased for large datasets.
    device : str, optional
        Device to use. If None, uses model's device.
    normalize : bool, default True
        If True, normalize attention weights per sample before aggregation.
    gene_feature_names : array-like, optional
        Names for gene features. If None, uses integer indices.
        If metadata is provided, this will be inferred from metadata['var'].index.
    nmf_feature_names : array-like, optional
        Names for NMF features. If None, uses integer indices.
    sample_names : array-like, optional
        Names/IDs for samples. If None, uses integer indices.
        Should match the order of samples in X_nmf.
        Can be obtained from metadata['obs'].index or metadata['obs']['sourceId'].
        If metadata is provided, this will be inferred from metadata['obs'].
    metadata : dict or AnnData-like object, optional
        Metadata object with 'var' and 'obs' keys (e.g., AnnData object).
        If provided, gene_feature_names and sample_names will be inferred from:
        - metadata['var'].index for gene names
        - metadata['obs']['obs_id'] if available, otherwise metadata['obs'].index for sample names
        This parameter takes precedence over gene_feature_names and sample_names if provided.
    verbose : bool, default True
        If True, print progress and summary information.
    nonzero_threshold : float, optional
        Threshold for counting "nonzero" attention. If None, uses percentile-based threshold:
        the 1st percentile of all attention values (identifies bottom 1% as noise).
        If normalize=True, all values are > 0 after normalization, so a threshold is needed
        to identify meaningful attention vs. uniform noise distribution.
    save_dir : str, optional
        Directory path to save the aggregated dataframes as parquet files.
        If provided, saves:
        - gene_aggregated_df as 'gene_attention_aggregated.parquet'
        - nmf_aggregated_df as 'nmf_attention_aggregated.parquet'
        If None, dataframes are not saved.
    force : bool, default False
        If True, overwrite existing parquet files. If False and files exist, raises an error.
    return_attention_matrices : bool, default False
        If True, also return the pre-aggregated attention matrices:
        - gene_attention_matrix: shape (n_samples, n_genes) - attention scores for each sample-gene pair
        - nmf_attention_matrix: shape (n_samples, n_nmf_components) - attention scores for each sample-NMF component pair
        Note: When True, this requires storing the full matrices in memory, which may be memory-intensive
        for very large datasets. If loading from existing files, matrices cannot be returned unless
        they were previously saved.
    
    Returns
    -------
    gene_aggregated_df : pd.DataFrame
        Aggregated statistics per gene feature with columns:
        - feature_index: Gene feature index
        - feature_name: Gene feature name (if provided)
        - mean_attention: Mean attention across all samples
        - min_attention: Minimum attention across all samples
        - max_attention: Maximum attention across all samples
        - n_samples_nonzero: Number of samples with nonzero attention for this gene
        - pct_samples_nonzero: Percentage of samples with nonzero attention
        - max_attention_sample: Sample name/ID with highest attention for this gene (if sample_names provided)
    nmf_aggregated_df : pd.DataFrame
        Aggregated statistics per NMF feature with columns:
        - feature_index: NMF feature index
        - feature_name: NMF feature name (if provided)
        - mean_attention: Mean attention across all samples
        - min_attention: Minimum attention across all samples
        - max_attention: Maximum attention across all samples
        - n_samples_nonzero: Number of samples with nonzero attention for this NMF feature
        - pct_samples_nonzero: Percentage of samples with nonzero attention
        - max_attention_sample: Sample name/ID with highest attention for this NMF feature (if sample_names provided)
    gene_attention_matrix : np.ndarray, optional
        Pre-aggregated attention matrix, shape (n_samples, n_genes).
        Only returned if return_attention_matrices=True.
        Contains continuous attention scores for each sample-gene pair.
    nmf_attention_matrix : np.ndarray, optional
        Pre-aggregated attention matrix, shape (n_samples, n_nmf_components).
        Only returned if return_attention_matrices=True.
        Contains continuous attention scores for each sample-NMF component pair.
    
    Examples
    --------
    >>> from sparse_nmf import extract_and_aggregate_attention
    >>> 
    >>> gene_df, nmf_df = extract_and_aggregate_attention(
    ...     model, X_nmf, nmf.H,
    ...     gene_feature_names=gene_names,
    ...     sample_names=sample_names
    ... )
    >>> 
    >>> # Get top genes by mean attention
    >>> top_genes = gene_df.nlargest(10, 'mean_attention')
    """
    import pandas as pd
    import time
    from pathlib import Path
    
    # Extract gene_feature_names and sample_names from metadata if provided
    if metadata is not None:
        # Extract gene names from metadata['var'].index
        if gene_feature_names is None:
            try:
                gene_feature_names = metadata['var'].index.values
            except (KeyError, AttributeError, TypeError):
                if verbose:
                    print(f"  Warning: Could not extract gene_feature_names from metadata['var'].index, using integer indices")
                gene_feature_names = None
        
        # Extract sample names from metadata['obs']
        if sample_names is None:
            try:
                obs = metadata['obs']
                if 'obs_id' in obs.columns:
                    sample_names = obs['obs_id'].values
                else:
                    sample_names = obs.index.values
            except (KeyError, AttributeError, TypeError):
                if verbose:
                    print(f"  Warning: Could not extract sample_names from metadata['obs'], using integer indices")
                sample_names = None
    
    # Check if files exist BEFORE doing any computation
    if save_dir is not None:
        save_path = Path(save_dir)
        gene_file = save_path / 'gene_attention_aggregated.parquet'
        nmf_file = save_path / 'nmf_attention_aggregated.parquet'
        
        # If both files exist and force=False, just load and return them
        if not force and gene_file.exists() and nmf_file.exists():
            if verbose:
                print(f"Loading existing aggregated attention data from {save_dir}...")
            gene_aggregated_df = pd.read_parquet(gene_file)
            nmf_aggregated_df = pd.read_parquet(nmf_file)
            if verbose:
                print(f"  ✓ Loaded: {gene_file}")
                print(f"  ✓ Loaded: {nmf_file}")
                print(f"  Gene features: {len(gene_aggregated_df):,} genes")
                print(f"  NMF features: {len(nmf_aggregated_df):,} components")
            
            # If return_attention_matrices is True, try to load the matrices
            if return_attention_matrices:
                gene_matrix_file = save_path / 'gene_attention_matrix.npy'
                nmf_matrix_file = save_path / 'nmf_attention_matrix.npy'
                if gene_matrix_file.exists() and nmf_matrix_file.exists():
                    if verbose:
                        print(f"  Loading attention matrices...")
                    gene_attention_matrix = np.load(gene_matrix_file)
                    nmf_attention_matrix = np.load(nmf_matrix_file)
                    if verbose:
                        print(f"  ✓ Loaded: {gene_matrix_file} (shape: {gene_attention_matrix.shape})")
                        print(f"  ✓ Loaded: {nmf_matrix_file} (shape: {nmf_attention_matrix.shape})")
                    return gene_aggregated_df, nmf_aggregated_df, gene_attention_matrix, nmf_attention_matrix
                else:
                    if verbose:
                        print(f"  Warning: Attention matrices not found. Set force=True to recompute with matrices.")
                    # Return None for matrices if they don't exist
                    return gene_aggregated_df, nmf_aggregated_df, None, None
            else:
                return gene_aggregated_df, nmf_aggregated_df
    
    start_time = time.time()
    
    # Initialize matrices to None (will be set if return_attention_matrices=True)
    gene_attention_matrix = None
    nmf_attention_matrix = None
    
    if verbose:
        n_samples = len(X_nmf)
        print(f"Extracting attention weights for all samples and aggregating statistics...")
        print(f"  Processing {n_samples:,} samples...")
        if gene_feature_names is not None:
            print(f"  Gene features: {len(gene_feature_names):,} genes")
    
    # Step 1: Extract attention weights for all samples (optimized, batched)
    # Auto-increase batch size for very large datasets
    n_samples = len(X_nmf)
    if n_samples > 500000:  # 500k+ samples
        effective_batch_size = max(8192, batch_size * 16)
    elif n_samples > 100000:  # 100k+ samples
        effective_batch_size = max(4096, batch_size * 8)
    elif n_samples > 10000:
        effective_batch_size = max(2048, batch_size * 4)
    else:
        effective_batch_size = batch_size
    
    if verbose:
        print(f"  Using batch size: {effective_batch_size:,} for attention extraction")
    
    attention_weights_nmf = extract_attention_weights(
        model, X_nmf, batch_size=effective_batch_size, device=device, verbose=verbose
    )
    
    if verbose:
        print(f"  ✓ Extracted attention weights: {attention_weights_nmf.shape}")
    
    # Step 2: Trace to genes (GPU-accelerated if possible, chunked for memory efficiency)
    # Determine if we should use GPU for large matrices
    # torch is already imported at module level
    use_gpu_for_aggregation = False
    
    # Get the model's device (model and input must be on same device)
    try:
        model_device = next(model.parameters()).device
    except:
        model_device = None
    
    # Try to detect device from model or user input
    if device is None:
        if hasattr(model, 'device'):
            device = model.device
        elif model_device is not None:
            device = model_device
        else:  # pragma: no cover  (only hit by parameterless models w/o .device — defensive fallback)
            device = None
    else:
        # User specified a device, but ensure it matches model's device
        device = torch.device(device) if isinstance(device, str) else device
        if model_device is not None and device != model_device:
            if verbose:
                print(f"  Warning: Model is on {model_device}, but device={device} specified. Using model's device {model_device}.")
            device = model_device
    
    # Check if GPU is available and use it
    if device is not None:
        device_str = str(device)
        if 'cuda' in device_str:
            try:
                use_gpu_for_aggregation = torch.cuda.is_available()
                if use_gpu_for_aggregation:
                    device = torch.device(device_str)  # Ensure it's a torch device
            except:
                pass
    else:  # pragma: no cover  (CPU-only CI: device is None + cuda probe both unreachable)
        # Try to use default CUDA device if available
        try:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                use_gpu_for_aggregation = True
        except:
            pass

    # Final check: if we still don't have a device but CUDA is available, use it
    if not use_gpu_for_aggregation:  # pragma: no cover  (CPU-only CI: cuda init fallback never fires)
        try:
            if torch.cuda.is_available():
                if model_device is not None and 'cuda' in str(model_device):
                    device = model_device
                    use_gpu_for_aggregation = True
                else:
                    device = torch.device('cuda:0')
                    use_gpu_for_aggregation = True
        except:
            pass
    
    # Convert to torch tensors for GPU processing if beneficial.
    # NB: ``use_gpu_for_aggregation`` is True only when CUDA is
    # available — the entire block below is unreachable on the
    # CPU-only CI runners that build the coverage badge. Marking with
    # ``# pragma: no cover`` so the badge reflects the code we
    # actually exercise; the CPU path that handles the same logic
    # starts at the matching ``else`` on line ~3195.
    if use_gpu_for_aggregation:  # pragma: no cover
        if verbose:
            print(f"  ✓ Using GPU for aggregation (device: {device})")
        # Convert to torch tensors on GPU
        if isinstance(attention_weights_nmf, np.ndarray):
            attention_weights_nmf_torch = torch.from_numpy(attention_weights_nmf).float().to(device)
        else:
            attention_weights_nmf_torch = attention_weights_nmf.float().to(device) if isinstance(attention_weights_nmf, torch.Tensor) else attention_weights_nmf
        
        if isinstance(nmf_H, torch.Tensor):
            nmf_H_torch = nmf_H.float().to(device)
        else:
            nmf_H_torch = torch.from_numpy(nmf_H).float().to(device)
        
        # Matrix multiplication on GPU (chunked to avoid OOM for very large matrices)
        n_samples, n_nmf_features = attention_weights_nmf_torch.shape
        n_genes = nmf_H_torch.shape[1]
        
        if verbose:
            print(f"  ✓ Tracing to genes (GPU, chunked): {attention_weights_nmf_torch.shape} @ {nmf_H_torch.shape}")
            print(f"  Output will be ({n_samples}, {n_genes}) - chunking to avoid OOM")
        
        # Estimate memory needed for full matrix: n_samples * n_genes * 4 bytes (float32)
        # If > 50GB, we must chunk
        estimated_memory_gb = (n_samples * n_genes * 4) / (1024**3)
        must_chunk = estimated_memory_gb > 50  # Chunk if > 50GB
        
        if must_chunk:
            # Chunked matrix multiplication - never create full matrix
            if verbose:
                print(f"  Chunking matrix multiplication (estimated {estimated_memory_gb:.1f} GB)")
            
            # Calculate safe chunk size based on available GPU memory
            # Each chunk creates (chunk_size, n_genes) matrix = chunk_size * n_genes * 4 bytes
            # We want to keep chunks under ~5GB to leave room for other operations and intermediate tensors
            max_chunk_memory_gb = 5.0  # Maximum memory per chunk in GB (conservative)
            max_chunk_size = int((max_chunk_memory_gb * (1024**3)) / (n_genes * 4))
            # But don't make chunks too small (inefficient) or too large
            # For 90k genes: 5GB / (90k * 4 bytes) ≈ 13,800 samples
            chunk_size = max(1000, min(max_chunk_size, 20000))  # Between 1k and 20k samples
            
            if verbose:
                print(f"  Using chunk size: {chunk_size:,} samples (max {max_chunk_size:,} based on {max_chunk_memory_gb}GB limit)")
            
            # Initialize aggregators (we'll accumulate as we process chunks)
            gene_max_sample_idx = torch.zeros(n_genes, dtype=torch.long, device=device)
            gene_mean = torch.zeros(n_genes, dtype=torch.float32, device=device)
            gene_min = torch.full((n_genes,), float('inf'), dtype=torch.float32, device=device)
            gene_max = torch.full((n_genes,), float('-inf'), dtype=torch.float32, device=device)
            gene_nonzero = torch.zeros(n_genes, dtype=torch.int32, device=device)
            
            # If return_attention_matrices=True, allocate full matrix (memory-intensive!)
            if return_attention_matrices:
                if verbose:
                    print(f"  Warning: return_attention_matrices=True requires storing full matrix ({estimated_memory_gb:.1f} GB)")
                gene_attention_weights_torch = torch.zeros((n_samples, n_genes), dtype=torch.float32, device=device)
            else:
                gene_attention_weights_torch = None
            
            # Process in chunks - compute gene attention, normalize, and aggregate all in one pass
            all_values_for_percentile = []
            
            chunk_iter = tqdm(range(0, n_samples, chunk_size), desc="  Processing chunks", disable=not verbose)
            for chunk_start in chunk_iter:
                chunk_end = min(chunk_start + chunk_size, n_samples)
                chunk_attention = attention_weights_nmf_torch[chunk_start:chunk_end]
                
                # Compute gene attention for this chunk only
                chunk_gene_attention = torch.mm(chunk_attention, nmf_H_torch)  # (chunk_size, n_genes)
                
                # Normalize if requested (in-place on chunk)
                if normalize:
                    row_sums = chunk_gene_attention.sum(dim=1, keepdim=True)
                    row_sums = torch.clamp(row_sums, min=1.0)
                    chunk_gene_attention = chunk_gene_attention / row_sums
                
                # Store chunk in full matrix if requested
                if return_attention_matrices:
                    gene_attention_weights_torch[chunk_start:chunk_end] = chunk_gene_attention
                
                # Update running statistics
                chunk_max, chunk_max_idx = torch.max(chunk_gene_attention, dim=0)
                chunk_max_idx = chunk_max_idx + chunk_start
                update_mask = chunk_max > gene_max
                gene_max[update_mask] = chunk_max[update_mask]
                gene_max_sample_idx[update_mask] = chunk_max_idx[update_mask]
                
                gene_mean += chunk_gene_attention.sum(dim=0)
                gene_min = torch.minimum(gene_min, chunk_gene_attention.min(dim=0)[0])
                
                # Sample values for percentile
                if nonzero_threshold is None:
                    sample_size = max(1000, chunk_gene_attention.numel() // 100)
                    flat_chunk = chunk_gene_attention.flatten()
                    sample_indices = torch.randint(0, flat_chunk.numel(), (min(sample_size, flat_chunk.numel()),), device=device)
                    all_values_for_percentile.append(flat_chunk[sample_indices])
                
                # Free chunk immediately (unless we're storing it)
                if not return_attention_matrices:
                    del chunk_gene_attention
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            gene_mean = gene_mean / n_samples
            
            # Compute threshold from sampled values (use CPU numpy for memory efficiency)
            if nonzero_threshold is None:
                if all_values_for_percentile:
                    # Convert to numpy and compute percentile (more memory-efficient)
                    # Don't concatenate all at once - process in batches if needed
                    if len(all_values_for_percentile) > 0:
                        # Convert first chunk to numpy
                        all_samples_np = all_values_for_percentile[0].cpu().numpy()
                        # Add remaining chunks
                        for chunk in all_values_for_percentile[1:]:
                            all_samples_np = np.concatenate([all_samples_np, chunk.cpu().numpy()])
                        threshold = np.percentile(all_samples_np, 1.0)
                        del all_values_for_percentile, all_samples_np
                    else:
                        threshold = gene_mean.min().item() * 0.1
                else:
                    threshold = gene_mean.min().item() * 0.1
            else:
                threshold = nonzero_threshold
            
            # Second pass: count nonzero (recompute chunks)
            if verbose:
                print(f"  ✓ Computing nonzero counts (second pass)")
            chunk_iter = tqdm(range(0, n_samples, chunk_size), desc="  Counting nonzero", disable=not verbose)
            for chunk_start in chunk_iter:
                chunk_end = min(chunk_start + chunk_size, n_samples)
                chunk_attention = attention_weights_nmf_torch[chunk_start:chunk_end]
                chunk_gene_attention = torch.mm(chunk_attention, nmf_H_torch)
                
                # Normalize if needed
                if normalize:
                    row_sums = chunk_gene_attention.sum(dim=1, keepdim=True)
                    row_sums = torch.clamp(row_sums, min=1.0)
                    chunk_gene_attention = chunk_gene_attention / row_sums
                
                gene_nonzero += (chunk_gene_attention > threshold).sum(dim=0).int()
                del chunk_gene_attention
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            gene_pct_nonzero = (gene_nonzero.float() / n_samples) * 100.0
            
            # Convert to CPU numpy for DataFrame construction (free GPU memory immediately)
            gene_max_sample_idx_np = gene_max_sample_idx.cpu().numpy()
            gene_mean_np = gene_mean.cpu().numpy()
            gene_min_np = gene_min.cpu().numpy()
            gene_max_np = gene_max.cpu().numpy()
            gene_nonzero_np = gene_nonzero.cpu().numpy()
            gene_pct_nonzero_np = gene_pct_nonzero.cpu().numpy()
            
            # Free GPU tensors immediately
            del gene_max_sample_idx, gene_mean, gene_min, gene_max, gene_nonzero, gene_pct_nonzero
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Move attention_weights_nmf to CPU for NMF aggregation to free GPU memory
            if verbose:
                print(f"  ✓ Moving attention weights to CPU for NMF aggregation (freeing GPU memory)")
            attention_weights_nmf_cpu = attention_weights_nmf_torch.cpu().numpy()
            # Preserve attention_weights_nmf if return_attention_matrices=True
            if return_attention_matrices:
                nmf_attention_matrix = attention_weights_nmf_cpu.copy()
            del attention_weights_nmf_torch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Convert gene_attention_weights to CPU if we stored it
            if return_attention_matrices:
                if verbose:
                    print(f"  ✓ Moving gene attention matrix to CPU")
                gene_attention_matrix = gene_attention_weights_torch.cpu().numpy()
                del gene_attention_weights_torch
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            # Note: gene_attention_matrix stays None if return_attention_matrices=False
            # (it was initialized to None at function start)
            
            # Reassign to original variable names for consistency with rest of code
            gene_max_sample_idx = gene_max_sample_idx_np
            gene_mean = gene_mean_np
            gene_min = gene_min_np
            gene_max = gene_max_np
            gene_nonzero = gene_nonzero_np
            gene_pct_nonzero = gene_pct_nonzero_np
            
            # Skip to NMF aggregation (gene aggregation is done)
            skip_gene_aggregation = True
            use_cpu_for_nmf = True  # Flag to use CPU for NMF aggregation
        else:
            # Full matrix multiplication (fits in memory)
            use_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 7  # Volta+
            if use_fp16:
                with torch.cuda.amp.autocast():
                    gene_attention_weights_torch = torch.mm(
                        attention_weights_nmf_torch.half(), 
                        nmf_H_torch.half()
                    ).float()
            else:
                gene_attention_weights_torch = torch.mm(attention_weights_nmf_torch, nmf_H_torch)
            
            # Normalize if requested (GPU)
            if normalize:
                if verbose:
                    print(f"  ✓ Normalizing gene attention weights (GPU)")
                row_sums = gene_attention_weights_torch.sum(dim=1, keepdim=True)
                row_sums = torch.clamp(row_sums, min=1.0)
                gene_attention_weights_torch = gene_attention_weights_torch / row_sums
            
            n_samples, n_genes = gene_attention_weights_torch.shape
            
            if verbose:
                print(f"  ✓ Traced to genes: {gene_attention_weights_torch.shape}")
            
            # Step 3: Aggregate gene features on GPU (optimized for large matrices)
            if verbose:
                print(f"  ✓ Aggregating gene features (GPU, optimized)")
            
            # Use vectorized operations - much faster than chunked loops for most GPUs
            # For very large matrices (>1M samples), still chunk to avoid OOM
            skip_gene_aggregation = False
            if n_samples > 1000000:
                chunk_size = 100000  # Process 100k samples at a time
                gene_max_sample_idx = torch.zeros(n_genes, dtype=torch.long, device=device)
                gene_mean = torch.zeros(n_genes, dtype=torch.float32, device=device)
                gene_min = torch.full((n_genes,), float('inf'), dtype=torch.float32, device=device)
                gene_max = torch.full((n_genes,), float('-inf'), dtype=torch.float32, device=device)
                
                # Process in chunks with progress
                chunk_iter = tqdm(range(0, n_samples, chunk_size), desc="  Aggregating", disable=not verbose)
                for chunk_start in chunk_iter:
                    chunk_end = min(chunk_start + chunk_size, n_samples)
                    chunk = gene_attention_weights_torch[chunk_start:chunk_end]
                    
                    # Update running statistics
                    chunk_max, chunk_max_idx = torch.max(chunk, dim=0)
                    chunk_max_idx = chunk_max_idx + chunk_start
                    update_mask = chunk_max > gene_max
                    gene_max[update_mask] = chunk_max[update_mask]
                    gene_max_sample_idx[update_mask] = chunk_max_idx[update_mask]
                    
                    gene_mean += chunk.sum(dim=0)
                    gene_min = torch.minimum(gene_min, chunk.min(dim=0)[0])
                
                gene_mean = gene_mean / n_samples
            else:
                # For smaller datasets, use full vectorized operations (much faster)
                gene_max, gene_max_sample_idx = torch.max(gene_attention_weights_torch, dim=0)
                gene_mean = gene_attention_weights_torch.mean(dim=0)
                gene_min = gene_attention_weights_torch.min(dim=0)[0]
            
            # Compute threshold using chunked percentile (memory-efficient)
            if nonzero_threshold is None:
                # Use approximate percentile: sample and compute
                # For very large arrays, compute percentile on a sample
                sample_size = min(100000, n_samples * n_genes)
                if sample_size < n_samples * n_genes:
                    # Sample random indices
                    flat_indices = torch.randint(0, n_samples * n_genes, (sample_size,), device=device)
                    sample_values = gene_attention_weights_torch.flatten()[flat_indices]
                    threshold = torch.quantile(sample_values, 0.01).item()
                else:
                    threshold = torch.quantile(gene_attention_weights_torch, 0.01).item()
            else:
                threshold = nonzero_threshold
            
            # Count nonzero (vectorized if possible, chunked for very large)
            if n_samples > 1000000:
                gene_nonzero = torch.zeros(n_genes, dtype=torch.int32, device=device)
                chunk_size = 100000
                chunk_iter = tqdm(range(0, n_samples, chunk_size), desc="  Counting nonzero", disable=not verbose)
                for chunk_start in chunk_iter:
                    chunk_end = min(chunk_start + chunk_size, n_samples)
                    chunk = gene_attention_weights_torch[chunk_start:chunk_end]
                    gene_nonzero += (chunk > threshold).sum(dim=0).int()
            else:
                gene_nonzero = (gene_attention_weights_torch > threshold).sum(dim=0).int()
            
            gene_pct_nonzero = (gene_nonzero.float() / n_samples) * 100.0
            
            # Preserve matrices if return_attention_matrices=True (before converting to CPU)
            if return_attention_matrices:
                if verbose:
                    print(f"  ✓ Preserving attention matrices for return")
                gene_attention_matrix = gene_attention_weights_torch.cpu().numpy()
                nmf_attention_matrix = attention_weights_nmf_torch.cpu().numpy()
            else:
                gene_attention_matrix = None
                nmf_attention_matrix = None
            
            # Convert to CPU numpy for DataFrame construction
            gene_max_sample_idx = gene_max_sample_idx.cpu().numpy()
            gene_mean = gene_mean.cpu().numpy()
            gene_min = gene_min.cpu().numpy()
            gene_max = gene_max.cpu().numpy()
            gene_nonzero = gene_nonzero.cpu().numpy()
            gene_pct_nonzero = gene_pct_nonzero.cpu().numpy()
            
            # Clean up GPU memory
            del gene_attention_weights_torch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Step 4: Aggregate NMF features
        # Check if we need to use CPU (memory was freed in chunked path)
        if 'use_cpu_for_nmf' in locals() and use_cpu_for_nmf:
            # Use CPU numpy array (already moved above)
            if verbose:
                print(f"  ✓ Aggregating NMF features (CPU)")
            nmf_max_sample_idx = np.argmax(attention_weights_nmf_cpu, axis=0)
            nmf_mean = np.mean(attention_weights_nmf_cpu, axis=0)
            nmf_min = np.min(attention_weights_nmf_cpu, axis=0)
            nmf_max = np.max(attention_weights_nmf_cpu, axis=0)
            
            # Compute NMF threshold (sampled for memory efficiency)
            if nonzero_threshold is None:
                sample_size = min(100000, attention_weights_nmf_cpu.size)
                flat_nmf = attention_weights_nmf_cpu.flatten()
                sample_indices = np.random.choice(flat_nmf.size, size=min(sample_size, flat_nmf.size), replace=False)
                nmf_threshold = np.percentile(flat_nmf[sample_indices], 1.0)
                del flat_nmf
            else:
                nmf_threshold = nonzero_threshold
            
            nmf_nonzero = np.sum(attention_weights_nmf_cpu > nmf_threshold, axis=0)
            nmf_pct_nonzero = (nmf_nonzero / n_samples) * 100.0
            
            # Clean up
            del attention_weights_nmf_cpu
        else:
            # Use GPU (original path for smaller datasets)
            nmf_max_sample_idx = torch.argmax(attention_weights_nmf_torch, dim=0).cpu().numpy()
            nmf_mean = attention_weights_nmf_torch.mean(dim=0).cpu().numpy()
            nmf_min = attention_weights_nmf_torch.min(dim=0)[0].cpu().numpy()
            nmf_max = attention_weights_nmf_torch.max(dim=0)[0].cpu().numpy()
            
            # Compute NMF threshold
            if nonzero_threshold is None:
                sample_size = min(100000, n_samples * n_nmf_features)
                if sample_size < n_samples * n_nmf_features:
                    flat_indices = torch.randint(0, n_samples * n_nmf_features, (sample_size,), device=device)
                    sample_values = attention_weights_nmf_torch.flatten()[flat_indices]
                    nmf_threshold = torch.quantile(sample_values, 0.01).item()
                else:
                    nmf_threshold = torch.quantile(attention_weights_nmf_torch, 0.01).item()
            else:
                nmf_threshold = nonzero_threshold
            
            nmf_nonzero = (attention_weights_nmf_torch > nmf_threshold).sum(dim=0).int().cpu().numpy()
            nmf_pct_nonzero = (nmf_nonzero / n_samples) * 100.0
            
            # Preserve NMF matrix if return_attention_matrices=True
            # Note: gene_attention_matrix was already preserved earlier (at line 3078 or 2965)
            if return_attention_matrices:
                # Get NMF matrix from torch tensor (it should exist here)
                # gene_attention_matrix should already be set from earlier in the function
                nmf_attention_matrix = attention_weights_nmf_torch.cpu().numpy()
            else:
                # Ensure it's None (already initialized to None at function start)
                nmf_attention_matrix = None
            
            # Clean up GPU memory
            if 'gene_attention_weights_torch' in locals():
                del gene_attention_weights_torch
            del attention_weights_nmf_torch, nmf_H_torch
            if 'cuda' in str(device):
                torch.cuda.empty_cache()
    else:
        # CPU path: use GPU for matrix multiplication if available, CPU for aggregation
        # This hybrid approach is much faster than pure CPU
        use_gpu_for_matmul = torch.cuda.is_available() if 'torch' in globals() else False
        
        if isinstance(nmf_H, torch.Tensor):
            nmf_H_numpy = nmf_H.detach().cpu().numpy()
            nmf_H_torch = nmf_H.to(device) if use_gpu_for_matmul and device is not None else None
        else:
            nmf_H_numpy = nmf_H
            nmf_H_torch = torch.from_numpy(nmf_H).float().to(device) if use_gpu_for_matmul and device is not None else None
        
        # For very large matrices, do everything in chunks
        n_samples, n_nmf_features = attention_weights_nmf.shape
        n_genes = nmf_H_numpy.shape[1]
        
        if verbose:
            if use_gpu_for_matmul:
                print(f"  ✓ Tracing to genes (GPU matmul, CPU aggregation): {attention_weights_nmf.shape} @ {nmf_H_numpy.shape}")
            else:
                print(f"  ✓ Tracing to genes (CPU, chunked, no full matrix): {attention_weights_nmf.shape} @ {nmf_H_numpy.shape}")
        
        # Initialize aggregators (we'll accumulate as we process chunks)
        gene_max_sample_idx = np.zeros(n_genes, dtype=np.int64)
        gene_mean = np.zeros(n_genes, dtype=np.float32)
        gene_min = np.full(n_genes, np.inf, dtype=np.float32)
        gene_max = np.full(n_genes, -np.inf, dtype=np.float32)
        gene_nonzero = np.zeros(n_genes, dtype=np.int32)
        
        # If return_attention_matrices=True, allocate full matrix (memory-intensive!)
        if return_attention_matrices:
            estimated_memory_gb = (n_samples * n_genes * 4) / (1024**3)
            if verbose:
                print(f"  Warning: return_attention_matrices=True requires storing full matrix ({estimated_memory_gb:.1f} GB)")
            gene_attention_weights = np.zeros((n_samples, n_genes), dtype=np.float32)
        else:
            gene_attention_weights = None
        
        # Process in chunks - compute gene attention, normalize, and aggregate all in one pass
        # Use larger chunks for better performance
        chunk_size = min(100000, n_samples)  # Increased from 50k to 100k
        all_values_for_percentile = []  # Collect samples for percentile
        
        chunk_iter = tqdm(range(0, n_samples, chunk_size), desc="  Processing chunks", disable=not verbose)
        for chunk_start in chunk_iter:
            chunk_end = min(chunk_start + chunk_size, n_samples)
            chunk_attention = attention_weights_nmf[chunk_start:chunk_end]
            
            # Compute gene attention for this chunk only
            # Use GPU for matrix multiplication if available (much faster)
            if use_gpu_for_matmul and nmf_H_torch is not None:
                chunk_attention_torch = torch.from_numpy(chunk_attention).float().to(device)
                chunk_gene_attention_torch = torch.mm(chunk_attention_torch, nmf_H_torch)
                chunk_gene_attention = chunk_gene_attention_torch.cpu().numpy()
                del chunk_attention_torch, chunk_gene_attention_torch
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                chunk_gene_attention = chunk_attention @ nmf_H_numpy  # (chunk_size, n_genes)
            
            # Normalize if requested (in-place on chunk)
            if normalize:
                row_sums = chunk_gene_attention.sum(axis=1, keepdims=True)
                np.clip(row_sums, a_min=1.0, a_max=None, out=row_sums)
                np.divide(chunk_gene_attention, row_sums, out=chunk_gene_attention)
            
            # Store chunk in full matrix if requested
            if return_attention_matrices:
                gene_attention_weights[chunk_start:chunk_end] = chunk_gene_attention
            
            # Update running statistics
            chunk_max_idx = np.argmax(chunk_gene_attention, axis=0) + chunk_start
            chunk_max = np.max(chunk_gene_attention, axis=0)
            update_mask = chunk_max > gene_max
            gene_max[update_mask] = chunk_max[update_mask]
            gene_max_sample_idx[update_mask] = chunk_max_idx[update_mask]
            
            gene_mean += chunk_gene_attention.sum(axis=0)
            gene_min = np.minimum(gene_min, np.min(chunk_gene_attention, axis=0))
            
            # Sample values for percentile (don't store all)
            if nonzero_threshold is None:
                # Sample 1% of values from this chunk for percentile estimation
                sample_size = max(1000, chunk_gene_attention.size // 100)
                flat_chunk = chunk_gene_attention.flatten()
                sample_indices = np.random.choice(flat_chunk.size, size=min(sample_size, flat_chunk.size), replace=False)
                all_values_for_percentile.append(flat_chunk[sample_indices])
            
            # Free chunk immediately (unless we're storing it)
            if not return_attention_matrices:
                del chunk_gene_attention
        
        gene_mean = gene_mean / n_samples
        
        # Compute threshold from sampled values (much more memory-efficient)
        if nonzero_threshold is None:
            if all_values_for_percentile:
                all_samples = np.concatenate(all_values_for_percentile)
                threshold = np.percentile(all_samples, 1.0)
                del all_values_for_percentile, all_samples
            else:
                # Fallback: use a simple heuristic
                threshold = gene_mean.min() * 0.1
        else:
            threshold = nonzero_threshold
        
        # Second pass: count nonzero (recompute chunks, but don't store)
        if verbose:
            print(f"  ✓ Computing nonzero counts (second pass)")
        chunk_iter = tqdm(range(0, n_samples, chunk_size), desc="  Counting nonzero", disable=not verbose)
        for chunk_start in chunk_iter:
            chunk_end = min(chunk_start + chunk_size, n_samples)
            chunk_attention = attention_weights_nmf[chunk_start:chunk_end]
            
            # Use GPU for matrix multiplication if available
            if use_gpu_for_matmul and nmf_H_torch is not None:
                chunk_attention_torch = torch.from_numpy(chunk_attention).float().to(device)
                chunk_gene_attention_torch = torch.mm(chunk_attention_torch, nmf_H_torch)
                chunk_gene_attention = chunk_gene_attention_torch.cpu().numpy()
                del chunk_attention_torch, chunk_gene_attention_torch
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                chunk_gene_attention = chunk_attention @ nmf_H_numpy
            
            # Normalize if needed (same as first pass)
            if normalize:
                row_sums = chunk_gene_attention.sum(axis=1, keepdims=True)
                np.clip(row_sums, a_min=1.0, a_max=None, out=row_sums)
                np.divide(chunk_gene_attention, row_sums, out=chunk_gene_attention)
            
            gene_nonzero += np.sum(chunk_gene_attention > threshold, axis=0).astype(np.int32)
            del chunk_gene_attention  # Free immediately
        
        gene_pct_nonzero = (gene_nonzero / n_samples) * 100.0
        
        if verbose:
            print(f"  ✓ Traced to genes: ({n_samples}, {n_genes})")
        
        # Step 4: Aggregate NMF features (chunked)
        nmf_max_sample_idx = np.argmax(attention_weights_nmf, axis=0)
        nmf_mean = np.mean(attention_weights_nmf, axis=0)
        nmf_min = np.min(attention_weights_nmf, axis=0)
        nmf_max = np.max(attention_weights_nmf, axis=0)
        
        # Compute NMF threshold (sampled)
        if nonzero_threshold is None:
            sample_size = min(100000, attention_weights_nmf.size)
            flat_nmf = attention_weights_nmf.flatten()
            sample_indices = np.random.choice(flat_nmf.size, size=min(sample_size, flat_nmf.size), replace=False)
            nmf_threshold = np.percentile(flat_nmf[sample_indices], 1.0)
            del flat_nmf
        else:
            nmf_threshold = nonzero_threshold
        
        nmf_nonzero = np.sum(attention_weights_nmf > nmf_threshold, axis=0)
        nmf_pct_nonzero = (nmf_nonzero / n_samples) * 100.0
        
        # Preserve matrices if return_attention_matrices=True
        if return_attention_matrices:
            gene_attention_matrix = gene_attention_weights
            nmf_attention_matrix = attention_weights_nmf.copy()
        else:
            gene_attention_matrix = None
            nmf_attention_matrix = None
            # Free the large matrix as soon as we're done with it
            del gene_attention_weights
    
    if verbose:
        print(f"  ✓ Aggregated statistics")
    
    # Step 5: Build DataFrames efficiently (pre-allocate arrays)
    # Convert sample names to array once if needed
    sample_names_array = None
    if sample_names is not None:
        sample_names_array = np.asarray(sample_names)
    
    # Build gene DataFrame (use list of arrays for faster construction)
    gene_cols = [
        np.arange(n_genes, dtype=np.int32),  # feature_index
        gene_mean.astype(np.float32),  # mean_attention
        gene_min.astype(np.float32),  # min_attention
        gene_max.astype(np.float32),  # max_attention
        gene_nonzero.astype(np.int32),  # n_samples_nonzero
        gene_pct_nonzero.astype(np.float32),  # pct_samples_nonzero
    ]
    gene_col_names = ['feature_index', 'mean_attention', 'min_attention', 'max_attention', 
                      'n_samples_nonzero', 'pct_samples_nonzero']
    
    if gene_feature_names is not None:
        gene_cols.append(np.asarray(gene_feature_names))
        gene_col_names.append('feature_name')
    
    if sample_names_array is not None:
        gene_cols.append(sample_names_array[gene_max_sample_idx])
        gene_col_names.append('max_attention_sample')
    
    # Create DataFrame from dict (faster than column-by-column)
    gene_aggregated_df = pd.DataFrame(dict(zip(gene_col_names, gene_cols)))
    
    # Sort by mean attention (descending) - use numpy argsort for speed, then reindex
    sort_idx = np.argsort(-gene_mean)  # Negative for descending
    gene_aggregated_df = gene_aggregated_df.iloc[sort_idx].reset_index(drop=True)
    
    # Build NMF DataFrame (same approach)
    nmf_cols = [
        np.arange(n_nmf_features, dtype=np.int32),  # feature_index
        nmf_mean.astype(np.float32),  # mean_attention
        nmf_min.astype(np.float32),  # min_attention
        nmf_max.astype(np.float32),  # max_attention
        nmf_nonzero.astype(np.int32),  # n_samples_nonzero
        nmf_pct_nonzero.astype(np.float32),  # pct_samples_nonzero
    ]
    nmf_col_names = ['feature_index', 'mean_attention', 'min_attention', 'max_attention',
                     'n_samples_nonzero', 'pct_samples_nonzero']
    
    # Always add feature_name column for NMF features
    if nmf_feature_names is not None:
        nmf_cols.append(np.asarray(nmf_feature_names))
    else:
        # Generate default names: factor1, factor2, factor3, ...
        nmf_cols.append(np.array([f'factor{i+1}' for i in range(n_nmf_features)]))
    nmf_col_names.append('feature_name')
    
    if sample_names_array is not None:
        nmf_cols.append(sample_names_array[nmf_max_sample_idx])
        nmf_col_names.append('max_attention_sample')
    
    nmf_aggregated_df = pd.DataFrame(dict(zip(nmf_col_names, nmf_cols)))
    
    # Sort by mean attention (descending)
    sort_idx = np.argsort(-nmf_mean)  # Negative for descending
    nmf_aggregated_df = nmf_aggregated_df.iloc[sort_idx].reset_index(drop=True)
    
    # Step 6: Add label column from metadata if available
    if metadata is not None and 'obs' in metadata:
        try:
            obs = metadata['obs']
            if 'label' in obs.columns and 'max_attention_sample' in gene_aggregated_df.columns:
                # Create a mapping from sample ID to label
                # Use obs_id if available, otherwise use index
                if 'obs_id' in obs.columns:
                    sample_to_label = dict(zip(obs['obs_id'], obs['label']))
                else:
                    sample_to_label = dict(zip(obs.index, obs['label']))
                
                # Add label column to gene_aggregated_df
                gene_aggregated_df['label'] = gene_aggregated_df['max_attention_sample'].map(sample_to_label)
                
                # Add label column to nmf_aggregated_df if it has max_attention_sample
                if 'max_attention_sample' in nmf_aggregated_df.columns:
                    nmf_aggregated_df['label'] = nmf_aggregated_df['max_attention_sample'].map(sample_to_label)
                
                if verbose:
                    n_labeled_genes = gene_aggregated_df['label'].notna().sum()
                    n_labeled_nmf = nmf_aggregated_df['label'].notna().sum() if 'label' in nmf_aggregated_df.columns else 0
                    print(f"  ✓ Added label column: {n_labeled_genes:,} genes, {n_labeled_nmf:,} NMF components")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not add label column from metadata: {e}")
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n✓ Completed! Processed {n_samples:,} samples in {elapsed:.2f}s")
        print(f"  Gene features aggregated: {len(gene_aggregated_df):,} genes")
        print(f"  NMF features aggregated: {len(nmf_aggregated_df):,} components")
        print(f"\nSummary Statistics:")
        print(f"  Gene attention - Mean: {gene_mean.mean():.6f}, Max: {gene_mean.max():.6f}")
        print(f"  NMF attention - Mean: {nmf_mean.mean():.6f}, Max: {nmf_mean.max():.6f}")
        print(f"  Genes with >0 attention in >1% samples: {(gene_pct_nonzero > 1).sum():,}")
        print(f"  NMF components with >0 attention in >1% samples: {(nmf_pct_nonzero > 1).sum():,}")
    
    # Save dataframes as parquet files if save_dir is provided
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        gene_file = save_path / 'gene_attention_aggregated.parquet'
        nmf_file = save_path / 'nmf_attention_aggregated.parquet'
        
        # Check if files exist (edge case: if only one file exists, or if force was changed)
        if not force:
            if gene_file.exists():
                raise FileExistsError(
                    f"File {gene_file} already exists. Set force=True to overwrite."
                )
            if nmf_file.exists():
                raise FileExistsError(
                    f"File {nmf_file} already exists. Set force=True to overwrite."
                )
        
        # Save as parquet
        if verbose:
            print(f"\n  Saving dataframes to {save_dir}...")
        
        gene_aggregated_df.to_parquet(gene_file, index=False, engine='pyarrow')
        nmf_aggregated_df.to_parquet(nmf_file, index=False, engine='pyarrow')
        
        if verbose:
            print(f"  ✓ Saved: {gene_file}")
            print(f"  ✓ Saved: {nmf_file}")
        
        # Save attention matrices if return_attention_matrices=True and they exist
        if return_attention_matrices and gene_attention_matrix is not None and nmf_attention_matrix is not None:
            gene_matrix_file = save_path / 'gene_attention_matrix.npy'
            nmf_matrix_file = save_path / 'nmf_attention_matrix.npy'
            if verbose:
                print(f"  Saving attention matrices...")
            np.save(gene_matrix_file, gene_attention_matrix)
            np.save(nmf_matrix_file, nmf_attention_matrix)
            if verbose:
                print(f"  ✓ Saved: {gene_matrix_file} (shape: {gene_attention_matrix.shape})")
                print(f"  ✓ Saved: {nmf_matrix_file} (shape: {nmf_attention_matrix.shape})")
    
    # Return with or without matrices based on return_attention_matrices
    if return_attention_matrices:
        return gene_aggregated_df, nmf_aggregated_df, gene_attention_matrix, nmf_attention_matrix
    else:
        return gene_aggregated_df, nmf_aggregated_df


def plot_nmf_factor_distributions(
    W: Union[np.ndarray, torch.Tensor],
    n_factors_to_plot: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
    bins: int = 50,
    kde: bool = True,
    title: Optional[str] = None,
    factor_names: Optional[list] = None,
    max_cols: int = 4,
    max_samples: int = 50000,
    sharex: bool = True,
    sharey: bool = False,
    log_x: bool = False,
    log_y: bool = False,
    filter_zeros: bool = False,
    zero_threshold: float = 1e-10,
    return_fig: bool = False,
) -> Optional["plt.Figure"]:  # type: ignore
    """
    Visualize the distribution of each NMF factor across all samples using a faceted plot.
    
    Creates a grid of subplots where each subplot shows the distribution (histogram + KDE)
    of one NMF factor across all samples. This helps identify factors with different
    distributions, sparsity patterns, or outliers.
    
    Optimized for large datasets by sampling when n_samples > max_samples.
    
    Parameters
    ----------
    W : np.ndarray or torch.Tensor
        NMF factor matrix of shape (n_samples, n_components).
        Each column represents one NMF factor, each row represents one sample.
    n_factors_to_plot : int, optional
        Number of factors to plot. If None, plots all factors.
        If specified, plots the first n_factors_to_plot factors.
    figsize : tuple, optional
        Figure size in inches (width, height). If None, auto-calculated based on grid size.
    bins : int, default 50
        Number of bins for histogram.
    kde : bool, default True
        If True, overlay a kernel density estimate (KDE) curve on the histogram.
        Automatically disabled for very large datasets (>100k samples) for performance.
    title : str, optional
        Overall plot title. If None, auto-generated.
    factor_names : list, optional
        Custom names for factors. If None, uses "Factor 1", "Factor 2", etc.
    max_cols : int, default 4
        Maximum number of columns in the subplot grid.
    max_samples : int, default 50000
        Maximum number of samples to use for plotting. If n_samples > max_samples,
        randomly samples max_samples for faster computation. Statistics (mean, std, etc.)
        are still computed on full dataset.
    sharex : bool, default True
        If True, share x-axis across all subplots. Makes it easier to compare value ranges.
    sharey : bool, default False
        If True, share y-axis across all subplots. Makes it easier to compare densities.
    log_x : bool, default False
        If True, use logarithmic scale for x-axis. Useful for visualizing sparse distributions
        with long tails. Automatically adds small epsilon to handle zeros.
    log_y : bool, default False
        If True, use logarithmic scale for y-axis (density). Useful when density values
        span multiple orders of magnitude.
    filter_zeros : bool, default False
        If True, exclude zero values from the distribution plot. Useful for focusing on
        the non-zero tail of sparse distributions. Statistics are still computed on full dataset.
    zero_threshold : float, default 1e-10
        Values below this threshold are considered "zero" when filter_zeros=True.
        Useful for filtering out numerical noise.
    return_fig : bool, default False
        If True, return the matplotlib figure for further customization.
        If False, display the plot and return None.
        
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The matplotlib figure if return_fig=True, otherwise None.
        
    Examples
    --------
    >>> from sparse_nmf import train_sparse_nmf, plot_nmf_factor_distributions
    >>> 
    >>> # Train NMF
    >>> X_nmf, nmf_model = train_sparse_nmf(X_sparse, n_components=256)
    >>> 
    >>> # Visualize first 12 factors (fast, even for large datasets)
    >>> plot_nmf_factor_distributions(X_nmf, n_factors_to_plot=12)
    >>> 
    >>> # Customize with factor names
    >>> factor_names = [f"Component {i+1}" for i in range(12)]
    >>> fig = plot_nmf_factor_distributions(
    ...     X_nmf, 
    ...     n_factors_to_plot=12,
    ...     factor_names=factor_names,
    ...     kde=True,
    ...     return_fig=True
    ... )
    >>> fig.savefig("nmf_distributions.png")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "plot_nmf_factor_distributions requires matplotlib. "
            "Install with: pip install matplotlib"
        )
    
    # Convert to numpy if needed
    if isinstance(W, torch.Tensor):
        W_np = W.cpu().numpy()
    else:
        W_np = np.asarray(W)
    
    if W_np.ndim != 2:
        raise ValueError(f"W must be 2D array (n_samples, n_components), got shape {W_np.shape}")
    
    # W_np shape: (n_samples, n_factors)
    # Each row is a sample, each column is a factor
    # We plot the distribution of each factor (column) across all samples (rows)
    n_samples, n_components = W_np.shape
    
    # Sample data if too large (for plotting only, stats use full data)
    use_sampling = n_samples > max_samples
    if use_sampling:
        # Random sample for plotting
        sample_indices = np.random.choice(n_samples, size=max_samples, replace=False)
        W_plot = W_np[sample_indices]
        n_plot_samples = max_samples
    else:
        W_plot = W_np
        n_plot_samples = n_samples
    
    # Determine how many factors to plot
    if n_factors_to_plot is None:
        n_factors_to_plot = n_components
    else:
        n_factors_to_plot = min(n_factors_to_plot, n_components)
    
    # Calculate grid dimensions
    n_cols = min(max_cols, n_factors_to_plot)
    n_rows = int(np.ceil(n_factors_to_plot / n_cols))
    
    # Auto-calculate figsize if not provided
    if figsize is None:
        width = n_cols * 3.5
        height = n_rows * 2.5
        figsize = (width, height)
    
    # Pre-compute statistics for all factors (vectorized, uses full dataset)
    # axis=0 means across samples (rows), giving one stat per factor (column)
    factor_stats = {
        'mean': W_np.mean(axis=0),      # Shape: (n_components,)
        'std': W_np.std(axis=0),         # Shape: (n_components,)
        'median': np.median(W_np, axis=0),  # Shape: (n_components,)
    }
    
    # Disable KDE for very large datasets (too slow)
    if kde and n_samples > 100000:
        kde = False
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, 
                            sharex=sharex, sharey=sharey)
    axes = axes.flatten()
    
    # Generate factor names if not provided
    if factor_names is None:
        factor_names = [f"Factor {i+1}" for i in range(n_components)]
    elif len(factor_names) < n_components:
        # Extend with default names if not enough provided
        factor_names = list(factor_names) + [f"Factor {i+1}" for i in range(len(factor_names), n_components)]
    
    # Pre-compute ranges - use filtered data if filter_zeros is True
    if filter_zeros:
        # Compute ranges on non-zero values only
        factor_ranges = []
        for i in range(n_factors_to_plot):
            factor_values = W_plot[:, i]
            factor_values_nonzero = factor_values[factor_values > zero_threshold]
            if len(factor_values_nonzero) > 0:
                factor_ranges.append((factor_values_nonzero.min(), factor_values_nonzero.max()))
            else:
                # All zeros, use full range as fallback
                factor_ranges.append((W_np[:, i].min(), W_np[:, i].max()))
    else:
        # Use full data range
        factor_ranges = [(W_np[:, i].min(), W_np[:, i].max()) for i in range(n_factors_to_plot)]
    
    # Plot each factor (vectorized where possible)
    for i in range(n_factors_to_plot):
        ax = axes[i]
        # W_plot[:, i] gets all samples (rows) for factor i (column i)
        # This gives us the distribution of factor i across all samples
        factor_values_plot = W_plot[:, i]  # Sampled data for plotting
        factor_values_full = W_np[:, i]     # Full data for statistics
        
        # Filter zeros if requested (for plotting only, stats use full data)
        if filter_zeros:
            factor_values_plot = factor_values_plot[factor_values_plot > zero_threshold]
            if len(factor_values_plot) == 0:
                # All zeros, skip plotting
                ax.text(0.5, 0.5, 'All zeros', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_title(factor_names[i], fontsize=10, fontweight='bold')
                continue
        
        # Handle log scale for x-axis
        if log_x:
            # For log scale, use the filtered range if filter_zeros is True
            if filter_zeros and len(factor_values_plot) > 0:
                x_min = max(factor_ranges[i][0], zero_threshold)
                x_max = factor_ranges[i][1]
            else:
                # Add small epsilon to handle zeros
                epsilon = zero_threshold
                x_min = max(factor_ranges[i][0], epsilon)
                x_max = factor_ranges[i][1]
            
            factor_values_plot_log = np.log10(factor_values_plot)
            # Compute histogram on log scale
            counts, bin_edges = np.histogram(factor_values_plot_log, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # Convert back to original scale for display
            bin_centers_display = 10**bin_centers
            bin_widths = 10**bin_edges[1:] - 10**bin_edges[:-1]
        else:
            # Compute histogram (fast, uses sampled data)
            # If filtering zeros, use the filtered range for better visualization
            if filter_zeros and len(factor_values_plot) > 0:
                # Use filtered range for better visualization
                counts, bin_edges = np.histogram(
                    factor_values_plot, 
                    bins=bins, 
                    density=True,
                    range=(factor_ranges[i][0], factor_ranges[i][1])
                )
            else:
                counts, bin_edges = np.histogram(factor_values_plot, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_centers_display = bin_centers
            bin_widths = bin_edges[1] - bin_edges[0]
        
        # Plot histogram using bar plot (faster than ax.hist)
        ax.bar(bin_centers_display, counts, width=bin_widths, 
               alpha=0.6, color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Set log scale if requested
        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        
        # Add KDE if requested (works with filtered data too)
        if kde and len(factor_values_plot) > 1:  # Need at least 2 points for KDE
            try:
                from scipy import stats
                # Use smaller sample for KDE if dataset is very large
                kde_sample_size = min(10000, len(factor_values_plot))
                if len(factor_values_plot) > kde_sample_size:
                    kde_indices = np.random.choice(len(factor_values_plot), size=kde_sample_size, replace=False)
                    kde_values = factor_values_plot[kde_indices]
                else:
                    kde_values = factor_values_plot
                
                # Compute KDE
                kde_obj = stats.gaussian_kde(kde_values)
                if log_x:
                    # For log scale, create range in log space then convert
                    x_min = max(factor_ranges[i][0], zero_threshold if filter_zeros else 1e-10)
                    x_max = factor_ranges[i][1]
                    x_range_log = np.linspace(np.log10(x_min), np.log10(x_max), 200)
                    x_range = 10**x_range_log
                else:
                    x_range = np.linspace(factor_ranges[i][0], factor_ranges[i][1], 200)
                kde_values_plot = kde_obj(x_range)
                ax.plot(x_range, kde_values_plot, 'r-', linewidth=2, label='KDE', alpha=0.8)
            except (ImportError, np.linalg.LinAlgError):
                # scipy not available or singular matrix, skip KDE
                pass
        
        # Get pre-computed statistics (from full dataset)
        mean_val = factor_stats['mean'][i]
        std_val = factor_stats['std'][i]
        median_val = factor_stats['median'][i]
        
        # Add vertical lines for mean and median
        ax.axvline(mean_val, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Median: {median_val:.3f}')
        
        # Set title and labels
        ax.set_title(factor_names[i], fontsize=10, fontweight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Simplified legend (faster rendering)
        if kde:
            ax.legend(fontsize=7, loc='best', framealpha=0.8)
        else:
            ax.legend(fontsize=7, loc='best', framealpha=0.8)
        
        # Add text box with statistics (computed from full dataset)
        stats_text = f'n={n_samples:,}'
        if use_sampling:
            stats_text += f' (plotted: {n_plot_samples:,})'
        if filter_zeros:
            n_nonzero = (factor_values_full > zero_threshold).sum()
            pct_nonzero = 100 * n_nonzero / n_samples
            stats_text += f'\nnon-zero: {n_nonzero:,} ({pct_nonzero:.1f}%)'
        stats_text += f'\nμ={mean_val:.3f}\nσ={std_val:.3f}\nmed={median_val:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for i in range(n_factors_to_plot, len(axes)):
        axes[i].axis('off')
    
    # Set overall title
    if title is None:
        title = f"NMF Factor Distributions (showing {n_factors_to_plot} of {n_components} factors)"
        if use_sampling:
            title += f" [sampled {n_plot_samples:,} of {n_samples:,} samples]"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave room for suptitle
    
    if return_fig:
        return fig
    else:
        plt.show()
        return None


def compute_attention_correlation(
    gene_attention_matrix: np.ndarray,
    X: Union[np.ndarray, spmatrix],
    obs_mask: Optional[np.ndarray] = None,
    stratify_by_unique_values: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute correlations between gene attention matrix and original data matrix.
    
    Computes Pearson and Spearman correlations between gene attention weights and
    the original gene expression/association matrix. Can stratify samples by the
    number of unique values per sample (useful for binary vs. continuous data).
    
    Parameters
    ----------
    gene_attention_matrix : np.ndarray
        Gene attention matrix, shape (n_samples, n_genes).
        Contains continuous attention scores for each sample-gene pair.
    X : np.ndarray or scipy.sparse matrix
        Original data matrix, shape (n_samples, n_genes).
        Can be sparse (scipy.sparse) or dense (numpy array).
    obs_mask : np.ndarray, optional
        Boolean mask to select subset of samples. If provided, X[obs_mask] is used.
        Should match the samples in gene_attention_matrix.
    stratify_by_unique_values : bool, default True
        If True, stratify samples by number of unique values per sample:
        - "2_unique": Binary data (2 unique values)
        - "3_unique": Ternary data (3 unique values)
        - "4+_unique": Continuous data (4+ unique values)
        If False, only computes global correlations.
    verbose : bool, default True
        If True, print summary information.
    
    Returns
    -------
    correlation_results_df : pd.DataFrame
        DataFrame with columns:
        - stratum: "2_unique", "3_unique", "4+_unique", or "all"
        - subset: "all" (all values) or "nonzero" (only nonzero values in X)
        - n_samples: Number of samples (for strata) or values (for global)
        - pearson: Pearson correlation coefficient
        - spearman: Spearman rank correlation coefficient
        - spearman_p: Spearman correlation p-value
    
    Examples
    --------
    >>> from sparse_nmf import compute_attention_correlation
    >>> import numpy as np
    >>> 
    >>> # Compute correlations
    >>> results_df = compute_attention_correlation(
    ...     gene_attention_matrix=gene_attention_matrix,
    ...     X=X,
    ...     obs_mask=obs_mask,
    ...     stratify_by_unique_values=True
    ... )
    >>> 
    >>> # Display results
    >>> print(results_df)
    """
    import pandas as pd
    from scipy.stats import spearmanr
    
    # Handle sparse matrices
    if obs_mask is not None:
        if hasattr(X, "toarray"):
            X_dense = X[obs_mask].toarray()
        else:
            X_dense = X[obs_mask]
    else:
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = X
    
    # Verify shapes match
    if gene_attention_matrix.shape != X_dense.shape:
        raise ValueError(
            f"Shape mismatch: gene_attention_matrix {gene_attention_matrix.shape} "
            f"vs X {X_dense.shape}. Ensure obs_mask matches if provided."
        )
    
    n_samples, n_genes = X_dense.shape
    flat_gene_attention = gene_attention_matrix.flatten()
    flat_x = X_dense.flatten()
    
    correlation_results = []
    
    if stratify_by_unique_values:
        # Compute unique counts per sample (row)
        num_unique_per_sample = np.apply_along_axis(lambda r: len(np.unique(r)), 1, X_dense)
        
        strata = {
            "2_unique": np.where(num_unique_per_sample == 2)[0],
            "3_unique": np.where(num_unique_per_sample == 3)[0],
            "4+_unique": np.where(num_unique_per_sample >= 4)[0],
        }
        
        if verbose:
            print(f"Stratifying samples by unique values:")
            for label, idx in strata.items():
                print(f"  {label}: {len(idx):,} samples")
        
        # Compute correlations for each stratum
        for label, idx in strata.items():
            if len(idx) == 0:
                # No samples in this group
                correlation_results.append({
                    "stratum": label,
                    "subset": "all",
                    "n_samples": 0,
                    "pearson": np.nan,
                    "spearman": np.nan,
                    "spearman_p": np.nan,
                })
                correlation_results.append({
                    "stratum": label,
                    "subset": "nonzero",
                    "n_samples": 0,
                    "pearson": np.nan,
                    "spearman": np.nan,
                    "spearman_p": np.nan,
                })
                continue
            
            # Get rows (samples) in this stratum
            orig_sub = X_dense[idx, :]
            attn_sub = gene_attention_matrix[idx, :]
            
            # -- Correlation for all values --
            flat_orig = orig_sub.flatten()
            flat_attn = attn_sub.flatten()
            
            try:
                pearson_corr_all = np.corrcoef(flat_attn, flat_orig)[0, 1]
            except Exception:
                pearson_corr_all = np.nan
            
            try:
                spearman_corr_all, spearman_p_all = spearmanr(flat_attn, flat_orig)
            except Exception:
                spearman_corr_all, spearman_p_all = np.nan, np.nan
            
            correlation_results.append({
                "stratum": label,
                "subset": "all",
                "n_samples": len(idx),
                "pearson": pearson_corr_all,
                "spearman": spearman_corr_all,
                "spearman_p": spearman_p_all,
            })
            
            # -- Correlation for ONLY values where X is nonzero --
            nonzero_mask = flat_orig != 0
            if np.any(nonzero_mask):
                flat_orig_nz = flat_orig[nonzero_mask]
                flat_attn_nz = flat_attn[nonzero_mask]
                try:
                    pearson_corr_nz = np.corrcoef(flat_attn_nz, flat_orig_nz)[0, 1]
                except Exception:
                    pearson_corr_nz = np.nan
                try:
                    spearman_corr_nz, spearman_p_nz = spearmanr(flat_attn_nz, flat_orig_nz)
                except Exception:
                    spearman_corr_nz, spearman_p_nz = np.nan, np.nan
            else:
                pearson_corr_nz, spearman_corr_nz, spearman_p_nz = np.nan, np.nan, np.nan
            
            correlation_results.append({
                "stratum": label,
                "subset": "nonzero",
                "n_samples": int(np.sum(nonzero_mask)),
                "pearson": pearson_corr_nz,
                "spearman": spearman_corr_nz,
                "spearman_p": spearman_p_nz,
            })
    
    # Global correlation over all samples
    # -- ALL values --
    try:
        pearson_corr = np.corrcoef(flat_gene_attention, flat_x)[0, 1]
    except Exception:
        pearson_corr = np.nan
    
    try:
        spearman_corr, spearman_p = spearmanr(flat_gene_attention, flat_x)
    except Exception:
        spearman_corr, spearman_p = np.nan, np.nan
    
    correlation_results.append({
        "stratum": "all",
        "subset": "all",
        "n_samples": n_samples,
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "spearman_p": spearman_p,
    })
    
    # -- Only nonzero values in X --
    nonzero_mask_all = flat_x != 0
    if np.any(nonzero_mask_all):
        flat_attn_nz_all = flat_gene_attention[nonzero_mask_all]
        flat_x_nz_all = flat_x[nonzero_mask_all]
        try:
            pearson_corr_nz_all = np.corrcoef(flat_attn_nz_all, flat_x_nz_all)[0, 1]
        except Exception:
            pearson_corr_nz_all = np.nan
        try:
            spearman_corr_nz_all, spearman_p_nz_all = spearmanr(flat_attn_nz_all, flat_x_nz_all)
        except Exception:
            spearman_corr_nz_all, spearman_p_nz_all = np.nan, np.nan
    else:
        pearson_corr_nz_all, spearman_corr_nz_all, spearman_p_nz_all = np.nan, np.nan, np.nan
    
    correlation_results.append({
        "stratum": "all",
        "subset": "nonzero",
        "n_samples": int(np.sum(nonzero_mask_all)),
        "pearson": pearson_corr_nz_all,
        "spearman": spearman_corr_nz_all,
        "spearman_p": spearman_p_nz_all,
    })
    
    # Convert to DataFrame
    correlation_results_df = pd.DataFrame(correlation_results)
    
    if verbose:
        print(f"\nCorrelation Analysis Summary:")
        print(f"  Total samples: {n_samples:,}")
        print(f"  Total genes: {n_genes:,}")
        print(f"  Global correlation (all values):")
        print(f"    Pearson: {correlation_results_df.loc[(correlation_results_df['stratum'] == 'all') & (correlation_results_df['subset'] == 'all'), 'pearson'].values[0]:.4f}")
        print(f"    Spearman: {correlation_results_df.loc[(correlation_results_df['stratum'] == 'all') & (correlation_results_df['subset'] == 'all'), 'spearman'].values[0]:.4f}")
    
    return correlation_results_df

