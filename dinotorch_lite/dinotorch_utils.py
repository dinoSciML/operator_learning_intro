# MIT License
# Copyright (c) 2025
#
# This is part of the dino_tutorial package
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# For additional questions contact Thomas O'Leary-Roseberry
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np



def squared_f_norm(A):
    return torch.sum(torch.square(A))

def squared_f_error(A_pred, A_true):
    return squared_f_norm(A_true - A_pred)

def f_mse(A_pred_batched, A_true_batched):
    return torch.mean(torch.vmap(squared_f_error, in_dims=(0, 0), out_dims=0)(A_pred_batched, A_true_batched), axis=0)

def normalized_f_mse(A_pred_batched, A_true_batched):
    err = f_mse(A_pred_batched, A_true_batched)
    normalization = torch.mean(torch.vmap(squared_f_norm)(A_true_batched), axis=0)
    return err / normalization


def weighted_l2_error(M: torch.sparse):
    def _weighted_l2_error(u_pred, u_true):
        x = u_pred-u_true
        Mx = torch.sparse.mm(M, torch.t(x))
        return torch.mean(torch.einsum("ij,ji->i", x, Mx),axis = 0)
    return _weighted_l2_error


# def weighted_l2_norm(M):
#     def _loss(x, reduction='sum'):
#         if x.dim() == 1:
#             # scalar: x^T M x
#             return x.dot(torch.sparse.mv(M, x))
#         # batch: x shape (B, n) -> per-sample quadratic forms
#         x2 = x.reshape(-1, x.size(-1))          # (B, n)
#         Mx = torch.sparse.mm(M, x2.T)           # (n, B)
#         q  = torch.einsum('bn,nb->b', x2, Mx)   # (B,)
#         if reduction == 'none': return q.view(*x.shape[:-1])
#         return q.sum() if reduction == 'sum' else q.mean()
#     return _loss

def weighted_l2_norm(M):
    spmm = (lambda A,B: A.matmul(B) if A.layout==torch.sparse_csr else torch.sparse.mm(A,B))
    def loss(x, reduction='sum'):
        X = x.reshape(-1, x.shape[-1])                 # (B,n)
        q = (X * spmm(M, X.t()).t()).sum(1)            # (B,)
        if x.dim()==1: return q[0]                     # scalar
        return q if reduction=='none' else (q.sum() if reduction=='sum' else q.mean())
    return loss

# def weighted_relative_mse(M: torch.sparse, tol=0.0):
#     def _weighted_relative_mse(pred, true):
#         return torch.mean(weighted_l2_norm(M, pred - true) / (weighted_l2_norm(M, true) + tol))

#     return _weighted_relative_mse

def weighted_squared_norm(M: torch.sparse, x: torch.Tensor) -> torch.Tensor:
    Mx = torch.sparse.mm(M, torch.t(x))
    return torch.einsum("ij,ji->i", x, Mx)


def weighted_relative_mse(M: torch.sparse, tol=0.0):
    def _weighted_relative_mse(pred, true):
        return torch.mean(weighted_squared_norm(M, pred - true) / (weighted_squared_norm(M, true) + tol))

    return _weighted_relative_mse




class L2Dataset(Dataset):
    """
    L2NO dataset
    Each sample is a pair of (m, u) where m is the parameter and u is the state
    """
    def __init__(self, m_data: torch.Tensor, u_data: torch.Tensor):
        """
        Initialize the dataset

        Input:
        - m_data: torch.Tensor, shape (n_data, m_dim)
        - u_data: torch.Tensor, shape (n_data, u_dim)
        """
        assert m_data.shape[0] == u_data.shape[0], "m_data and u_data must have the same number of samples"

        self.m_data = m_data
        self.u_data = u_data


    def __len__(self):
        return self.m_data.shape[0]


    def __getitem__(self, idx):
        return self.m_data[idx], self.u_data[idx]

class DINODataset(Dataset):
    """
    DINO dataset
    Each sample is a triplet of (m, u, J) where m is the parameter, u is the state and j is the jacobian
    """
    def __init__(self, m_data: torch.Tensor, u_data: torch.Tensor, J_data: torch.Tensor):
        """
        Initialize the dataset

        Input:
        - m_data: torch.Tensor, shape (n_data, m_dim)
        - u_data: torch.Tensor, shape (n_data, u_dim)
        - J_data: torch.Tensor, shape (n_data, u_dim, m_dim)
        """
        assert m_data.shape[0] == u_data.shape[0] == J_data.shape[0], "m_data, u_data and j_data must have the same number of samples"

        self.m_data = m_data
        self.u_data = u_data
        self.J_data = J_data


    def __len__(self):
        return self.m_data.shape[0]


    def __getitem__(self, idx):
        return self.m_data[idx], self.u_data[idx], self.J_data[idx]



from scipy.sparse import csr_matrix


def scipy_csr_to_torch_csr(A: csr_matrix) -> torch.Tensor:
    # SciPy's index arrays are usually int32; PyTorch CSR needs int64.
    crow = torch.from_numpy(A.indptr.astype(np.int64, copy=False))  # may copy only if type differs
    col  = torch.from_numpy(A.indices.astype(np.int64, copy=False)) # may copy only if type differs
    val  = torch.from_numpy(A.data)                                 # shares memory (no copy)
    return torch.sparse_csr_tensor(crow, col, val, size=A.shape)


