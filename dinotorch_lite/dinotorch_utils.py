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







