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


# Some of the code here is borrowed from 
# https://github.com/nickhnelsen/fourier-neural-mappings/tree/main


# def _get_act(act):
#     """
#     https://github.com/NeuralOperator/PINO/blob/master/models/utils.py
#     """
#     if act == 'tanh':
#         func = F.tanh
#     elif act == 'gelu':
#         func = F.gelu
#     elif act == 'relu':
#         func = F.relu_
#     elif act == 'elu':
#         func = F.elu_
#     elif act == 'leaky_relu':
#         func = F.leaky_relu_
#     else:
#         raise ValueError(f'{act} is not supported')
#     return func



# class GenericDense(nn.Module):
#     def __init__(self,  input_dim=50, hidden_layer_dim = 256, output_dim=20):
#         super().__init__()

#         self.hidden1 = nn.Linear(input_dim, hidden_layer_dim)
#         self.act1 = nn.GELU()
#         self.hidden2 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
#         self.act2 = nn.GELU()
#         self.hidden3 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
#         self.act3 = nn.GELU()
#         self.hidden4 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
#         self.act4 = nn.GELU()
#         self.output = nn.Linear(hidden_layer_dim, output_dim)

#     def forward(self, x):
#         x = self.act1(self.hidden1(x))
#         x = self.act2(self.hidden2(x))
#         x = self.act3(self.hidden3(x))
#         x = self.act4(self.hidden4(x))
#         x = self.output(x)
#         return x


class MLP(nn.Module):
    def __init__(self, input_size,  output_size, hidden_layer_list = 4*[256], activation_fn = torch.nn.GELU()):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_layer_size = input_size
        for hidden_size in hidden_layer_list:
            self.hidden_layers.append(nn.Linear(prev_layer_size, hidden_size))
            prev_layer_size = hidden_size
        self.output_layer = nn.Linear(prev_layer_size, output_size)
        self.activation_fn = activation_fn

    def __call__(self, x):
        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))
        x = self.output_layer(x)
        return x


class TwoLayerMLP(nn.Module):
    """
    Pointwise single hidden layer fully-connected neural network applied to last axis of input
    """
    def __init__(self, channels_in, channels_hid, channels_out, activation_fn = torch.nn.GELU()):
        super(TwoLayerMLP, self).__init__()
        
        self.fc1 = nn.Linear(channels_in, channels_hid)
        self.act = activation_fn
        self.fc2 = nn.Linear(channels_hid, channels_out)

    def forward(self, x):
        """
        Input shape (of x):     (..., channels_in)
        Output shape:           (..., channels_out)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x




