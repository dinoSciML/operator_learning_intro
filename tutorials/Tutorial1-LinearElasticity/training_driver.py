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

import os, sys
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('../../')


from dinotorch_lite import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-output_type', '--output_type', type=str, default='full_state',\
                                                 help="Output type")
parser.add_argument('-architecture','--architecture', type=str, default= 'rbno', help = 'what kind of architecture')
parser.add_argument('-formulation','--formulation', type=str, default= 'l2', help = 'training formulation')

# Training hyperparameters
parser.add_argument('-batch_size', '--batch_size', type=int, default=32, help="Training batch size")
parser.add_argument('-n_epochs', '--n_epochs', type=int, default=100, help="Training epochs")
parser.add_argument('-n_train', '--n_train', type=int, default=800, help="Number of training data")
parser.add_argument('-n_test', '--n_test', type=int, default=200, help="Number of testing data")

# Architectural hyperparameters
parser.add_argument('-depth', '--depth', type=int, default=4, help="Number of hidden layers")
parser.add_argument('-width', '--width', type=int, default=256, help="Dimension for hidden latent representation")

# RBNO Hyperparameters
parser.add_argument('-rM', '--rM', type=int, default=100, help="Reduced dimension for input (RBNO)")
parser.add_argument('-rQ', '--rQ', type=int, default=100, help="Reduced dimension for output (RBNO)")

# FNO Hyperparameters
parser.add_argument('-modes1', '--modes1', type=int, default=4, help="Modes for the first dimension")
parser.add_argument('-modes2', '--modes2', type=int, default=4, help="Modes for the second dimension")

# DON Hyperparameters


args = parser.parse_args()
################################################################################
print(80*'#')
print(f' Architecture: {args.architecture.upper()}, Formulation: {args.formulation.upper()}'.center(80))
print(f' Output Type: {args.output_type.upper()}'.center(80))

# Initial checks
output_type = args.output_type.lower()
assert output_type in ['full_state','observable']
data_dir = 'data/'+output_type+'/'

architecture = args.architecture.lower()
assert architecture in ['rbno', 'fno', 'don']
if architecture in ['fno','don']:
    assert output_type == 'full_state'

formulation = args.formulation.lower()
assert formulation in ['l2','h1']

if formulation == 'h1':
    assert architecture == 'rbno', 'DIFNO and DIDON not implemented'

batch_size = args.batch_size
n_epochs = args.n_epochs

################################################################################
# Data loading
n_test = args.n_test

if architecture == 'rbno':
    rM = args.rM
    rQ = args.rQ
    mq_data_dict = np.load(data_dir+'mq_data_reduced.npz')
    m_data = mq_data_dict['m_data'][:,:rM]
    q_data = mq_data_dict['q_data'][:,:rQ]
    n_data,dM = m_data.shape
    _n_data, dQ = q_data.shape
    assert n_data == _n_data
    if formulation == 'h1':
        J_data_dict = np.load(data_dir+'JstarPhi_data_reduced.npz')
        J_data = J_data_dict['J_data'][:,:rQ,:rM]
        assert J_data.shape == (n_data,dQ,dM)

elif architecture in ['fno','don']:
    mq_data_dict = np.load(data_dir+'mq_data.npz')
    q_data = mq_data_dict['q_data']
    m_data = mq_data_dict['m_data']
    n_data,dM = m_data.shape
    _n_data, dQ = q_data.shape
    assert n_data == _n_data
    if architecture == 'fno':
        fno_metadata = np.load(data_dir+'fno_metadata.npz')
        d2v = fno_metadata['d2v_param']
        v2d = fno_metadata['v2d_param']
        nx = fno_metadata['nx']
        ny = fno_metadata['ny']

assert args.n_train + n_test  <= n_data and args.n_train > 0

m_train = torch.Tensor(m_data[:args.n_train])
q_train = torch.Tensor(q_data[:args.n_train])
m_test = torch.Tensor(m_data[-n_test:])
q_test = torch.Tensor(q_data[-n_test:])

# Set up datasets and loaders
if formulation == 'h1':
    J_train = torch.Tensor(J_data[:args.n_train])
    J_test = torch.Tensor(J_data[-n_test:])
    train_dataset = DINODataset(m_train,q_train, J_train)
    test_dataset = DINODataset(m_test,q_test, J_test)

else:
    train_dataset = L2Dataset(m_train,q_train)
    test_dataset = L2Dataset(m_test,q_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################################################
# Load the networks

if architecture == 'rbno':
    hidden_layer_list = args.depth*[args.width]
    model = MLP(input_size = dM, output_size=dQ, hidden_layer_list = hidden_layer_list).to(device)

elif architecture == 'fno':
    model_settings = fno2d_settings(modes1=args.modes1, modes2=args.modes2, width=args.width, n_layers=args.depth, d_out=2)
    model = VectorFNO2D(v2d=[d2v, d2v], d2v=[v2d, v2d], nx=nx, ny=ny, dim=2, settings=model_settings).to(device) 

elif architecture == 'don':
    raise


################################################################################
# Training loop

lr_scheduler = None
optimizer = torch.optim.Adam(model.parameters())

loss_func_l2 = normalized_f_mse

if formulation == 'l2':
    network, history = l2_training(model,loss_func_l2,train_loader, validation_loader,\
                     optimizer,lr_scheduler=lr_scheduler,n_epochs = n_epochs)
else:
    loss_func_jac = normalized_f_mse
    network, history = h1_training(model,loss_func_l2, loss_func_jac, train_loader, validation_loader,\
                             optimizer,lr_scheduler=lr_scheduler,n_epochs = n_epochs, verbose=True)


rel_error = evaluate_l2_error(model,validation_loader)

print('L2 relative error after training = ', rel_error)

model_name = architecture + '_'
if output_type == 'observable':
    model_name += output_type+'_'

model_name += formulation + '_ndata_'+str(args.n_train)

model_name += '.pth'

torch.save(model.state_dict(), data_dir+model_name)


print('Process completed.'.center(80))
