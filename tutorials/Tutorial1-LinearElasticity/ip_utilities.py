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

# This code was written mostly by Blake Christierson

import torch
import torch.optim as opt
from dinotorch_lite.misfits import WeightedQuadraticMisfit
from dinotorch_lite.dinotorch_utils import weighted_l2_norm


def map_estimate(neural_op,  model, R=None, iterations = 100,\
         output_type = 'observable', verbose = False):

    assert output_type.lower() in ['observable','full_state']
    if output_type.lower() == 'full_state':
        assert R is not None

    mean = model.prior.mean

    m = torch.tensor(mean.copy()[:], dtype=torch.float32)
    m.requires_grad = True
    
    m0 = torch.tensor(mean.copy()[:], dtype=torch.float32)
    m0.requires_grad = False
    
    # Optimize
    likelihood = WeightedQuadraticMisfit.from_hippylib(model.misfit)

    if output_type == 'observable':
        regularization = lambda m_r_surr:  torch.linalg.vector_norm(m - m0, 2)**2 
    else:
        # R = R.to(device)
        regularization = lambda m: 0.5*weighted_l2_norm(R)(m- m0)
    lbfgs = opt.LBFGS([m], line_search_fn='strong_wolfe')

    
    def closure():
        lbfgs.zero_grad()
        m_in = m.unsqueeze(0)  
        if output_type == 'observable':
            objective = likelihood(q=neural_op(m_in).squeeze(0)) + regularization(m)
        else:
            objective = likelihood(u=neural_op(m_in).squeeze(0)) + regularization(m)
        objective.backward(retain_graph=True)
        return objective
    
    iteration, gradnorm = 0, 1
    lbfgs_history = []
    for iteration in range(iterations):
        m_in = m.unsqueeze(0)  
        if output_type == 'observable':
            history = {'L': likelihood(q=neural_op(m_in).squeeze(0)).item(), 
                    'R': regularization(m)}
        else:
            history = {'L': likelihood(u=neural_op(m_in).squeeze(0)).item(), 
                    'R': regularization(m)}
        lbfgs_history.append(history)
        lbfgs.step(closure)
        if iteration %10 == 0 and verbose:
            print(' | '.join([f"Iteration: {iteration}"] + [f"{k}: {v:.3}" for k,v in lbfgs_history[-1].items()]))
    

    m = m.squeeze()
    m = m.to("cpu")

    return m.detach().numpy()

# def map_estimate(neural_op, m_r, model, iterations = 100,\
#          output_type = 'observable', verbose = False):

#     assert output_type.lower() in ['observable','full_state']

#     m_r_surr = torch.tensor(m_r.copy()[:], dtype=torch.float32)
#     m_r_surr.requires_grad = True
    
#     m_r_surr0 = torch.tensor(m_r.copy()[:], dtype=torch.float32)
#     m_r_surr0.requires_grad = False
    
#     # Optimize
#     likelihood = WeightedQuadraticMisfit.from_hippylib(model.misfit)
#     # likelihood.output_decoder = output_decoder

#     regularization = lambda m_r_surr:  torch.linalg.vector_norm(m_r_surr - m_r_surr0, 2)**2 
#     lbfgs = opt.LBFGS([m_r_surr], line_search_fn='strong_wolfe')
    
#     def closure():
#         lbfgs.zero_grad()
#         objective = likelihood(q=neural_op(m_r_surr)) + regularization(m_r_surr)
#         objective.backward(retain_graph=True)
#         return objective
    
#     iteration, gradnorm = 0, 1
#     lbfgs_history = []
#     for iteration in range(iterations):
#         lbfgs_history.append({
#             'L': likelihood(q=neural_op(m_r_surr)).item(), 
#             'R': regularization(m_r_surr)})
#         lbfgs.step(closure)
#         if iteration %10 == 0 and verbose:
#             print(' | '.join([f"Iteration: {iteration}"] + [f"{k}: {v:.3}" for k,v in lbfgs_history[-1].items()]))
    
#     q_surr = neural_op(m_r_surr)

#     return m_r_surr.detach().numpy()




