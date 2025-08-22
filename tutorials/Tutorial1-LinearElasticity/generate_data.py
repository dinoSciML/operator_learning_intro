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

import math
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
import sys, os
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
import hippyflow as hf

from linear_elasticity_model import *

sys.path.append('../../')

from dinotorch_lite.hp_utils import operator_to_array_with_dummy_vectors

################################################################################
# Set up the model

formulation = 'full_state'

assert formulation.lower() in ['full_state', 'pointwise']

settings = linear_elasticity_settings()
model = linear_elasticity_model(settings)

data_dir = 'data/'+formulation+'/'

################################################################################
# Generate training data

Vh = model.problem.Vh

mesh = Vh[hp.STATE].mesh()

if formulation.lower() == 'full_state':
	q_trial = dl.TrialFunction(Vh[hp.STATE])
	q_test = dl.TestFunction(Vh[hp.STATE])
	M = dl.PETScMatrix(mesh.mpi_comm())
	dl.assemble(dl.inner(q_trial,q_test) * dl.dx, tensor=M)
	B = hf.StateSpaceIdentityOperator(M, use_mass_matrix=False)
	output_decoder = None
elif formulation.lower() == 'pointwise':
	B = model.misfit.B
	q = dl.Vector()
	B.init_vector(q,0)
	dQ = q.get_local().shape[0]
	# Since the problems are
	output_decoder = np.eye(dQ)
else:
	raise

observable = hf.LinearStateObservable(model.problem,B)
prior = model.prior

dataGenerator = hf.DataGenerator(observable,prior)

nsamples = 1000
n_samples_pod = 250
pod_rank = 200


if formulation.lower() == 'full_state':
	dataGenerator.two_step_generate(nsamples,n_samples_pod = n_samples_pod, derivatives = (1,0),\
		 pod_rank = pod_rank, data_dir = data_dir)
elif formulation.lower() == 'pointwise':
	dataGenerator.generate(nsamples, derivatives = (1,0),output_decoder = output_decoder, data_dir = data_dir)
else: 
	raise

d2v_state = dl.dof_to_vertex_map(Vh[hp.STATE])
v2d_state = dl.vertex_to_dof_map(Vh[hp.STATE])
d2v_state = d2v_state.astype(np.int64)
v2d_state = v2d_state.astype(np.int64)

d2v_param = dl.dof_to_vertex_map(Vh[hp.PARAMETER])
v2d_param = dl.vertex_to_dof_map(Vh[hp.PARAMETER])
d2v_param = d2v_param.astype(np.int64)
v2d_param = v2d_param.astype(np.int64)

nx = settings['nx']
ny = settings['ny']

np.savez(data_dir+'fno_metadata.npz',nx = nx, ny = ny,
			d2v_state = d2v_state, d2v_param = d2v_param,
			v2d_state = v2d_state, v2d_param = v2d_param)

mat = dl.as_backend_type(M).mat()
row, col, val = mat.getValuesCSR()

import scipy.sparse as sp

M_csr = sp.csr_matrix((val,col,row)) 
M_csr = M_csr.astype(np.float32)
sp.save_npz(data_dir+'M_output_csr',M_csr)


input_vector = dl.Function(model.problem.Vh[hp.PARAMETER]).vector()
R_np = operator_to_array_with_dummy_vectors(model.prior.R, input_vector, input_vector)

np.save(data_dir+'R.npy',R_np)

print('Process completed'.center(80))

