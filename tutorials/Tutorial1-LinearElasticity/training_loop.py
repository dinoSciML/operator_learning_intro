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

import os


def default_settings():
	settings = {}
	settings['architecture'] = 'rbno'
	settings['formulation'] = 'l2'
	settings['output_type'] = 'full_state'
	settings['n_train'] = 800
	settings['width'] = 256
	settings['n_epochs'] = 200
	settings['rM'] = 100
	settings['rQ'] = 100
	settings['channels'] = 32

	return settings


def build_string(settings):
	command = 'python training_driver.py'
	command+=' '
	for key,value in settings.items():
		command += '-'+key
		command += ' '
		command += str(value)
		command += ' '
	return command

n_trains = [25,50,100,200,400,800]

#################################################################################
# RBNO
architecture = 'rbno'
output_types = ['full_state', 'observable']
formulations = ['l2','h1']

for n_train in n_trains:
	for output_type in output_types:
		for formulation in formulations:
			settings = default_settings()
			settings['n_train'] = n_train
			settings['architecture'] = architecture
			settings['output_type'] = output_type
			settings['formulation'] = formulation
			print(build_string(settings))
			os.system(build_string(settings))

#################################################################################
# FNO
# os.system('export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True')
architecture = 'fno'
output_type = 'full_state'
formulations = ['l2','h1']

for n_train in n_trains:
	for formulation in formulations:
		settings = default_settings()
		settings['n_train'] = n_train
		settings['architecture'] = architecture
		settings['output_type'] = output_type
		settings['formulation'] = formulation
		if formulation == 'h1':
			settings['channels'] = int(settings['channels']/2)

		print(build_string(settings))
		os.system(build_string(settings))

#################################################################################
# DeepONet

architecture = 'don'
output_type = 'full_state'
formulations = ['l2','h1']

for n_train in n_trains:
	for formulation in formulations:
		settings = default_settings()
		settings['n_train'] = n_train
		settings['architecture'] = architecture
		settings['output_type'] = output_type
		settings['formulation'] = formulation
		print(build_string(settings))
		os.system(build_string(settings))


