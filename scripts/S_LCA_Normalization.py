"""
% LOS ALAMOS NATIONAL LABORATORY CONFIDENTIAL AND PROPRIETARY
%
% Copyright 2022 Los Alamos National Laboratory.
%
% This software and the related documents are Department of Energy (DOE)
% copyrighted materials, and your use of them is governed by the express
% license under which they were provided to you (License). Unless
% the License provides otherwise, you may not use, modify, copy,
% publish, distribute, disclose or transmit  this software or the
% related documents without the DOE's prior written permission.

Name: Diego Chavez
Division: CCS - 3

Filename: S_LCA_Normalization.py
Version: 1

"""

import os, sys
import matplotlib.pyplot as plt
import numpy as np

# TODO: Add function to save versions and parameters of the python code.

from S_LCA.S_LCA.network_parameters import params
from S_LCA.S_LCA.S_LCA_Normalization_Network import NormalizationNet


# Control variables
probing = False
training = False
plot = False
dataset = 'MNIST10'
interval = 2
max_rate = 1

# Dataset setup
params['dataset'] = dataset
params['dataset_interval'] = interval
params['dataset_maxrate'] = max_rate

params['probing'] = probing
params['training'] = training
params['plot'] = plot
params['hardware'] = 'INRC' #loihi or INRC Cloud
params['pyCharm'] = 'YES'


# TODO: Create the file for the input of the MNIST
# Input of the probes

# Check if the net is created
s_lca = NormalizationNet(params, debug = 1, verbose = True)

# Check the seed value

