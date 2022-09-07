"""
LOS ALAMOS NATIONAL LABORATORY CONFIDENTIAL AND PROPRIETARY

Copyright ? 2022 Los Alamos National Laboratory.

This software and the related documents are Department of Energy (DOE)
copyrighted materials, and your use of them is governed by the express
license under which they were provided to you (License). Unless
the License provides otherwise, you may not use, modify, copy,
publish, distribute, disclose or transmit  this software or t
related documents without the DOE's prior written permission.

This software and the related documents are provided as is, with
no express or implied warranties, other than those that are
expressly stated in the License.

-------------------------------------------------------------------------
Name: network_parameters.py
Version: 1
Author: Diego Chavez
Division: CCS - 3
-------------------------------------------------------------------------
General Information
-------------------------------------------------------------------------

This file contains the parameters to implement the 2 layer.
Where it specifies the parameters for each of the layers that we are using
"""

# Libraries
import numpy as np
from S_LCA.loihi.compartment import create_Compartment_Prototype


params = {}
# Figure out the parameters of each of the layers.
params['num_trials'] = 10
params['num_populations'] = {}
params['num_populations']['hid'] = 784
params['num_populations']['gat'] = 1

# Threshold parameters
params['weight_exponent'] = 0
params['lca_threshold'] = 102
# Learning Parameters
params['x1TimeConstant'] = 1
params['y1TimeConstant'] = 1
params['x1Impulse'] = 0
params['y1Impulse'] = 0
params['r1Impulse'] = 1

bias = 0

slca_neuron_params = {
    'tau_v': 1,
    'tau_i': 1,
    'threshold': params['lca_threshold'],
    'refracotry': 0,
    'i_cost': bias,
    'enableNoise': 0,
    'enableLearning': 0,
    'num_neurons': params['num_populations']['hid'],
    'neuron_creator': create_Compartment_Prototype,
    'start_core': 2,
    'end_core': 3 # 1024 neurons per core
}


# Layer parameters
# Layer learning
# Gating
# Reward
# Input

