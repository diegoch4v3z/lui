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

Filename: S_LCA_Normalization_Network.py
Version: 1

"""

import os, pickle, sys, time, re
import matplotlib.pyplot as plt
import numpy as np
import nxsdk.api.n2a as nx
from S_LCA.environment.environment import loihi_hardware
from S_LCA.datasets.datasets import loadMnist
from S_LCA.loihi.spikeGeneration import spikeTimes


class NormalizationNet():
    def __init__(self, params, debug = 0, verbose = False):
        if debug > 0:
            self.verbose = True
        else:
            self.verbose = False
        self.params = params
        self.hardware = params['hardware']
        self.IDE = params['pyCharm']
        self.dataset = params['dataset']
        self.dataset_interval = params['dataset_interval']
        self.max_rate = params['dataset_maxrate']


        # Setup environment
        loihi_hardware(hardware=self.hardware, pyCharm=self.IDE)
        # Load input
        mnist_train_data, mnist_test_data, mnist_train_labels, mnist_test_labels, trainLabel = loadMnist(m=28, resize=False, shuffle=False, save=False, numImages=10, verbose = False)

        spikeTimes(mnist_train_data, self.dataset_interval, self.max_rate, num_neurons=1)
        # Input
        # To give input to the loihi we create a spike generation process
        # This spike generation process is going to input the image based on the indices of the input
        # Image





