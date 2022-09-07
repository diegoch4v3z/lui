# coding: utf-8
"""
% LOS ALAMOS NATIONAL LABORATORY CONFIDENTIAL AND PROPRIETARY
%
% Copyright © 2022 Los Alamos National Laboratory.
%
% This software and the related documents are Department of Energy (DOE)
% copyrighted materials, and your use of them is governed by the express
% license under which they were provided to you (License). Unless
% the License provides otherwise, you may not use, modify, copy,
% publish, distribute, disclose or transmit  this software or the
% related documents without the DOE's prior written permission.
"""

import numpy as np
import nxsdk.api.n2a as nx
import S_LCA
import os, warnings

def loihi_hardware(hardware = 'INRC', pyCharm = 'NO'):
    if pyCharm == 'YES':
        os.environ[
            "PATH"] = "/slurm/intel-archi/sbin:/slurm/intel-archi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
        os.environ["SLURM"] = "1"
        haveDisplay = "DISPLAY" in os.environ
    if hardware == 'INRC':
        print('Running on INRC Cloud')
        os.environ['PARTITION'] = 'loihi_2h'
        os.environ['BOARD'] = 'ncl-ext-ghrd-05'
        os.environ['SLURM'] = '1'
    elif hardware == 'loihi':
        print('Running on KapohoBay')
        os.environ['KAPOHOBAY'] = '1'
    else:
        warnings.warn('No hardware specified')
    return print('Using: ', hardware)
