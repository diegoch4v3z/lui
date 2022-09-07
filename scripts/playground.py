"""                                                                                                         
Diego Chavez Arana                                                                                          
"""
# Libraries                                                                                                 
import os
from datetime import datetime
from pprint import pprint

import nxsdk.api.n2a as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from S_LCA.loihi.compartmentTools import createCompartmentParameters
from S_LCA.datasets.datasets import loadMnist
from S_LCA.loihi.spikeGeneration import spikeTimes, repeatInput

from S_LCA.environment.environment import loihi_hardware
from S_LCA.loihi.weight import effectiveWeight, precisionWeight, mantExp, noiseVal
from S_LCA.loihi.weight import weightInitialization, connectionPrototype, createConnection

sns.set()

# Hardware
loihi_hardware(hardware='INRC', pyCharm='YES')

net = nx.NxNet()
numSrc = 10
numDst = 20
# Spike generator
sg = net.createSpikeGenProcess(numPorts=numSrc)
# Prototype
cxProto = nx.CompartmentPrototype(vThMant=2000,
                                  compartmentCurrentDecay=int(1/10*2**12))
cxGrp = net.createCompartmentGroup(size=numDst, prototype=cxProto)

# Connection of the spike generator to the compartement group with a mixed signs
connProto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.MIXED)
np.random.seed(0)

sg.connect(cxGrp,
           prototype=connProto,
           weight=np.random.randint(-100, 100, size =(numDst,numSrc)),
           delay = np.random.randint(0,7, size=(1,numSrc)))

print('Set up completed')

numSteps = 200
spikingProbabilility = 0.05
inputSpikes = np.random.rand(numSrc, numSteps) < spikingProbabilility
spikeTimes1 = []

for i in range(numSrc):
    st = np.where(inputSpikes[i,:])[0].tolist()
    spikeTimes1.append(st)





