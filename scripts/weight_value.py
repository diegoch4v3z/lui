"""
% LOS ALAMOS NATIONAL LABORATORY CONFIDENTIAL AND PROPRIETARY
%
% Copyright ? 2022 Los Alamos National Laboratory.
%
% This software and the related documents are Department of Energy (DOE)
% copyrighted materials, and your use of them is governed by the express
% license under which they were provided to you (License). Unless
% the License provides otherwise, you may not use, modify, copy,
% publish, distribute, disclose or transmit  this software or the
% related documents without the DOE's prior written permission.
"""

## Libraries
import os, io, contextlib, re, time, math, pickle, bitstring, nxsdk
from datetime import datetime
from pprint import pprint

import nxsdk.api.n2a as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# NXSDK
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator
from S_LCA.environment.environment import loihi_hardware
from nxsdk.arch.n2a.n2board import N2Board
from nxsdk.api.enums.api_enums import ProbeParameter
from nxsdk.graph.monitor.probes import PerformanceProbeCondition
from nxsdk.graph.nxinputgen.nxinputgen import BasicSpikeGenerator
from nxsdk.graph.nxenergy_time import EnergyProbe

sns.set()

# Hardware
loihi_hardware(hardware='INRC', pyCharm='YES')

class SpikeAccumulator:
    def __init__(self):
        self.data = []

    def __call__(self, *args, **kwargs):
        self.data.extend(args[0])

class randomNeuron():

    # Constants

    # Probes
    probeParameters = [nx.ProbeParameter.COMPARTMENT_CURRENT,
                    nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                    nx.ProbeParameter.SPIKE]


    def __init__(self, probing=False):

        self.placement_manager = pd.DataFrame(columns=['proto', 'core_id'])
        self.detailed_placement_manager = pd.DataFrame(columns=['neuron', 'core_id'])

        self._probing = probing

        self.constructNetwork()

    def run(self, runtime = 10):
        t0 = datetime.now()
        print('before run:', t0.strftime("%H:%M:%S"))
        self._net.run(runtime)
        self._net.disconnect()
        t1 = datetime.now()
        print('after run:', t1.strftime("%H:%M:%S"))
        print('run time: {:.2f} seconds'.format((t1 - t0).total_seconds()))


    ## Compartment Prototypes
    def createRandomNeuronCompartment(self, size):
        # vth = vThMant*2**6
        self.randomNeuronPrototype = nx.CompartmentPrototype(
            biasMant=0,
            biasExp=0,
            vThMant=254,
            compartmentVoltageDecay=4096,
            compartmentCurrentDecay=4096,
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            logicalCoreId=0,
            enableNoise=1,
            randomizeVoltage=1,
            noiseMantAtCompartment=0,
            noiseExpAtCompartment=8)
        return [self._net.createCompartmentGroup(size=1, prototype=self.randomNeuronPrototype) for _ in range(size)]

    def createNormalNeuronPrototype(self, size):
        self.normalNeuronProtype = nx.CompartmentPrototype(
            biasMant=0,
            biasExp=0,
            vThMant=130,
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            compartmentVoltageDecay=4096,
            compartmentCurrentDecay=4096)
        return [self._net.createCompartmentGroup(size=1, prototype=self.normalNeuronProtype) for _ in range(size)]

    def constructNetwork(self):
        # Net
        self._net = nx.NxNet()

        # Input
        on_times = np.linspace(1, 10, 10).tolist()
        on_sg = self._net.createSpikeGenProcess(numPorts=1)
        on_sg.addSpikes(spikeTimes=on_times, spikeInputPortNodeIds=0)

        # Connection prototype
        connection_proto = nx.ConnectionPrototype(weight=2, weightExponent=1)
        # Neurons
        normalNeurons = self.createNormalNeuronPrototype(1)

        # Probes
        self._normalNeurons = normalNeurons
        if self._probing:
            self._normalNeurons_probes = [n.probe(self.probeParameters) for n in self._normalNeurons]

        # Connections

        on_sg.connect(normalNeurons[0], prototype=connection_proto)

random = randomNeuron(probing=True)
random.run(50)


normalNeuron_sProbe = random._normalNeurons_probes[0][2]
normalNeuron_vProbe = random._normalNeurons_probes[0][1]
normalNeuron_uProbe = random._normalNeurons_probes[0][0]

print(random._normalNeurons_probes[0][1].data)




# Neuron dynamics
fig = plt.figure(4000)

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 11})

plt.subplot(1, 3, 1)
normalNeuron_uProbe.plot()
plt.title('Normal')
plt.xlabel('Time')
plt.ylabel('Current')

plt.subplot(1, 3, 2)
normalNeuron_vProbe.plot()
plt.title('Normal')
plt.xlabel('Time')
plt.ylabel('Voltage')

plt.subplot(1, 3, 3)
normalNeuron_sProbe.plot()
plt.title('Normal')
plt.xlabel('Time')
plt.ylabel('Spikes')

plt.tight_layout()
plt.show()

