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
from S_LCA.loihi.compartmentTools import createCompartmentParameters
# NXSDK
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator
from S_LCA.environment.environment import loihi_hardware
from nxsdk.arch.n2a.n2board import N2Board
from nxsdk.api.enums.api_enums import ProbeParameter
from nxsdk.graph.monitor.probes import PerformanceProbeCondition
from nxsdk.graph.nxinputgen.nxinputgen import BasicSpikeGenerator
from nxsdk.graph.nxenergy_time import EnergyProbe
from S_LCA.loihi.weight import effectiveWeight, precisionWeight, mantExp, noiseVal


sns.set()

# This script creates a neuron and injects spikes to a compartment
# to check the random threshold.

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
        self.runtime = 20

        self.constructNetwork()

    def run(self):
        t0 = datetime.now()
        print('before run:', t0.strftime("%H:%M:%S"))
        self._net.run(self.runtime)
        self._net.disconnect()
        t1 = datetime.now()
        print('after run:', t1.strftime("%H:%M:%S"))
        print('run time: {:.2f} seconds'.format((t1 - t0).total_seconds()))


    ## Compartment Prototypes
    def createRandomNeuronCompartment(self, size, noiseMant, noiseExp):
        #vth = vThMant*2**6
        valueMax, valueMin = noiseVal(noiseMant,noiseExp)
        print('Limit of random noise values: {} and {}'.format(valueMax, valueMin))
        self.randomNeuronPrototype = nx.CompartmentPrototype(
            logicalCoreId=1,
            vThMant=2,
            enableNoise=1,
            randomizeVoltage=1,
            noiseMantAtCompartment=noiseMant,
            noiseExpAtCompartment=noiseExp,
            compartmentVoltageDecay=4096,
            compartmentCurrentDecay=4096)
    #     self.randomNeuronPrototype = createCompartmentPrototype(compartmentCurrentTimeConstant=1,
    #                                                 compartmentVoltageTimeConstant=1,
    #                                                 bias_Exp=0,
    #                                                 bias_Mant=0,
    #                                                 vTh=1 * 64,
    #                                                 randomizeVoltage=1,
    #                                                 enableNoise=1,
    #                                                 noiseMantAtCompartment=0,
    #                                                 noiseExpAtCompartment=11,
    #                                                 logicalCoreId = 0)
        return [self._net.createCompartmentGroup(size=1, prototype=self.randomNeuronPrototype) for _ in range(size)]

    def createNormalNeuronPrototype(self, size):
        self.normalNeuronProtype = nx.CompartmentPrototype(
            biasMant=1,
            biasExp=0,
            vThMant=4,
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            compartmentVoltageDecay=4096,
            compartmentCurrentDecay=4096)
        return [self._net.createCompartmentGroup(size=1, prototype=self.normalNeuronProtype) for _ in range(size)]




    def constructNetwork(self):
        # Net
        self._net = nx.NxNet()

        # Input
        on_times = np.linspace(1, self.runtime, self.runtime).tolist()
        on_sg = self._net.createSpikeGenProcess(numPorts=1)
        on_sg.addSpikes(spikeTimes=on_times, spikeInputPortNodeIds=0)


        # Connection prototype

        connection_proto = nx.ConnectionPrototype(weight=2)

        # Neurons

        # randomNeuron, registry = createCompartmentParameters(2,
        #                                            net=self._net,
        #                                            tauI=1,
        #                                            tauV=1,
        #                                            vTh=64,
        #                                            refractoryDelay=1,
        #                                            biasMant=0,
        #                                            biasExp=0,
        #                                            startCore=0,
        #                                            endCore=0,
        #                                            name='random')
        randomNeurons = self.createRandomNeuronCompartment(1, noiseMant=0, noiseExp=7)
        normalNeurons = self.createNormalNeuronPrototype(1)


        # Probes
        #self._randomNeuron = randomNeuron
        self._randomNeurons = randomNeurons
        self._normalNeurons = normalNeurons
        if self._probing:
            #(self.randomNeuron_uProbe, self.randomNeuron_vProbe, self.randomNeuron_sProbe) = randomNeuron.probe(self.probeParameters)
            self._randomNeurons_probes = [n.probe(self.probeParameters) for n in self._randomNeurons]
            self._normalNeurons_probes = [n.probe(self.probeParameters) for n in self._normalNeurons]

        # Connections

        on_sg.connect(normalNeurons[0], prototype=connection_proto)
        #normalNeurons[0].connect(randomNeurons[0], prototype=connection_proto)
        on_sg.connect(randomNeurons[0], prototype=connection_proto)


        # on_sg.connect(randomNeuron, prototype=connection_proto,
        #               weight=np.array([[4],
        #                                [4]]),
        #               connectionMask=np.array([[1],
        #                                        [0]]))
        # for ii in range(len(self._randomNeurons)):
        #     on_sg.connect(randomNeurons[ii], prototype=connection_proto)


random = randomNeuron(probing=True)
random.run()




normalNeuron_sProbe = random._normalNeurons_probes[0][2]
normalNeuron_vProbe = random._normalNeurons_probes[0][1]
normalNeuron_uProbe = random._normalNeurons_probes[0][0]

randomNeuron_sProbe = random._randomNeurons_probes[0][2]
randomNeuron_vProbe = random._randomNeurons_probes[0][1]
randomNeuron_uProbe = random._randomNeurons_probes[0][0]


# plt.figure(2, figsize=(18,10))
# ax1=plt.subplot(4,1,1)
# uh = random.randomNeuron_uProbe.plot()
# plt.title('Destination compartmentCurrent')
# ax1.legend(uh, ['dstCx=%d'%(i) for i in range(len(uh))])
#
# plt.tight_layout()
# plt.show()

# Create the one with one neuron connecting two random ones.
# Map out the histogram of the spikes




# Neuron dynamics
fig = plt.figure(6000)

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 11})

plt.subplot(3, 2, 1)
normalNeuron_uProbe.plot()
plt.title('Normal')
plt.xlabel('Time')
plt.ylabel('Current')

plt.subplot(3, 2, 3)
normalNeuron_vProbe.plot()
plt.title('Normal')
plt.xlabel('Time')
plt.ylabel('Voltage')

plt.subplot(3, 2, 5)
normalNeuron_sProbe.plot()
plt.title('Normal')
plt.xlabel('Time')
plt.ylabel('Spikes')

plt.subplot(3, 2, 2)
randomNeuron_uProbe.plot()
plt.title('Random Neuron')
plt.xlabel('Time')
plt.ylabel('Currrent')

plt.subplot(3, 2, 4)
randomNeuron_vProbe.plot()
plt.title('Random Neuron')
plt.xlabel('Time step')
plt.ylabel('Voltage')

plt.subplot(3, 2, 6)
randomNeuron_sProbe.plot()
plt.title('Random Neuron')
plt.xlabel('Time step')
plt.ylabel('Spike')

plt.tight_layout()
plt.show()

#
# # Random Neuron Voltage
# randomNeurons_uProbe_data = [random._randomNeurons_probes[i][0] for i in range(len(random._randomNeurons_probes))]
# randomNeurons_vProbe_data = [random._randomNeurons_probes[i][1] for i in range(len(random._randomNeurons_probes))]
# randomNeurons_sProbe_data = [random._randomNeurons_probes[i][2] for i in range(len(random._randomNeurons_probes))]
#
#
#
#
# fig = plt.figure(4000)
#
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams.update({'font.size': 11})
#
# plt.subplot(3, 2, 1)
# randomNeurons_uProbe_data[0].plot()
# plt.title('Random 1')
# plt.xlabel('Time')
# plt.ylabel('Current')
#
# plt.subplot(3, 2, 3)
# randomNeurons_vProbe_data[0].plot()
# plt.title('Random 1')
# plt.xlabel('Time')
# plt.ylabel('Voltage')
#
# plt.subplot(3, 2, 5)
# randomNeurons_sProbe_data[0].plot()
# plt.title('Random 1')
# plt.xlabel('Time')
# plt.ylabel('Spikes')
#
# plt.subplot(3, 2, 2)
# randomNeurons_uProbe_data[1].plot()
# plt.title('Random 2')
# plt.xlabel('Time')
# plt.ylabel('Currrent')
#
# plt.subplot(3, 2, 4)
# randomNeurons_vProbe_data[1].plot()
# plt.title('Random 2')
# plt.xlabel('Time step')
# plt.ylabel('Voltage')
#
# plt.subplot(3, 2, 6)
# randomNeurons_sProbe_data[1].plot()
# plt.title('Random 2')
# plt.xlabel('Time step')
# plt.ylabel('Spike')
#
# plt.tight_layout()
# plt.show()
#
# # Histograms
#
# low, high = 0, 255
# n_bins = 256
# bins = np.linspace(0, 255, 256).tolist()
#
