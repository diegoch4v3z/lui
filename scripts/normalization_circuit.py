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
# LIBRARIES
import os
from datetime import datetime
from pprint import pprint
import bitstring
import io
import contextlib
import re
import time
import math
import pickle
import bitstring
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nxsdk.api.n2a as nx
import nxsdk
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator

from S_LCA.environment.environment import loihi_hardware
from nxsdk.arch.n2a.n2board import N2Board
from nxsdk.api.enums.api_enums import ProbeParameter
from nxsdk.graph.monitor.probes import PerformanceProbeCondition
from nxsdk.graph.nxinputgen.nxinputgen import BasicSpikeGenerator
from nxsdk.graph.nxenergy_time import EnergyProbe

sns.set()

loihi_hardware(hardware='INRC', pyCharm='YES')

class SpikeAccumulator:
    def __init__(self):
        self.data = []

    def __call__(self, *args, **kwargs):
        self.data.extend(args[0])


class randomNeuron():
    # Constants
    _exp = 2**6
    _max_decay = 2**12
    _max_delay = 6

    _max_comp_per_core = 1024

    _value = 10
    _threshold = _value - 1

    _probes_refs = [nx.ProbeParameter.COMPARTMENT_CURRENT,
                    nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                    nx.ProbeParameter.SPIKE]

    def __init__(self, n_neurons=10, probing = False):
        self._n_neurons = n_neurons
        self._sample_time = n_neurons
        self.probing = probing

        self._time_per_product = self._n_neurons * (self._sample_time)
        self._total_time = self._time_per_product

        self.placement_manager = pd.DataFrame(columns=['proto', 'core_id'])
        self.detailed_placement_manager = pd.DataFrame(columns=['neuron', 'core_id'])

        self._next_core_id = 0
        self._rnd_core_id = 0
        self._stm_core_id = 0
        self._rnd_core_id = 0
        self._n_current_stm_compartments_on_core = 0
        self._n_current_rnd_compartments_on_core = 0

        self._construct_basic_network()

    def random(self, draw = False):
        t0 = datetime.now()
        print('before run:', t0.strftime("%H:%M:%S"))
        self._net.run(self._total_time)
        self._net.disconnect()
        t1 = datetime.now()
        print('after run:', t1.strftime("%H:%M:%S"))
        print('run time: {:.2f} seconds'.format((t1 - t0).total_seconds()))

        # if self.probing:
        #     result_bin = np.array([p[0].data[:, -1] for p in self._probes]).reshape(self.n, self.n_bits)
        #     result_dec = [self.b2d(r) for r in result_bin]
        #     return result_dec
        # else:
        #     return

        return

    def draw(self, coord):
        assert (self.probing)
        # data
        data = np.asarray([self._probes[i][0].data for i in range(len(self._probes))]).reshape(self.n, self.n_bits, self.total_time)

        fig = plt.figure(figsize=(10, 4))
        neurons, times = data[coord].nonzero()
        plt.scatter(times, neurons, marker='>')
        plt.axvline(0.7, c='r', linewidth=0.75)
        plt.axvline(self._sample_time + 0.7, c='r', linewidth=0.75)
        # adjustments
        plt.gca().set_xlim([0, self._total_time + 1])
        plt.gca().set_ylim([-0.5, self._n_neurons - 0.5])
        plt.gca().set_yticks(np.arange(self._n_neurons))
        plt.gca().set_xlabel('time')
        plt.gca().set_ylabel('resum_2 neurons')
        #         plt.gca().legend(names, bbox_to_anchor=(1.05, 1.00), loc='upper left')
        plt.tight_layout()

        return fig

    def _create_rnd_compartments(self, cg_size):
        self._n_current_rnd_compartments_on_core += cg_size
        self._rnd_neuron_proto = nx.CompartmentPrototype(
            biasMant=0,
            biasExp=0,
            vThMant=randomNeuron._threshold,
            functionalState=2,
            compartmentVoltageDecay=randomNeuron._max_decay,
            compartmentCurrentDecay=0,
            enableNoise=1,
            randomizeVoltage=1,
            #noiseMantAtCompartment=1,
            #noiseExpAtCompartment=1,
            logicalCoreId=self._next_core_id)
        self.placement_manager.loc[len(self.placement_manager)] = ['rnd_neuron', self._next_core_id]
        self._rnd_core_id = self._next_core_id
        self._n_current_rnd_compartments_on_core = cg_size
        self._next_core_id += 1
        return [self._net.createCompartmentGroup(size=1, prototype=self._rnd_neuron_proto) for _ in range(cg_size)]

    def _create_stm_compartments(self, cg_size):
        self._n_current_stm_compartments_on_core += cg_size
        self._stm_neuron_proto = nx.CompartmentPrototype(
            vThMant=randomNeuron._threshold,
            compartmentVoltageDecay=randomNeuron._max_decay,
            functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
            logicalCoreId=self._next_core_id)
        self.placement_manager.loc[len(self.placement_manager)] = ['stm', self._next_core_id]
        self._stm_core_id = self._next_core_id
        self._n_current_stm_compartments_on_core = cg_size
        self._next_core_id += 1
        return [self._net.createCompartmentGroup(size=1, prototype=self._stm_neuron_proto) for _ in range(cg_size)]



    def _construct_basic_network(self):
        # Net
        self._net = nx.NxNet()

        # Input
        on_times = [1]
        off_times = [self._time_per_product - 1]
        on_sg = self._net.createSpikeGenProcess(numPorts=1)
        off_sg = self._net.createSpikeGenProcess(numPorts=1)
        on_sg.addSpikes(spikeInputPortNodeIds=0, spikeTimes=on_times)
        off_sg.addSpikes(spikeInputPortNodeIds=0, spikeTimes=off_times)

        # Output
        self._output_sa = SpikeAccumulator()
        self._output_sr = nx.SpikeReceiver(self._net)
        self._output_sr.callback(self._output_sa)

        # Prototype
        pos_conn_proto = nx.ConnectionPrototype(weight=randomNeuron._value)
        neg_conn_proto = nx.ConnectionPrototype(weight=-randomNeuron._value)

        # Neuron
        clock_1_neurons = self._create_stm_compartments(self._sample_time)
        self.detailed_placement_manager.loc[len(self.detailed_placement_manager)] = ['clock_1', self._stm_core_id]
        rnd_neurons = self._create_rnd_compartments(self._sample_time)
        self.detailed_placement_manager.loc[len(self.detailed_placement_manager)] = ['random', self._rnd_core_id]

        # Probes
        self._clock_neurons = clock_1_neurons
        self._rnd_neurons = rnd_neurons
        if self.probing:
            self._clock_probes = [n.probe(self._probes_refs) for n in self._clock_neurons]
            self._rnd_probes = [n.probe(self._probes_refs) for n in self._rnd_neurons]


        # Connections
        on_sg.connect(clock_1_neurons[0], prototype=pos_conn_proto)
        for ii in range(self._sample_time):
            off_sg.connect(clock_1_neurons[ii], prototype=neg_conn_proto)
        for ii in range(self._sample_time - 1):
            next_index = (ii + 1) % self._sample_time
            clock_1_neurons[ii].connect(clock_1_neurons[next_index], prototype = pos_conn_proto)
        clock_1_neurons[-1].connect(clock_1_neurons[0], prototype=pos_conn_proto)
        for ii in range(self._sample_time):
            for iii in range(self._sample_time):
                clock_1_neurons[ii].connect(rnd_neurons[iii], prototype = pos_conn_proto)
        # for ii in range(self._sample_time):
        #     rnd_neurons[ii].connect(rnd_neurons[ii], prototype=neg_conn_proto)



random = randomNeuron(n_neurons=10, probing=True)
random.random()

# Clock Plot
clock_data = np.asarray([random._clock_probes[i][0].data for i in range(len(random._clock_probes))])
neurons_index, other, spike_time = np.nonzero(clock_data)
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots(figsize = (10, 6))
plt.scatter(spike_time, neurons_index, marker='4')
ax.set_xticks(np.arange(len(clock_data[0][0]))[::40])  #Label number linspace
ax.set_xticks(np.arange(len(clock_data[0][0])), minor=True)       # Minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=0.2)   # Gridlines based on minor ticks

plt.axvline()
plt.gca().set_xlabel('Timestep')
plt.gca().set_ylabel('Clock Neuron Index Number')

plt.tight_layout()
plt.show()

# Random Neurons Plot
random_data = np.asarray([random._rnd_probes[i][0].data for i in range(len(random._rnd_probes))])
neurons_index, other, spike_time = np.nonzero(random_data)
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots(figsize = (10, 6))
plt.scatter(spike_time, neurons_index, marker='4')
ax.set_xticks(np.arange(len(random_data[0][0]))[::40])  #Label number linspace
ax.set_xticks(np.arange(len(random_data[0][0])), minor=True)       # Minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=0.2)   # Gridlines based on minor ticks

plt.axvline()
plt.gca().set_xlabel('Timestep')
plt.gca().set_ylabel('Random Neuron Index Number')

plt.tight_layout()
plt.show()




"""
def setupNetwork(net):
    # Create a compartment prototype with the following parameters:
    # biasMant: Configure bias mantissa.  Actual numerical bias is
    #   biasMant*(2^biasExp) = 100*(2^6) = 6400..
    # biasExp: Configure bias exponent.  Actual numerical bias is
    #   biasMant*(2^biasExp) = 100*(2^6) = 6400.
    # vThMant: Configuring voltage threshold value mantissa. Actual numerical
    #   threshold is vThMant*(2^6) = 1000*2^6 = 64000
    # functionalState: The setting to PHASE_IDLE = 2 allows the compartment to
    #   be driven by a bias current alone.  See user guide for all possible
    #   phases.
    # compartmentVoltageDecay: Set to 1/16 (2^12 factor for fixed point
    #   implementation)

    p = nx.CompartmentPrototype(biasMant=100,
                                biasExp=6,
                                vThMant=1000,
                                functionalState=2,
                                compartmentVoltageDecay=256)

    # Create a compartment in the network using the prototype
    compartment = net.createCompartment(p)
    # Create a compartment probe to probe the compartment states: compartment current(U) and compartment voltage(V)
    # probeConditions=None implies default probe conditions will be used for each probe
    probes = compartment.probe([nx.ProbeParameter.COMPARTMENT_CURRENT,
                                nx.ProbeParameter.COMPARTMENT_VOLTAGE], probeConditions=None)

    return probes


# -----------------------------------------------------------------------------
# Run the tutorial
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # Create a network
    net = nx.NxNet()

    # -------------------------------------------------------------------------
    # Configure network
    # -------------------------------------------------------------------------
    probes = setupNetwork(net)

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------

    net.run(100)
    net.disconnect()

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    fig = plt.figure(1001)
    # Since there are no incoming spikes and noise is disabled by default, u
    # remains constant at 0
    plt.subplot(1, 2, 1)
    uProbe = probes[0]
    uProbe.plot()
    plt.title('u')

    # v increases due to the bias current.  The concave nature of the curve
    # when the voltage is increasing is due to the decay constant.  Upon
    # reaching the threshold of 64000, the compartment spikes and resets to 0.
    # Since there is no refractory period, the voltage immediate begins to
    # increase again.
    plt.subplot(1, 2, 2)
    vProbe = probes[1]
    vProbe.plot()
    plt.title('v')

    plt.show()
    
"""


