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
from S_LCA.loihi.spikeGeneration import spikeTimes, repeatInput, spaceInput
from S_LCA.environment.environment import loihi_hardware
from S_LCA.loihi.weight import effectiveWeight, precisionWeight, mantExp, noiseVal, weightInitialization, connectionPrototype, createConnection
sns.set()


# Hardware
loihi_hardware(hardware='INRC', pyCharm='YES')

class SpikeAccumulator:
    def __init__(self):
        self.data = []

    def __call__(self, *args, **kwargs):
        self.data.extend(args[0])

class probabilityDistribution():
    #Constants
    probeParameters = [nx.ProbeParameter.COMPARTMENT_CURRENT,
                       nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                       nx.ProbeParameter.SPIKE]
    testVariable = np.random.randint(1,10)
    _runtime = 10
    def __init__(self, probing=False):
        self.placementManager = pd.DataFrame(columns=['proto', 'coreId'])
        self.detailedPlacementManager = pd.DataFrame(columns=['neuron', 'coreId'])
        self._probing = probing
        self.constructNetwork()

    ## Compartment Prototypes
    def createRandomNeuronCompartment(self, size, noiseMant, noiseExp):
        # vth = vThMant*2**6
        valueMax, valueMin = noiseVal(noiseMant, noiseExp)
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
        return [self._net.createCompartmentGroup(size=1, prototype=self.randomNeuronPrototype) for _ in range(size)]
    def createNormalNeuronPrototype(self, size, arrayGroup = True, learning = False):
        if learning:
            self.normalNeuronProtype = nx.CompartmentPrototype(
               biasMant=1,
               biasExp=0,
               vThMant=2,
               functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
               compartmentVoltageDecay=4096,
               compartmentCurrentDecay=4096,
               enableSpikeBackprop=1,
               enableSpikeBackpropFromSelf=1) 
        else:
            self.normalNeuronProtype = nx.CompartmentPrototype(
                biasMant=1,
                biasExp=0,
                vThMant=2,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                compartmentVoltageDecay=4096,
                compartmentCurrentDecay=4096)

        if arrayGroup:
            return [self._net.createCompartmentGroup(size=1, prototype=self.normalNeuronProtype) for _ in range(size)]
        else:
            return self._net.createCompartmentGroup(size = size, prototype= self.normalNeuronProtype)



    def constructNetwork(self):
        self._net = nx.NxNet()

        # MNIST Input
        # Spike input just accepts a list where that neuron is going to spike.
        # Generate the times where that neuron is going to spike that represents the input of the mnist dictionary across time.
        m = 10
        mnist_train_data, mnist_test_data, mnist_train_labels, mnist_test_labels, trainOutput = loadMnist(m=m, resize=True, shuffle=False, save=False, numImages=1)
        mnist_train_data = repeatInput(mnist_train_data, 10, binarize=True)
        mnist_train_data = spaceInput(mnist_train_data, 1)
        IDX, T, input, num_comps, comp_size = spikeTimes(mnist_train_data, binary=True, flatten=True)
        input = input.transpose()
        # Create spike generator
        sg = self._net.createSpikeGenProcess(numPorts=comp_size)
        onTimes = []
        for i,j in enumerate(T):
            onTimes = np.append(onTimes, np.unique(j[0]))
        for i in onTimes:
            nodeId = np.int_(i).tolist()
            spike =  np.int_(IDX[1]).tolist()
            sg.addSpikes(spikeInputPortNodeIds = nodeId,spikeTimes=spike)
        # Connection prototype
        #connection_proto = nx.ConnectionPrototype(weight=2)
        normalNeurons = self.createNormalNeuronPrototype(comp_size, arrayGroup = False)
        normalNeuronsPost = self.createNormalNeuronPrototype(comp_size, arrayGroup=False, learning=True)

        params = {'num_populations': {'in': m*m,
                                      'hid': m*m,
                                      'out': m*m},
                  'learning': {'traceImp': 1,
                               'traceTau': 1,
                               'runTime': self._runtime,
                               'rImp': 1,
                               'rTimeConstant': 1,
                               'tEpoch': 4}}

        # lr = net.createLearningRule(dw=conn_parameters['lr_w'],
        #                             dt=lr_t,
        #                             x1Impulse=conn_parameters['x1Impulse'],
        #                             x1TimeConstant=conn_parameters['x1TimeConstant'],
        #                             y1Impulse=conn_parameters['y1Impulse'],
        #                             y1TimeConstant=conn_parameters['y1TimeConstant'],
        #                             r1Impulse=conn_parameters['r1Impulse'],
        #                             r1TimeConstant=conn_parameters['r1TimeConstant'],
        #                             tEpoch=1)

        # # g_sfc_params['voltage_init'] = 'rand'
        # g_sfc_params['S_factor'] = 0.99  # 0.96  # 0.96  # 0.96
        # g_sfc_params['x1TimeConstant'] = 1  # 4  # 3
        # g_sfc_params['y1TimeConstant'] = 1  # 4  # 3
        # g_sfc_params['x1Impulse'] = 1  # 0.005  # 0.015
        # g_sfc_params['y1Impulse'] = 1  # 0.015
        #
        # g_sfc_params['r1TimeConstant'] = 0  # 2
        # g_sfc_params['r1Impulse'] = 1

        # Connection
        wMatrix0, wMatrix1 = weightInitialization(params, type='testLearning', shift = 0, positive=True, scale=2)
        # Connection prototype
        #normalNeuronsPrototype, wMatrix0 = connectionPrototype(wMatrix0)

        connectionGroup, conn  = createConnection(self._net, params, sg, normalNeurons, wMatrix1, name = '1', type = 0, learning = True)
        connectionGroup2, conn2 = createConnection(self._net,params, normalNeurons, normalNeuronsPost,  wMatrix1, name='2', type = 0, learning=False)
        connectionGroup3, conn3 = createConnection(self._net,params, sg, normalNeuronsPost,  wMatrix1, name='3', type = 0, learning=False)
        # Probes
        self._normalNeurons = normalNeurons
        self._conn = conn
        self._conn2 = conn2
        if self._probing:
           self._normalNeurons_probes = [n.probe(self.probeParameters) for n in self._normalNeurons]
           self.ProbeResult1 = normalNeurons.probe([nx.ProbeParameter.COMPARTMENT_CURRENT,
                                                   nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                                                    nx.ProbeParameter.SPIKE])
           self.ProbeResult2 = normalNeuronsPost.probe([nx.ProbeParameter.COMPARTMENT_CURRENT,
                                                  nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                                                  nx.ProbeParameter.SPIKE])
           self.connProbeResult = [n.probe(nx.ProbeParameter.SYNAPSE_WEIGHT)[0] for n in self._conn]
           self.conn2ProbeResult = [n.probe(nx.ProbeParameter.SYNAPSE_WEIGHT)[0] for n in self._conn2]
    def run(self, runtime):
        self._runtime = runtime
        t0 = datetime.now()
        print('Before run: ', t0.strftime("%H:%M:%S"))
        self._net.run(runtime)
        self._net.disconnect()
        t1 = datetime.now()
        print('after run:', t1.strftime("%H:%M:%S"))
        print('run time: {:.2f} seconds'.format((t1 - t0).total_seconds()))




        #for i in range(comp_size):
        #    sg.connect(normalNeurons[i],)
probability = probabilityDistribution(probing=True)

# Parameters

probability.run(runtime = 120)
print('Simulation Sucessful')
probes = probability._normalNeurons_probes
probesAlt1 = probability.ProbeResult1
probesAlt2 = probability.ProbeResult2
probesAltconn1 = probability.connProbeResult
probesAltconn2 = probability.conn2ProbeResult


sData = np.zeros((len(probes), len(probes[0][2].data)))
vData = np.zeros((len(probes), len(probes[0][1].data)))
uData = np.zeros((len(probes), len(probes[0][0].data)))

for i, j in enumerate(probes):
    # Spiking [0][2]
    sData[i] = np.asarray(j[2].data)
    # Voltage [0][1]
    vData[i] = np.asarray(j[1].data)
    # Current [0][0]
    uData[i] = np.asarray(j[0].data)

# TODO function to plot synapses, compartment, voltage and curretn depending on its nature
plt.figure(1, figsize=(18,14))

# Plot the destination compartment current
ax1 = plt.subplot(3,1,1)
u = probesAlt1[0].plot()
plt.title('Destination compartment 1 current')
# Plot destination compartment voltage
ax2 = plt.subplot(3,1,2)
v = probesAlt1[1].plot()
plt.title('Destination compartment 1 voltage')
# Plot destination spikes
ax3 = plt.subplot(3,1,3)
s = probesAlt1[2].plot(colors=[h.get_color() for h in u])
#ax3.set_xlim(ax1.get_xlim())
plt.title('Destination Spikes 1')
# Plot Source Spikes
plt.show()

plt.figure(1, figsize=(18,14))
ax3 = plt.subplot(1,1,1)
s = probesAlt1[2].plot(colors=[h.get_color() for h in u])
#ax3.set_xlim(ax1.get_xlim())
plt.title('Destination Spikes 1')
# Plot Source Spikes
plt.show()

# Plot the destination compartment current
ax1 = plt.subplot(4,1,1)
u = probesAlt2[0].plot()
plt.title('Destination compartment 2 current')
# Plot destination compartment voltage
ax2 = plt.subplot(4,1,2)
v = probesAlt2[1].plot()
plt.title('Destination compartment 2 voltage')
# Plot destination spikes
ax3 = plt.subplot(4,1,3)
s = probesAlt2[2].plot(colors=[h.get_color() for h in u])
ax3.set_xlim(ax1.get_xlim())
plt.title('Destination Spikes 2')
# Plot Source Spikes
plt.show()


# Extract data from the synaptic weights
def extractData(probes):
    data = np.zeros((np.shape(probes)[0],np.shape(probes[0].data)[0]) )
    for i in range(len(probes)):
        temp = probesAltconn1[i].data
        data[i,:] = temp
    return data


data = extractData(probesAltconn1)
fig, ax = plt.subplots(nrows=len(data[1]), ncols=1, figsize=(15, 15))
fig.supxlabel('Weights of connection 1')
t = np.linspace(1,len(data[1]),len(data[1]))
for i, j in enumerate(data):
    ax[i].plot(t,j)
plt.tight_layout()
plt.show()

data = extractData(probesAltconn2)
fig, ax = plt.subplots(nrows=len(data[1]), ncols=1, figsize=(15, 15))
fig.supxlabel('Weights of connection 2')
t = np.linspace(1,len(data[1]),len(data[1]))
for i, j in enumerate(data):
    ax[i].plot(t,j)
plt.tight_layout()
plt.show()

print('Plot Completed')



