import matplotlib.pyplot as plt
import numpy as np
import os
from copy import copy
import nxsdk.api.n2a as nx
from datetime import datetime
from S_LCA.environment.environment import loihi_hardware

loihi_hardware(hardware='INRC', pyCharm='YES')

class weight_Precision():


    def __init__(self):
        self.setupNetwork()

    def run(self, runTime = 10):
        t0 = datetime.now()
        print('Time before run:', t0.strftime("%H:%M:%S"))
        self._net.run(runTime)
        self._net.disconnect()
        t1 = datetime.now()
        print('Time after run : ', t1.strftime("%H:%M:%S"))
        print('Run time {:.2f} seconds'.format((t1-t0).total_seconds()))
    def setupNetwork(self):
        self._net = nx.NxNet()

        # Compartment Prototype
        compartment_Prototype1 = nx.CompartmentPrototype(biasMant=1,
                                                        biasExp=6,
                                                        vThMant=10,
                                                        functionalState=2,
                                                        compartmentCurrentDecay=1)
        compartment_Prototype2 = nx.CompartmentPrototype(compartmentCurrentDecay=1)

        # Compartments
        compartment1 = self._net.createCompartment(compartment_Prototype1)
        compartment2 = self._net.createCompartment(compartment_Prototype2)
        compartment3 = self._net.createCompartment(compartment_Prototype2)
        compartment4 = self._net.createCompartment(compartment_Prototype2)
        compartment5 = self._net.createCompartment(compartment_Prototype2)

        # Connection Prototype
        connectionPrototype1 = nx.ConnectionPrototype(weight=20, numWeightBits=4, compressionMode=0,
                                                      signMode=2)
        self._net._createConnection(compartment1, compartment2, connectionPrototype1)

        #####
        connectionPrototype2 = copy(connectionPrototype1)


        connectionPrototype2.weight = 16
        connectionPrototype2.numWeightBits = -1

        self._net._createConnection(compartment1, compartment3, connectionPrototype2)

        connectionPrototype3 = copy(connectionPrototype1)
        connectionPrototype3.weight = 10
        connectionPrototype3.numWeightBits = 6
        connectionPrototype3.signMode = 1
        self._net._createConnection(compartment1, compartment4, connectionPrototype3)

        connectionPrototype4 = copy(connectionPrototype1)
        connectionPrototype4.weight = 8
        connectionPrototype4.numWeightBits = -1
        connectionPrototype4.signMode = 1
        self._net._createConnection(compartment1, compartment5, connectionPrototype4)

        self.probes = []

        self.compartment1Probe = compartment1.probe([nx.ProbeParameter.COMPARTMENT_CURRENT,
                                                nx.ProbeParameter.COMPARTMENT_VOLTAGE],
                                               probeConditions = None)
        self.probes.append(self.compartment1Probe)


        for compartment in [compartment2, compartment3, compartment4, compartment5]:
            self.compartmentProbe = compartment.probe([nx.ProbeParameter.COMPARTMENT_CURRENT],
                                                 probeConditions = None)
        self.probes.append(self.compartmentProbe)






# Initialize the script
test = weight_Precision()
test.run(10)









