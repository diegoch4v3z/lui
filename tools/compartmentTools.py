import os, warnings
import numpy as np
import nxsdk.api.n2a as nx
import pandas as pd

def printPrototypeInformation(prototype):
    print('Creating compartment group with the following:')
    print('Current decay (I): {} - Current tau (t): {}'.format(prototype.compartmentCurrentDecay,prototype.compartmentCurrentTimeConstant))
    print('Voltage decay (V): {} - Voltage tau (t): {}'.format(prototype.compartmentVoltageDecay,prototype.compartmentVoltageTimeConstant))
    print('Bias: {} - biasMant: {} - biasExp: {}'.format(prototype.bias, prototype.biasMant, prototype.biasExp))
    print('Compartment Threshold: {}\nFunctional State: {}\nRefractory Delay: {}'.format(prototype.compartmentThreshold,prototype.functionalState,
                                                                                         prototype.refractoryDelay))
    print('Comparment Logical Core Id: {}'.format(prototype.logicalCoreId))
    if prototype.enableNoise == 1:
        print('Noise enabled in the neuron\nNoise Mantissa: {}\nNoise Exp: {}'.format(prototype.noiseMantAtCompartment, prototype.noiseExpAtCompartment))


def createCompartmentPrototype(compartmentVoltageTimeConstant,
                                 compartmentCurrentTimeConstant,
                                 vTh,
                                 refractoryDelay = 1,
                                 verbose=True,
                                 bias_Mant=0,
                                 bias_Exp=0,
                                 **kwargs):
    compartmentVoltageDecay = int(2**12/compartmentVoltageTimeConstant)
    compartmentCurrentDecay = int(2**12/compartmentCurrentTimeConstant)
    randomizeVoltage = kwargs.get('randomizeVoltage')
    prototype = nx.CompartmentPrototype(vThMant=vTh//64,
                                        # enableSpikeBackprop=0,
                                        # enableSpikeBackpropFromSelf=0,
                                        # noiseExpAtCompartment=0,
                                        # noiseMantAtCompartment=0,
                                        # randomizeVoltage=0,
                                        # randomizeCurrent=0,
                                        # enableNoise=0,
                                        # numDendriticAccumulators=8,
                                        functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                                        compartmentCurrentDecay=compartmentCurrentDecay,
                                        compartmentVoltageDecay=compartmentVoltageDecay,
                                        biasMant=bias_Mant,
                                        biasExp=bias_Exp,
                                        refractoryDelay=refractoryDelay,
                                        **kwargs
                                        )
    if verbose:
        printPrototypeInformation(prototype)
    return prototype

def _createCompartmentGroup(net, startCore, endCore, name, prototype, size, type='distributed',
                             neuronsPerCore=1024):

    registry = pd.DataFrame(columns=['Neurons', 'coreId'])
    compartmentGroup = net.createCompartmentGroup(name, size=0, prototype=prototype)
    num_cores = endCore-startCore
    if num_cores == 0:
        num_cores = 1
    if type == 'distributed':
        neuronsAssigned = np.ceil(size / num_cores)
        for i in range(size):
            core_id = startCore + (i//neuronsAssigned)
            prototype.logicalCoreId = core_id
            compartment = net.createCompartmentGroup(prototype=prototype, size=1)
            compartmentGroup.addCompartments(compartment)
            registry.loc[len(registry)] = [name, prototype.logicalCoreId]
    elif type == 'sequential':
        neuronsAssigned=np.ceil(size / neuronsPerCore)
        for i in range(size):
            core_id = startCore + np.floor(i/neuronsPerCore)
            prototype.logicalCoreId = core_id
            compartment = net.createCompartmentGroup(prototype=prototype)
            compartmentGroup.addCompartments(compartment)
            registry.loc[len(registry)] = [name, prototype.logicalCoreId]
    elif type == 'sequential_2':
        num_neurons = size
        group = net.createCompartmentGroup(name, size=num_neurons, prototype=prototype)
        for i in range(num_neurons):
            core_id = startCore + (i // num_neurons)
            compartmentGroup[i].logicalCoreId = core_id
            registry.loc[len(registry)] = [name, prototype.logicalCoreId]
    elif type == 'zero':
        num_neurons = size
        group = net.createCompartmentGroup(name, size=0, prototype=prototype)
        for i in range(num_neurons):
            prototype.logicalCoreId = 0
            comp = net.createCompartment(prototype=prototype)
            compartmentGroup.addCompartments(comp)
            registry.loc[len(registry)] = [name, prototype.logicalCoreId]
    else:
        warnings.warn('Type of compartment distribution not specified')
    return compartmentGroup, registry

def createCompartmentParameters(size=1, net=None, tauV=1, tauI=1,
                                vTh=1*64, refractoryDelay=1, biasMant=0,
                                biasExp=0, startCore=0, endCore=0, type='distributed',
                                name=None, **kwargs):
    # Defaults are specified on the function:
    # tauI - Inf
    # vTh - 100 -> Might not be the correct value
    compartmentPrototype = createCompartmentPrototype(compartmentCurrentTimeConstant=tauI,
                                                      compartmentVoltageTimeConstant=tauV,
                                                      vTh=vTh,
                                                      refractoryDelay=refractoryDelay,
                                                      bias_Mant=biasMant,
                                                      bias_Exp=biasExp,
                                                      **kwargs)
    compartmentGroup, registry = _createCompartmentGroup(net=net, name=name, size=size,
                                           prototype=compartmentPrototype, startCore=startCore,
                                           endCore=endCore, type=type)
    return compartmentGroup, registry






