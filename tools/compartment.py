import os, warnings
import numpy as np
import nxsdk.api.n2a as nx

def print_Prototype_Information(prototype):
    print('Creating compartment group with the following:\n')
    print('Current decay (I): {} - Current tau (t): {}\n'.format(prototype.compartmentCurrentDecay,prototype.compartmentCurrentTimeConstant))
    print('Voltage decay (V): {} - Voltage tau (t): {}\n'.format(prototype.compartmentVoltageDecay,prototype.compartmentVoltageTimeConstant))
    print('Bias: {} - biasMant: {} - biasExp: {}\n'.format(prototype.bias, prototype.biasMant, prototype.biasExp))
    print('Compartment Threshold: {}\nFunctional State: {}\nRefractory Delay: {}\n'.format(prototype.compartmentThreshold,prototype.functionalState,
                                                                                         prototype.refractoryDelay))
def create_Compartment_Prototype(compartmentVoltageTimeConstant,
                               compartmentCurrentTimeConstant,
                               vTh,refractoryDelay = 1, verbose=True, bias_Mant=0, bias_Exp=0,
                               **CompartmentPrototype_kwargs):
    compartmentVoltageDecay = int(2**12/compartmentVoltageTimeConstant)
    compartmentCurrentDecay = int(2**12/compartmentCurrentTimeConstant)
    prototype = nx.CompartmentPrototype(vThMant=vTh//64,
                                        functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                                        compartmentCurrentDecay=compartmentCurrentDecay,
                                        compartmentVoltageDecay=compartmentVoltageDecay,
                                        biasMant=bias_Mant, biasExp=bias_Exp,
                                        refractoryDelay=refractoryDelay,
                                        **CompartmentPrototype_kwargs)
    if verbose:
        print_Prototype_Information(prototype)
    return prototype

def create_Compartment_Group(net, startCore, endCore, name, prototype, size, verbose=True, type='distributed',
                             neurons_per_core=1024):
    # Max of 128 cores per chip
    # Max of 1024 neurons per core

    group = net.createCompartmentGroup(name, size=0, prototype=prototype)
    num_cores = endCore-startCore
    if type == 'distributed':
        neuronsAssigned = np.ceil(size / num_cores)
        for i in range(size):
            core_id = startCore + (i//neuronsAssigned)
            prototype.logicalCoreId = core_id
            compartment = net.createCompartmentGroup(prototype=prototype)
            group.addCompartments(compartment)
    elif type == 'sequential':
        neuronsAssigned=np.ceil(size / neurons_per_core)
        for i in range(size):
            core_id = startCore + np.floor(i/neurons_per_core)
            prototype.logicalCoreId = core_id
            compartment = net.createCompartmentGroup(prototype=prototype)
            group.addCompartments(compartment)
    elif type == 'sequential_2':
        num_neurons = size
        group = net.createCompartmentGroup(name, size=num_neurons, prototype=prototype)
        for i in range(num_neurons):
            core_id = startCore + (i // num_neurons)
            group[i].logicalCoreId = core_id
    elif type == 'zero':
        num_neurons = size
        group = net.createCompartmentGroup(name, size=0, prototype=prototype)
        for i in range(num_neurons):
            prototype.logicalCoreId = 0
            comp = net.createCompartment(prototype=prototype)
            group.addCompartments(comp)
    else:
        warnings.warn('Type of compartment distribution not specified')
    if verbose:
        print_Prototype_Information(prototype)

    return group

def createCompartmentParameters(size=1, net=None, tauV=1, tauI=None,
                                threshold=100, refractorDelay=1,bias_mant=0,
                                bias_exp=0, start_core=0, end_core=0,
                                name=None, **kwargs):
    # Defaults are specified on the function:
    # tauI - Inf
    # vTh - 100 -> Might not be the correct value
    compPrototype = create_Compartment_Prototype(compartmentCurrentTimeConstant=tauI,
                                                 compartmentVoltageTimeConstant=tauV,
                                                 vTh=threshold, refractoryDelay=refractorDelay,
                                                 bias_Mant=bias_mant, bias_Exp=bias_exp,
                                                 **kwargs)
    group = create_Compartment_Group(net=net, name=name, size=size,
                                           prototype=compPrototype, startCore=start_core,
                                           endCore=end_core, type='distributed')
    return group


# Create a pandas group so I can register which neurons go with each group.




