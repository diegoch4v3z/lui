import numpy as np
import scipy.sparse as sparse
import time
import nxsdk.api.n2a as nx
import warnings
from SFC_backprop.SFC_backprop import weight_init

def effectiveWeight(numWeightBits=6, IS_MIXED=0, weight=255, exp=0):
    numLsbBits = 8 - numWeightBits - IS_MIXED
    actWeight = (weight >> numLsbBits) << numLsbBits
    print('Original weight: ', weight)
    print('Actual weight: ', actWeight)
    print('Num LSB Bits: ', numLsbBits)
    print('weight (with exponent):', actWeight * 2 ** exp)
    print('weight effect on current (with exponent):', actWeight * 2 ** (6 + exp))
    return actWeight
def noiseVal(noiseMantAtCompartment=0, noiseExpAtCompartment=0):
    valueMax = (0-2**7+noiseMantAtCompartment*2**6)*2**(noiseExpAtCompartment-7)
    valueMin = (255-2**7+noiseMantAtCompartment*2**6)*2**(noiseExpAtCompartment-7)
    return valueMax, valueMin
def weightInitialization(params, type='random', shift=0.5, scale = 1, positive = False, file=None):
    num_populations = params['num_populations']
    if type == 'randomUniform':
    #randn returns the value from a "standard normal distribution"
    #rand returns the value from

        wMatrix0 = (np.random.randn(num_populations['in'], num_populations['hid']) + shift)*scale
        wMatrix1 = (np.random.randn(num_populations['hid'], num_populations['out']) + shift)*scale

    elif type == 'random':
        wMatrix0 = (np.random.rand(num_populations['in'], num_populations['hid']) + shift)*scale
        wMatrix1 = (np.random.rand(num_populations['hid'], num_populations['out'])+ shift)*scale
    else:
        print('No type')
    if type == 'testLearning':
    #randn returns the value from a "standard normal distribution"
    #rand returns the value from
        wMatrix0 = np.zeros((num_populations['hid'], num_populations['out']))
        wMatrix1 = (np.ones((num_populations['in'], num_populations['hid'])) + shift)*scale


    if positive:
        wMatrix0 = np.abs(wMatrix0)
        wMatrix1 = np.abs(wMatrix1)
    return wMatrix0, wMatrix1


def precisionWeight(value, IS_MIXED=0):
    valueDes=value
    valueAct=value
    numWeightBits=8
    while numWeightBits > 0 and valueAct == valueDes:
        numWeightBits -= 1
        numLsbBits = 8 - numWeightBits - IS_MIXED
        valueAct = (value >> numLsbBits) << numLsbBits
    if (numWeightBits == 6):
        numWeightBits =+ 1
    return numWeightBits + 1
def mantExp(value, precision=8, name="", exponentBounds=(-8,7), weightPrecision=7, verbose=True):
    value = np.asarray(value)
    exponentBounds = np.asarray(exponentBounds)
    if verbose:
        print('Desired value: ', value )
    valDes = value
    exp = 0
    while np.max(np.abs(value)) >= (2**precision) and not exp == exponentBounds[1]:
        value = value/2
        exp += 1
    exponentBounds[0] = np.max([exponentBounds[0], weightPrecision-8])
    while np.abs(np.max(value)) < (2**precision/2) and np.abs(np.max(value)) != 0 and not exp == exponentBounds[0]:
        value = value*2
        exp += -1

    value = np.asarray(np.round(value), dtype=int)
    if verbose:
        print('Actual value of ', name, ':', value*2**exp, 'mantissa:', value, 'exponent:', exp)
    if (valDes != (value*2**exp)).any():
        if verbose:
            print('Rounding error for ' + name + '!')
        warnings.warn('Rounding error for ' + name + '!')
    return value, exp
def connectionPrototype(weightMatrix, weightExp=None, verbose=True, **kwargs):
    scalar = False
    ## Check if a function is iterable
    if not hasattr(weightMatrix, '__iter__'):
        weightMatrix = np.asarray([weightExp])
        scalar = True
    elif type(weightMatrix) == list:
        weightMatrix = np.asarray(weightMatrix)
    try:
        weightMatrix = weightMatrix.tocsr()
        maxAbs = abs(weightMatrix).max()
    except:
        maxAbs = np.max(np.abs(weightMatrix))

    maxVal = np.max(weightMatrix)
    minVal = np.min(weightMatrix)

    if maxVal > 0 and minVal < 0:
        signMode = nx.SYNAPSE_SIGN_MODE.MIXED
        weightPrecision = 7
    elif minVal > 0:
        signMode = nx.SYNAPSE_SIGN_MODE.EXCITATORY
        weightPrecision = 8
    elif maxVal < 0:
        signMode = nx.SYNAPSE_SIGN_MODE.INHIBITORY
        weightPrecision = 8
    else:
        signMode = nx.SYNAPSE_SIGN_MODE.MIXED
        weightPrecision = 7

    if weightExp is None:
        _ , weightExp = mantExp(maxAbs, precision=8, name='weight', weightPrecision= weightPrecision,
                                verbose=verbose)
    weightMatrix = weightMatrix*2**(-weightExp)
    weightMatrix = np.asarray(np.round(weightMatrix), dtype=int)

    if verbose:
        print('Max weight: ', np.max(weightMatrix))
        print('Min weight: ', np.min(weightMatrix))

    try:
        numDelayBits = kwargs['numDelayBits']
    except KeyError:
        kwargs['numDelayBits'] = 2

    try:
        numWeightBits = kwargs['numWeightBits']
    except KeyError:
        kwargs['numWeightBits'] = 8

    try:
        numTagBits = kwargs['numTagBits']
    except KeyError:
        kwargs['numTagBits'] = 1
    prototype = nx.ConnectionPrototype(signMode=signMode,
                                       weightExponent=weightExp,
                                       **kwargs)
    if scalar:
        weightMatrix = weightMatrix[0]

    return prototype, weightMatrix
def createConnectionMask(weightMatrix, threshold, type = 0):
    if type == 0:
        mask = np.eye(np.shape(weightMatrix)[0], np.shape(weightMatrix)[1])
    else:
        mask = np.abs(weightMatrix) > np.max(np.abs(weightMatrix))*threshold
    return mask
def createConnection(net, params, sourceGroup, targetGroup, weightMatrix, name = '', type = 0, learning = False):
    startTime = time.time()
    if learning:
        print('Learning enabled...')
        paramsL = params['learning']
        lr = net.createLearningRule(dw='y0*x0*r1*4 - y0*x0*2',
                                    #'2^-2*x1*y0 - 2^-2*y1*x0'
                                    x1Impulse=paramsL['traceImp'],
                                    x1TimeConstant=paramsL['traceTau'],
                                    y1Impulse=paramsL['traceImp'],
                                    y1TimeConstant=paramsL['traceTau'],
                                    r1Impulse=paramsL['rImp'],
                                    r1TimeConstant=paramsL['rTimeConstant'],
                                    tEpoch=paramsL['tEpoch'])
        connProto, weightMatrix = connectionPrototype(weightMatrix,
                                                      enableLearning=1,
                                                      learningRule=lr)
    else:
        connProto, weightMatrix = connectionPrototype(weightMatrix)
        # One to one connection
    mask = createConnectionMask(weightMatrix=weightMatrix, threshold=0.1, type = type)
    print('Connecting: ', np.sum(mask), name, ' synapses...')
    connGroup = sourceGroup.connect(targetGroup,
                                    prototype=connProto,
                                    weight=weightMatrix,
                                    connectionMask=mask)
    print('Took: ', time.time() - startTime)
    return connGroup, connGroup
if __name__ == '__main__':
    #weight = effectiveWeight(numWeightBits=6, IS_MIXED=0, weight=255, exp=0)
    #precisionWeight = precisionWeight(255, IS_MIXED=0)
    #value, exp = mantExp(2000)
    #print('hello')
    params = {'num_populations': {'in': 10,
                                  'hid': 10,
                                  'out': 10}}
    num_populations = params['num_populations']
    effectiveWeight(6, 0, 2,0)
    #w0, w1 = weightInitialization(params, type='randomUniform')