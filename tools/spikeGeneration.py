# TODO: add the information necessary

import numpy as np
from numpy import newaxis
import warnings
from nxsdk.net.process.basicspikegen import BasicSpikeGen
from S_LCA.datasets.datasets import loadMnist

def spikeTimes(input, binary = False, flatten = False):
    num_comps = np.shape(input)[0]
    comp_size = np.shape(input)[1] * np.shape(input)[2]
    if flatten:
        num_comps = np.shape(input)[0]
        comp_size = np.shape(input)[1] * np.shape(input)[2]
        temp = np.zeros((num_comps, comp_size))
        for i, j in enumerate(input):
            length = np.shape(j.flatten())
            temp[i] = j.flatten()
        input = temp

    if binary:
        input = (input > 0).astype(np.int_)

    IDX = []
    T = []
    for i, j in enumerate(input):
        idx = []
        t = []
        size = np.shape(j)[0]
        if flatten:
            idx = np.where(j)[0]
            t = i*np.ones((1,len(idx)))
        else:
            for ii in range(size):
                print('enter loop')
                if len(np.where(j[ii])[0]) == 0:
                    print('row doesnt have values to store')
                else:
                    print('row has value to store')
                    idx = np.append(idx, np.where(j[ii])[0])
                    t = np.append(t, np.ones(len(np.where(j[ii])[0])) * ii)

        IDX += [idx]
        T += [t]
    return IDX, T, input, num_comps, comp_size

def repeatInput(input, times, binarize = True, type = 1):
    size = np.shape(input)
    if type == 1: # Batch repetition
       output = np.repeat(input, times, axis = 0)
    elif type == 0: # Element repetition
        # it is going to repeat that element t times
        temp = np.zeros((size[0] * times, size[1], size[2]))
        for i, j in enumerate(input):
            for ii in range(times):
                temp[ii + i*times] = j
            #temp[i:i+times] = np.repeat(j, times, axis = 0)
        output = temp
    if binarize:
        output = (output > 0).astype(np.int_)
    return output

def spaceInput(input, spacing = 1,verbose = True):
    if verbose:
        print('Generating spaced array:')
        indices = []
    for i in range(len(input)):
        dataInsert = np.empty(np.shape(input[i]))[newaxis,:,:]
        for ii in range(spacing):
            dataInsert = np.append(dataInsert, np.zeros(np.shape(input[i]))[newaxis,:,:], axis = 0)
        dataInsert = np.delete(dataInsert,0,0)
        input = np.insert(input, (spacing*i+i), dataInsert, axis = 0)
    return input




if __name__ == '__main__':
    m = 28
    mnist_train_data, mnist_test_data, mnist_train_labels, mnist_test_labels, trainOutput = loadMnist(m=m, resize=False, shuffle=False, save=False, numImages=2)

    #spikeTimes(mnist_train_data, interval=3, max_rate=1, num_neurons=1, T=1, binary=True, flatten= True)
    mnist_train_data = repeatInput(mnist_train_data, 2, True, type=0)
    IDX, T = spikeTimes(mnist_train_data, binary = True, flatten=True)



    # the function should have number of neurons where the spike generation is being made
    # indices where the generation is going ot happen.