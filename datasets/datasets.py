import os
import keras.datasets.mnist
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
import skimage.transform as skt
import tensorflow as tf
import tensorflow_datasets as tfds


def loadMnist(m=28, resize=False, crop=0, shuffle=False, verbose=True, save=True, numImages = 1000, tag = 'default'):
    if numImages > 60000:
        (mnist_train_data2, mnist_train_labels2), (mnist_test_data2, mnist_test_labels2) = keras.datasets.mnist.load_data()
        (mnist_train_data, mnist_train_labels), (mnist_test_data, mnist_test_labels) = keras.datasets.mnist.load_data()
        for i in range((numImages-60000)//60000):
            mnist_train_data = np.concatenate((mnist_train_data, mnist_train_data2), axis=0)
            mnist_train_labels = np.concatenate((mnist_train_labels, mnist_train_labels2))
        mnist_train_data = np.concatenate((mnist_train_data, mnist_train_data2[0:(numImages-((numImages-60000)//60000+1)*60000)]), axis=0)
        mnist_train_labels = np.concatenate((mnist_train_labels, mnist_train_labels2[0:(numImages-((numImages-60000)//60000+1)*60000)]))
        mnist_test_data = mnist_test_data[:numImages//6, :, :]
        mnist_test_labels = mnist_test_labels[:numImages//6]
    elif numImages <= 60000:
        (mnist_train_data, mnist_train_labels),(mnist_test_data, mnist_test_labels) = keras.datasets.mnist.load_data()
        mnist_train_data = mnist_train_data[:numImages, :, :]
        mnist_train_labels = mnist_train_labels[:numImages]
        mnist_test_data = mnist_test_data[:numImages//6, :, :]
        mnist_test_labels = mnist_test_labels[:numImages//6]
    else:
        print('Invalid number of images')
        raise ValueError('Invalid numImages')
    if crop > 0:
        mnist_train_data = mnist_train_data[:, crop:(28-crop), crop:(28-crop)]
        mnist_test_data = mnist_test_data[:, crop:(28-crop), crop:(28-crop)]
    if verbose:
        print('MNIST train data array dimensions: ', np.shape(mnist_train_data))
        print('MNIST train labels array dimensions: ', np.shape(mnist_train_labels))
        print('MNIST test data array dimensions: ', np.shape(mnist_test_data))
        print('MNIST test labels array dimensions: ', np.shape(mnist_test_labels))
    if resize:
        if verbose:
            print('Original shape: ', mnist_train_data.shape)
        mnist_train_data_resized = np.zeros((1,m,m))
        for i in range(np.shape(mnist_train_data)[0]):
            temp = skt.resize(mnist_train_data[i,:,:],(m,m))[newaxis, :, :]
            mnist_train_data_resized = np.append(mnist_train_data_resized, temp, axis=0)
        mnist_test_data_resized = np.zeros((1, m, m))
        for i in range(np.shape(mnist_test_data)[0]):
            temp = skt.resize(mnist_test_data[i, :, :], (m, m))[newaxis, :, :]
            mnist_test_data_resized = np.append(mnist_test_data_resized, temp, axis=0)
        mnist_train_data = mnist_train_data_resized
        mnist_test_data = mnist_test_data_resized
        mnist_train_data = np.delete(mnist_train_data, 0, axis=0)
        mnist_test_data = np.delete(mnist_test_data, 0, axis=0)
        if verbose:
            print('Rescaled shape MNIST train data: ', mnist_train_data.shape)
            print('Rescaled shape MNIST test data: ', mnist_test_data.shape)
    if shuffle:
        perm = np.random.permutation(len(mnist_train_data))
        mnist_train_data = mnist_train_data[perm]
        mnist_train_labels = mnist_train_labels[perm]
    if save:
        np.savez_compressed('/homes/diegochav3z/S_LCA/S_LCA/datasets/mnist_' + str(m) + '_' + str(numImages) + '_' + str(tag), train_data = mnist_train_data,
                            test_data=mnist_test_data, train_labels=mnist_train_labels, test_labels=mnist_train_labels)
    trainLabel = np.zeros((numImages, 10))
    for i, j in enumerate(mnist_train_labels[0:numImages]):
        trainLabel[i,j] = 1
    return mnist_train_data, mnist_test_data, mnist_train_labels, mnist_test_labels, trainLabel
def loadXOR(numSamples=1000):
    trainData = np.random.randint(0,2,size=(numSamples,2))
    trainLabel = trainData[:,0]^trainData[:,1]
    return trainData, trainLabel
def loadAND(numSamples=1000):
    trainData = np.random.randint(0,2,size=(numSamples,2))
    trainLabel = trainData[:,0]&trainData[:,1]
    return trainData, trainLabel
def normalizeData(dataset):
    dataset = (dataset - dataset.min())/(dataset.max()-dataset.min())
    return dataset
def binarizeData(dataset):
    dataset = np.where(dataset > 0, 1, 0)
    return dataset

if __name__ == '__main__':
    m = 28
    mnist_train_data, mnist_test_data, mnist_train_labels, mnist_test_labels, trainOutput = loadMnist(m=m, resize=False, shuffle=False, save=False, numImages=10)
    mnist_train_data_resized, mnist_test_data_resized, mnist_train_labels_resized, mnist_test_labels_resized, trainOutput_resized = loadMnist(m=10, resize=True, shuffle=False, save= False, numImages=10)
    # trainDataXOR, trainLabelXOR = loadXOR(numSamples=5)
    # trainDataAND, trainLabelAND = loadAND(numSamples=5)
    #mnistTrainDataN, mnistTestDataN, mnistTrainLabelsN, mnistTestDataN
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(mnist_test_data[0])
    ax[1].imshow(mnist_test_data_resized[0])
    plt.show()




  