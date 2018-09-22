


import numpy as np
import h5py
import time
import copy
from random import randint
from scipy import signal
# load MNIST data
MNIST_data = h5py.File('../MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))
MNIST_data.close()


# Implementation of stochastic gradient descent algorithm
# number of inputs
num_inputs = 28*28
# number of outputs
# 1. Initialization
num_outputs = 10
dH = 100
model = {}
# Three Channel
global channel
channel = 3
model['K'] = np.random.randn(3, 3, channel) / np.sqrt(num_inputs)
model['W'] = np.random.randn(10, 26, 26, 3) / np.sqrt(num_inputs)
model['b'] = np.zeros((num_outputs, 1))
model_grads = copy.deepcopy(model)


# Define Components of one layer neural network
def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ


def relu(Z):
    """
    Implement the RELU function.
    """
    A = np.maximum(0, Z)

    assert(A.shape == Z.shape)
    return A


def drelu(z):
    '''
    derivative of relu function
    '''
    # This is to avoid bug when the impossible happens
    z[z > 0] = 1
    z[z <= 0] = 0
    return z

# helper function for backward


def e(y, num_outputs=10):
    '''
    e(y) function
    '''
    ret = np.zeros((num_outputs, 1), dtype=np.int)
    ret[y] = 1
    return ret


def conv(X, K):
    a, b = X.shape[0], X.shape[1]
    f = K.shape[0]
    re = np.zeros((a-f+1)**2).reshape(a-f+1, b-f+1)
    for i in range(a-f+1):
        for j in range(b-f+1):
            # print(X.shape)
            temp1 = np.arange(f) + i
            temp2 = np.arange(f) + j
            # print(temp1, temp2)
            subX = X[np.ix_(temp1, temp2)]
            # print(subX, K)
            # print(i,j,np.sum(subX * K))
            #print(subX.shape, X.shape)
            re[i, j] = np.sum(subX * K)
    return re


def conv3D(X, K):
    a, b = X.shape[0], X.shape[1]
    f = K.shape[0]
    re = np.zeros((a - f + 1, b - f + 1, channel))
    # print(re.shape)
    for i in range(channel):
        re[:, :, i] = conv(X, K[:, :, i])
    return re


def forward(x, y, model):
    x = x.reshape(28, 28)
    Z = conv3D(x, model['K'])
    H = relu(Z)
    U = np.tensordot(model['W'], H, axes=3).reshape(num_outputs, 1) + model['b']
    p = softmax_function(U)
    return p, H, Z


# In[7]:


def backward(x, y, p, H, Z, model, model_grads):
    x = x.reshape(28, 28)
    drhodU = -(e(y) - p)
    drhodb = drhodU
    delta = np.tensordot(drhodU.squeeze(), model['W'], axes=1)
    drhodW = np.tensordot(drhodU.squeeze(), H, axes=0)
    dsigmaZ = drelu(Z)
    dfilter = np.multiply(dsigmaZ, delta)

    drhodK = copy.deepcopy(model['K'])

    drhodK = conv3D(x, dfilter)
    # for i in range(channel):

    # drhodK[:,:,i] = signal.correlate2d(x, dfilter[:, :, i], mode='valid',  boundary='wrap')

    model_grads['K'] = drhodK
    model_grads['W'] = drhodW
    model_grads['b'] = drhodb

    return model_grads


import time
time1 = time.time()
LR = 0.01
num_epochs = 10
for epochs in range(num_epochs):
    # Learning rate schedule
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
    total_correct = 0
    for n in range(len(x_train)):
        n_random = randint(0, len(x_train)-1)
        y = y_train[n_random]
        x = x_train[n_random][:]
        # x = x.reshape(28, 28)
        p, H, Z = forward(x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x, y, p, H, Z, model, model_grads)
        model['K'] = model['K'] - LR*model_grads['K']
        model['W'] = model['W'] - LR*model_grads['W']
        model['b'] = model['b'] - LR*model_grads['b']
        # print('h')
    print('epochs: '+str(epochs), ' | Train Acc: ' + str(total_correct/np.float(len(x_train))))
time2 = time.time()
print('Run Time:', time2-time1)


############### Test Set Accuracy ############################
# test data
total_correct = 0
for n in range(len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    p, _, _ = forward(x, y, model)
    prediction = np.argmax(p)
    # print(p, np.argmax(p))
    if (prediction == y):
        total_correct += 1
print('Test Set Accuracy:', total_correct/np.float(len(x_test)))
