
# coding: utf-8

# # HW1
# Implement and train a neural network from scratch in Python for the MNIST dataset (no PyTorch). The neural network should be trained on the Training Set using stochastic gradient descent. It should achieve 97-98% accuracy on the Test Set. For full credit, submit via Compass (1) the code and (2) a paragraph (in a PDF document) which states the Test Accuracy and briefly describes the implementation. Due September 7 at 5:00 PM.

# In[1]:


import numpy as np
import h5py
import time
import copy
from random import randint
# load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))
MNIST_data.close()


# In[2]:


# Implementation of stochastic gradient descent algorithm
# number of inputs
num_inputs = 28*28
# number of outputs
# 1. Initialization
num_outputs = 10
dH = 100
model = {}
model['W1'] = np.random.randn(dH, num_inputs) / np.sqrt(num_inputs)
model['b1'] = np.random.randn(dH)
model['b2'] = np.random.randn(num_outputs)
model['C'] = np.random.randn(num_outputs, dH) / np.sqrt(num_inputs)
model_grads = copy.deepcopy(model)


# In[3]:


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
    if z-0.0 == 0:
        z = z+1e-8
    if z < 0:
        return 0.0
    else:
        return 1.0


def forward(x, y, model):
    '''
    Forward propagation of one hidden layer neural network
    return p, output
    H and Z are for back-propagation
    '''
    Z = np.dot(model['W1'], x) + model['b1']
    H = np.array([relu(z) for z in Z])
    U = np.dot(model['C'], H) + model['b2']
    p = softmax_function(U)
    return p, H, Z


# In[4]:


# helper function for backward
def e(y, num_outputs=10):
    '''
    e(y) function
    '''
    ret = np.zeros(num_outputs, dtype=np.int)
    ret[y] = 1.0
    return ret


# In[5]:


# backward propagation and store grads in model_grads
def backward(x, y, p, H, Z, model, model_grads):
    '''
    neural network backward propagation
    '''
    drhodU = - e(y) + p
    drhodb2 = drhodU

    drhodC = np.dot(drhodU.reshape(len(drhodU), 1), np.transpose(H.reshape(len(H), 1)))

    delta = np.dot(np.transpose(model['C']), drhodU)

    drhodb1 = delta*[drelu(z) for z in Z]

    drhodW = np.dot(drhodb1.reshape(len(drhodb1), 1), np.transpose(x.reshape(len(x), 1)))

    model_grads['C'] = drhodC
    model_grads['b2'] = drhodb2
    model_grads['W1'] = drhodW
    model_grads['b1'] = drhodb1
    assert model_grads['C'].shape == drhodC.shape
    assert model_grads['b2'].shape == drhodb2.shape
    assert model_grads['W1'].shape == drhodW.shape
    assert model_grads['b1'].shape == drhodb1.shape
    return model_grads


# In[6]:


import time
time1 = time.time()
LR = 0.01
num_epochs = 12
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
        p, H, Z = forward(x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x, y, p, H, Z, model, model_grads)
        model['W1'] = model['W1'] - LR*model_grads['W1']
        model['C'] = model['C'] - LR*model_grads['C']
        model['b2'] = model['b2'] - LR*model_grads['b2']
        model['b1'] = model['b1'] - LR*model_grads['b1']
    print('epochs: '+str(epochs), ' | Train Acc: ' + str(total_correct/np.float(len(x_train))))
time2 = time.time()
print('Run Time:', time2-time1)


# In[7]:


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
