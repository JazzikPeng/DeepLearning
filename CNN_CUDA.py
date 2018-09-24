
# coding: utf-8

# Created on Sept. 19, 2018
'''
HW3: Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset.
The convolution network should use (A) dropout, (B) trained with RMSprop or ADAM, and (C)
data augmentation. For 10% extra credit, compare dropout test accuracy (i) using the heuristic
prediction rule and (ii) Monte Carlo simulation. For full credit, the model should achieve 80-90%
Test Accuracy. Submit via Compass (1) the code and (2) a paragraph (in a PDF document) which
reports the results and briefly
describes the model architecture.
Due September 28 at 5:00 PM.
'''
import numpy as np
import torch
import h5py
import time
# import matplotlib.pyplot as plt
import copy


# PyTorch Function
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn


# # Load Data Set #

# In[58]:


CIFAR10 = h5py.File('../CIFAR10.hdf5', 'r')
x_train = CIFAR10['X_train'][:]
y_train = np.array(CIFAR10['Y_train'])
x_test = CIFAR10['X_test'][:]
y_test = np.array(CIFAR10['Y_test'])
# print([i for i in CIFAR10.keys()])
CIFAR10.close()


# # Data Augmentation
# 1. Color Shift, randomly add numbers from -1 to 1 to all three RGB Channel

# In[59]:


x_train_aug = copy.deepcopy(x_train)
y_train_aug = copy.deepcopy(y_train)


# In[60]:


# def imshow(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, axes=(1, 2, 0)))
#     plt.show()


# ## Color Shift ##

# In[67]:


# Step 1 Color shift
def color_shift(inputs):
    img = copy.deepcopy(inputs)
    # img = copy.deepcopy(inputs)
    C1_shift = np.random.randint(0, 30)*np.random.randint(-1, 2) / 255
    C2_shift = np.random.randint(0, 30)*np.random.randint(-1, 2) / 255
    C3_shift = np.random.randint(0, 30)*np.random.randint(-1, 2) / 255
    img[0] = img[0] + C1_shift
    img[1] = img[1] + C2_shift
    img[2] = img[2] + C3_shift
    return img

# imshow(torchvision.utils.make_grid(torch.tensor((color_shift(x_train[0])))))


# ## random horizontal flip

# In[68]:


def flip_horizontal(inputs):
    img = copy.deepcopy(inputs)
    flip_dir = np.random.randint(0, 2)  # 1 flip horizontally; 0 dont flip
    if flip_dir == 1:
        img[0] = np.flip(img[0], 1)
        img[1] = np.flip(img[1], 1)
        img[2] = np.flip(img[2], 1)
    return img
# imshow(torchvision.utils.make_grid(torch.tensor(flip_horizontal(img))))


# ## Random Vertical flip

# In[69]:


def flip_vertical(inputs):
    img = copy.deepcopy(inputs)
    flip_dir = np.random.randint(0, 2)  # 1 flip horizontally; 0 dont flip
    if flip_dir == 1:
        img[0] = np.flip(img[0], 0)
        img[1] = np.flip(img[1], 0)
        img[2] = np.flip(img[2], 0)
    return img


# ## Apply Transformation to x_train and add training data to the training set

# In[64]:


for i in range(len(x_train_aug)):
    img = x_train_aug[i]
   # img = color_shift(img)
    img = flip_horizontal(img)
    img = flip_vertical(img)
    x_train_aug[i] = img


# In[65]:


x_train = np.array([x_train, x_train_aug]).reshape([2*50000, 3, 32, 32])
y_train = np.array([y_train, y_train_aug]).reshape(2*50000)
assert all(y_train[:50000] == y_train[50000:])


# In[66]:


# i = 13
# imshow(torchvision.utils.make_grid(torch.tensor(x_train[i])))
# imshow(torchvision.utils.make_grid(torch.tensor(x_train[50000+i])))


# ## 2.5  Randomize the training set

# In[70]:


arr = np.arange(0, 100000)
np.random.shuffle(arr)


# In[71]:


x_train = x_train[arr]
y_train = y_train[arr]


# In[73]:


# imshow(torchvision.utils.make_grid(torch.tensor(x_train[156])))
# print(y_train[156])


# In[74]:


print('Dataset Shape')
print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('x_test:', x_train.shape)
print('y_test:', y_test.shape)


# # Transform data to mini batchs #
#

########## Parameters ##########
batch_size = 16

# Load data
x_train = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=False, num_workers=4)
y_train = torch.utils.data.DataLoader(y_train, batch_size=batch_size, shuffle=False, num_workers=4)
x_test = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=False, num_workers=4)
y_test = torch.utils.data.DataLoader(y_test, batch_size=batch_size, shuffle=False, num_workers=4)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# # Visualize Our Dataset #


visualization = False
if visualization:
    dataiter = iter(x_train)
    images = dataiter.next()
    dataiter2 = iter(y_train)
    labels = dataiter2.next()

    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(16)))


# # Build PyTorch Model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1. 64 channels, k = 4,s = 1, P = 2.
        self.conv1 = nn.Conv2d(3, 64, 4, 1, 2)
        # Batch normalization
        self.conv1_bn = nn.BatchNorm2d(64)
        # 2. 64 channels, k = 4,s = 1, P = 2.
        self.conv2 = nn.Conv2d(64, 64, 4, 1, 2)
        # Max Pooling: s = 2, k = 2.
        self.pool2 = nn.MaxPool2d(2, 2)
        # Dropout 50%
        self.dropout2 = nn.Dropout(0.5)
        # Convolution layer 3: 64 channels, k = 4,s = 1, P = 2.
        self.conv3 = nn.Conv2d(64, 64, 4, 1, 2)
        self.conv3_bn = nn.BatchNorm2d(64)
        # Convolution layer 4: 64 channels, k = 4,s = 1, P = 2.
        self.conv4 = nn.Conv2d(64, 64, 4, 1, 2)
        # Max Pooling: s = 2, k = 2
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.5)
        # Convolution layer 5: 64 channels, k = 4,s = 1, P = 2
        self.conv5 = nn.Conv2d(64, 64, 4, 1, 2)
        self.conv5_bn = nn.BatchNorm2d(64)
        # Convolution layer 6: 64 channels, k = 3,s = 1, P = 0.
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 0)
        self.dropout6 = nn.Dropout(0.5)
        # Convolution layer 7: 64 channels, k = 3,s = 1, P = 0
        self.conv7 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv7_bn = nn.BatchNorm2d(64)
        # Convolution layer 8: 64 channels, k = 3,s = 1, P = 0.
        self.conv8 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv8_bn = nn.BatchNorm2d(64)
        self.dropout8 = nn.Dropout(0.5)
        # Fully connected layer 1: 500 units.
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # Fully connected layer 2: 500 units.
        self.fc2 = nn.Linear(500, 500)

    def forward(self, x):
        # print("input:", x.shape)
        x = self.conv1_bn(F.relu(self.conv1(x)))
        # print("1", x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        # print("2", x.shape)
        x = self.dropout2(x)
        x = self.conv3_bn(F.relu(self.conv3(x)))
        # print("3", x.shape)
        x = self.pool4(F.relu(self.conv4(x)))
        # print("4", x.shape)
        x = self.dropout4(x)
        x = self.conv5_bn(F.relu(self.conv5(x)))
        # print("5", x.shape)
        x = F.relu(self.conv6(x))
        # print("6", x.shape)
        x = self.dropout6(x)
        x = self.conv7_bn(F.relu(self.conv7(x)))
        # print("7", x.shape)
        x = self.conv8_bn(F.relu(self.conv8(x)))
        # print("8", x.shape)
        x = self.dropout8(x)
        # print("9", x.shape)
        x = x.view(-1, 64 * 4 * 4)
        # print("10",x.shape)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
net = CNN()
is_cuda = torch.cuda.is_available()
print('Is there CUDA Core?:', is_cuda)
if is_cuda:
    net.cuda()
    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    print(torch.cuda.device_count())
    cudnn.benchmark = True


# # Define Optimizer and loss function

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())


# In[ ]:

net.train()
start = time.time()
for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(zip(x_train, y_train), 0):
        # get the inputs
        inputs, labels = data
        labels = labels.long()
        #inputs, labels = inputs.cuda(), labels.cuda()
        if is_cuda:
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        if(epoch > 0):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if(state['step'] >= 1024):
                        state['step'] = 1000
        optimizer.step()
        # print statistics
        if torch.__version__ == '0.4.1':
            running_loss += loss.item()
        else:
            running_loss += loss.data[0]
        if i % 1000 == 999:    # print every 1000 mini-batches
            print('epoch: %d, Sample: %5d | loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0
    # Print Acc every 10 epochs
    if epoch % 10 == 0:
        correct = 0
        total = 0
        for data in zip(x_test, y_test):
            inputs, labels = data
            labels = labels.long()
            if is_cuda:
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy()
            labels = labels.data.cpu().numpy()
            # print(labels.size(0))
            total += len(labels)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))
end = time.time()
print('Finished Training, Time Cost:', start-end)


# # Test Accuracy on Test Set

# In[11]:

net.eval()
correct = 0
total = 0

for data in zip(x_test, y_test):
    images, labels = data
    labels = labels.long()
    if is_cuda:
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
    else:
        images, labels = Variable(images), Variable(labels)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.cpu().numpy()
    labels = labels.data.cpu().numpy()
    # print(labels.size(0))
    total += len(labels)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))
