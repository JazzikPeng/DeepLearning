
# coding: utf-8

# In[33]:



# Created on Oct. 5th, 2018
'''
Fine-tune a pre-trained ResNet-18 model and achieve at least 70% test accuracy.

'''

# coding: utf-8

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
import torch.utils.data as data
import torch.backends.cudnn as cudnn


# # read ResNEt
# 

# In[ ]:


# read ResNet
net = torchvision.models.resnet18(pretrained=True)
# Change ResNet Output to 100
net = torch.nn.Sequential(net, torch.nn.Linear(net.fc.out_features,100))


# In[ ]:


# Load Dataset
# Compute mean and std
train_transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR100(root='./data/', train=True, download=True, transform=train_transform)
#print(vars(train_set))
print(train_set.train_data.shape)
mean = np.mean(train_set.train_data, axis=(0,1,2))/255  
std = np.std(train_set.train_data, axis=(0,1,2))/255
print(mean)
print(std)

# Read in Data
print('==> Preparing dataset %s' % 'CIFAR100')
transform_train = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])

transform_test = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

dataloader = torchvision.datasets.CIFAR100
num_classes = 100

trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=500, shuffle=True, num_workers=4)

testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
testloader = data.DataLoader(testset, batch_size=500, shuffle=False, num_workers=4)


# # Train

# In[ ]:


is_cuda = torch.cuda.is_available()
print('Is there CUDA Core?:', is_cuda)
if is_cuda:
    net.cuda()
    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    print(torch.cuda.device_count())
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())


best_test_acc = 0
net.train()
start = time.time()
for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # print(i)
        # get the inputs
        inputs, labels = data
        # print(inputs.shape)
        labels = labels.long()
        if is_cuda:
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        # print(outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
#         if(epoch > 0):
#             for group in optimizer.param_groups:
#                 for p in group['params']:
#                     state = optimizer.state[p]
#                     if(state['step'] >= 1024):
#                         state['step'] = 1000
        optimizer.step()
        # print('Complete one')
        # print statistics
        if torch.__version__ == '0.4.1':
            running_loss += loss.item()
        else:
            running_loss += loss.data[0]
        if i % 100 == 99:    # print every 10 mini-batches
            print('epoch: %d, Sample: %5d | loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    # Print Acc every 10 epoch
    if epoch % 10 == 9:
        correct = 0
        total = 0
        net.eval()
        for i, data in enumerate(testloader):
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
        test_acc = (100 * correct / total)
        print('Accuracy of the network on the 10000 test images: %f %%' % test_acc)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        net.train()
end = time.time()
print('Finished Training, Time Cost:', end-start)

# Save final Model After Train


# In[ ]:

print('The best_test_acc', best_test_acc)

## Test Accuracy on Test Set Heuristic Method ##
net.eval()
correct = 0
total = 0

for i, data in enumerate(testloader):
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

