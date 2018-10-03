# coding: utf-8

# Created on Sept. 19, 2018
'''
Build the Residual Network specified in Figure 1 and achieve at least 60% test accuracy.
In the homework, you should define your “Basic Block” as shown in Figure 2. For each
weight layer, it should contain 3 × 3 filters for a specific number of input channels and
output channels. The output of a sequence of ResNet basic blocks goes through a max
pooling layer with your own choice of filter size, and then goes to a fully-connected
layer. The hyperparameter specification for each component is given in Figure 1. Note
that the notation follows the notation in He et al. (2015).

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
import torch.utils.data as data
import torch.backends.cudnn as cudnn


# Compute mean and std
train_transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR100(root='./data/', train=True, download=True, transform=train_transform)
# print(vars(train_set))
print(train_set.train_data.shape)
mean = np.mean(train_set.train_data, axis=(0, 1, 2))/255
std = np.std(train_set.train_data, axis=(0, 1, 2))/255
print(mean)
print(std)


# Read in Data
print('==> Preparing dataset %s' % 'CIFAR100')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

dataloader = torchvision.datasets.CIFAR100
num_classes = 100


trainset = dataloader(root='./data', train=True, download=False, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4)

testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)


# # Construct The Model
# ## Construct the basic block


class basic_block(nn.Module):
    def __init__(self, in_channel, out_channel, s, p):
        super(basic_block, self).__init__()
        self.conv2d = nn.Conv2d(in_channel, out_channel, 3, s, p)
        self.bn = nn.BatchNorm2d(out_channel)
        self.conv2d_2 = nn.Conv2d(out_channel, out_channel, 3, 1, p)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.residual = nn.Sequential()
        if s != 1 or in_channel != out_channel:  # Keep the dim consistent
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                          kernel_size=1, stride=s),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv2d_2(x)
        x = self.bn2(x)

        x += self.residual(x)
        return x


class resNet(nn.Module):
    def __init__(self, basic_block, num_blocks=[2, 4, 4, 2]):
        '''
        num_blocks is a list
        '''
        super(resNet, self).__init__()
        # 1. 32 channels, k = 3,s = 1, P = 2.
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        # Batch normalization
        self.conv1_bn = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.5)
        self.conv2_x = self.block_helper(basic_block, num_blocks[0], 32, 32, 1, 1)
        self.conv3_x = self.block_helper(basic_block, num_blocks[1], 32, 64, 2, 1)
        self.conv4_x = self.block_helper(basic_block, num_blocks[2], 64, 128, 2, 1)
        self.conv5_x = self.block_helper(basic_block, num_blocks[3], 128, 256, 2, 1)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(256*2*2, 100)

    def block_helper(self, basic_block, num, in_channel, out_channel, s, p):
        blocks = []
        for i in range(num):
            # stride for other following conv uses stride of 2
            if i == 0:
                blocks.append(basic_block(in_channel, out_channel,  s, p))
            else:
                blocks.append(basic_block(out_channel, out_channel,  1, p))
        return nn.Sequential(*blocks)

    def forward(self, x):
        # print("input:", x.shape)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.dropout1(x)
        # print('step after dropout1', x.shape)
        x = self.conv2_x(x)

        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.maxpool(x)
        #print('fc1', x.shape)

        x = x.view(x.size(0), -1)
        #print('fc2', x.shape)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# # Train and Test


net = resNet(basic_block)
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

best_test_acc = 0
net.train()
start = time.time()
for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # print(i)
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
#         if(epoch > 0):
#             for group in optimizer.param_groups:
#                 for p in group['params']:
#                     state = optimizer.state[p]
#                     if(state['step'] >= 1024):
#                         state['step'] = 1000
        optimizer.step()
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

end = time.time()
print('Finished Training, Time Cost:', end-start)

# Save final Model After Train
torch.save(net, './Resnet_V1')


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
