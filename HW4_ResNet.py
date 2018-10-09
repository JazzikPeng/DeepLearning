# coding: utf-8

# Created on Sept. 19, 2018
# Tune paramters according to TA
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



##################### Parameters Setup #####################
batch_size = 128




############################################################

# Compute mean and std
train_transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR100(root='./data/', train=True, download=True, transform=train_transform)
# print(vars(train_set))
print(train_set.train_data.shape)
#mean = np.mean(train_set.train_data, axis=(0, 1, 2))/255
#std = np.std(train_set.train_data, axis=(0, 1, 2))/255
#print('mean:', mean)
#print('sdt', std)


# Read in Data
print('==> Preparing dataset %s' % 'CIFAR100')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    # transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])

dataloader = torchvision.datasets.CIFAR100
num_classes = 100


trainset = dataloader(root='./data', train=True, download=False, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

class basic_block(nn.Module):
    def __init__(self, in_channel, out_channel, s, p):
        super(basic_block, self).__init__()
        self.conv2d = nn.Conv2d(in_channel, out_channel, 3,  s, p)
        self.bn = nn.BatchNorm2d(out_channel)
        self.conv2d_2 = nn.Conv2d(out_channel, out_channel, 3, 1, p)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.residual = nn.Sequential()
        if s != 1 or in_channel != out_channel:  # Keep the dim consistent
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                          kernel_size=1, stride=s),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.conv2d(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.conv2d_2(out)
        out = self.bn2(out)

        out += self.residual(x)
        return out

class resNet(nn.Module):
    def __init__(self, num_blocks=[2, 4, 4, 2]):
        '''
        num_blocks is a list
        '''
        super(resNet, self).__init__()
        # 1. 32 channels, k = 3,s = 1, P = 2.
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        # Batch normalization
        self.conv1_bn = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.2)
        self.conv2_x = self.block_helper(num_blocks[0], 32, 32, 1, 1)
        self.conv3_x = self.block_helper(num_blocks[1], 32, 64, 2, 1)
        self.conv4_x = self.block_helper(num_blocks[2], 64, 128, 2, 1)
        self.conv5_x = self.block_helper(num_blocks[3], 128, 256, 2, 1)

        self.maxpool = nn.MaxPool2d(3, 2, 1) # kernel size 3, stride 2, padding 1
        self.fc = nn.Linear(256*2*2, 100)

    def block_helper(self, num, in_channel, out_channel, s, p):
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
        return x


# # Train and Test
def learning_rate_decay(optimizer, factor=2):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] / factor

net = resNet()
is_cuda = torch.cuda.is_available()
print('Is there CUDA Core?:', is_cuda)
if is_cuda:
    net.cuda()
    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    print(torch.cuda.device_count())
    cudnn.benchmark = True


# # Define Optimizer and loss function

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters(), lr=0.001)

best_test_acc = 0
net.train()
start = time.time()
for epoch in range(500):  # loop over the dataset multiple times
    running_loss = 0.0
    break_flag = False
    if epoch % 30 ==29:
        learning_rate_decay(optimizer, factor=2)
    for i, data in enumerate(trainloader):
        # get the inputs
        inputs, labels = data
        labels = labels.long()

       # for j in range(len(inputs)): # Augmentation
           # inputs[j] = data_augmentation(inputs[j])

        if is_cuda:
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        predicted = outputs.data.max(1)[1]
        train_accuracy = (float(predicted.eq(labels.data).sum()) / float(batch_size))

        loss = criterion(outputs, labels)
        loss.backward()
       # if(epoch > 0):
       #     for group in optimizer.param_groups:
       #         for p in group['params']:
       #             state = optimizer.state[p]
       #            try:
       #                 if(state['step'] >= 1024):
       #                     state['step'] = 1000
       #             except:
       #                 pass
        optimizer.step()
        # print statistics
        if torch.__version__ == '0.4.1':
            running_loss += loss.item()
        else:
            running_loss += loss.data[0]
        if i % 20 == 19:    # print every 10 mini-batches
            print('epoch: %d, Sample: %5d | loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
    # Print Acc every 10 epoch
    if epoch % 10 == 9:
        net.eval()
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
        test_acc = correct / total
        print('training acc on this epoch', train_accuracy)
        print('Accuracy of the network on the 10000 test images: %f %%' % (100*test_acc))
        if test_acc > 0.63:
            print('ready to break')
            break_flag = True
        #if train_accuracy>0.90 and test_acc<0.5:
            #print('This is a fucking overfit')
            #break_flag = True
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        print('So far best acc is:', best_test_acc)
        net.train()
    if break_flag:
        break

end = time.time()
print('Finished Training, Time Cost:', end-start)

# Save final Model After Train
torch.save(net, './Resnet_V1')



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
