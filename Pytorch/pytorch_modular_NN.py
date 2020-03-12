#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:46:49 2020

@author: anaraquelpengelly
"""
import torch
import torch.nn as nn
# =============================================================================
# Define NN structure
# =============================================================================
class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output
    
    
    
class SimpleNet(nn.Module):
    def __init__(self,num_classes=2):
        super(SimpleNet,self).__init__()
        
        #Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3,out_channels=224)
        self.unit2 = Unit(in_channels=224, out_channels=224)
        self.unit3 = Unit(in_channels=224, out_channels=224)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=224, out_channels=224)
        self.unit5 = Unit(in_channels=448, out_channels=448)
        self.unit6 = Unit(in_channels=448, out_channels=448)
        self.unit7 = Unit(in_channels=448, out_channels=448)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=448, out_channels=896)
        self.unit9 = Unit(in_channels=896, out_channels=896)
        self.unit10 = Unit(in_channels=896, out_channels=896)
        self.unit11 = Unit(in_channels=896, out_channels=896)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=896, out_channels=896)
        self.unit13 = Unit(in_channels=896, out_channels=896)
        self.unit14 = Unit(in_channels=896, out_channels=896)

        self.avgpool = nn.AvgPool2d(kernel_size=4)
        
        #Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 ,self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=896,out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1,896)
        output = self.fc(output)
        return output    
    
# =============================================================================
# Loading the data   
# =============================================================================
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

#1-Define transformations to be applied to the train data:
'''
NOTE: Here, since we are using the padded and resized images, we only 
need to send the images to tensor and normalise them.
'''
train_transformations = transforms.Compose([
    transforms.ToTensor(),
    #here we use the computed means I found in script quickCode.py:
    transforms.Normalize([0.1879,0.1499,0.1592], [0.3263, 0.2615, 0.2764])
])

#2-Load the train dataset using torchvision
train_set = ImageFolder(root='/data_path', transform=train_transformations)
#3-create an instance of the DataLoader to hold the train images:
train_loader = DataLoader(train_set,batch_size=100,shuffle=True,num_workers=4)

#4-Define transformations for the test set
test_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.1879,0.1499,0.1592], [0.3263, 0.2615, 0.2764])

])

#5-Load the test set, note that train is set to False
test_set = ImageFolder(root="./data", transform=test_transformations)

#6-Create a loder for the test set, note that both shuffle is set to false for the test loader
test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=4)

# =============================================================================
# Training the data
# =============================================================================
from torch.optim import Adam
# Check if gpu support is available
cuda_avail = torch.cuda.is_available()

# Create model, optimizer and loss function
model = SimpleNet(num_classes=2)

#if cuda is available, move the model to the GPU
if cuda_avail:
    model.cuda()

#Define the optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()

