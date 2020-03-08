#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:20:21 2020

@author: anaraquelpengelly
"""


import torch
import torchvision
from torchvision.transforms import Compose, CenterCrop, ToTensor
from torch.utils.data import DataLoader
import os
import multiprocessing as mp
os.getcwd()
os.chdir('/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/')
image_transform = Compose([CenterCrop(256),ToTensor()])

dset = torchvision.datasets.ImageFolder('cell_images/',
                                        transform = image_transform)

loader = DataLoader(dset, batch_size = len(dset),
                    num_workers = mp.cpu_count()-1)

for (image, label) in loader:
    meanR = image[:,0,:,:].mean()
    meanG = image[:,1,:,:].mean()
    meanB = image[:,2,:,:].mean()
    sdR = image[:,0, :,:].std()
    sdG = image[:,1,:,:].std()
    sdB = image[:,2,:,:].std()
    
    


    
    d
