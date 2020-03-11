#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:20:21 2020

@author: anaraquelpengelly
"""


import torch
import torchvision
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader
import os
import multiprocessing as mp
import pandas as pd
from PIL import Image, ImageOps
from shutil import copy2
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
#os.getcwd()
os.chdir('/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/')






image_transform = Compose([ToTensor()])

dset = torchvision.datasets.ImageFolder('padded_cell_images/', transform = image_transform)

loader = DataLoader(dset, batch_size = len(dset),
                    num_workers = mp.cpu_count()-1)

# for (image, label) in loader:
#     meanR = image[:,0,:,:].mean()
#     print(image.shape)
#     meanG = image[:,1,:,:].mean()
#     meanB = image[:,2,:,:].mean()
#     sdR = image[:,0, :,:].std()
#     sdG = image[:,1,:,:].std()
#     sdB = image[:,2,:,:].std()
#     break

    
    
image_transform_2 = Compose([ToTensor(), Normalize([0.1879,0.1499,0.1592], [0.3263, 0.2615, 0.2764])])
#means=[tensor(), tensor(), tensor(0.1592)
#sds=[tensor(0.3263), tensor(0.2615), tensor(0.2764)]
dset = torchvision.datasets.ImageFolder('padded_cell_images/',
                                        transform = image_transform_2)


                 
                  
                    
#%%                   


# train_dset_leng = int(len(dset)*0.8)
# test_dset_leng = len(dset) - train_dset_leng

# lengths = [train_dset_leng, test_dset_leng]

# train_dset, test_dset = torch.utils.data.random_split(dset, lengths)  
# print(len(train_dset))
# print(len(test_dset))

# train_dataloader = DataLoader(
#     train_dset, 
#     batch_size = 100, 
#     num_workers = mp.cpu_count()-1,
#     )

# test_dataloader = Dataloader(
#     test_dset, 
#     batch_size = 100,
#     num_workers = mp.cpu_count()-1,
#     )

    
