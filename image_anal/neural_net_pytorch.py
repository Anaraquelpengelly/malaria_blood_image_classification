#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 19:55:24 2020

@author: anaraquelpengelly
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
#%% 
#you do this so that the sklearn_ana script is found:
import sys
sys.path.append(r"/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/scripts/image_anal")

#%%

import  sklearn_ana as sk


#%%
def get_channel_mean_sd(df, colname, desired_size, max_size, im_path):
    em = np.empty([0, 3, 224, 224])
    m=[]
    s=[]
    for index, row in df.iterrows():
    
        filename=im_path+row[colname]
        im=Image.open(filename)
        #HERE USE FUNCTION TO PAD THE IMAGE:
 # old_size[0] is in (width, height) format
        new_im=sk.pad_crop_image(desired_size, max_size, im)
        #save image as np array:
        new_im=np.asarray(new_im)
        im_trans=new_im.transpose((2, 0, 1))#might not need this if processing downstream with pytorch.
        im_4D=im_trans[np.newaxis, :, :, :]
        em=np.append(em, im_4D, 0)
#now add the means and sds of each of the channels to the m and s variable: 
    m.append(em[:, 0].mean()/255)
    m.append(em[:, 1].mean()/255)
    m.append(em[:, 2].mean()/255)
    s.append(em[:, 0].std()/255)
    s.append(em[:, 1].std()/255)
    s.append(em[:, 2].std()/255)
    print(f"The shape of your tensor is {em.shape}")
    return m, s
#%%