#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:40:04 2020

@author: anaraquelpengelly
this should go into the trianing/test script 
from pytorch_dataset import MalariaDataset


malaria_train = MalariaDateset(train_DF)
malaria_test = MalariaDateset(test_DF)
train_dataset = torch.utils.data.DataLoader(malaria_train, batch_size=10)

train(..., malaria_train):
    for batch_idx, (images, labels) in enumerate(malaria_train):
        images
        labels

"""
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#%%
class MalariaDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataframe, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.name_labeldf = dataframe
    def __len__(self):
        return len(self.name_labeldf)

    def __getitem__(self, idx):
        im=Image.open(path+name_labeldf[idx,"0"]
        label=name_labeldf[idx, "column"]
        #add function to padd and crop and normalise
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

        # if self.transform:
        #     sample = self.transform(sample)

        return image, label