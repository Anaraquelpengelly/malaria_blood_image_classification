#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:06:16 2020

@author: anaraquelpengelly
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps
import sklearn
from sklearn import neighbors, ensemble, svm, metrics 
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
import logging
#function used:
def pad_crop_extract_images(path, label_df, max_size, desired_size):
    max_size=max_size+15
    all_images_as_array=[]
    label=[]    
    for filename in os.listdir(path):
        for index, row in label_df.iterrows():
            if filename==row["0"]:
                # 1-Feed the label vector:
                if row["infect_status"]==1:
                    label.append(1)
                else:
                    label.append(0)
                # 2- read the image:
                im = Image.open(path+filename)
                cropped_im = pad_crop_image(desired_size, max_size, im)
        #create np_array from new image:  
        #dimensions are fine! 
                np_array = np.asarray(cropped_im)
                l,b,c = np_array.shape
                np_array = np_array.reshape(l*b*c,)
                all_images_as_array.append(np_array)

    return np.array(all_images_as_array), np.array(label)
#


