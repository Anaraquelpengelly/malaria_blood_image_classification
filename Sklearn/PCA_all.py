#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:50:56 2020

@author: anaraquelpengelly
"""


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps
import sklearn
from sklearn import neighbors, ensemble, svm, metrics 
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
import logging

#Functions: 

def pad_crop_image(desired_size, max_size, image):
    old_size = image.size
    max_size=max_size+14
    delta_w=max_size - old_size[0]
    delta_h=max_size - old_size[1]
    padding= (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    pad_im=ImageOps.expand(image, padding)
    
    left = (max_size-desired_size)/2
    top =  (max_size-desired_size)/2
    right = max_size-(max_size-desired_size)/2
    bottom = max_size-(max_size-desired_size)/2
    cropped_im = pad_im.crop((left, top, right, bottom))
    return cropped_im


#########
def norm_pad_crop_extract_images(path, label_df, max_size=394, desired_size=224):
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
                
                all_images_as_array.append(np_array)
    #normalise the images:            
    X4D=np.array(all_images_as_array)            
    X4D_centered= X4D - X4D.mean(axis=0, keepdims=True)
    X4D_sd=X4D.std(axis=0, keepdims=True)
    X4D_divided =np.divide(X4D_centered, X4D_sd , out=np.zeros_like(X4D_centered), where=X4D_sd!=0)            
    
    X4D_norm = X4D_divided.reshape(X4D_divided.shape[0],l*b*c)

    return X4D_norm, np.array(label)



#1- read images
path="/rds/general/user/arp219/home/ML_project/Malaria_blood_image_classification/sklearn_HPC/"
im_path="/rds/general/user/arp219/home/ML_project/Malaria_blood_image_classification/sklearn_HPC/all_images/"
fig_path="/rds/general/user/arp219/home/ML_project/Malaria_blood_image_classification/sklearn_HPC/results/"

data=pd.read_csv(path+"labels.csv")

X,y=norm_pad_crop_extract_images(im_path, data, 394, 224)


#2-get plot to determine how many compnents explain how much of the variance:

pca_test2 = PCA().fit(X)
plt.plot(np.cumsum(pca_test2.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.savefig((fig_path+"PCA_scree_plot.png"), dpi=300)



