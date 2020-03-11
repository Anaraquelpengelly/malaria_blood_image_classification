#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:46:41 2020

@author: anaraquelpengelly
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image, ImageOps
from shutil import copy2
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

def pad_crop_image(image, desired_size=224, max_size=394):
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

im_path="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/cell_images/"

folder_names = ['Parasitized', 'Uninfected']
image_names = [os.listdir(im_path+i) for i in folder_names]

# _path = f'{im_path}{folder_name}/{image_name}'
# image = Image.open(_path)

# image = pad_crop_image(image)

# Image.save(image, f'{im_path}padded_{folder_name}/{image_name}')
            
j = 1
print(j)
for i, folder_name in enumerate(folder_names): 
        for image_name in image_names[i]:
            if j % 100 == 0:
                print(j)
            try:
                _path = f'{im_path}{folder_name}/{image_name}'
                image = Image.open(_path)
                
                image = pad_crop_image(image)
                
                image.save(f'{im_path}padded_{folder_name}/{image_name}')
            except: 
                pass
            j += 1
    
print('got here')

#%%
#make the test and the train set: 
#1 Read the train and test sets: 
path="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/"
train_df=pd.read_csv(path+"training_labels.csv")
test_df=pd.read_csv(path+"test_labels.csv")
train_path_para=path+"train_images/parazitised"
train_path_unin=path+"train_images/uninfected"
test_path_para=path+"test_images/parazitised"
test_path_unin=path+"test_images/uninfected"



for filename in os.listdir(path+'padded_cell_images/Parasitized'):
            for index, row in train_df.iterrows():
                if filename==row["0"]:
                    copy2((path+f'padded_cell_images/Parasitized/{filename}'), train_path_para) 
                    
#%% attempt with multiprocessing:                    
 
def copy_image( df, source_path=path+'padded_cell_images/Uninfected', dest_path=train_path_unin):
    for filename in os.listdir(source_path):
            for index, row in df.iterrows():
                if filename==row["0"]:
                    copy2((source_path+f"/{filename}"), dest_path)
                    
                    

cores=mp.cpu_count()
df_split = np.array_split(train_df, cores, axis=0)
# create the multiprocessing pool
pool = Pool(cores)
# process the DataFrame by mapping function to each df across the pool
df_out = np.vstack(pool.map(copy_image, df_split))
  # close down the pool and join
pool.close()
pool.clear()