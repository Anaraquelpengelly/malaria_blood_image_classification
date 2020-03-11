#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:02:34 2020

@author: anaraquelpengelly
"""
# =============================================================================
# Import packages needed
# =============================================================================

import numpy as np
import os
import pandas as pd
from PIL import Image, ImageOps

# =============================================================================
# Functions needed
# =============================================================================
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




def get_channel_mean_sd(df, colname, desired_size, max_size, im_path):
    em = np.empty([0, 3, 224, 224])
    m=[]
    s=[]
    for index, row in df.iterrows():
    
        filename=im_path+row[colname]
        im=Image.open(filename)
        #HERE USE FUNCTION TO PAD THE IMAGE:
 # old_size[0] is in (width, height) format
        new_im=pad_crop_image(desired_size, max_size, im)
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

# =============================================================================
# Set variables needed:
# =============================================================================
im_path=
df_path=
results_path=
max_size=394
desired_size= 224
all_image_df= pd.read_csv(df_path+"shuffled_labels.csv")
colname="0"

# =============================================================================
# perfom process
# =============================================================================

m_a, s_a=nn.get_channel_mean_sd(all_image_df, colname, desired_size, max_size, im_path)

print(m_a, s_a)
# =============================================================================
# save the results in a txt in path
# =============================================================================
D={"mean":m_a;"SD":s_a}
res_df=pd.DataFrame(D)
res_df.to_csv((results_path+"results_ana"), index=False)