#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 18:25:42 2020

@author: anaraquelpengelly
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image, ImageOps


# =============================================================================
# Function to open image as a np.array:
# =============================================================================

def image_open_arr(filename):
    im=Image.open(filename)
    im_arr=np.asarray(im)
    return im_arr

# =============================================================================
# To resize images: 
# =============================================================================

def resize_image(desired_size, image):
    old_size=image.size
    ratio=float(desired_size)/max(old_size)
    new_size=tuple([int(x*ratio) for x in old_size])
    resized_im =image.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(resized_im, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))
    return new_im

# =============================================================================
# To pad and crop images:
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

# =============================================================================
# Function to extract and resize images and labels 
# =============================================================================

def resize_extract_images(path, label_df, desired_size):
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
                #here call resize function:
                new_im = resize_image(desired_size, im)
                
                
#create np_array from new image:  
        #dimensions are fine! 
                im_array = np.asarray(new_im)
                #normalise images:
                all_images_as_array.append(im_array)
                l,b,c = im_array.shape
#THIS NEEDS TO BE TESTED ON A SMALLARRAY OF IMAGES                
    # X_4d =all_images_as_array
    # X4D_centered= X_4d - X_4d.mean(axis=0, keepdims=True)
    # X4D_sd=X_4d.std(axis=0, keepdims=True)
    # X4D_divided =np.divide(X4D_centered, X4D_sd , out=np.zeros_like(X4D_centered), where=X4D_sd!=0)
    #TODO ! need to put things in here!! check if this works! 
                
    #flatterned_array=all_images_as_array.reshape(len(label),l*b*c)
    return np.array(all_images_as_array), np.array(label)
# =============================================================================
#  Function to extract and pad+crop images and labels 
# =============================================================================
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

# =============================================================================
# Function to pad, crop, and normalise images 
# =============================================================================
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