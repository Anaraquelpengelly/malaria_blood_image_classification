#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:46:44 2020

@author: anaraquelpengelly
"""

path="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/"
im_path="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/cell_images/all/"

'''
#make function to extract images and resample them to wanted pixel size and 
rescale them:

'''
#download required libraries
#%%
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
#important things to consider: 
#for running things in parallel: torch.nn.DataParallel

#%%
#one image todo tests: 

filename1 = os.path.join(im_path,'C101P62ThinF_IMG_20150918_151006_cell_61.png')
filename2 = os.path.join(im_path, "C99P60ThinF_IMG_20150918_142334_cell_9.png")
#%%
def image_open(filename):
    im=Image.open(filename)
    im_arr=np.asarray(im)
    return im_arr

#%%
 
img_t=image_open(filename1)    
#%%

img1 = image_open(filename1)
img2 = image_open(filename2)


#playing with RGB  
##for two images:
#%%
###This now works!

#resize images:
#open images with PIL:
img1=Image.open(filename1)
img2=Image.open(filename2)
#img1
old_size_1 = img1.size  # old_size[0] is in (width, height) format
ratio_1 = float(224)/max(old_size_1)
new_size_1 = tuple([int(x*ratio_1) for x in old_size_1])
# use resize() method to resize the input image
resized_img1 = img1.resize(new_size_1, Image.ANTIALIAS)
# create a new image and paste the resized on it
new_img1 = Image.new("RGB", (224, 224))
new_img1.paste(resized_img1, ((224-new_size_1[0])//2,
                    (224-new_size_1[1])//2))  
#img2
old_size_2 = img2.size  # old_size[0] is in (width, height) format
ratio_2 = float(224)/max(old_size_2)
new_size_2 = tuple([int(x*ratio_2) for x in old_size_2])
# use resize() method to resize the input image
resized_img2 = img2.resize(new_size_2, Image.ANTIALIAS)
# create a new image and paste the resized on it
new_img2 = Image.new("RGB", (224, 224))
new_img2.paste(resized_img2, ((224-new_size_1[0])//2,
                    (224-new_size_1[1])//2))
#save as np arrays:
new_img1=np.asarray(new_img1)
new_img2=np.asarray(new_img2) 
                   
#transpose the RGB to the front, then width and then height: 
img_trans1=new_img1.transpose((2, 0, 1))
img_trans2=new_img2.transpose((2, 0, 1))
#for each image create an extra dimension:
img1_4d=img_trans1[np.newaxis, :, :, :]
img2_4d=img_trans2[np.newaxis, :, :, :]

#stack all images: 
img_stacked = np.append(img1_4d, img2_4d, 0)
#mean of the Red channel
m0=img_stacked[:, 0].mean()
#mean of the green channel:
m1=img_stacked[:, 1].mean()
#mean of the blue channel:
m2=img_stacked[:,2].mean() 

#sd s:
s0=img_stacked[:,0].std()
s1=img_stacked[:,1].std()
s2=img_stacked[:,2].std()
#%%
#for loop:
bar = np.empty([0, 3, 3])
foo = np.random.uniform(-1, 1, [3, 3]) 
foo = foo[np.newaxis, :, :]
bar = np.append(bar, foo, 0)
print(bar, foo)
#%%
#4D_tensor =np.empty([0,3,3])

em = np.empty([0, 3, 3])
#%%


toy_training=pd.read_csv((path+"training_toy.csv"))
toy_training.head()
toy_test=pd.read_csv(path+"test_toy.csv")
toy_test.head()

#%% The for loop below now is functionnal!
#tests for the for loop:
em = np.empty([0, 3, 224, 224])
m=[]
s=[]

for index, row in toy_training.iterrows():
    
    filename=im_path+row["0"]
    im=Image.open(filename)
    #HERE USE FUNCTION TO RESIZE OR PAD THE IMAGE:
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(224)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # use resize() method to resize the input image
    resized_im = im.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (224, 224))
    new_im.paste(resized_im, ((224-new_size[0])//2,
                    (224-new_size[1])//2))
    #save image as np array:
    new_im=np.asarray(new_im)
    im_trans=new_im.transpose((2, 0, 1))#might not need this if processing downstream with pytorch.
    im_4D=im_trans[np.newaxis, :, :, :]
    em=np.append(em, im_4D, 0)
#now add the means and sds of each of the channels to the m and s variable: 
m.append(em[:, 0].mean())
m.append(em[:, 1].mean())
m.append(em[:, 2].mean())
s.append(em[:, 0].std())
s.append(em[:, 1].std())
s.append(em[:, 2].std())
    
print(em.shape, "mean is m={} and std={}".format(m, s))  
    
#%%
#Now we can do the pytorch neural network:
transform = transforms.Compose([            
                transforms.ToTensor(),                     
                transforms.Normalize(                      
                    mean=m,                
                    std=s                  
                    )])


#%%
'''
For the neural nets you don't need to extract the images into a numpy array 
(to extract the features) so the function would be as follows:
    Attempt at a function, needs quite a lot of work!
'''
#TODO: do the following for tow images to check that it works!
#TODO: seach for the biggest image (width or height and then first resize all images to that and then crop to the right size. )
#this step is for KNN or SVM models.
#reads, resizes and extracts images into a numpy array and get the corresponding labels:
def resize_extract_images_NN(path, label_df, desired_size):
    m=[]
    s=[]
    label=[]
    em = np.empty([0, desired_size, desired_size, 3])    
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
                old_size = im.size  # old_size[0] is in (width, height) format
                ratio = float(desired_size)/max(old_size)
                new_size = tuple([int(x*ratio) for x in old_size])
# use resize() method to resize the input image
                resized_im = im.resize(new_size, Image.ANTIALIAS)
# create a new image and paste the resized on it
                #TODO: make a function to rezise and pad an image!
                new_im = Image.new("RGB", (desired_size, desired_size))
                new_im.paste(resized_im, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))
#TODO here call the function for the padding + cropping:                        
                #transpose for pytorch
                im_trans=new_im.transpose(2, 0, 1)
                #add extra dimension
                im_4D=im_trans[np.newaxis, :, :, :]
                #append to the empty array
                em=np.append(em, im_4D, 0)
                
                
                
                #this should be done outside the for loop
    #mean of each each CHANNEL of your image            
    
    m.append([em[:, 0].mean(), em[:, 1].mean(), em[:, 2].mean()])
    #sd of each CHANNEL of your images
    s.append([em[:, 0].sd(), em[:, 1].sd(), em[:, 2].sd()])
    
    transform = transforms.Compose([            #[1]
                transforms.ToTensor(),                     #[4]
                transforms.Normalize(                      #[5] scaling!
                    mean=m,                #[6]
                    std=s                  #[7]
                    )])
                
#np.stack
#img[:,:,0].shape                


'''
                Form CDM:
                    
 Note that the output of the extract_images() function is a 4D array. 
 The reason is that the convolutational layer in Pytorch takes 4D arrays of shape  
 𝑁×𝐶×𝑋×𝑌  as input.
 
 OJO: you need to scale your images as well! you need to figure out the average 
 of each channel (R, G, B) overall images and  then do scaling with the 
 "transforms.normalize" parameter of the transform function in pytorch. 
 Look for examples.
 multiply by: value-mean/sd
 ===>>>> 
 
 Need to do: 
  
m1, m2, m3=mean of each each dimension of your image
s1, s2, s3=sd of each dimension of your image

transform = transforms.Compose([            #[1]
transforms.ToTensor(),                     #[4]
transforms.Normalize(                      #[5] scaling!
mean=[0.485, 0.456, 0.406],                #[6]
std=[0.229, 0.224, 0.225]                  #[7]
 )])


need to check what format the convolutional layer in Pytorch for resenet is!
     

 def extract_images(f_name):
    """ Extract the images into a 4D uint8 numpy array [index, rows, cols, 1]. """
    print('Extracting', f_name)
    with gzip.open(f_name, 'rb') as f:
        # Read file header
        buffer = f.read(16)
        magic, num_images, rows, cols = struct.unpack(">IIII", buffer)
        if magic != 2051:
            raise ValueError('Invalid magic number {0} in MNIST image file {1}.'.format(magic, f_name))

        # Read data
        buffer = f.read(rows * cols * num_images)
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num_images, 1, rows, cols)
        return data
gpy:  gaussian process latent varaible model.
'''