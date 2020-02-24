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
from torchvision import transforms


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
#for two images:

#transpose the RGB to the front, then width and then height: 
img_trans1=img1.transpose((2, 0, 1))
img_trans2=img2.transpose((2, 0, 1))
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
s0=img_stacked[:,0].sd()
s1=img_stacked[:,1].sd()
s2=img_stacked[:,2].sd()
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

#%%
em = np.empty([0, 224, 224, 3])

#%%
for index, row in toy_training.iterrows():
    
    filename=im_path+row["0"]
    im=image_open(filename)
    #HERE USE FUNCTION TO RESIZE OR PAD THE IMAGE:
    
    im_trans=im.transpose(2, 0, 1)#might not need this if processing downstream with pytorch.
    im_4D=im_trans[np.newaxis, :, :, :]
    em=np.append(em, im_4D, 0)
    

print(em.shape)  
    
  

#%%

#read datasets:
toy_training=pd.read_csv((path+"training_toy.csv"))
toy_training.head()
toy_test=pd.read_csv(path+"test_toy.csv")
toy_test.head()

#%%
'''
For the neural nets you don't need to extract the images into a numpy array 
(to extract the features) so the function would be as follows:
'''
#this step is for KNN or SVM models.
#reads, resizes and extracts images into a numpy array and get the corresponding labels:
def resize_extract_images_NN(path, label_df, desired_size):
    means=[]
    sds=[]
    all_images_as_array=[]
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
                new_im = Image.new("RGB", (desired_size, desired_size))
                new_im.paste(resized_im, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))       
####here need something to save the new images.... maybe make a new directory?
                
                #transpose for pytorch
                im_trans=new_im.transpose(2, 0, 1)
                #add extra dimension
                im_4D=im_trans[np.newaxis, :, :, :]
                #append to the empty array
                em=np.append(em, im_4D, 0)
                
                
                
                #this should be done outside the for loop
    #mean of each each CHANNEL of your image            
    
    mean.append(m1)
    mean.append(m2)
    mean.append(m3)
   #sd of each CHANNEL of your images
    sd.append(s1)
    sd.append(s2)
    sd.append(s3)
    
    
    
    
    transform = transforms.Compose([            #[1]
                transforms.ToTensor(),                     #[4]
                transforms.Normalize(                      #[5] scaling!
                    mean=[m1, m2, m3],                #[6]
                    std=[s1, s2, s3]                  #[7]
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