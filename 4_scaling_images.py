#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:31:13 2020

@author: anaraquelpengelly
"""

'''
As we saw in our image display, the images are different sizes, so we need to 
scale them. For this purpose I will use the toy dataset I made in 3. 
From the paper:
The images were re-sampled to 100 × 100, 224 × 224, 227 × 227 and 
299 × 299 pixel resolutions to suit the input requirements of customized 
and pre-trained CNNs and normalized to assist in faster convergence.

'''
#%%
#import necesary packages: 


path="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/"
im_path="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/cell_images/all/"


#make function to extract images and resample them to wanted pixel size:
##here, first do it with one image to see what the function does!

#First extract the image
import numpy as np
import matplotlib.pyplot as plt

import os
filename = os.path.join(im_path,'C101P62ThinF_IMG_20150918_151006_cell_61.png')
'''
will not use this but is useful to know:
from skimage import io
img = io.imread(filename)
'''
#%%
#different strategy: 
#desired size: 224 × 224

from PIL import Image, ImageOps

desired_size = 224
import os
filename = os.path.join(im_path,'C101P62ThinF_IMG_20150918_151006_cell_61.png')

im = Image.open(filename)
old_size = im.size  # old_size[0] is in (width, height) format

ratio = float(desired_size)/max(old_size)
new_size = tuple([int(x*ratio) for x in old_size])
# use thumbnail() or resize() method to resize the input image

# thumbnail is a in-place operation

# im.thumbnail(new_size, Image.ANTIALIAS)

resized_im = im.resize(new_size, Image.ANTIALIAS)
# create a new image and paste the resized on it




new_im = Image.new("RGB", (desired_size, desired_size))
new_im.paste(resized_im, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))




#this pops the image open! 
new_im.show()
#this does a pretty good job! Now lets try with the Image Ops:
delta_w=desired_size - old_size[0]
delta_h=desired_size - old_size[1]

padding= (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
new_im2=ImageOps.expand(im, padding)
new_im2.show()
#plot the three images as before to compare: 
fig, axes=plt.subplots(1, 3)
ax=axes.flatten()
ax[0].imshow(im)
ax[0].set_title("Original image. \n Size:{}.".format(im.size))
ax[1].imshow(new_im)
ax[1].set_title("Resized image. \n Size:{}.".format(new_im.size))
ax[2].imshow(new_im2)
ax[2].set_title("Padded image. \n Size:{}.".format(new_im2.size))
plt.tight_layout()
plt.savefig((path+"figures/image_resizing_options.png"), dpi=300)
plt.show()
#Now do a for loop to get all the images of the toy (in my computer) and eventually 
#of the real dataset (on the HPC) to the right size. 
#%%
import pandas as pd
toy_training=pd.read_csv((path+"training_toy.csv"))
toy_training.head()


#%%
import os
#this step is for KNN or SVM models.
#reads, resizes and extracts images into a numpy array and get the corresponding labels:
def resize_extract_images(path, label_df, desired_size):
    all_images_as_array=[]
    label=[]
    #TODO:here create if statement!id filename in df...might be better below..
    for filename in os.listdir(path):
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
#create np_array from new image:  
        #TODO: here there is a problem with the dimensions of the vectorised image find it! 
        np_array = np.asarray(new_im)
        l,b,c = np_array.shape
        np_array = np_array.reshape(l*b*c,)
        all_images_as_array.append(np_array)
        for index, row in label_df.iterrows():
            if filename==row["0"]:
                if row["infect_status"]==1:
                    label.append(1)
                else:
                    label.append(0)
         

    return np.array(all_images_as_array), np.array(label)



X_train,y_train = resize_extract_images(im_path,toy_training,desired_size = 224)
print('X_train set : ',X_train)
print('y_train set : ',y_train)
print("The shape of the X_train set is: {} \n and the shape of the y_train is: {}.".format(X_train.shape, y_train.shape))

#TODO: thisX is not the right dimensions, it contains all the images from the "all" folder.
#might need to make a new folder with the toy dataset. Or put the looping through the labels before reading the images in the function. 
'''
Do I now need to save the images? or can I just proceed to feeding them to 
the transforming into a vector( for the k nearest neighbours and for the 
random forests)?
'''

#%%
'''
For the neural nets you don't need to extract the images into a numpy array 
(to extract the features).
'''
 