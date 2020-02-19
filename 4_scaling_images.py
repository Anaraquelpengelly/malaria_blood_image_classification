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
#play arround with vectorising the image: 

a = np.asarray(new_im)# a is readonly
l,b,c = a.shape
a_r = a.reshape(l*b*c,)
 
i = Image.fromarray(a)
i.show()
plt.imshow(a)
plt.show()
print(a_r.shape)

#%%
import pandas as pd
toy_training=pd.read_csv((path+"training_toy.csv"))
toy_training.head()

#%%
#This now works:
import os
#this step is for KNN or SVM models.
#reads, resizes and extracts images into a numpy array and get the corresponding labels:
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
        #dimensions are fine! 
                np_array = np.asarray(new_im)
                l,b,c = np_array.shape
                np_array = np_array.reshape(l*b*c,)
                all_images_as_array.append(np_array)
        
         

    return np.array(all_images_as_array), np.array(label)



X_train,y_train = resize_extract_images(im_path,toy_training,desired_size = 224)
print('X_train set : ',X_train)
print('y_train set : ',y_train)
print("The shape of the X_train set is: {} \n and the shape of the y_train is: {}.".format(X_train.shape, y_train.shape))


#%%
'''
For the neural nets you don't need to extract the images into a numpy array 
(to extract the features) so the function would be as follows:
'''
import os
#this step is for KNN or SVM models.
#reads, resizes and extracts images into a numpy array and get the corresponding labels:
def resize_extract_images_NN(path, label_df, desired_size):
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
                '''
                Form CDM:
                    
 Note that the output of the extract_images() function is a 4D array. 
 The reason is that the convolutational layer in Pytorch takes 4D arrays of shape  
 𝑁×𝐶×𝑋×𝑌  as input.
 ===>>>> 
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
        '''
         

    



X_train,y_train = resize_extract_images(im_path,toy_training,desired_size = 224)
print('X_train set : ',X_train)
print('y_train set : ',y_train)
print("The shape of the X_train set is: {} \n and the shape of the y_train is: {}.".format(X_train.shape, y_train.shape))

