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
plt.show()
plt.savefig((path+"figures/image_resizing_options.png"), dpi=300)
'''
Do I now need to save the images? or can I just proceed to feeding them to 
the transforming into a vector( for the k nearest neighbours and for the 
random forests)?
'''
#%%