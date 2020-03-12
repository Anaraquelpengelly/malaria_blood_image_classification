#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:31:08 2020

@author: anaraquelpengelly
"""
import os
import shutil 
import pandas as pd
path="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/all_images/"
val=pd.read_csv("/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/validation_set.csv")
destination="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/vali_images/"


val["0"]=val["0"].str.replace(" ", "")
val["0"]=val["0"].str.replace("'", "")

for index, row in val.iterrows():
    for filename in os.listdir(path):
            if filename==row["0"]:
                shutil.move( path+f"{filename}", destination+f'{filename}')
                
#%% Now to remove the images form the padded folder: 

test_para="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/test_images/parazitised/"

test_unin="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/test_images/uninfected/"

train_para="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/train_images/parazitised/"
train_unin = "/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/train_images/uninfected/"               
padded="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/padded_cell_images/"
destination2="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/vali_images_padded/"
   
l=[test_para, test_unin, train_unin, train_para]          
names=["Uninfected", "Parasitized"]
for name in names:
    for index, row in val.iterrows():
        for filename in os.listdir(padded):
                if filename==row["0"]:
                    shutil.copy2(padded+name+f"/{filename}", destination2)
                    print("done!")
#not working!!!                  
 #below worked!                   
for i, folder_name in enumerate(names):
    for index, row in val.iterrows():
        for filename in os.listdir(padded+folder_name):
                if filename==row["0"]:
                    shutil.copy2(padded+folder_name+f"/{filename}", destination2)
    
print('got here')
#%%
#Check that it worked:

image_names = os.listdir(destination2)
#for i, name in enumerate(os.listdir(path))
for i, image in enumerate(image_names):
    for filename in os.listdir(test_para):
        if image_names[i]==filename:
            print("still there!")
        else:
            print("not there!")
            
#### function
            
def finding_images(path_target, path_query):
    image_names = os.listdir(path_query)
    names=[]
    for i, image in enumerate(image_names):
        for filename in os.listdir(path_target):
            if image_names[i]==filename:
                names.append(image_names[i])
    return(names)


finding_images(test_unin, destination2)
finding_images(train_para, destination2) 
finding_images(train_unin, destination2) 
finding_images(test_para, destination2)  

#all of those returned an empty list, lets make a test to see what happens:

finding_images(destination2, destination2)            
#this returns the list of images, so function works!    