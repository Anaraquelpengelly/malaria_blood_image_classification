#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:06:16 2020

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
import joblib


# =============================================================================
# Define functions
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


#####

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



# =============================================================================
# define paths
# =============================================================================
path="/rds/general/user/arp219/home/ML_project/Malaria_blood_image_classification/sklearn_HPC/"   
#TODO: change path to the correct one!
im_path="/rds/general/user/arp219/home/ML_project/Malaria_blood_image_classification/sklearn_HPC/all_images_minus_val/"
fig_path="/rds/general/user/arp219/home/ML_project/Malaria_blood_image_classification/sklearn_HPC/results/"

# =============================================================================
# Script
# =============================================================================
#1- Read images:

data=pd.read_csv(path+"labels.csv")

X,y=norm_pad_crop_extract_images(im_path, data, 394, 224)

X_train, X_test,y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

#2- PCA and transform the data:

n_components = 160

pca = PCA(n_components=n_components, svd_solver='randomized',).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

#3- Train models:

##SVM with gaussian kernel 


SVM_model=svm.SVC(kernel="rbf")
SVM_model.fit(X_train, y_train)
y_pred_SVM=SVM_model.predict(X_test)

##SVM on PCA transformed:
SVM_PCA=svm.SVC(kernel="rbf")
SVM_PCA.fit(X_train_pca, y_train)
y_pred_PCA=SVM_PCA.predict(X_test_pca)

##SVM with grid search:
param_grid={"C":[0.5, 5, 100, 1000, 5000],"gamma":[0.01, 0.1, 0.3, 0.5]}
grid =GridSearchCV(svm.SVC(kernel="rbf", class_weight="balanced"), param_grid, n_jobs=-1)
grid=grid.fit(X_train, y_train)
model =grid.best_estimator_
y_pred_SVM_grid=model.predict(X_test)

##SVM with grid search and PCA:
grid_PCA=grid.fit(X_train_pca, y_train)
model_PCA=grid_PCA.best_estimator_
y_pred_SVM_grid_PCA=model_PCA.predict(X_test_pca)

#3-Evaluate models:

conf_mat_SVM_gauss=metrics.confusion_matrix(y_test, y_pred_SVM)
conf_mat_SVM_PCA=metrics.confusion_matrix(y_test, y_pred_PCA)
conf_mat_SVM_grid=metrics.confusion_matrix(y_test, y_pred_SVM_grid) 
conf_mat_SVM_grid_PCA=metrics.confusion_matrix(y_test, y_pred_SVM_grid_PCA)

##make plot:
fig, (ax1,ax2, ax3, ax4)=plt.subplots(1, 4, sharex=True, sharey=True)
g1=sns.heatmap(conf_mat_SVM_gauss, ax=ax1, annot=True, linewidths=1,linecolor="grey", fmt="g",cbar=None)
g1.set(ylim=(2,-0.5),ylabel="True values" , title= " SVM gauss ")
g1.set_yticklabels(labels=g1.get_yticklabels(), rotation=0)
g2=sns.heatmap(conf_mat_SVM_PCA, ax=ax2, annot=True,linewidths=1, linecolor="grey",fmt="g", cbar=None)
g2.set(ylim=(2,-0.5), title= " SVM gauss\nPCA")
g3=sns.heatmap(conf_mat_SVM_grid, ax=ax3, annot=True, linewidths=1,linecolor="grey", fmt="g", cbar=None)
g3.set(ylim=(2,-0.5),  xlabel="Predicted values", title= "SVM gauss\ngrid")
g4=sns.heatmap(conf_mat_SVM_grid_PCA, ax=ax4, annot=True, linewidths=1,linecolor="grey", fmt="g")
g4.set(ylim=(2,-0.5), title= "SVM gauss\ngrid PCA")
fig.tight_layout()
plt.savefig((fig_path+"conf_matrix_sk_gauss_all.png"), dpi=300)


##save accuracy results: 
SVM_gauss_report=metrics.classification_report(y_test, y_pred_SVM, output_dict=True)
SVM_gauss_report=df = pd.DataFrame(SVM_gauss_report).transpose()
SVM_gauss_report.to_csv(fig_path+"SVM_gauss.csv")

SVM_gauss_PCA=metrics.classification_report(y_test, y_pred_PCA, output_dict=True)
SVM_gauss_PCA=pd.DataFrame(SVM_gauss_PCA).transpose()
SVM_gauss_PCA.to_csv(fig_path+"SVM_gauss_PCA.csv")

SVM_gauss_grid=metrics.classification_report(y_test, y_pred_SVM_grid, output_dict=True)
SVM_gauss_grid=pd.DataFrame(SVM_gauss_grid).transpose()
SVM_gauss_grid.to_csv(fig_path+"SVM_gauss_grid.csv")

SVM_gauss_grid_PCA=metrics.classification_report(y_test, y_pred_SVM_grid_PCA, output_dict=True)
SVM_gauss_grid_PCA=pd.DataFrame(SVM_gauss_grid_PCA).transpose()
SVM_gauss_grid_PCA.to_csv(fig_path+"SVM_gauss_grdi_PCA.csv")

##save the models: 

filename = fig_path+'SVM_model.joblib'
joblib.dump(SVM_model, filename)

filename_2 = fig_path+'SVM_PCA_model.joblib'
joblib.dump(SVM_PCA, filename_2)

filename_3 = fig_path+'grid_PCA_model.joblib'
joblib.dump(model_PCA, filename_3)

filename_4 = fig_path+"grid_model.joblib"
joblib.dump(model, filename_4)
