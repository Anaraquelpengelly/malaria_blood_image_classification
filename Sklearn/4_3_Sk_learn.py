#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 21:05:21 2020

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
import logging
import joblib
#%%
import sys
sys.path.append(r"/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/scripts/image_anal")
import sklearn_ana as sk

#%% Test for the function with normalising: 

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
                cropped_im = sk.pad_crop_image(desired_size, max_size, im)
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

path="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/"
im_path="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/all_images/"

toy_test=pd.read_csv(path+"test_toy.csv")

X_x, y_y=norm_pad_crop_extract_images(im_path, toy_test)


#%%
#1- read images

path="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/"
im_path="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/all_images/"

data=pd.read_csv(path+"labels.csv")

data=pd.read_csv(path+"toy_df.csv")
import time as time
start_time=time.time()
X,y=sk.norm_pad_crop_extract_images(im_path, data, 394, 224)

print(f"It took {time.time()-start_time}s to open 200 images")
#It took 617.9523899555206s for 200 images
#so 10 min!!!
X_train, X_test,y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
#%%
#1.1 Dimensionality reduction with PCA
##get how many components explain 50% of the variance: 
pca_test = PCA(0.50).fit(X)
print(pca_test.n_components_)
#6!!!
##get plot to determine how many compnents explain how much of the variance:

pca_test2 = PCA().fit(X)
plt.plot(np.cumsum(pca_test2.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.savefig((path+"figures/PCA_scree_plot.png"), dpi=300)




#Do PCA and transform your data
n_components = 160

t0 = time.time()
pca = PCA(n_components=n_components, svd_solver='randomized',).fit(X_train)
print(f"done in {(time.time() - t0)}s" )


print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time.time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time.time() - t0))
#%%
#2-Train models:
##K nearest neighbors
KNN_model=neighbors.KNeighborsClassifier(3)
KNN_model.fit(X_train, y_train)
y_pred_KNN=KNN_model.predict(X_test)


##SVM with gaussian kernel 


SVM_model=svm.SVC(kernel="rbf")
SVM_model.fit(X_train, y_train)
y_pred_SVM=SVM_model.predict(X_test)

#SVM on PCA transformed:
SVM_PCA=svm.SVC(kernel="rbf")
SVM_PCA.fit(X_train_pca, y_train)
y_pred_PCA=SVM_PCA.predict(X_test_pca)


##SVM with grid search
t0=time.time()
param_grid={"C":[0.5, 5, 100, 1000, 5000],"gamma":[0.01, 0.1, 0.3, 0.5]}
grid =GridSearchCV(svm.SVC(kernel="rbf", class_weight="balanced"), param_grid, n_jobs=-1)
grid=grid.fit(X_train, y_train)
model =grid.best_estimator_
y_pred_SVM_grid=model.predict(X_test)
print(f"time it took ={time.time()-t0}")
#time it took =275.04812693595886
print(grid.best_params_)
#{'C': 1000, 'gamma': 0.01}
grid_PCA=grid.fit(X_train_pca, y_train)
model_PCA=grid_PCA.best_estimator_
y_pred_SVM_grid_PCA=model_PCA.predict(X_test_pca)

# =============================================================================
# Grid search:
# =============================================================================
# param_grid={"C":[0.5, 5, 100, 1000, 5000],"gamma":[0.01, 0.1, 0.3, 0.5]}
# grid_poly=GridSearchCV(svm.SVC(kernel="poly", class_weight="balanced"), param_grid, n_jobs=-1 )
# grid_poly.fit(X_train, y_train)
# model_poly=grid_poly.best_estimator_
# y_pred_poly_grid=model_poly.predict(X_test)
# grid =GridSearchCV(svm.SVC(kernel="rbf", class_weight="balanced"), param_grid, n_jobs=-1)
# grid=grid.fit(X_train, y_train)
# model =grid.best_estimator_
# y_pred_SVM_grid=model.predict(X_test)
# print(grid.best_params_)
#{'C': 1000, 'gamma': 0.01}
# grid_PCA=grid.fit(X_train_pca, y_train)
# model_PCA=grid_PCA.best_estimator_
# y_pred_SVM_grid_PCA=model_PCA.predict(X_test_pca)

#randomised search from sklearn ! you can make it run through all Cs and all gammas! 
#C belongs to the 


#%%
#3-Evaluate models:
##confusion matrix: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
## example: 

conf_mat_KNN=metrics.confusion_matrix(y_test, y_pred_KNN)
conf_mat_SVM_gauss=metrics.confusion_matrix(y_test, y_pred_SVM)
conf_mat_SVM_PCA=metrics.confusion_matrix(y_test, y_pred_PCA)
conf_mat_SVM_grid=metrics.confusion_matrix(y_test, y_pred_SVM_grid) 
conf_mat_SVM_grid_PCA=metrics.confusion_matrix(y_test, y_pred_SVM_grid_PCA)

#%%
#make function!
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
plt.savefig((path+"figures/conf_matrix_sk_gauss_norm_test.png"), dpi=300)
plt.show()



#%%
##ROC_curve: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
##accuracy_score:
# acc_KNN=metrics.accuracy_score(y_test, y_pred_KNN)
# acc_SVM=metrics.accuracy_score(y_test, y_pred_SVM)
# acc_SVM_grid=metrics.accuracy_score(y_test, y_pred_SVM_grid)

SVM_gauss_report=metrics.classification_report(y_test, y_pred_SVM, output_dict=True)
SVM_gauss_report=df = pd.DataFrame(SVM_gauss_report).transpose()
SVM_gauss_report.to_csv(path+"figures/"+"SVM_gauss.csv")

SVM_gauss_PCA=metrics.classification_report(y_test, y_pred_PCA, output_dict=True)
SVM_gauss_PCA=pd.DataFrame(SVM_gauss_PCA).transpose()
SVM_gauss_PCA.to_csv(path+"figures/"+"SVM_gauss_PCA.csv")

SVM_gauss_grid=metrics.classification_report(y_test, y_pred_SVM_grid, output_dict=True)
SVM_gauss_grid=pd.DataFrame(SVM_gauss_grid).transpose()
SVM_gauss_grid.to_csv(path+"figures/"+"SVM_gauss_grid.csv")

SVM_gauss_grid_PCA=metrics.classification_report(y_test, y_pred_SVM_grid_PCA, output_dict=True)
SVM_gauss_grid_PCA=pd.DataFrame(SVM_gauss_grid_PCA).transpose()
SVM_gauss_grid_PCA.to_csv(path+"figures/"+"SVM_gauss_grdi_PCA.csv")



#%%
#save models:
filename = path+'finalized_model.sav'
joblib.dump(SVM_model, filename)
 
# some time later...
 
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)
 

