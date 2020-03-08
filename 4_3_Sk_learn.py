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
from image_anal import sklearn_ana as sk
import sklearn
from sklearn import neighbors, ensemble, svm, metrics 

#%%
#1- read images

path="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/"
im_path="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/all_images/"

data=pd.read_csv(path+"")
#train test split from sklearn
train=
test=

#Could do the following OR 
#TODO: could do it on the whole (toy) data and then use test_train_split from sklearn.
X_train,y_train=sk.pad_crop_extract_images(path, train, 394, 224)
X_test, y_test=sk.pad_crop_extract_images(path, test, 394, 224)
#%%
#2-Train models:
##K nearest neighbors
KNN_model=neighbors.KNeighborsClassifier(3)
KNN_model.fit(X_train, y_train)
y_pred_KNN=KNN_model.predict(X_test)


##SVM with gaussian kernel 
#TODO:cross validation with grid search!!!Do it!
SVM_model=svm.SVC(kernel="rbf")
SVM_model.fit(X_train, y_train)
y_pred_SVM=SVM_model.predict(X_test)

#%%
#3-Evaluate models:
##confusion matrix: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
## example: 
conf_met_NN=metrics.confusion_matrix(y_test, y_pred)
ax= plt.figure()
ax=sns.heatmap(conf_met_NN, annot=True, fmt="g")
ax.set(ylim=(2,-0.5), xlabel="Predicted values", ylabel="True values", title="Confusion matrix for the K-nearest neighbour classifier with 3 neighbours")
ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=0)
plt.show()
##ROC_curve: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
##accuracy_score:


#%%
#Do clustering with the images to see how it looks!



 

