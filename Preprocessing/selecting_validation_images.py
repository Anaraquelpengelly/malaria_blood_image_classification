#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:05:06 2020

@author: anaraquelpengelly
"""


import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
import os
import imageio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import seaborn as sns

path="/Users/anaraquelpengelly/Desktop/MSC_health_data_science/term_2/machine_learning/project_malaria/Malaria_blood_image_classification/"
labels_x=pd.read_csv(path+"shuffled_labels.csv")

validation=labels_x.sample(20, replace=False, random_state=30)

validation["infect_status"]
validation["0"]
validation.to_csv(path+"validation_set.csv")
