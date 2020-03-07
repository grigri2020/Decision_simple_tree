#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:32:37 2020

@author: mishrap1
"""


import os,sys,io
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.externals.six import StringIO 
from pydot import graph_from_dot_data
from IPython.display import Image




os.chdir("/Users/mishrap1/KAGGLE/Decision_tree_kaggle")

x  = pd.read_csv("input/kag_risk_factors_cervical_cancer.csv", sep=",")
features = x.loc[:, x.columns != 'Dx:Cancer']
features= x[['Age', 'STDs', 'IUD']]
target   = x['Dx:Cancer'].astype('bool')
x=x.replace('?','0')



#Drop the NA values if exists
features = x.loc[:, x.columns != 'Dx:Cancer']
target   = x['Dx:Cancer'].astype('bool')
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=1)
dec_tree = DecisionTreeClassifier()

feature_names =  features_train.columns
dec_tree.fit(features_train,target_train)
dot_data = StringIO()
export_graphviz(dec_tree, out_file=dot_data,feature_names=feature_names)
(graph, ) = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
#y_pred = dec_tree.predict(features_test)
#print(y_pred)
graph.write_png("Decision_Tree_cervical.png")

