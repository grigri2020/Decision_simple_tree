#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:32:37 2019

@author: Pamela Mishra
The one who will climb all 7 summits and more! 
"""


import os,sys,io
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.externals.six import StringIO 
from pydot import graph_from_dot_data
from IPython.display import Image

#REFER DECISION TREE TUTORIAL:
#https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/ml-decision-tree/tutorial/


class Decision_tree:
    def __init__(self, file_df):
        self.file_df = file_df
        
    #read file into a dataframe
    def read_file(self, file_name):
        x  = pd.read_csv(file_name, sep=",")
        return x

    #Make feature set for the decision tree, Excluding 1 column
    #All the features used to predict whether the patient will get cancer 
    #or not
    def make_features(self, data_frame, column_not_included):
        features = data_frame.loc[:, data_frame.columns != column_not_included]
        return features
    
    #Make the target to be predicted
    #iN THIS CASE: ITS WHETHER THE PATIENT HAS CANCER OR NOT
    def make_target(self, data_frame, column_included):
        target   = data_frame[column_included].astype('bool')
        return target

    #Split the data into traning and testing
    #Use random seed =1 so that this can be replicated
    def split_data_frame(self, features, target):
        features_train, features_test, target_train, target_test =\
        train_test_split(features, target, random_state=1)
        return(features_train, features_test, target_train, target_test)
        
    #Input whether the set is test or train and 
    def prediction_for_tree(self, decision_tree_obj, set_t):
        dec_tree_pred =  decision_tree_obj.predict(set_t)
        return(dec_tree_pred)
    

def main():
    input = "input/kag_risk_factors_cervical_cancer.csv"
    my_dec = Decision_tree(input)
    df = my_dec.read_file(input)
    df=df.replace('?','0')
    df_features = my_dec.make_features(df,"Dx:Cancer")
    df_target  =  my_dec.make_target(df, "Dx:Cancer")
    (features_train, features_test, target_train, target_test ) = my_dec.split_data_frame(df_features, df_target)
    
    
    #Make a decision tree cxlassifier object
    dec_tree = DecisionTreeClassifier(min_samples_split=5)
    feature_names =  features_train.columns
    dec_tree.fit(features_train,target_train)
    
    dec_tree_train_pred =  my_dec.prediction_for_tree(dec_tree, features_train)
    dec_tree_test_pred  =  my_dec.prediction_for_tree(dec_tree, features_test)
    
    #print(type(dec_tree_train_pred))
    dot_file = StringIO()
    export_graphviz(dec_tree, out_file=dot_file,feature_names=feature_names)
    (graph, ) = graph_from_dot_data(dot_file.getvalue())
    Image(graph.create_png())
    graph.write_png("Decision_Tree_cervical_entropy.png")
    
    
    print('Accuracy on training: ', accuracy_score(y_true=target_train, y_pred=dec_tree_train_pred))
    print('Accuracy on test: ', accuracy_score(y_true=target_test, y_pred=dec_tree_test_pred))

    print ('Remember if the Accuracy score for training data is 1 then it  is overfitted')
    print ('This issue can be avoided by many ways but easiest is changing min_samples_split')
#This is where it all began
if __name__ == "__main__": main()


