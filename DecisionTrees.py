#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 16:08:13 2025

@author: pavanpaj
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer
import matplotlib.pyplot as plt

seed = 69
np.random.seed(seed = seed)

data = load_breast_cancer()
X = pd.DataFrame(data = data.data, columns = data.feature_names)
y = pd.Series(data = data.target, name = 'target')

def remove_highly_correlated_features(X,y, low_threshold = 0.05):
    X_y = pd.concat([X,y], axis = 1)
    target_corr = X_y.corr()['target'].drop('target')
    low_info_cols = target_corr[target_corr.abs() < low_threshold].index
    X = X.drop(columns = low_info_cols)
    return X,y
    
X,y = remove_highly_correlated_features(X,y)


data = pd.concat([X,y], axis = 1)
train_data = data[:int(0.8*len(data))]
test_data = data[int(0.8*len(data)):]


class Node():
    def __init__(self, feature = None, threshold = None, left = None, right = None, gain = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value

class DecisionTree():
    def __init__(self,min_samples = 2, max_depth = 2):
        self.min_samples = min_samples
        self.max_depth = max_depth
                
    def entropy(self, y):
        entropy = 0
        labels = np.unique(y)
        for label in labels:
            yi = len(y[y==label])/len(y)
            entropy += (-yi*np.log2(yi)) + (-(1-yi)*np.log2(1-yi))
        return entropy
    
    def information_gain(self, parent, left, right):
        parent_entropy = self.entropy(parent)
        left_weight = len(left)/len(parent)
        right_weight = len(right)/len(parent)
        child_entropy = left_weight*self.entropy(left) + right_weight*self.entropy(right)
        gain = parent_entropy - child_entropy
        return gain
    
    def best_split(self, dataset):
        X,y = dataset[:,:-1],dataset[:,-1]
        num_features = X.shape[1]
        best_split = {
            'gain':0,
            'feature':None,
            'threshold':None,
            'left_dataset':None,
            'right_dataset':None
            }
        for feature_idx in range(num_features):
            thresholds = np.unique(X[:,feature_idx]) 
            for threshold in thresholds:
                left_dataset = dataset[dataset[:,feature_idx]<=threshold]
                right_dataset = dataset[dataset[:,feature_idx]>threshold]
                if len(left_dataset) and len(right_dataset):
                    gain = self.information_gain(y,left_dataset[:,-1],right_dataset[:,-1])
                    if gain > best_split['gain']:
                        best_split['gain'] = gain
                        best_split['feature'] = feature_idx
                        best_split['threshold'] = threshold
                        best_split['left_dataset'] = left_dataset
                        best_split['right_dataset'] = right_dataset        
        return best_split
                
        
    def build_tree(self, dataset, depth = 0):
        X,y = dataset[:,:-1],dataset[:,-1]
        samples = len(X)
        if samples > self.min_samples and depth <= self.max_depth:
            best_split = self.best_split(dataset)     
            if best_split['gain']:
                left_node = self.build_tree(best_split['left_dataset'], depth + 1)
                right_node = self.build_tree(best_split['right_dataset'], depth + 1)
                return Node(best_split['feature'], best_split['threshold'], left_node, right_node, best_split['gain'])
        y = list(y)
        value = max(y, key = y.count)
        return Node(value)
        
        
    def fit(self, X, y):
        dataset = np.concatenate((X,y.reshape(-1,1)), axis = 1)
        self.root = self.build_tree(dataset)
            

X_train, y_train = train_data.iloc[:,:-1].values, train_data.iloc[:,-1].values
X_test, y_test = test_data.iloc[:,:-1].values, test_data.iloc[:,-1].values


dt = DecisionTree()
dt.fit(X_train,y_train)
    













































