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
    
    def __repr__(self):
        return f"Node(feature = {self.feature}, threshold = {self.threshold}, value = {self.value})"

class DecisionTree():
    def __init__(self,min_samples = 2, max_depth = 2):
        self.min_samples = min_samples
        self.max_depth = max_depth
                
    def entropy(self, y):
        entropy = 0
        labels, counts = np.unique(y, return_counts=True)
        probabilities = counts/len(y)
        for p in probabilities:
            if p > 0:
                entropy += (-p*np.log2(p))
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
                return Node(
                            feature=best_split['feature'],
                            threshold=best_split['threshold'],
                            left=left_node,
                            right=right_node,
                            gain=best_split['gain'],
                            value=None
                            )
        y = list(y)
        value = max(y, key = y.count)
        return Node(
                feature=None,
                threshold=None,
                left=None,
                right=None,
                gain=None,
                value=value
            )
        
        
    def fit(self, X, y):
        dataset = np.concatenate((X,y.reshape(-1,1)), axis = 1)
        self.root = self.build_tree(dataset)
        
    def predict(self, X):
        return np.array([self.predict_single(x, self.root) for x in X])
    
    def predict_single(self, x, Node):
        if Node.value is not None:
            return Node.value
        else:
            feature = x[int(Node.feature)]
            if feature <= Node.threshold:
                return self.predict_single(x, Node.left)
            if feature > Node.threshold:
                return self.predict_single(x, Node.right)
            
            

class ClassificationMetrics:
    @staticmethod 
    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred)/len(y_true)
    
    @staticmethod  
    def precision(y_true, y_pred):
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        if (true_positives + false_positives) == 0:
            return 0
        return np.divide(true_positives, (true_positives + false_positives))
    
    @staticmethod 
    def recall(y_true, y_pred):
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        if (true_positives + false_negatives) == 0: 
            return 0
        return np.divide(true_positives, (true_positives + false_negatives))
    
    @staticmethod 
    def f1_score(y_true, y_pred):
        precision = ClassificationMetrics.precision(y_true, y_pred)
        recall = ClassificationMetrics.recall(y_true, y_pred)
        if (precision + recall) == 0:
            return 0
        return np.divide(2*precision*recall, (precision + recall))
    
    @staticmethod 
    def perf_metrics(y_true, y_pred_prob, threshold = 0.5):
        tp = fp = tn = fn = 0
        y_pred = np.where(y_pred_prob > threshold, 1, 0)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = tp/(tp + fn) if (tp + fn) > 0 else 0 # Sensitivity
        fpr = fp/(fp + tn) if (fp + tn) > 0 else 0 # 1- Specificity
        return fpr, tpr
    
    @staticmethod 
    def roc_auc_score(y_true, y_pred_prob, plot = False):
       thresholds = np.linspace(0, 1, 100)
       fpr_list, tpr_list = [], []
       
       for threshold in thresholds:
           fpr, tpr = ClassificationMetrics.perf_metrics(y_true, y_pred_prob, threshold = threshold)
           fpr_list.append(fpr)
           tpr_list.append(tpr)
           
       sorted_pairs = sorted(zip(fpr_list, tpr_list)) # Sort by Fprlist
       fpr_sorted, tpr_sorted = zip(*sorted_pairs)
       auc = np.trapz(tpr_sorted, fpr_sorted)
       
       if plot:
           plt.plot(fpr_sorted, tpr_sorted, 'r-', lw=2)
           plt.plot([0, 1], [0, 1], 'k--', lw=1)
           plt.xlabel("False Positive Rate")
           plt.ylabel("True Positive Rate")
           plt.title(f"ROC Curve (AUC = {auc:.3f})")
           plt.grid(True)
           plt.show()
           
       return auc

X_train, y_train = train_data.iloc[:,:-1].values, train_data.iloc[:,-1].values
X_test, y_test = test_data.iloc[:,:-1].values, test_data.iloc[:,-1].values


dt = DecisionTree()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
    
test_accuracy = ClassificationMetrics.accuracy(y_test, y_pred)
test_precision = ClassificationMetrics.precision(y_test, y_pred)
test_recall = ClassificationMetrics.recall(y_test, y_pred)
test_f1_score = ClassificationMetrics.f1_score(y_test, y_pred)
#test_auc_score = ClassificationMetrics.roc_auc_score(y_test, y_pred_prob, plot = True)


print(f"Final testing Accuracy Score: {test_accuracy:.4f}")
print(f"Final testing Precision Score: {test_precision:.4f}")
print(f"Final testing Recall Score: {test_recall:.4f}")
print(f"Final testing f1_score Score: {test_f1_score:.4f}")
#print(f"Final Training roc_auc Score: {test_auc_score:.4f}")












































