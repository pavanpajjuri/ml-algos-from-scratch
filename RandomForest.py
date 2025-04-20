#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 23:19:26 2025

@author: pavanpaj
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from DecisionTrees import DecisionTree, ClassificationMetrics




def remove_lowly_correlated_features(X,y,threshold = 0.05):
    X_y = pd.concat([X,y], axis = 1)
    target_corr = X_y.corr()['target'].drop('target')
    low_info_cols = target_corr[target_corr.abs() < threshold].index
    X = X.drop(columns = low_info_cols)
    return X,y


class RandomForest:
    def __init__(self, n_trees = 7, max_depth = 4, min_samples = 2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees = []
        
    def fit(self, X, y):
        
        X = pd.DataFrame(X)
        y = pd.Series(y)
        dataset = pd.concat([X,y], axis = 1)
        n_samples, n_features  = X.shape[0], X.shape[1]
        
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples, max_features=int(np.sqrt(n_features)))
            dataset_sample = dataset.iloc[np.random.choice(n_samples, n_samples, replace = True)]
            X_sample, y_sample = dataset_sample.iloc[:,:-1].values, dataset_sample.iloc[:,-1].values
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def predict(self, X):
        predictions = np.array([tree.predict(X)[0] for tree in self.trees])
        predictions = predictions.T # To change the matrix to (n_samples, n_trees)
        majority_preds = np.array([np.bincount(row.astype(int)).argmax() for row in predictions])
        
        probabilities = np.array([tree.predict(X)[1] for tree in self.trees])
        pred_prob = np.mean(probabilities, axis = 0)
        
        return majority_preds, pred_prob
    
if __name__ == "__main__":
    seed = 69
    np.random.seed(seed = seed)
    
    data = load_breast_cancer()
    X = pd.DataFrame(data = data.data, columns = data.feature_names)
    y = pd.Series(data = data.target, name = 'target')
    
    X,y = remove_lowly_correlated_features(X, y)
    
    data = pd.concat([X,y], axis = 1)
    train_data = data[:int(0.8*len(data))]
    test_data = data[int(0.8*len(data)):]
    
    X_train,y_train = train_data.iloc[:,:-1].values,train_data.iloc[:,-1].values
    X_test,y_test = test_data.iloc[:,:-1].values,test_data.iloc[:,-1].values
    
    
    
    rf = RandomForest(n_trees = 7, max_depth = 4, min_samples = 2)
    rf.fit(X_train,y_train)
    
    y_pred, y_pred_prob = rf.predict(X_test)
    
    
    
    test_accuracy = ClassificationMetrics.accuracy(y_test, y_pred)
    test_precision = ClassificationMetrics.precision(y_test, y_pred)
    test_recall = ClassificationMetrics.recall(y_test, y_pred)
    test_f1_score = ClassificationMetrics.f1_score(y_test, y_pred)
    test_auc_score = ClassificationMetrics.roc_auc_rank_based(y_test, y_pred_prob)
    
    
    print(f"Final testing Accuracy Score: {test_accuracy:.4f}")
    print(f"Final testing Precision Score: {test_precision:.4f}")
    print(f"Final testing Recall Score: {test_recall:.4f}")
    print(f"Final testing f1_score Score: {test_f1_score:.4f}")
    print(f"Final testing roc_auc Score: {test_auc_score:.4f}")
    
    
    
    
    """from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier(n_estimators = 7, max_depth = 4,random_state = seed,min_samples_split=2)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_pred_prob = rf.predict_proba(X_test)[:,1]
    
    etst_accuracy = ClassificationMetrics.accuracy(y_test, y_pred)
    test_precision = ClassificationMetrics.precision(y_test, y_pred)
    test_recall = ClassificationMetrics.recall(y_test, y_pred)
    test_f1_score = ClassificationMetrics.f1_score(y_test, y_pred)
    test_auc_score = ClassificationMetrics.roc_auc_rank_based(y_test, y_pred_prob)
    
    print()
    print(f"Final testing Accuracy Score: {test_accuracy:.4f}")
    print(f"Final testing Precision Score: {test_precision:.4f}")
    print(f"Final testing Recall Score: {test_recall:.4f}")
    print(f"Final testing f1_score Score: {test_f1_score:.4f}")
    print(f"Final Training roc_auc Score: {test_auc_score:.4f}")"""
        
        