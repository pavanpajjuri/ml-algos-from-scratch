#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 02:11:17 2025

@author: pavanpaj
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from DecisionTrees import DecisionTree, ClassificationMetrics



def remove_lowly_correlated_features(X,y, threshold = 0.05):
    X_y = pd.concat([X,y], axis = 1)
    target_corr = X_y.corr()['target'].drop('target')
    low_info_cols = target_corr[target_corr.abs() < threshold].index
    X = X.drop(columns = low_info_cols)
    return X,y



class AdaBoost:
    def __init__(self, n_trees = 10, max_depth = 1, min_samples = 2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.alphas = []
        self.trees = []
    
    def fit(self, X, y):
        y = np.where(y == 0, -1, 1)  # convert labels
        n_samples, n_features = X.shape
        w = np.ones(n_samples)/n_samples
        
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples, max_features=int(np.sqrt(n_features)))
            tree.fit(X,y,sample_weights = w)
            y_pred, y_pred_prob = tree.predict(X)
            err = np.sum(w*(y_pred != y))/np.sum(w)
            alpha = 0.5*np.log((1-err)/(err + 1e-10))
            self.alphas.append(alpha)
            self.trees.append(tree)
            w = w * np.exp(-alpha*y*y_pred)
            w = w/np.sum(w)
            
    def predict(self, X):
        preds = np.zeros(X.shape[0])
        weighted_probs = np.zeros(X.shape[0])
        total_weight = 0
        for tree, alpha in zip(self.trees, self.alphas):
            pred, pred_prob = tree.predict(X)
            preds += alpha * pred
            weighted_probs += alpha*pred_prob
            total_weight += alpha
        return np.sign(preds).astype(int), weighted_probs/total_weight

if __name__=="__main__":
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
    
    adb = AdaBoost(n_trees = 10, max_depth = 1, min_samples = 2)
    adb.fit(X_train,y_train)
    
    y_pred,y_pred_prob = adb.predict(X_test)
    y_pred = np.where(y_pred == -1, 0, 1)
    
    
    
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
    
    
    from sklearn.ensemble import AdaBoostClassifier
    
    adb = AdaBoostClassifier(n_estimators = 10,random_state=seed)
    adb.fit(X_train, y_train)
    
    y_pred = adb.predict(X_test)
    y_pred_prob = adb.predict_proba(X_test)[:,1]
    
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
    print(f"Final Training roc_auc Score: {test_auc_score:.4f}")
