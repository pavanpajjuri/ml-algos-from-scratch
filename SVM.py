#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 23:37:45 2025

@author: pavanpaj
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from DecisionTrees import ClassificationMetrics

class SVM:
    def __init__(self, kernel = 'poly', degree = 2, sigma = 0.1, epochs = 1000, learning_rate = 0.001):
        self.alpha = None
        self.b = 0
        self.degree = degree
        self.C = 1
        self.c = 1
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        if kernel == 'poly':
            self.kernel = self.polynomial_kernel
        elif kernel == 'rbf':
            self.kernel = self.gaussian_kernel
    
    
    def polynomial_kernel(self, X, Z):
        return (self.c + X.dot(Z.T))**self.degree
    
    def gaussian_kernel(self,X,Z):
        return np.exp(-(1/self.sigma**2)*np.linalg.norm(X[:,np.newaxis]-Z[np.newaxis,:],axis = 2)**2)
    
    def fit(self, X,y):
        m = X.shape[0]
        y = np.where(y==0, -1, 1)
        self.alpha = np.random.random(m)
        K = self.kernel(X,X)
        y_mul_kernel = np.outer(y,y)*K
        self.loss = []
        
        for i in range(self.epochs):
            gradient = np.ones(m) - self.alpha.dot(y_mul_kernel)
            self.alpha = self.alpha + self.learning_rate*gradient
            
            self.alpha[self.alpha > self.C] = self.C
            self.alpha[self.alpha < 0] = 0
            
            loss = np.sum(self.alpha) - 0.5*np.sum((np.outer(self.alpha, self.alpha)*y_mul_kernel))
            self.loss.append(loss)
        
        alpha_index = np.where(self.alpha > 0)[0]
        
        self.b = np.mean(y[alpha_index] - (self.alpha*y).dot(K)[alpha_index])
        self.sv_alpha = self.alpha[alpha_index]
        self.sv_y = y[alpha_index]
        self.sv_X = X[alpha_index]
        return loss
    
    def predict(self,X):
        y = (self.sv_alpha*self.sv_y).dot(self.kernel(self.sv_X,X)) + self.b
        return np.sign(y)
    
            
 

if __name__ == "__main__":
    seed = 69
    np.random.seed(seed = seed)

    data = load_breast_cancer()
    X = pd.DataFrame(data = data.data, columns = data.feature_names)
    y = pd.Series(data = data.target, name = 'target')
    #y = y.replace({0:-1, 1:1})
    
    data = pd.concat([X, y], axis = 1)
    train_data = data[:int(0.8*len(data))]
    test_data = data[int(0.8*len(data)):]
    
    X_train,y_train = train_data.iloc[:,:-1].values, train_data.iloc[:,-1].values
    X_test, y_test = test_data.iloc[:,:-1].values, test_data.iloc[:,-1].values
    
    """K = np.random.random((10,10))
    y = np.random.randint(0,2,10)
    y_mul_kernel = np.outer(y,y)*K
    alpha = np.random.rand(10)
    grad = alpha.dot(y_mul_kernel)
    alpha_index = np.where(alpha > 0.5)[0]
    support_vectors = y[alpha_index]-(alpha*y).dot(K)[alpha_index]"""
    
    svm = SVM(kernel = 'poly', degree = 1)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    y_pred = np.where(y_pred==-1, 0, 1)

    
    test_accuracy = ClassificationMetrics.accuracy(y_test, y_pred)
    test_precision = ClassificationMetrics.precision(y_test, y_pred)
    test_recall = ClassificationMetrics.recall(y_test, y_pred)
    test_f1_score = ClassificationMetrics.f1_score(y_test, y_pred)
    #test_auc_score = ClassificationMetrics.roc_auc_rank_based(y_test, y_pred_prob)
    
    print(f"Final testing Accuracy Score: {test_accuracy:.4f}")
    print(f"Final testing Precision Score: {test_precision:.4f}")
    print(f"Final testing Recall Score: {test_recall:.4f}")
    print(f"Final testing f1_score Score: {test_f1_score:.4f}")
    #print(f"Final testing roc_auc Score: {test_auc_score:.4f}")
    

     
    
    
    
    
    

