#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 19:27:49 2025

@author: pavanpaj
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

def standardize_data(X):
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    X = (X-mean)/std
    return X

def remove_lowly_correlated_features(X,y, threshold = 0.05):
    df = pd.concat([X,y], axis = 1)
    df_corr = df.corr()['target'].drop('target')
    low_info_cols = df_corr[df_corr.abs()<threshold].index
    X.drop(columns = low_info_cols, inplace = True)
    return X

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        
    def fit(self, X):
        
        """cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:,::-1]
        self.components = eigenvectors[:,:self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance/np.sum(eigenvalues)"""
        
        
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        eigenvalues = S**2/(X.shape[0] - 1)
        self.components = Vt[:self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance/np.sum(eigenvalues)
        
        return np.dot(X, self.components.T)
        
    
    def transform(self,X):
        return np.dot(X, self.components)
    
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

    
if __name__ == '__main__':    
    seed = 69
    np.random.seed(seed = seed)
    
    data = load_breast_cancer()
    X = pd.DataFrame(data = data.data, columns=data.feature_names)
    y = pd.Series(data = data.target, name = 'target')
    
    data = pd.concat([X,y], axis = 1)
    X = standardize_data(X)
    X = remove_lowly_correlated_features(X,y)
    
    pca = PCA(n_components=2)
    X_transformed = pca.fit(X)
    
    plt.figure('Reduced Dimensions')
    plt.scatter(X_transformed[:,0], X_transformed[:,1])
    plt.show()
    
    print(pca.explained_variance_ratio_)

    
    
    
    
    
    
    
    
