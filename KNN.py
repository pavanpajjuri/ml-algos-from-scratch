#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 17:50:35 2025

@author: pavanpaj
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_iris
import matplotlib.pyplot as plt

seed = 69
np.random.seed(seed)

data = load_iris()
X = pd.DataFrame(data = data.data, columns = data.feature_names)
y = pd.Series(data = data.target, name = 'target')


"""# Generate synthetic dataset with 10,000 samples and 20 numerical features
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_repeated=0,
    n_classes=2,
    weights=[0.7, 0.3],  # Class imbalance like breast cancer
    random_state=seed
)

# Wrap into DataFrame/Series like your original code
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y = pd.Series(y, name='target')"""

data = pd.concat([X,y], axis = 1)
data = data.sample(frac = 1, random_state = seed).reset_index()

train_data = data[:int(0.8*len(data))]
test_data = data[int(0.8*len(data)):]

train_data = train_data.dropna()
test_data = test_data.dropna()

def standardize_data(X_train, X_test):
    mean = np.mean(X_train, axis = 0)
    std = np.std(X_train, axis = 0)
    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std
    
    return X_train, X_test

class KNN:
    def __init__(self, n_neighbours = 5):
        self.n_neighbours = n_neighbours
    
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        
    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict(x) for x in X])   
    
    def _predict(self, x):
    
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # Get indices of k closest points
        k_indices = np.argsort(distances)[:self.n_neighbours]
        k_labels = self.y_train[k_indices].tolist()
        most_common = max(k_labels, key = k_labels.count)
        return most_common
        

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


X_train, X_test = standardize_data(X_train, X_test)

knn = KNN(7)
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)

test_accuracy = ClassificationMetrics.accuracy(y_test, y_pred)
test_precision = ClassificationMetrics.precision(y_test, y_pred)
test_recall = ClassificationMetrics.recall(y_test, y_pred)
test_f1_score = ClassificationMetrics.f1_score(y_test, y_pred)

print(f"Final testing Accuracy Score: {test_accuracy:.4f}")
print(f"Final testing Precision Score: {test_precision:.4f}")
print(f"Final testing Recall Score: {test_recall:.4f}")
print(f"Final testing f1_score Score: {test_f1_score:.4f}")


# Sklearn Model KNN
from sklearn.neighbors import KNeighborsClassifier
skmodel = KNeighborsClassifier(n_neighbors=7)
skmodel.fit(X_train, y_train)

sk_pred = skmodel.predict(X_test)
sk_test_accuracy = ClassificationMetrics.accuracy(y_test, sk_pred)
sk_test_precision = ClassificationMetrics.precision(y_test, sk_pred)
sk_test_recall = ClassificationMetrics.recall(y_test, sk_pred)
sk_test_f1_score = ClassificationMetrics.f1_score(y_test, sk_pred)


print(f"Final testing Accuracy Score: {sk_test_accuracy:.4f}")
print(f"Final testing Precision Score: {sk_test_precision:.4f}")
print(f"Final testing Recall Score: {sk_test_recall:.4f}")
print(f"Final testing f1_score Score: {sk_test_f1_score:.4f}")

