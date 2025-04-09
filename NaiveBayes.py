#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 18:52:11 2025

@author: pavanpaj
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer
import matplotlib.pyplot as plt

seed = 69
np.random.seed(seed)

data = load_breast_cancer()
X = pd.DataFrame(data = data.data, columns = data.feature_names)
y = pd.Series(data = data.target, name = 'target')


def remove_highly_correlated_features(df, threshold=0.9):
    # Dropping features that are highly correlated to match conditional independence of features according to Naive Bayes
    corr_matrix = df.corr().abs()

    # Upper triangle mask to avoid duplicates
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identify columns to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    return df.drop(columns=to_drop), to_drop

# Apply to feature columns only
X, dropped_columns = remove_highly_correlated_features(X, threshold=0.8)

data = pd.concat([X,y], axis = 1)
data = data.sample(frac = 1, random_state = seed).reset_index()

train_data = data[:int(0.8*len(data))]
test_data = data[int(0.8*len(data)):]

def standardize_data(X_train, X_test):
    mean = np.mean(X_train, axis = 0)
    std = np.std(X_train, axis = 0)
    
    X_train = (X_train-mean)/std
    X_test = (X_test-mean)/std
    
    return X_train, X_test

class NaiveBayes:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.m, self.n = X.shape
        self.unique_classes = np.unique(y)
        self.n_unique = len(self.unique_classes)
        
        # Arrays for mean, variance and priors for each feature per each class
        self.mean = np.zeros((self.n_unique, self.n))
        self.variance = np.zeros((self.n_unique, self.n))
        self.prior = np.zeros(self.n_unique)
        
        for i,c in enumerate(self.unique_classes):
            X_c = X[y==c]
            self.mean[i] = np.mean(X_c, axis = 0)
            self.variance[i] = np.var(X_c, axis = 0)
            self.prior[i] = X_c.shape[0]/self.m 
        
    def GaussianNB(self,X,y):
        mean = self.mean[y]
        variance = self.variance[y] +1e-9
        const = -0.5*np.log(2*np.pi*variance)
        prob = -((X-mean)**2)/(2*variance)
        return const+prob        
        
    def predict(self, X):
        posteriors = []
        for i,y in enumerate(self.unique_classes):
            log_prior = np.log(self.prior[i])
            log_likelihood = np.sum(self.GaussianNB(X,i), axis = 1)
            posterior = log_prior + log_likelihood
            posteriors.append(posterior)
        
        posteriors = np.array(posteriors)
        predictions = self.unique_classes[np.argmax(posteriors, axis = 0)]
        
        log_probs = posteriors.T
        max_log = np.max(log_probs, axis = 1, keepdims= True)
        probs = np.exp(log_probs - max_log)
        probs = probs/np.sum(probs, axis = 1, keepdims = True)
        positive_class_prob = probs[:, np.where(self.unique_classes == 1)[0][0]]
        
        return predictions, positive_class_prob
    
    
    

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


nb = NaiveBayes()
nb.fit(X_train,y_train)

y_pred,y_pred_prob = nb.predict(X_test)


test_accuracy = ClassificationMetrics.accuracy(y_test, y_pred)
test_precision = ClassificationMetrics.precision(y_test, y_pred)
test_recall = ClassificationMetrics.recall(y_test, y_pred)
test_f1_score = ClassificationMetrics.f1_score(y_test, y_pred)
test_auc_score = ClassificationMetrics.roc_auc_score(y_test, y_pred_prob, plot = True)


print(f"Final testing Accuracy Score: {test_accuracy:.4f}")
print(f"Final testing Precision Score: {test_precision:.4f}")
print(f"Final testing Recall Score: {test_recall:.4f}")
print(f"Final testing f1_score Score: {test_f1_score:.4f}")
print(f"Final Training roc_auc Score: {test_auc_score:.4f}")



"""
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:,1]


test_accuracy = ClassificationMetrics.accuracy(y_test, y_pred)
test_precision = ClassificationMetrics.precision(y_test, y_pred)
test_recall = ClassificationMetrics.recall(y_test, y_pred)
test_f1_score = ClassificationMetrics.f1_score(y_test, y_pred)
test_auc_score = ClassificationMetrics.roc_auc_score(y_test, y_pred_prob, plot = True)

print()
print(f"Final testing Accuracy Score: {test_accuracy:.4f}")
print(f"Final testing Precision Score: {test_precision:.4f}")
print(f"Final testing Recall Score: {test_recall:.4f}")
print(f"Final testing f1_score Score: {test_f1_score:.4f}")
print(f"Final testing roc_auc Score: {test_auc_score:.4f}")
"""