#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 20:23:41 2025

@author: pavanpaj
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
import matplotlib.pyplot as plt

seed = 69
np.random.seed(seed)


"""
data = load_breast_cancer()
X = pd.DataFrame(data = data.data, columns = data.feature_names)
y = pd.Series(data = data.target, name = 'target')
"""


# Generate synthetic dataset with 10,000 samples and 20 numerical features
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
y = pd.Series(y, name='target')


print(y.value_counts(normalize=True))


data = pd.concat([X,y], axis = 1)
train_data = data[:int(0.8*len(data))]
test_data = data[int(0.8*len(data)):]

train_data = train_data.dropna()
test_data = test_data.dropna()

def sigmoid(z):
    z = np.clip(z, -500, 500) # To prevent overflow
    return 1/(1 + np.exp(-z))

def standardize_data(X_train, X_test):
    X_train = (X_train - np.mean(X_train, axis = 0))/np.std(X_train, axis = 0)
    X_test = (X_test - np.mean(X_train, axis = 0))/np.std(X_train, axis = 0) # We donot want the bias of X_test to creep in during evaluation hence move the data according to mean and std of X_train
    
    return X_train, X_test

class LogisticRegression:
    def __init__(self, learning_rate, convergence_tol = 1e-6):
        self.learning_rate = learning_rate
        self.convergence_tol = convergence_tol
        self.W = None
        self.b = None
        
    def initialize_parameters(self, num_features):
        self.W = np.random.randn(num_features)/100
        self.b = 0
        
    def forward(self, X):
        Z = np.dot(X, self.W) + self.b
        z = sigmoid(Z)
        return z
    
    def compute_cost(self, predictions, y):
        m = len(predictions)
        cost = np.sum( (-y * np.log(predictions) ) + \
                      (-(1-y) * np.log(1-predictions))) / m
        return cost
    
    def backward(self, predictions, X, y):
        m = len(predictions)
        dW = X.T@(predictions - y)/m
        db = np.sum(predictions - y)/m
        return dW, db
    
    def fit(self, X, y, iterations):
        self.initialize_parameters(X.shape[1])
        costs = []
        for i in range(iterations):
            predictions = self.forward(X)
            costs.append(self.compute_cost(predictions, y))
            dW, db = self.backward(predictions, X, y)
            
            self.W = self.W - self.learning_rate*dW
            self.b = self.b - self.learning_rate*db
            
            if(i % 100 == 0):
                print(f"Iteration {i} Cost: {costs[-1]:.5f}")
            
            if i > 0 and abs(costs[-1] - costs[-2]) < self.convergence_tol:
                print(f"Converged after {i} iterations")
                break
            
        plt.plot(np.arange(0,len(costs)), costs, linestyle = '-', linewidth = 2)
        plt.title('Training Loss over iterations')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Binary Cross Entropy)')
        plt.show()
        
    def predict(self, X, threshold = 0.5):
        prediction_prob = self.forward(X)
        predictions = np.where(prediction_prob > threshold, 1, 0)
        return predictions, prediction_prob
  

X_train, y_train = train_data.iloc[:,:-1].values, train_data.iloc[:,-1].values
X_test, y_test = test_data.iloc[:,:-1].values, test_data.iloc[:,-1].values


X_train, X_test = standardize_data(X_train, X_test)

lg = LogisticRegression(0.01)
lg.fit(X_train,y_train, 1000)


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
           
        
        
 
    
y_train_pred, y_train_pred_prob = lg.predict(X_train)
train_accuracy = ClassificationMetrics.accuracy(y_train, y_train_pred)
train_precision = ClassificationMetrics.precision(y_train, y_train_pred)
train_recall = ClassificationMetrics.recall(y_train, y_train_pred)
train_f1_score = ClassificationMetrics.f1_score(y_train, y_train_pred)
train_auc_score = ClassificationMetrics.roc_auc_score(y_train, y_train_pred_prob, plot = True)

print(f"Final Training Accuracy Score: {train_accuracy:.4f}")
print(f"Final Training Precision Score: {train_precision:.4f}")
print(f"Final Training Recall Score: {train_recall:.4f}")
print(f"Final Training f1_score Score: {train_f1_score:.4f}")
print(f"Final Training roc_auc Score: {train_auc_score:.4f}")


#print(pd.Series(y_test).value_counts(normalize=False))
print()


y_test_pred, y_test_pred_prob = lg.predict(X_test)
test_accuracy = ClassificationMetrics.accuracy(y_test, y_test_pred)
test_precision = ClassificationMetrics.precision(y_test, y_test_pred)
test_recall = ClassificationMetrics.recall(y_test, y_test_pred)
test_f1_score = ClassificationMetrics.f1_score(y_test, y_test_pred)
test_auc_score = ClassificationMetrics.roc_auc_score(y_test, y_test_pred_prob, plot = True)

print(f"Final testing Accuracy Score: {test_accuracy:.4f}")
print(f"Final testing Precision Score: {test_precision:.4f}")
print(f"Final testing Recall Score: {test_recall:.4f}")
print(f"Final testing f1_score Score: {test_f1_score:.4f}")
print(f"Final Training roc_auc Score: {test_auc_score:.4f}")






"""
# Comparing with original sklearn's Logistic Regression Model'
from sklearn.linear_model import LogisticRegression

# Create and fit the model
sk_model = LogisticRegression(solver='lbfgs', max_iter=1000)
sk_model.fit(X_train, y_train)

# Predict probabilities for class 1
y_train_probs_sk = sk_model.predict_proba(X_train)[:, 1]
y_test_probs_sk = sk_model.predict_proba(X_test)[:, 1]

# Predict class labels using threshold = 0.5
y_train_preds_sk = (y_train_probs_sk >= 0.5).astype(int)
y_test_preds_sk = (y_test_probs_sk >= 0.5).astype(int)

# TRAIN
train_acc = ClassificationMetrics.accuracy(y_train, y_train_preds_sk)
train_prec = ClassificationMetrics.precision(y_train, y_train_preds_sk)
train_rec = ClassificationMetrics.recall(y_train, y_train_preds_sk)
train_f1 = ClassificationMetrics.f1_score(y_train, y_train_preds_sk)
train_auc = ClassificationMetrics.roc_auc_score(y_train, y_train_probs_sk)

# TEST
test_acc = ClassificationMetrics.accuracy(y_test, y_test_preds_sk)
test_prec = ClassificationMetrics.precision(y_test, y_test_preds_sk)
test_rec = ClassificationMetrics.recall(y_test, y_test_preds_sk)
test_f1 = ClassificationMetrics.f1_score(y_test, y_test_preds_sk)
test_auc = ClassificationMetrics.roc_auc_score(y_test, y_test_probs_sk)

# Print Results
print()
print("------ SKLEARN Logistic Regression ------")
print(f"Train Accuracy:  {train_acc:.4f}")
print(f"Train Precision: {train_prec:.4f}")
print(f"Train Recall:    {train_rec:.4f}")
print(f"Train F1 Score:  {train_f1:.4f}")
print(f"Train ROC AUC:   {train_auc:.4f}")
print()
print(f"Test Accuracy:   {test_acc:.4f}")
print(f"Test Precision:  {test_prec:.4f}")
print(f"Test Recall:     {test_rec:.4f}")
print(f"Test F1 Score:   {test_f1:.4f}")
print(f"Test ROC AUC:    {test_auc:.4f}")

"""


















    
    