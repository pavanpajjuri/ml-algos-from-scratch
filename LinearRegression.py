# %% 
import numpy as np
import math
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

np.random.seed(69)

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns = data.feature_names)[:5000]
y = pd.Series(data.target, name = 'target')[:5000]

data = pd.concat([X,y],axis=1)
train_data = data[:4000]
test_data = data[4000:]

# Dropping any null values
train_data = train_data.dropna()
test_data = test_data.dropna()


train_data.head()

# Data Preprocessing

def standardize_data(X_train, X_test):
    # Standardize (Mean 0 and std dev 1) the values as Linear regression is susceptible to outlier. Also faster convergence.
    # Also called z-score normalization

    mean = np.mean(X_train, axis = 0)
    std = np.std(X_train, axis = 0)
    
    X_train = (X_train-mean)/std
    X_test = (X_test-mean)/std
    
    return X_train, X_test

# Model Implementation

class LinearRegression:
    def __init__(self, learning_rate, convergence_tol = 1e-6):
        self.learning_rate = learning_rate
        self.convergence_tol = convergence_tol
        self.W = None
        self.b = None
        
    def initialize_parameters(self, n_features):
        self.W = np.random.randn(n_features)/100
        self.b = 0
        
    def forward(self, X):
        return np.dot(X, self.W) + self.b
    
    
    def compute_cost(self, predictions):
        return (1/(2*len(predictions)))*np.sum((self.y - predictions)**2)
    
    def backward(self, predictions):
        n = len(predictions)
        self.dW = np.dot((predictions - self.y),self.X)/n
        self.db = np.sum(predictions - self.y)/n 
        
    def fit(self, X,y, iterations):
        self.X = X
        self.y = y
        self.initialize_parameters(X.shape[1])
        self.costs = []
        
        for i in range(iterations):
            predictions = self.forward(X)
            self.costs.append(self.compute_cost(predictions))
            self.backward(predictions)
            
            self.W -= self.learning_rate*self.dW
            self.b -= self.learning_rate*self.db
            
            if(i % 10 == 0):
                print(f"Iteration {i} Cost: {self.costs[-1]}")
            
            if i > 0 and abs(self.costs[-1] - self.costs[-2]) < self.convergence_tol:
                print(f"Converged after {i} iterations")
                break
            
            
        plt.plot(np.arange(0,len(self.costs)), self.costs, linestyle = '-', linewidth = 2)
        plt.title('Training Loss over iterations')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.show()
            
    def predict(self, X):
        return self.forward(X)        
        
        
X_train, y_train =  train_data.iloc[:,:-1].values, train_data.iloc[:,-1].values     
X_test, y_test =  test_data.iloc[:,:-1].values, test_data.iloc[:,-1].values     

X_train, X_test = standardize_data(X_train, X_test)

lr = LinearRegression(0.01)
lr.fit(X_train,y_train, 1000)





















      