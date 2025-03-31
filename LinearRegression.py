# %% 
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, make_regression
import matplotlib.pyplot as plt

seed = 69
np.random.seed(seed)

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns = data.feature_names)[:5000]
y = pd.Series(data.target, name = 'target')[:5000]


"""# Generate 500 samples, 3 features, small noise
X, y = make_regression(n_samples=500, n_features=3, noise=10, random_state=42)
X = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3'])
y = pd.Series(y, name='target')"""


data = pd.concat([X,y],axis=1)
train_data = data[:int(0.8*len(data))]
test_data = data[int(0.8*len(data)):]




# Dropping any null values
train_data = train_data.dropna()
test_data = test_data.dropna()


train_data.head()

# Data Preprocessing

def standardize_data(X_train, X_test):
    # Standardize (Mean 0 and std dev 1) the values as Linear regression is susceptible to outlier. Helps with faster convergence.
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
    
    
    def compute_cost(self,X,y,predictions):
        return (1/(2*len(predictions)))*np.sum((y - predictions)**2)
    
    def backward(self,X,y,predictions):
        n = y.shape[0]
        dW = np.dot((predictions - y),X)/n
        db = np.sum(predictions - y)/n 
        return dW, db
        
    def fit(self, X,y, iterations):
        self.initialize_parameters(X.shape[1])
        costs = []
        
        for i in range(iterations):
            predictions = self.forward(X)
            costs.append(self.compute_cost(X,y,predictions))
            dW, db = self.backward(X,y,predictions)
            
            self.W -= self.learning_rate*dW
            self.b -= self.learning_rate*db
            
            if(i % 10 == 0):
                print(f"Iteration {i} Cost: {costs[-1]:.5f}")
            
            if i > 0 and abs(costs[-1] - costs[-2]) < self.convergence_tol:
                print(f"Converged after {i} iterations")
                break
            
            
        plt.plot(np.arange(0,len(costs)), costs, linestyle = '-', linewidth = 2)
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


class RegressionMetrics:
    @staticmethod
    def MSE(y_true, y_pred):
        mse = np.mean((y_true-y_pred)**2)
        return mse
    @staticmethod
    def RMSE(y_true, y_pred):
        mse = RegressionMetrics.MSE(y_true, y_pred)
        rmse = np.sqrt(mse)
        return rmse
    @staticmethod
    def R2(y_true, y_pred):
        sse = np.sum((y_true - y_pred)**2)
        sst = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - (sse/sst)
        return r2
    
    
y_train_pred = lr.predict(X_train)
train_r2 = RegressionMetrics.R2(y_train, y_train_pred)
print(f"Final Training R² Score: {train_r2:.4f}")



y_pred = lr.predict(X_test)
mse = RegressionMetrics.MSE(y_test, y_pred)
rmse = RegressionMetrics.RMSE(y_test, y_pred)
r2 = RegressionMetrics.R2(y_test, y_pred)


print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (Coefficient of Determination): {r2}")















      