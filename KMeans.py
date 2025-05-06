#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  4 23:55:05 2025

@author: pavanpaj
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, K):
        self.K = K
    
    def initialize_centroids(self, X):
        centroid_idx = np.random.permutation(X.shape[0])[:K]
        self.centroids = X[centroid_idx]
    
    def assign_points_to_centroids(self, X):
        X_expanded =  X[:,np.newaxis,:]
        centroids_expanded = self.centroids[np.newaxis,:,:]
        diff = X_expanded - centroids_expanded
        dist = np.sqrt(np.sum(diff**2, axis = 2))
        labels = np.argmin(dist, axis = 1)
        return labels
    
    def compute_mean(self, X, labels):
        centroids = np.zeros((self.K, X.shape[1]))
        for i in range(self.K):
            if np.any(labels == i):
                centroids[i] = X[labels == i].mean(axis=0)
            else:
                centroids[i] = X[np.random.randint(0, X.shape[0])]
        return centroids
        
    def fit(self, X, iterations = 10):
        self.initialize_centroids(X)
        for i in range(iterations):
            labels = self.assign_points_to_centroids(X)
            new_centroids = self.compute_mean(X, labels)
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return self.centroids, labels
    
    def predict(self, X_new):
        return self.assign_points_to_centroids(X_new)


def silhouette_score(X, labels):
    n_samples = len(X)
    
    # Replace sklearn's pairwise_distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    distance_matrix = np.linalg.norm(diff, axis=2)

    silhouette_scores = []

    for i in range(n_samples):
        same_cluster = (labels == labels[i])

        a = np.mean(distance_matrix[i][same_cluster]) if np.sum(same_cluster) > 1 else 0
        b = np.min([np.mean(distance_matrix[i][labels == lbl]) for lbl in np.unique(labels) if lbl != labels[i]])

        s = (b - a) / max(a, b) if max(a, b) > 0 else 0
        silhouette_scores.append(s)
    
    return np.mean(silhouette_scores)


def plot_clusters_direct(X, labels, centroids):
    plt.figure(figsize=(8, 5))
    
    for i in np.unique(labels):
        cluster = X[labels == i]
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Cluster {i}", alpha=0.6)
    
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, marker='X', color='black', label='Centroids')
    plt.title("KMeans Clusters (Feature 1 vs Feature 2)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    seed = 69
    np.random.seed(seed = seed)
    
    data = load_iris()
    X = pd.DataFrame(data = data.data, columns = data.feature_names)
    y = pd.Series(data = data.target, name = 'target')
    
    X = np.array(X)
    K = 3
        
    kmeans = KMeans(K = K)
    centroids, labels = kmeans.fit(X, 100)
    print(silhouette_score(X, labels))
    
    plot_clusters_direct(X, labels, centroids)
    
    
    
    """from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
        
    # Fit KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    
    # Compute Silhouette Score
    score = silhouette_score(X, labels)
    print(f"Silhouette Score Sklearn: {score:.4f}")
            """
    
    
    
