# Exercise 3.14
from sklearn.cluster import KMeans
import numpy as np

X = np.array([
    [1, 2, 3], [1, 4, 2], [1, 0, 3],   
    [2, 1, 2], [1, 3, 1],               
    [10, 2, 4], [9, 4, 3], [11, 0, 2],  
    [10, 5, 2], [8, 2, 3]               
])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print("Labels:", kmeans.labels_)
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Prediction:", kmeans.predict([[12, 3, 1]]))