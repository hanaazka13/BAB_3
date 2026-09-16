# Exercise 3.13 K-means Clustering with make_blobs
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=100, n_features=3, centers=2, random_state=0)

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

print("Labels:\n", kmeans.labels_)

print("\nCluster Centers:\n", kmeans.cluster_centers_)

p = kmeans.predict([[12, 3, 1]])
print("\nPrediction for [12, 3, 1]:\n", p)