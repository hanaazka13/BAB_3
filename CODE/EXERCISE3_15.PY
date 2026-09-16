# Exercise 3.15
from sklearn.cluster import KMeans
import numpy as np

X = np.array([
    [1, 2, 3], [1, 4, 2], [1, 0, 3],     # Kelompok 1
    [10, 2, 4], [9, 4, 3], [11, 0, 2],    # Kelompok 2
    [20, 2, 4], [21, 4, 3], [19, 0, 2]    # Kelompok 3 (Grup Baru)
])

# Ubah n_clusters menjadi 3 karena sekarang ada 3 grup
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print("Labels:", kmeans.labels_)
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Prediction:", kmeans.predict([[12, 3, 1]]))