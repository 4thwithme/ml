import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv("./Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

num_of_clusters = 21
wcss = []
for i in range(1, num_of_clusters):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, num_of_clusters), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c="red", label="Cluster 1")
plt.scatter(
    X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c="blue", label="Cluster 2"
)
plt.scatter(
    X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c="green", label="Cluster 3"
)
plt.scatter(
    X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c="cyan", label="Cluster 4"
)
plt.scatter(
    X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c="magenta", label="Cluster 5"
)
plt.scatter(
    X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s=100, c="orange", label="Cluster 6"
)
plt.scatter(
    X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s=100, c="purple", label="Cluster 7"
)
plt.scatter(
    X[y_kmeans == 7, 0], X[y_kmeans == 7, 1], s=100, c="brown", label="Cluster 8"
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c="yellow",
    label="Centroids",
)
plt.title("Clusters of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()

plt.show()
