import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv("./Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

dendogram = sch.dendrogram(sch.linkage(X, method="ward"))

plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distances")
plt.show()

hc = AgglomerativeClustering(n_clusters=8, metric="euclidean", linkage="ward")
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c="red", label="Cluster 1")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c="blue", label="Cluster 2")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c="green", label="Cluster 3")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c="cyan", label="Cluster 4")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c="magenta", label="Cluster 5")
plt.scatter(X[y_hc == 5, 0], X[y_hc == 5, 1], s=100, c="black", label="Cluster 6")
plt.scatter(X[y_hc == 6, 0], X[y_hc == 6, 1], s=100, c="yellow", label="Cluster 7")
plt.scatter(X[y_hc == 7, 0], X[y_hc == 7, 1], s=100, c="orange", label="Cluster 8")
plt.title("Clusters of customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
